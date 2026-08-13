"""Wide random sweep over (frequency x tendon/motor/control) space on MJX, vmapped.

Goal: locate the best BASIN (freq level + stiffness structure), not polish a design.
Frequency levels get their own exactly-windowed rollout builder; within a level the
61-dim design genomes are evaluated vmapped in batches.

    uv run python opt/wide_sweep.py --freqs 8 10 12 14 16 --n 1536 --batch 256
"""
import argparse, json, math, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_control as oc

ROOT = Path(__file__).resolve().parents[1]
K = 5
LIFT_FLOOR = 0.347  # N (0.9 x baseline 385 mN)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_level(tg, f_hz):
    scene = ROOT / f"opt/ws_{int(f_hz*10)}.xml"
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = oc.setup(
        tg, scene, aero_substep=2, integrator="euler", timestep_ms=0.06)
    omega = math.tau * f_hz
    period = 1.0 / f_hz
    dt = model.opt.timestep
    spc = max(1, round(period / dt))
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names], jnp.float32)
    metrics = oc.build_rollout(
        mx, dx0, drive, pitch, aero, tip_ids, omega, K,
        math.ceil(8.0 * period / dt), math.ceil(3.0 * period / dt),
        steps_per_cycle=spc, aero_substep=2, base_a0=float(drive.a0),
        tune_control=True, control_fixed=jnp.zeros(2 * K + 1, jnp.float32),
        tune_frequency=False, tune_tendons=True, tendons=tendons,
        tune_motor=True, motor_ids=motor_ids, flap_t1_mask=mask)
    return jax.jit(jax.vmap(metrics)), tendons


def sample_genomes(rng, n, n_tend, tendon_only=False):
    # Global stiffness level log-uniform 50..800 N/m + per-tendon heterogeneity, so
    # single draws already contain soft/stiff MIXTURES (the basin structure we want).
    lvl = rng.uniform(math.log(0.25), math.log(4.0), (n, 1))
    stiff = lvl + rng.normal(0, 0.35, (n, n_tend))
    pre = rng.normal(0, 0.18, (n, n_tend))
    if tendon_only:  # baseline control: isolate the tendon-set axis
        a0 = np.zeros((n, 1)); harm = np.zeros((n, 2 * K))
    else:
        a0 = rng.uniform(-0.2, 0.6, (n, 1))
        harm = rng.normal(0, 0.15, (n, 2 * K))
    motor = rng.uniform(-0.6, 0.6, (n, 2))
    return np.concatenate([a0, harm, stiff, pre, motor], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", type=Path, default=ROOT / "flight-orientations/hovering/shoulder.tg")
    p.add_argument("--freqs", type=float, nargs="+", default=[8.0, 10.0, 12.0, 14.0, 16.0])
    p.add_argument("--n", type=int, default=1536)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tendon-only", action="store_true",
                   help="baseline control (a0=base, no harmonics); sample only tendons/motor")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "wide_sweep_out")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for f_hz in args.freqs:
        log(f"level f={f_hz}Hz: building")
        vm, tendons = build_level(args.tg, f_hz)
        n_tend = int(np.asarray(tendons[0]).shape[0])
        G = sample_genomes(rng, args.n, n_tend, tendon_only=args.tendon_only)
        M = []
        for i in range(0, args.n, args.batch):
            t0 = time.time()
            M.append(np.asarray(vm(jnp.asarray(G[i:i + args.batch]))).T)
            log(f"level f={f_hz}Hz: batch {i//args.batch+1}/{args.n//args.batch} {time.time()-t0:.0f}s")
        M = np.concatenate(M, 0)  # (n, 13): L,tau,A,S,Clc,Pin,Ovl,swing,Psup,Cv,alpha,k,conv
        ok = np.isfinite(M).all(1)
        for j in np.where(ok)[0]:
            all_rows.append(dict(f=f_hz, L=float(M[j, 0]), tau=float(M[j, 1]), A=float(M[j, 2]),
                                 Psup=float(M[j, 8]), k=float(M[j, 11]),
                                 genome=[float(x) for x in G[j]]))
        feas = [r for r in all_rows if r["f"] == f_hz and r["L"] >= LIFT_FLOOR]
        best = sorted(feas, key=lambda r: r["tau"])[:5]
        log(f"level f={f_hz}Hz done: {len(feas)}/{args.n} lift-feasible; best tau: "
            + ", ".join(f"{r['tau']*1e3:.0f}mNm/L{r['L']*1e3:.0f}" for r in best))

    feas = [r for r in all_rows if r["L"] >= LIFT_FLOOR]
    feas.sort(key=lambda r: r["tau"])
    (args.out / "top50.json").write_text(json.dumps(feas[:50], indent=1))
    lite = [{k: v for k, v in r.items() if k != "genome"} for r in all_rows]
    (args.out / "all_rows.json").write_text(json.dumps(lite))
    print("\n=== TOP 15 (lift >= %.0f mN, by peak torque) ===" % (LIFT_FLOOR * 1e3), flush=True)
    for r in feas[:15]:
        print(f"f={r['f']:4.1f}Hz tau={r['tau']*1e3:6.1f}mNm L={r['L']*1e3:6.0f}mN "
              f"A={r['A']*1e3:4.0f}mm P={r['Psup']*1e3:5.0f}mW k={r['k']:.2f}", flush=True)
    print(f"[done] {len(feas)} feasible of {len(all_rows)}; top50 -> {args.out}/top50.json", flush=True)


if __name__ == "__main__":
    main()
