"""Score every reward arm on ONE shared robustness bank + a CPU transfer check.

Each arm in opt/run_reward_queue.sh trains with no in-loop gates, because on this
box a distinct jitted rollout costs ~30 min to compile and six arms would spend a
day compiling the same six bank cells over and over. This pays for them once: the
bank is built in a single process, and every arm's three thetas are pushed through
all of it.

Per arm it reports the objective (in its own units, robust = worst cell) at:
    baseline    the seed
    it010       the preharmonics snapshot -- a0/frequency only
    final       the run's selected design -- a0 + harmonics
so the frequency gain is (baseline -> it010) and the HARMONIC gain, which is what
the campaign is actually looking for, is (it010 -> final). An arm only counts if
that harmonic gain also survives the CPU rollout at the deploy dt and a holdout dt.

    uv run --no-sync python opt/certify_arms.py --arms swing tau_rms ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from opt.optimize_control import (_INTEGRATORS, OBJ_METRIC_IDX, ROOT, build_rollout,
                                  cpu_validate_design, load_control_seed, setup)
from runtime.mjx import aero_runtime

# Robust direction per metric: lift is a floor (worst = min), the rest are costs.
WORST_IS_MIN = {"lift"}


def obj_value(objective, row, lift_floor):
    if objective == "energy":
        return row["power"] / max(row["lift"], lift_floor)
    if objective == "swing_rms":
        return 0.5 * (row["swing"] / row["_s0"] + row["tau_rms"] / row["_r0"])
    return row[objective]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path,
                   default=ROOT / "flight-orientations" / "hovering-soft-it009p" / "shoulder.tg")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--arms", nargs="+",
                   default=("swing", "tau_rms", "power_crest", "overlap",
                            "lift_ripple", "swing_rms"))
    p.add_argument("--runs-root", type=Path, default=ROOT / "opt")
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--lift-floor", type=float, default=0.335)
    p.add_argument("--timestep", type=float, default=0.06)
    p.add_argument("--dts", type=float, nargs="+", default=(0.06, 0.09, 0.13))
    p.add_argument("--integrators", nargs="+", default=("euler", "implicitfast"))
    p.add_argument("--cpu-dts", type=float, nargs="*", default=(0.06, 0.03),
                   help="CPU transfer check: deploy dt first, then a holdout dt. Pass "
                        "with no values to skip CPU entirely -- a numpy rollout is ~40x "
                        "the wall time of an MJX bank cell, so the usual order is one "
                        "MJX-only pass to rank every arm, then CPU on the movers")
    p.add_argument("--warmup-periods", type=float, default=8.0)
    p.add_argument("--sample-periods", type=float, default=8.0)
    p.add_argument("--aero-substep", type=int, default=1)
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "certify.json")
    args = p.parse_args()

    K = args.harmonics
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator="euler",
        timestep_ms=args.timestep, lock_wings=True)
    omega, period, base_a0 = float(control.omega), math.tau / float(control.omega), float(drive.a0)
    seed = jnp.asarray(load_control_seed(None, K, drive, base_a0), dtype=jnp.float32)
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    t1_mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names],
                          dtype=jnp.float32)

    print(f"[certify] building the {len(args.integrators)}x{len(args.dts)} bank once "
          f"(this is the whole campaign's compile budget)", flush=True)
    bank = {}
    for integ in args.integrators:
        for dt_ms in args.dts:
            dt = dt_ms * 1e-3
            model.opt.integrator = _INTEGRATORS[integ]
            model.opt.timestep = dt
            f = build_rollout(
                mjx.put_model(model), dx0, drive, pitch,
                aero_runtime.build_aero_config(model, substep=args.aero_substep),
                tip_ids, omega, K,
                math.ceil(args.warmup_periods * period / dt),
                math.ceil(args.sample_periods * period / dt),
                steps_per_cycle=max(1, round(period / dt)), aero_substep=args.aero_substep,
                base_a0=base_a0, tune_control=True, control_fixed=seed,
                tune_frequency=False, tune_tendons=False, tendons=tendons,
                motor_ids=motor_ids, flap_t1_mask=t1_mask,
                avg_periods=max(1, int(args.sample_periods)))
            bank[(integ, dt_ms)] = jax.jit(f)
    model.opt.integrator = _INTEGRATORS["euler"]
    model.opt.timestep = args.timestep * 1e-3

    def robust_row(theta):
        """Worst cell across the bank, per metric."""
        cells = [{k: float(v) for k, v in zip(
            ("lift", "tau_lp", "A_stroke", "S", "C_lc", "P_inactive", "overlap", "swing",
             "power", "C_valid", "alpha", "k_eff", "conv", "tau_peak", "drift",
             "tau_rms", "power_crest", "lift_ripple"), f(theta))} for f in bank.values()]
        row = {}
        for k in cells[0]:
            vals = [c[k] for c in cells]
            row[k] = min(vals) if k in WORST_IS_MIN else max(vals)
        row["_cells"] = len(cells)
        return row

    def cpu_row(theta):
        if not args.cpu_dts:
            return {}
        t = np.asarray(theta)
        a0 = base_a0 * math.exp(float(t[0]))
        cp = (a0, t[1:1 + K], t[1 + K:], omega * math.exp(float(t[0])))
        out = {}
        for dt_ms in args.cpu_dts:
            try:
                L, T, P = cpu_validate_design(args.scene, cp, None, "euler",
                                              warmup=args.warmup_periods, periods=6.0,
                                              phase_lag=math.radians(6), timestep_ms=dt_ms,
                                              aero_substep=args.aero_substep)
                out[f"dt{dt_ms:g}"] = {"lift": L, "tau_peak": T, "power": P}
            except Exception as e:
                out[f"dt{dt_ms:g}"] = {"error": str(e)[:120]}
        return out

    def theta_of(path):
        f = path / "theta_full.json"
        if not f.exists():
            return None
        return jnp.asarray(json.loads(f.read_text())["theta"][:2 * K + 1], dtype=jnp.float32)

    def final_theta(run):
        """Last checkpoint, not theta_full.json at the run root.

        The root file holds the SELECTED best, and selection is a no-op when
        rotate-batch is on with full checks off -- it stays at the seed. The last
        checkpoint is the design the run actually ended on."""
        cps = sorted((run / "checkpoints").glob("it*")) if (run / "checkpoints").exists() else []
        for c in reversed(cps):
            t = theta_of(c)
            if t is not None:
                return t, c.name
        return theta_of(run), "root"

    base_row = robust_row(seed)
    base_cpu = cpu_row(seed)
    s0, r0 = max(base_row["swing"], 1e-12), max(base_row["tau_rms"], 1e-12)
    print(f"[baseline] robust over {base_row['_cells']} cells: "
          f"lift={base_row['lift']*1e3:.1f} mN  swing={base_row['swing']*1e3:.2f} mNm  "
          f"tau_rms={base_row['tau_rms']*1e3:.2f}  pcrest={base_row['power_crest']:.4f}  "
          f"ovl={base_row['overlap']:.4f}  lripple={base_row['lift_ripple']:.4f}  "
          f"P={base_row['power']:.3f} W", flush=True)

    results = {}
    for arm in args.arms:
        run = args.runs_root / f"runs_{arm}"
        stages = {"it010": theta_of(run / "preharmonics")}
        fin, fin_src = final_theta(run)
        stages["final"] = fin
        if stages["it010"] is None and fin is None:
            print(f"[{arm}] no thetas on disk ({run}) -- arm did not produce output")
            continue
        entry = {"baseline": {"robust": base_row, "cpu": base_cpu, "obj": None}}
        for label, th in stages.items():
            if th is None:
                print(f"[{arm}] {label}: missing")
                continue
            rr = robust_row(th)
            rr["_s0"], rr["_r0"] = s0, r0
            entry[label] = {"robust": rr, "cpu": cpu_row(th)}
        bb = dict(base_row); bb["_s0"], bb["_r0"] = s0, r0
        o_base = obj_value(arm, bb, args.lift_floor)
        entry["baseline"]["obj"] = o_base
        line = [f"[{arm}] target={arm} ({fin_src})  baseline={o_base:.6g}"]
        for label in ("it010", "final"):
            if label not in entry:
                continue
            o = obj_value(arm, entry[label]["robust"], args.lift_floor)
            entry[label]["obj"] = o
            line.append(f"{label}={o:.6g} ({(o - o_base) / abs(o_base) * 100:+.2f}%)")
        if "it010" in entry and "final" in entry:
            o10, ofin = entry["it010"]["obj"], entry["final"]["obj"]
            entry["freq_gain_pct"] = (o10 - o_base) / abs(o_base) * 100
            entry["harm_gain_pct"] = (ofin - o10) / abs(o10) * 100
            line.append(f"| freq {entry['freq_gain_pct']:+.2f}%  "
                        f"HARMONICS {entry['harm_gain_pct']:+.2f}%")
            fl = entry["final"]["robust"]["lift"]
            entry["lift_ok"] = bool(fl >= args.lift_floor)
            line.append(f"| liftmin={fl*1e3:.1f} mN {'OK' if entry['lift_ok'] else 'BELOW FLOOR'}")
        print("  ".join(line), flush=True)
        results[arm] = entry

    ranked = sorted((a for a in results if "harm_gain_pct" in results[a]),
                    key=lambda a: results[a]["harm_gain_pct"])
    print("\n=== ranked by certified HARMONIC gain (most negative = waveform helped most) ===")
    print(f"{'arm':14s} {'freq %':>9s} {'harm %':>9s} {'total %':>9s} {'lift':>10s}")
    for a in ranked:
        e = results[a]
        tot = (e["final"]["obj"] - e["baseline"]["obj"]) / abs(e["baseline"]["obj"]) * 100
        print(f"{a:14s} {e['freq_gain_pct']:+9.2f} {e['harm_gain_pct']:+9.2f} {tot:+9.2f} "
              f"{'OK' if e['lift_ok'] else 'FLOOR':>10s}")
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
