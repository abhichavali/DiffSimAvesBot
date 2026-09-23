"""Which quantity can the CONTROL SIGNAL robustly move? A gradient screen.

Runs one forward-mode jacobian of a bag of candidate objectives w.r.t. the shared
Fourier theta [a0_log, a_1..a_K, b_1..b_K], under several (integrator, dt) variants
from the robustness bank, and scores each candidate on two axes:

  ELASTICITY  d(log m)/d(theta) -- how much a unit step in the control genome moves
              the metric, in percent. Reported separately for a0 (= crank speed =
              flap frequency) and for the HARMONICS (= waveform shape), because the
              standing verdict is "harmonics are dead" and only frequency moves
              anything. A target worth optimizing needs harmonic elasticity.
  CONSENSUS   min pairwise cosine similarity of the harmonic gradient across the
              variants. 1.0 = every dt/integrator agrees which way to push; <=0 =
              the "improvement" is a property of one discretization, i.e. exactly
              the MJX-overfit that the CPU transfer gate later rejects. A big
              elasticity with low consensus is a trap, not a target.

The screen is control-only (2K+1 params, no tendon/pitch genes) both because that
is the question and because it keeps the jacfwd narrow enough to compile in minutes.

    uv run --no-sync python opt/screen_targets.py \
        --tg flight-orientations/hovering-soft-it009p/shoulder.tg --lock-wings
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from opt.optimize_control import (ROOT, build_rollout, load_control_seed, setup,
                                  structural_tendon_pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path,
                   default=ROOT / "flight-orientations" / "hovering-soft-it009p" / "shoulder.tg")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--lock-wings", action="store_true")
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--timestep", type=float, default=0.06, help="center dt [ms]")
    p.add_argument("--dts", type=float, nargs="+", default=(0.04, 0.06, 0.09, 0.13))
    p.add_argument("--integrators", nargs="+", default=("euler", "implicitfast"))
    p.add_argument("--warmup-periods", type=float, default=8.0)
    p.add_argument("--sample-periods", type=float, default=4.0)
    p.add_argument("--aero-substep", type=int, default=1)
    args = p.parse_args()

    K = args.harmonics
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator="euler",
        timestep_ms=args.timestep, lock_wings=args.lock_wings)
    omega = float(control.omega)
    period = math.tau / omega
    base_a0 = float(drive.a0)
    theta = jnp.asarray(load_control_seed(None, K, drive, base_a0), dtype=jnp.float32)
    print(f"[screen] {args.tg.parent.name}  lock_wings={args.lock_wings}  "
          f"theta={theta.shape[0]} control params (a0_log + {K} cos + {K} sin)")

    import mujoco
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    flap_t1_mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names],
                               dtype=jnp.float32)

    import mujoco.mjx as mjx
    from runtime.mjx import aero_runtime
    from opt.optimize_control import _INTEGRATORS

    jacs = {}   # (integrator, dt) -> {metric: grad vector}
    values = {}
    for integ in args.integrators:
        for dt_ms in args.dts:
            dt = dt_ms * 1e-3
            model.opt.integrator = _INTEGRATORS[integ]
            model.opt.timestep = dt
            mx_v = mjx.put_model(model)
            aero_v = aero_runtime.build_aero_config(model, substep=args.aero_substep)
            spc = max(1, round(period / dt))
            wu = math.ceil(args.warmup_periods * period / dt)
            sm = math.ceil(args.sample_periods * period / dt)
            m = build_rollout(mx_v, dx0, drive, pitch, aero_v, tip_ids, omega, K, wu, sm,
                              steps_per_cycle=spc, aero_substep=args.aero_substep,
                              base_a0=base_a0, tune_control=True, control_fixed=theta,
                              tune_frequency=False, tune_tendons=False, tendons=tendons,
                              motor_ids=motor_ids, flap_t1_mask=flap_t1_mask,
                              avg_periods=max(1, int(args.sample_periods)),
                              extra_outputs=True)
            val, jac = jax.jit(lambda t, f=m: (f(t), jax.jacfwd(f)(t)))(theta)
            key = (integ, dt_ms)
            vals = {k: float(v) for k, v in val.items()}
            grads = {k: np.asarray(v, dtype=np.float64) for k, v in jac.items()}
            if (not all(np.isfinite(x) for x in vals.values())
                    or not all(np.isfinite(g).all() for g in grads.values())):
                # A cell that diverges at this dt has nothing to say about which
                # direction is robust; keeping it would zero every consensus score.
                print(f"  SKIP {integ}@{dt_ms:g}ms: non-finite (unstable at this dt)", flush=True)
                continue
            values[key], jacs[key] = vals, grads
            print(f"  rolled {integ}@{dt_ms:g}ms: "
                  + "  ".join(f"{k}={values[key][k]:.4g}" for k in ("lift", "power", "spec_power")),
                  flush=True)

    keys = list(jacs)
    assert len(keys) >= 2, f"need >=2 stable cells to score consensus, got {len(keys)}"
    names = list(jacs[keys[0]])

    def cos(u, v):
        n = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / n) if n > 1e-30 else 0.0

    rows = []
    for name in names:
        # Elasticity in % per unit theta step, split a0 (theta[0]) vs harmonics (rest).
        el_a0, el_h, mags = [], [], []
        for k in keys:
            g, v = jacs[k][name], values[k][name]
            scale = abs(v) + 1e-12
            el_a0.append(abs(g[0]) / scale * 100.0)
            el_h.append(float(np.linalg.norm(g[1:])) / scale * 100.0)
            mags.append(v)
        # Consensus: worst pairwise agreement of the HARMONIC gradient direction.
        hs = [jacs[k][name][1:] for k in keys]
        consensus = min(cos(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs)))
        spread = (max(mags) - min(mags)) / (abs(np.mean(mags)) + 1e-12) * 100.0
        rows.append((name, float(np.mean(mags)), float(np.mean(el_a0)),
                     float(np.mean(el_h)), consensus, spread))

    # Rank by what the task asks for: a target the WAVEFORM can move, in a direction
    # every discretization agrees on. Elasticity alone ranks traps first.
    rows.sort(key=lambda r: -(r[3] * max(r[4], 0.0)))
    print(f"\n{'metric':16s} {'value':>11s} {'el_a0 %':>9s} {'el_harm %':>10s} "
          f"{'consensus':>10s} {'dt-spread %':>11s}  {'score':>8s}")
    print("-" * 84)
    for name, v, ea, eh, c, sp in rows:
        print(f"{name:16s} {v:11.4g} {ea:9.2f} {eh:10.2f} {c:10.3f} {sp:11.1f}  "
              f"{eh*max(c,0.0):8.2f}")
    print("\nel_harm = |d metric / d harmonics| / |metric| x100 (waveform authority)\n"
          "consensus = worst pairwise cosine of the harmonic gradient across the "
          f"{len(keys)} (integrator, dt) cells; <=0 means the direction is a\n"
          "            property of one discretization -- an MJX-overfit trap, not a target.\n"
          "score = el_harm x consensus: robust waveform authority, the thing to optimize.")


if __name__ == "__main__":
    main()
