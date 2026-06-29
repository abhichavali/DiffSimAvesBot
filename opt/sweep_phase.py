"""Sweep the crank phase lag (drive_phase_offset) and report the value that best
staggers the two motors -- minimum torque overlap / inactive power -- while
holding lift. The lag is an initial crank-angle offset on the t4 ("orange") crank
(the two cranks always counter-rotate); see crank_sync.apply_crank_phase_lag.

  uv run python opt/sweep_phase.py --added-mass [--n 25] [--init-control theta_best.json]

Evaluates a forward rollout at the seed control by default (use --init-control to
evaluate at an optimized control instead). One rollout per lag.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "opt"):
    sys.path.insert(0, str(p))

import jax.numpy as jnp
import mujoco
import numpy as np

from optimize_control import setup, build_rollout, load_control_seed, set_unsteady_aero


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=ROOT / "design" / "baseline_shoulder.tg")
    ap.add_argument("--scene", type=Path, default=ROOT / "opt" / "sweep_scene.xml")
    ap.add_argument("--n", type=int, default=25, help="lag samples over [0, 360) deg")
    ap.add_argument("--harmonics", type=int, default=5)
    ap.add_argument("--warmup-periods", type=float, default=2.0)
    ap.add_argument("--sample-periods", type=float, default=2.0)
    ap.add_argument("--added-mass", action="store_true")
    ap.add_argument("--init-control", type=Path, default=None,
                    help="evaluate at an optimized control (theta_best.json) instead of the seed")
    ap.add_argument("--out", type=Path, default=ROOT / "opt" / "sweep_phase_result.json")
    args = ap.parse_args()

    set_unsteady_aero(args.added_mass)
    lags = np.linspace(0.0, 2 * math.pi, args.n, endpoint=False)
    print(f"sweeping {args.n} phase lags over [0,360) deg  added_mass={args.added_mass}", flush=True)
    rows = []
    for lag in lags:
        model, mx, dx0, control, drive, pitch, aero, tip_id, tendons, motor_ids = setup(
            args.tg, args.scene, integrator="euler", phase_lag=float(lag))
        K = args.harmonics
        omega = float(control.omega)
        dt = model.opt.timestep
        period = control.period
        steps_per_cycle = max(1, round(period / dt))
        warmup_steps = math.ceil(args.warmup_periods * period / dt)
        sample_steps = math.ceil(args.sample_periods * period / dt)
        base_a0 = float(drive.a0)
        seed = jnp.asarray(load_control_seed(args.init_control, K, drive, base_a0), dtype=jnp.float32)
        flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                      for i in np.asarray(drive.flap_actuator_ids)]
        mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names], dtype=jnp.float32)
        metrics = build_rollout(mx, dx0, drive, pitch, aero, tip_id, omega, K,
                                warmup_steps, sample_steps, steps_per_cycle=steps_per_cycle,
                                aero_substep=4, base_a0=base_a0, tune_control=True,
                                control_fixed=seed, tune_frequency=False, tune_tendons=False,
                                tendons=tendons, flap_t1_mask=mask)
        Lbar, tau, A, S, Clc, Pin, Ovl = (float(x) for x in metrics(seed))
        rows.append({"lag_deg": math.degrees(float(lag)), "lift_mN": Lbar * 1e3,
                     "tau_mNm": tau * 1e3, "A_mm": A * 1e3, "Pin": Pin, "overlap": Ovl})
        print(f"  lag={math.degrees(float(lag)):6.1f}deg  L={Lbar*1e3:6.1f}mN  "
              f"tau={tau*1e3:6.1f}mNm  A={A*1e3:5.1f}mm  Pin={Pin:.3g}  ovl={Ovl:.3g}", flush=True)

    best_ovl = min(rows, key=lambda r: r["overlap"])
    best_pin = min(rows, key=lambda r: r["Pin"])
    print(f"\nbest overlap : lag={best_ovl['lag_deg']:.1f} deg  ovl={best_ovl['overlap']:.3g}  "
          f"(L={best_ovl['lift_mN']:.0f}mN tau={best_ovl['tau_mNm']:.0f}mNm)")
    print(f"best inactive: lag={best_pin['lag_deg']:.1f} deg  Pin={best_pin['Pin']:.3g}  "
          f"(L={best_pin['lift_mN']:.0f}mN tau={best_pin['tau_mNm']:.0f}mNm)")
    args.out.write_text(json.dumps({"rows": rows, "best_overlap_deg": best_ovl["lag_deg"],
                                    "best_pin_deg": best_pin["lag_deg"]}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
