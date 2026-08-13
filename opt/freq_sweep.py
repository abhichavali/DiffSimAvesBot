"""Frequency-response sweep of a design: lift/torque/amplitude vs drive frequency.

    uv run python opt/freq_sweep.py [--freqs 10 12.5 15 17.5 20] [--stiff-factor 1.0]

For each frequency the rollout is rebuilt with omega'=2*pi*f (steps/warmup/sample
rescaled to the new period), evaluated with (a) the design a0 and (b) a0 scaled by
f/f0 (constant commanded stroke). --stiff-factor scales ALL structural tendon
stiffnesses (e.g. 2.5 for the 500 N/m regime).
"""
import argparse, math, sys
from pathlib import Path

import jax, jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_control as oc

ROOT = Path(__file__).resolve().parents[1]
K = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", type=Path, default=ROOT / "flight-orientations/hovering/shoulder.tg")
    p.add_argument("--freqs", type=float, nargs="+", default=[10.0, 12.5, 15.0, 17.5, 20.0])
    p.add_argument("--stiff-factor", type=float, default=1.0)
    args = p.parse_args()

    for f in args.freqs:
        scene = ROOT / f"opt/fsw_{int(f*10)}.xml"
        model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = oc.setup(
            args.tg, scene, aero_substep=2, integrator="euler", timestep_ms=0.06)
        if args.stiff_factor != 1.0:
            l_ids, r_ids = np.asarray(tendons[0]), np.asarray(tendons[1])
            st = np.asarray(mx.tendon_stiffness).copy()
            st[l_ids] *= args.stiff_factor; st[r_ids] *= args.stiff_factor
            mx = mx.replace(tendon_stiffness=jnp.asarray(st))
        f0 = control.omega / math.tau
        omega = math.tau * f
        period = 1.0 / f
        dt = model.opt.timestep
        spc = max(1, round(period / dt))
        flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                      for i in np.asarray(drive.flap_actuator_ids)]
        mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names], jnp.float32)
        for label, a0_log in (("a0 fixed", 0.0), ("a0~f", math.log(f / f0))):
            metrics = oc.build_rollout(
                mx, dx0, drive, pitch, aero, tip_ids, omega, K,
                math.ceil(8.0 * period / dt), math.ceil(3.0 * period / dt),
                steps_per_cycle=spc, aero_substep=2, base_a0=float(drive.a0) * math.exp(a0_log),
                tune_control=False, control_fixed=jnp.zeros(2 * K + 1, jnp.float32),
                tune_frequency=False, tune_tendons=False, tendons=tendons,
                tune_motor=False, motor_ids=motor_ids, flap_t1_mask=mask)
            m = np.asarray(jax.jit(metrics)(jnp.zeros(0, jnp.float32)))
            print(f"f={f:5.1f}Hz {label:9s} stiff x{args.stiff_factor:g}  "
                  f"L={m[0]*1e3:6.1f}mN tau={m[1]*1e3:6.1f}mNm A={m[2]*1e3:5.0f}mm "
                  f"P={m[8]*1e3:6.0f}mW k={m[11]:.2f}", flush=True)


if __name__ == "__main__":
    main()
