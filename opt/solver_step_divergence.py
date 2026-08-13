"""One-step CPU MuJoCo vs MJX solver map from identical CPU pre-step states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from optimize_control import _INTEGRATORS, setup, set_unsteady_aero
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.morphology_runtime import load_design_morphology
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


def diff_norm(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", type=Path, default=ROOT / "flight-orientations" / "hovering-tuned-hacked" / "shoulder.tg")
    ap.add_argument("--scene", type=Path, default=Path("/tmp/solver_step_divergence_scene.xml"))
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--aero-substep", type=int, default=4)
    ap.add_argument("--integrator", choices=list(_INTEGRATORS), default="euler")
    args = ap.parse_args()

    set_unsteady_aero(False)
    model, mx, _dx0, _control, _drive, _pitch, _aero, _tip_ids, _tendons, _motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator=args.integrator
    )
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, load_design_morphology().motors.crank_phase_lag_rad)
    aero = AeroBladeRuntime(model, substep=args.aero_substep)
    pitch = WingPitchLimiter(model, data)
    control = load_control_for_morphology(args.tg)
    motor_dir = load_design_morphology().motors.motor_direction
    aids = [get_actuator_id(model, name) for name in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    mjx_step = jax.jit(mjx.step)

    print(f"jax_enable_x64={jax.config.jax_enable_x64} integrator={args.integrator} dt_ms={model.opt.timestep*1e3:g}")
    print("step qpos qvel qacc qfrc_actuator qfrc_passive qfrc_constraint actuator_force ten_length")
    for step in range(args.steps):
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ctrl = motor_dir * eval_control(control, data.time)
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = ctrl

        before = mujoco.MjData(model)
        before.qpos[:] = data.qpos
        before.qvel[:] = data.qvel
        before.act[:] = data.act
        before.time = data.time
        before.ctrl[:] = data.ctrl
        before.qfrc_applied[:] = data.qfrc_applied
        before.qacc_warmstart[:] = data.qacc_warmstart
        mujoco.mj_forward(model, before)
        before.qfrc_applied[:] = data.qfrc_applied
        before.ctrl[:] = data.ctrl

        dx_same = mjx.put_data(model, before)
        dx_next = mjx_step(mx, dx_same)
        mujoco.mj_step(model, data)
        print(
            f"{step:03d} "
            f"{diff_norm(data.qpos, dx_next.qpos):.3e} "
            f"{diff_norm(data.qvel, dx_next.qvel):.3e} "
            f"{diff_norm(data.qacc, dx_next.qacc):.3e} "
            f"{diff_norm(data.qfrc_actuator, dx_next.qfrc_actuator):.3e} "
            f"{diff_norm(data.qfrc_passive, dx_next.qfrc_passive):.3e} "
            f"{diff_norm(data.qfrc_constraint, dx_next.qfrc_constraint):.3e} "
            f"{diff_norm(data.actuator_force, dx_next.actuator_force):.3e} "
            f"{diff_norm(data.ten_length, dx_next.ten_length):.3e}"
        )


if __name__ == "__main__":
    main()
