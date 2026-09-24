"""Component-level CPU vs MJX divergence check for one design: per step, the aero, pitch,
applied-force and post-step state differences between MuJoCo CPU and MJX.

    uv run python opt/diag_mjx_cpu_parity.py --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from codesign.rig import INTEGRATORS, set_unsteady_aero, setup
from runtime.aero import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology
from runtime.crank import sync_crank_disks_at_rest
from runtime.mjx import aero_runtime as maero
from runtime.mjx import rollout as mroll
from runtime.mjx import wing_pitch as mwing
from runtime.morphology_runtime import DESIGNS_DIR, load_design_morphology
from runtime.realtime import get_actuator_id
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest


def norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x)))


def diff_norm(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def make_cpu(model, tg: Path, substep: int):
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, load_design_morphology().motors.crank_phase_lag_rad)
    aero = AeroBladeRuntime(model, substep=substep)
    pitch = WingPitchLimiter(model, data)
    control = load_control_for_morphology(tg)
    motor_dir = load_design_morphology().motors.motor_direction
    aids = [get_actuator_id(model, name) for name in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]

    def pre_forces():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        aero_q = data.qfrc_applied.copy()
        pitch.apply(model, data)
        pitch_q = data.qfrc_applied.copy() - aero_q
        ctrl = motor_dir * eval_control(control, data.time)
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = ctrl
        return ctrl, aero_q, pitch_q, data.qfrc_applied.copy()

    return data, pre_forces


def make_mjx_pre(mx, drive, pitch, aero, substep: int):
    def pre(carry):
        dx, state, cq, cnfw, i = carry
        ctrl_vec = mroll.drive_ctrl(dx.ctrl, dx.time, drive)
        base = dx.replace(ctrl=ctrl_vec, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
        pitch_q = mwing.wing_pitch_qfrc(base.qpos, base.qvel, pitch)

        def recompute(_):
            return maero.aero_wrench(mx, base, aero, state)

        def hold(_):
            return cq, cnfw, state

        aero_q, net_force_world, new_state = jax.lax.cond((i % substep) == 0, recompute, hold, operand=None)
        pre_step = base.replace(qfrc_applied=aero_q + pitch_q)
        return ctrl_vec, aero_q, pitch_q, pre_step, new_state, net_force_world

    return jax.jit(pre)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", type=Path, default=DESIGNS_DIR / "hummer" / "checkpoints" / "tuned-dt-hacked" / "shoulder.tg")
    ap.add_argument("--scene", type=Path, default=Path("/tmp/diag_mjx_cpu_parity_scene.xml"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--aero-substep", type=int, default=4)
    ap.add_argument("--integrator", choices=list(INTEGRATORS), default="euler")
    args = ap.parse_args()

    set_unsteady_aero(False)
    model, mx, dx, _control, drive, pitch, aero, _tip_ids, _tendons, _motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator=args.integrator
    )
    cpu, cpu_pre = make_cpu(model, args.tg, args.aero_substep)
    mjx_pre = make_mjx_pre(mx, drive, pitch, aero, args.aero_substep)
    state = maero.init_aero_state(aero)
    carry = (dx, state, jnp.zeros_like(dx.qvel), jnp.zeros(3, dtype=dx.qvel.dtype), jnp.int32(0))

    print(f"jax_enable_x64={jax.config.jax_enable_x64} dx_dtype={dx.qpos.dtype} pitch_dtype={pitch.center.dtype}")
    print(
        "step state qvel cpu_aero mjx_aero aero_diff cpu_pitch mjx_pitch pitch_diff "
        "applied_diff ctrl_diff post_state qacc_diff actuator_diff "
        "qfrc_actuator_diff qfrc_passive_diff qfrc_constraint_diff ten_length_diff"
    )
    for step in range(args.steps):
        ctrl_cpu, cpu_aero, cpu_pitch, cpu_applied = cpu_pre()
        ctrl_vec, mjx_aero, mjx_pitch, pre_dx, new_state, net_force = mjx_pre(carry)
        ctrl_diff = float(abs(ctrl_cpu - np.asarray(ctrl_vec)[0]))
        state_diff = diff_norm(cpu.qpos, carry[0].qpos) + diff_norm(cpu.qvel, carry[0].qvel)
        qvel_diff = diff_norm(cpu.qvel, carry[0].qvel)
        print(
            f"{step:03d} {state_diff:.3e} {qvel_diff:.3e} "
            f"{norm(cpu_aero):.3e} {norm(mjx_aero):.3e} {diff_norm(cpu_aero, mjx_aero):.3e} "
            f"{norm(cpu_pitch):.3e} {norm(mjx_pitch):.3e} {diff_norm(cpu_pitch, mjx_pitch):.3e} "
            f"{diff_norm(cpu_applied, pre_dx.qfrc_applied):.3e} {ctrl_diff:.3e}",
            end=" ",
        )
        mujoco.mj_step(model, cpu)
        next_dx = mjx.step(mx, pre_dx)
        post_state = diff_norm(cpu.qpos, next_dx.qpos) + diff_norm(cpu.qvel, next_dx.qvel)
        qacc_diff = diff_norm(cpu.qacc, next_dx.qacc)
        actuator_diff = diff_norm(cpu.actuator_force, next_dx.actuator_force)
        print(
            f"{post_state:.3e} {qacc_diff:.3e} {actuator_diff:.3e} "
            f"{diff_norm(cpu.qfrc_actuator, next_dx.qfrc_actuator):.3e} "
            f"{diff_norm(cpu.qfrc_passive, next_dx.qfrc_passive):.3e} "
            f"{diff_norm(cpu.qfrc_constraint, next_dx.qfrc_constraint):.3e} "
            f"{diff_norm(cpu.ten_length, next_dx.ten_length):.3e}"
        )
        carry = mroll.match_dtypes((next_dx, new_state, mjx_aero, net_force, carry[4] + 1), carry)


if __name__ == "__main__":
    main()
