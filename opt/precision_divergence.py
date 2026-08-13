"""Compare CPU MuJoCo and MJX state divergence under the same drive.

Run with and without JAX x64 before import, e.g.

    JAX_ENABLE_X64=0 uv run python opt/precision_divergence.py
    JAX_ENABLE_X64=1 uv run python opt/precision_divergence.py
"""

from __future__ import annotations

import argparse
import os
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
from runtime.mjx import aero_runtime as maero
from runtime.mjx import rollout as mroll
from runtime.mjx import wing_pitch as mwing
from runtime.morphology_runtime import load_design_morphology
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


def state_norm(a, b) -> tuple[float, float, float, float]:
    qpos = np.asarray(a.qpos) - np.asarray(b.qpos)
    qvel = np.asarray(a.qvel) - np.asarray(b.qvel)
    both = np.concatenate([qpos.ravel(), qvel.ravel()])
    return float(np.linalg.norm(qpos)), float(np.linalg.norm(qvel)), float(np.linalg.norm(both)), float(np.max(np.abs(both)))


def make_cpu(model: mujoco.MjModel, tg: Path, *, aero_substep: int):
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, load_design_morphology().motors.crank_phase_lag_rad)
    aero = AeroBladeRuntime(model, substep=aero_substep)
    pitch = WingPitchLimiter(model, data)
    control = load_control_for_morphology(tg)
    motor_dir = load_design_morphology().motors.motor_direction
    aids = [get_actuator_id(model, name) for name in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]

    def step():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ctrl = motor_dir * eval_control(control, data.time)
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = ctrl
        mujoco.mj_step(model, data)

    return data, step


def make_mjx_step(mx, drive, pitch, aero, *, aero_substep: int):
    flap = drive.flap_actuator_ids

    def step(carry):
        dx, state, cq, cnfw, i = carry
        dx = dx.replace(ctrl=mroll.drive_ctrl(dx.ctrl, dx.time, drive), qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
        dx = mwing.apply(dx, pitch)

        def recompute(_):
            return maero.aero_wrench(mx, dx, aero, state)

        def hold(_):
            return cq, cnfw, state

        qfrc, net_force_world, state = jax.lax.cond((i % aero_substep) == 0, recompute, hold, operand=None)
        dx = dx.replace(qfrc_applied=dx.qfrc_applied + qfrc)
        dx = mjx.step(mx, dx)
        return mroll._match_dtypes((dx, state, qfrc, net_force_world, i + 1), carry)

    return jax.jit(step)


def summarize(label: str, rows: list[tuple[int, float, float, float, float]]) -> None:
    print(label)
    print("step,qpos_norm,qvel_norm,state_norm,state_max")
    for idx in sorted({0, 1, 2, 4, 8, 16, 32, 64, 128, len(rows) - 1}):
        if 0 <= idx < len(rows):
            print("{},{:.6e},{:.6e},{:.6e},{:.6e}".format(*rows[idx]))
    first = next((r for r in rows if r[3] > 1e-6), None)
    print("first_state_norm_gt_1e-6={}".format(first[0] if first else "none"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", type=Path, default=ROOT / "flight-orientations" / "hovering-tuned-hacked" / "shoulder.tg")
    ap.add_argument("--scene", type=Path, default=Path("/tmp/precision_divergence_scene.xml"))
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--integrator", choices=list(_INTEGRATORS), default="euler")
    ap.add_argument("--timestep", type=float, default=None, help="override timestep in ms")
    ap.add_argument("--aero-substep", type=int, default=4)
    ap.add_argument("--perturb", type=float, default=1e-7)
    args = ap.parse_args()

    set_unsteady_aero(False)
    model, mx, dx0, _control, drive, pitch, aero, _tip_ids, _tendons, _motor_ids = setup(
        args.tg,
        args.scene,
        aero_substep=args.aero_substep,
        integrator=args.integrator,
        timestep_ms=args.timestep,
    )
    print(f"jax_enable_x64={jax.config.jax_enable_x64}")
    print(f"mjx_qpos_dtype={dx0.qpos.dtype} mjx_qvel_dtype={dx0.qvel.dtype} drive_a_dtype={drive.a.dtype} pitch_center_dtype={pitch.center.dtype}")
    print(f"dt_ms={model.opt.timestep * 1e3:g} integrator={args.integrator} steps={args.steps} tg={args.tg}")

    cpu, cpu_step = make_cpu(model, args.tg, aero_substep=args.aero_substep)
    mjx_step = make_mjx_step(mx, drive, pitch, aero, aero_substep=args.aero_substep)
    carry = (dx0, maero.init_aero_state(aero), jnp.zeros_like(dx0.qvel), jnp.zeros(3, dtype=dx0.qvel.dtype), jnp.int32(0))
    cpu_rows: list[tuple[int, float, float, float, float]] = [(0, *state_norm(cpu, dx0))]
    for step in range(1, args.steps + 1):
        cpu_step()
        carry = mjx_step(carry)
        cpu_rows.append((step, *state_norm(cpu, carry[0])))
    summarize("CPU64_vs_MJX", cpu_rows)

    carry_a = (dx0, maero.init_aero_state(aero), jnp.zeros_like(dx0.qvel), jnp.zeros(3, dtype=dx0.qvel.dtype), jnp.int32(0))
    dxp = dx0.replace(qvel=dx0.qvel.at[0].add(args.perturb))
    carry_b = (dxp, maero.init_aero_state(aero), jnp.zeros_like(dx0.qvel), jnp.zeros(3, dtype=dx0.qvel.dtype), jnp.int32(0))
    pert_rows: list[tuple[int, float, float, float, float]] = [(0, *state_norm(carry_a[0], carry_b[0]))]
    for step in range(1, args.steps + 1):
        carry_a = mjx_step(carry_a)
        carry_b = mjx_step(carry_b)
        pert_rows.append((step, *state_norm(carry_a[0], carry_b[0])))
    summarize(f"MJX_vs_MJX_perturb_{args.perturb:g}", pert_rows)


if __name__ == "__main__":
    main()
