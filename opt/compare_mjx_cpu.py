"""Diagnose the MJX vs CPU lift gap: run the SAME design in both, Euler, same
aero substep, same initial state, and compare the per-step lift trace.

If they match step-1 and diverge slowly -> trajectory sensitivity. If they
differ from step 1 -> an aero/force computation difference (the 2x suspect).

    uv run python opt/compare_mjx_cpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from optimize_control import setup, set_unsteady_aero
from runtime.mjx import aero_runtime as mar
from runtime.mjx import wing_pitch as mwp
from runtime.mjx import rollout as mroll
from runtime.aero_runtime import AeroBladeRuntime
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_limit import WingPitchLimiter
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.crank_sync import sync_crank_disks_at_rest

TG = ROOT / "design" / "baseline_shoulder.tg"  # meshdir 'assets' resolves from repo root
SCENE = Path("/tmp/cmp_scene.xml")
NSTEPS = 60  # ~ a few crank revolutions of comparison


def mjx_trace(model, mx, dx0, drive, pitch, aero, n):
    flap = drive.flap_actuator_ids
    a0 = float(drive.a0)
    substep = aero.substep if hasattr(aero, "substep") else 4

    def step(carry, _):
        dx, st, cq, cn, i = carry
        ctrl = jnp.zeros_like(dx.ctrl).at[flap].set(a0)
        dx = dx.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
        dx = mwp.apply(dx, pitch)
        def recompute(_):
            q, nf, ns = mar.aero_wrench(mx, dx, aero, st)
            return q, nf, ns
        def hold(_):
            return cq, cn, st
        q, nf, st2 = jax.lax.cond((i % substep) == 0, recompute, hold, operand=None)
        dx = dx.replace(qfrc_applied=dx.qfrc_applied + q)
        dx = mjx.step(mx, dx)
        carry = mroll._match_dtypes((dx, st2, q, nf, i + 1), carry)
        return carry, nf[2]

    st0 = mar.init_aero_state(aero)
    c0 = (dx0, st0, jnp.zeros_like(dx0.qvel), jnp.zeros(3), jnp.int32(0))
    _, lifts = jax.lax.scan(step, c0, xs=None, length=n)
    return np.asarray(lifts)


def cpu_trace(model, n):
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    from runtime.control_signal import load_control_for_morphology
    ctrl = load_control_for_morphology(TG)
    morph_dir = __import__("runtime.morphology_runtime", fromlist=["load_design_morphology"])
    a0 = morph_dir.load_design_morphology().motors.motor_direction * ctrl.a0
    aids = [get_actuator_id(model, nm) for nm in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    lifts = []
    for _ in range(n):
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = a0
        lifts.append(sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings))
        mujoco.mj_step(model, data)
    return np.array(lifts)


def dof_compare(model, mx, dx0, drive, pitch, aero):
    """Step both MJX and CPU 8 steps from the same IC; compare crank vel + wing pitch."""
    jid_crank = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_t1_disk_hinge")
    jid_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_wing_pitch")
    dofadr_crank = model.jnt_dofadr[jid_crank]
    qadr_pitch = model.jnt_qposadr[jid_pitch]
    flap = drive.flap_actuator_ids
    a0 = float(drive.a0)
    # CPU
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    caero = AeroBladeRuntime(model); cpitch = WingPitchLimiter(model, data)
    aids = [get_actuator_id(model, nm) for nm in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    from runtime.morphology_runtime import load_design_morphology
    md = load_design_morphology().motors.motor_direction
    cpu_rows = []
    for _ in range(8):
        data.qfrc_applied[:] = 0; data.ctrl[:] = 0
        caero.apply(model, data); cpitch.apply(model, data)
        for aid in aids:
            if aid >= 0: data.ctrl[aid] = md * a0
        cpu_rows.append((float(data.qvel[dofadr_crank]), float(data.qpos[qadr_pitch])))
        mujoco.mj_step(model, data)
    # MJX
    dx = dx0; st = mar.init_aero_state(aero)
    mjx_rows = []
    step_fn = jax.jit(lambda mx, dx, st: _mjx_one(mx, dx, st, pitch, aero, flap, a0))
    for _ in range(8):
        mjx_rows.append((float(dx.qvel[dofadr_crank]), float(dx.qpos[qadr_pitch])))
        dx, st = step_fn(mx, dx, st)
    print("step | crank_vel MJX/CPU | wing_pitch MJX/CPU")
    for i in range(8):
        print(f"  {i} | {mjx_rows[i][0]:8.4f} {cpu_rows[i][0]:8.4f} | "
              f"{mjx_rows[i][1]*57.3:7.3f} {cpu_rows[i][1]*57.3:7.3f} deg")


def _mjx_one(mx, dx, st, pitch, aero, flap, a0):
    ctrl = jnp.zeros_like(dx.ctrl).at[flap].set(a0)
    dx = dx.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
    dx = mwp.apply(dx, pitch)
    q, nf, st = mar.aero_wrench(mx, dx, aero, st)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied + q)
    dx = mjx.step(mx, dx)
    return dx, st


def main():
    import sys as _sys
    on = "--off" not in _sys.argv
    set_unsteady_aero(on)  # added-mass ON by default; must precede the jitted rollout
    print(f"added-mass (unsteady aero): {'ON' if on else 'OFF'}  integrator=euler")
    model, mx, dx0, control, drive, pitch, aero, tip_id, tendons, motor_ids = setup(
        TG, SCENE, integrator="euler")
    dof_compare(model, mx, dx0, drive, pitch, aero)
    lm = mjx_trace(model, mx, dx0, drive, pitch, aero, NSTEPS)
    # rebuild a fresh euler model for CPU (same scene)
    cm_model = mujoco.MjModel.from_xml_path(str(SCENE))
    cm_model.opt.integrator = 0  # euler
    lc = cpu_trace(cm_model, NSTEPS)
    print(f"steps compared: {NSTEPS}")
    print(f"MJX lift mean {lm.mean()*1e3:7.2f} mN   first5 {np.round(lm[:5]*1e3,2)}")
    print(f"CPU lift mean {lc.mean()*1e3:7.2f} mN   first5 {np.round(lc[:5]*1e3,2)}")
    print(f"ratio MJX/CPU mean = {lm.mean()/ (lc.mean()+1e-12):.3f}")
    d = np.abs(lm - lc)
    print(f"per-step |diff| mean {d.mean()*1e3:.3f} mN  first divergence>0.1mN at step "
          f"{int(np.argmax(d>1e-4)) if np.any(d>1e-4) else -1}")
    print(f"step0  MJX {lm[0]*1e3:.3f}  CPU {lc[0]*1e3:.3f}")


if __name__ == "__main__":
    main()
