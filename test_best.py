"""Test the promoted design (design/shoulder.tg + design/control.tg) on CPU.

IMPORTANT: the design was optimized with the unsteady/added-mass aero term
DISABLED -- that term's MJX and CPU implementations diverge ~4x and break
sim-to-sim transfer, so the quasi-steady lift is the faithful, reproducible
number. This script disables it (the optimizer default) and reports mean lift
and peak motor torque under both integrators.

    uv run python test_best.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import mujoco

# Disable the added-mass term (clip FD accel to 0) BEFORE importing the aero runtime.
import runtime.aero_runtime as aero_rt
aero_rt.OMEGA_DOT_CAP = 0.0
aero_rt.LIN_ACC_CAP = 0.0

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter

TG = ROOT / "design" / "shoulder.tg"
SCENE = ROOT / "scene.xml"
INTEGRATORS = {"euler": 0, "implicitfast": 3}


def run(integrator: str, periods=3.0, warmup=2.0):
    import os
    os.environ["MORPHOLOGY_PATH"] = str(TG)
    os.environ["MUJOCO_XML_PATH"] = str(SCENE)
    compile_tg(TG, SCENE)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.integrator = INTEGRATORS[integrator]
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    morph = load_morphology(TG)
    ctrl = load_control_for_morphology(TG)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    aids = [get_actuator_id(model, n) for n in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]

    def step():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        cmd = morph.motors.motor_direction * eval_control(ctrl, data.time)
        for a in aids:
            if a >= 0:
                data.ctrl[a] = cmd
        lift = sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings)
        mujoco.mj_step(model, data)
        return lift, float(np.max(np.abs(data.actuator_force[aids])))

    for _ in range(int(warmup * ctrl.period / model.opt.timestep)):
        step()
    rec = [step() for _ in range(int(periods * ctrl.period / model.opt.timestep))]
    L = np.array([r[0] for r in rec]); T = np.array([r[1] for r in rec])
    return L.mean(), T.max()


if __name__ == "__main__":
    print("Testing design/ (added-mass disabled, the optimized regime):")
    for integ in INTEGRATORS:
        lift, torque = run(integ)
        print(f"  CPU {integ:13s}: mean lift = {lift*1e3:6.1f} mN   peak torque = {torque*1e3:6.1f} mNm")
    print("\n(target: ~222 mN lift / ~121 mNm torque, hovers ~23 g)")
