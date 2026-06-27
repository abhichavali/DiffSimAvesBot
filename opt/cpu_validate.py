"""CPU-validate an optimizer design: mean lift + peak motor torque on MuJoCo CPU.

Applies run_dir's design onto the base morphology, compiles, and runs the CPU
reference pipeline (numpy aero, substep=4) for `periods` cycles after warmup.
Reports under both `euler` (the MJX-matching integrator) and `implicitfast`
(the design's nominal physics) so the sim2sim spread is visible.

    uv run python opt/cpu_validate.py opt/auto/run_0 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco
import numpy as np

from apply_design import write_design  # noqa: E402 (opt/ on path via __main__)
sys.path.insert(0, str(ROOT / "opt"))


def cpu_metrics(shoulder_tg: Path, scene_xml: Path, integrator: str,
                periods: float = 3.0, warmup: float = 2.0) -> dict:
    os.environ["MORPHOLOGY_PATH"] = str(shoulder_tg)
    os.environ["MUJOCO_XML_PATH"] = str(scene_xml)
    from compile_tg import compile_tg
    from morphology.tg_parser import load_morphology
    from runtime.aero_runtime import AeroBladeRuntime
    from runtime.control_signal import eval_control, load_control_for_morphology
    from runtime.crank_sync import sync_crank_disks_at_rest
    from runtime.viewer_runtime import get_actuator_id
    from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
    from runtime.wing_pitch_limit import WingPitchLimiter

    compile_tg(shoulder_tg, scene_xml)
    integ = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}[integrator]
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.integrator = integ
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    morph = load_morphology(shoulder_tg)
    ctrl = load_control_for_morphology(shoulder_tg)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    aids = [get_actuator_id(model, n) for n in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]

    def step():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        bc = morph.motors.motor_direction * eval_control(ctrl, data.time)
        for a in aids:
            if a >= 0:
                data.ctrl[a] = bc
        lift = sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings)
        mujoco.mj_step(model, data)
        return lift, float(np.max(np.abs(data.actuator_force[aids])))

    for _ in range(int(warmup * ctrl.period / model.opt.timestep)):
        step()
    rec = [step() for _ in range(int(periods * ctrl.period / model.opt.timestep))]
    L = np.array([r[0] for r in rec])
    T = np.array([r[1] for r in rec])
    return {"mean_lift_N": float(L.mean()), "peak_torque_Nm": float(T.max()),
            "finite": bool(np.all(np.isfinite(L)) and np.all(np.isfinite(T)))}


def validate(run_dir: Path, base_shoulder: Path, integrators=("euler", "implicitfast")) -> dict:
    tmp = Path(tempfile.mkdtemp())
    write_design(run_dir, base_shoulder, tmp / "shoulder.tg", tmp / "control.tg")
    # control_file in the temp shoulder.tg points at ./control.tg (same dir).
    out = {}
    for integ in integrators:
        try:
            out[integ] = cpu_metrics(tmp / "shoulder.tg", tmp / "scene.xml", integ)
        except Exception as e:  # robust: one integrator failing shouldn't sink the run
            out[integ] = {"error": str(e)[:200]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--base", type=Path, default=ROOT / "design" / "shoulder.tg")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = validate(args.run_dir, args.base)
    if args.json:
        print(json.dumps(res))
    else:
        for integ, m in res.items():
            print(integ, m)


if __name__ == "__main__":
    main()
