from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import numpy as np

from compile_tg import compile_tg
from morphology.motor_drive import motor_actuator_gains
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")


def run(tg_path: Path, scene_path: Path, periods: float, warmup_periods: float) -> None:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    control = load_control_for_morphology(tg_path)
    period = control.period
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    gains = motor_actuator_gains(morph)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}

    def step() -> float:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        base_ctrl = morph.motors.motor_direction * eval_control(control, data.time)
        for name, aid in actuator_ids.items():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl * float(gains.get(name, 1.0))
        lift = sum(float(np.sum([f[2] for f in (wing.cached_forces or [])])) for wing in aero.wings)
        mujoco.mj_step(model, data)
        return lift

    for _ in range(math.ceil(warmup_periods * period / model.opt.timestep)):
        step()

    lifts = np.array([step() for _ in range(math.ceil(periods * period / model.opt.timestep))])
    print(f"control_fundamental_frequency_hz {control.fundamental_frequency_hz:.6g}")
    print(f"control_harmonics {control.harmonics}")
    print(f"periods {periods:.6g}")
    print(f"samples {lifts.size}")
    print(f"mean_lift_N {float(lifts.mean()):.9g}")
    print(f"rms_lift_N {float(np.sqrt(np.mean(lifts * lifts))):.9g}")
    print(f"min_lift_N {float(lifts.min()):.9g}")
    print(f"max_lift_N {float(lifts.max()):.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compiled morphology with its design control signal.")
    parser.add_argument("--tg", type=Path, default=ROOT / "design" / "shoulder.tg")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--periods", type=float, default=5.0)
    parser.add_argument("--warmup-periods", type=float, default=2.0)
    args = parser.parse_args()
    run(args.tg, args.scene, args.periods, args.warmup_periods)


if __name__ == "__main__":
    main()
