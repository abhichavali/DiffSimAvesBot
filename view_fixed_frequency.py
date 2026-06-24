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
import mujoco.viewer

from compile_tg import compile_tg
from morphology.motor_drive import motor_actuator_gains
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import RealtimeSync, get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")


def main() -> None:
    parser = argparse.ArgumentParser(description="View the compiled morphology at fixed crank frequency.")
    parser.add_argument("--tg", type=Path, default=ROOT / "design" / "shoulder.tg")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--frequency", type=float, default=10.0, help="Crank frequency in Hz")
    parser.add_argument("--speed", type=float, default=0.2, help="Viewer realtime factor")
    args = parser.parse_args()

    os.environ["MORPHOLOGY_PATH"] = str(args.tg)
    os.environ["MUJOCO_XML_PATH"] = str(args.scene)
    compile_tg(args.tg, args.scene)

    morph = load_morphology(args.tg)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    omega = math.tau * args.frequency
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    gains = motor_actuator_gains(morph)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}

    def step_forces(_model, _data) -> None:
        aero.apply(model, data)
        pitch.apply(model, data)
        data.ctrl[:] = 0.0
        for name, aid in actuator_ids.items():
            if aid >= 0:
                data.ctrl[aid] = omega * float(gains.get(name, 1.0))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sync = RealtimeSync(data, realtime_factor=args.speed)
        while viewer.is_running():
            sync.step_and_sync(model, data, viewer, pre_mj_step_hook=step_forces)


if __name__ == "__main__":
    main()
