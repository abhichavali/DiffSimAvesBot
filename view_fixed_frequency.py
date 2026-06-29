from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import mujoco.viewer

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import RealtimeSync, get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")


def main() -> None:
    parser = argparse.ArgumentParser(description="View the compiled morphology with its design control signal.")
    parser.add_argument("--tg", type=Path, default=ROOT / "design" / "shoulder.tg")
    parser.add_argument("--control", type=Path, default=None, help="Control .tg (default: coupled to morphology)")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--speed", type=float, default=0.2, help="Viewer realtime factor")
    parser.add_argument("--speed-log-interval", type=float, default=1.0, help="Wall seconds between speed logs")
    parser.add_argument("--color", action="store_true", help="Color structural tendons to match GUI plots")
    args = parser.parse_args()

    os.environ["MORPHOLOGY_PATH"] = str(args.tg)
    os.environ["MUJOCO_XML_PATH"] = str(args.scene)
    compile_tg(args.tg, args.scene, color_structural_tendons=args.color)

    morph = load_morphology(args.tg)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    control = load_control(args.control) if args.control else load_control_for_morphology(args.tg)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = [get_actuator_id(model, name) for name in ACTUATORS]

    def step_forces(_model, _data) -> None:
        aero.apply(model, data)
        pitch.apply(model, data)
        data.ctrl[:] = 0.0
        base_ctrl = morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids:
            if aid >= 0:
                data.ctrl[aid] = base_ctrl

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sync = RealtimeSync(data, realtime_factor=args.speed)
        configured_slowdown = 1.0 / sync.realtime_factor
        print(
            f"viewer_speed target={sync.realtime_factor:.3f}x realtime "
            f"slowdown={configured_slowdown:.3f}x "
            f"control_f0={control.fundamental_frequency_hz:.6g}Hz",
            flush=True,
        )
        next_speed_log = time.time() + max(0.1, args.speed_log_interval)
        while viewer.is_running():
            sync.step_and_sync(model, data, viewer, pre_mj_step_hook=step_forces)
            now = time.time()
            if now >= next_speed_log:
                wall_elapsed = now - sync.wall_start
                sim_elapsed = data.time - sync.sim_start
                realtime = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 0.0
                slowdown = wall_elapsed / sim_elapsed if sim_elapsed > 0 else float("inf")
                print(
                    f"viewer_speed actual={realtime:.3f}x realtime "
                    f"slowdown={slowdown:.3f}x sim_t={data.time:.3f}s",
                    flush=True,
                )
                next_speed_log = now + max(0.1, args.speed_log_interval)


if __name__ == "__main__":
    main()
