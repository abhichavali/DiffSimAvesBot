from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import mujoco.viewer

from compile_tg import compile_tg
from morphology.model import PLATE_BIRDS, require_bird
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control, load_control_for_morphology, ramp_factor
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.disk_orientation import DiskOrientation
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
    parser.add_argument(
        "--stroke-plane-demo",
        action="store_true",
        help="Sweep the disk mount (tilt, pitch) in a slow circle to show the stroke plane moving",
    )
    parser.add_argument("--stroke-plane-amp", type=float, default=0.15, help="Demo sweep amplitude (rad)")
    parser.add_argument("--stroke-plane-hz", type=float, default=0.25, help="Demo sweep frequency (Hz)")
    args = parser.parse_args()

    os.environ["MORPHOLOGY_PATH"] = str(args.tg)
    os.environ["MUJOCO_XML_PATH"] = str(args.scene)
    compile_tg(args.tg, args.scene, color_structural_tendons=args.color)

    morph = load_morphology(args.tg)
    require_bird(morph, *PLATE_BIRDS, tool="view_fixed_frequency")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    # A learned stroke-plane signal next to the design (optimizer artifact) is
    # applied automatically; --stroke-plane-demo overrides it with the test circle.
    learned_sp = None
    sp_file = args.tg.parent / "strokeplane_optimized.json"
    if not args.stroke_plane_demo and sp_file.exists():
        spd = json.loads(sp_file.read_text())
        _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spd["phase_ref_joint"])
        learned_sp = (DiskOrientation(model), int(model.jnt_qposadr[_jid]),
                      spd["tilt_coeffs"] + spd["pitch_coeffs"],
                      float(spd["tilt0_rad"]), float(spd["pitch0_rad"]),
                      float(spd["max_delta_rad"]))
        print(f"applying learned stroke-plane signal from {sp_file}")

    disk_orient = None
    if args.stroke_plane_demo:
        disk_orient = DiskOrientation(model)
        tilt0 = morph.motors.disk.tilt_rad
        pitch0 = morph.motors.disk.pitch_rad
        # Round-trip check: at the baseline angles we must reproduce the
        # compiled body_quat exactly (guards the euler-seq/mirror convention).
        import numpy as np

        body_ids = [bid for bid, _ in disk_orient.bodies]
        compiled_quats = model.body_quat[body_ids].copy()
        disk_orient.set(tilt0, pitch0)
        assert np.allclose(model.body_quat[body_ids], compiled_quats, atol=1e-9)

    control = load_control(args.control) if args.control else load_control_for_morphology(args.tg)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = [get_actuator_id(model, name) for name in ACTUATORS]
    drive_start = data.time
    ramp_seconds = 2.0 * control.period

    def step_forces(_model, _data) -> None:
        if learned_sp is not None:
            do, qadr, coef, t0, p0, mx_ = learned_sp
            th = float(data.qpos[qadr])
            basis = [1.0, math.cos(th), math.sin(th), math.cos(2.0 * th), math.sin(2.0 * th)]
            raw_t = sum(c * b for c, b in zip(coef[:5], basis))
            raw_p = sum(c * b for c, b in zip(coef[5:], basis))
            do.set(t0 + mx_ * math.tanh(raw_t / mx_), p0 + mx_ * math.tanh(raw_p / mx_))
        if disk_orient is not None:
            # Circle in (tilt, pitch): both DOFs sweep, the stroke-plane
            # normal precesses — the eventual learned signal replaces this.
            phase = 2.0 * math.pi * args.stroke_plane_hz * (data.time - drive_start)
            disk_orient.set(
                tilt0 + args.stroke_plane_amp * math.sin(phase),
                pitch0 + args.stroke_plane_amp * math.sin(phase + math.pi / 2.0),
            )
        aero.apply(model, data)
        pitch.apply(model, data)
        data.ctrl[:] = 0.0
        ramp = ramp_factor(data.time, drive_start, ramp_seconds)
        base_ctrl = ramp * morph.motors.motor_direction * eval_control(control, data.time)
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
