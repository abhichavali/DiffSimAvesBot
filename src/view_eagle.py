"""View (or record) the AvesEagle three-linkage wing flapping.

    mjpython src/view_eagle.py
    uv run python src/view_eagle.py --gif out/eagle.gif

The tensegrity shoulder flaps the humerus exactly as it flaps the hummingbird's
plate wing; `runtime.eagle_drive` adds the elbow/wrist fold and the feathering
twist, both phase-locked to the crank and both calibrated against this scene
before the run starts.
"""

from __future__ import annotations

import argparse
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

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.eagle_drive import calibrated_drive, drive_cranks
from runtime.viewer_runtime import RealtimeSync


def build(args):
    os.environ["MORPHOLOGY_PATH"] = str(args.tg)
    os.environ["MUJOCO_XML_PATH"] = str(args.scene)
    compile_tg(args.tg, args.scene)
    morph = load_morphology(args.tg)
    if morph.wing.bones is None:
        raise SystemExit(f"{args.tg} has no `bones:` section — that is the hummingbird plate wing")
    control = load_control_for_morphology(args.tg)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    sync_crank_disks_at_rest(model, data)
    drive = calibrated_drive(
        model,
        morph,
        control,
        fold=args.fold,
        twist=math.radians(args.twist),
        level=args.level,
        lead=math.radians(args.lead),
    )
    print(
        f"stroke phase (top of upstroke) {math.degrees(drive.calib.top_phase) % 360:.1f} deg; "
        f"flap {control.fundamental_frequency_hz:.3g} Hz; wing half-span "
        f"{morph.wing.bones.span * 1000:.0f} mm",
        flush=True,
    )
    return model, data, morph, control, drive


def step(model, data, morph, control, drive) -> None:
    data.ctrl[:] = 0.0
    drive_cranks(model, data, morph, control, 0.0, 2.0 * control.period)
    drive.apply(data)


def record_gif(args, model, data, morph, control, drive) -> None:
    from PIL import Image

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    period = control.period
    start = args.settle * period
    frames, taken = [], 0
    total = int(args.gif_cycles * args.fps / control.fundamental_frequency_hz)
    while taken < total:
        step(model, data, morph, control, drive)
        mujoco.mj_step(model, data)
        if data.time >= start + taken / args.fps:
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(Image.fromarray(renderer.render()))
            taken += 1
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
    )
    print(f"wrote {args.gif} ({len(frames)} frames, {args.width}x{args.height})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tg", type=Path, default=ROOT / "design" / "eagle.tg")
    p.add_argument("--scene", type=Path, default=ROOT / "eagle_scene.xml")
    p.add_argument("--speed", type=float, default=1.0, help="Viewer realtime factor")
    p.add_argument("--fold", type=float, default=0.60, help="Elbow/wrist flexion, 0..1 of range")
    p.add_argument("--twist", type=float, default=14.0, help="Peak pronation (deg)")
    p.add_argument("--lead", type=float, default=35.0, help="Fold lead ahead of the flap (deg)")
    p.add_argument("--level", type=float, default=1.0, help="Roll cancellation, 0..1")
    p.add_argument("--gif", type=Path, default=None, help="Record to this GIF instead of viewing")
    p.add_argument("--gif-cycles", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--settle", type=float, default=5.0, help="Cycles to settle before recording")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=620)
    p.add_argument("--azimuth", type=float, default=145.0)
    p.add_argument("--elevation", type=float, default=-14.0)
    p.add_argument("--distance", type=float, default=0.8)
    p.add_argument("--lookat", default="0.03,0,0.24")
    args = p.parse_args()

    model, data, morph, control, drive = build(args)
    if args.gif is not None:
        record_gif(args, model, data, morph, control, drive)
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sync = RealtimeSync(data, realtime_factor=args.speed)
        next_log = time.time() + 2.0
        while viewer.is_running():
            sync.step_and_sync(
                model,
                data,
                viewer,
                pre_mj_step_hook=lambda m, d: step(m, d, morph, control, drive),
            )
            if time.time() >= next_log:
                elapsed = time.time() - sync.wall_start
                print(f"sim_t={data.time:.2f}s realtime={(data.time - sync.sim_start) / elapsed:.2f}x")
                next_log = time.time() + 2.0


if __name__ == "__main__":
    main()
