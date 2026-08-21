"""View (or record) the AvesEagle three-linkage wing stroking.

    mjpython src/view_eagle.py
    uv run python src/view_eagle.py --gif out/eagle.gif

The tensegrity shoulder swings the humerus exactly as it swings the hummingbird's
plate wing — a wide side-to-side sweep in a near-horizontal stroke plane.
`runtime.eagle_drive` adds the elbow/wrist fold and the wing flip, both
phase-locked to the crank and both calibrated against this scene before the run
starts.
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
from runtime.eagle_drive import CrankDrive, calibrated_drive, check_stable
from runtime.wing_cables import apply_tension, cable_names, release_fold_servos
from runtime.viewer_runtime import RealtimeSync


def _tension_cables(model, morph, args) -> None:
    """Set the wing cables' rest lengths, if the design has any.

    The compiler cannot do this: a cable is tensioned when its spring is shorter
    than the distance its anchors currently span, so the fractions in the .tg only
    mean something once the extended pose exists to measure. Without this the
    cables are emitted but inert.
    """
    cables = morph.wing.bones.cables if morph.wing.bones else None
    if cables is None or not cable_names(model):
        if args.cables not in ("slack", "extend"):
            raise SystemExit(f"{args.tg} declares no `cables:` section, so --cables has nothing to set")
        return
    fractions = {"extend": cables.extend, "contract": cables.contract, "slack": (1.0, 1.0)}[args.cables]
    apply_tension(model, fractions)
    if args.cable_only:
        # Let the cables alone hold the wing: a stiff position servo on the elbow
        # would otherwise decide the pose no matter how they are tensioned.
        release_fold_servos(model)
    print(f"wing cables: {args.cables} (cranial, caudal) = {tuple(fractions)}"
          + ("; elbow servo released" if args.cable_only else ""), flush=True)


def build(args):
    os.environ["MORPHOLOGY_PATH"] = str(args.tg)
    os.environ["MUJOCO_XML_PATH"] = str(args.scene)
    compile_tg(args.tg, args.scene)
    morph = load_morphology(args.tg)
    if morph.wing.bones is None:
        raise SystemExit(f"{args.tg} has no `bones:` section — that is the hummingbird plate wing")
    control = load_control_for_morphology(args.tg)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    _tension_cables(model, morph, args)
    data = mujoco.MjData(model)
    sync_crank_disks_at_rest(model, data)
    if args.paused:
        # Nothing is driven and nothing is stepped, so skip the calibration
        # rollouts too — they exist to schedule motion there is none of. This is
        # the compiled design exactly as the .tg describes it.
        mujoco.mj_forward(model, data)
        print(
            f"paused at the compiled rest pose; wing half-span "
            f"{morph.wing.bones.span * 1000:.0f} mm",
            flush=True,
        )
        return model, data, None, control, None
    cranks = CrankDrive(model, morph, control)
    extra = {} if args.attack is None else {"attack": math.radians(args.attack)}
    drive = calibrated_drive(
        model,
        morph,
        control,
        fold=args.fold,
        twist=math.radians(args.twist),
        level=args.level,
        lead=math.radians(args.lead),
        **extra,
    )
    calib = drive.calib
    print(
        f"stroke: {calib.axis} — {calib.sweep_deg:.0f} deg of sweep against "
        f"{calib.elevation_deg:.0f} deg of elevation, reversing at crank "
        f"{math.degrees(calib.reversal_phase) % 360:.0f} deg; "
        f"{control.fundamental_frequency_hz:.3g} Hz; wing half-span "
        f"{morph.wing.bones.span * 1000:.0f} mm; wing plane held at "
        f"{math.degrees(drive.attack):.0f} deg (0 = flat, 90 = on edge)",
        flush=True,
    )
    return model, data, cranks, control, drive


def step(model, data, cranks, drive) -> None:
    """Command one physics step. ``cranks is None`` means paused — do nothing."""
    if cranks is None:
        return
    # check_stable here too, not just on the recording path: MuJoCo auto-resets a
    # diverged model and rewinds the clock, so without it an interactive session
    # on a marginal timestep silently restarts instead of saying anything.
    check_stable(model, data, "viewer")
    data.ctrl[:] = 0.0
    cranks.apply(data)
    drive.apply(data)


def record_gif(args, model, data, cranks, control, drive) -> None:
    from PIL import Image

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    if cranks is None:  # paused: one still frame, no settling and no stepping
        renderer.update_scene(data, camera=cam, scene_option=opt)
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(args.gif)
        print(f"wrote {args.gif} (1 still frame, {args.width}x{args.height})")
        return
    period = 1.0 / control.fundamental_frequency_hz
    frames, taken = [], 0
    total = int(args.gif_cycles * args.fps / control.fundamental_frequency_hz)
    # Step-counted, not clock-counted: MuJoCo rewinds data.time when it resets a
    # diverged model, and a clock-driven loop then never finishes.
    start_step = round(args.settle * period / model.opt.timestep)
    every = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    stepped = 0
    while taken < total:
        step(model, data, cranks, drive)
        mujoco.mj_step(model, data)
        stepped += 1
        if stepped >= start_step and (stepped - start_step) % every == 0:
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
    p.add_argument("--fold", type=float, default=0.35, help="Elbow/wrist flexion at the stroke reversals, 0..1 of range")
    p.add_argument("--twist", type=float, default=30.0, help="Peak wing flip between half-strokes (deg)")
    p.add_argument("--lead", type=float, default=35.0, help="Fold lead ahead of the stroke (deg)")
    p.add_argument("--level", type=float, default=1.0, help="Roll-hold authority, 0..1")
    p.add_argument("--attack", type=float, default=None,
                   help="Wing-plane roll to hold (deg). 0 lies flat and slices the sweep, "
                        "~90 stands the wing on edge so it paddles. Default: the mount's own roll")
    p.add_argument("--cables", choices=("slack", "extend", "contract"), default="slack",
                   help="Wing-cable tension preset from the .tg (default: slack)")
    p.add_argument("--cable-only", action="store_true",
                   help="Release the elbow servo so the cables alone hold the wing's pose")
    p.add_argument("--paused", action="store_true",
                   help="Freeze at the compiled rest pose: nothing driven, no physics stepped")
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

    model, data, cranks, control, drive = build(args)
    if args.gif is not None:
        record_gif(args, model, data, cranks, control, drive)
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sync = RealtimeSync(data, realtime_factor=args.speed)
        next_log = time.time() + 2.0
        while viewer.is_running():
            if cranks is None:
                viewer.sync()
                time.sleep(1.0 / 60.0)
                continue
            sync.step_and_sync(
                model,
                data,
                viewer,
                pre_mj_step_hook=lambda m, d: step(m, d, cranks, drive),
            )
            if time.time() >= next_log:
                elapsed = time.time() - sync.wall_start
                print(f"sim_t={data.time:.2f}s realtime={(data.time - sync.sim_start) / elapsed:.2f}x")
                next_log = time.time() + 2.0


if __name__ == "__main__":
    main()
