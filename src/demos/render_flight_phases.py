"""Render ONE multimodal clip: hover -> forward flight -> glide.

  1. hover    — level stroke plane (disk pitch 0), full flapping.
  2. cruise   — the stroke plane tilts forward; flapping continues.
  3. glide    — the motor drive fades out, the cranks damp to a stop so the wings
                settle extended, and the stroke plane swings to 90 deg forward.

Free flight is uncontrolled until the closed-loop hover policy drives it.

The skin runs the opposite fade to render_configs.py: it starts nearly invisible
(bare tensegrity joint) and fills in as the flight progresses.

Scene build and ffmpeg pipe come from common.py; the aero, control, crank-sync,
pitch and disk-orientation runtimes are the same ones the viewer uses.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402

import flying  # noqa: E402
from common import (  # noqa: E402
    airframe_xml,
    ffmpeg_pipe,
    lift_free_bodies,
    save_still,
    scene_option,
    smoothstep,
)
from design_paths import DEFAULT_HUMMER_TG  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.control_signal import (  # noqa: E402
    eval_control,
    load_control_for_morphology,
    ramp_factor,
)
from runtime.disk_orientation import DiskOrientation  # noqa: E402
from runtime.sim import PlateWingSim  # noqa: E402

# Skin materials, faded together so the whole bird materializes over the joint.
# (torso ends translucent so the mechanism still shows through at the end).
SKIN_MATS = {"bird_body_mat": 0.75, "bird_mat": 1.0, "bird_wing_mat": 1.0}


class Schedule:
    """Phase clock in sim seconds since the drive started."""

    def __init__(self, args):
        self.hover = args.hover
        self.transition = args.transition
        self.cruise = args.cruise
        self.glide = args.glide
        self.t_fwd = self.hover                                  # start tilting forward
        self.t_glide = self.hover + self.transition + self.cruise
        self.total = self.t_glide + self.glide
        self.a = args

    def stroke_pitch(self, dt: float) -> float:
        """0 (hover) -> cruise_pitch -> 90 deg (glide), in radians."""
        fwd = self.a.cruise_pitch * smoothstep((dt - self.t_fwd) / self.transition)
        to90 = (math.pi / 2 - self.a.cruise_pitch) * smoothstep(
            (dt - self.t_glide) / self.a.tilt90_seconds
        )
        return fwd + to90

    def drive(self, dt: float) -> float:
        return 1.0 - smoothstep((dt - self.t_glide) / self.a.stop_seconds)

    def skin_alpha(self, dt: float) -> float:
        return smoothstep(dt / (self.a.fill_frac * self.total))


def render(args):
    model = mujoco.MjModel.from_xml_string(
        flying.add_visual_bird(airframe_xml(args.tg, args.scene), assets_dir=args.bird_assets))
    morph = load_morphology(args.tg)
    control = load_control_for_morphology(args.tg)
    sched = Schedule(args)

    sim = PlateWingSim(model, morph)
    data = sim.data
    lift_free_bodies(model, data, args.lift)

    motor_dir = morph.motors.motor_direction
    ramp_seconds = 2.0 * control.period
    drive_start = data.time

    disk = DiskOrientation(model)
    airframe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flying.AIRFRAME_BODY)
    air_dofadr = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)])
    z0 = float(data.subtree_com[airframe_bid][2])
    dt = float(model.opt.timestep)

    def physics_step():
        t = data.time - drive_start
        disk.set(0.0, sched.stroke_pitch(t))
        # Drive: normal flap, then fade to zero. ctrl 0 leaves the velocity servos
        # as pure dampers, so the cranks coast to a stop and the tendon net pulls
        # the wings back out to their extended rest pose.
        env = ramp_factor(data.time, drive_start, ramp_seconds) * sched.drive(t)
        sim.step(env * motor_dir * eval_control(control, data.time))

    mats = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in SKIN_MATS}
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = airframe_bid
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    scn_opt = scene_option()

    steps_per_frame = max(1, round((args.realtime_factor / args.fps) / dt))
    n_frames = int(sched.total / args.realtime_factor * args.fps)
    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / f"{args.name}.mp4"
    proc = ffmpeg_pipe(mp4, args.width, args.height, args.fps)
    still_frames = {int(n_frames * f) for f in args.still_fracs}

    print(f"[{args.name}] {n_frames} frames, {sched.total:.2f}s sim "
          f"({steps_per_frame} steps/frame) -> {mp4}", flush=True)
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            physics_step()
        t = data.time - drive_start
        u = sched.skin_alpha(t)
        for name, full in SKIN_MATS.items():
            if mats[name] >= 0:
                model.mat_rgba[mats[name]][3] = args.alpha0 + (full - args.alpha0) * u
        # Same curve as the skin fade: tight on the bare joint, pulling back to the
        # whole bird exactly as the opacity hardens.
        cam.distance = args.distance + args.zoom_out * u
        # Climb over the glide: the wings are flat and horizontal by then, and the
        # near one is edge-on (a sliver) from the flapping camera. Looking down on
        # the planform is what makes "wings extended" read.
        cam.elevation = args.elevation + args.elev_drop * smoothstep(
            (t - sched.t_glide) / sched.glide)
        renderer.update_scene(data, camera=cam, scene_option=scn_opt)
        px = renderer.render()
        proc.stdin.write(px.tobytes())
        if frame in still_frames:
            save_still(args.out / f"{args.name}_f{frame:04d}.png", px)
        if frame % 15 == 0:
            # signed nose-down lean: z-component of the body x axis in world.
            lean = math.degrees(math.asin(max(-1.0, min(1.0, -data.xmat[airframe_bid][6]))))
            print(f"  frame {frame}/{n_frames} t={t:.2f}s "
                  f"pitch={math.degrees(sched.stroke_pitch(t)):.0f}deg "
                  f"x={data.subtree_com[airframe_bid][0]:+.3f} "
                  f"vx={data.qvel[air_dofadr]:+.2f} "
                  f"dz={data.subtree_com[airframe_bid][2] - z0:+.4f} "
                  f"lean={lean:+.0f}deg", flush=True)
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"[{args.name}] done: {mp4}", flush=True)


def self_check(args):
    """The schedule is the only real logic here — assert each phase is what it says."""
    s = Schedule(args)
    deg = math.degrees
    assert deg(s.stroke_pitch(0.0)) == 0.0 and s.drive(0.0) == 1.0, "hover: level + flapping"
    mid = s.t_glide - args.cruise / 2
    assert abs(deg(s.stroke_pitch(mid)) - 30.0) < 1e-6, "cruise: 30 deg forward tilt"
    assert s.drive(mid) == 1.0, "cruise: still flapping"
    end = s.total
    assert abs(deg(s.stroke_pitch(end)) - 90.0) < 1e-6, "glide: stroke plane 90 deg"
    assert s.drive(end) == 0.0, "glide: motors off"
    assert s.skin_alpha(0.0) == 0.0 and s.skin_alpha(end) == 1.0, "skin fades in"
    print("render_flight_phases self-check OK")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true", help="assert the phase schedule, no render")
    p.add_argument("--name", default="flight_hover_cruise_glide")
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "out")
    p.add_argument("--bird-assets", default=flying.BIRD_ASSETS_DIR)
    # phase durations, sim seconds
    p.add_argument("--hover", type=float, default=1.4)
    p.add_argument("--transition", type=float, default=0.8)
    p.add_argument("--cruise", type=float, default=2.2)
    p.add_argument("--glide", type=float, default=4.0)
    p.add_argument("--cruise-pitch", type=float, default=math.radians(30.0),
                   help="forward stroke-plane tilt at cruise (rad)")
    p.add_argument("--tilt90-seconds", type=float, default=0.5,
                   help="time to swing the stroke plane from cruise tilt to 90 deg")
    p.add_argument("--stop-seconds", type=float, default=0.25, help="motor fade-out (s)")
    p.add_argument("--lift", type=float, default=0.55,
                   help="initial airframe height above the compiled pose (m)")
    # skin fade-IN (opposite of render_configs)
    p.add_argument("--alpha0", type=float, default=0.05)
    p.add_argument("--fill-frac", type=float, default=0.8,
                   help="fraction of the clip over which the skin fills in")
    # camera / output
    p.add_argument("--azimuth", type=float, default=130.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--elev-drop", type=float, default=-22.0,
                   help="camera elevation change over the glide (deg)")
    # The tracking camera centres the airframe COM, which sits below the crank
    # disks -- closer than ~0.4 and the mechanism runs off the top of the frame.
    p.add_argument("--distance", type=float, default=0.40, help="opening framing, on the joint")
    p.add_argument("--zoom-out", type=float, default=0.65, help="pull-back by the time the skin is solid")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--realtime-factor", type=float, default=0.32)
    p.add_argument("--still-fracs", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.05, 0.25, 0.45, 0.6, 0.8, 0.97])
    args = p.parse_args()
    (self_check if args.self_check else render)(args)


if __name__ == "__main__":
    main()
