"""Render ONE multimodal clip: hover -> forward flight -> glide.

  1. hover    — level stroke plane (disk pitch 0), full flapping, station keeping.
  2. cruise   — the stroke plane tilts forward and a soft speed servo brings the
                airframe up to cruise; flapping continues.
  3. glide    — the motor drive fades out, the cranks damp to a stop so the wings
                settle extended, the stroke plane swings to 90 deg forward, and
                the bird sinks slowly on the airflow over the wings.

The skin runs the opposite fade to render_configs.py: it starts nearly invisible
(bare tensegrity joint) and fills in as the flight progresses.

Reuses render_configs for the scene build / ffmpeg pipe; the aero, control,
crank-sync, pitch and disk-orientation runtimes are the same ones the viewer uses.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402

import flying  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from render_configs import ACTUATORS, _ffmpeg, _save_still, build_model  # noqa: E402
from runtime.aero import AeroBladeRuntime  # noqa: E402
from runtime.control_signal import (  # noqa: E402
    eval_control,
    load_control_for_morphology,
    ramp_factor,
)
from runtime.crank import sync_crank_disks_at_rest  # noqa: E402
from runtime.disk_orientation import DiskOrientation  # noqa: E402
from runtime.morphology_runtime import DEFAULT_HUMMER_TG  # noqa: E402
from runtime.realtime import get_actuator_id  # noqa: E402
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest  # noqa: E402

# The nose points +x (bird_head_geom at x=+0.14, tail at -0.09) and the wings
# span +-y, so "forward" is +x, the stroke plane tilts about y (disk pitch), and a
# positive rotation about +y is nose-down (the forward lean of cruising flight).
PITCH_AXIS = np.array([0.0, 1.0, 0.0])

# Skin materials, faded together so the whole bird materializes over the joint.
# (torso ends translucent so the mechanism still shows through at the end).
SKIN_MATS = {"bird_body_mat": 0.75, "bird_mat": 1.0, "bird_wing_mat": 1.0}


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


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

    def speed_target(self, dt: float) -> float:
        return self.a.cruise_speed * smoothstep((dt - self.t_fwd) / self.transition)

    def lean(self, dt: float) -> float:
        """Commanded body pitch: slightly nose-up hovering, nose-down once moving."""
        h, c = self.a.hover_lean, self.a.cruise_lean
        return h + (c - h) * smoothstep((dt - self.t_fwd) / self.transition)

    def bob(self, dt: float):
        """Station-keeping wander (sway, rise) in metres, in phase with the drive.

        The physical bob at stroke frequency is a fraction of a millimetre, so it
        reads as a bird welded to an invisible floor. This is the slow hunting a
        hovering bird actually shows; it dies with the motors at the glide.
        """
        e = self.drive(dt)
        w = 2.0 * math.pi * self.a.bob_hz
        return (self.a.sway_amp * e * math.sin(0.61 * w * dt),
                self.a.bob_amp * e * math.sin(w * dt))

    def sink(self, dt: float) -> float:
        """Metres of altitude given up so far (glide only)."""
        return self.a.sink_rate * max(0.0, dt - self.t_glide)

    def skin_alpha(self, dt: float) -> float:
        return smoothstep(dt / (self.a.fill_frac * self.total))


def render(args):
    model = build_model("flying", args.tg, args.scene,
                        skinned=True, bird_assets=args.bird_assets)
    morph = load_morphology(args.tg)
    control = load_control_for_morphology(args.tg)
    sched = Schedule(args)

    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[int(model.jnt_qposadr[j]) + 2] += args.lift
    mujoco.mj_forward(model, data)

    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = [get_actuator_id(model, n) for n in ACTUATORS]
    motor_dir = morph.motors.motor_direction
    ramp_seconds = 2.0 * control.period
    drive_start = data.time

    disk = DiskOrientation(model)
    airframe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flying.AIRFRAME_BODY)
    air_dofadr = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)])
    m_total = float(model.body_subtreemass[airframe_bid])
    g = float(-model.opt.gravity[2])
    z0 = float(data.subtree_com[airframe_bid][2])
    prev_z = [z0]
    dt = float(model.opt.timestep)
    qconj = np.zeros(4)
    q_tgt = np.zeros(4)
    q_err = np.zeros(4)
    att_err = np.zeros(3)
    x_ref = [float(data.subtree_com[airframe_bid][0])]

    def force_hook():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        t = data.time - drive_start
        disk.set(0.0, sched.stroke_pitch(t))
        aero.apply(model, data)
        pitch.apply(model, data)

        # Flight assist at the airframe COM (no spurious torque): weight cancel +
        # altitude hold about a sinking setpoint, and a soft speed servo forward.
        # ponytail: this is the "artificially controlled" part -- the stroke plane
        # tilt tells the story, the servo guarantees the shot.
        com = np.asarray(data.subtree_com[airframe_bid])
        vz = (float(com[2]) - prev_z[0]) / dt
        prev_z[0] = float(com[2])
        vel = data.qvel[air_dofadr : air_dofadr + 3]
        # Track a moving setpoint, not just a speed: a pure velocity servo settles
        # at a standing offset against the flapping thrust, so "hover" crept
        # forward at ~0.6 m/s. The integrated reference pins station keeping.
        x_ref[0] += sched.speed_target(t) * dt
        sway, rise = sched.bob(t)
        f = np.array([
            m_total * (args.pos_gain * (x_ref[0] - float(com[0]))
                       + args.speed_gain * (sched.speed_target(t) - float(vel[0]))),
            m_total * (args.pos_gain * (sway - float(com[1]))
                       - args.speed_gain * float(vel[1])),
            m_total * g + args.z_k * (z0 + rise - sched.sink(t) - float(com[2]))
            - args.z_c * vz,
        ])
        # Attitude PD toward a commanded lean (upright hovering, nose-down once
        # moving forward). Without it the forward stroke-plane tilt pitches the
        # airframe past 90 deg and it tumbles. Free-joint angular qvel is
        # body-local, mj_applyFT wants a world torque -> rotate it out.
        rot = np.asarray(data.xmat[airframe_bid]).reshape(3, 3)
        mujoco.mju_axisAngle2Quat(q_tgt, PITCH_AXIS, sched.lean(t))
        mujoco.mju_negQuat(qconj, data.xquat[airframe_bid])
        mujoco.mju_mulQuat(q_err, q_tgt, qconj)
        mujoco.mju_quat2Vel(att_err, q_err, 1.0)
        tau = args.level_kp * att_err - args.level_kd * (
            rot @ data.qvel[air_dofadr + 3 : air_dofadr + 6]
        )
        mujoco.mj_applyFT(model, data, f, tau, com, airframe_bid, data.qfrc_applied)

        # Drive: normal flap, then fade to zero. ctrl 0 leaves the velocity servos
        # as pure dampers, so the cranks coast to a stop and the tendon net pulls
        # the wings back out to their extended rest pose.
        env = ramp_factor(data.time, drive_start, ramp_seconds) * sched.drive(t)
        base = env * motor_dir * eval_control(control, data.time)
        for aid in actuator_ids:
            if aid >= 0:
                data.ctrl[aid] = base

    mats = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in SKIN_MATS}
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = airframe_bid
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    scn_opt = mujoco.MjvOption()
    for gi in range(len(scn_opt.geomgroup)):
        scn_opt.geomgroup[gi] = 1 if gi <= 2 else 0
    for si in range(len(scn_opt.sitegroup)):
        scn_opt.sitegroup[si] = 0

    steps_per_frame = max(1, round((args.realtime_factor / args.fps) / dt))
    n_frames = int(sched.total / args.realtime_factor * args.fps)
    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / f"{args.name}.mp4"
    proc = _ffmpeg(mp4, args.width, args.height, args.fps)
    still_frames = {int(n_frames * f) for f in args.still_fracs}

    print(f"[{args.name}] {n_frames} frames, {sched.total:.2f}s sim "
          f"({steps_per_frame} steps/frame) -> {mp4}", flush=True)
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            force_hook()
            mujoco.mj_step(model, data)
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
            _save_still(args.out / f"{args.name}_f{frame:04d}.png", px)
        if frame % 15 == 0:
            # signed nose-down lean: z-component of the body x axis in world.
            lean = math.degrees(math.asin(max(-1.0, min(1.0, -data.xmat[airframe_bid][6]))))
            print(f"  frame {frame}/{n_frames} t={t:.2f}s "
                  f"pitch={math.degrees(sched.stroke_pitch(t)):.0f}deg "
                  f"x={data.subtree_com[airframe_bid][0]:+.3f} "
                  f"vx={data.qvel[air_dofadr]:+.2f} "
                  f"dz={data.subtree_com[airframe_bid][2] - z0:+.4f} "
                  f"lean={lean:+.0f}deg(cmd {math.degrees(sched.lean(t)):+.0f})", flush=True)
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"[{args.name}] done: {mp4}", flush=True)


def self_check(args):
    """The schedule is the only real logic here — assert each phase is what it says."""
    s = Schedule(args)
    deg = math.degrees
    assert deg(s.stroke_pitch(0.0)) == 0.0 and s.drive(0.0) == 1.0, "hover: level + flapping"
    assert s.speed_target(s.t_fwd) == 0.0, "hover: station keeping"
    assert s.lean(s.t_fwd) == args.hover_lean < args.cruise_lean, "hover leans back of cruise"
    assert s.bob(0.0) == (0.0, 0.0), "bob starts at zero -- no startup jump"
    rise = [s.bob(0.02 * i)[1] for i in range(int(s.t_fwd / 0.02))]
    assert max(rise) > 0.9 * args.bob_amp and min(rise) < -0.9 * args.bob_amp, "hover bobs"
    mid = s.t_glide - args.cruise / 2
    assert abs(deg(s.stroke_pitch(mid)) - 30.0) < 1e-6, "cruise: 30 deg forward tilt"
    assert abs(s.lean(mid) - args.cruise_lean) < 1e-9, "cruise: leaning forward"
    assert abs(s.speed_target(mid) - args.cruise_speed) < 1e-9 and s.drive(mid) == 1.0
    end = s.total
    assert abs(deg(s.stroke_pitch(end)) - 90.0) < 1e-6, "glide: stroke plane 90 deg"
    assert s.drive(end) == 0.0 and s.sink(end) > 0.0, "glide: motors off, sinking"
    assert s.sink(s.t_glide) == 0.0, "no sink before the glide"
    assert s.bob(end) == (0.0, 0.0), "glide: bob off with the motors"
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
    p.add_argument("--hover", type=float, default=1.4, help="~2 bob cycles at --bob-hz")
    p.add_argument("--transition", type=float, default=0.8)
    p.add_argument("--cruise", type=float, default=2.2)
    p.add_argument("--glide", type=float, default=4.0)
    p.add_argument("--cruise-pitch", type=float, default=math.radians(30.0),
                   help="forward stroke-plane tilt at cruise (rad)")
    p.add_argument("--tilt90-seconds", type=float, default=0.5,
                   help="time to swing the stroke plane from cruise tilt to 90 deg")
    p.add_argument("--stop-seconds", type=float, default=0.25, help="motor fade-out (s)")
    p.add_argument("--cruise-speed", type=float, default=1.5, help="forward cruise speed (m/s)")
    p.add_argument("--speed-gain", type=float, default=20.0, help="speed servo gain (1/s)")
    p.add_argument("--pos-gain", type=float, default=80.0, help="station-keeping gain (1/s^2)")
    p.add_argument("--sink-rate", type=float, default=0.08, help="glide descent (m/s)")
    p.add_argument("--cruise-lean", type=float, default=math.radians(22.0),
                   help="commanded nose-down body pitch at cruise (rad)")
    p.add_argument("--hover-lean", type=float, default=math.radians(-6.0),
                   help="commanded body pitch while hovering (rad, negative = nose up)")
    p.add_argument("--bob-amp", type=float, default=0.018, help="hover bob amplitude (m)")
    p.add_argument("--sway-amp", type=float, default=0.008, help="hover lateral wander (m)")
    p.add_argument("--bob-hz", type=float, default=1.5, help="hover bob rate (Hz)")
    p.add_argument("--z-k", type=float, default=20.0)
    p.add_argument("--z-c", type=float, default=0.6)
    # kp has to beat the flapping pitch-up moment or the commanded lean never lands:
    # at 0.3 the body drooped 11 deg nose-down whatever was asked of it.
    p.add_argument("--level-kp", type=float, default=0.9, help="attitude spring (N m/rad)")
    p.add_argument("--level-kd", type=float, default=0.05, help="attitude damping (N m s/rad)")
    # taller hover = more altitude to spend on the glide (sink_rate * glide)
    p.add_argument("--lift", type=float, default=0.55)
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
