"""Render ONE clip: many different wingtip paths out of one unchanged machine.

Nothing about the bird changes across the clip -- same morphology, same tendons,
same control signal. Only two things move:

  * the STROKE PLANE -- the mount angle of the four crank disks. ``pitch``
    swings the plane fore/aft, ``tilt`` rolls it (mirrored left/right).
  * the DRIVE SPEED -- how fast that same control signal is played back.

Each setting is held a few wingbeats, then blended into the next. Two things
make the cause and the effect visible at once:

  * the wingtip TRAIL, recorded in the airframe's own frame, so it is the path
    the bird sees rather than a smear of where the bird has flown;
  * a translucent STROKE-PLANE disk, taken straight from the crank disk's own
    world orientation, so you watch the plane tilt that produced the path.

No force arrows: they overshot the wings and tangled with the trails, and this
clip is about the path, not the load.

The assist holds altitude and attitude ONLY. x/y are free, so a tilted stroke
plane genuinely flies the bird somewhere -- that translation is the point.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402

import flying  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from render_configs import (  # noqa: E402
    ACTUATORS,
    TIP_SITES,
    WING_COLORS,
    _ffmpeg,
    _save_still,
    build_model,
    draw_trail,
)
from runtime.aero import AeroBladeRuntime  # noqa: E402
from runtime.control_signal import (  # noqa: E402
    FourierControl,
    eval_control,
    load_control_for_morphology,
    ramp_factor,
)
from runtime.crank_sync import sync_crank_disks_at_rest  # noqa: E402
from runtime.disk_orientation import DiskOrientation  # noqa: E402
from runtime.morphology_runtime import DEFAULT_HUMMER_TG  # noqa: E402
from runtime.viewer_runtime import get_actuator_id  # noqa: E402
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest  # noqa: E402
from runtime.wing_pitch_limit import WingPitchLimiter  # noqa: E402

PITCH_AXIS = np.array([0.0, 1.0, 0.0])
MAC_FONT = Path("/System/Library/Fonts/Helvetica.ttc")

# (label, stroke-plane tilt deg, stroke-plane pitch deg, drive speed x)
# Ordered so each step changes ONE thing against the one before it.
SEGMENTS = [
    ("level stroke plane", 0.0, 0.0, 1.00),
    ("plane pitched forward", 0.0, 32.0, 1.00),
    ("plane pitched aft", 0.0, -28.0, 1.00),
    ("level plane, slow drive", 0.0, 0.0, 0.60),
    ("level plane, fast drive", 0.0, 0.0, 1.40),
    ("plane rolled", 24.0, 0.0, 1.00),
]


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class Sequence:
    """Which (tilt, pitch, speed) is commanded at time t, and its label."""

    def __init__(self, args, segments=SEGMENTS):
        self.seg = segments
        self.hold = args.hold
        self.blend = args.blend
        self.step = args.hold + args.blend
        self.total = self.step * len(self.seg)

    def _pair(self, t: float):
        i = min(int(t / self.step), len(self.seg) - 1)
        j = min(i + 1, len(self.seg) - 1)
        u = smoothstep((t - i * self.step - self.hold) / self.blend)
        return i, j, u

    def at(self, t: float):
        """(tilt_rad, pitch_rad, speed) -- held flat, then blended into the next."""
        i, j, u = self._pair(t)
        a, b = self.seg[i], self.seg[j]
        return (
            math.radians(a[1] + (b[1] - a[1]) * u),
            math.radians(a[2] + (b[2] - a[2]) * u),
            a[3] + (b[3] - a[3]) * u,
        )

    def label(self, t: float) -> str:
        """The segment being held, or the one being blended toward past halfway."""
        i, j, u = self._pair(t)
        k = j if u > 0.5 else i
        name, tilt, pitch, speed = self.seg[k]
        return f"{name}   pitch {pitch:+.0f}°  tilt {tilt:+.0f}°  drive {speed:.2f}x"

    def windows(self):
        """(label, t_start, t_end) per segment, for burning captions in."""
        return [(self.seg[i][0], i * self.step, (i + 1) * self.step)
                for i in range(len(self.seg))]


def draw_stroke_plane(scene, disk_bid, data, radius, rgba):
    """A translucent disk at the crank disk's REAL world orientation.

    Read back off the model rather than re-derived from the commanded angles, so
    it cannot drift out of agreement with what the cranks are actually doing.
    """
    if scene.ngeom >= scene.maxgeom or disk_bid < 0:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        # cylinder size is (radius, HALF-LENGTH) -- passing radius twice gives a
        # drum that swallows the bird, not a plane.
        np.array([radius, 0.0007, 0.0]),
        np.asarray(data.xpos[disk_bid]).copy(),
        np.asarray(data.xmat[disk_bid]).copy(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render(args):
    model = build_model("flying", args.tg, args.scene, skinned=False,
                        bird_assets=args.bird_assets)
    morph = load_morphology(args.tg)
    control = load_control_for_morphology(args.tg)
    seq = Sequence(args)

    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[int(model.jnt_qposadr[j]) + 2] += args.lift
    mujoco.mj_forward(model, data)

    aero = AeroBladeRuntime(model)
    pitch_limit = WingPitchLimiter(model, data)
    actuator_ids = [get_actuator_id(model, n) for n in ACTUATORS]
    motor_dir = morph.motors.motor_direction
    drive_start = data.time
    ramp_seconds = 2.0 * control.period

    disk = DiskOrientation(model)
    disk_bid = disk.bodies[0][0] if disk.bodies else -1
    airframe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flying.AIRFRAME_BODY)
    air_dofadr = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)])
    tip_ids = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
               for s, n in TIP_SITES.items()}
    missing = [s for s, i in tip_ids.items() if i < 0]
    if missing:
        raise ValueError(f"missing wingtip sites: {missing}")

    m_total = float(model.body_subtreemass[airframe_bid])
    g = float(-model.opt.gravity[2])
    z0 = float(data.subtree_com[airframe_bid][2])
    prev_z = [z0]
    dt = float(model.opt.timestep)
    psi = [0.0]  # accumulated drive phase (rad)
    qconj, q_tgt, q_err, att_err = (np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3))

    # Sample the trail off the physics clock, not the frame clock: one flap is
    # ~`control.period`, and a 30 fps frame is a tenth of that, so a per-frame
    # trail is a hexagon rather than a stroke path.
    trail_every = max(1, round(control.period / (dt * args.trail_pts)))
    keep = int(args.trail_pts * args.trail_cycles)
    trail = {s: deque(maxlen=keep) for s in TIP_SITES}
    step_n = [0]

    def force_hook():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        t = data.time - drive_start
        tilt, sp_pitch, speed = seq.at(t)
        disk.set(tilt, sp_pitch)
        aero.apply(model, data)
        pitch_limit.apply(model, data)

        com = np.asarray(data.subtree_com[airframe_bid])
        vz = (float(com[2]) - prev_z[0]) / dt
        prev_z[0] = float(com[2])
        # Weight cancel + altitude hold ONLY. No x/y station keeping: a tilted
        # stroke plane is supposed to take the bird somewhere, and hiding that
        # under a position servo would hide the entire point of the clip.
        f = np.array([0.0, 0.0,
                      m_total * g + args.z_k * (z0 - float(com[2])) - args.z_c * vz])
        rot = np.asarray(data.xmat[airframe_bid]).reshape(3, 3)
        mujoco.mju_axisAngle2Quat(q_tgt, PITCH_AXIS, args.body_lean)
        mujoco.mju_negQuat(qconj, data.xquat[airframe_bid])
        mujoco.mju_mulQuat(q_err, q_tgt, qconj)
        mujoco.mju_quat2Vel(att_err, q_err, 1.0)
        tau = args.level_kp * att_err - args.level_kd * (
            rot @ data.qvel[air_dofadr + 3 : air_dofadr + 6]
        )
        mujoco.mj_applyFT(model, data, f, tau, com, airframe_bid, data.qfrc_applied)

        # Integrate the drive PHASE rather than scaling time. eval_control(t*speed)
        # teleports the crank the instant speed changes -- a step input onto
        # near-zero-inertia disks. Accumulating phase changes the rate only.
        psi[0] += control.omega * speed * dt
        env = ramp_factor(data.time, drive_start, ramp_seconds)
        base = env * motor_dir * eval_control(control, psi[0] / control.omega)
        for aid in actuator_ids:
            if aid >= 0:
                data.ctrl[aid] = base

        step_n[0] += 1
        if step_n[0] % trail_every == 0:
            rot_t = rot.T
            for side, sid in tip_ids.items():
                trail[side].append(rot_t @ (np.asarray(data.site_xpos[sid]) - com))

    renderer = mujoco.Renderer(model, height=args.height, width=args.width,
                               max_geom=args.max_geom)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = airframe_bid
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.distance = args.distance
    scn_opt = mujoco.MjvOption()
    for gi in range(len(scn_opt.geomgroup)):
        scn_opt.geomgroup[gi] = 1 if gi <= 2 else 0
    for si in range(len(scn_opt.sitegroup)):
        scn_opt.sitegroup[si] = 0

    steps_per_frame = max(1, round((args.realtime_factor / args.fps) / dt))
    n_frames = int(seq.total / args.realtime_factor * args.fps)
    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / f"{args.name}.mp4"
    proc = _ffmpeg(mp4, args.width, args.height, args.fps)
    caption = _captioner(args.height, args.no_captions)
    still_frames = {int(n_frames * f) for f in args.still_fracs}

    print(f"[{args.name}] {n_frames} frames, {seq.total:.2f}s sim "
          f"({steps_per_frame} steps/frame, trail every {trail_every} steps) -> {mp4}",
          flush=True)
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            force_hook()
            mujoco.mj_step(model, data)
        t = data.time - drive_start
        renderer.update_scene(data, camera=cam, scene_option=scn_opt)
        com = np.asarray(data.subtree_com[airframe_bid])
        rot = np.asarray(data.xmat[airframe_bid]).reshape(3, 3)
        draw_stroke_plane(renderer.scene, disk_bid, data, args.plane_radius,
                          args.plane_rgba)
        draw_trail(renderer.scene, trail, com, rot, args.trail_width)
        px = caption(renderer.render(), seq.label(t))
        proc.stdin.write(px.tobytes())
        if frame in still_frames:
            _save_still(args.out / f"{args.name}_f{frame:04d}.png", px)
        if frame % 20 == 0:
            tilt, sp_pitch, speed = seq.at(t)
            print(f"  frame {frame}/{n_frames} t={t:.2f}s "
                  f"pitch={math.degrees(sp_pitch):+.0f}deg "
                  f"tilt={math.degrees(tilt):+.0f}deg speed={speed:.2f}x "
                  f"x={com[0]:+.3f} y={com[1]:+.3f} "
                  f"v=({data.qvel[air_dofadr]:+.2f},{data.qvel[air_dofadr + 1]:+.2f}) "
                  f"| {seq.label(t)}", flush=True)
    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"[{args.name}] done: {mp4}", flush=True)


def _captioner(height, disabled):
    """Stamp the caption into the frame itself.

    Not ffmpeg's drawtext: that filter needs an ffmpeg built with libfreetype,
    which the one on this machine is not, and a missing filter kills the encode
    outright. PIL is already a matplotlib dependency, and stamping pixels also
    puts the caption on the saved stills for free.
    """
    if disabled or not MAC_FONT.exists():
        return lambda px, text: px
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(MAC_FONT), max(11, int(height * 0.047)))

    def stamp(px, text):
        img = Image.fromarray(px)
        draw = ImageDraw.Draw(img, "RGBA")
        _, _, tw, th = draw.textbbox((0, 0), text, font=font)
        x, y = (img.width - tw) // 2, img.height - th - int(height * 0.075)
        pad = max(6, int(height * 0.017))
        draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                       fill=(0, 0, 0, 115))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 245))
        return np.asarray(img)

    return stamp


def self_check(args):
    """The sequence and the phase integrator are the only real logic here."""
    seq = Sequence(args)
    for i, (name, tilt, pitch, speed) in enumerate(SEGMENTS):
        mid = i * seq.step + args.hold / 2
        got = seq.at(mid)
        assert abs(math.degrees(got[0]) - tilt) < 1e-9, f"{name}: tilt held flat"
        assert abs(math.degrees(got[1]) - pitch) < 1e-9, f"{name}: pitch held flat"
        assert abs(got[2] - speed) < 1e-9, f"{name}: speed held flat"
        assert name in seq.label(mid), f"{name}: labelled while held"

    # Every segment must differ from its neighbour, or the clip shows nothing.
    for a, b in zip(SEGMENTS, SEGMENTS[1:]):
        assert a[1:] != b[1:], f"{a[0]} -> {b[0]} changes nothing"

    # The blend is monotone between the two settings it joins, never overshooting.
    lo, hi = SEGMENTS[0][2], SEGMENTS[1][2]
    prev = -1e9
    for k in range(21):
        t = args.hold + args.blend * k / 20
        p = math.degrees(seq.at(t)[1])
        assert min(lo, hi) - 1e-9 <= p <= max(lo, hi) + 1e-9, "blend overshoots"
        assert p >= prev - 1e-9, "blend is not monotone"
        prev = p
    assert seq.total == seq.step * len(SEGMENTS)

    # Phase integration vs naive time scaling: only one of them is continuous
    # across a speed change, and the drive lands on near-zero-inertia cranks.
    ctl = FourierControl(harmonics=1, fundamental_frequency_hz=10.0, a0=0.0,
                         a=(1.0,), b=(0.0,))
    dt, t_sw = 1e-4, 0.37
    psi, naive_prev, psi_prev, worst_psi, worst_naive = 0.0, None, None, 0.0, 0.0
    for k in range(8000):
        t = k * dt
        speed = 1.0 if t < t_sw else 1.6
        psi += ctl.omega * speed * dt
        v_psi = eval_control(ctl, psi / ctl.omega)
        v_naive = eval_control(ctl, t * speed)
        if psi_prev is not None and abs(t - t_sw) < 5 * dt:
            worst_psi = max(worst_psi, abs(v_psi - psi_prev))
            worst_naive = max(worst_naive, abs(v_naive - naive_prev))
        psi_prev, naive_prev = v_psi, v_naive
    assert worst_psi < 1e-2, f"integrated phase jumped {worst_psi:.3f} at the speed change"
    assert worst_naive > 10 * worst_psi, "naive scaling was expected to jump; it did not"

    # The trail is stored body-local, so flying the bird must not deform it.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tip, com = np.array([0.1, 0.2, 0.3]), np.array([5.0, -2.0, 1.0])
    local = rot.T @ (tip - com)
    assert np.allclose(com + rot @ local, tip), "trail round-trip is not the identity"
    moved = com + np.array([3.0, 3.0, 0.0])
    assert np.allclose(rot.T @ ((tip + np.array([3.0, 3.0, 0.0])) - moved), local), \
        "pure translation changed the body-local trail"

    print("render_stroke_plane self-check OK")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true", help="assert the sequence, no render")
    p.add_argument("--name", default="stroke_plane_paths")
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "out")
    p.add_argument("--bird-assets", default=flying.BIRD_ASSETS_DIR)
    # In wingbeats, which is the unit that matters here: the flap period is
    # ~0.1 s (10 Hz), so 0.30/0.20 is roughly 3 beats held and 2 beats to morph.
    p.add_argument("--hold", type=float, default=0.30, help="seconds each setting is held")
    p.add_argument("--blend", type=float, default=0.20, help="seconds to morph to the next")
    # trail
    # The tip is far faster at mid-stroke than at the reversals, so a coarse
    # sample rate draws the fast arc as a couple of straight streaks.
    p.add_argument("--trail-pts", type=int, default=115, help="trail samples per flap cycle")
    # exactly one cycle: the loop closes on itself, and overlapping history stops
    # spilling out of frame.
    p.add_argument("--trail-cycles", type=float, default=1.0, help="cycles of trail kept")
    p.add_argument("--trail-width", type=float, default=0.0011)
    # stroke-plane disk
    p.add_argument("--plane-radius", type=float, default=0.026,
                   help="just past the crank disk -- enough to read the plane's tilt "
                        "without the disk competing with the wingtip trails")
    p.add_argument("--plane-rgba", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.98, 0.86, 0.20, 0.30])
    # assist
    p.add_argument("--z-k", type=float, default=20.0)
    p.add_argument("--z-c", type=float, default=0.6)
    p.add_argument("--level-kp", type=float, default=0.9)
    p.add_argument("--level-kd", type=float, default=0.05)
    p.add_argument("--body-lean", type=float, default=0.0,
                   help="commanded body pitch (rad); 0 keeps the body a flat reference")
    p.add_argument("--lift", type=float, default=0.45)
    # camera / output
    p.add_argument("--azimuth", type=float, default=118.0)
    p.add_argument("--elevation", type=float, default=-8.0)
    p.add_argument("--distance", type=float, default=1.12,
                   help="the trail loops reach well past the wingtips -- frame for those")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-geom", type=int, default=8000)
    p.add_argument("--realtime-factor", type=float, default=0.18)
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--still-fracs", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.07, 0.24, 0.40, 0.57, 0.74, 0.90])
    args = p.parse_args()
    (self_check if args.self_check else render)(args)


if __name__ == "__main__":
    main()
