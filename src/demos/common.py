"""What the offline MP4 renderers in src/demos/ share.

Scene dressing (wing colours, the wingtip trail, per-blade aero arrows, burned-in
captions, the render option), the ffmpeg pipe and PNG stills, the free-flight
pieces (the airframe MJCF, lifting it off the floor, the attitude PD) and the
crank-phase helpers of the two L/R phase clips. Each render_*.py keeps its own
CPU drive loop.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import flying
from compile_tg import compile_tg
from runtime.control_signal import eval_control
from runtime.realtime import get_actuator_id

ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")

# Nose points +x, wings span +-y, so forward is +x and a positive rotation about
# +y is nose-down (the forward lean of cruising flight).
PITCH_AXIS = np.array([0.0, 1.0, 0.0])

WING_COLORS = {
    "L": np.array([0.10, 0.55, 1.0, 0.95], np.float32),
    "R": np.array([1.0, 0.45, 0.10, 0.95], np.float32),
}

# Outermost aero blade site on each wing — the closest thing the model has to a
# wingtip, and what the trail is traced from.
TIP_SITES = {"L": "L_wing_aero_blade_04", "R": "R_wing_aero_blade_04"}

MAC_FONT = Path("/System/Library/Fonts/Helvetica.ttc")


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------- #
# Scene building
# --------------------------------------------------------------------------- #
def airframe_xml(tg, scene) -> str:
    """Compile the design and cut it free: the MJCF of the whole robot as one free airframe."""
    os.environ["MORPHOLOGY_PATH"] = str(tg)
    os.environ["MUJOCO_XML_PATH"] = str(scene)
    # Colored structural tendons read as the tensegrity joint — vivid both bare
    # and revealed through the fading skin.
    compile_tg(tg, scene, color_structural_tendons=True)
    xml = Path(scene).read_text()
    xml = flying.to_airframe(xml)
    return flying.shrink_pillar_visual(xml)


def lift_free_bodies(model, data, dz: float) -> None:
    """Raise every free body by dz so the rig reads as airborne.

    The airframe AND the mobile struts it is tendon-coupled to must rise
    together, or the tendons stretch and explode.
    """
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[int(model.jnt_qposadr[j]) + 2] += dz
    mujoco.mj_forward(model, data)


def attitude_pd(data, bid, dofadr: int, lean: float, kp: float, kd: float) -> np.ndarray:
    """World-frame PD torque on free body `bid` toward a pitch of `lean` about PITCH_AXIS.

    Free-joint angular qvel is body-local and mj_applyFT wants a world torque, so
    the damping term is rotated out.
    """
    q_tgt, qconj, q_err, att_err = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
    rot = np.asarray(data.xmat[bid]).reshape(3, 3)
    mujoco.mju_axisAngle2Quat(q_tgt, PITCH_AXIS, lean)
    mujoco.mju_negQuat(qconj, data.xquat[bid])
    mujoco.mju_mulQuat(q_err, q_tgt, qconj)
    mujoco.mju_quat2Vel(att_err, q_err, 1.0)
    return kp * att_err - kd * (rot @ data.qvel[dofadr + 3 : dofadr + 6])


# --------------------------------------------------------------------------- #
# Crank phase (the L/R phase clips)
# --------------------------------------------------------------------------- #
def crank_gears(model):
    """Each crank hinge's rotation direction, from its actuator gear.

    The right shoulder is the Y-mirror of the left, so its two cranks run the
    opposite gear (see emit_mjcf._add_actuators). A phase offset has to follow
    each crank's OWN direction or it breaks the t1-vs-t4 relationship inside the
    shoulder -- which is invisible at 180 deg (+-180 are the same angle) and
    wrong everywhere else.
    """
    gears = {}
    for name in ACTUATORS:
        side, station = name.split("_")[0], name.split("_")[2]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{station}_disk_hinge")
        aid = get_actuator_id(model, name)
        if jid >= 0 and aid >= 0:
            gears[(side, station)] = (jid, float(np.sign(model.actuator_gear[aid][0])))
    return gears


def relative_phase(model, data, gears) -> float:
    """Right-vs-left crank phase in radians, each side read in its own direction."""
    jl, gl = gears[("L", "t1")]
    jr, gr = gears[("R", "t1")]
    return gr * float(data.qpos[model.jnt_qposadr[jr]]) - gl * float(data.qpos[model.jnt_qposadr[jl]])


def ctrl_per_phase_rate(model, morph, control) -> float:
    """Extra motor ctrl per rad/s of relative phase rate.

    The nominal drive turns each crank exactly one revolution per control
    period, so an extra ctrl of nominal*r on one side adds r rev of relative
    phase per period. That is the whole phase-rate calibration.
    """
    ts = np.arange(0.0, control.period, model.opt.timestep)
    nominal = float(np.mean([morph.motors.motor_direction * eval_control(control, t) for t in ts]))
    return nominal * control.period / (2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def scene_option():
    opt = mujoco.MjvOption()
    # Show visual geoms (groups 0-2), hide collision group 3 and the anchor sites.
    for gi in range(len(opt.geomgroup)):
        opt.geomgroup[gi] = 1 if gi <= 2 else 0
    for si in range(len(opt.sitegroup)):
        opt.sitegroup[si] = 0
    # No mjVIS_CONTACTPOINT: it pops a big yellow disk over the wing at the exact
    # moment the strike should be readable. The crushed beat and the wingtip trail
    # show the hit; a debug marker on top of them only hides it.
    return opt


def _seg(scene, a, b, rgba, width):
    """One capsule between two points -- a continuous tube.

    `_arrow` would put a head on every sample, which reads as a dashed line
    rather than a path.
    """
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         a.astype(np.float64), b.astype(np.float64))
    scene.ngeom += 1


def draw_trail(scene, trail, com, rot, width):
    """Body-local trail points, replayed into world at the bird's current pose."""
    for side, pts in trail.items():
        if len(pts) < 2:
            continue
        base = WING_COLORS[side]
        world = [com + rot @ p for p in pts]
        for k in range(1, len(world)):
            rgba = base.copy()
            # Older samples fade out, so the trail reads as a direction of travel.
            rgba[3] = 0.10 + 0.80 * (k / (len(world) - 1))
            _seg(scene, world[k - 1], world[k], rgba, width)


def _arrow(scene, a, b, rgba, width):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), rgba
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_ARROW, width, a.astype(np.float64), b.astype(np.float64)
    )
    scene.ngeom += 1


def draw_aero_arrows(scene, aero, scale):
    """Per-blade force arrows (thin) plus a bold per-wing net-force arrow."""
    for wing in aero.wings:
        if wing.cached_forces is None:
            continue
        col = WING_COLORS[wing.side]
        thin = col.copy()
        thin[3] = 0.55
        net_f = np.zeros(3)
        net_p = np.zeros(3)
        for f, p in zip(wing.cached_forces, wing.cached_points):
            _arrow(scene, np.asarray(p), np.asarray(p) + np.asarray(f) * scale, thin, 0.0013)
            net_f += np.asarray(f)
            net_p += np.asarray(p)
        n = len(wing.cached_forces)
        if n:
            c = net_p / n
            _arrow(scene, c, c + net_f * scale, col, 0.0032)


def captioner(height, disabled):
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


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def ffmpeg_pipe(path, w, h, fps):
    """An ffmpeg process that encodes raw RGB frames written to its stdin as H.264."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def save_still(path, px):
    import matplotlib.image as mpimg

    mpimg.imsave(str(path), px)
