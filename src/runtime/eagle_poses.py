"""Held poses for the eagle wing: an elbow angle, an elbow fold sweep, the glide.

These are the non-stroking commands ``src/view_eagle.py`` offers (--elbow,
--elbow-sweep, --glide): servo targets for the elbow and humerus twist, and the
crank park that sets where the span sits. ``runtime.eagle_drive`` is the stroking
drive; this is everything the viewer holds still instead.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from runtime.eagle_drive import check_stable, wing_plane_roll


def elbow_targets(model, fraction: float) -> dict[int, float]:
    """Servo commands holding both elbows at ``fraction`` of their range.

    0 is the extended stop and 1 the flexed one, so --elbow 1 is the wing's one
    remaining hinge at full contraction.
    """
    out: dict[int, float] = {}
    for side in ("L", "R"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_elbow_servo")
        if aid < 0:
            continue
        lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
        out[aid] = hi + (lo - hi) * fraction
    return out


def elbow_sweep(model, seconds: float):
    """Elbow commands that fold and unfold once every ``seconds``.

    A raised cosine, so it leaves and returns to the extended stop with zero
    velocity — a linear ramp would step the servo at each turnaround and ring the
    tensegrity, which is the one thing you cannot tell apart from a real fold.
    Returns a function of sim time, which is what makes it a schedule rather than
    the held pose ``elbow_targets`` gives.
    """
    def targets(t: float) -> dict[int, float]:
        return elbow_targets(model, 0.5 - 0.5 * math.cos(2.0 * math.pi * t / seconds))

    return targets


def settle_elbow(model, data, targets, control, cycles: float = 2.0) -> None:
    """Step the wing to rest at a held elbow angle, with the shoulder quiet.

    Not a matter of setting qpos: with a tensegrity wrist the hand is a free body,
    so posing the forearm and calling mj_forward leaves the hand where it was. The
    only way to see the pose is to let the model reach it.
    """
    for _ in range(round(cycles * control.period / model.opt.timestep)):
        data.ctrl[:] = 0.0
        for aid, target in targets.items():
            data.ctrl[aid] = target
        mujoco.mj_step(model, data)
        check_stable(model, data, "elbow settle")


def park_cranks(model, data, park_rad: float) -> None:
    """Turn every crank ``park_rad`` off its synced angle, keeping the mirror.

    The two cranks on a shoulder counter-rotate and the right shoulder mirrors the
    left, so the same advance in the world is the opposite step in qpos on three
    of the four disks — the same rule ``crank.apply_crank_phase_lag`` follows
    for the t4 lag. Get it wrong and the wings park in different postures.
    """
    for side in ("L", "R"):
        for station in ("t1", "t4"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{station}_disk_hinge")
            if jid < 0:
                continue
            sign = 1.0 if (side == "L") == (station == "t1") else -1.0
            data.qpos[model.jnt_qposadr[jid]] += sign * park_rad
    mujoco.mj_forward(model, data)


def glide_pose(model, data, control, park_rad: float, cycles: float = 1.2):
    """Park the motors and hold the wing out flat — the glide posture.

    Nothing about this is the rest pose. The mount stands the wing on edge by
    89 deg so it paddles the stroke, and the tensegrity shoulder presents the span
    at whatever elevation the parked crank angle puts it. So two things are set
    here: the crank park, which trades span elevation against sweep, and the twist
    servo, which takes the mount's roll back out until the wing plane is level.

    How much twist is measured, not baked. On a left wing the plane's roll IS the
    twist joint's angle, so one correction lands it and the second confirms — and
    the correction has to be clipped, because the servo's 70 deg cannot always
    cover the mount's 89. Both wings take the same command: the right hinge axis
    is mirrored and its bank is the negative of the left's, and the two signs
    cancel (the same rule ``EagleWingDrive`` follows).

    Returns a schedule rather than a dict, which is what tells the viewer to keep
    stepping: the pose is static but the feathers hanging off it are not.
    """
    park_cranks(model, data, park_rad)
    targets = elbow_targets(model, 0.0)  # 0 = the extended stop: a glide is spread
    twist = {}
    for side in ("L", "R"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_humerus_twist_servo")
        if aid >= 0:
            twist[aid] = 0.0

    def settle():
        for _ in range(round(cycles * control.period / model.opt.timestep)):
            data.ctrl[:] = 0.0
            for aid, target in {**targets, **twist}.items():
                data.ctrl[aid] = target
            mujoco.mj_step(model, data)
            check_stable(model, data, "glide settle")

    def plane_roll() -> float:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
        return wing_plane_roll(np.asarray(data.xmat[bid]).reshape(3, 3), "L")

    settle()
    for _ in range(2):
        error = plane_roll()
        for aid in twist:
            lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
            twist[aid] = float(np.clip(twist[aid] - error, lo, hi))
        settle()

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    span = np.asarray(data.xmat[bid]).reshape(3, 3)[:, 2]
    print(
        f"glide: cranks parked at {math.degrees(park_rad):.0f} deg; span "
        f"{math.degrees(math.asin(float(np.clip(span[2], -1, 1)))):+.1f} deg above level, "
        f"swept {math.degrees(math.atan2(span[0], span[1])):+.1f} deg; wing plane "
        f"{math.degrees(plane_roll()):+.1f} deg off flat "
        f"(0 = flat, 90 = on edge)",
        flush=True,
    )
    held = {**targets, **twist}
    return lambda _t: held
