"""Fold and feathering schedules for the three-linkage avian wing.

The tensegrity shoulder does the flapping — the humerus is rigid on the mobile
strut, so the tendon network swings it exactly as it swings the hummingbird's
plate wing. What this module adds is everything distal to that: an eagle extends
its wing through the downstroke and flexes the elbow and wrist on the upstroke,
and pronates the wing into the downstroke and supinates it on the way up. Those
are the three position servos ``emit_mjcf`` puts on ``elbow``, ``wrist`` and
``humerus_twist``, and all three are written here as functions of one clock: the
crank angle.

The clock's offset is *measured*, not baked in. Where in the crank cycle the wing
actually tops out depends on ``crank_phase_lag``, the disk pitch and the humerus
mount — retune any of those in the ``.tg`` and a hard-coded phase would silently
desync the fold from the flap. ``calibrate_stroke`` runs a short scratch rollout
and reads the answer off the wing itself, along with the roll the tendon network
imposes on the wing plane, which the twist servo then cancels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import mujoco
import numpy as np

from runtime.control_signal import FourierControl, eval_control, ramp_factor

CRANK_ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")
PHASE_JOINT = "L_t1_disk_hinge"


def _ids(model) -> tuple[list[int], int, int]:
    acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in CRANK_ACTUATORS]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PHASE_JOINT)
    if jid < 0:
        raise ValueError(f"no crank joint {PHASE_JOINT!r} — is this an eagle scene?")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    if bid < 0:
        raise ValueError("no L_humerus body — compile a .tg that declares wing bones")
    return [a for a in acts if a >= 0], int(model.jnt_qposadr[jid]), bid


def drive_cranks(model, data, morph, control: FourierControl, t0: float, ramp: float) -> None:
    """Command every crank disk at the design velocity, ramped from ``t0``."""
    acts, _, _ = _ids(model)
    cmd = ramp_factor(data.time, t0, ramp) * morph.motors.motor_direction * eval_control(
        control, data.time
    )
    for aid in acts:
        data.ctrl[aid] = cmd


HARMONICS = 6


def wing_plane_roll(rot: np.ndarray) -> float:
    """Signed roll of the wing plane about the span axis (rad); 0 = plane level.

    ``h`` is the horizontal direction perpendicular to the span — where the chord
    would point on a perfectly level wing — so the angle from ``h`` to the actual
    chord, measured about the span, is the bank.
    """
    span, chord = rot[:, 2], rot[:, 0]
    h = np.cross(span, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(h))
    if n < 1e-9:  # span vertical: bank is undefined, and the wing is edge-on anyway
        return 0.0
    h /= n
    return float(math.atan2(float(np.dot(np.cross(h, chord), span)), float(np.dot(h, chord))))


@dataclass(frozen=True)
class StrokeCalibration:
    """What the shoulder does on its own, measured once per scene.

    ``top_phase`` is the crank angle at the top of the upstroke — the clock every
    schedule is written against. ``bank_coeffs`` is a Fourier fit of the roll the
    tensegrity imposes on the wing plane over a cycle; the twist servo subtracts
    it so the wing stays a wing instead of turning edge-on twice a beat.
    """

    top_phase: float
    bank_coeffs: tuple[float, ...]

    def bank_at(self, theta: float) -> float:
        phase = theta - self.top_phase
        value = self.bank_coeffs[0]
        for n in range(1, HARMONICS + 1):
            value += self.bank_coeffs[2 * n - 1] * math.cos(n * phase)
            value += self.bank_coeffs[2 * n] * math.sin(n * phase)
        return value


def calibrate_stroke(
    model,
    morph,
    control: FourierControl,
    cycles: int = 7,
    drive: "EagleWingDrive | None" = None,
) -> StrokeCalibration:
    """Measure the shoulder's own stroke on a scratch rollout (caller state untouched).

    Everything downstream depends on where in the crank cycle the wing tops out
    and how much the tendon network rolls it on the way. Both shift when the .tg
    changes ``crank_phase_lag``, the disk pitch or the humerus mount, so both are
    measured here rather than written down.

    Pass a ``drive`` to re-measure with the fold and feathering already running:
    flexing the wing moves its mass inboard, which changes what the strut does,
    so one pass leaves a residual. ``calibrated_drive`` does that loop for you.
    """
    from runtime.crank_sync import sync_crank_disks_at_rest

    data = mujoco.MjData(model)
    sync_crank_disks_at_rest(model, data)
    _, qadr, bid = _ids(model)
    period = control.period
    settle = max(2.0, cycles - 2.0) * period
    end = cycles * period
    best_elev, top = -2.0, 0.0
    thetas: list[float] = []
    banks: list[float] = []
    while data.time < end:
        data.ctrl[:] = 0.0
        drive_cranks(model, data, morph, control, 0.0, 2.0 * period)
        if drive is not None:
            drive.apply(data)
        mujoco.mj_step(model, data)
        if data.time < settle:
            continue
        rot = np.asarray(data.xmat[bid]).reshape(3, 3)
        if not np.isfinite(rot).all():
            raise RuntimeError("stroke calibration diverged — check tendon/timestep settings")
        elev = float(rot[2, 2])  # z of the span axis
        if elev > best_elev:
            best_elev, top = elev, float(data.qpos[qadr])
        thetas.append(float(data.qpos[qadr]))
        # With a drive already running, what is left to cancel is the residual on
        # top of what it is commanding, so fold its own twist target back in.
        residual = wing_plane_roll(rot)
        if drive is not None:
            residual -= drive.schedule(float(data.qpos[qadr]))[0]
        banks.append(residual)
    phase = np.asarray(thetas) - top
    basis = [np.ones_like(phase)]
    for n in range(1, HARMONICS + 1):
        basis += [np.cos(n * phase), np.sin(n * phase)]
    coeffs, *_ = np.linalg.lstsq(np.stack(basis, 1), np.asarray(banks), rcond=None)
    return StrokeCalibration(top_phase=top, bank_coeffs=tuple(float(c) for c in coeffs))


def calibrated_drive(model, morph, control: FourierControl, passes: int = 2, **kwargs):
    """Build an :class:`EagleWingDrive` whose roll cancellation has converged.

    Pass 1 measures the bare shoulder; each later pass re-measures with the drive
    running and folds the leftover roll into the correction. Two passes is enough
    in practice — the third moves the residual by under a degree.
    """
    calib = calibrate_stroke(model, morph, control)
    drive = EagleWingDrive(model, calib=calib, **kwargs)
    for _ in range(max(0, passes - 1)):
        drive.calib = calibrate_stroke(model, morph, control, drive=drive)
    return drive


@dataclass
class EagleWingDrive:
    """Servo targets for both wings' elbow, wrist and twist, from the crank angle.

    ``fold`` is a 0..1 fraction of each joint's compiled range, so the amplitude
    stays inside the limits the ``.tg`` declared; every target is clipped to the
    servo's ctrlrange regardless. ``lead`` advances the fold ahead of the flap:
    real wings start extending before the downstroke begins, and a servo with
    finite bandwidth needs the head start anyway.

    Both wings take the *same* twist command. The R side's hinge axis is mirrored
    (-z vs +z) and its bank is the negative of L's, and the two sign flips cancel
    — the same rule the plate wing's pitch hinge follows.
    """

    model: object
    calib: StrokeCalibration
    fold: float = 0.60  # fraction of the elbow range used at full flexion
    wrist_fold: float = 1.0  # scales `fold` at the wrist
    twist: float = math.radians(14.0)  # peak deliberate pronation (rad; negate to flip sense)
    level: float = 1.0  # how much of the measured roll to cancel (1 = fully level)
    lead: float = math.radians(35.0)  # fold leads the flap by this crank angle

    def __post_init__(self) -> None:
        _, self._qadr, _ = _ids(self.model)
        self._servos: list[tuple[int, float, float, bool, float]] = []
        for side in ("L", "R"):
            for name in ("humerus_twist", "elbow", "wrist"):
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{name}_servo"
                )
                if aid < 0:
                    raise ValueError(f"missing servo {side}_{name}_servo")
                lo, hi = (float(v) for v in self.model.actuator_ctrlrange[aid])
                scale = self.wrist_fold if name == "wrist" else 1.0
                self._servos.append((aid, lo, hi, name == "humerus_twist", scale))

    def schedule(self, theta: float) -> tuple[float, float]:
        """(twist target in rad, flexion in 0..1) at crank angle ``theta``.

        Phase 0 is the top of the upstroke and pi the bottom of the downstroke.
        Flexion peaks at the top and vanishes at the bottom. The twist target is
        two terms: cancel the roll the tensegrity imposes (which otherwise turns
        the wing edge-on twice a beat), then add back a deliberate pronation that
        leads the downstroke by a quarter cycle — nose-down through the fastest
        descent, nose-up on the recovery, the way a bird feathers.
        """
        phase = theta - self.calib.top_phase
        twist = -self.level * self.calib.bank_at(theta) + self.twist * math.sin(phase)
        return twist, 0.5 * (1.0 + math.cos(phase + self.lead))

    def apply(self, data) -> None:
        twist, flexion = self.schedule(float(data.qpos[self._qadr]))
        for aid, lo, hi, is_twist, scale in self._servos:
            if is_twist:
                target = twist
            else:
                target = hi + (lo - hi) * self.fold * scale * flexion
            data.ctrl[aid] = float(np.clip(target, lo, hi))
