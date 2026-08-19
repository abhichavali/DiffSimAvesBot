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


class CrankDrive:
    """Command every crank disk, and hold the t1/t4 offset that picks the flap mode.

    ``crank_phase_lag`` is the eagle knob: it selects which compliant mode of the
    tendon network the two counter-rotating cranks excite. ``crank_sync`` sets it
    once at t=0 and its docstring claims the offset then stays put because "both
    integrate the same speed". They do not — the two disks are velocity servos
    sitting at different points of the tendon load cycle, so their achieved rates
    differ by about 0.1% and the lag walks off at ~0.31 deg per beat. Left alone
    the stroke leaves the eagle regime after a couple of hundred cycles: flap
    amplitude 60.6 -> 38.3 deg and wing-plane roll 30.7 -> 55.2 deg by cycle 255.

    Because the pair counter-rotates, q_t1 + q_t4 is an invariant of the intended
    motion, so the drift shows up in that sum and nowhere else. Trimming the t4
    command to hold the sum at its startup value holds the lag with no effect on
    the stroke itself.
    """

    #: Trim gain (1/s). Loose enough not to fight the tendon load within a beat,
    #: stiff enough to kill a 0.3 deg/beat drift.
    GAIN = 2.0

    def __init__(self, model, morph, control: FourierControl, ramp_periods: float = 2.0):
        self.model = model
        self.morph = morph
        self.control = control
        self.ramp = ramp_periods * control.period
        self._pairs = []
        for side in ("L", "R"):
            ids = {}
            for station in ("t1", "t4"):
                aid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_motor_{station}"
                )
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{station}_disk_hinge"
                )
                if aid < 0 or jid < 0:
                    ids = None
                    break
                ids[station] = (aid, int(model.jnt_qposadr[jid]))
            if ids:
                gear = float(model.actuator_gear[ids["t4"][0]][0]) or 1.0
                self._pairs.append((ids["t1"], ids["t4"], gear))
        if not self._pairs:
            raise ValueError("no crank disks — is this a compiled shoulder scene?")
        self._target: list[float] | None = None

    def command(self, t: float, t0: float = 0.0) -> float:
        return (
            ramp_factor(t, t0, self.ramp)
            * self.morph.motors.motor_direction
            * eval_control(self.control, t)
        )

    def lag_error(self, data) -> float:
        """How far the t1/t4 offset has walked from its startup value (rad), worst side."""
        if self._target is None:
            return 0.0
        return max(
            abs(float(data.qpos[t1[1]] + data.qpos[t4[1]]) - target)
            for (t1, t4, _), target in zip(self._pairs, self._target)
        )

    def apply(self, data, t0: float = 0.0) -> None:
        cmd = self.command(data.time, t0)
        if self._target is None:
            self._target = [
                float(data.qpos[t1[1]] + data.qpos[t4[1]]) for t1, t4, _ in self._pairs
            ]
        for (t1, t4, gear), target in zip(self._pairs, self._target):
            data.ctrl[t1[0]] = cmd
            error = float(data.qpos[t1[1]] + data.qpos[t4[1]]) - target
            # q_dot = gear * ctrl, so d(sum)/dt = gear_t4 * trim.
            data.ctrl[t4[0]] = cmd - self.GAIN * error / gear


def crank_period(control: FourierControl) -> float:
    """Seconds per crank revolution, from the velocity command itself.

    Not ``control.period``: that is ``fundamental_frequency``, a phase reference
    the .tg keeps by hand and which nothing validates. The calibration window and
    the roll fit are functions of *crank angle*, so a window sized off a stale
    fundamental can cover well under one revolution and the fit then extrapolates
    over phase it never saw — at a0 = 3 rad/s against a 2.5 Hz reference it sees
    0.38 of a turn and the "correction" ends up rolling the wing more than doing
    nothing at all.
    """
    speed = abs(float(control.a0))
    if speed < 1e-6:
        return control.period
    return math.tau / speed


def check_stable(model, data, what: str) -> None:
    """Raise if the model has gone unstable.

    Not a finite check on the state: MuJoCo resets MjData on mjWARN_BADQACC, which
    also rewinds ``data.time``, so by the time anyone looks the state is finite
    again and any ``while data.time < end`` loop has silently restarted. The
    warning counter is the only thing that remembers.
    """
    if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
        raise RuntimeError(
            f"{what} diverged (mjWARN_BADQACC) — world.timestep is likely past the "
            "tendon stiffness limit; design/eagle.tg is stable to 0.20 ms"
        )


# Harmonics in the roll fit, trading levelling against how fast the twist servo
# has to move. Measured on design/eagle.tg (residual roll peak-to-peak / peak
# twist rate): 2 -> 19.7 deg / 535 deg-s, 3 -> 10.4 / 785, 4 -> 5.2 / 921,
# 6 -> 2.5 / 1066. Past 4 the extra flatness is bought with high-frequency
# content that reads on screen as the wing snapping rather than feathering.
HARMONICS = 4


_MIRROR_Y = np.diag([1.0, -1.0, 1.0])

#: Fraction of a joint's range kept clear of each limit by the fold schedule.
_STOP_MARGIN = 0.02


def wing_plane_roll(rot: np.ndarray, side: str = "L") -> float:
    """Signed roll of the wing plane about the span axis (rad); 0 = plane level.

    ``h`` is the horizontal direction perpendicular to the span — where the chord
    would point on a perfectly level wing — so the angle from ``h`` to the actual
    chord, measured about the span, is the roll. On a left wing that equals the
    twist joint's own angle, which is why the servo can cancel it directly.

    The reference ``h`` picks out the chord on a left wing (span +y) and its
    opposite on a right one (span -y), so a right wing must be mirrored into the
    left convention first or it reads 180 deg off and wraps every cycle. Pass
    ``side="R"`` for the right humerus; both then report the same number for a
    mirror-symmetric pose.
    """
    if side == "R":
        rot = _MIRROR_Y @ rot @ _MIRROR_Y
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
    # Explicit lag: sync_crank_disks_at_rest otherwise re-reads MORPHOLOGY_PATH,
    # and a stale one (eagle scene, hummingbird env var) mis-phases the whole
    # calibration without a word — eagle.tg lags 160 deg, shoulder.tg 12.
    sync_crank_disks_at_rest(model, data, lag_rad=morph.motors.crank_phase_lag_rad)
    _, qadr, bid = _ids(model)
    period = crank_period(control)
    settle_steps = round(max(2.0, cycles - 2.0) * period / model.opt.timestep)
    total_steps = round(cycles * period / model.opt.timestep)
    cranks = CrankDrive(model, morph, control)
    best_elev, top = -2.0, 0.0
    thetas: list[float] = []
    banks: list[float] = []
    for step in range(total_steps):
        data.ctrl[:] = 0.0
        cranks.apply(data)
        if drive is not None:
            drive.apply(data)
        mujoco.mj_step(model, data)
        check_stable(model, data, "stroke calibration")
        if step < settle_steps:
            continue
        rot = np.asarray(data.xmat[bid]).reshape(3, 3)
        elev = float(rot[2, 2])  # z of the span axis
        if elev > best_elev:
            best_elev, top = elev, float(data.qpos[qadr])
        thetas.append(float(data.qpos[qadr]))
        # With a drive already running, what is left to cancel is the residual on
        # top of what it is commanding, so fold its own twist target back in.
        residual = wing_plane_roll(rot)
        if drive is not None:
            residual -= drive.twist_target(float(data.qpos[qadr]))
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
    running and folds the leftover roll into the correction. Two is enough: on
    design/eagle.tg the residual roll goes 7.2 -> 2.5 deg peak-to-peak, and a
    third pass only reaches 2.2 (a fourth is back at 2.3, i.e. noise).
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
                if name == "humerus_twist":
                    self._twist_range = (lo, hi)

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
        # A positive twist angle carries the trailing edge down, i.e. nose-UP, so
        # pronation is the negative direction.
        twist = -self.level * self.calib.bank_at(theta) - self.twist * math.sin(phase)
        return twist, 0.5 * (1.0 + math.cos(phase + self.lead))

    def twist_target(self, theta: float) -> float:
        """The twist the servo is actually asked for, clipped as ``apply`` clips it.

        The roll cancellation alone peaks near 36 deg of the .tg's 50 deg range,
        so a large ``twist`` saturates. Calibration has to subtract what was
        commanded, not what was wanted, or the next pass fits a signal that never
        happened.
        """
        lo, hi = self._twist_range
        return float(np.clip(self.schedule(theta)[0], lo, hi))

    def apply(self, data) -> None:
        twist, flexion = self.schedule(float(data.qpos[self._qadr]))
        for aid, lo, hi, is_twist, scale in self._servos:
            if is_twist:
                target = twist
            else:
                # Stop a hair short of the joint limit: at fold=0 the target is
                # exactly `hi`, and the servo then parks the hinge on its own
                # constraint stop for the whole run instead of resting extended.
                span = hi - lo
                target = hi - _STOP_MARGIN * span + (lo - hi + 2 * _STOP_MARGIN * span) * (
                    self.fold * scale * flexion
                )
            data.ctrl[aid] = float(np.clip(target, lo, hi))
