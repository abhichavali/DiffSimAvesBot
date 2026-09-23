"""Fold and feathering schedules for the three-linkage avian wing.

The tensegrity shoulder does the stroking — the humerus is rigid on the mobile
strut, so the tendon network swings it exactly as it swings the hummingbird's
plate wing, and the stroke is the hummingbird's: a wide sweep in a near-horizontal
stroke plane, not an eagle's up-and-down flap. What this module adds is everything
distal to that: the wing flexes slightly through each reversal and extends across
the sweep, and it pronates into one half-stroke and supinates into the other, the
way a hummingbird flips its wing at each end of the row. Those are the three
position servos ``emit_mjcf`` puts on ``elbow``, ``wrist`` and ``humerus_twist``,
and all three are written here as functions of one clock: the crank angle.

The clock's offset is *measured*, not baked in. Where in the crank cycle the
stroke reaches its extreme depends on ``crank_phase_lag``, the disk pitch and the
humerus mount — retune any of those in the ``.tg`` and a hard-coded phase would
silently desync the fold from the sweep. ``calibrate_stroke`` runs a short scratch
rollout and reads the answer off the wing itself, along with the roll the tendon
network imposes on the wing plane, which the twist servo then cancels.
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

    The two shoulders drift apart the same way and for the same reason. ``crank_sync``
    matches them once at t=0 and its docstring claims they then "begin together",
    but nothing holds them there: each side integrates its own velocity command
    against its own tendon load, and the L/R phase walks off at ~0.014 deg per
    beat — slow, but open-loop, so it only ever grows, and the wings are visibly
    out of phase after a few thousand cycles. The R crank mirrors L, so it
    counter-rotates and q_L_t1 + q_R_t1 is the invariant; the same trim on the
    right t1 holds it. Everything distal is slaved to the L crank
    (``EagleWingDrive`` reads ``PHASE_JOINT``), so this also keeps the right
    wing's fold and feathering on its own stroke rather than on a stale copy of
    the left's.
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
        # Cross-shoulder lock, in the same terms: (L t1 qpos, R t1 actuator, R t1
        # qpos, R t1 gear). One shoulder — a scene with a side missing — has no
        # phase to hold.
        self._lr: tuple[int, int, int, float] | None = None
        if len(self._pairs) == 2:
            (left_t1, _, _), (right_t1, _, _) = self._pairs
            self._lr = (
                left_t1[1],
                right_t1[0],
                right_t1[1],
                float(model.actuator_gear[right_t1[0]][0]) or 1.0,
            )
        self._lr_target: float | None = None

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

    def phase_error(self, data) -> float:
        """How far the two shoulders have walked apart since startup (rad)."""
        if self._lr is None or self._lr_target is None:
            return 0.0
        left_q, _, right_q, _ = self._lr
        return float(data.qpos[left_q] + data.qpos[right_q]) - self._lr_target

    def capture(self, data) -> None:
        """Take the phases as they stand now as the ones to hold (apply/trim do this on first use)."""
        self._target = [float(data.qpos[t1[1]] + data.qpos[t4[1]]) for t1, t4, _ in self._pairs]
        if self._lr is not None:
            self._lr_target = float(data.qpos[self._lr[0]] + data.qpos[self._lr[2]])

    def apply(self, data, t0: float = 0.0) -> None:
        cmd = self.command(data.time, t0)
        for t1, t4, _ in self._pairs:
            data.ctrl[t1[0]] = cmd
            data.ctrl[t4[0]] = cmd
        self.trim(data)

    def trim(self, data) -> None:
        """Hold the phases on top of whatever velocity command ``data.ctrl`` already carries.

        Trims each t4 against q_t1 + q_t4, and the right t1 against q_L_t1 + q_R_t1.
        Any drive that writes one command per disk can call this after it does.
        """
        if self._target is None:
            self.capture(data)
        for (t1, t4, gear), target in zip(self._pairs, self._target):
            error = float(data.qpos[t1[1]] + data.qpos[t4[1]]) - target
            # q_dot = gear * ctrl, so d(sum)/dt = gear_t4 * trim.
            data.ctrl[t4[0]] -= self.GAIN * error / gear
        if self._lr is not None:
            # Trim the right t1 the same way, against the L/R sum. The right t4's
            # own loop then follows it, so the shoulder stays internally phased
            # while it is pulled back level with the left.
            _, right_aid, _, right_gear = self._lr
            data.ctrl[right_aid] -= self.GAIN * self.phase_error(data) / right_gear


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

    ``reversal_phase`` is the crank angle at one end of the stroke — the clock
    every schedule is written against. Which end depends on the design: the
    shoulder sweeps the wing in azimuth *and* elevation, and the axis that swings
    furthest is the stroke. On the hummingbird settings this design uses that is
    azimuth (118 deg of sweep against 38 of elevation), so the reference is the
    forward-most point of the sweep; on a vertical-flap tuning it would be the top
    of the upstroke instead. ``axis`` records which was picked.

    ``bank_coeffs`` is a Fourier fit of the roll the tensegrity imposes on the
    wing plane over a cycle; the twist servo subtracts it so the wing stays a
    wing instead of turning edge-on twice a beat.
    """

    reversal_phase: float
    bank_coeffs: tuple[float, ...]
    axis: str = "azimuth"
    sweep_deg: float = 0.0
    elevation_deg: float = 0.0

    def bank_at(self, theta: float) -> float:
        phase = theta - self.reversal_phase
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

    Everything downstream depends on where in the crank cycle the stroke reverses
    and how much the tendon network rolls the wing on the way. Both shift when the
    .tg changes ``crank_phase_lag``, the disk pitch or the humerus mount, so both
    are measured here rather than written down — including *which* axis the stroke
    runs along, so the same code serves a horizontal sweep and a vertical flap.

    Pass a ``drive`` to re-measure with the fold and feathering already running:
    flexing the wing moves its mass inboard, which changes what the strut does,
    so one pass leaves a residual. ``calibrated_drive`` does that loop for you.
    """
    from runtime.crank_sync import sync_crank_disks_at_rest

    data = mujoco.MjData(model)
    # Explicit lag: sync_crank_disks_at_rest otherwise re-reads MORPHOLOGY_PATH,
    # and a stale one mis-phases the whole calibration without a word.
    sync_crank_disks_at_rest(model, data, lag_rad=morph.motors.crank_phase_lag_rad)
    _, qadr, bid = _ids(model)
    period = crank_period(control)
    settle_steps = round(max(2.0, cycles - 2.0) * period / model.opt.timestep)
    total_steps = round(cycles * period / model.opt.timestep)
    cranks = CrankDrive(model, morph, control)
    thetas: list[float] = []
    banks: list[float] = []
    azims: list[float] = []
    elevs: list[float] = []
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
        span = rot[:, 2]
        elevs.append(math.degrees(math.asin(float(np.clip(span[2], -1.0, 1.0)))))
        azims.append(math.degrees(math.atan2(float(span[0]), float(span[1]))))
        thetas.append(float(data.qpos[qadr]))
        # With a drive already running, what is left to cancel is the residual on
        # top of what it is commanding, so fold its own twist target back in.
        residual = wing_plane_roll(rot)
        if drive is not None:
            residual -= drive.twist_target(float(data.qpos[qadr]))
        banks.append(residual)

    theta = np.asarray(thetas)
    azim, elev = np.asarray(azims), np.asarray(elevs)
    sweep, rise = float(np.ptp(azim)), float(np.ptp(elev))
    # The stroke is whichever axis swings furthest; reference the clock to that
    # axis' maximum. A horizontal sweep and a vertical flap then both get a phase
    # zero at a real stroke reversal instead of at whichever wobble happens to be
    # measured, and every schedule below stays meaningful across a retune.
    axis = "azimuth" if sweep >= rise else "elevation"
    reversal = float(theta[int(np.argmax(azim if axis == "azimuth" else elev))])

    phase = theta - reversal
    basis = [np.ones_like(phase)]
    for n in range(1, HARMONICS + 1):
        basis += [np.cos(n * phase), np.sin(n * phase)]
    coeffs, *_ = np.linalg.lstsq(np.stack(basis, 1), np.asarray(banks), rcond=None)
    return StrokeCalibration(
        reversal_phase=reversal,
        bank_coeffs=tuple(float(c) for c in coeffs),
        axis=axis,
        sweep_deg=sweep,
        elevation_deg=rise,
    )


def mount_roll_at_rest(model) -> float:
    """Wing-plane roll the mount was built with, at the compiled rest pose (rad).

    This is the attitude the twist servo should HOLD, not cancel. Levelling to
    zero is only right for a wing meant to lie flat; roll the mount up on edge so
    it paddles the sweep instead of slicing it, and zero would be the servo
    fighting the design back down. Reading it off the model keeps the setpoint and
    the mount from drifting apart — rotate ``bones.mount`` and this follows.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, _, bid = _ids(model)
    return wing_plane_roll(np.asarray(data.xmat[bid]).reshape(3, 3), "L")


def calibrated_drive(model, morph, control: FourierControl, passes: int = 2, **kwargs):
    """Build an :class:`EagleWingDrive` whose roll cancellation has converged.

    Pass 1 measures the bare shoulder; each later pass re-measures with the drive
    running and folds the leftover roll into the correction. Two is enough in
    practice — a third moves the residual by well under a degree.
    """
    kwargs.setdefault("attack", mount_roll_at_rest(model))
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
    servo's ctrlrange regardless. Keep it modest: a hummingbird's wing is nearly
    rigid across the sweep, and past ~0.5 the wing visibly shortens at each end
    instead of rowing. ``lead`` advances the fold ahead of the stroke, because a
    servo with finite bandwidth needs the head start.

    Both wings take the *same* twist command. The R side's hinge axis is mirrored
    (-z vs +z) and its bank is the negative of L's, and the two sign flips cancel
    — the same rule the plate wing's pitch hinge follows.
    """

    model: object
    calib: StrokeCalibration
    fold: float = 0.35  # fraction of the elbow range used at full flexion
    wrist_fold: float = 1.0  # scales `fold` at the wrist
    twist: float = math.radians(30.0)  # peak wing flip (rad; negate to reverse the sense)
    level: float = 1.0  # how much of the measured roll deviation to cancel
    # Wing-plane roll the servo holds (rad). 0 lays the wing flat, which slices
    # through a horizontal sweep; ~90 deg stands it on edge so it paddles.
    # calibrated_drive defaults this to the mount's own built-in roll.
    attack: float = 0.0
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
                    # A tensegrity wrist has no wrist joint to servo — the hand is
                    # held by its own tendon net and follows the forearm passively.
                    # Anything else missing is a real error.
                    if name == "wrist":
                        continue
                    raise ValueError(f"missing servo {side}_{name}_servo")
                lo, hi = (float(v) for v in self.model.actuator_ctrlrange[aid])
                scale = self.wrist_fold if name == "wrist" else 1.0
                self._servos.append((aid, lo, hi, name == "humerus_twist", scale))
                if name == "humerus_twist":
                    self._twist_range = (lo, hi)

    def schedule(self, theta: float) -> tuple[float, float]:
        """(twist target in rad, flexion in 0..1) at crank angle ``theta``.

        Phase 0 is one end of the stroke and pi the other. Flexion peaks at the
        ends and vanishes in the middle, so the wing tucks slightly through each
        reversal and is fully spread across the fast part of the sweep — which is
        what a hummingbird does, and the opposite of folding once per beat.

        The twist target is two terms: hold the wing plane at ``attack`` against
        the roll the tensegrity imposes (which otherwise swings the plane through
        ~41 deg twice a beat), then add a deliberate flip that leads by a quarter
        cycle, so the wing leans one way through the forward half-stroke and the
        other way coming back — what a hummingbird does at each end of the row.
        """
        phase = theta - self.calib.reversal_phase
        # A positive twist angle carries the trailing edge down, i.e. nose-UP, so
        # pronation is the negative direction.
        twist = -self.level * (self.calib.bank_at(theta) - self.attack) - self.twist * math.sin(phase)
        # cos(2*phase) so the tuck happens at BOTH reversals, once per half-stroke.
        return twist, 0.5 * (1.0 + math.cos(2.0 * phase + self.lead))

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
