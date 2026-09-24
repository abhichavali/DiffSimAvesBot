"""The crank disks: their startup sync, the t1/t4 phase lag, and the drive that holds both.

``sync_crank_disks_at_rest`` runs once after the rest settle; ``CrankDrive`` commands
the disks every step and trims away the phase drift the open-loop drive accumulates.
``check_stable`` is the guard every clock-bounded rollout loop needs.

The rest settle (``wing_pitch.calibrate_wing_pitch_at_rest``) steps the model for
~2000 frames with no motor command. The velocity-driven disks creep a few degrees
under tendon/gravity load and end up at slightly different angles *and* at nonzero
speed, asymmetrically between the two shoulders. The result: when the flap ramp
begins the L/R cranks are already out of phase and mid-spin, so the wings appear to
start at different times. The sync resets every crank disk to a matched,
zero-velocity phase so both shoulders begin together.
"""

from __future__ import annotations

import math

import mujoco

from morphology.tg_parser import load_morphology
from runtime.control_signal import FourierControl, eval_control, ramp_factor
from runtime.morphology_runtime import resolve_morphology_path

CRANK_ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")


def sync_crank_disks_at_rest(model, data, lag_rad: float | None = None, morph=None) -> None:
    """Mirror R onto L at every motor station, stop the disks, then apply the t4 lag.

    ``morph`` names the stations (and, with ``lag_rad=None``, the lag); without it
    the design at ``MORPHOLOGY_PATH`` is parsed.
    """
    if morph is None:
        morph = load_morphology(resolve_morphology_path())
    feet = morph.motors.feet

    def joint_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    changed = False
    # R_t{s}_disk is the Y-mirror of L_t{s}_disk (same station -- verified: every
    # L_/R_ body and site pair sits at mirrored xpos), and its hinge axis mirrors
    # with it, so the mirrored angle is the NEGATED one. The settle pass already
    # leaves them that way (+0.330 / -0.330 deg); this just cleans up the creep.
    # This used to pair L_t{a} with R_t{b} across the sigma station swap (1<->4)
    # and copy the angle unnegated, which took a mirror-correct rest state and
    # un-mirrored it: the two wings then traced paths 0.6-0.8% of stroke apart
    # for the whole run, on top of whatever the .tg baked in. Per-station and
    # negated, the wingtips agree to solver noise (tests/test_wing_lock.py).
    for foot in feet:
        left = joint_id(f"L_t{foot.station}_disk_hinge")
        right = joint_id(f"R_t{foot.station}_disk_hinge")
        if left < 0 or right < 0:
            continue
        data.qpos[model.jnt_qposadr[right]] = -float(data.qpos[model.jnt_qposadr[left]])
        data.qvel[model.jnt_dofadr[left]] = 0.0
        data.qvel[model.jnt_dofadr[right]] = 0.0
        changed = True

    if changed:
        mujoco.mj_forward(model, data)
    apply_crank_phase_lag(
        model,
        data,
        morph.motors.crank_phase_lag_rad if lag_rad is None else lag_rad,
        feet,
    )


def apply_crank_phase_lag(model, data, lag_rad: float, feet) -> None:
    """Phase-lag the non-reference (t4 'orange') cranks by ``lag_rad`` vs ``feet[0]`` (t1).

    The two cranks ALWAYS counter-rotate (actuator gear -1); this adds an initial
    crank-angle offset on top. The offset does not hold by itself under the shared
    velocity command -- ``CrankDrive`` explains the drift and trims it away.
    ``crank_phase_lag`` in the .tg sets it; sweep it to stagger the two motors'
    torque peaks (cuts overlap / inactive power).
    """
    if not lag_rad:
        return
    ref = feet[0].station
    changed = False
    for foot in feet:
        if foot.station == ref:
            continue
        for side in ("L", "R"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_t{foot.station}_disk_hinge")
            if jid < 0:
                continue
            # R shoulder is the Y-mirror of L: its t4 crank runs the opposite gear
            # (see emit_mjcf._add_actuators). Adding the same qpos offset to both
            # sides would advance them in opposite world directions, mirror-breaking
            # the orange disks. Flip the sign on R so the offset is a true mirror and
            # both wings see the same t4-vs-t1 lag.
            sign = 1.0 if side == "L" else -1.0
            data.qpos[model.jnt_qposadr[jid]] += sign * float(lag_rad)
            changed = True
    if changed:
        mujoco.mj_forward(model, data)


class CrankDrive:
    """Command every crank disk, and hold the t1/t4 offset that picks the flap mode.

    ``crank_phase_lag`` is the eagle knob: it selects which compliant mode of the
    tendon network the two counter-rotating cranks excite. The sync above sets it
    once at t=0, but it does not stay put: the two disks are velocity servos
    sitting at different points of the tendon load cycle, so their achieved rates
    differ by about 0.1% and the lag walks off at ~0.31 deg per beat. Left alone
    the stroke leaves the eagle regime after a couple of hundred cycles: flap
    amplitude 60.6 -> 38.3 deg and wing-plane roll 30.7 -> 55.2 deg by cycle 255.

    Because the pair counter-rotates, q_t1 + q_t4 is an invariant of the intended
    motion, so the drift shows up in that sum and nowhere else. Trimming the t4
    command to hold the sum at its startup value holds the lag with no effect on
    the stroke itself.

    The two shoulders drift apart the same way and for the same reason. The sync
    matches them once at t=0, but nothing holds them there: each side integrates
    its own velocity command against its own tendon load, and the L/R phase walks
    off at ~0.014 deg per beat -- slow, but open-loop, so it only ever grows, and
    the wings are visibly out of phase after a few thousand cycles. The R crank
    mirrors L, so it counter-rotates and q_L_t1 + q_R_t1 is the invariant; the
    same trim on the right t1 holds it. Everything distal on the eagle is slaved
    to the L crank (``eagle_drive.EagleWingDrive``), so this also keeps the right
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
            "tendon stiffness limit; designs/eagle/shoulder.tg is stable to 0.20 ms"
        )
