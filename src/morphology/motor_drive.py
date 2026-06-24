"""Motor disk drive phasing — actuator gains and joint equality offset."""

from __future__ import annotations

import math
import re

from morphology.model import Morphology

_ACTUATOR_RE = re.compile(r"^[LR]_motor_t(\d+)$")


def drive_phase_offset_rad(morph: Morphology) -> float:
    return float(morph.motors.drive_phase_offset_rad)


def shoulder_drive_sign(morph: Morphology, shoulder: str) -> float:
    """Per-shoulder velocity sign for flap actuators.

    When ``inphase_left_right_drive`` is on, R crank reversal is compiled into
    MJCF via ``gear="-1"`` on R velocity actuators (see ``emit_mjcf``). Runtime
    gains stay +1 on both shoulders so control code does not double-negate.
    """
    del morph, shoulder
    return 1.0


def disk_equality_polycoef(morph: Morphology) -> str:
    """MuJoCo joint equality: q2 = offset + 1.0 * q1."""
    offset = drive_phase_offset_rad(morph)
    return f"{offset} 1.0 0.0 0.0 0.0"


def motor_actuator_gain(morph: Morphology, actuator_name: str) -> float:
    """Relative velocity gain for a crank actuator vs the reference foot (feet[0]).

    At ``drive_phase_offset = π`` the second foot spins opposite the first
    (cos π = −1). At 0 they co-rotate; at π/2 the second foot is quiescent.
    """
    match = _ACTUATOR_RE.match(actuator_name)
    if not match:
        return 1.0
    shoulder = actuator_name[0]
    station = int(match.group(1))
    ref_station = morph.motors.feet[0].station
    other_station = morph.motors.feet[1].station
    side_sign = shoulder_drive_sign(morph, shoulder)
    if station == ref_station:
        return side_sign
    if station == other_station:
        return side_sign * math.cos(drive_phase_offset_rad(morph))
    return 1.0


def motor_actuator_gains(morph: Morphology) -> dict[str, float]:
    gains: dict[str, float] = {}
    for shoulder in ("L", "R"):
        for foot in morph.motors.feet:
            name = f"{shoulder}_motor_t{foot.station}"
            gains[name] = motor_actuator_gain(morph, name)
    return gains
