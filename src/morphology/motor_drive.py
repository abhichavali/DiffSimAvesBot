"""Motor disk drive phasing."""

from __future__ import annotations

from morphology.model import Morphology


def drive_phase_offset_rad(morph: Morphology) -> float:
    return float(morph.motors.drive_phase_offset_rad)


def motor_actuator_gains(morph: Morphology) -> dict[str, float]:
    """Both cranks on a shoulder drive identically (+1).

    The blue/orange offset comes purely from the cranks' initial angles (see
    ``crank_sync``); the right wing is inverted via ``gear="-1"`` on its R
    actuators in MJCF (see ``emit_mjcf``). So every runtime gain is +1 and the
    two cranks share one control signal and one direction.
    """
    return {
        f"{shoulder}_motor_t{foot.station}": 1.0
        for shoulder in ("L", "R")
        for foot in morph.motors.feet
    }


def demo() -> None:
    """Self-check: every crank gain is +1 (both cranks, both sides, same dir).

    Run with src on the path: ``PYTHONPATH=src python -m morphology.motor_drive``.
    """
    from pathlib import Path

    from morphology.tg_parser import load_morphology

    tg = Path(__file__).resolve().parents[2] / "design" / "shoulder.tg"
    gains = motor_actuator_gains(load_morphology(tg))
    assert set(gains) == {"L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4"}, gains
    assert all(g == 1.0 for g in gains.values()), gains
    print("ok", gains)


if __name__ == "__main__":
    demo()
