"""The .tg parser: units on angle literals.

Run: uv run pytest -q tests/test_tg_parser.py
"""

import math

from design_paths import DEFAULT_HUMMER_TG
from morphology.tg_parser import _TgParser

HOVER = DEFAULT_HUMMER_TG.read_text()


def parse(old: str, new: str):
    assert old in HOVER, old
    return _TgParser(HOVER.replace(old, new, 1)).parse()


def test_angles_follow_their_unit():
    rad = math.radians
    disk = parse("tilt: 0.0 rad, pitch: 20 deg", "tilt: 5 deg, pitch: 2 deg, crank_phase: 90 deg").motors.disk
    assert (disk.tilt_rad, disk.pitch_rad, disk.crank_phase_rad) == (rad(5), rad(2), rad(90))
    assert parse("pitch: 20 deg", "pitch: 0.35 rad").motors.disk.pitch_rad == 0.35
    assert parse("tilt: 0.0 rad", "tilt: 5 rad").motors.disk.tilt_rad == 5.0
    pitch = parse("initial    = 0.0 rad", "initial    = 10 deg").wing.pitch
    assert pitch.initial_rad == rad(10)
    assert parse("range      = [-0.7, 0.7] rad", "range      = [-40, 40] deg").wing.pitch.range_rad == (rad(-40), rad(40))
    feet = parse("attach_along_edge = 0.0 rad", "attach_along_edge = 10 deg").motors.feet
    assert feet[0].attach_along_edge_rad == rad(10)


def test_attach_along_edge_defaults_to_motor_foot():
    feet = parse("    attach_along_edge = 0.0 rad\n", "").motors.feet
    assert feet[0].attach_along_edge_rad == 0.0
