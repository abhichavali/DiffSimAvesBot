"""Flight control .tg files: per-crank harmonics, the flight keys, and a short closed-loop hover."""
import math

import numpy as np
import pytest

from design_paths import DEFAULT_HUMMER_TG, FREE_HOVER_TG
from hover import launch as L
from hover.policy import has_flight_policy, load_flight_policy, write_flight_control
from runtime.control_signal import CRANKS, drive_command, eval_control, load_control

FLIGHT = FREE_HOVER_TG.parent / "control.tg"


def test_per_crank_harmonics_drive_each_crank():
    c = load_control(FLIGHT)
    assert c.per_crank is not None and len(c.per_crank) == len(CRANKS)
    cmd = drive_command(c, 1.0, 0.037)
    assert isinstance(cmd, list) and len(cmd) == 4 and len(set(np.round(cmd, 9))) == 4


def test_plain_controls_are_unchanged():
    c = load_control(DEFAULT_HUMMER_TG.parent / "control.tg")
    assert c.per_crank is None
    assert drive_command(c, -1.0, 0.05) == -1.0 * eval_control(c, 0.05)
    assert not has_flight_policy(DEFAULT_HUMMER_TG.parent / "control.tg")


def test_unknown_per_crank_key_is_rejected(tmp_path):
    p = tmp_path / "c.tg"
    p.write_text("bird = hummer\nharmonics = 1\nfundamental_frequency = 10 Hz\na0 = 62.83\na_2.L_t1 = 1\n")
    with pytest.raises(ValueError, match="per-crank"):
        load_control(p)


def test_write_then_load_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    da, db = rng.normal(0, 0.03, (4, 3)), rng.normal(0, 0.03, (4, 3))
    att = np.array([0.98, -0.02, 0.18, 0.0])
    write_flight_control(tmp_path / "c.tg", a0=53.7, depth_a=da, depth_b=db, attitude=att, phase_lock=15)
    fp = load_flight_policy(tmp_path / "c.tg")
    assert fp.mlp is None and fp.phase_lock == 15 and fp.theta[0] == 0.0
    np.testing.assert_allclose(fp.theta[1:], np.concatenate([da.ravel(), db.ravel()]), atol=1e-9)
    np.testing.assert_allclose(fp.attitude, att / np.linalg.norm(att), atol=1e-12)
    assert np.all(fp.correction(np.zeros(12)) == 0.0)


def test_closed_loop_hovers():
    fp = load_flight_policy(FLIGHT)
    assert fp.mlp is not None and has_flight_policy(FLIGHT)
    bird = L.load_bird(FREE_HOVER_TG, control=FLIGHT)
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock)
    tr = L.fly(bird, rel, fp.theta, 12, policy=fp.correction, kicks={2: np.r_[np.zeros(6), 0.2, 0, 0, 1.0, 0, 0]})
    assert len(tr) == 13                                           # not lost
    assert math.degrees(np.linalg.norm(tr[-1, 3:5])) < 2.0         # recovering from the push
