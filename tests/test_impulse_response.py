"""Impulse-response sim behind the GUI pane: phase bookkeeping and the open-loop torque pulse.

Run: uv run --with pytest pytest -q tests/test_impulse_response.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plot_fixed_frequency_gui import ImpulseSim, cut_wraps, wrap_deg  # noqa: E402

TG = ROOT / "designs" / "hummer" / "hover" / "shoulder.tg"
CONTROL = TG.with_name("control.tg")


def test_wrap_and_cut():
    assert wrap_deg(370.0) == 10.0 and wrap_deg(-10.0) == 350.0 and wrap_deg(190.0, 0.0) == -170.0
    x, y = cut_wraps(np.array([350.0, 355.0, 0.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(x) == 5 and np.isnan(x[2]) and np.isnan(y[2]) and y[3] == 3.0


def stroke_deviation(sim: ImpulseSim, key: str, cycles: float = 2.0) -> float:
    """Largest gap (deg) between the last `cycles` of history and the tiled undisturbed stroke."""
    w = sim.window(cycles)
    x, y = sim.reference_tiled(key, w["t"][0], w["t"][-1])
    ok = ~np.isnan(x)
    return float(np.max(np.abs(w[key] - np.interp(w["t"], x[ok], y[ok]))))


def test_phases_hold_the_designed_lag_and_the_pulse_reaches_the_wing(tmp_path):
    sim = ImpulseSim(TG, tmp_path / "scene.xml", CONTROL, settle_periods=1.0)
    lag = math.degrees(sim.morph.motors.crank_phase_lag_rad)
    assert set(sim.motors) == {"L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4"}
    assert set(sim.sides) == {"L", "R"}

    sim.advance(2.0 * sim.period)  # two steady cycles: one to become the reference stroke
    for name in sim.motors:  # every crank turns forward in its own sense; only the 360 -> 0 wraps go backwards
        d = np.diff(np.array(sim.hist[f"phase:{name}"]))
        assert (d[d > -180.0] >= 0.0).all(), name
        assert (d < -180.0).sum() == 2, name
    assert abs(sim.relative_deg("L_motor_t4") + lag) < 1.5  # t4 cranks sit at -crank_phase_lag, both sides
    assert abs(sim.relative_deg("R_motor_t4") + lag) < 1.5
    assert abs(sim.relative_deg("R_motor_t1")) < 1.5
    for side in sim.sides:  # a real stroke, and a stroke plane read off its reversals
        cyc = sim.stroke_cycle(side)
        assert 40.0 < cyc["amplitude"] < 180.0, (side, cyc)
        assert -80.0 < cyc["plane"] < 80.0, (side, cyc)
    w = sim.window(1.0)
    assert np.isfinite(w["aoa:L"]).mean() > 0.8  # AoA is undefined only around stroke reversal
    assert 0.0 <= np.nanmax(w["aoa:L"]) <= 90.0

    before = {name: sim.relative_deg(name) for name in sim.motors}
    torque, duration = 100.0, 20.0
    assert sim.inject(torque, duration, ["L_motor_t1", "bogus"]) == ("L_motor_t1",)
    assert sim.reference is not None and abs(sim.reference["period"] - sim.period) < 0.05 * sim.period
    sim.advance(0.5e-3 * duration)
    # loop open: the servo delivers last cycle's profile plus the pulse, and only on the pulsed shaft
    want = sim.nominal["L_motor_t1"](sim.phase_deg("L_motor_t1")) + 1e-3 * torque * sim.ctrl_sign
    assert abs(float(sim.data.actuator_force[sim.aid["L_motor_t1"]]) - want) < 5e-3
    assert sim.applied["L_motor_t1"] != 0.0 and sim.applied["L_motor_t4"] == 0.0
    sim.advance(0.5e-3 * duration + 2.0 * sim.period)
    assert sim.applied["L_motor_t1"] == 0.0  # pulse over, loop closed again
    after = {name: sim.relative_deg(name) for name in sim.motors}
    for name in sim.motors:  # L t1 gained phase, so the other three read behind it
        if name != sim.ref:
            assert -15.0 < after[name] - before[name] < -4.0, (name, before[name], after[name])
    assert stroke_deviation(sim, "sweep:L") > 2.0  # the pulsed wing moved off its stroke...
    assert stroke_deviation(sim, "sweep:R") < 1.5  # ...the other wing did not
    assert np.count_nonzero(np.array(sim.hist["applied:L_motor_t1"])) > 0
    assert not np.any(np.array(sim.hist["applied:R_motor_t4"]))

    w = sim.window(1.0)
    assert len(w["t"]) == int(sim.period / (sim.SAMPLE_EVERY * sim.model.opt.timestep)) + 1
    assert w["t"][-1] == sim.hist["t"][-1] and np.all(np.diff(w["t"]) > 0)


def test_feedforward_alone_reproduces_the_stroke(tmp_path):
    """A zero pulse still opens the loop; the nominal profile must then hold the stroke."""
    sim = ImpulseSim(TG, tmp_path / "scene.xml", CONTROL, settle_periods=1.0)
    sim.advance(2.0 * sim.period)
    # how much the stroke drifts cycle to cycle on its own, with no pulse at all
    sim.capture_reference()
    sim.advance(2.0 * sim.period)
    drift = {key: stroke_deviation(sim, key) for key in ("sweep:L", "roll:L")}
    sim.inject(0.0, 20.0, sim.motors)
    sim.advance(0.02 + 2.0 * sim.period)
    for key, natural in drift.items():
        assert stroke_deviation(sim, key) < max(1.5, natural + 1.0), (key, natural, stroke_deviation(sim, key))


def test_hold_phase_pins_the_lag_and_lets_a_pulse_step_decay(tmp_path):
    """The plain drive lets t4 walk off t1; CrankDrive's trim holds it and pulls a kicked crank back."""
    lag = math.degrees(ImpulseSim(TG, tmp_path / "a.xml", CONTROL, 1.0).morph.motors.crank_phase_lag_rad)
    free = ImpulseSim(TG, tmp_path / "free.xml", CONTROL, settle_periods=1.0)
    held = ImpulseSim(TG, tmp_path / "held.xml", CONTROL, settle_periods=1.0)
    held.hold_phase = True
    for sim in (free, held):
        sim.advance(30.0 * sim.period)
    assert free.relative_deg("L_motor_t4") + lag > 2.0  # walked off the designed -lag
    assert abs(held.relative_deg("L_motor_t4") + lag) < 1.0  # pinned at it
    assert abs(held.relative_deg("R_motor_t1")) < 1.0

    held.inject(100.0, 20.0, ["L_motor_t1"])
    held.advance(0.04 + 1.0 * held.period)
    kicked = held.relative_deg("L_motor_t4") + lag
    assert abs(kicked) > 3.0  # the step is there right after the pulse...
    held.advance(12.0 * held.period)  # ...and decays over a few 1/GAIN = 0.5 s
    assert abs(held.relative_deg("L_motor_t4") + lag) < 0.35 * abs(kicked)
