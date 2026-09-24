"""PlateWingSim driven by hand must reproduce the library CPU loops bit for bit.

Checked against codesign.cpu_check.cpu_rollout (harmonics + learned stroke
plane, explicit phase lag) and run_fixed_frequency.run (ramped Fourier drive,
design lag), on a short window of the hover-soft-it009p design.
"""
import contextlib
import io
import json
import math
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import pytest

import run_fixed_frequency
from codesign.cpu_check import CpuDesign, cpu_rollout
from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.control_signal import drive_command, load_control_for_morphology
from runtime.disk_orientation import DiskOrientation
from runtime.morphology_runtime import load_design_morphology
from runtime.sim import PlateWingSim

TG = Path(__file__).resolve().parents[1] / "designs" / "hummer" / "hover-soft-it009p" / "shoulder.tg"


@pytest.fixture
def design(tmp_path, monkeypatch):
    """The design compiled to a scratch scene, with the env the legacy loops read pointed at it."""
    scene = tmp_path / "scene.xml"
    monkeypatch.setenv("MORPHOLOGY_PATH", str(TG))
    monkeypatch.setenv("MUJOCO_XML_PATH", str(scene))
    load_design_morphology.cache_clear()
    compile_tg(TG, scene)
    yield scene, load_morphology(TG), load_control_for_morphology(TG)
    load_design_morphology.cache_clear()


@contextlib.contextmanager
def final_data():
    """Capture the MjData of the last mujoco.mj_step inside the block."""
    last = {}
    step = mujoco.mj_step

    def recording(m, d, *a, **k):
        step(m, d, *a, **k)
        last["data"] = d

    with mock.patch.object(mujoco, "mj_step", recording):
        yield last


def state(d):
    return [np.array(x) for x in (d.time, d.qpos, d.qvel, d.qfrc_applied, d.ctrl, d.actuator_force)]


def assert_same(a, b):
    for x, y in zip(a, b):
        assert x.tobytes() == y.tobytes()


def test_matches_cpu_rollout(design):
    scene, morph, control = design
    lag = morph.motors.crank_phase_lag_rad
    a0 = morph.motors.motor_direction * control.a0
    a = np.asarray([x / control.a0 for x in control.a])
    b = np.asarray([x / control.a0 for x in control.b])
    omega, warmup, periods = control.omega, 0.1, 0.15
    sp = json.loads((TG.parent / "strokeplane_optimized.json").read_text())
    sp_params = (sp["tilt_coeffs"] + sp["pitch_coeffs"], sp["tilt0_rad"], sp["pitch0_rad"], sp["max_delta_rad"])

    with final_data() as last:
        expected = tuple(cpu_rollout(scene, CpuDesign((a0, a, b, omega), strokeplane=sp_params), "euler",
                                     periods=periods, warmup=warmup, phase_lag=lag, aero_substep=4))
    expected_state = state(last["data"])

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    sim = PlateWingSim(model, morph, phase_lag=lag, aero_substep=4)
    data, ids = sim.data, sim.crank_ids
    disks = DiskOrientation(model)
    coef, tilt0, pitch0, span = np.asarray(sp_params[0], dtype=float), *sp_params[1:]
    phase_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_t1_disk_hinge")]
    harm = np.arange(1, len(a) + 1)

    def step():
        th = float(data.qpos[phase_q])
        basis = np.array([1.0, math.cos(th), math.sin(th), math.cos(2.0 * th), math.sin(2.0 * th)])
        disks.set(tilt0 + span * math.tanh(float(coef[:5] @ basis) / span),
                  pitch0 + span * math.tanh(float(coef[5:] @ basis) / span))
        phase = omega * data.time
        lift = sim.step(a0 * (1.0 + float(np.sum(a * np.cos(harm * phase) + b * np.sin(harm * phase)))))
        power = data.actuator_force[ids] * data.actuator_velocity[ids]
        return lift, float(np.max(np.abs(data.actuator_force[ids]))), float(np.sum(np.maximum(0.0, power)))

    period, dt = math.tau / omega, model.opt.timestep
    for _ in range(int(warmup * period / dt)):
        step()
    rec = np.array([step() for _ in range(int(periods * period / dt))])
    L, T, P = rec.T
    spc = max(1, int(round(period / dt)))
    cycle_peaks = [float(np.max(T[i * spc:min((i + 1) * spc, len(T))])) for i in range(max(1, len(T) // spc))]
    assert (float(L.mean()), float(T.max()), float(P.mean()), cycle_peaks, float(np.std(T))) == expected
    assert_same(state(data), expected_state)


def test_matches_run_fixed_frequency(design):
    scene, morph, control = design
    warmup, periods = 0.1, 0.15
    with final_data() as last, contextlib.redirect_stdout(io.StringIO()) as out:
        run_fixed_frequency.run(TG, scene, periods, warmup)
    expected_state = state(last["data"])

    model = mujoco.MjModel.from_xml_path(str(scene))
    sim = PlateWingSim(model, morph)
    data, start, ramp = sim.data, sim.data.time, warmup * control.period

    def step():
        return sim.step(drive_command(control, morph.motors.motor_direction, data.time, start, ramp))

    dt = model.opt.timestep
    for _ in range(math.ceil(warmup * control.period / dt)):
        step()
    lifts = np.array([step() for _ in range(math.ceil(periods * control.period / dt))])
    assert f"mean_lift_N {float(lifts.mean()):.9g}" in out.getvalue().splitlines()
    assert f"max_lift_N {float(lifts.max()):.9g}" in out.getvalue().splitlines()
    assert_same(state(data), expected_state)
