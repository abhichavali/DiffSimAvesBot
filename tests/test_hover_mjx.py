"""The MJX free flight (hover.mjx_flight) and sensor port (hover.mjx_sensors) against the CPU reference.

The device code is exact to float64 round-off when JAX runs in x64: a short CPU-JAX flight must match
hover.launch.fly to ~1e-9 (the float32 GPU default is a precision choice, measured by opt/hover_gpu.py parity).
"""
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from hover import mjx_flight as F

ROOT = Path(__file__).resolve().parents[1]


def test_quaternion_helpers_match_mujoco():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p, q = rng.normal(size=4), rng.normal(size=4)
        p /= np.linalg.norm(p); q /= np.linalg.norm(q)
        ref = np.zeros(4); mujoco.mju_mulQuat(ref, p, q)
        assert np.allclose(F.qmul(p, q), ref, atol=1e-6)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
        assert np.allclose(F.quat2mat(q), R.reshape(3, 3), atol=1e-6)
        rv = rng.normal(size=3) * 0.3
        qa = np.zeros(4); mujoco.mju_axisAngle2Quat(qa, rv / np.linalg.norm(rv), np.linalg.norm(rv))
        assert np.allclose(F.rotvec2quat(rv), qa, atol=1e-6)
    assert np.allclose(F.rotvec2quat(np.zeros(3)), [1, 0, 0, 0])


def test_finite_difference_tasks_recover_a_linear_map():
    K = 3
    rng = np.random.default_rng(1)
    A9, B0, d0 = rng.normal(size=(12, 9)), rng.normal(size=(12, 8 * K)), rng.normal(size=12)
    perts, dus = F.lqr_tasks(K)
    assert len(perts) == 1 + 2 * 9 + 2 * 8 * K
    out = d0 + perts[:, 3:] @ A9.T + dus @ B0.T
    A, B, drift = F.ab_from_tasks(out, K)
    assert np.allclose(A[:, 3:], A9) and np.allclose(A[:3, :3], np.eye(3)) and np.allclose(B, B0) and np.allclose(drift, d0)


SCRIPT = r"""
import json, sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from design_paths import FREE_HOVER_TG
from hover import launch as L, mjx_flight as F
from hover.policy import load_flight_policy
C = FREE_HOVER_TG.parent / "control.tg"
fp = load_flight_policy(C)
bird = L.load_bird(FREE_HOVER_TG, control=C)
rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock, n_cradle=3, n_avg=1)
Kg = np.load(FREE_HOVER_TG.parent / "lqr.npz")["K"]
kick = np.r_[0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.5, 0]
cpu = L.fly(bird, rel, fp.theta, 2, policy=lambda s: np.concatenate([[0.0], np.clip(-Kg @ s, -0.15, 0.15)]), kicks={1: kick})
fl = F.Flight(bird, rel, fp.theta, fp.K, k_phi=fp.phase_lock)
f = F.batch_flyer(fl, 2, "linear")
S, U, A = f(fl.arrays, jnp.asarray(np.stack([np.zeros(12), kick])[None]), jnp.asarray(Kg[None]), jnp.zeros((1, 12)))
out = {"flight": float(np.abs(np.asarray(S[0]) - cpu).max()), "alive": bool(np.asarray(A).all()),
       "scale": float(np.abs(cpu).max())}
try:
    from hover import sensors as Sn, reduced as Rd, mjx_sensors as MS
except ImportError:
    print(json.dumps(out)); sys.exit()
if "C_now" not in np.load(sys.argv[1]).files:            # an lqr.npz from before the sensor model
    print(json.dumps(out)); sys.exit()
ctl = Rd.design(Rd.load_model(sys.argv[1]), "pos,att,w", "lqg", noise=Sn.SensorNoise(scale=0.0))
sen = Sn.Sensors(Sn.SensorNoise.zero(), att_mode="filter")
cpu = L.fly(bird, rel, fp.theta, 2, policy=Rd.LinearController(ctl), kicks={1: kick}, sensor=sen)
sp = MS.SensorPipeline(fl, Sn.SensorNoise.zero(), att_mode="filter", ctl=ctl)
g = MS.batch_sensor_flyer(fl, sp, 2)
prm = sp.params()
S, Y, U, A = g(fl.arrays, jnp.asarray(np.stack([np.zeros(12), kick])[None]), jax.random.split(jax.random.PRNGKey(0), 1),
               jnp.zeros(1), jax.tree_util.tree_map(lambda x: x[None], prm.ctl))
out.update(sensor_flight=float(np.abs(np.asarray(S[0]) - cpu).max()),
           sensor_y=float(np.abs(np.asarray(Y[0]) - np.array(sen.log)).max()))
print(json.dumps(out))
"""


def test_mjx_flight_matches_cpu_in_x64():
    """Two wingbeats (LQR closed loop, a push at wingbeat 1) on CPU-JAX x64 vs hover.launch.fly; with the sensor
    model present (hover.sensors), also a noise-free LQG through the sensor pipeline."""
    lqr_sensor = ROOT / "designs" / "hummer" / "free-hover" / "lqr.npz"
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(lqr_sensor)], capture_output=True, text=True,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu", "PATH": ""}, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["alive"] and out["scale"] > 1e-3
    assert out["flight"] < 1e-8, out
    if "sensor_flight" in out:
        assert out["sensor_flight"] < 1e-8 and out["sensor_y"] < 1e-8, out
