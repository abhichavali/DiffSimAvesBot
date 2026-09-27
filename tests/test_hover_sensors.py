"""Sensor model (hover.sensors), reduced-observation controllers (hover.reduced) and the flight plumbing."""
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from hover import reduced as Rd
from hover import sensors as S
from hover.lqr import lqr

DT = 1e-4                        # fake physics step; IMU every 10 steps at 1 kHz
Z = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------- channel masks
def test_parse_obs_groups_axes_aliases():
    assert list(S.parse_obs("state")) == list(range(12))
    assert list(S.parse_obs("pos_z")) == [2]
    assert list(S.parse_obs("att_xy, vel_z")) == [3, 4, 8]
    assert list(S.parse_obs("w gyro")) == list(range(9, 15))
    assert list(S.parse_obs("imu")) == [3, 4, 5] + list(range(9, 18))
    assert list(S.parse_obs("acc,att,acc")) == [3, 4, 5, 15, 16, 17]          # sorted, unique
    assert S.parse_obs("none").size == 0
    assert S.n_non_imu(S.parse_obs("pos,att,vel_z")) == 4
    assert S.obs_names(S.parse_obs("gyro_y")) == ["gyro_y"]
    with pytest.raises(ValueError):
        S.parse_obs("mag")
    with pytest.raises(ValueError):
        S.parse_obs("pos_q")


# ------------------------------------------------------------------------------------------ a fake rigid body
class Body:
    """Minimal stand-in for (bird, rel, sim) with one free body: qpos = [p, quat], qvel = [v world, w body]."""

    def __init__(self, quat=(1, 0, 0, 0), w_body=(0, 0, 0)):
        self.bird = SimpleNamespace(qadr=0, vadr=0, af=0, model=None)
        q = np.asarray(quat, float) / np.linalg.norm(quat)
        self.d = SimpleNamespace(qpos=np.r_[0.0, 0, 0, q], qvel=np.r_[0.0, 0, 0, w_body], subtree_com=np.zeros((1, 3)),
                                 xmat=np.zeros((1, 9)))
        self.sim = SimpleNamespace(data=self.d, model=SimpleNamespace(opt=SimpleNamespace(timestep=DT)))
        self.rel = SimpleNamespace(com_ref=np.zeros(3), q_ref=q.copy(), x_orbit=np.zeros(12))
        self._sync()

    def _sync(self):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, self.d.qpos[3:7]); self.d.xmat[0] = R

    def advance(self):
        """One physics step of free rotation at the constant body rate."""
        mujoco.mju_quatIntegrate(self.d.qpos[3:7], self.d.qvel[3:6], DT)
        mujoco.mju_normalize4(self.d.qpos[3:7])
        self._sync()

    def state(self):
        R = self.d.xmat[0].reshape(3, 3)
        from hover.sensors import att_error
        return np.r_[np.zeros(3), att_error(self.d.qpos[3:7], self.rel.q_ref), np.zeros(3), R @ self.d.qvel[3:6]]

    def up(self):
        return self.d.xmat[0].reshape(3, 3)[2]


def fly_fake(body, sen, n_reads, steps_per_read=1000, ref=None):
    ref = ref if ref is not None else (np.zeros(3), np.array([0, 0, S.G]), None)
    sen.start(body.bird, body.rel, body.sim, ref=ref)
    Y = [sen.read(body.d, body.state())]
    for _ in range(n_reads):
        for _ in range(steps_per_read):
            body.advance()
            sen.step(body.d)
        Y.append(sen.read(body.d, body.state()))
    return np.array(Y)


# ------------------------------------------------------------------------------------------- noise statistics
def test_white_noise_and_bias_statistics():
    nz = S.SensorNoise(gyro_bias=0.0, acc_bias=0.0, gyro_cal_s=0.0)
    Y = fly_fake(Body(), S.Sensors(nz, att_mode="truth", seed=3), 400, steps_per_read=100)
    g = Y[1:, 12:15]
    np.testing.assert_allclose(g.std(0), nz.gyro_sample_std(), rtol=0.15)       # ND sqrt(ODR) per sample
    a = Y[1:, 15:18]                                                             # mean of 10 IMU samples
    np.testing.assert_allclose(a.std(0), nz.acc_sample_std() / math.sqrt(10), rtol=0.15)
    assert abs(a.mean()) < 3 * nz.acc_sample_std() / math.sqrt(10 * 400)
    p, v = Y[1:, 0:3], Y[1:, 6:9]
    np.testing.assert_allclose(p.std(0), nz.pos_std, rtol=0.15)
    np.testing.assert_allclose(v.std(0), nz.vel_std, rtol=0.15)


def test_bias_is_constant_per_flight_and_random_across_flights():
    nz = S.SensorNoise(gyro_nd=0.0, acc_nd=0.0, gyro_cal_s=0.0)
    biases = []
    for seed in range(200):
        Y = fly_fake(Body(), S.Sensors(nz, att_mode="truth", seed=seed), 3, steps_per_read=20)
        assert np.allclose(Y[1:, 12:15], Y[1, 12:15])                          # constant within a flight
        biases.append(Y[1, 12:15])
    np.testing.assert_allclose(np.std(biases, 0), nz.gyro_bias, rtol=0.2)


def test_seed_reproducible_and_scale_zero_is_exact():
    y1 = fly_fake(Body(w_body=(0.1, 0.2, 0.3)), S.Sensors(seed=7), 5, 50)
    y2 = fly_fake(Body(w_body=(0.1, 0.2, 0.3)), S.Sensors(seed=7), 5, 50)
    assert np.array_equal(y1, y2)
    b = Body(quat=(0.9, 0.1, -0.3, 0.2), w_body=(0.4, -0.2, 0.7))
    Y = fly_fake(b, S.Sensors(S.SensorNoise.zero(), att_mode="truth"), 3, 50)
    assert np.array_equal(Y[-1, :12], b.state())                                # perfect sensors = true state


# ------------------------------------------------------------------------------------------ frame conversion
def test_gyro_is_body_frame_and_w_is_world_frame():
    wb = np.array([0.5, -1.0, 2.0])
    b = Body(quat=(0.8, 0.3, -0.4, 0.2), w_body=wb)
    for mode in ("truth", "filter"):
        Y = fly_fake(b, S.Sensors(S.SensorNoise.zero(), att_mode=mode), 4, 100)
        np.testing.assert_allclose(Y[-1, 12:15], wb, atol=1e-12)                # IMU frame = body frame
        R = b.d.xmat[0].reshape(3, 3)
        np.testing.assert_allclose(Y[-1, 9:12], R @ wb, atol=2e-3)             # filter: integrated attitude
        assert not np.allclose(R @ wb, wb, atol=0.1)


# ------------------------------------------------------------------------------------------- attitude filter
def test_filter_tracks_rotation_and_settles_at_the_accel_bias_tilt():
    ba = np.array([0.15, -0.1, 0.0])                                             # accel bias [m/s^2]
    nz = S.SensorNoise(gyro_nd=0, acc_nd=0, gyro_bias=0, acc_bias=0, pos_std=0, vel_std=0)
    sen = S.Sensors(nz, att_mode="filter", seed=0)
    b = Body(quat=(0.95, 0.05, 0.3, 0.0))
    sen.start(b.bird, b.rel, b.sim, ref=(np.zeros(3), np.array([0, 0, S.G]), None))
    sen.ba = ba                                                                  # a known bias
    sen.q_hat = b.d.qpos[3:7].copy()                                             # start at truth: must walk to the bias tilt
    for _ in range(80):
        for _ in range(1000):
            b.advance(); sen.step(b.d)
        y = sen.read(b.d, b.state())
    # steady state: the estimated up (body) points along the biased accelerometer
    R_hat = np.zeros(9); mujoco.mju_quat2Mat(R_hat, sen.q_hat)
    up_est = R_hat.reshape(3, 3)[2]
    f = S.G * b.up() + ba
    assert np.degrees(np.arccos(np.clip(up_est @ f / np.linalg.norm(f), -1, 1))) < 0.05
    expected = np.degrees(np.arccos(b.up() @ f / np.linalg.norm(f)))           # the bias's tilt, ~0.9 deg
    assert expected > 0.5 and abs(np.degrees(np.linalg.norm(y[3:6])) - expected) < 0.05


def test_filter_learns_roll_pitch_gyro_bias_but_yaw_drifts():
    bg = np.array([0.01, -0.02, 0.015])                                          # rad/s
    nz = S.SensorNoise(gyro_nd=0, acc_nd=0, gyro_bias=0, acc_bias=0, pos_std=0, vel_std=0, gyro_cal_s=0)
    sen = S.Sensors(nz, att_mode="filter")
    b = Body()
    sen.start(b.bird, b.rel, b.sim, ref=(np.zeros(3), np.array([0, 0, S.G]), None))
    sen.bg = bg
    for _ in range(300):                                                         # 30 s at 10 reads/s
        for _ in range(1000):
            b.advance(); sen.step(b.d)
        y = sen.read(b.d, b.state())
    np.testing.assert_allclose(sen.bg_hat[:2], bg[:2], atol=1e-3)
    assert np.degrees(np.linalg.norm(y[3:5])) < 0.5                              # roll/pitch held
    assert abs(y[5]) > 0.5 * bg[2] * 30                                          # yaw drifts ~ bias x time


# --------------------------------------------------------------------------- reduced controllers on a toy bird
def toy_model(T=0.1):
    """12-state toy: pos += T vel, att += T w, vel += T g tilt, w += T u; 24 inputs (first 4 used)."""
    A = np.eye(12)
    A[0:3, 6:9] = T * np.eye(3); A[3:6, 9:12] = T * np.eye(3)
    A[6, 4], A[7, 3] = T * 9.81, -T * 9.81
    A[9:12, 9:12] *= 1.02                                                        # mildly unstable rates
    B = np.zeros((12, 24)); B[9:12, 0:3] = T * 10 * np.eye(3); B[8, 3] = T * 5
    Cn = np.zeros((18, 12)); Cn[:12] = np.eye(12); Cn[12:15, 9:12] = np.eye(3)
    Cp = np.zeros((18, 12)); Cp[15:18, 6:9] = -0.5 * np.eye(3)                  # acc ~ drag on last beat's velocity
    return {"A": A, "B": B, "K": lqr(A, B), "drift": np.zeros(12), "C_now": Cn, "C_prev": Cp, "J": np.zeros((18, 24))}


def test_observability_report():
    m = toy_model()
    r = Rd.observability(m, "att,w")                    # the toy's velocity reaches no attitude/rate channel
    assert r["unobservable_states"] == ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"] and not r["design_detectable"]
    r = Rd.observability(m, "att,w,acc")                # the accelerometer's drag term sees it
    assert r["unobservable_states"] == ["pos_x", "pos_y", "pos_z"] and not r["detectable"] and r["design_detectable"]
    r = Rd.observability(m, "pos_z,att,w")
    assert r["unobservable_states"][:2] == ["pos_x", "pos_y"] and "pos_z" in r["observable_states"]
    assert Rd.observability(m, "state")["detectable"]
    assert list(Rd.design_states(S.parse_obs("pos_z,att"))) == [2] + list(range(3, 12))


def test_sof_is_stable_and_no_better_than_lqr():
    m = toy_model()
    Ky, info = Rd.sof(m, "pos,att,w,acc")              # acc's drag term stands in for the velocity
    assert info["ok"] and info["rho"] < 1 and info["J_ratio"] >= 1 - 1e-9
    assert not Rd.sof(m, "pos,att,w")[1]["ok"]         # no velocity: no static gain damps the position chain
    Ky2, info2 = Rd.sof(m, "state")                                              # full state: SOF = LQR
    assert info2["J_ratio"] < 1 + 1e-6


def test_lqg_observer_recovers_unmeasured_velocity():
    m = toy_model()
    nz = S.SensorNoise(scale=0.2)
    ctl = Rd.design(m, "pos,att,w", "lqg", noise=nz, beat_s=0.1)
    assert ctl["info"]["rho_closed_loop"] < 1
    ctl["u_max"] = 10.0                                                          # the toy's inputs are weak: no saturation
    pol = Rd.LinearController(ctl)
    rng = np.random.default_rng(0)
    x, x_prev = 0.2 * np.r_[0.01, -0.02, 0.0, 0.02, -0.03, 0.01, 0.1, -0.05, 0.08, 0.2, -0.1, 0.1], np.zeros(12)
    std = nz.channel_std(0.1)
    err = []
    for k in range(80):
        y = m["C_now"] @ x + m["C_prev"] @ x_prev + std * rng.standard_normal(18)
        u = pol(y)[1:]
        err.append(np.abs(pol.log[-1][6:9] - x[6:9]))                           # design states = all 12 here
        x_prev, x = x, m["A"] @ x + m["B"] @ u + Rd.PROCESS_STD * rng.standard_normal(12)
    assert np.mean(err[-40:]) < 0.01                                             # vel estimated from pos + model
    rms, pstd = np.sqrt(np.mean(np.square(err[-40:]))), np.mean(ctl["P_post_std"][6:9])
    assert 0.5 * pstd < rms < 2 * pstd                                           # the filter's own error estimate is right
    assert np.abs(x).max() < 0.3                                                 # regulated (process noise keeps it moving)


def test_lqg_imu_only_drops_position():
    m = toy_model()
    ctl = Rd.design(m, "att,w,acc", "lqg", noise=S.SensorNoise(), beat_s=0.1)
    assert list(ctl["xs"]) == list(range(3, 12)) and ctl["info"]["rho_closed_loop"] < 1


def test_dead_reckoning_holds_position_against_a_constant_disturbance():
    m = toy_model()
    m["drift"] = np.zeros(12); m["drift"][6] = 2e-3                             # a constant push on vel_x per beat
    final = {}
    for dr in (False, True):
        ctl = Rd.design(m, "att,w,acc", "lqg", noise=S.SensorNoise(scale=0.0), beat_s=0.1, dead_reckon=dr)
        assert ("dr" in ctl) == dr and (not dr or list(ctl["dr"]) == [0, 1, 2])
        ctl["u_max"] = 10.0
        pol, x, x_prev = Rd.LinearController(ctl), np.zeros(12), np.zeros(12)
        for k in range(400):
            u = pol(m["C_now"] @ x + m["C_prev"] @ x_prev)[1:]
            x_prev, x = x, m["A"] @ x + m["B"] @ u + m["drift"]
        final[dr] = x
    assert abs(final[False][6]) > 1e-3 and abs(final[False][0]) > 0.3             # without: a steady velocity, drifting
    assert abs(final[True][6]) < 1e-6 and abs(final[True][0]) < 0.05              # with: velocity nulled, position held


def test_lqg_bias_state_removes_a_constant_attitude_offset():
    m = toy_model()
    b = np.r_[0.03, -0.02]                                                       # the attitude filter's tilt error
    final = {}
    for bias in (None, "att_xy"):
        ctl = Rd.design(m, "pos,att,w", "lqg", noise=S.SensorNoise(scale=0.2), beat_s=0.1, bias=bias, bias_walk=1e-3)
        assert ctl["Az"].shape[0] == 24 + (2 if bias else 0) and ctl["info"]["rho_closed_loop"] < 1
        ctl["u_max"] = 10.0
        pol, x, x_prev = Rd.LinearController(ctl), np.zeros(12), np.zeros(12)
        for k in range(1500):
            y = m["C_now"] @ x + m["C_prev"] @ x_prev
            y[3:5] += b
            u = pol(y)[1:]
            x_prev, x = x, m["A"] @ x + m["B"] @ u
        final[bias] = x
    assert np.abs(final[None][:2]).max() > 0.01                                  # the offset parks the bird off target
    assert np.abs(final["att_xy"][:2]).max() < 1e-3                              # estimated: position held
    with pytest.raises(ValueError):
        Rd.design(m, "pos,w", "lqg", beat_s=0.1, bias="att_xy")                  # bias channels must be read


def test_gyro_channel_subtracts_the_cradle_calibration():
    nz = S.SensorNoise(gyro_nd=0.0, acc_nd=0.0, acc_bias=0.0, gyro_cal_s=0.7)   # bias, calibrated exactly (no white)
    Y = fly_fake(Body(w_body=(0.1, 0.2, 0.3)), S.Sensors(nz, att_mode="truth", seed=4), 3, 50)
    np.testing.assert_allclose(Y[-1, 12:15], [0.1, 0.2, 0.3], atol=1e-12)


def test_state_controller_matches_full_state_lqr_bit_for_bit():
    m = toy_model()
    ctl = {"kind": "state", "obs": np.arange(12), "K": m["K"], "u_max": 0.15}
    s = np.random.default_rng(1).normal(0, 0.05, 12)
    y18 = np.r_[s, np.ones(6)]
    ref = np.concatenate([[0.0], np.clip(-m["K"] @ s, -0.15, 0.15)])
    pol = Rd.LinearController(ctl)
    assert np.array_equal(pol(s), ref) and np.array_equal(pol(y18), ref)
    with pytest.raises(ValueError):
        Rd.LinearController({**ctl, "obs": S.parse_obs("gyro"), "K": m["K"][:, :3]})(s)


def test_controller_round_trips_through_npz(tmp_path):
    m = toy_model()
    ctl = Rd.design(m, "att,w", "sof", beat_s=0.1)
    Rd.save_controller(tmp_path / "c.npz", ctl)
    back = Rd.load_controller(tmp_path / "c.npz")
    assert back["kind"] == "sof" and Rd.is_controller(tmp_path / "c.npz")
    y = np.random.default_rng(2).normal(0, 0.1, 18)
    assert np.array_equal(Rd.LinearController(ctl)(y), Rd.LinearController(back)(y))


# ------------------------------------------------------------------------------------- the real bird (a few s)
def test_perfect_sensors_fly_bit_identical_to_the_true_state():
    from design_paths import FREE_HOVER_TG
    from hover import launch as L
    from hover.policy import load_flight_policy
    control = FREE_HOVER_TG.parent / "control.tg"
    fp = load_flight_policy(control)
    bird = L.load_bird(FREE_HOVER_TG, control=control)
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock)
    kicks = {1: np.r_[np.zeros(6), 0.2, 0, 0, 1.0, 0, 0]}
    tr0 = L.fly(bird, rel, fp.theta, 3, policy=fp.correction, kicks=kicks)
    sen = S.Sensors(S.SensorNoise.zero(), att_mode="truth")
    tr1 = L.fly(bird, rel, fp.theta, 3, policy=lambda y: fp.correction(y[:12]), kicks=kicks, sensor=sen)
    assert np.array_equal(tr0, tr1)
    assert np.array_equal(np.array(sen.log)[:, :12], tr1)
