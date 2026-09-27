"""What the per-wingbeat hover policy would really read: an IMU + motion-capture sensor model.

``hover.launch.fly(..., sensor=Sensors(...))`` runs this pipeline alongside the physics: the IMU is sampled
at its own output data rate (every ``imu_every`` physics steps) with white noise and a constant per-flight
bias, an on-board attitude filter integrates it at that rate, and once per wingbeat (at the strobe) the
policy reads the channel vector ``y`` (18) instead of the true strobed state. Every channel is a deviation
from its value on the release orbit:

    group  index   what                                                            source
    pos    0:3     COM position [m], world                                         mocap (non-IMU)
    att    3:6     attitude rotation vector vs the release attitude [rad], world   IMU attitude filter
    vel    6:9     COM velocity [m/s], world                                       mocap / flow (non-IMU)
    w      9:12    angular velocity [rad/s], world = R_est (gyro - bias_est)       IMU (gyro + filter)
    gyro   12:15   angular rate [rad/s] in the BODY (IMU = airframe) frame, minus  IMU
                   the cradle bias calibration (not the attitude filter's estimate)
    acc    15:18   specific force [m/s^2], body frame, mean over the last wingbeat IMU

The first 12 channels are the strobed state's layout: with perfect sensors (``SensorNoise.zero()``,
``att_mode="truth"``) ``y[:12]`` equals ``hover.launch.strobe - x_orbit`` bit for bit.

Firmware processing (why these choices):
- gyro / w: one IMU sample taken synchronously with the strobe (the firmware triggers a read on the crank
  index pulse). White noise at the ODR has per-sample std ND*sqrt(ODR) (Kalibr convention
  sigma_d = sigma_c / sqrt(dt)), so drawing the strobe-aligned sample directly is identical to generating the
  whole stream and picking that sample. Noise is small anyway: 0.014 deg/s/rtHz * sqrt(1 kHz) = 0.008 rad/s.
- acc: under flapping the specific force swings by several g within a wingbeat, so the only usable accel
  quantity is its mean over one wingbeat (a boxcar of the stream: this is generated at the ODR, sample by
  sample, from finite differences of the IMU point velocity). Its deviation is set by the input over the beat
  and by the aero (drag ~ velocity): the linear model gives it its own terms (hover.reduced: C_prev, J).
- att: a Mahony complementary filter. Gyro integration at the ODR; once per wingbeat a correction from the
  beat-mean specific force (compared with the beat-mean of the estimated "up" in the body, so the flapping
  wobble cancels) with gain ``att_kp`` per wingbeat and a gyro-bias integrator ``att_ki``. Only roll/pitch are
  corrected: yaw drifts with the residual z gyro bias (no magnetometer). During a push the thrust tilts with
  the body and misleads the correction: that is the real multirotor trade-off, kept on purpose.
- Initial filter state at release: the filter has converged in the cradle, so roll/pitch sit at the tilt the
  accelerometer bias implies (the steady-state error), yaw error 0 (heading is defined at release); the gyro
  bias estimate comes from averaging the gyro over ``gyro_cal_s`` in the cradle (residual std
  ND / sqrt(gyro_cal_s)); ``gyro_cal_s = 0`` = no calibration.

Everything stochastic scales with ``noise_scale`` (``SensorNoise.scale``); pos / vel have their own extra
factors. Latency (deterministic) and the filter gains do not scale.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, replace

import mujoco
import numpy as np

G = 9.81          # matches the model's gravity (hover.launch.release sets (0, 0, -9.81))
DEG = math.pi / 180

GROUPS = {"pos": 0, "att": 3, "vel": 6, "w": 9, "gyro": 12, "acc": 15}
CHANNELS = [f"{g}_{ax}" for g in GROUPS for ax in "xyz"]
N_CH = len(CHANNELS)
STATE_CHANNELS = tuple(range(12))       # y[:12] is the strobed state's layout
NON_IMU = ("pos", "vel")
INTERVAL = tuple(range(15, 18))         # channels that are a mean over the last wingbeat (acc)
ALIASES = {"state": "pos,att,vel,w", "imu": "att,w,gyro,acc", "all": ",".join(GROUPS), "none": ""}
# O(1) input scaling per channel for MLPs (pos/5 cm, att/0.1 rad, vel/0.2 m/s, rates/1 rad/s, acc/1 m/s^2),
# the first 12 = hover.env.OBS_SCALE
CHANNEL_SCALE = np.array([20.0] * 3 + [10.0] * 3 + [5.0] * 3 + [1.0] * 3 + [1.0] * 3 + [1.0] * 3)


def parse_obs(spec) -> np.ndarray:
    """Channel indices (sorted, unique) for a subset spec: comma/space separated groups (pos att vel w gyro acc),
    aliases (state imu all none), or per-axis picks ``<group>_<axes>`` (pos_z, att_xy, vel_xz). An int
    sequence is passed through."""
    if spec is None:
        spec = "state"
    if not isinstance(spec, str):
        idx = np.unique(np.asarray(spec, int))
        if idx.size and (idx.min() < 0 or idx.max() >= N_CH):
            raise ValueError(f"channel index out of range 0..{N_CH - 1}: {idx}")
        return idx
    out = []
    for tok in (t for t in re.split(r"[,\s+]+", spec.strip().lower()) if t):
        if tok in ALIASES:
            out += list(parse_obs(ALIASES[tok])) if ALIASES[tok] else []
            continue
        m = re.fullmatch(r"(pos|att|vel|w|gyro|acc)(?:_([xyz]+))?", tok)
        if not m:
            raise ValueError(f"unknown observation {tok!r}: use groups {list(GROUPS)}, aliases {list(ALIASES)} "
                             f"or <group>_<axes> like pos_z / att_xy")
        out += [GROUPS[m[1]] + "xyz".index(a) for a in (m[2] or "xyz")]
    return np.unique(np.asarray(out, int))


def obs_names(idx) -> list[str]:
    return [CHANNELS[i] for i in idx]


def n_non_imu(idx) -> int:
    return sum(CHANNELS[i].split("_")[0] in NON_IMU for i in idx)


@dataclass(frozen=True)
class SensorNoise:
    """Sensor noise parameters. Defaults = a realistic hobby-grade IMU (datasheet numbers below) + a lab mocap.

    Gyro (BMI088, Bosch BST-BMI088-DS001 Table 2: rate noise density 0.014 deg/s/rtHz, zero-rate offset
    +-1 deg/s; ICM-42688-P, TDK DS-000347: 0.0028 deg/s/rtHz, ZRO +-0.5 deg/s). Bias instability of both is a
    few deg/h (<0.01 deg over a 15 s flight): ignored, the bias is a constant per flight.
    Accel (BMI088: 175 ug/rtHz typ (160-190), zero-g offset +-20 mg; ICM-42688-P: 70 ug/rtHz, +-20 mg).
    Mocap (Vicon / OptiTrack marker RMS 0.1-0.5 mm; end-to-end latency 3-10 ms, off by default).
    Velocity (mocap-differentiated + filtered, or optical flow): 0.02 m/s, an assumption, not a datasheet.
    """
    imu_hz: float = 1000.0                   # IMU output data rate [Hz]
    gyro_nd: float = 0.014 * DEG             # gyro white noise density [rad/s/rtHz] (BMI088)
    gyro_bias: float = 0.5 * DEG             # per-axis std of the per-flight gyro bias [rad/s]
    gyro_cal_s: float = 0.7                  # cradle gyro calibration time [s] (6 wingbeats); 0 = none
    acc_nd: float = 175e-6 * 9.80665         # accel white noise density [m/s^2/rtHz] (BMI088)
    acc_bias: float = 20e-3 * 9.80665        # per-axis std of the per-flight accel bias [m/s^2]
    pos_std: float = 0.3e-3                  # mocap position white noise [m]
    pos_latency_ms: float = 0.0              # mocap latency [ms] (deterministic; realistic 3-10)
    vel_std: float = 0.02                    # velocity white noise [m/s]
    att_kp: float = 0.1                      # attitude filter: fraction of the tilt error corrected per wingbeat
    att_ki: float = 0.0025                   # bias integrator: kp^2/4 = critically damped [rad/s per rad per wingbeat / T]
    scale: float = 1.0                       # noise_scale: multiplies every stochastic term
    pos_scale: float = 1.0                   # extra factor on pos_std (mocap quality)
    vel_scale: float = 1.0                   # extra factor on vel_std

    @classmethod
    def zero(cls, **kw) -> "SensorNoise":
        return cls(scale=0.0, **kw)

    def scaled(self, scale: float) -> "SensorNoise":
        return replace(self, scale=float(scale))

    # effective (scaled) stds
    def gyro_sample_std(self) -> float:
        return self.scale * self.gyro_nd * math.sqrt(self.imu_hz)

    def acc_sample_std(self) -> float:
        return self.scale * self.acc_nd * math.sqrt(self.imu_hz)

    def gyro_bias_resid(self) -> float:
        """Std of (true bias - estimated bias) at release."""
        if self.gyro_cal_s > 0:
            return self.scale * self.gyro_nd / math.sqrt(self.gyro_cal_s)
        return self.scale * self.gyro_bias

    def channel_std(self, beat_s: float, flight_s: float = 14.0) -> np.ndarray:
        """Per-channel std of one per-wingbeat reading (for the Kalman filter's V). Biases are folded in as
        extra white variance (they are constant per flight: an approximation, stated)."""
        g = math.hypot(self.gyro_sample_std(), self.gyro_bias_resid())
        a = math.hypot(self.acc_sample_std() / math.sqrt(max(1.0, beat_s * self.imu_hz)), self.scale * self.acc_bias)
        tilt = self.scale * self.acc_bias / G                                # accel-bias tilt of the filter
        yaw = self.gyro_bias_resid() * flight_s / 2                          # mean yaw drift over a flight
        att = [tilt, tilt, max(yaw, 1e-3 * self.scale)]
        return np.array([self.scale * self.pos_scale * self.pos_std] * 3 + att + [self.scale * self.vel_scale * self.vel_std] * 3
                        + [g] * 3 + [g] * 3 + [a] * 3)


def _rotvec_quat(v) -> np.ndarray:
    q = np.array([1.0, 0, 0, 0])
    a = float(np.linalg.norm(v))
    if a > 0:
        mujoco.mju_axisAngle2Quat(q, np.asarray(v, float) / a, a)
    return q


def _quat_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(q, float))
    return R.reshape(3, 3)


def att_error(q, q_ref) -> np.ndarray:
    """Rotation vector (small-angle, world) of q vs q_ref, exactly as hover.launch.strobe computes it."""
    qc = np.zeros(4); mujoco.mju_negQuat(qc, q_ref); qe = np.zeros(4); mujoco.mju_mulQuat(qe, q, qc)
    return 2 * np.sign(qe[0]) * qe[1:]


class Sensors:
    """The sensor pipeline for one flight at a time (``start`` resets it; the RNG stream continues, so seed
    per flight for reproducible Monte Carlo). ``att_mode``: "filter" (the on-board estimate) or "truth"."""

    def __init__(self, noise: SensorNoise | None = None, *, att_mode="filter", seed=0, reference=None):
        if att_mode not in ("filter", "truth"):
            raise ValueError("att_mode must be 'filter' or 'truth'")
        self.noise = noise if noise is not None else SensorNoise()
        self.att_mode = att_mode
        self.rng = np.random.default_rng(seed)
        self._fixed_ref = reference           # (gyro_ref, acc_ref) or None = compute from the orbit
        self._ref_cache = {}
        self.log = []

    # -- orbit reference --------------------------------------------------------------------------------------
    def reference(self, bird, rel, theta):
        """Noise-free (gyro, acc, up) readings on the release orbit: body rate at release, and the specific force
        and the true "up" in the body (R^T z), both averaged over the first unperturbed wingbeat. Cached per
        released state. acc vs up is the attitude filter's calibration: on the orbit the beat-mean specific
        force at the IMU is NOT along the beat-mean up (lever arm of the IMU to the COM under the ~4 rad/s
        flapping wobble, ~5 deg here); firmware learns this constant body-frame offset in the cradle."""
        if self._fixed_ref is not None:
            return self._fixed_ref
        key = (id(rel), np.asarray(theta, float).tobytes(), bird.model.opt.timestep, self.noise.imu_hz)
        if key not in self._ref_cache:
            from hover import launch as L
            probe = Sensors(SensorNoise.zero(imu_hz=self.noise.imu_hz), att_mode="truth",
                            reference=(np.zeros(3), np.zeros(3), None))
            L.fly(bird, rel, theta, 1, sensor=probe)
            gyro0 = rel.sim.data.qvel[bird.vadr + 3:bird.vadr + 6].copy()
            self._ref_cache[key] = (gyro0, probe.log[-1][15:18].copy(), probe.last_up_true)
        return self._ref_cache[key]

    # -- per flight -------------------------------------------------------------------------------------------
    def start(self, bird, rel, sim, theta=None, *, ref=None):
        """Reset for a flight starting now from ``sim.data`` (the released or perturbed state)."""
        nz, d, m = self.noise, sim.data, sim.model
        self.bird, self.rel = bird, rel
        self.ref = ref if ref is not None else self.reference(bird, rel, theta)
        self.q_cal = None                                  # rotation (body) taking the orbit's mean f to its mean up
        if self.ref[2] is not None:
            f0, u0 = self.ref[1] / np.linalg.norm(self.ref[1]), self.ref[2] / np.linalg.norm(self.ref[2])
            ax = np.cross(f0, u0); s0 = np.linalg.norm(ax)
            self.q_cal = _quat_mat(_rotvec_quat(ax / s0 * math.atan2(s0, float(f0 @ u0)) if s0 > 0 else np.zeros(3)))
        self.dt = m.opt.timestep
        self.every = max(1, round(1.0 / (nz.imu_hz * self.dt)))
        self.dt_imu = self.every * self.dt
        self.k = 0
        self.sg, self.sa = nz.gyro_sample_std(), nz.acc_sample_std()
        self.bg = self.rng.normal(0, nz.scale * nz.gyro_bias, 3)
        self.ba = self.rng.normal(0, nz.scale * nz.acc_bias, 3)
        self.bg_hat = (self.bg + self.rng.normal(0, nz.scale * nz.gyro_nd / math.sqrt(nz.gyro_cal_s), 3)
                       if nz.gyro_cal_s > 0 else np.zeros(3))
        self.bg_cal = self.bg_hat.copy()                  # the cradle calibration the gyro channel subtracts
        qa = bird.qadr
        q_true = d.qpos[qa + 3:qa + 7].copy()
        # converged-in-the-cradle filter: its "up" = the biased accelerometer's up
        up = _quat_mat(q_true).T @ np.array([0, 0, 1.0])
        up_b = (G * up + self.ba) / np.linalg.norm(G * up + self.ba)
        ax = np.cross(up_b, up)                           # R_hat = R Rot, Rot takes up_b to up
        s_ = np.linalg.norm(ax)
        rot = _rotvec_quat(ax / s_ * math.atan2(s_, float(up_b @ up)) if s_ > 0 else np.zeros(3))
        self.q_hat = np.zeros(4); mujoco.mju_mulQuat(self.q_hat, q_true, rot)
        self.v_prev = d.qvel[bird.vadr:bird.vadr + 3].copy()
        self.sum_f, self.sum_up, self.sum_up_true, self.n_acc = np.zeros(3), np.zeros(3), np.zeros(3), 0
        self.last_up_true = None
        self.lag = int(round(nz.pos_latency_ms * 1e-3 / self.dt))
        self.buf = deque([d.subtree_com[0].copy()] * self.lag, maxlen=self.lag) if self.lag else None
        self.log = []

    def step(self, d):
        """After every physics step (Drive.beat's on_step)."""
        if self.buf is not None:
            self.buf.append(d.subtree_com[0].copy())      # COM one step ago (kinematics run before the step)
        self.k += 1
        if self.k % self.every:
            return
        b = self.bird
        R = _quat_mat(d.qpos[b.qadr + 3:b.qadr + 7])
        v = d.qvel[b.vadr:b.vadr + 3].copy()
        a = (v - self.v_prev) / self.dt_imu
        self.v_prev = v
        n = self.rng.standard_normal(6)
        f = R.T @ (a + np.array([0, 0, G])) + self.ba + self.sa * n[3:]
        self.sum_f += f
        self.sum_up_true += R[2]
        self.n_acc += 1
        if self.att_mode == "filter":
            wg = d.qvel[b.vadr + 3:b.vadr + 6] + self.bg + self.sg * n[:3]
            mujoco.mju_quatIntegrate(self.q_hat, wg - self.bg_hat, self.dt_imu)
            mujoco.mju_normalize4(self.q_hat)
            self.sum_up += _quat_mat(self.q_hat)[2]      # R_hat^T z = third row of R_hat

    def read(self, d, s):
        """Channel vector y (18) at the strobe; ``s`` = the true strobed deviation state (12), kinematics fresh."""
        nz, b, rel = self.noise, self.bird, self.rel
        x0 = rel.x_orbit
        y = np.zeros(N_CH)
        if self.n_acc:
            fbar = self.sum_f / self.n_acc
            y[15:18] = fbar - self.ref[1]
            self.last_up_true = self.sum_up_true / self.n_acc
            if self.att_mode == "filter":
                fc = self.q_cal @ fbar if self.q_cal is not None else fbar
                e = np.cross(fc / np.linalg.norm(fc), self.sum_up / np.linalg.norm(self.sum_up))
                mujoco.mju_quatIntegrate(self.q_hat, nz.att_kp * e, 1.0)
                mujoco.mju_normalize4(self.q_hat)
                self.bg_hat = self.bg_hat - nz.att_ki / (self.n_acc * self.dt_imu) * e
        self.sum_f, self.sum_up, self.sum_up_true, self.n_acc = np.zeros(3), np.zeros(3), np.zeros(3), 0
        n = self.rng.standard_normal(9)
        if self.lag:
            y[0:3] = self.buf[0] - rel.com_ref - x0[0:3] + nz.scale * nz.pos_scale * nz.pos_std * n[0:3]
        else:
            y[0:3] = s[0:3] + nz.scale * nz.pos_scale * nz.pos_std * n[0:3]
        y[6:9] = s[6:9] + nz.scale * nz.vel_scale * nz.vel_std * n[3:6]
        gyro = d.qvel[b.vadr + 3:b.vadr + 6] + self.bg + self.sg * n[6:9]
        y[12:15] = gyro - self.bg_cal - self.ref[0]
        if self.att_mode == "truth":
            y[3:6] = s[3:6]
            y[9:12] = d.xmat[b.af].reshape(3, 3) @ (gyro - self.bg_hat) - x0[9:12]
        else:
            y[3:6] = att_error(self.q_hat, rel.q_ref) - x0[3:6]
            y[9:12] = _quat_mat(self.q_hat) @ (gyro - self.bg_hat) - x0[9:12]
        self.log.append(y)
        return y
