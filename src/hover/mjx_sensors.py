"""The hover sensor model and the reduced-observation controllers on the device (JAX port of hover.sensors +
hover.reduced.LinearController), so noise Monte Carlo runs as thousands of batched MJX flights (hover.mjx_flight).

Same pipeline as the CPU reference (read hover.sensors' docstring for the physics): the IMU is sampled every
``every`` physics steps (gyro + accelerometer white noise, constant per-flight biases), a Mahony attitude filter
integrates the gyro at the IMU rate and is corrected once per wingbeat from the beat-mean specific force, the
mocap position can lag, and once per wingbeat the controller reads the 18-channel vector y. The controller
(state / sof / lqg, optional dead reckoning, optional distilled MLP) carries its observer across wingbeats.

Randomness: one PRNG key per flight (``jax.random``), so a device flight and a CPU flight with the same "seed"
draw different noise: compare noise statistics, not traces. With zero noise the two pipelines are
deterministic and match to the precision of the flight itself (tests/test_hover_mjx.py).

Per-flight parameters (batchable with vmap): ``Params(scale, ctl)``: the noise scale and the controller arrays
(an LQG designed per noise scale has the same shapes for every scale). Everything structural (channels, kind,
latency, IMU rate, attitude mode) is static.

    sp = SensorPipeline(fl, noise, att_mode="filter", ctl=hover.reduced.design(...))
    f = batch_sensor_flyer(fl, sp, n_beats)          # jit(vmap) over (kicks, keys, params)
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hover import mjx_flight as F

G = 9.81
N_CH = 18


class Params(NamedTuple):
    scale: jax.Array          # noise scale (SensorNoise.scale)
    ctl: dict                 # controller arrays (see CTL_KEYS)


CTL_KEYS = ("K", "Ky", "J", "Az", "Bz", "Cz", "cz", "L", "K_dr", "A_dr", "B_dr", "d_dr")


def ctl_arrays(ctl: dict | None) -> dict:
    """The numeric arrays of a hover.reduced controller dict (what may differ per flight)."""
    if ctl is None:
        return {}
    return {k: jnp.asarray(np.asarray(ctl[k], float), F.fdtype()) for k in CTL_KEYS if k in ctl}


def _quat_integrate(q, w, scale):
    """mju_quatIntegrate: q <- q * quat(w scale), then normalized (mju_normalize4)."""
    q = F.qmul(q, F.rotvec2quat(w * scale))
    return q / jnp.linalg.norm(q)


def _rot_between(a, b):
    """Rotation vector taking unit a to unit b (hover.sensors' axis-angle construction)."""
    ax = jnp.cross(a, b)
    s2 = ax @ ax
    safe = s2 > 1e-24                                    # not just > 0: the GPU flushes denormals (see F.rotvec2quat)
    s = jnp.sqrt(jnp.where(safe, s2, 1.0))
    return jnp.where(safe, ax / s * jnp.arctan2(s, a @ b), ax)


class SensorPipeline:
    """Static structure of one sensor + controller configuration.

    ``noise``: a hover.sensors.SensorNoise (its ``scale`` is only the default of :meth:`params`).
    ``ctl``: a hover.reduced controller dict (kind state / sof / lqg), or None with ``state_mlp`` = a
    hover.policy MLP dict (the flight control's state MLP reading y[:12]), or neither (open loop).
    ``reference``: (gyro0, acc0, up0) on the release orbit; None = measure it on the device (one noise-free
    wingbeat, as hover.sensors.Sensors.reference does on CPU)."""

    def __init__(self, fl: F.Flight, noise, *, att_mode="filter", ctl=None, state_mlp=None, reference=None):
        if att_mode not in ("filter", "truth"):
            raise ValueError("att_mode must be 'filter' or 'truth'")
        S = fl.static
        self.noise, self.att_mode = noise, att_mode
        self.every = max(1, round(1.0 / (noise.imu_hz * S.dt)))
        self.dt_imu = self.every * S.dt
        self.lag = int(round(noise.pos_latency_ms * 1e-3 / S.dt))
        self.kind = None if ctl is None else str(ctl["kind"])
        self.obs = None if ctl is None else np.asarray(ctl["obs"], int)
        self.nx = None if ctl is None or "xs" not in ctl else len(ctl["xs"])
        self.dr = ctl is not None and "dr" in ctl
        self.ctl = ctl
        self.u_max = 0.15 if ctl is None else float(ctl["u_max"])
        mlp = ctl.get("mlp") if ctl is not None else state_mlp
        self.mlp = None if mlp is None else ([(jnp.asarray(W, F.fdtype()), jnp.asarray(b, F.fdtype())) for W, b in mlp["layers"]],
                                             jnp.asarray(mlp["obs_scale"], F.fdtype()))
        self.state_mlp = ctl is None and state_mlp is not None
        self.nu = 8 * S.K
        self.ref = None
        self.q_cal = None
        if reference is None:
            reference = measure_reference(fl, self)
        self.set_reference(*reference)

    def set_reference(self, gyro0, acc0, up0):
        self.ref = (jnp.asarray(gyro0, F.fdtype()), jnp.asarray(acc0, F.fdtype()))
        if up0 is None:
            self.q_cal = None
        else:
            f0, u0 = np.asarray(acc0, float), np.asarray(up0, float)
            rv = _rot_between(jnp.asarray(f0 / np.linalg.norm(f0)), jnp.asarray(u0 / np.linalg.norm(u0)))
            self.q_cal = F.quat2mat(F.rotvec2quat(rv)).astype(F.fdtype())

    def params(self, scale=None) -> Params:
        return Params(jnp.asarray(self.noise.scale if scale is None else scale, F.fdtype()), ctl_arrays(self.ctl))

    # ------------------------------------------------------------------ sensor state (hover.sensors.Sensors.start)
    def init(self, S, ar, dx, key, prm: Params):
        nz, dt = self.noise, dx.qpos.dtype
        sc = prm.scale
        k = jax.random.split(key, 5)
        bg = sc * nz.gyro_bias * jax.random.normal(k[0], (3,), dt)
        ba = sc * nz.acc_bias * jax.random.normal(k[1], (3,), dt)
        bg_hat = (bg + sc * nz.gyro_nd / math.sqrt(nz.gyro_cal_s) * jax.random.normal(k[2], (3,), dt)
                  if nz.gyro_cal_s > 0 else jnp.zeros(3, dt))
        qa = S.qadr
        q_true = dx.qpos[qa + 3:qa + 7]
        up = F.quat2mat(q_true).T @ jnp.array([0.0, 0, 1.0], dt)
        up_b = (G * up + ba) / jnp.linalg.norm(G * up + ba)
        q_hat = F.qmul(q_true, F.rotvec2quat(_rot_between(up_b, up)))
        z3 = jnp.zeros(3, dt)
        st = {"key": k[3], "rkey": k[4], "k": jnp.int32(0), "beat": jnp.int32(0), "v_prev": dx.qvel[S.vadr:S.vadr + 3],
              "sum_f": z3, "sum_up": z3, "sum_up_true": z3, "n_acc": jnp.int32(0), "q_hat": q_hat,
              "bg": bg, "ba": ba, "bg_hat": bg_hat, "bg_cal": bg_hat, "last_f": z3, "last_up_true": z3,
              "prm": prm}
        if self.lag:
            st["buf"] = jnp.repeat(dx.subtree_com[0][None], self.lag, 0)
            st["w"] = jnp.int32(0)
        n = 0 if self.ctl is None else (self.ctl["Az"].shape[0] if "Az" in self.ctl else 0)
        st["ctl"] = {"z": jnp.zeros(n, dt), "first": jnp.asarray(True), "u_prev": jnp.zeros(self.nu, dt),
                     "p": jnp.zeros(len(self.ctl["dr"]) if self.dr else 0, dt),
                     "xh_prev": jnp.zeros(self.nx or 0, dt)}
        return st

    # ------------------------------------------------------------------ every physics step (Sensors.step)
    def on_step(self, st, dx):
        S_ = self._S
        nz = self.noise
        sc = st["prm"].scale
        if self.lag:
            st = st | {"buf": st["buf"].at[st["w"]].set(dx.subtree_com[0]), "w": (st["w"] + 1) % self.lag}
        k = st["k"] + 1
        sample = (k % self.every) == 0
        R = F.quat2mat(dx.qpos[S_.qadr + 3:S_.qadr + 7])
        v = dx.qvel[S_.vadr:S_.vadr + 3]
        a = (v - st["v_prev"]) / self.dt_imu
        n = jax.random.normal(jax.random.fold_in(st["key"], k), (6,), v.dtype)
        f = R.T @ (a + jnp.array([0.0, 0, G], v.dtype)) + st["ba"] + sc * nz.acc_nd * math.sqrt(nz.imu_hz) * n[3:]
        new = {"sum_f": st["sum_f"] + f, "sum_up_true": st["sum_up_true"] + R[2], "n_acc": st["n_acc"] + 1, "v_prev": v}
        if self.att_mode == "filter":
            wg = dx.qvel[S_.vadr + 3:S_.vadr + 6] + st["bg"] + sc * nz.gyro_nd * math.sqrt(nz.imu_hz) * n[:3]
            q_hat = _quat_integrate(st["q_hat"], wg - st["bg_hat"], self.dt_imu)
            new |= {"q_hat": q_hat, "sum_up": st["sum_up"] + F.quat2mat(q_hat)[2]}
        st = st | {kk: jnp.where(sample, vv, st[kk]) for kk, vv in new.items()}
        return st | {"k": k}

    # ------------------------------------------------------------------ at the strobe (Sensors.read + the controller)
    def read(self, S, ar, st, dx, s):
        # keeps the once-per-wingbeat filter update out of the flight's big fusions (a defensive guard next to
        # the real fix, the einsum quaternion products in hover.mjx_flight; costs nothing measurable)
        st, dx, s = jax.lax.optimization_barrier((st, dx, s))
        nz = self.noise
        sc = st["prm"].scale
        dt = s.dtype
        x0 = ar.x_orbit
        has = st["n_acc"] > 0
        nacc = jnp.maximum(st["n_acc"], 1).astype(dt)
        fbar = st["sum_f"] / nacc
        y_acc = jnp.where(has, fbar - self.ref[1], 0.0)
        q_hat, bg_hat = st["q_hat"], st["bg_hat"]
        if self.att_mode == "filter":
            fc = self.q_cal @ fbar if self.q_cal is not None else fbar
            e = jnp.cross(fc / jnp.maximum(jnp.linalg.norm(fc), 1e-30), st["sum_up"] / jnp.maximum(jnp.linalg.norm(st["sum_up"]), 1e-30))
            q_hat = jnp.where(has, _quat_integrate(q_hat, nz.att_kp * e, 1.0), q_hat)
            bg_hat = jnp.where(has, bg_hat - nz.att_ki / (nacc * self.dt_imu) * e, bg_hat)
        n = jax.random.normal(jax.random.fold_in(st["rkey"], st["beat"]), (9,), dt)
        pos = (st["buf"][st["w"]] - ar.com_ref - x0[0:3]) if self.lag else s[0:3]
        pos = pos + sc * nz.pos_scale * nz.pos_std * n[0:3]
        vel = s[6:9] + sc * nz.vel_scale * nz.vel_std * n[3:6]
        gyro = dx.qvel[S.vadr + 3:S.vadr + 6] + st["bg"] + sc * nz.gyro_nd * math.sqrt(nz.imu_hz) * n[6:9]
        if self.att_mode == "truth":
            att = s[3:6]
            w = dx.xmat[S.af] @ (gyro - bg_hat) - x0[9:12]
        else:
            att = _att_error(q_hat, ar.q_ref) - x0[3:6]
            w = F.quat2mat(q_hat) @ (gyro - bg_hat) - x0[9:12]
        y = jnp.concatenate([pos, att, vel, w, gyro - st["bg_cal"] - self.ref[0], y_acc])
        z3 = jnp.zeros(3, dt)
        st = st | {"q_hat": q_hat, "bg_hat": bg_hat, "sum_f": z3, "sum_up": z3, "sum_up_true": z3, "n_acc": jnp.int32(0),
                   "last_f": jnp.where(has, fbar, st["last_f"]),
                   "last_up_true": jnp.where(has, st["sum_up_true"] / nacc, st["last_up_true"]), "beat": st["beat"] + 1}
        return st, y

    def _act_mlp(self, v):
        layers, scale = self.mlp
        x = v * scale
        for W, b in layers[:-1]:
            x = jnp.tanh(W @ x + b)
        W, b = layers[-1]
        return self.u_max * jnp.tanh(W @ x + b)

    def control(self, st, y):
        """hover.reduced.LinearController.__call__ (or the state MLP on y[:12]); returns (st, u)."""
        if self.ctl is None:
            if self.state_mlp:
                return st, self._act_mlp(y[:12])
            return st, jnp.zeros(self.nu, y.dtype)
        c, cs = st["prm"].ctl, st["ctl"]
        ys = y[self.obs]
        clip = lambda u: jnp.clip(u, -self.u_max, self.u_max)
        if self.kind == "state":
            u = clip(-c["K"] @ ys)
        elif self.kind == "sof":
            yt = ys - c["J"] @ cs["u_prev"]
            u = self._act_mlp(yt) if self.mlp else clip(-c["Ky"] @ yt)
        else:
            yt = ys - c["J"] @ cs["u_prev"]
            zp = jnp.where(cs["first"], 0.0, c["Az"] @ cs["z"] + c["Bz"] @ cs["u_prev"] + c["cz"])
            z = zp + c["L"] @ (yt - c["Cz"] @ zp)
            xh = z[:self.nx]
            v, K = xh, c["K"]
            p = cs["p"]
            if self.dr:
                p = jnp.where(cs["first"], 0.0, p + c["A_dr"] @ cs["xh_prev"] + c["B_dr"] @ cs["u_prev"] + c["d_dr"])
                v, K = jnp.concatenate([xh, p]), jnp.concatenate([c["K"], c["K_dr"]], axis=1)
            u = self._act_mlp(v) if self.mlp else clip(-K @ v)
            cs = cs | {"z": z, "p": p, "xh_prev": xh}
        return st | {"ctl": cs | {"u_prev": u, "first": jnp.asarray(False)}}, u

    def act(self, S, ar, st, dx, s):
        st, y = self.read(S, ar, st, dx, s)
        st, u = self.control(st, y)
        return st, u, y

    def bind(self, S):
        self._S = S
        return self


def _att_error(q, q_ref):
    qe = F.qmul(q, q_ref * jnp.array([1.0, -1.0, -1.0, -1.0], q.dtype))
    return 2.0 * jnp.where(qe[0] >= 0, 1.0, -1.0) * qe[1:]


def fly(S, ar, sp: SensorPipeline, n_beats, kicks, key, prm: Params, pert=None):
    """hover.launch.fly(..., policy=LinearController, sensor=Sensors) on the device. Returns (S (n+1, 12) true
    strobed states, Y (n+1, 18) channel readings, U (n, 8K), alive (n+1,))."""
    sp.bind(S)
    c = F.carry0(ar)
    dx = F.forward(ar, c.dx)
    if pert is not None:
        dx = F.rotate(S, ar, dx, pert[3:6])
        dx = F.kick(S, ar, dx, pert)
    c = c._replace(dx=dx)
    st = sp.init(S, ar, dx, key, prm)
    c_rest = c

    def one(state, kick_n):
        c, alive, st = state
        dx = F.kick(S, ar, c.dx, kick_n)
        s = F.strobe(S, ar, dx)
        alive = alive & jnp.all(jnp.isfinite(s)) & (jnp.linalg.norm(s[3:5]) <= F.LOST_TILT)
        st, u, y = sp.act(S, ar, st, dx, s)
        u = jnp.where(alive, u, 0.0)
        c, st = F.beat(S, ar, F.park(alive, c._replace(dx=dx), c_rest), ar.theta + jnp.concatenate([jnp.zeros(1, u.dtype), u]), sp.on_step, st)
        return (c, alive, st), (s, y, u, alive)

    (c, alive, st), (Ss, Ys, Us, As) = jax.lax.scan(one, (c, jnp.asarray(True), st), kicks)
    dx = F.forward(ar, c.dx)
    s = F.strobe(S, ar, dx)
    st, y = sp.read(S, ar, st, dx, s)
    last = alive & jnp.all(jnp.isfinite(s))
    return (jnp.concatenate([Ss, s[None]]), jnp.concatenate([Ys, y[None]]), Us, jnp.concatenate([As, last[None]]))


def batch_sensor_flyer(fl: F.Flight, sp: SensorPipeline, n_beats: int):
    """jit(vmap) of :func:`fly` over (kicks (B, n, 12), keys (B, 2), scales (B,), ctl arrays (B, ...)).
    Call as ``f(fl.arrays, kicks, keys, scales, ctl)``."""
    S = fl.static

    def one(ar, kicks, key, scale, ctl):
        return fly(S, ar, sp, n_beats, kicks, key, Params(scale, ctl))
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0)))


def measure_reference(fl: F.Flight, sp: SensorPipeline):
    """(gyro0, acc0, up0) on the release orbit: hover.sensors.Sensors.reference on the device (one noise-free
    wingbeat, truth attitude, no controller)."""
    from dataclasses import replace
    S, ar = fl.static, fl.arrays
    probe = SensorPipeline.__new__(SensorPipeline)
    probe.__dict__.update(sp.__dict__)
    probe.noise = replace(sp.noise, scale=0.0)
    probe.att_mode, probe.ctl, probe.kind, probe.mlp, probe.state_mlp, probe.dr = "truth", None, None, None, False, False
    probe.ref, probe.q_cal = (jnp.zeros(3, F.fdtype()), jnp.zeros(3, F.fdtype())), None

    def run(ar):
        probe.bind(S)
        c = F.carry0(ar)
        dx = F.forward(ar, c.dx)
        st = probe.init(S, ar, dx, jax.random.PRNGKey(0), probe.params(0.0))
        c, st = F.beat(S, ar, c._replace(dx=dx), ar.theta, probe.on_step, st)
        dx = F.forward(ar, c.dx)
        st, _ = probe.read(S, ar, st, dx, F.strobe(S, ar, dx))
        return st["last_f"], st["last_up_true"]
    f0, up0 = jax.jit(run)(ar)
    gyro0 = np.asarray(ar.dx0.qvel[S.vadr + 3:S.vadr + 6])
    return gyro0, np.asarray(f0), np.asarray(up0)


def linearize_obs(fl: F.Flight, chunk=None):
    """hover.lqr.linearize with its sensor model, on the device: (A, B, drift, obs) where obs = hover.lqr's
    observation_model (C_now, C_prev, J, y_drift) from noise-free truth-attitude channels read right after each
    perturbation and one wingbeat later (hover.lqr.one_beat_obs)."""
    from hover import lqr as Lq
    from hover import sensors as Sn
    S = fl.static
    probe = SensorPipeline(fl, Sn.SensorNoise.zero(), att_mode="truth", reference=(np.zeros(3), np.zeros(3), None))

    def one(ar, pert, du):
        probe.bind(S)
        c = F.carry0(ar)
        dx = F.forward(ar, c.dx)
        dx = F.rotate(S, ar, dx, pert[3:6])
        dx = F.kick(S, ar, dx, pert)
        st = probe.init(S, ar, dx, jax.random.PRNGKey(0), probe.params(0.0))
        s0 = F.strobe(S, ar, dx)
        st, y0 = probe.read(S, ar, st, dx, s0)
        c, st = F.beat(S, ar, c._replace(dx=dx), ar.theta + jnp.concatenate([jnp.zeros(1, du.dtype), du]), probe.on_step, st)
        dx = F.forward(ar, c.dx)
        s1 = F.strobe(S, ar, dx)
        st, y1 = probe.read(S, ar, st, dx, s1)
        return s1, y0, y1, s0
    perts, dus = F.lqr_tasks(S.K)
    f = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
    s1, y0, y1, s0 = F.run_batched(f, fl.arrays, perts, dus, chunk=chunk)
    A, B, drift = F.ab_from_tasks(s1, S.K)
    res = [(s1[i].astype(float), y0[i].astype(float), y1[i].astype(float), s0[i].astype(float)) for i in range(len(s1))]
    return A, B, drift, Lq.observation_model(res, F.SX, np.full(8 * S.K, F.SU))
