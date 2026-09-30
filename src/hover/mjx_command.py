"""Command-following flights on the device (MJX, GPU): hover.command's servo flown batched with vmap.

The scan body is hover.mjx_flight.fly's (kick, strobe, loss rule, park, one wingbeat under theta + [0, u]); the
policy is hover.command's servo law with every run-specific quantity a traced per-flight array (per-beat x_ref,
u_ff, dither, kicks and the gain Ka), so the ILC / envelope pass, the evaluation and the baseline arm are all
one compiled program: B flights x N wingbeats, chained call after call through a batched carry (the flight
state, the loss flag and the integral).

The sensor variant (:class:`CmdSensorPipeline`) runs hover.mjx_sensors' IMU / mocap pipeline and a padded LQG
observer (all 18 channels, zero gains on the unread ones) so the mocap-position and IMU-only controllers share
one compiled program too.

    .venv/bin/python opt/hover_cmd.py learn --out opt/runs_cmd/<run>      # on the GPU box (gpu-box skill)
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from hover import mjx_flight as F
from hover import mjx_sensors as MS
from runtime.mjx import rollout
from hover.command import ALLOC_FREE, U_MAX, ZETA_MAX, fista_betas


def allocate(ud, H, u_max, iters):
    """hover.command.allocate on the device (``iters`` static)."""
    u = jnp.clip(ud, -u_max, u_max)

    def body(carry, b):
        u, y = carry
        un = jnp.clip(y - H @ (y - ud), -u_max, u_max)
        return (un, un + b * (un - u)), None
    (u, _), _ = jax.lax.scan(body, (u, u), jnp.asarray(fista_betas(iters), ud.dtype))
    return u


def law(s, zeta, x_ref, u_ff, dither, Ka, u_max=U_MAX, zeta_max=ZETA_MAX, cp=1.0, cv=0.0, T=0.0, H=None, iters=0):
    """hover.command.servo_law on the device: (u, zeta'). ``iters`` (static) > 0: the H-metric allocation. A
    13 / 14-wide ``x_ref`` (static shape) carries the integral handoff factor / gate (hover.command.integral_keep /
    integral_gate)."""
    e = s - x_ref[:12]
    keep = x_ref[12] if x_ref.shape[-1] > 12 else 1.0
    gate = x_ref[13] if x_ref.shape[-1] > 13 else 1.0
    raw = u_ff + dither - Ka[:, :12] @ e - Ka[:, 12:] @ zeta
    if iters:
        u = allocate(raw, H, u_max, iters)
        d, fb = u - raw, raw - u_ff - dither
        free = jnp.where(d @ H @ d <= ALLOC_FREE * (fb @ H @ fb), 1.0, 0.0).astype(s.dtype)
    else:
        u = jnp.clip(raw, -u_max, u_max)
        free = jnp.where(jnp.any(jnp.abs(raw) >= u_max), 0.0, 1.0).astype(s.dtype)
    dz = cp * e[:3] + cv * T * e[6:9]
    return u, jnp.clip(keep * zeta + gate * free * dz, -zeta_max, zeta_max)


def release_carry(fl: F.Flight, B: int):
    """(carry, alive, zeta) of B flights at release, and the unbatched release carry the lost flights park on.
    mjx_flight.fly runs mj_forward on carry0 first; here the first wingbeat's kick does (a zero kick = forward
    only), which saves a compile (tests/test_hover_command.py checks the two agree)."""
    c = F.carry0(fl.arrays)
    cb = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)), c)
    return (cb, jnp.ones(B, bool), jnp.zeros((B, 3), F.fdtype())), c


def fly_cmd(S, ar, c, alive, zeta, c_rest, kicks, x_ref, u_ff, dither, Ka, lost_tilt, u_max=U_MAX, zeta_max=ZETA_MAX,
            H=None, iters=0):
    """N wingbeats of the servo from a GIVEN carry (calls chain). kicks / x_ref / u_ff / dither are per beat
    (N, .); x_ref is 12 wide, or 13 / 14 with the integral handoff factor / gate. Returns (S (N, 12), U (N, nu),
    alive (N,), carry', alive', zeta', s_end (12), alive_end): s_end is the
    strobe after the last beat (the next call's first) with mjx_flight.fly's final loss rule. ``lost_tilt``
    (traced, per flight) is the loss rule's tilt [rad]: mjx_flight.LOST_TILT (60 deg) for evaluation; the ILC
    passes use a lower one, since a tumbling flight slows the whole vmapped batch (the solver's while loop).
    ``H`` / ``iters``: the allocation metric (shared) and its static iteration count (0 = clip)."""

    def one(state, xs):
        c, alive, zeta = state
        kick_n, xr, uf, dth = xs
        dx = F.kick(S, ar, c.dx, kick_n)
        s = F.strobe(S, ar, dx)
        alive = alive & jnp.all(jnp.isfinite(s)) & (jnp.linalg.norm(s[3:5]) <= lost_tilt)
        u, zn = law(s, zeta, xr, uf, dth, Ka, u_max, zeta_max, H=H, iters=iters)
        u = jnp.where(alive, u, 0.0)
        zeta = jnp.where(alive, zn, zeta)
        c = F.park(alive, c._replace(dx=dx), c_rest)
        c, _ = F.beat(S, ar, c, ar.theta + jnp.concatenate([jnp.zeros(1, u.dtype), u]))
        # x64 promotes a few int32 index fields of the MJX data: keep the carry type (and the chained calls' shape)
        return rollout.match_dtypes((c, alive, zeta), state), (s, u, alive)

    (c, alive, zeta), (Ss, Us, As) = jax.lax.scan(one, (c, alive, zeta), (kicks, x_ref, u_ff, dither))
    s_end = F.strobe(S, ar, F.forward(ar, c.dx))
    return Ss, Us, As, c, alive, zeta, s_end, alive & jnp.all(jnp.isfinite(s_end))


def batch_cmd_flyer(fl: F.Flight, N: int, u_max=U_MAX, zeta_max=ZETA_MAX, alloc_iters=0):
    """jit(vmap(fly_cmd)) over (carry, alive, zeta, kicks, x_ref, u_ff, dither, Ka, lost_tilt, H); ar and the park
    carry are shared. Call as ``f(ar, c, alive, zeta, c_rest, kicks, x_ref, u_ff, dither, Ka, lost_tilt, H)``
    (H = I allocates exactly as clip: the baseline arm)."""
    S = fl.static

    def one(ar, c, alive, zeta, c_rest, kicks, x_ref, u_ff, dither, Ka, lost_tilt, H):
        return fly_cmd(S, ar, c, alive, zeta, c_rest, kicks, x_ref, u_ff, dither, Ka, lost_tilt, u_max, zeta_max,
                       H, alloc_iters)
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0)), donate_argnums=(1,))


class Chain:
    """Chained N-beat calls of one compiled flight program; ``compile_s`` = the first lower + compile (or
    persistent-cache load) time, measured apart from the flights."""

    def __init__(self, fl: F.Flight, B: int, N: int, u_max=U_MAX, zeta_max=ZETA_MAX, H=None, alloc_iters=0):
        self.fl, self.B, self.N = fl, B, N
        self.H = np.eye(8 * fl.static.K) if H is None else np.asarray(H, float)
        self.f = batch_cmd_flyer(fl, N, u_max, zeta_max, alloc_iters if H is not None else 0)
        self.compiled = None
        self.compile_s = None

    def _compile(self, args):
        t = time.time()
        self.compiled = self.f.lower(*args).compile()
        self.compile_s = time.time() - t

    def run(self, state, c_rest, kicks, x_ref, u_ff, dither, Ka, n_calls, start=0, log=None, lost_tilt=None, H=None):
        """Fly n_calls x N beats from ``state`` = (carry, alive, zeta), per-beat arrays indexed from beat ``start``.
        ``H``: per-flight allocation metric (B, nu, nu) (default: the chain's H for every flight).
        Returns (S (B, n_calls N + 1, 12), U, alive (B, n_calls N + 1), state', per-call seconds)."""
        dt = F.fdtype()
        ar = self.fl.arrays
        Ka = jnp.asarray(Ka, dt)
        Hb = jnp.asarray(np.broadcast_to(self.H if H is None else H, (self.B,) + self.H.shape), dt)
        lt = jnp.asarray(np.broadcast_to(F.LOST_TILT if lost_tilt is None else lost_tilt, (self.B,)), dt)
        Ss, Us, As, secs = [], [], [], []
        c, alive, zeta = state
        s_end = a_end = None
        for j in range(n_calls):
            sl = slice(start + j * self.N, start + (j + 1) * self.N)
            args = (ar, c, alive, zeta, c_rest, jnp.asarray(kicks[:, sl], dt), jnp.asarray(x_ref[:, sl], dt),
                    jnp.asarray(u_ff[:, sl], dt), jnp.asarray(dither[:, sl], dt), Ka, lt, Hb)
            if self.compiled is None:
                self._compile(args)
                if log:
                    log(f"  flight program {self.B} x {self.N}: compile / cache load {self.compile_s:.1f} s")
            t = time.time()
            S_, U_, A_, c, alive, zeta, s_end, a_end = self.compiled(*args)
            Ss.append(np.asarray(S_)); Us.append(np.asarray(U_)); As.append(np.asarray(A_))
            secs.append(time.time() - t)
            if log:
                log(f"  call {j + 1}/{n_calls} (beats {sl.start}-{sl.stop}): {secs[-1]:.1f} s")
        S = np.concatenate(Ss + [np.asarray(s_end)[:, None]], 1)
        A = np.concatenate(As + [np.asarray(a_end)[:, None]], 1)
        return S, np.concatenate(Us, 1), A, (c, alive, zeta), secs


# ---------------------------------------------------------------------------------------------- sensors

def pad_observer(ctl: dict, nz: int, nu: int) -> dict:
    """A hover.reduced lqg dict as 18-channel arrays padded to nz observer states: J (18, nu), Az (nz, nz),
    Bz (nz, nu), cz (nz), Cz (18, nz), L (nz, 18), Sx (12, nz) (x_hat = Sx z, zeros on undesigned states)."""
    obs, xs = np.asarray(ctl["obs"], int), np.asarray(ctl["xs"], int)
    n0 = ctl["Az"].shape[0]
    if n0 > nz:
        raise ValueError("nz smaller than the observer")
    J = np.zeros((MS.N_CH, nu)); J[obs] = ctl["J"]
    Az = np.zeros((nz, nz)); Az[:n0, :n0] = ctl["Az"]
    Bz = np.zeros((nz, nu)); Bz[:n0] = ctl["Bz"]
    cz = np.zeros(nz); cz[:n0] = ctl["cz"]
    Cz = np.zeros((MS.N_CH, nz)); Cz[obs, :n0] = ctl["Cz"]
    L = np.zeros((nz, MS.N_CH)); L[:n0, obs] = ctl["L"]
    Sx = np.zeros((12, nz)); Sx[xs, np.arange(len(xs))] = 1.0
    return {"J": J, "Az": Az, "Bz": Bz, "cz": cz, "Cz": Cz, "L": L, "Sx": Sx}


class CmdSensorPipeline(MS.SensorPipeline):
    """mjx_sensors.SensorPipeline whose controller is the (padded) LQG observer + the servo on the estimate.
    Per-flight ctl dict: pad_observer arrays + Ka (nu, 15), x_ref (N, 12), u_ff (N, nu), cp, cv, T (scalars:
    cp = 1 integrates the estimated position error, cv = 1 the estimated velocity error x T; optional H (nu, nu):
    the allocation metric, used with ``alloc_iters`` > 0; optional cz_seq (N, nz): the per-beat observer offset,
    hover.command.trim_consistent_cz padded to nz, in place of the constant cz)."""

    def __init__(self, fl, noise, nz, *, att_mode="filter", u_max=U_MAX, zeta_max=ZETA_MAX, reference=None, alloc_iters=0):
        nu = 8 * fl.static.K
        tmpl = {"kind": "lqg", "obs": np.arange(MS.N_CH), "xs": np.arange(12), "u_max": u_max,
                "Az": np.zeros((nz, nz)), "K": np.zeros((nu, 12))}
        super().__init__(fl, noise, att_mode=att_mode, ctl=tmpl, reference=reference)
        self.nz, self.zeta_max, self.alloc_iters = nz, zeta_max, alloc_iters

    def init(self, S, ar, dx, key, prm):
        st = super().init(S, ar, dx, key, prm)
        dt = dx.qpos.dtype
        st["ctl"] = {"z": jnp.zeros(self.nz, dt), "first": jnp.asarray(True), "u_prev": jnp.zeros(self.nu, dt),
                     "k": jnp.int32(0), "zeta": jnp.zeros(3, dt), "xh": jnp.zeros(12, dt)}
        return st

    def control(self, st, y):
        c, cs = st["prm"].ctl, st["ctl"]
        yt = y - c["J"] @ cs["u_prev"]
        k = cs["k"]
        cz = c["cz_seq"][k] if "cz_seq" in c else c["cz"]
        zp = jnp.where(cs["first"], 0.0, c["Az"] @ cs["z"] + c["Bz"] @ cs["u_prev"] + cz)
        z = zp + c["L"] @ (yt - c["Cz"] @ zp)
        xh = c["Sx"] @ z
        u, zeta = law(xh, cs["zeta"], c["x_ref"][k], c["u_ff"][k], 0.0, c["Ka"], self.u_max, self.zeta_max,
                      c["cp"], c["cv"], c["T"], c.get("H"), self.alloc_iters if "H" in c else 0)
        return st | {"ctl": cs | {"z": z, "u_prev": u, "first": jnp.asarray(False), "k": k + 1, "zeta": zeta, "xh": xh}}, u


def fly_cmd_sensor(S, ar, sp: CmdSensorPipeline, n_beats, kicks, key, prm: MS.Params):
    """mjx_sensors.fly with the command servo; also returns the per-beat estimate x_hat. Returns (S (n+1, 12)
    true strobed states, Y (n+1, 18), Xh (n, 12), U (n, nu), alive (n+1,))."""
    sp.bind(S)
    c = F.carry0(ar)
    dx = F.forward(ar, c.dx)
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
        c, st = F.beat(S, ar, F.park(alive, c._replace(dx=dx), c_rest),
                       ar.theta + jnp.concatenate([jnp.zeros(1, u.dtype), u]), sp.on_step, st)
        return (c, alive, st), (s, y, st["ctl"]["xh"], u, alive)

    (c, alive, st), (Ss, Ys, Xs, Us, As) = jax.lax.scan(one, (c, jnp.asarray(True), st), kicks)
    dx = F.forward(ar, c.dx)
    s = F.strobe(S, ar, dx)
    st, y = sp.read(S, ar, st, dx, s)
    last = alive & jnp.all(jnp.isfinite(s))
    return (jnp.concatenate([Ss, s[None]]), jnp.concatenate([Ys, y[None]]), Xs, Us, jnp.concatenate([As, last[None]]))


def batch_cmd_sensor_flyer(fl: F.Flight, sp: CmdSensorPipeline, n_beats: int):
    """jit(vmap) of :func:`fly_cmd_sensor` over (kicks (B, n, 12), keys (B, 2), scales (B,), ctl dict (B, ...)).
    Call as ``f(fl.arrays, kicks, keys, scales, ctl)``."""
    S = fl.static

    def one(ar, kicks, key, scale, ctl):
        return fly_cmd_sensor(S, ar, sp, n_beats, kicks, key, MS.Params(scale, ctl))
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0)))
