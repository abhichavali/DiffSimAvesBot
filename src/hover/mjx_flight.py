"""Free flight of the released bird on MJX (GPU): the CPU hover.launch.fly, step for step, batched with vmap.

The CPU side (hover.launch) settles the bird in the launch cradle and releases it, once per (design, dt: ~10 s;
hover.mjx_cradle runs the cradle on the device for the batched trim); :class:`Flight` puts that released state on the device - the MuJoCo data, the aero hold (held generalized
force, step counter, finite-difference history) and the drive (phase, encoder targets) - and flies it with the
same phase-locked per-crank Fourier drive, the same per-wingbeat strobe, pushes and loss rule as
``hover.launch.fly``. Everything is a pure function of arrays, so ``jax.vmap`` batches flights over pushes,
policies, perturbations and noise.

What makes it match the CPU reference (docs/findings/2026-09-26-mjx-free-flight-parity.md):
  * the full solver (the model's own ``opt.iterations`` / tolerance), never ``iterations=1``;
  * the aero hold continues the CPU's: the same step counter phase, the same held force at release;
  * ``mjx.forward`` at every wingbeat start, as the CPU's ``mj_forward`` before the strobe (it refreshes the
    actuator lengths the phase lock reads, which ``mj_step`` leaves one step stale);
  * float32 never accumulates a large angle: the drive phase restarts every wingbeat (phi0 + j w dt: integer
    steps per wingbeat), the crank angles are unwound by whole turns every wingbeat, and the phase lock carries
    the encoder ERROR e = L_des - L (updated with the exact float32 increment of the crank angle) instead of
    the target L_des, whose float32 running sum drifts ~1e-4 rad per wingbeat and walks the trim;
  * float32 is the default; ``jax_enable_x64`` (``enable_x64()`` before building) gives float64.

Policies (static ``kind``): "open", "linear" (u = clip(-K s)), "mlp" (hover.policy MLP); policies that read
the sensor channels and carry an observer are flown by hover.mjx_sensors.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from hover import launch as L
from runtime.mjx import aero_runtime, rollout, wing_pitch

LOST_TILT = math.radians(60)          # hover.launch.fly's loss rule
TWO_PI = 2 * math.pi


def enable_x64() -> None:
    """float64 on device (call before building a Flight). Consumer GPUs run FP64 at 1/64 rate."""
    jax.config.update("jax_enable_x64", True)


def fdtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class Arrays(NamedTuple):
    """Device-side state of a released flight (a pytree: pass it into jitted functions)."""

    mx: Any
    dx0: Any
    pitch: wing_pitch.WingPitchConfig
    aero_st0: aero_runtime.AeroState
    qfrc0: jax.Array          # the aero generalized force the CPU holds at release
    i0: jax.Array             # the CPU aero step counter at release
    theta: jax.Array          # open-loop theta (1 + 8K)
    e0: jax.Array             # encoder error at release: L_des - gear * crank angle
    com_ref: jax.Array
    q_ref: jax.Array
    x_orbit: jax.Array
    body_mass: jax.Array


@dataclass(frozen=True)
class Static:
    """Trace-time constants (shapes, indices, rates)."""

    K: int
    n_step: int               # physics steps per wingbeat
    substep: int
    dt: float
    w_flap: float
    speed: float              # a0 e^g (signed actuator space)
    k_phi: float
    phi0: float               # drive phase at release, mod 2 pi
    qadr: int
    vadr: int
    af: int
    crank_ids: tuple
    crank_qadr: tuple
    crank_gear: tuple
    free: tuple               # ((qposadr, dofadr, body), ...) of every free joint
    aero: Any = None          # aero_runtime.AeroConfig (holds strings: a trace-time constant, not an argument)


def _cast(tree):
    dt = fdtype()
    return jax.tree_util.tree_map(lambda x: x.astype(dt) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


class Flight:
    """A released CPU flight on the device. ``bird``/``rel`` from hover.launch (load_bird, release)."""

    def __init__(self, bird: L.Bird, rel: L.Released, theta, K, *, k_phi=L.K_PHI, solver_iterations=None,
                 ls_iterations=None, aero_substep=None, disable_contact=False):
        m = copy.deepcopy(bird.model)              # never mutate the CPU model
        if solver_iterations is not None:
            m.opt.iterations = int(solver_iterations)
        if ls_iterations is not None:
            m.opt.ls_iterations = int(ls_iterations)
        if disable_contact:
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        sim = rel.sim
        d = sim.data
        theta = np.asarray(theta, float)
        g = float(theta[0])
        w_flap = bird.omega * math.exp(g)
        n_step = round(TWO_PI / w_flap / m.opt.timestep)
        substep = int(aero_substep or sim.aero.substep)
        if substep != sim.aero.substep:
            raise ValueError("aero substep must match the released CPU sim's")
        crank_ids = tuple(int(i) for i in sim.crank_ids)
        crank_qadr = tuple(int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in crank_ids)
        free = tuple((int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), int(m.jnt_bodyid[j]))
                     for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.static = Static(K=int(K), n_step=n_step, substep=substep, dt=float(m.opt.timestep), w_flap=w_flap,
                             speed=float(bird.a0 * math.exp(g)), k_phi=float(k_phi),
                             phi0=float(math.fmod(rel.drive.phi, TWO_PI)), qadr=bird.qadr, vadr=bird.vadr, af=bird.af,
                             crank_ids=crank_ids, crank_qadr=crank_qadr,
                             crank_gear=tuple(float(m.actuator_gear[i, 0]) for i in crank_ids), free=free,
                             aero=_cast(aero_runtime.build_aero_config(m, bird.morph, substep=substep)))
        dt = fdtype()
        pl = sim.pitch
        pitch = wing_pitch.WingPitchConfig(
            qpos_adr=jnp.asarray([j.qpos_adr for j in pl.joints], jnp.int32),
            dof_adr=jnp.asarray([j.dof_adr for j in pl.joints], jnp.int32),
            center=jnp.asarray([j.center for j in pl.joints], dt),
            upstroke_pitch=jnp.asarray(pl.upstroke_pitch, dt), downstroke_pitch=jnp.asarray(pl.downstroke_pitch, dt),
            upstroke_stiffness=jnp.asarray(pl.upstroke_stiffness, dt),
            downstroke_stiffness=jnp.asarray(pl.downstroke_stiffness, dt), damping=jnp.asarray(pl.damping, dt))
        wings = sim.aero.wings
        init = all(w.previous_omega_local is not None for w in wings)
        st0 = aero_runtime.AeroState(
            prev_omega_local=jnp.asarray(np.stack([w.previous_omega_local if init else np.zeros(3) for w in wings]), dt),
            prev_velocity_local=jnp.asarray(np.stack([w.previous_velocity_local if init else np.zeros(3) for w in wings]), dt),
            initialized=jnp.asarray(1.0 if init else 0.0, dt))
        # unwind whole crank turns on the host too (exact in float64), so float32 starts small
        d = copy.deepcopy(d)
        L_des = np.asarray(rel.drive.L_des, float).copy()
        for c, (qa, gear) in enumerate(zip(crank_qadr, self.static.crank_gear)):
            turns = math.floor(d.qpos[qa] / TWO_PI)
            d.qpos[qa] -= TWO_PI * turns
            L_des[c] -= gear * TWO_PI * turns
            d.actuator_length[crank_ids[c]] -= gear * TWO_PI * turns
        self.model = m
        self.arrays = Arrays(
            mx=mjx.put_model(m), dx0=mjx.put_data(m, d), pitch=pitch, aero_st0=st0,
            qfrc0=jnp.asarray(sim.aero._qfrc, dt), i0=jnp.asarray(sim.aero._step_counter % substep, jnp.int32),
            theta=jnp.asarray(theta, dt),
            e0=jnp.asarray(L_des - np.array([g_ * d.qpos[qa] for qa, g_ in zip(crank_qadr, self.static.crank_gear)]), dt), com_ref=jnp.asarray(rel.com_ref, dt),
            q_ref=jnp.asarray(rel.q_ref, dt), x_orbit=jnp.asarray(rel.x_orbit, dt),
            body_mass=jnp.asarray(m.body_mass, dt))


# ---------------------------------------------------------------------------------------------- kinematics helpers

def _qmul_table():
    C = np.zeros((4, 4, 4))
    for i, terms in enumerate(([(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, -1)], [(0, 1, 1), (1, 0, 1), (2, 3, 1), (3, 2, -1)],
                               [(0, 2, 1), (1, 3, -1), (2, 0, 1), (3, 1, 1)], [(0, 3, 1), (1, 2, 1), (2, 1, -1), (3, 0, 1)])):
        for j, k, sg in terms:
            C[i, j, k] = sg
    T = np.zeros((3, 3, 4, 4))                  # R = I + T q q (mju_quat2Mat's 1 - 2(..) form)
    for (a, b), terms in {(0, 0): [(2, 2, -2), (3, 3, -2)], (0, 1): [(1, 2, 2), (3, 0, -2)], (0, 2): [(1, 3, 2), (2, 0, 2)],
                          (1, 0): [(1, 2, 2), (3, 0, 2)], (1, 1): [(1, 1, -2), (3, 3, -2)], (1, 2): [(2, 3, 2), (1, 0, -2)],
                          (2, 0): [(1, 3, 2), (2, 0, -2)], (2, 1): [(2, 3, 2), (1, 0, 2)], (2, 2): [(1, 1, -2), (2, 2, -2)]}.items():
        for k, l, v in terms:
            T[a, b, k, l] = v
    return C, T


_QMUL, _QMAT = _qmul_table()
# Quaternion products as contractions with constant tables, not stacks of scalar expressions: XLA's Triton
# block-level fusion of the stacked form (a 4-way concatenate fed by small matvecs) failed to compile on an RTX
# 5070 and silently produced NaNs on an RTX 5090 inside the vmapped sensor flights (opt/runs_gpu notes).


def qmul(p, q):
    """mju_mulQuat."""
    return jnp.einsum("ijk,j,k->i", jnp.asarray(_QMUL, p.dtype), p, q)


def quat2mat(q):
    """mju_quat2Mat (unit q)."""
    return jnp.eye(3, dtype=q.dtype) + jnp.einsum("abkl,k,l->ab", jnp.asarray(_QMAT, q.dtype), q, q)


def rotvec2quat(rv):
    """mju_axisAngle2Quat(rv / |rv|, |rv|), branch-free. Below 0.01 rad a Taylor series (exact to float64
    round-off there) replaces sin(a/2)/a: on the GPU the divided form returned NaN for ~1e-4 rad rotations
    inside large vmapped programs (the MC attitude filter), though not when run alone."""
    a2 = jnp.sum(rv * rv)
    small = a2 < 1e-4
    a = jnp.sqrt(jnp.where(small, 1.0, a2))
    s = jnp.where(small, 0.5 - a2 / 48 + a2 * a2 / 3840, jnp.sin(a / 2) / a)
    c = jnp.where(small, 1.0 - a2 / 8 + a2 * a2 / 384, jnp.cos(a / 2))
    return jnp.concatenate([c[None], s * rv])


def _com(ar: Arrays, dx):
    return (ar.body_mass[:, None] * dx.xipos).sum(0) / ar.body_mass.sum()


def forward(ar: Arrays, dx):
    """mj_forward (the CPU calls it before every strobe, after a kick and after a rotation)."""
    return mjx.forward(ar.mx, dx)


def kick(S: Static, ar: Arrays, dx, d12):
    """hover.launch.kick: rigid velocity kick [.., dv (6:9), dw (9:12)] of the whole bird, then forward."""
    com = _com(ar, dx)
    qvel = dx.qvel
    for qa, va, b in S.free:
        R = dx.xmat[b]
        qvel = qvel.at[va:va + 3].add(d12[6:9] + jnp.cross(d12[9:12], dx.xpos[b] - com))
        qvel = qvel.at[va + 3:va + 6].add(R.T @ d12[9:12])
    return forward(ar, dx.replace(qvel=qvel))


def rotate(S: Static, ar: Arrays, dx, rv):
    """hover.launch.rotate_bird by the rotation vector ``rv`` about the COM, then forward."""
    q = rotvec2quat(rv)
    R = quat2mat(q)
    com = _com(ar, dx)
    qpos = dx.qpos
    for qa, va, b in S.free:
        qpos = qpos.at[qa:qa + 3].set(com + R @ (qpos[qa:qa + 3] - com))
        qpos = qpos.at[qa + 3:qa + 7].set(qmul(q, qpos[qa + 3:qa + 7]))
    return forward(ar, dx.replace(qpos=qpos))


def strobe(S: Static, ar: Arrays, dx):
    """hover.launch.strobe(...) - x_orbit, from dx's (fresh) kinematics."""
    q, v, af = S.qadr, S.vadr, S.af
    com = dx.subtree_com[0]
    w_world = dx.xmat[af] @ dx.qvel[v + 3:v + 6]
    qe = qmul(dx.xquat[af], ar.q_ref * jnp.array([1.0, -1.0, -1.0, -1.0], dx.qpos.dtype))
    att = 2.0 * jnp.where(qe[0] >= 0, 1.0, -1.0) * qe[1:]
    vcom = dx.qvel[v:v + 3] + jnp.cross(w_world, com - dx.qpos[q:q + 3])
    return jnp.concatenate([com - ar.com_ref, att, vcom, w_world]) - ar.x_orbit


def unwind(S: Static, dx):
    """Subtract whole turns from the crank angles (and their stale actuator lengths); the encoder error is
    relative, so it is unchanged."""
    qpos, alen = dx.qpos, dx.actuator_length
    for qa, gear, ai in zip(S.crank_qadr, S.crank_gear, S.crank_ids):
        turns = jnp.floor(qpos[qa] / TWO_PI)
        qpos = qpos.at[qa].add(-TWO_PI * turns)
        alen = alen.at[ai].add(-gear * TWO_PI * turns)
    return dx.replace(qpos=qpos, actuator_length=alen)


# ---------------------------------------------------------------------------------------------- one wingbeat

class Carry(NamedTuple):
    dx: Any
    st: aero_runtime.AeroState
    qfrc: jax.Array
    i: jax.Array
    e: jax.Array              # encoder error L_des - gear * crank angle (small: no float32 accumulation)


def carry0(ar: Arrays) -> Carry:
    return Carry(ar.dx0, ar.aero_st0, ar.qfrc0, ar.i0, ar.e0)


def beat(S: Static, ar: Arrays, c: Carry, theta, on_step=None, sens=None):
    """Advance one wingbeat under drive ``theta`` (only the harmonic depths may differ from the open loop: the
    crank speed, hence the wingbeat length, is static). ``on_step(sens, dx) -> sens`` after every step."""
    K = S.K
    a = theta[1:1 + 4 * K].reshape(4, K)
    b = theta[1 + 4 * K:].reshape(4, K)
    dtype = c.dx.qpos.dtype
    ks = jnp.arange(1, K + 1, dtype=dtype)
    crank = jnp.asarray(S.crank_ids, jnp.int32)
    cq = jnp.asarray(S.crank_qadr, jnp.int32)
    gear = jnp.asarray(S.crank_gear, dtype)

    def step(state, j):
        c, sens = state
        dx = c.dx
        phi = S.phi0 + S.w_flap * S.dt * j
        w = S.speed * (1.0 + jnp.sum(a * jnp.cos(ks * phi) + b * jnp.sin(ks * phi), axis=1))
        q_c = dx.qpos[cq]
        # L_des - actuator_length, with the (one step stale) actuator length as the CPU reads it
        ctrl = jnp.zeros_like(dx.ctrl).at[crank].set(w + S.k_phi * (c.e + gear * q_c - dx.actuator_length[crank]))

        def aero(d):
            def recompute(_):
                qf, _, ns, _, _ = aero_runtime.aero_wrench(ar.mx, d, S.aero, c.st)
                return qf, ns
            qf, ns = jax.lax.cond(c.i % S.substep == 0, recompute, lambda _: (c.qfrc, c.st), None)
            return qf, (qf, ns)

        dx, (qf, st) = rollout.physics_step(ar.mx, dx, ctrl, ar.pitch, aero)
        nc = Carry(dx, st, qf, (c.i + 1) % S.substep, c.e + w * S.dt - gear * (dx.qpos[cq] - q_c))
        if on_step is not None:
            sens = on_step(sens, dx)
        return rollout.match_dtypes((nc, sens), (c, sens)), None

    (c, sens), _ = jax.lax.scan(step, (c, sens), jnp.arange(S.n_step, dtype=dtype))
    return c._replace(dx=unwind(S, c.dx)), sens


# ---------------------------------------------------------------------------------------------- policies

def linear_policy(Kg, u_max=0.15):
    return lambda s: jnp.clip(-(Kg @ s), -u_max, u_max)


def mlp_policy(mlp: dict):
    """hover.policy.FlightPolicy.correction (without the leading 0) as a pure function; ``mlp`` = load_mlp()."""
    layers = [(jnp.asarray(W, fdtype()), jnp.asarray(b, fdtype())) for W, b in mlp["layers"]]
    scale, u_max = jnp.asarray(mlp["obs_scale"], fdtype()), float(mlp["u_max"])

    def act(s):
        x = s * scale
        for W, b in layers[:-1]:
            x = jnp.tanh(W @ x + b)
        W, b = layers[-1]
        return u_max * jnp.tanh(W @ x + b)
    return act


# ---------------------------------------------------------------------------------------------- flights

def park(alive, c: Carry, c_rest: Carry) -> Carry:
    """A lost flight (alive False) continues from ``c_rest`` (the release state) instead of its tumbling state:
    its results are masked anyway, and a tumbling or non-finite state would make the solver's while loop run to
    its iteration cap for the whole vmapped batch."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(alive, x, y), c, c_rest)


def fly(S: Static, ar: Arrays, n_beats: int, kicks, *, policy=None, pert=None, theta=None):
    """hover.launch.fly on the device. ``kicks`` (n_beats, 12) are applied at the start of each wingbeat
    (zeros = none), ``pert`` (12) rotates (att entries 3:6) and kicks the bird before wingbeat 0 as hover.lqr
    does, ``policy(s) -> du (8K)`` is pure (hover.mjx_sensors flies policies that read sensors and carry an
    observer). Returns (S (n_beats+1, 12), U (n_beats, 8K), alive (n_beats+1,)): beats after the CPU's loss rule
    fires (non-finite or tilt > 60 deg) have alive=False (the CPU stops there; see :func:`lost_at`).
    """
    theta = ar.theta if theta is None else theta
    nu = 8 * S.K
    c = carry0(ar)
    dx = forward(ar, c.dx)
    if pert is not None:
        dx = rotate(S, ar, dx, pert[3:6])
        dx = kick(S, ar, dx, pert)
    c = c._replace(dx=dx)
    c_rest = c

    def one(state, kick_n):
        c, alive = state
        dx = kick(S, ar, c.dx, kick_n)              # zero kick = mj_forward only
        s = strobe(S, ar, dx)
        alive = alive & jnp.all(jnp.isfinite(s)) & (jnp.linalg.norm(s[3:5]) <= LOST_TILT)
        u = jnp.zeros(nu, dx.qpos.dtype) if policy is None else policy(s)
        u = jnp.where(alive, u, 0.0)
        c = park(alive, c._replace(dx=dx), c_rest)
        c, _ = beat(S, ar, c, theta + jnp.concatenate([jnp.zeros(1, u.dtype), u]))
        return (c, alive), (s, u, alive)

    (c, alive), (Ss, Us, As) = jax.lax.scan(one, (c, jnp.asarray(True)), kicks)
    dx = forward(ar, c.dx)
    s = strobe(S, ar, dx)
    last_alive = alive & jnp.all(jnp.isfinite(s))
    return jnp.concatenate([Ss, s[None]]), Us, jnp.concatenate([As, last_alive[None]])


def one_beat(S: Static, ar: Arrays, pert, du):
    """hover.lqr.one_beat: the strobed state one wingbeat after a rigid perturbation ``pert`` under input ``du``."""
    c = carry0(ar)
    dx = forward(ar, c.dx)
    dx = rotate(S, ar, dx, pert[3:6])
    dx = kick(S, ar, dx, pert)
    c, _ = beat(S, ar, c._replace(dx=dx), ar.theta + jnp.concatenate([jnp.zeros(1, du.dtype), du]))
    return strobe(S, ar, forward(ar, c.dx))


SX = np.array([0.01] * 3 + [0.02] * 3 + [0.1] * 3)      # hover.lqr.linearize's steps: att, vel, w
SU = 0.01                                                 # ... and harmonic depths


def lqr_tasks(K, sx=SX, su=SU):
    """hover.lqr.linearize's central-difference tasks, in its order: (perts (n, 12), dus (n, 8K))."""
    nu = 8 * K
    perts, dus = [np.zeros(12)], [np.zeros(nu)]
    for i, s in enumerate(sx):
        for sg in (1, -1):
            d = np.zeros(12); d[3 + i] = sg * s; perts.append(d); dus.append(np.zeros(nu))
    for j in range(nu):
        for sg in (1, -1):
            du = np.zeros(nu); du[j] = sg * su; perts.append(np.zeros(12)); dus.append(du)
    return np.array(perts), np.array(dus)


def ab_from_tasks(out, K, sx=SX, su=SU):
    """(A, B, drift) from the one-beat results of :func:`lqr_tasks` (hover.lqr.linearize's formulas)."""
    out = np.asarray(out, float)
    A = np.zeros((12, 12)); A[:3, :3] = np.eye(3)
    for i, s in enumerate(sx):
        A[:, 3 + i] = (out[1 + 2 * i] - out[2 + 2 * i]) / (2 * s)
    o = 1 + 2 * len(sx)
    B = np.stack([(out[o + 2 * j] - out[o + 1 + 2 * j]) / (2 * su) for j in range(8 * K)], 1)
    return A, B, out[0]


def linearize(fl: Flight, chunk=None):
    """Per-wingbeat (A, B, drift) by central differences, all one-beat rollouts in one vmapped batch."""
    perts, dus = lqr_tasks(fl.static.K)
    f = jax.jit(jax.vmap(lambda ar, p, du: (one_beat(fl.static, ar, p, du),), in_axes=(None, 0, 0)))
    (out,) = run_batched(f, fl.arrays, perts, dus, chunk=chunk)
    return ab_from_tasks(out, fl.static.K)


def linearize_jacfwd(fl: Flight):
    """(A, B, drift) of the one-beat map by forward-mode AD at the release state (tangents vmapped)."""
    S, K = fl.static, fl.static.K
    nu = 8 * K

    def g(z, ar):
        pert = jnp.concatenate([jnp.zeros(3, z.dtype), z[:9]])
        return one_beat(S, ar, pert, z[9:])
    z0 = jnp.zeros(9 + nu, fdtype())
    J = jax.jit(jax.jacfwd(g))(z0, fl.arrays)
    drift = jax.jit(g)(z0, fl.arrays)
    J = np.asarray(J, float)
    A = np.zeros((12, 12)); A[:3, :3] = np.eye(3); A[:, 3:] = J[:, :9]
    return A, J[:, 9:], np.asarray(drift, float)


def cpu_trace(S, A):
    """Post-process a device flight like the CPU list: states up to and including the first lost beat."""
    return np.asarray(S)[:int(lost_at(A)) + 1]


def lost_at(alive) -> np.ndarray:
    """Per flight: the index of the last strobe the CPU would return (= n_beats if it flew them all)."""
    alive = np.asarray(alive)
    return np.where(alive.all(-1), alive.shape[-1] - 1, np.argmin(alive, -1))


def batch_flyer(fl: Flight, n_beats: int, kind: str = "linear", mlp: dict | None = None):
    """jit(vmap) of :func:`fly` over (kicks (B, n_beats, 12), gains (B, 8K, 12), perts (B, 12)); ``kind`` =
    "open" | "linear" (u = clip(-K s), K per flight; K = 0 flies the open loop) | "mlp" (one shared MLP).
    Call as ``f(fl.arrays, kicks, gains, perts)``."""
    S = fl.static

    def one(ar, kicks, Kg, pert):
        pol = {"open": None, "linear": linear_policy(Kg), "mlp": mlp_policy(mlp) if mlp else None}[kind]
        return fly(S, ar, n_beats, kicks, policy=pol, pert=pert)
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))


def run_batched(f, ar, *batched, chunk=None):
    """Call ``f(ar, *batched)`` (a jit(vmap) over the leading axis) in chunks of ``chunk`` flights, padding the
    last chunk to the same width (one compile); returns host numpy arrays."""
    B = len(batched[0])
    chunk = min(chunk or B, B)
    outs = []
    for s in range(0, B, chunk):
        part = [np.asarray(x[s:s + chunk]) for x in batched]
        n = len(part[0])
        if n < chunk:
            part = [np.concatenate([p, np.repeat(p[-1:], chunk - n, 0)]) for p in part]
        res = f(ar, *[jnp.asarray(p, fdtype()) for p in part])
        outs.append([np.asarray(r)[:n] for r in res])
    return [np.concatenate(o) for o in zip(*outs)]
