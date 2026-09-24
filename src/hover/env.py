"""MJX free flight from a released state, with a per-wingbeat MLP closed loop (GPU, differentiable).

The CPU side (hover.launch) settles the bird on its open-loop trim in the launch cradle and releases it;
this module puts that released state on device and flies it with the SAME phase-locked per-crank Fourier
drive. Once per wingbeat the MLP reads the strobed deviation state (hover.launch.strobe, relative to the
orbit's own release state) and outputs bounded corrections to the 24 harmonic depths (the crank speed,
hence the wingbeat length, stays fixed so every strobe lands at the same drive phase). Nothing but physics
acts on the bird after release.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from hover import launch as L
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.sim import rest_state

N_OBS = 12
U_MAX = 0.15
# obs scaling to O(1): pos/5 cm, att/0.1 rad, vel/0.2 m/s, rates/1 rad/s
OBS_SCALE = np.array([20.0] * 3 + [10.0] * 3 + [5.0] * 3 + [1.0] * 3, np.float32)


class Env(NamedTuple):
    mx: mjx.Model
    dx0: mjx.Data
    pitch_cfg: wing_pitch.WingPitchConfig
    aero_cfg: aero_runtime.AeroConfig
    crank_ids: jax.Array
    kick_map: jax.Array      # (nv, 6): qvel change of a rigid [dv, dw] kick of the whole bird
    theta: jax.Array         # open-loop theta (1 + 8K)
    phi0: float
    L_des0: jax.Array
    com_ref: jax.Array
    q_ref: jax.Array
    x_orbit: jax.Array
    a0: float
    w_flap: float
    dt: float
    n_step: int              # physics steps per wingbeat (integer by construction of dt)
    substep: int
    qadr: int
    vadr: int
    af: int
    K: int


def make(bird: L.Bird, rel: L.Released, theta, K, *, aero_substep=4, solver_iterations=1) -> Env:
    """Device-side env from a CPU-released state. ``solver_iterations=1`` lets reverse mode trace mjx.step."""
    m = bird.model
    m.opt.iterations = solver_iterations
    m.opt.gravity[:] = 0
    rest = rest_state(m, bird.morph)               # pitch dead-zone centers, as the CPU sim calibrated them
    m.opt.gravity[:] = (0, 0, -9.81)
    d = rel.sim.data
    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    km = np.zeros((m.nv, 6))
    for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        v, b = m.jnt_dofadr[j], m.jnt_bodyid[j]
        R = d.xmat[b].reshape(3, 3); r = d.xpos[b] - com
        km[v:v + 3, 0:3] = np.eye(3)
        km[v:v + 3, 3:6] = -np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])   # w x r
        km[v + 3:v + 6, 3:6] = R.T
    g = float(theta[0])
    return Env(
        mx=mjx.put_model(m), dx0=mjx.put_data(m, d),
        pitch_cfg=wing_pitch.build_wing_pitch_config(m, bird.morph, rest),
        aero_cfg=aero_runtime.build_aero_config(m, bird.morph, substep=aero_substep),
        crank_ids=jnp.asarray(rel.sim.crank_ids, jnp.int32), kick_map=jnp.asarray(km, jnp.float32),
        theta=jnp.asarray(theta, jnp.float32), phi0=float(rel.drive.phi), L_des0=jnp.asarray(rel.drive.L_des, jnp.float32),
        com_ref=jnp.asarray(rel.com_ref, jnp.float32), q_ref=jnp.asarray(rel.q_ref, jnp.float32),
        x_orbit=jnp.asarray(rel.x_orbit, jnp.float32), a0=float(bird.a0), w_flap=float(bird.omega * math.exp(g)),
        dt=float(m.opt.timestep), n_step=round(2 * math.pi / (bird.omega * math.exp(g)) / m.opt.timestep),
        substep=aero_substep, qadr=bird.qadr, vadr=bird.vadr, af=bird.af, K=K)


def _qmul(p, q):
    return jnp.stack([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3], p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                      p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1], p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])


def observe(env: Env, dx):
    """Strobed deviation state (as hover.launch.strobe - x_orbit), from fresh kinematics."""
    d = mjx.com_pos(env.mx, mjx.kinematics(env.mx, dx))
    q, v, af = env.qadr, env.vadr, env.af
    com = d.subtree_com[0]
    w_world = d.xmat[af] @ dx.qvel[v + 3:v + 6]
    qe = _qmul(d.xquat[af], env.q_ref * jnp.array([1.0, -1.0, -1.0, -1.0], dx.qpos.dtype))
    att = 2.0 * jnp.where(qe[0] >= 0, 1.0, -1.0) * qe[1:]
    vcom = dx.qvel[v:v + 3] + jnp.cross(w_world, com - dx.qpos[q:q + 3])
    return jnp.concatenate([com - env.com_ref, att, vcom, w_world]) - env.x_orbit


def mlp_init(key, sizes, out_scale=0.0):
    params = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        scale = out_scale if i == len(sizes) - 2 else 1.0
        params.append((scale * jax.random.normal(k, (n_out, n_in)) / math.sqrt(n_in), jnp.zeros(n_out)))
    return params


def policy(params, s):
    """Strobed state -> bounded corrections to the 24 harmonic depths (crank speed untouched)."""
    x = s * OBS_SCALE
    for w, b in params[:-1]:
        x = jnp.tanh(w @ x + b)
    w, b = params[-1]
    return U_MAX * jnp.tanh(w @ x + b)


def fly(env: Env, params, kick, n_beats: int, closed_loop: bool = True):
    """Release (+ rigid kick [dv, dw]) and fly ``n_beats`` wingbeats. Returns (states (n_beats+1, 12), u (n_beats, 24))."""
    K = env.K
    ks = jnp.arange(1, K + 1, dtype=jnp.float32)
    dx0 = env.dx0.replace(qvel=env.dx0.qvel + env.kick_map @ kick)
    st0 = aero_runtime.init_aero_state(env.aero_cfg)
    carry0 = (dx0, st0, jnp.zeros_like(dx0.qvel), jnp.asarray(env.phi0, jnp.float32), env.L_des0, jnp.int32(0))

    def beat(carry, _):
        dx = carry[0]
        s = observe(env, dx)
        u = policy(params, s) if closed_loop else jnp.zeros(8 * K)
        th = env.theta + jnp.concatenate([jnp.zeros(1), u])
        a = th[1:1 + 4 * K].reshape(4, K); b = th[1 + 4 * K:].reshape(4, K)
        speed = env.a0 * jnp.exp(th[0])

        def step(c, _):
            dx, st, cq, phi, L_des, i = c
            w = speed * (1.0 + jnp.sum(a * jnp.cos(ks * phi) + b * jnp.sin(ks * phi), axis=1))
            ctrl = jnp.zeros_like(dx.ctrl).at[env.crank_ids].set(w + L.K_PHI * (L_des - dx.actuator_length[env.crank_ids]))

            def aero(d):
                def recompute(_):
                    qf, _, ns, _, _ = aero_runtime.aero_wrench(env.mx, d, env.aero_cfg, st)
                    return qf, ns
                qf, ns = jax.lax.cond(i % env.substep == 0, recompute, lambda _: (cq, st), None)
                return qf, (qf, ns)

            dx, (qf, st) = rollout.physics_step(env.mx, dx, ctrl, env.pitch_cfg, aero)
            nxt = (dx, st, qf, phi + env.w_flap * env.dt, L_des + w * env.dt, i + 1)
            return rollout.match_dtypes(nxt, c), None

        carry, _ = jax.lax.scan(step, carry, None, length=env.n_step)
        return carry, (s, u)

    carry, (S, U) = jax.lax.scan(jax.checkpoint(beat), carry0, None, length=n_beats)
    return jnp.concatenate([S, observe(env, carry[0])[None]]), U


# per-wingbeat cost weights on the strobed state: pos/2 cm, att/3 deg, vel/0.1 m/s, rates/0.5 rad/s
COST_W = jnp.array([1 / 0.02**2] * 3 + [1 / math.radians(3)**2] * 3 + [1 / 0.1**2] * 3 + [1 / 0.5**2] * 3)


def loss(params, env: Env, kicks, n_beats: int, w_u=0.2):
    S, U = jax.vmap(lambda k: fly(env, params, k, n_beats))(kicks)
    per_beat = jnp.sum(COST_W * S[:, 1:] ** 2, axis=-1)                 # (B, n_beats)
    return jnp.mean(jnp.log1p(per_beat)) + w_u * jnp.mean(jnp.sum((U / U_MAX) ** 2, -1)), (S, U)
