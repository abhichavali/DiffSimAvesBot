"""Apply the JAX blade-element aero (``aero.py``) inside an MJX rollout.

JAX/MJX port of ``runtime.aero.AeroBladeRuntime``. For each wing it:

1. reads the root-site spatial velocity from the site Jacobian
   (``mjx.jac``) — the MJX equivalent of ``mj_objectVelocity`` — and rotates
   it into the root-site frame;
2. finite-differences the angular velocity against the previous step, carried in
   :class:`AeroState` (no Python-side mutable state, so the rollout stays pure);
3. evaluates :func:`runtime.mjx.aero.aero_wing` to get per-strip force/torque in
   the root frame, rotates them to world, and applies each strip's wrench at its
   blade site — the ``mj_applyFT`` equivalent ``jacp @ f + jacr @ t`` — summing
   into a ``qfrc_applied`` contribution.

The host-side wing geometry (``length``/``chords``/``span_cutoffs``) and the
site/body indices are resolved once on a CPU model in :func:`build_aero_config`.

Verified conventions (machine precision vs CPU MuJoCo):
``omega_global = jacr.T @ qvel``, ``vel_global = jacp.T @ qvel``,
``qfrc += jacp @ force + jacr @ torque``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from runtime.mjx.aero import RHO_AIR, AeroParams, aero_wing, aero_wing_diagnostics
from morphology.wing_geometry import load_default_wing_geometry

# Match the CPU runtime: quasi-steady by default. set_unsteady_aero(True) opts
# into the explicit finite-difference added-mass hack for diagnostics.
OMEGA_DOT_CAP = 0.0
LIN_ACC_CAP = 0.0

_DEFAULT_SIDES = ("L", "R")


class AeroConfig(NamedTuple):
    """Static aero wiring resolved from a CPU model + wing geometry."""

    root_site_id: jax.Array  # (n_wings,) int
    blade_site_ids: jax.Array  # (n_wings, num_blade) int
    wing_body_id: jax.Array  # (n_wings,) int
    chords: jax.Array  # (num_blade,)
    span_cutoffs: jax.Array  # (num_blade,)
    length: float  # static — wing span Rw
    num_blade: int  # static
    rho: float
    dt: float  # finite-difference window (timestep * substep)
    params: AeroParams
    sides: tuple[str, ...]  # static — per-wing mirror sign selector


class AeroState(NamedTuple):
    """Carry for the added-mass finite difference (one row per wing)."""

    prev_omega_local: jax.Array  # (n_wings, 3)
    prev_velocity_local: jax.Array  # (n_wings, 3)
    initialized: jax.Array  # scalar in {0.0, 1.0}


def build_aero_config(
    model: mujoco.MjModel,
    *,
    params: AeroParams = AeroParams(),
    num_blades: int | None = None,
    rho: float = RHO_AIR,
    dt: float | None = None,
    substep: int = 1,
    sides: Sequence[str] = _DEFAULT_SIDES,
) -> AeroConfig:
    """Resolve aero sites, body ids, and wing geometry into a JAX-ready config."""
    geom = load_default_wing_geometry(num_blades=num_blades)
    num_blade = int(len(geom.chords))

    def site_id(name: str) -> int:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Required aerodynamic site not found: {name}")
        return sid

    root_ids: list[int] = []
    blade_ids: list[list[int]] = []
    body_ids: list[int] = []
    for side in sides:
        root = site_id(f"{side}_wing_Root_Site")
        blades = [site_id(f"{side}_wing_aero_blade_{b + 1:02d}") for b in range(num_blade)]
        root_ids.append(root)
        blade_ids.append(blades)
        body_ids.append(int(model.site_bodyid[root]))

    timestep = float(model.opt.timestep) if dt is None else float(dt)
    return AeroConfig(
        root_site_id=jnp.asarray(root_ids, dtype=jnp.int32),
        blade_site_ids=jnp.asarray(blade_ids, dtype=jnp.int32),
        wing_body_id=jnp.asarray(body_ids, dtype=jnp.int32),
        chords=jnp.asarray(geom.chords),
        span_cutoffs=jnp.asarray(geom.span_cutoffs),
        length=float(geom.length),
        num_blade=num_blade,
        rho=float(rho),
        dt=timestep * float(substep),
        params=params,
        sides=tuple(sides),
    )


def init_aero_state(cfg: AeroConfig) -> AeroState:
    n = len(cfg.sides)
    return AeroState(
        prev_omega_local=jnp.zeros((n, 3)),
        prev_velocity_local=jnp.zeros((n, 3)),
        initialized=jnp.zeros(()),
    )


def _wing_qfrc(m, d, cfg: AeroConfig, wing_idx: int, side: str, state: AeroState):
    """qfrc contribution for one wing + its (omega_local, vel_local) for the carry."""
    sid = cfg.root_site_id[wing_idx]
    bid = cfg.wing_body_id[wing_idx]
    rot = d.site_xmat[sid].reshape(3, 3)  # root-site world rotation
    point = d.site_xpos[sid]

    jacp, jacr = mjx.jac(m, d, point, bid)
    omega_global = jacr.T @ d.qvel
    vel_global = jacp.T @ d.qvel
    omega_local = rot.T @ omega_global
    vel_local = rot.T @ vel_global

    # First step: no history -> zero angular accel and vel_h_prev == vel_local.
    prev_omega = jnp.where(
        state.initialized > 0.0, state.prev_omega_local[wing_idx], omega_local
    )
    prev_vel = jnp.where(
        state.initialized > 0.0, state.prev_velocity_local[wing_idx], vel_local
    )
    omega_dot = jnp.clip((omega_local - prev_omega) / cfg.dt, -OMEGA_DOT_CAP, OMEGA_DOT_CAP)
    dv_cap = LIN_ACC_CAP * cfg.dt
    prev_vel = vel_local - jnp.clip(vel_local - prev_vel, -dv_cap, dv_cap)

    forces_local, torques_local = aero_wing(
        omega_local,
        jnp.zeros(3),
        vel_local,
        omega_dot,
        prev_vel,
        side=side,
        chords=cfg.chords,
        span_cutoffs=cfg.span_cutoffs,
        params=cfg.params,
        length=cfg.length,
        num_blade=cfg.num_blade,
        rho=cfg.rho,
        dt=cfg.dt,
    )
    alpha, speed = aero_wing_diagnostics(
        omega_local, jnp.zeros(3), vel_local,
        chords=cfg.chords, length=cfg.length, num_blade=cfg.num_blade,
    )

    # Rotate strip wrenches root-frame -> world: row i is rot @ f_i  ==  f @ rot.T.
    forces_world = forces_local @ rot.T
    torques_world = torques_local @ rot.T

    def blade_qfrc(blade_sid, f_world, t_world):
        bp, br = mjx.jac(m, d, d.site_xpos[blade_sid], bid)
        return bp @ f_world + br @ t_world

    qfrc = jax.vmap(blade_qfrc)(
        cfg.blade_site_ids[wing_idx], forces_world, torques_world
    ).sum(axis=0)
    net_force_world = forces_world.sum(axis=0)  # (3,) — this wing's net aero force
    return qfrc, net_force_world, omega_local, vel_local, jnp.max(alpha), jnp.mean(speed)


def aero_wrench(m, d, cfg: AeroConfig, state: AeroState):
    """Aero generalized force, net world force, and updated carry.

    ``qfrc`` is shaped like ``qvel`` (what gets added to ``qfrc_applied``);
    ``net_force_world`` is the (3,) world-frame aerodynamic force summed over both
    wings — its z-component is the instantaneous aerodynamic **lift**.
    """
    qfrc, net_force_world, new_state, _alpha, _speed = aero_wrench_with_diagnostics(m, d, cfg, state)
    return qfrc, net_force_world, new_state


def aero_wrench_with_diagnostics(m, d, cfg: AeroConfig, state: AeroState):
    """Aero wrench plus max folded AoA and mean relative strip speed."""
    qfrc = jnp.zeros_like(d.qvel)
    net_force_world = jnp.zeros(3)
    omegas: list[jax.Array] = []
    vels: list[jax.Array] = []
    alphas: list[jax.Array] = []
    speeds: list[jax.Array] = []
    for wing_idx, side in enumerate(cfg.sides):
        q, fworld, omega_local, vel_local, alpha, speed = _wing_qfrc(
            m, d, cfg, wing_idx, side, state)
        qfrc = qfrc + q
        net_force_world = net_force_world + fworld
        omegas.append(omega_local)
        vels.append(vel_local)
        alphas.append(alpha)
        speeds.append(speed)
    new_state = AeroState(
        prev_omega_local=jnp.stack(omegas),
        prev_velocity_local=jnp.stack(vels),
        initialized=jnp.ones(()),
    )
    return qfrc, net_force_world, new_state, jnp.max(jnp.stack(alphas)), jnp.mean(jnp.stack(speeds))


def aero_qfrc(m, d, cfg: AeroConfig, state: AeroState):
    """Total aero generalized force (shaped like ``qvel``) and the updated carry."""
    qfrc, _net_force_world, new_state = aero_wrench(m, d, cfg, state)
    return qfrc, new_state

