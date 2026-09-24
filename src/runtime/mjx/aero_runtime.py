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

The host-side wing geometry (``length``/``root``/``chords``/``x_rot``)
and the site/body indices are resolved once on a CPU model in :func:`build_aero_config`.

Verified conventions (machine precision vs CPU MuJoCo):
``omega_global = jacr.T @ qvel``, ``vel_global = jacp.T @ qvel``,
``qfrc += jacp @ force + jacr @ torque``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from morphology.wing_geometry import load_wing_geometry
from runtime import aero as cpu_aero
from runtime.aero import RHO_AIR, AeroParams
from runtime.mjx.aero import aero_wing


class AeroConfig(NamedTuple):
    """Static aero wiring resolved from a CPU model + wing geometry."""

    root_site_id: jax.Array  # (n_wings,) int
    blade_site_ids: jax.Array  # (n_wings, num_blade) int
    wing_body_id: jax.Array  # (n_wings,) int
    chords: jax.Array  # (num_blade,)
    x_rot: jax.Array  # (num_blade,) hinge behind the leading edge, in chords
    length: float  # static — wing span Rw
    root: float  # static — hinge to wing root along the span
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


def build_aero_config(model: mujoco.MjModel, morph, *, substep: int = 1) -> AeroConfig:
    """Resolve aero sites, body ids, and ``morph``'s wing geometry into a JAX-ready config."""
    geom = load_wing_geometry(morph)
    num_blade = int(len(geom.chords))

    def site_id(name: str) -> int:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Required aerodynamic site not found: {name}")
        return sid

    root_ids: list[int] = []
    blade_ids: list[list[int]] = []
    body_ids: list[int] = []
    sides = ("L", "R")
    for side in sides:
        root = site_id(f"{side}_wing_Root_Site")
        blades = [site_id(f"{side}_wing_aero_blade_{b + 1:02d}") for b in range(num_blade)]
        root_ids.append(root)
        blade_ids.append(blades)
        body_ids.append(int(model.site_bodyid[root]))

    return AeroConfig(
        root_site_id=jnp.asarray(root_ids, dtype=jnp.int32),
        blade_site_ids=jnp.asarray(blade_ids, dtype=jnp.int32),
        wing_body_id=jnp.asarray(body_ids, dtype=jnp.int32),
        chords=jnp.asarray(geom.chords),
        x_rot=jnp.asarray(geom.x_rot),
        length=float(geom.length),
        root=float(geom.root_offset),
        num_blade=num_blade,
        rho=RHO_AIR,
        dt=float(model.opt.timestep) * float(substep),
        params=AeroParams(),
        sides=sides,
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
    # The added-mass caps are the CPU aero's, read at trace time (set_unsteady_aero).
    omega_cap = cpu_aero.OMEGA_DOT_CAP
    omega_dot = jnp.clip((omega_local - prev_omega) / cfg.dt, -omega_cap, omega_cap)
    dv_cap = cpu_aero.LIN_ACC_CAP * cfg.dt
    prev_vel = vel_local - jnp.clip(vel_local - prev_vel, -dv_cap, dv_cap)

    forces_local, torques_local, alpha, speed = aero_wing(
        omega_local,
        jnp.zeros(3),
        vel_local,
        omega_dot,
        prev_vel,
        side=side,
        chords=cfg.chords,
        x_rot=cfg.x_rot,
        params=cfg.params,
        length=cfg.length,
        root=cfg.root,
        num_blade=cfg.num_blade,
        rho=cfg.rho,
        dt=cfg.dt,
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
    """Aero ``(qfrc, net_force_world, new_state, alpha_max, speed_mean)``.

    ``qfrc`` is shaped like ``qvel`` (what gets added to ``qfrc_applied``);
    ``net_force_world`` is the (3,) world-frame aerodynamic force summed over both
    wings — its z-component is the instantaneous aerodynamic **lift**. ``alpha_max``
    is the max folded AoA and ``speed_mean`` the mean relative strip speed.

    Like the CPU runtime it first brings the kinematics up to ``d.qpos`` (after
    ``mjx.step`` they still hold the pre-step pose); the refreshed data stays local.
    """
    d = mjx.com_pos(m, mjx.kinematics(m, d))
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
