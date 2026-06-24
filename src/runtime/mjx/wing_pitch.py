"""Dead-zone restorative spring for the wing pitch joints, in JAX.

JAX port of :class:`runtime.wing_pitch_limit.WingPitchLimiter`. The pitch hinges
are emitted passively free (zero stiffness/damping in MJCF); this applies, every
step, a constant velocity ``damping`` plus a restorative spring that engages only
once the pitch deviates more than ``free_angle`` from its leveled-neutral rest:

    torque = -damping * qvel
             - stiffness * max(|dev| - free_angle, 0) * sign(dev)

where ``dev = qpos - center``. It is branch-free (``clip``/``sign``), so it is
fully ``jit``/``vmap``-able and differentiable for use inside an ``mjx.step``
rollout. The CPU loop zeroes ``qfrc_applied`` each step then accumulates this and
the aero force into it (see ``viewer_runtime.step_and_sync``); :func:`apply`
mirrors that by *adding* to the current ``qfrc_applied``, so the rollout must
zero (or freshly build) ``qfrc_applied`` before composing force sources.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import mujoco

from runtime.runtime_config import load_runtime_config

_DEFAULT_JOINTS = ("L_wing_pitch", "R_wing_pitch")


class WingPitchConfig(NamedTuple):
    """Static-per-build configuration for the dead-zone pitch controller.

    A JAX pytree: pass it straight into a jitted step. ``qpos_adr``/``dof_adr``
    index the (possibly batched) state vectors; ``center`` is the leveled rest
    angle each joint's dead zone is measured from.
    """

    qpos_adr: jax.Array  # (n_joints,) int
    dof_adr: jax.Array  # (n_joints,) int
    center: jax.Array  # (n_joints,) float — leveled-neutral rest angle
    free_angle: jax.Array  # scalar float — half-width of the zero-torque zone
    stiffness: jax.Array  # scalar float — spring gain beyond the free zone
    damping: jax.Array  # scalar float — velocity damping applied everywhere
    # Up/down stroke asymmetry (default neutral → symmetric spring). The spring
    # is stiffer/softer and the free zone wider/narrower on the negative-dev
    # (one stroke half) side, identically on both wings — so it breaks the
    # up↓/down↑ force cancellation that zeroes net lift, WITHOUT breaking L/R
    # mirror symmetry. ``stiffness_asym``/``free_asym`` scale the neg-dev side.
    stiffness_asym: jax.Array = jnp.asarray(1.0, dtype=jnp.float32)
    free_asym: jax.Array = jnp.asarray(1.0, dtype=jnp.float32)


def build_wing_pitch_config(
    model: mujoco.MjModel,
    *,
    free_angle: float,
    stiffness: float,
    damping: float,
    centers: Sequence[float] | None = None,
    joint_names: Sequence[str] = _DEFAULT_JOINTS,
) -> WingPitchConfig:
    """Resolve joint addresses on a CPU model into a JAX-ready config.

    ``centers`` is the leveled rest angle per joint (from
    ``calibrate_wing_pitch_at_rest`` on CPU at build time). If omitted, the
    model's ``qpos0`` for each joint is used.
    """
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    resolved_centers: list[float] = []
    for idx, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        qpos_adr.append(qadr)
        dof_adr.append(int(model.jnt_dofadr[joint_id]))
        if centers is None:
            resolved_centers.append(float(model.qpos0[qadr]))
        else:
            resolved_centers.append(float(centers[idx]))

    return WingPitchConfig(
        qpos_adr=jnp.asarray(qpos_adr, dtype=jnp.int32),
        dof_adr=jnp.asarray(dof_adr, dtype=jnp.int32),
        center=jnp.asarray(resolved_centers, dtype=jnp.float32),
        free_angle=jnp.asarray(float(free_angle), dtype=jnp.float32),
        stiffness=jnp.asarray(float(stiffness), dtype=jnp.float32),
        damping=jnp.asarray(float(damping), dtype=jnp.float32),
    )


def config_from_runtime(
    model: mujoco.MjModel,
    *,
    centers: Sequence[float] | None = None,
    joint_names: Sequence[str] = _DEFAULT_JOINTS,
) -> WingPitchConfig:
    """Build a config using the design params from ``load_runtime_config``."""
    params = (load_runtime_config() or {}).get("wing_pitch_parameters", {})
    return build_wing_pitch_config(
        model,
        free_angle=float(params.get("free_angle", 0.0)),
        stiffness=float(params.get("limit_stiffness", 0.5)),
        damping=float(params.get("limit_damping", 0.0005)),
        centers=centers,
        joint_names=joint_names,
    )


def wing_pitch_qfrc(qpos: jax.Array, qvel: jax.Array, cfg: WingPitchConfig) -> jax.Array:
    """Dead-zone pitch torque as a generalized force vector, shaped like ``qvel``.

    Returns ``zeros_like(qvel)`` with the per-joint torque scattered into each
    pitch DOF — add it to ``qfrc_applied``. Pure and branch-free.
    """
    dev = qpos[cfg.qpos_adr] - cfg.center
    # Per-stroke-half scaling: neg-dev side scaled by *_asym, pos-dev side by 1.
    neg = dev < 0.0
    k = cfg.stiffness * jnp.where(neg, cfg.stiffness_asym, 1.0)
    fa = cfg.free_angle * jnp.where(neg, cfg.free_asym, 1.0)
    excess = jnp.clip(jnp.abs(dev) - fa, 0.0)
    torque = -cfg.damping * qvel[cfg.dof_adr] - k * excess * jnp.sign(dev)
    return jnp.zeros_like(qvel).at[cfg.dof_adr].add(torque)


def apply(data, cfg: WingPitchConfig):
    """Add the pitch torque to ``data.qfrc_applied`` (MJX Data in, Data out)."""
    qfrc = wing_pitch_qfrc(data.qpos, data.qvel, cfg)
    return data.replace(qfrc_applied=data.qfrc_applied + qfrc)
