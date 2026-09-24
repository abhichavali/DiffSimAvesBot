"""Dead-zone restorative spring for the wing pitch joints, in JAX.

JAX port of :class:`runtime.wing_pitch.WingPitchLimiter`. The pitch hinges
are emitted passively free (zero stiffness/damping in MJCF); this applies, every
step, a constant velocity ``damping`` plus a restorative spring that engages only
once the pitch deviates more than the one-sided pitch deadband from its
leveled-neutral rest:

    torque = -damping * qvel
             - stiffness_side * max(|dev| - pitch_side, 0) * sign(dev)

where ``dev = qpos - center``. It is branch-free (``clip``/``sign``), so it is
fully ``jit``/``vmap``-able and differentiable for use inside an ``mjx.step``
rollout. The CPU step (``runtime.sim.PlateWingSim``) zeroes ``qfrc_applied`` each
step then accumulates the aero force and this into it; :func:`apply` mirrors that by
*adding* to the current ``qfrc_applied``, so the rollout must zero (or freshly build)
``qfrc_applied`` before composing force sources (``rollout.physics_step`` does).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco

_JOINTS = ("L_wing_pitch", "R_wing_pitch")


class WingPitchConfig(NamedTuple):
    """Static-per-build configuration for the dead-zone pitch controller.

    A JAX pytree: pass it straight into a jitted step. ``qpos_adr``/``dof_adr``
    index the (possibly batched) state vectors; ``center`` is the leveled rest
    angle each joint's dead zone is measured from.
    """

    qpos_adr: jax.Array  # (n_joints,) int
    dof_adr: jax.Array  # (n_joints,) int
    center: jax.Array  # (n_joints,) float — leveled-neutral rest angle
    upstroke_pitch: jax.Array  # scalar float — positive-dev zero-torque angle
    downstroke_pitch: jax.Array  # scalar float — negative-dev zero-torque angle
    upstroke_stiffness: jax.Array  # scalar float — positive-dev spring gain
    downstroke_stiffness: jax.Array  # scalar float — negative-dev spring gain
    damping: jax.Array  # scalar float — velocity damping applied everywhere


def build_wing_pitch_config(model: mujoco.MjModel, morph, data: mujoco.MjData) -> WingPitchConfig:
    """Resolve the pitch joints on a CPU model into a JAX-ready config for ``morph.wing.pitch``.

    Each dead zone is centered on the joint's angle in ``data``: the settled rest pose
    (``runtime.sim.rest_state``).
    """
    pitch = morph.wing.pitch
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    resolved_centers: list[float] = []
    for name in _JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        dof_adr.append(int(model.jnt_dofadr[joint_id]))
        resolved_centers.append(float(data.qpos[qpos_adr[-1]]))

    return WingPitchConfig(
        qpos_adr=jnp.asarray(qpos_adr, dtype=jnp.int32),
        dof_adr=jnp.asarray(dof_adr, dtype=jnp.int32),
        center=jnp.asarray(resolved_centers, dtype=jnp.float32),
        upstroke_pitch=jnp.asarray(float(pitch.upstroke_pitch_rad), dtype=jnp.float32),
        downstroke_pitch=jnp.asarray(float(pitch.downstroke_pitch_rad), dtype=jnp.float32),
        upstroke_stiffness=jnp.asarray(float(pitch.upstroke_stiffness), dtype=jnp.float32),
        downstroke_stiffness=jnp.asarray(float(pitch.downstroke_stiffness), dtype=jnp.float32),
        damping=jnp.asarray(float(pitch.damping), dtype=jnp.float32),
    )


def wing_pitch_qfrc(qpos: jax.Array, qvel: jax.Array, cfg: WingPitchConfig) -> jax.Array:
    """Dead-zone pitch torque as a generalized force vector, shaped like ``qvel``.

    Returns ``zeros_like(qvel)`` with the per-joint torque scattered into each
    pitch DOF — add it to ``qfrc_applied``. Pure and branch-free.
    """
    dev = qpos[cfg.qpos_adr] - cfg.center
    neg = dev < 0.0
    k = jnp.where(neg, cfg.downstroke_stiffness, cfg.upstroke_stiffness)
    fa = jnp.where(neg, cfg.downstroke_pitch, cfg.upstroke_pitch)
    excess = jnp.clip(jnp.abs(dev) - fa, 0.0)
    torque = -cfg.damping * qvel[cfg.dof_adr] - k * excess * jnp.sign(dev)
    return jnp.zeros_like(qvel).at[cfg.dof_adr].add(torque)


def apply(data, cfg: WingPitchConfig):
    """Add the pitch torque to ``data.qfrc_applied`` (MJX Data in, Data out)."""
    qfrc = wing_pitch_qfrc(data.qpos, data.qvel, cfg)
    return data.replace(qfrc_applied=data.qfrc_applied + qfrc)
