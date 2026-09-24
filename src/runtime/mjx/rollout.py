"""MJX drive + one physics step in the CPU runtime's order, and a plain design-drive rollout.

physics_step() is the step every MJX rollout shares. It mirrors the canonical CPU step,
``runtime.sim.PlateWingSim``: zero ``qfrc_applied``, set the motor ``ctrl``, add the
dead-zone pitch torque (:mod:`runtime.mjx.wing_pitch`) and the blade-element aero
force (:mod:`runtime.mjx.aero_runtime`), then ``mjx.step``. PlateWingSim adds aero
before pitch; with just those two terms on a zeroed ``qfrc_applied`` the order is
exact either way (a + p == p + a).

codesign.metrics runs physics_step under a learned command, run_mjx_fixed_frequency
under its own scan (step drive, lift recording). simulate() runs it over the design's
ramped Fourier drive with aero every step; only the golden harness calls it, as its
MJX reference rollout (mjx/simulate).

Autodiff: differentiate with **forward mode** (``jax.jacfwd``). MJX's Newton/CG
solver steps with a ``lax.while_loop``, which reverse mode (``jax.grad``) cannot
trace unless ``model.opt.iterations == 1``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from runtime.crank import CRANK_ACTUATORS
from runtime.mjx import aero_runtime, wing_pitch
from runtime.mjx.wing_pitch import WingPitchConfig

class DriveConfig(NamedTuple):
    """Velocity-actuator drive from the design Fourier control signal."""

    flap_actuator_ids: jax.Array  # (k,) int
    omega: jax.Array | float
    a0: jax.Array | float
    a: jax.Array
    b: jax.Array
    ramp_duration: float


def build_drive_config(model: mujoco.MjModel, morph, control, *, ramp_duration: float = 2.0) -> DriveConfig:
    """The crank actuators and ``control`` (a FourierControl) signed by ``morph``'s motor direction."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in CRANK_ACTUATORS]
    direction = morph.motors.motor_direction
    return DriveConfig(
        flap_actuator_ids=jnp.asarray([i for i in ids if i >= 0], dtype=jnp.int32),
        omega=float(control.omega),
        a0=float(direction * control.a0),
        a=jnp.asarray(tuple(direction * value for value in control.a)),
        b=jnp.asarray(tuple(direction * value for value in control.b)),
        ramp_duration=float(ramp_duration),
    )


def drive_ctrl(ctrl_like: jax.Array, t: jax.Array, cfg: DriveConfig) -> jax.Array:
    """Ramped Fourier velocity command vector, shaped like ``data.ctrl``."""
    ramp = jnp.where(
        cfg.ramp_duration > 0.0, jnp.clip(t / cfg.ramp_duration, 0.0, 1.0), 1.0
    )
    harmonics = jnp.arange(1, cfg.a.shape[0] + 1, dtype=ctrl_like.dtype)
    phase = cfg.omega * t
    base = cfg.a0 + jnp.sum(cfg.a * jnp.cos(harmonics * phase) + cfg.b * jnp.sin(harmonics * phase))
    command = base * ramp
    return jnp.zeros_like(ctrl_like).at[cfg.flap_actuator_ids].set(command)


def match_dtypes(tree, ref_tree):
    """Cast every leaf of ``tree`` to the dtype of the matching leaf in ``ref_tree``.

    ``mjx.step`` promotes some int32 index fields (tendon wrap / contact) to int64
    when x64 is enabled, which makes the ``lax.scan`` carry type drift from the
    initial ``put_data`` carry and aborts the scan. Casting back keeps the carry
    type stable in both float32 (no-op) and x64 modes; the affected fields are
    small indices, so the down-cast is lossless.

    Only leaves whose dtype actually differs are cast (the ``a.dtype == ref.dtype``
    check is static, evaluated at trace time). Casting *every* leaf — even no-op
    same-dtype casts — inserts convert ops that cost ~20 ms/step in float32, so the
    skip keeps the hot path fast while still fixing the x64 carry drift.
    """
    return jax.tree_util.tree_map(
        lambda a, ref: a if a.dtype == ref.dtype else a.astype(ref.dtype),
        tree,
        ref_tree,
    )


def physics_step(mx, dx, ctrl, pitch_cfg: WingPitchConfig, aero_qfrc: Callable):
    """One step in the CPU runtime's force order; returns ``(data, aux)``.

    ``aero_qfrc(data) -> (qfrc, aux)`` supplies the aero generalized force (and
    whatever the caller wants back, e.g. the updated aero state); it sees the data
    after ctrl and pitch torque are set.
    """
    dx = dx.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
    dx = wing_pitch.apply(dx, pitch_cfg)
    qfrc, aux = aero_qfrc(dx)
    dx = dx.replace(qfrc_applied=dx.qfrc_applied + qfrc)
    return mjx.step(mx, dx), aux


def simulate(model: mujoco.MjModel, data: mujoco.MjData, morph, control, n_steps: int):
    """Put a settled CPU model/data on device and run ``n_steps`` of the design drive
    (``morph`` + its ``control``), aero recomputed every step. Returns
    ``(final_data, qpos trajectory)``.

    The pitch dead zone is centered on ``data``'s settled pitch angles.
    """
    drive_cfg = build_drive_config(model, morph, control)
    pitch_cfg = wing_pitch.build_wing_pitch_config(model, morph, data)
    aero_cfg = aero_runtime.build_aero_config(model, morph)

    def run(mx, dx):
        def step(carry, _):
            dx, aero_state = carry
            ctrl = drive_ctrl(dx.ctrl, dx.time, drive_cfg)

            def aero_qfrc(d):
                qfrc, _, new_state, _, _ = aero_runtime.aero_wrench(mx, d, aero_cfg, aero_state)
                return qfrc, new_state

            dx, aero_state = physics_step(mx, dx, ctrl, pitch_cfg, aero_qfrc)
            carry = match_dtypes((dx, aero_state), carry0)
            return carry, carry[0].qpos

        carry0 = (dx, aero_runtime.init_aero_state(aero_cfg))
        return jax.lax.scan(step, carry0, xs=None, length=n_steps)

    (dx_final, _aero_state), traj = jax.jit(run)(mjx.put_model(model), mjx.put_data(model, data))
    return dx_final, traj
