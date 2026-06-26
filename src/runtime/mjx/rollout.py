"""Scan-based MJX rollout: motor drive + pitch + aero + ``mjx.step``.

Composes the migrated force sources into one jittable, differentiable rollout,
mirroring the CPU loop in ``run_hanging_poc`` / ``viewer_runtime.step_and_sync``:
each step zeroes ``qfrc_applied``, sets the ramped motor ``ctrl``, adds the
dead-zone pitch torque (:mod:`runtime.mjx.wing_pitch`) and the blade-element aero
force (:mod:`runtime.mjx.aero_runtime`), then steps the physics. The aero
finite-difference carry (:class:`~runtime.mjx.aero_runtime.AeroState`) is threaded
through the ``lax.scan`` so no Python-side state is needed.

    model, data = load_model()              # CPU; calibrates wing pitch
    final, traj = simulate(model, data, n_steps=4000)

``simulate`` reads the calibrated pitch rest angles out of ``data`` so the
dead-zone is centered exactly as the CPU pipeline leaves it.

Autodiff note: differentiate the rollout with **forward mode** (``jax.jacfwd``).
MJX's Newton/CG solver steps with a ``lax.while_loop``, which reverse-mode
(``jax.grad``) cannot trace; forward mode handles it and is efficient for the
small parameter sets here (e.g. the ~14 ``AeroParams``). Pass the parameter being
optimized through ``aero_runtime.build_aero_config(..., params=...)`` so it stays
a traced input rather than a baked constant.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from runtime.mjx import aero_runtime, wing_pitch
from runtime.mjx.aero import AeroParams
from runtime.mjx.aero_runtime import AeroConfig, AeroState
from runtime.mjx.wing_pitch import WingPitchConfig
from runtime.runtime_config import get_control_parameters

_PITCH_JOINTS = ("L_wing_pitch", "R_wing_pitch")


class DriveConfig(NamedTuple):
    """Velocity-actuator drive from the design Fourier control signal."""

    flap_actuator_ids: jax.Array  # (k,) int
    omega: jax.Array | float
    a0: jax.Array | float
    a: jax.Array
    b: jax.Array
    ramp_duration: float


def build_drive_config(
    model: mujoco.MjModel,
    *,
    ramp_duration: float | None = None,
) -> DriveConfig:
    """Resolve flap actuators from the control parameters."""
    control = get_control_parameters()
    names: Sequence[str] = control["flap_actuator_names"]

    ids: list[int] = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        ids.append(aid)

    return DriveConfig(
        flap_actuator_ids=jnp.asarray(ids, dtype=jnp.int32),
        omega=float(control["control_frequency_rad_s"]),
        a0=float(control["control_a0"]),
        a=jnp.asarray(control["control_a"]),
        b=jnp.asarray(control["control_b"]),
        ramp_duration=float(
            control["ramp_duration"] if ramp_duration is None else ramp_duration
        ),
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


def _pitch_centers(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    centers: list[float] = []
    for name in _PITCH_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        centers.append(float(data.qpos[model.jnt_qposadr[jid]]))
    return centers


def _match_dtypes(tree, ref_tree):
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


def make_step_fn(
    mx,
    drive_cfg: DriveConfig,
    pitch_cfg: WingPitchConfig,
    aero_cfg: AeroConfig,
    measure_fn: Callable,
    carry_template,
):
    """Build the ``lax.scan`` body: (data, aero_state) -> (carry, measurement)."""

    def step(carry, _):
        dx, aero_state = carry
        ctrl = drive_ctrl(dx.ctrl, dx.time, drive_cfg)
        dx = dx.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
        dx = wing_pitch.apply(dx, pitch_cfg)
        dx, aero_state = aero_runtime.apply(mx, dx, aero_cfg, aero_state)
        dx = mjx.step(mx, dx)
        carry = _match_dtypes((dx, aero_state), carry_template)
        return carry, measure_fn(carry[0])

    return step


def rollout(
    mx,
    dx,
    n_steps: int,
    drive_cfg: DriveConfig,
    pitch_cfg: WingPitchConfig,
    aero_cfg: AeroConfig,
    *,
    aero_state: AeroState | None = None,
    measure_fn: Callable | None = None,
):
    """Run ``n_steps`` of the composed rollout via ``lax.scan``.

    Returns ``(final_data, final_aero_state, trajectory)`` where ``trajectory`` is
    the stacked per-step output of ``measure_fn`` (default: ``data.qpos``).
    """
    if aero_state is None:
        aero_state = aero_runtime.init_aero_state(aero_cfg)
    if measure_fn is None:
        measure_fn = lambda d: d.qpos  # noqa: E731

    init_carry = (dx, aero_state)
    step = make_step_fn(mx, drive_cfg, pitch_cfg, aero_cfg, measure_fn, init_carry)
    (dx_final, aero_state), traj = jax.lax.scan(
        step, init_carry, xs=None, length=n_steps
    )
    return dx_final, aero_state, traj


def set_reverse_mode_solver(model: mujoco.MjModel, *, iterations: int = 1) -> mujoco.MjModel:
    """Configure ``model`` so ``mjx.step`` is reverse-mode differentiable.

    MJX's ``solver.solve`` runs the main constraint loop as a single ``body()``
    call (no ``lax.while_loop``) **iff** ``opt.iterations == 1``; the line search
    is already a static-length scan. So ``jax.grad`` through ``mjx.step`` works
    only with one solver iteration. Mutates ``model.opt.iterations`` in place and
    returns it. Trade-off: one constraint iteration is looser than the default —
    validate drift before relying on it (see the accuracy gate).
    """
    if iterations != 1:
        raise ValueError(
            "Only iterations=1 avoids MJX's while_loop solver; for N>1 a custom "
            "scan-based solve is required (see the documented fallback)."
        )
    model.opt.iterations = int(iterations)
    return model


def simulate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
    *,
    params: AeroParams = AeroParams(),
    ramp_duration: float | None = None,
    measure_fn: Callable | None = None,
    differentiable: bool = False,
    jit: bool = True,
):
    """Convenience: put a (calibrated) CPU model/data on device and roll out.

    ``data`` is assumed already calibrated (e.g. via ``load_model``); its pitch
    rest angles seed the dead-zone centers. With ``differentiable=True`` the
    model's solver is set to one iteration so the rollout is reverse-mode
    differentiable (see :func:`set_reverse_mode_solver`). Returns
    ``(final_data, trajectory)``.
    """
    if differentiable:
        set_reverse_mode_solver(model)
    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)

    drive_cfg = build_drive_config(model, ramp_duration=ramp_duration)
    pitch_cfg = wing_pitch.config_from_runtime(model, centers=_pitch_centers(model, data))
    aero_cfg = aero_runtime.build_aero_config(model, params=params)

    # Close over the configs (their str/int fields are static, used at trace time)
    # and jit only over the device arrays (model + data).
    def _run(mx_, dx_):
        return rollout(
            mx_, dx_, n_steps, drive_cfg, pitch_cfg, aero_cfg, measure_fn=measure_fn
        )

    run = jax.jit(_run) if jit else _run
    dx_final, _aero_state, traj = run(mx, dx)
    return dx_final, traj
