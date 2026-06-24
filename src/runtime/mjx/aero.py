"""Quasi-steady blade-element wing aerodynamics, in JAX.

JAX port of ``utils.aero.Aerodynamics_left`` / ``Aerodynamics_right`` (plus the
``CL_hovering`` / ``CD_hovering`` / ``ry_rad`` helpers). Computes the per-strip
aerodynamic force and pitching moment for a wing, expressed in the wing root-site
frame — the same quantities the numpy model returns, ready to be rotated to world
and applied at each blade site by the MJX runtime layer.

Two design choices serve the autodiff goal:

* Branch-free. The ``norm_v == 0`` guard, the ``rdi`` span cutoff, the
  ``lift_sign`` flip, and the CoP clip are all expressed with ``where``/``clip``
  so the function is smooth enough to ``jit``/``vmap``/``grad``.
* The Sane & Dickinson polar coefficients (and the other empirical constants)
  live in :class:`AeroParams`, a JAX pytree, so gradients can flow into them.

Left vs right wings differ only by a fixed set of sign flips (mirror symmetry);
``side`` ("L"/"R") selects them at trace time. The geometry inputs
(``length``/``chords``/``span_cutoffs``) come from the host-side
``wing_geometry`` STL parse and are passed in as constants.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

RHO_AIR = 1.1839
_PI = math.pi


class AeroParams(NamedTuple):
    """Empirical aero coefficients (defaults: canonical fruit-fly fit).

    A JAX pytree, so any field can be optimized via ``grad``. ``a..d`` are the
    drag polar, ``e..h`` the lift polar (Sane & Dickinson 2002, revolving wing);
    the rest are the rotational-drag / added-mass / CoP constants.
    """

    a: jax.Array | float = 1.92
    b: jax.Array | float = 1.55
    c: jax.Array | float = 2.04
    d: jax.Array | float = 9.82
    e: jax.Array | float = 0.225
    f: jax.Array | float = 1.58
    g: jax.Array | float = 2.13
    h: jax.Array | float = 7.2
    Crd: jax.Array | float = 5.0
    Cadd2: jax.Array | float = 0.0
    c4: jax.Array | float = 1.0
    ct1: jax.Array | float = 0.82
    ct2: jax.Array | float = 0.05
    x_rot: jax.Array | float = 0.0


class _SideSigns(NamedTuple):
    Fik: float  # rotational-drag chordwise force sign
    lift: float  # FLi sign in the force/torque sums
    rot: float  # F_rot sign in the force/torque sums
    addo: float  # lamda_xo (rotational added-mass) sign
    add2: float  # F_add2 sign
    cof: float  # chordwise CoP lever-arm sign


def _side_signs(side: str) -> _SideSigns:
    if side == "L":
        return _SideSigns(Fik=1.0, lift=-1.0, rot=-1.0, addo=1.0, add2=-1.0, cof=-1.0)
    if side == "R":
        return _SideSigns(Fik=-1.0, lift=1.0, rot=1.0, addo=-1.0, add2=1.0, cof=1.0)
    raise ValueError(f"side must be 'L' or 'R', got {side!r}")


def _cd(alpha: jax.Array, p: AeroParams) -> jax.Array:
    alpha_deg = alpha * 180.0 / _PI
    return p.a - p.b * jnp.cos((p.c * alpha_deg - p.d) * _PI / 180.0)


def _cl(alpha: jax.Array, p: AeroParams) -> jax.Array:
    alpha_deg = alpha * 180.0 / _PI
    return p.e + p.f * jnp.sin((p.g * alpha_deg - p.h) * _PI / 180.0)


def _ry_apply(theta: jax.Array, v: jax.Array) -> jax.Array:
    """Apply a right-handed rotation about the Y axis to vector ``v``."""
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    return jnp.stack(
        [
            cos * v[0] + sin * v[2],
            v[1],
            -sin * v[0] + cos * v[2],
        ]
    )


def aero_strip(
    omega_w: jax.Array,
    omega_b: jax.Array,
    vel_h: jax.Array,
    omega_w_d: jax.Array,
    vel_h_prev: jax.Array,
    span_fraction: jax.Array,
    chord: jax.Array,
    *,
    side: str,
    span_cutoffs: jax.Array,
    params: AeroParams,
    length: float,
    num_blade: int,
    rho: float = RHO_AIR,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Force and pitching moment for one blade strip, in the root-site frame.

    All vector inputs are length-3 in the root-site frame. ``span_fraction`` is
    the strip's fractional span position (``i`` in the numpy model), ``chord`` its
    chord. ``dt`` is the finite-difference window (timestep * substep) used for
    the added-mass acceleration term. Returns ``(force[3], torque[3])``.
    """
    p = params
    s = _side_signs(side)
    x_rot = p.x_rot
    Rw = length
    # Active numpy code zeroes both CoP offsets regardless of x_rot.
    r_cof_rot = 0.0
    r_cof_add = 0.0

    # --- Rotational drag, summed over all strips (gated by span cutoff) ---
    m = (jnp.arange(1, num_blade + 1).astype(span_cutoffs.dtype)) / num_blade
    rdi = jnp.where(Rw * span_fraction <= span_cutoffs, 1.0, 0.0)
    wsum = omega_w[1] + omega_b[1]
    abswsum = jnp.abs(wsum)
    span = Rw / num_blade
    local = chord * (x_rot - m)
    base = rdi * 0.5 * rho * wsum * abswsum * span * p.Crd
    Fik_lt = s.Fik * jnp.sum(base * local * jnp.abs(local) * chord / num_blade)
    Tik_lt = -jnp.sum(base * (local**2) * jnp.abs(local) * chord / num_blade)

    # --- Blade-element velocity at the strip leading edge ---
    rle = -span_fraction * Rw
    lever = jnp.stack([jnp.zeros_like(rle), rle, jnp.zeros_like(rle)])
    vle = jnp.cross(omega_w + omega_b, lever) + vel_h
    vle = vle.at[1].set(0.0)

    omega_wLI = omega_w - omega_w_d * dt
    vleLI = jnp.cross(omega_wLI + omega_b, lever) + vel_h_prev
    vleLI = vleLI.at[1].set(0.0)

    acc_xw = (vle[0] - vleLI[0]) / dt
    omega_yw_dot = omega_w_d[1]

    # --- Velocity direction & folded angle of attack (branch-free) ---
    norm_v = jnp.linalg.norm(vle)
    safe = norm_v > 0.0
    denom = jnp.where(safe, norm_v, 1.0)
    dir_vle = jnp.where(safe, vle / denom, jnp.zeros(3))
    alpha = jnp.arctan2(jnp.abs(dir_vle[0]), jnp.abs(dir_vle[2]))

    dir_drag = -dir_vle
    dir_drag = dir_drag.at[1].set(0.0)
    lift_sign = jnp.where(vle[0] >= 0.0, 1.0, -1.0)
    dir_lift = _ry_apply(-lift_sign * _PI / 2.0, dir_vle)

    # --- Translational lift & drag ---
    q = 0.5 * rho * norm_v**2
    span_chord = chord * (Rw / num_blade)
    FDi = q * _cd(alpha, p) * span_chord * dir_drag
    FLi = q * _cl(alpha, p) * span_chord * dir_lift

    r_cof_trans = x_rot - (p.ct1 * jnp.abs(alpha) / _PI + p.ct2)
    r_cof_trans = jnp.clip(r_cof_trans, x_rot - 1.0, x_rot) * chord

    # --- Rotational circulation force ---
    circulation_rot = _PI * (0.75 - x_rot) * chord**2 * omega_w[1]
    f_rot_x = p.c4 * jnp.abs(rle) * (1.0 / num_blade) * rho * circulation_rot * norm_v
    F_rot_i = jnp.stack([f_rot_x, jnp.zeros_like(f_rot_x), jnp.zeros_like(f_rot_x)])

    # --- Added mass ---
    cr = (-x_rot + 0.5) * chord
    lamda_x = _PI * rho * (0.5 * chord) ** 2
    lamda_xo = -_PI * rho * (0.5 * chord) ** 2 * cr
    F_addx_i = -lamda_x * acc_xw + s.addo * lamda_xo * omega_yw_dot
    F_add_i = (Rw / num_blade) * jnp.stack(
        [F_addx_i, jnp.zeros_like(F_addx_i), jnp.zeros_like(F_addx_i)]
    )
    f_add2 = (
        s.add2
        * p.Cadd2
        * _PI
        * rho
        * (0.5 * chord) ** 2
        * Rw
        * (Rw / num_blade)
        * omega_w[2]
        * omega_w[1]
    )
    F_add2_vec = jnp.stack([f_add2, jnp.zeros_like(f_add2), jnp.zeros_like(f_add2)])

    # --- Sums (force in root frame, pure chordwise pitching moment) ---
    force = (
        FDi
        + s.lift * FLi
        + F_add_i
        + s.rot * F_rot_i
        + jnp.stack([Fik_lt, jnp.zeros_like(Fik_lt), jnp.zeros_like(Fik_lt)])
        + F_add2_vec
    )
    torque = (
        jnp.cross(jnp.array([0.0, 0.0, s.cof * r_cof_trans]), FDi + s.lift * FLi)
        + jnp.cross(jnp.array([0.0, 0.0, s.cof * r_cof_add]), F_add_i + F_add2_vec)
        + jnp.cross(jnp.array([0.0, 0.0, s.cof * r_cof_rot]), s.rot * F_rot_i)
        + jnp.stack([jnp.zeros_like(Tik_lt), Tik_lt, jnp.zeros_like(Tik_lt)])
    )
    return force, torque


def aero_wing(
    omega_w: jax.Array,
    omega_b: jax.Array,
    vel_h: jax.Array,
    omega_w_d: jax.Array,
    vel_h_prev: jax.Array,
    *,
    side: str,
    chords: jax.Array,
    span_cutoffs: jax.Array,
    params: AeroParams = AeroParams(),
    length: float,
    num_blade: int,
    rho: float = RHO_AIR,
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Per-strip forces/torques for a whole wing, in the root-site frame.

    Returns ``(forces[num_blade, 3], torques[num_blade, 3])`` — one row per blade
    strip, matching the numpy model's per-strip calls. Strip span fractions are
    ``(j + 0.5) / num_blade``.
    """
    span_fracs = (jnp.arange(num_blade, dtype=chords.dtype) + 0.5) / num_blade

    def strip(span_fraction, chord):
        return aero_strip(
            omega_w,
            omega_b,
            vel_h,
            omega_w_d,
            vel_h_prev,
            span_fraction,
            chord,
            side=side,
            span_cutoffs=span_cutoffs,
            params=params,
            length=length,
            num_blade=num_blade,
            rho=rho,
            dt=dt,
        )

    return jax.vmap(strip)(span_fracs, chords)
