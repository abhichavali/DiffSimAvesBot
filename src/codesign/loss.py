"""The co-design reward: a robust (soft worst-case) loss over variant rollouts, and what --objective means.

make_loss() turns a tuple of metrics functions (the same design under other integrators,
timesteps, warmups) into jit(jacfwd(loss)). Every term is smooth so gradients flow.
objective_norm/objective_value are the selection-side twins of the loss's objective
term, so a candidate is trained and picked on the same quantity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from codesign.metrics import Metrics, soft_extreme, soft_hinge

# --objective NAME -> Metrics field. Everything here is minimized as (robust value) /
# (its own baseline), so the seed scores ~1. "torque", "energy" (watts per newton) and
# "swing_rms" (a 50/50 composite) are special-cased because they combine fields.
OBJ_METRIC = {"power": "P_supplied", "swing": "swing", "overlap": "C_overlap",
              "tau_rms": "tau_rms", "power_crest": "power_crest", "lift_ripple": "lift_ripple"}
OBJ_CHOICES = ("torque", "energy", "swing_rms") + tuple(OBJ_METRIC)
LIFT_ABOVE_WEIGHT = 0.25  # weight of the above-cap lift penalty relative to the floor


class LossAux(NamedTuple):
    """make_loss's aux: the primary rollout's Metrics plus the robust aggregates."""
    Lbar: jnp.ndarray
    tau_hat: jnp.ndarray
    A_stroke: jnp.ndarray
    S: jnp.ndarray
    C_lc: jnp.ndarray
    P_inactive: jnp.ndarray
    C_overlap: jnp.ndarray
    swing: jnp.ndarray
    P_supplied: jnp.ndarray
    loss: jnp.ndarray
    tau_impl: jnp.ndarray  # tau_hat of variant 1
    tau_rk4: jnp.ndarray  # tau_hat of variant 2
    tau_robust: jnp.ndarray  # max tau_hat over variants
    tau_spread: jnp.ndarray  # max - min tau_hat over variants
    C_valid: jnp.ndarray
    alpha_pk: jnp.ndarray
    k_eff: jnp.ndarray
    conv_err: jnp.ndarray
    tau_peak: jnp.ndarray  # max exact peak over variants
    tau_drift: jnp.ndarray  # max |drift| over variants
    tau_rms: jnp.ndarray
    power_crest: jnp.ndarray
    lift_ripple: jnp.ndarray


def variant_set(base, impl=None, rk4=None, extra=()):
    """The variant tuple make_loss expects: slots 1 and 2 (logged as tau_impl/tau_rk4)
    fall back to the base rollout, which then counts again in the robust aggregates."""
    return (base, impl or base, rk4 or base, *extra)


def make_loss(variants, *, lift_floor, lift_cap, lift_smooth, A_target, weights, w_lc,
              w_power, w_overlap, w_swing, lift0=None, tradeoff=None, w_gap=0.0,
              amp_band_pct=0.02, amp_floor=None, w_drift=0.0, objective="torque",
              obj_norm=None, tau_guard=4.0):
    """jit(jacfwd(loss)): (theta, tau_max, robust_beta) -> (grad, LossAux); loss = -R.

    ``variants`` (see variant_set) are metrics functions of the SAME design; the loss
    optimizes their robust envelope (soft max/min at ``robust_beta``). Terms:
      * Lift: one-sided soft floor below ``lift_floor`` (optional cap), or with
        ``tradeoff`` a controlled lift<->torque trade anchored at ``lift0``.
      * Stroke amplitude: two-sided band of +/-amp_band_pct around A_target (``wK``),
        or a one-sided absolute ``amp_floor``.
      * Peak torque: ``(tau/tau_max)^2`` + 4x extra above the ceiling (``wtau``); under a
        non-torque ``objective`` the objective replaces it and torque becomes a guard.
      * Torque swing ``w_swing``, inactive power ``w_power``, motor overlap ``w_overlap``,
        harmonic energy ``ws``, limit cycle ``w_lc``, cross-variant spread ``w_gap``,
        per-cycle drift ``w_drift``.
    """
    wL, wtau, wK, ws = weights
    # --objective energy: lift enters DIVISIVELY (P/L), never as a linear credit (a
    # linear credit is a score fountain for any lift bought below its fixed rate). The
    # denominator is floored at lift_floor so a collapsed stroke cannot divide by a
    # vanishing lift. softplus(0)=0.693 offsets the hinge AT the floor, so it smooths on
    # 0.002*floor.
    l_hinge = max(1e-9, 0.002 * lift_floor)
    lift_den = lambda L: lift_floor + soft_hinge(L - lift_floor, l_hinge)
    obj_norm = tuple(max(float(x), 1e-12) for x in (obj_norm or (1.0,)))
    if amp_floor is not None:
        # One-sided absolute amplitude FLOOR [m]: shrink below it is walled off, growth
        # is free; the penalty reaches wK at one smoothing width (2% of the floor) short.
        amp_lo, amp_hi = amp_floor, 1e9
        A_smooth = max(1e-9, 0.02 * amp_floor)
    else:
        amp_lo = A_target * (1.0 - amp_band_pct)
        amp_hi = A_target * (1.0 + amp_band_pct)
        A_smooth = max(1e-9, 0.2 * amp_band_pct * A_target)

    def loss(theta, tau_max, robust_beta):
        outs = {}
        for f in variants:  # each distinct rollout once, in order
            if f not in outs:
                outs[f] = f(theta)
        rows = [outs[f] for f in variants]
        m = rows[0]
        taus = jnp.stack([r.tau_hat for r in rows])
        lifts = jnp.stack([r.Lbar for r in rows])
        amps = jnp.stack([r.A_stroke for r in rows])
        swings = jnp.stack([r.swing for r in rows])
        tau_hard = jnp.max(taus)
        tau_used = soft_extreme(taus, robust_beta)
        tau_spread = tau_hard - jnp.min(taus)
        lift_used = soft_extreme(lifts, robust_beta, sign=-1.0)
        lift_high = soft_extreme(lifts, robust_beta)
        amp_low = soft_extreme(amps, robust_beta, sign=-1.0)
        amp_high = soft_extreme(amps, robust_beta)
        swing_used = soft_extreme(swings, robust_beta)
        Pgap = w_gap * (tau_spread / tau_max) ** 2
        Pswing = (swing_used / tau_max) ** 2
        amp_short = soft_hinge(amp_lo - amp_low, A_smooth)
        amp_over = soft_hinge(amp_high - amp_hi, A_smooth)
        if amp_floor is not None:
            Ckin = amp_short**2 / A_smooth**2  # wall, not the weak /A_target**2 slope
        else:
            Ckin = (amp_short**2 + amp_over**2) / A_target**2
        # Worst exact peak + worst |per-cycle drift|. Drift is linear (not squared) so a
        # few mNm of beat still has authority while honest designs pay ~nothing.
        tau_peak_hard = jnp.max(jnp.stack([r.tau_peak for r in rows]))
        drift_hard = jnp.max(jnp.stack([jnp.abs(r.tau_drift) for r in rows]))
        P_drift = w_drift * drift_hard / tau_max
        if objective != "torque":
            # Robust objective (soft-max over variants; lift soft-min), normalized by its
            # baseline; peak torque drops to a soft ceiling guard.
            rob = lambda field: soft_extreme(jnp.stack([getattr(r, field) for r in rows]),
                                             robust_beta)
            over_g = soft_hinge(tau_used - tau_max, 0.02 * tau_max)
            guard = tau_guard * over_g**2 / tau_max**2
            if objective == "energy":  # watts per newton, floored denominator
                P_obj = (rob("P_supplied") / lift_den(lift_used)) / obj_norm[0] + guard
            elif objective == "swing_rms":  # composite: both torque-shape terms, 50/50
                P_obj = 0.5 * (rob("swing") / obj_norm[0] + rob("tau_rms") / obj_norm[1]) + guard
            else:
                P_obj = rob(OBJ_METRIC[objective]) / obj_norm[0] + guard
        if tradeoff is not None:
            # A `tradeoff`% peak-torque cut is worth a 1% lift drop below lift0; the
            # reverse, a `tradeoff`% lift gain pays for a 1% torque rise.
            Ptau = tau_used / tau_max if objective == "torque" else P_obj
            l_smooth = max(1e-9, 0.02 * abs(lift0))
            Pdrop = soft_hinge(lift0 - lift_used, l_smooth) / (abs(lift0) + 1e-9)
            Pgain = jnp.maximum(0.0, lift_used - lift0) / (abs(lift0) + 1e-9)
            # No linear lift credit off the torque objective ('energy' already credits
            # lift inside P_obj; paying twice is the fountain again).
            w_gain = wtau / tradeoff if objective == "torque" else 0.0
            R = (-wtau * Ptau - wtau * tradeoff * Pdrop + w_gain * Pgain
                 - w_swing * Pswing - wK * Ckin - ws * m.S
                 - w_lc * m.C_lc - w_power * m.P_inactive - w_overlap * m.C_overlap
                 - Pgap - P_drift)
        else:
            below = soft_hinge(lift_floor - lift_used, lift_smooth)
            above = soft_hinge(lift_high - lift_cap, lift_smooth) if lift_cap is not None else 0.0
            # Normalized by lift_smooth: the penalty hits wL at ONE smoothing width short
            # (a wall with a thin soft boundary layer).
            PhiL = -(below**2 + LIFT_ABOVE_WEIGHT * above**2) / max(lift_smooth, 1e-9)**2
            over = soft_hinge(tau_used - tau_max, 0.02 * tau_max)
            Ptau = ((tau_used**2 + 4.0 * over**2) / tau_max**2 if objective == "torque"
                    else P_obj)
            R = (wL * PhiL - wtau * Ptau - w_swing * Pswing - wK * Ckin - ws * m.S
                 - w_lc * m.C_lc - w_power * m.P_inactive - w_overlap * m.C_overlap
                 - Pgap - P_drift)
        aux = LossAux(m.Lbar, m.tau_hat, m.A_stroke, m.S, m.C_lc, m.P_inactive, m.C_overlap,
                      m.swing, m.P_supplied, -R, rows[1].tau_hat, rows[2].tau_hat, tau_hard,
                      tau_spread, m.C_valid, m.alpha_pk, m.k_eff, m.conv_err, tau_peak_hard,
                      drift_hard, m.tau_rms, m.power_crest, m.lift_ripple)
        return -R, aux

    return jax.jit(jax.jacfwd(loss, argnums=0, has_aux=True))  # (grad, aux) in one fwd sweep


def objective_norm(objective: str, m0: Metrics, lift_floor: float) -> tuple:
    """The baseline scale(s) that make the objective ~1 at the seed (make_loss obj_norm)."""
    if objective == "energy":
        return (max(m0.P_supplied, 1e-9) / max(m0.Lbar, lift_floor),)
    if objective == "swing_rms":
        return (max(m0.swing, 1e-12), max(m0.tau_rms, 1e-12))
    if objective == "torque":
        return (max(m0.tau_hat, 1e-12),)
    return (max(getattr(m0, OBJ_METRIC[objective]), 1e-12),)


def objective_value(objective: str, row: dict, lift_floor: float, obj_norm: tuple) -> float:
    """The scalar --objective minimizes, from a dict of robust metric values (keys tau,
    power, lift, swing, overlap, tau_rms, power_crest, lift_ripple)."""
    if objective == "torque":
        return row["tau"]
    if objective == "energy":
        return row["power"] / max(row["lift"], lift_floor)
    if objective == "swing_rms":
        return 0.5 * (row["swing"] / obj_norm[0] + row["tau_rms"] / obj_norm[1]) * obj_norm[0]
    return row[objective]
