"""The differentiable MJX rollout the co-design optimizers score: theta -> Metrics.

build_rollout() closes over a Rig and a Genome and returns metrics(theta), a jittable
lax.scan over the CPU runtime's step order (crank command, stroke plane, pitch spring,
aero held between recomputes, mjx.step) that reduces the recorded window to Metrics.
Differentiate it in forward mode (jax.jacfwd): MJX's solver loop is not reverse-mode
differentiable, and the parameter count is small.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from codesign.genome import Genome, ricker_atoms, wavelet_dictionary
from runtime.mjx import aero_runtime, rollout

LP = 10.0  # power of the smooth Lp peak proxies (tau_hat, alpha_pk)
ALPHA_STALL, ALPHA_SMOOTH = math.radians(45.0), math.radians(3.0)  # C_valid AoA hinge
K_MAX, K_SMOOTH = 0.10, 0.01  # C_valid reduced-frequency hinge


class Metrics(NamedTuple):
    """One rollout's scores over its recorded window (SI units)."""
    Lbar: jnp.ndarray  # mean aero lift
    tau_hat: jnp.ndarray  # soft-max over cycles of the per-cycle Lp10 peak |torque| (gradient surrogate)
    A_stroke: jnp.ndarray  # weaker wing's 3D tip-path bounding-box diagonal
    S: jnp.ndarray  # harmonic energy sum n^2 (a_n^2 + b_n^2)
    C_lc: jnp.ndarray  # relative cross-cycle lift variance (0 = stable limit cycle)
    P_inactive: jnp.ndarray  # braking fraction of motor power
    C_overlap: jnp.ndarray  # t1/t4 |torque| cosine similarity
    swing: jnp.ndarray  # std of the |torque| trace
    P_supplied: jnp.ndarray  # mean motoring power
    C_valid: jnp.ndarray  # soft stall + reduced-frequency hinge
    alpha_pk: jnp.ndarray  # Lp10 of the max strip AoA
    k_eff: jnp.ndarray  # reduced frequency
    conv_err: jnp.ndarray  # qpos RMS change over the last cycle
    tau_peak: jnp.ndarray  # exact per-cycle max |torque|: the certification metric
    tau_drift: jnp.ndarray  # last-minus-first cycle peak
    tau_rms: jnp.ndarray
    power_crest: jnp.ndarray  # rms / mean supplied power
    lift_ripple: jnp.ndarray  # within-window lift std / |mean|


def recorded_peak(values, record):
    """Exact peak over recorded samples; matches CPU validation."""
    return jnp.max(jnp.where(record, values, -jnp.inf))


def recorded_lp(values, record, p):
    """Smooth peak proxy over recorded samples."""
    mask = record.astype(values.dtype)
    n = jnp.maximum(jnp.sum(mask), 1.0)
    return (jnp.sum((jnp.abs(values) ** p) * mask) / n) ** (1.0 / p)


def soft_hinge(x, s):
    """Smooth max(0, x): softplus(x/s)*s. C-infinity, no kink at 0."""
    return jax.nn.softplus(x / s) * s


def soft_extreme(values, beta, sign=1.0):
    """Smooth max/min with softmax weights; log-mean avoids a batch-size offset."""
    v = jnp.asarray(values)
    b = jnp.asarray(beta, dtype=v.dtype)
    n = jnp.log(jnp.asarray(v.size, dtype=v.dtype))
    return sign * (jax.nn.logsumexp(sign * b * v) - n) / b


def build_rollout(rig, genome: Genome, omega, *, warmup_steps, sample_steps, steps_per_cycle,
                  aero_substep, control_fixed, flap_t1_mask, strokeplane=None, avg_periods=3):
    """Return metrics(theta, mx=None, dx0=None) -> Metrics as a jittable fn.

    ``theta`` packs the Genome's active groups (see codesign.genome); a frozen control
    reads as ``control_fixed``. a0 and the tendon/pitch/motor genes are log-factors of
    the design base, so one learning rate moves them all sensibly. Tendon genes perturb
    the MJX model arrays directly. ``rig`` supplies the model, start state, drive,
    pitch and aero configs; ``strokeplane`` is rig.strokeplane_info(...) when the
    Genome carries stroke-plane genes. The recorded window is the last ``avg_periods``
    whole cycles.
    """
    mx_base, dx0_base, drive, pitch, aero = rig.mx, rig.dx0, rig.drive, rig.pitch, rig.aero
    tip_ids, motor_ids = rig.tip_ids, rig.motor_ids
    K = genome.K
    base_a0 = float(drive.a0)
    flap_ids = drive.flap_actuator_ids
    harmonics = jnp.arange(1, K + 1, dtype=jnp.float32)
    if genome.wavelet:  # the 2K control coeffs are atom amplitudes of a dyadic Ricker frame
        wavelet = wavelet_dictionary(2 * K)
        wl_centers = jnp.asarray(wavelet[0], dtype=jnp.float32)
        wl_sigmas = jnp.asarray(wavelet[1], dtype=jnp.float32)
        wl_freqs = jnp.asarray(wavelet[2], dtype=jnp.float32)
    l_ids, r_ids = rig.tendons[0], rig.tendons[1]
    total_steps = warmup_steps + sample_steps
    n_full = max(1, total_steps // steps_per_cycle)
    n_avg = max(1, min(int(avg_periods), n_full))
    window_start = (n_full - n_avg) * steps_per_cycle
    window_end = n_full * steps_per_cycle
    record = (jnp.arange(total_steps) >= window_start) & (jnp.arange(total_steps) < window_end)
    qdim = int(dx0_base.qpos.shape[0])
    if genome.strokeplane:
        sp_qadr = int(strokeplane["qadr"])
        sp_body_ids = strokeplane["body_ids"]
        sp_tilt_signs = strokeplane["tilt_signs"]
        sp_tilt0, sp_pitch0 = float(strokeplane["tilt0"]), float(strokeplane["pitch0"])
        sp_max = float(strokeplane["max_rad"])

    def perturbed_model(mxb, g):
        # Log-factors (seeded exp(0)=1) scale the design base; exp keeps values > 0.
        # The base comes off `mxb`, not the captured arrays, so a per-genome model
        # (evolve_codesign's geometry genes) scales ITS OWN derived rest lengths.
        stiff, lenspring = mxb.tendon_stiffness, mxb.tendon_lengthspring
        base_stiff, base_lenspring = stiff, lenspring
        if genome.tendons:  # per tendon (or mirrored pair: l_ids/r_ids)
            sp = base_stiff[l_ids] * jnp.exp(g.stiff)
            stiff = stiff.at[l_ids].set(sp).at[r_ids].set(sp)
            lpl = base_lenspring[l_ids] * jnp.exp(g.pre)[:, None]
            lpr = base_lenspring[r_ids] * jnp.exp(g.pre)[:, None]
            lenspring = lenspring.at[l_ids].set(lpl).at[r_ids].set(lpr)
        if genome.motor:  # one shared stiffness + one shared pretension over the 4 motor tendons
            stiff = stiff.at[motor_ids].set(base_stiff[motor_ids] * jnp.exp(g.motor[0]))
            lenspring = lenspring.at[motor_ids].set(base_lenspring[motor_ids] * jnp.exp(g.motor[1]))
        return mxb.replace(tendon_stiffness=stiff, tendon_lengthspring=lenspring)

    def metrics(theta, mx_g=None, dx_g=None) -> Metrics:
        """``mx_g``/``dx_g`` override the model and start state for THIS call (same
        array shapes, so one trace serves evolve_codesign's per-genome vmap)."""
        mx, dx0 = (mx_base if mx_g is None else mx_g), (dx0_base if dx_g is None else dx_g)
        g = genome.split(theta, control_fixed)
        ctrl = g.control
        a0 = base_a0 * jnp.exp(ctrl[0])
        a, b = ctrl[1 : 1 + K], ctrl[1 + K :]
        wl_coeff = ctrl[1:]  # wavelet mode: all 2K coeffs are atom amplitudes
        # Crank-LOCKED harmonic clock: a0 (=ctrl[0]) is the crank velocity command, so it
        # IS the flap frequency; the modulation rides the same clock.
        omega_eff = omega * jnp.exp(g.f_log + ctrl[0])
        mx_eff = perturbed_model(mx, g) if (genome.tendons or genome.motor) else mx
        pc = pitch
        if genome.pitch:  # deadband + stiffness as log-factors of the design base
            pc = pitch._replace(
                upstroke_pitch=pitch.upstroke_pitch * jnp.exp(g.pitch[0]),
                downstroke_pitch=pitch.downstroke_pitch * jnp.exp(g.pitch[1]),
                upstroke_stiffness=pitch.upstroke_stiffness * jnp.exp(g.pitch[2]),
                downstroke_stiffness=pitch.downstroke_stiffness * jnp.exp(g.pitch[3]),
            )

        def step(carry, rec):
            dx, state, cq, cnfw, ca, cu, i = carry
            mx_step = mx_eff
            if genome.strokeplane:
                # Stroke-plane signal over crank phase (L_t1 hinge angle): per DOF mean +
                # 2 harmonics, tanh-bounded to +/-sp_max around the compiled mount angles.
                # body_quat is re-derived every mjx.step, so the whole flap arc follows.
                th = dx.qpos[sp_qadr]
                basis = jnp.stack([jnp.ones_like(th), jnp.cos(th), jnp.sin(th),
                                   jnp.cos(2.0 * th), jnp.sin(2.0 * th)])
                tilt = sp_tilt0 + sp_max * jnp.tanh(jnp.dot(g.sp[:5], basis) / sp_max)
                pitv = sp_pitch0 + sp_max * jnp.tanh(jnp.dot(g.sp[5:], basis) / sp_max)
                # euler "xyz" with z=0 -> quat = qx(tilt) * qy(pitch), wxyz; tilt
                # sign-mirrored on R (matches compile.py _mirror_shoulder).
                hx, hy = 0.5 * tilt * sp_tilt_signs, 0.5 * pitv
                cx, sx, cy, sy = jnp.cos(hx), jnp.sin(hx), jnp.cos(hy), jnp.sin(hy)
                quats = jnp.stack([cx * cy, sx * cy, cx * sy, sx * sy], axis=1)
                mx_step = mx_eff.replace(body_quat=mx_eff.body_quat.at[sp_body_ids].set(
                    quats.astype(mx_eff.body_quat.dtype)))
            phase = omega_eff * dx.time
            if genome.wavelet:
                mod = jnp.sum(wl_coeff * ricker_atoms(phase, wl_centers, wl_sigmas, jnp))
            else:
                mod = jnp.sum(a * jnp.cos(harmonics * phase) + b * jnp.sin(harmonics * phase))
            cmd = a0 * (1.0 + mod)
            ctrl_vec = jnp.zeros_like(dx.ctrl).at[flap_ids].set(cmd)

            def held_aero(dx):
                # Recompute every `aero_substep` steps and hold in between; the aero
                # state (FD previous velocity) only advances on a recompute.
                def recompute(_):
                    return aero_runtime.aero_wrench_with_diagnostics(mx_step, dx, aero, state)

                def hold(_):
                    return cq, cnfw, state, ca, cu

                q, nfw, ns, alpha, speed = jax.lax.cond(
                    (i % aero_substep) == 0, recompute, hold, operand=None)
                return q, (q, nfw, ns, alpha, speed)

            dx, (qfrc, net_force_world, state, alpha, speed) = rollout.physics_step(
                mx_step, dx, ctrl_vec, pc, held_aero)
            carry = rollout.match_dtypes((dx, state, qfrc, net_force_world, alpha, speed, i + 1), carry)
            tau = jnp.max(jnp.abs(dx.actuator_force[flap_ids]))
            # Motor mechanical power per crank actuator: P>0 drives, P<0 brakes.
            pm = dx.actuator_force[flap_ids] * dx.actuator_velocity[flap_ids]
            power = jnp.sum(pm)
            p_sup = jnp.sum(jnp.maximum(0.0, pm))
            tips = dx.site_xpos[tip_ids].reshape(-1)  # [Lx,Ly,Lz, Rx,Ry,Rz]
            absf = jnp.abs(dx.actuator_force[flap_ids])
            tauA = jnp.sum(absf * flap_t1_mask)  # t1 ("blue", upstroke)
            tauB = jnp.sum(absf * (1.0 - flap_t1_mask))  # t4 ("orange", downstroke)
            keep = rec.astype(net_force_world.dtype)
            return carry, jnp.concatenate(
                [jnp.array([net_force_world[2] * keep, tau, power, tauA, tauB, p_sup,
                            alpha, speed]), tips, dx.qpos])

        state0 = aero_runtime.init_aero_state(aero)
        carry0 = (dx0, state0, jnp.zeros_like(dx0.qvel), jnp.zeros(3),
                  jnp.array(0.0), jnp.array(0.0), jnp.int32(0))
        _, rec = jax.lax.scan(step, carry0, record)
        lift_z, tau, power = rec[:, 0], rec[:, 1], rec[:, 2]
        tauA, tauB, p_sup = rec[:, 3], rec[:, 4], rec[:, 5]
        alpha_trace, speed_trace = rec[:, 6], rec[:, 7]
        tips = rec[:, 8:14].reshape(-1, 2, 3)  # (T, wing[L,R], xyz)
        qpos = rec[:, 14:14 + qdim]
        mask = record.astype(rec.dtype)
        n_win = jnp.maximum(jnp.sum(mask), 1.0)

        Lbar = jnp.sum(lift_z) / n_win
        # Per-cycle exact torque peaks over the window. tau_peak (max over cycles) is what
        # CPU certification measures; tau_drift (last minus first cycle peak) exposes slow
        # beats a short window would sample as fake gains. tau_hat (the gradient surrogate)
        # is the soft-max over per-cycle Lp, which neither a narrower spike nor one good
        # parked cycle can lower.
        whole_cycles = (window_end <= total_steps
                        and (window_end - window_start) == n_avg * steps_per_cycle)
        if whole_cycles:
            tau_cyc = tau[window_start:window_end].reshape(n_avg, steps_per_cycle)
            pc_exact = jnp.max(tau_cyc, axis=1)
            tau_peak = jnp.max(pc_exact)
            tau_drift = pc_exact[-1] - pc_exact[0]
            pc_lp = jnp.mean(tau_cyc ** LP, axis=1) ** (1.0 / LP)
            tau_hat = soft_extreme(pc_lp, 300.0)
        else:  # window shorter than one cycle (smoke tests): whole-window fallback
            tau_hat = recorded_lp(tau, record, LP)
            tau_peak = recorded_peak(tau, record)
            tau_drift = jnp.array(0.0)
        # 3D stroke amplitude per wing: the bounding-box diagonal of the tip path. The MIN
        # over wings keeps BOTH flapping (unmirrored tendons let one wing compensate).
        tip_w = tips[window_start:window_end]
        ptp = jnp.max(tip_w, axis=0) - jnp.min(tip_w, axis=0)
        A_stroke = jnp.min(jnp.sqrt(jnp.sum(ptp**2, axis=1)))
        # Energy regularizer: fine scales pay more, like harmonic order n for Fourier.
        if genome.wavelet:
            S = jnp.sum(wl_freqs**2 * wl_coeff**2)
        else:
            S = jnp.sum(harmonics**2 * (a**2 + b**2))

        # Limit-cycle stability: relative cross-cycle variance of the lift trace (needs
        # >= 2 whole cycles, else 0).
        if n_full >= 2:
            lift_cyc = lift_z[window_start:window_end].reshape(n_avg, steps_per_cycle)
            C_lc = jnp.mean(jnp.var(lift_cyc, axis=0)) / (jnp.mean(lift_cyc**2) + 1e-12)
        else:
            C_lc = jnp.array(0.0)

        # Braking fraction of motor power, and mean SUPPLIED (motoring) power [W].
        p_s = jnp.where(record, power, 0.0)
        P_brake = jnp.sum(jnp.maximum(0.0, -p_s)) / n_win
        P_scale = jnp.sum(jnp.abs(p_s)) / n_win + 1e-9
        P_inactive = P_brake / P_scale
        P_supplied = jnp.sum(jnp.where(record, p_sup, 0.0)) / n_win

        # Anti-overlap: uncentered cosine similarity of the t1 and t4 |torque| traces.
        a_s = jnp.where(record, tauA, 0.0)
        b_s = jnp.where(record, tauB, 0.0)
        C_overlap = jnp.sum(a_s * b_s) / (jnp.sqrt(jnp.sum(a_s**2) * jnp.sum(b_s**2)) + 1e-9)

        tau_win = tau[window_start:window_end]
        swing = jnp.sqrt(jnp.mean((tau_win - jnp.mean(tau_win)) ** 2))

        alpha_pk = recorded_lp(alpha_trace, record, LP)
        Ubar = jnp.sum(speed_trace * mask) / n_win + 1e-9
        k_eff = omega_eff * jnp.mean(aero.chords) / (2.0 * Ubar)
        C_valid = ((soft_hinge(alpha_pk - ALPHA_STALL, ALPHA_SMOOTH) ** 2) / (ALPHA_STALL ** 2)
                   + (soft_hinge(k_eff - K_MAX, K_SMOOTH) ** 2) / (K_MAX ** 2))
        if n_full >= 2:
            qcyc = qpos[:n_full * steps_per_cycle].reshape(n_full, steps_per_cycle, qdim)
            conv_err = jnp.sqrt(jnp.mean((qcyc[-1] - qcyc[-2]) ** 2))
        else:
            conv_err = jnp.array(0.0)
        tau_w = jnp.where(record, tau, 0.0)
        tau_rms = jnp.sqrt(jnp.sum(tau_w ** 2) / n_win)
        p_w = jnp.where(record, p_sup, 0.0)
        p_mean = jnp.sum(p_w) / n_win
        p_rms = jnp.sqrt(jnp.sum(p_w ** 2) / n_win)
        power_crest = p_rms / (p_mean + 1e-9)
        l_w = jnp.where(record, lift_z, 0.0)
        l_var = jnp.sum((l_w - Lbar) ** 2 * mask) / n_win  # WITHIN-cycle lift ripple
        lift_ripple = jnp.sqrt(l_var) / (jnp.abs(Lbar) + 1e-9)
        return Metrics(Lbar, tau_hat, A_stroke, S, C_lc, P_inactive, C_overlap, swing,
                       P_supplied, C_valid, alpha_pk, k_eff, conv_err, tau_peak, tau_drift,
                       tau_rms, power_crest, lift_ripple)

    return metrics
