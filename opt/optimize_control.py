"""Gradient-based optimization of the Fourier velocity-servo control signal.

Warm-start phase: hold morphology/tendons fixed and optimize only the angular-
velocity command driving the crank motors

    u(t) = a0 + sum_{n=1..K} a_n cos(n w t) + b_n sin(n w t)   (w fixed)

against the design objective (see aves_control_diff.org):

    R = wL*PhiL(Lbar) - wtau*Ptau(tau_hat) - wK*Ckin - ws*S - w_lc*C_lc  (maximize)

C_lc is a limit-cycle stability penalty: the relative cross-cycle variance of the
lift trace (0 iff every sampled cycle traces the same orbit). It rejects transient
lift, cleans the gradient, and rewards the resonant steady-state gait.

Gradients come from the MJX differentiable rollout via *forward mode*
(jax.jacfwd): with only 2K+1 parameters forward mode is memory-light (constant,
vs reverse mode storing the whole multi-thousand-step trajectory) and keeps the
full-accuracy solver (reverse mode would need the 1-iteration solver). Optimizer
is plain Adam (low lr). A torque-budget curriculum tightens tau_max each stage.

Run on a CUDA box (the 5090):  uv run python opt/optimize_control.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.mjx.rollout import _match_dtypes
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest

WINGTIP_SITES = ("L_wing_aero_blade_04", "R_wing_aero_blade_04")  # L/R tip, full 3D stroke
G = 9.81
TAU_CURRICULUM_NM = (0.300, 0.250, 0.200, 0.150, 0.100)  # 300 -> 250 -> 200 -> 150 -> 100 mNm


_INTEGRATORS = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}


def recorded_peak(values, record):
    """Exact peak over recorded samples; matches CPU validation."""
    return jnp.max(jnp.where(record, values, -jnp.inf))


def recorded_lp(values, record, p):
    """Smooth peak proxy over recorded samples."""
    mask = record.astype(values.dtype)
    n = jnp.maximum(jnp.sum(mask), 1.0)
    return (jnp.sum((jnp.abs(values) ** p) * mask) / n) ** (1.0 / p)


def set_unsteady_aero(enabled: bool):
    """Toggle the added-mass (unsteady) aero term in BOTH the MJX and CPU aero.

    It is a capped finite-difference hack whose JAX (MJX) and numpy (CPU)
    implementations diverge ~4x -- THE source of the MJX<->CPU lift mismatch
    (MJX 371 vs CPU 93 mN with it on; both ~160 mN with it off, ratio 0.995).
    Disabling (clip FD accel to 0) makes MJX a faithful CPU proxy; the
    quasi-steady lift is the transferable quantity. Must be set before the
    rollout is jitted (the caps are captured at trace time).
    """
    import runtime.aero_runtime as ca
    import runtime.mjx.aero_runtime as ma
    o, l = (5.0e4, 1.0e3) if enabled else (0.0, 0.0)
    ca.OMEGA_DOT_CAP, ca.LIN_ACC_CAP = o, l
    ma.OMEGA_DOT_CAP, ma.LIN_ACC_CAP = o, l


def setup(tg_path: Path, scene_path: Path, *, aero_substep: int = 4, integrator: str = "euler",
          phase_lag: float | None = None, timestep_ms: float | None = None):
    """Compile, build a calibrated CPU model/data and the device-side configs.

    ``aero_substep`` matches the CPU runtime (4): aero is recomputed every N
    physics steps and held between, with the FD window = N*dt. The MJX default
    is substep=1 (every-step, 1*dt FD), which gives a much noisier added-mass
    force (RMS ~2 N vs ~0.2 N) and is the main CPU<->MJX lift mismatch.

    ``integrator`` defaults to **euler**: it is the one integrator where MJX and
    CPU MuJoCo agree per-step (implicitfast diverges between the two solvers, so
    designs optimized under implicitfast overfit MJX and fail CPU validation).
    """
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    # compile_tg writes SHARED files (templates/generated-joint/*.xml) regardless of
    # scene_path; concurrent optimizer processes racing here produce empty/partial
    # XML (observed: ParseXML empty _motor_tendon_probe.xml). Serialize compile+load.
    import fcntl
    with open(ROOT / "opt" / ".compile.lock", "w") as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        compile_tg(tg_path, scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = _INTEGRATORS[integrator]
    if timestep_ms is not None:  # override the tg's dt (must precede put_model)
        model.opt.timestep = timestep_ms * 1e-3
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    lag = (load_morphology(tg_path).motors.crank_phase_lag_rad
           if phase_lag is None else phase_lag)
    sync_crank_disks_at_rest(model, data, lag)  # main-dialect: sync applies the lag internally

    control = load_control_for_morphology(tg_path)
    drive = rollout.build_drive_config(model, ramp_duration=0.0)
    pitch = wing_pitch.config_from_runtime(model, centers=rollout._pitch_centers(model, data))
    aero = aero_runtime.build_aero_config(model, substep=aero_substep)
    tip_ids = jnp.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
                           for s in WINGTIP_SITES], dtype=jnp.int32)  # [L_tip, R_tip]

    mx = mjx.put_model(model)
    dx0 = mjx.put_data(model, data)
    tendons = structural_tendon_pairs(model)
    motor_ids = motor_tendon_ids(model)
    return model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids


def structural_tendon_pairs(model, individual: bool = True):
    """Structural tendon indices + base stiffness/rest-length arrays.

    Returns (l_ids, r_ids, base_stiffness(ntendon,), base_lengthspring(ntendon,2)).
    ``individual=True`` (default) drops the L=R mirror so EVERY structural tendon
    is tuned on its own (l_ids = all structural ids, r_ids = same; the mirror
    .set in perturbed_model becomes idempotent) -> 24 tendons * 2 = 48 params.
    ``individual=False`` keeps the old 12 mirror pairs (24 params).
    """
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(model.ntendon)]
    pairs: dict[str, dict[str, int]] = {}
    for i, n in enumerate(names):
        if not n or "motor" in n or n[:2] not in ("L_", "R_"):
            continue
        pairs.setdefault(n[2:], {})[n[0]] = i
    l_ids, r_ids = [], []
    for d in pairs.values():
        if "L" in d and "R" in d:
            l_ids.append(d["L"])
            r_ids.append(d["R"])
    if individual:  # each tendon its own (stiffness, pretension); no L=R mirror
        ids = l_ids + r_ids
        l_ids = r_ids = ids
    return (
        jnp.asarray(l_ids, dtype=jnp.int32),
        jnp.asarray(r_ids, dtype=jnp.int32),
        jnp.asarray(model.tendon_stiffness),
        jnp.asarray(model.tendon_lengthspring),
    )


def motor_tendon_ids(model):
    """The 4 motor-tendon indices (L/R x t1/t4). Tuned with one shared stiffness +
    one shared pretension factor (2 params), like the structural pairs."""
    ids = [i for i in range(model.ntendon)
           if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)) and "motor" in n]
    return jnp.asarray(ids, dtype=jnp.int32)


def wavelet_dictionary(n_atoms):
    """A dyadic real-wavelet (periodic Ricker / Mexican-hat) frame of exactly
    ``n_atoms`` atoms on the crank cycle [0, 2*pi).

    Level L has ``2**L`` equal-spaced translations, each a Ricker atom of scale
    sigma = pi / 2**L (coarse-to-fine). The optimizer learns one amplitude per
    atom. Unlike Fourier harmonics (global sinusoids), each atom is LOCALIZED in
    crank phase -- the whole point of this experiment: can time-local crank-speed
    modulations transfer where global harmonics didn't (methodology sec 13).

    Returns (centers, sigmas, freqs) as float64 numpy arrays. ``freqs`` = 2**L is
    the atom's characteristic frequency, the wavelet analog of harmonic number n
    for the scale-weighted energy regularizer.

    ponytail: the last level is truncated to hit n_atoms exactly (e.g. 6 = 1+2+3
    of the level-2 quartet), leaving one coarse-only phase gap near 3*pi/2. Fine
    for an exploration basis; swap in a full 2^L frame if that gap ever matters.
    """
    centers, sigmas, freqs, level = [], [], [], 0
    while len(centers) < n_atoms:
        n = 2 ** level
        for k in range(n):
            centers.append(2.0 * math.pi * k / n)
            sigmas.append(math.pi / n)
            freqs.append(float(n))
            if len(centers) >= n_atoms:
                break
        level += 1
    return np.asarray(centers), np.asarray(sigmas), np.asarray(freqs)


def ricker_atoms(phase, centers, sigmas, xp):
    """Ricker (Mexican-hat) atom values at scalar ``phase`` for all dictionary
    atoms. ``xp`` is jnp (MJX rollout) or np (CPU validation) -- both share mod/
    exp so the SAME waveform is evaluated on both sides (no basis mismatch).

    d = signed shortest angular distance to each center (periodic); the Gaussian
    envelope makes the antipodal wrap-kink negligible for fine atoms.
    """
    d = xp.mod(phase - centers + math.pi, 2.0 * math.pi) - math.pi
    u = d / sigmas
    return (1.0 - u * u) * xp.exp(-0.5 * u * u)


def build_rollout(mx_base, dx0_base, drive, pitch, aero, tip_ids, omega, K,
                  warmup_steps, sample_steps,
                  *, steps_per_cycle, aero_substep, base_a0, tune_control, control_fixed,
                  tune_frequency, tune_tendons, tendons, tune_pitch=False, tune_motor=False,
                  motor_ids=None, flap_t1_mask=None, lp_power=10.0, avg_periods=3,
                  conv_eps=1e-4, alpha_stall=math.radians(45.0),
                  alpha_smooth=math.radians(3.0), k_max=0.10, k_smooth=0.01,
                  swing_mode="std", stiff_range=None, tune_strokeplane=False,
                  strokeplane=None, wavelet=None):
    """Return metrics(theta) -> (Lbar, tau_hat, A_stroke, S, C_lc) as a jittable fn.

    ``theta`` packs only the *active* groups, concatenated in this order:
      [control(2K+1)]  if tune_control     -- a0_log, a_1..a_K, b_1..b_K
      [f_log(1)]       if tune_frequency   -- log drive-frequency factor (resonance)
      [stiff_log(12)]  if tune_tendons     -- per-pair log stiffness factor
      [pre_log(12)]    if tune_tendons     -- per-pair log pretension (rest-length) factor
    a0 and frequency are **log-factors** (a0 = base_a0*exp(a0_log), w = base_w*exp(f_log))
    so a single learning rate moves them sensibly despite their large absolute scale
    (~63 rad/s, 10 Hz). Tendon params perturb the MJX model arrays directly.
    """
    flap_ids = drive.flap_actuator_ids
    harmonics = jnp.arange(1, K + 1, dtype=jnp.float32)
    if wavelet is not None:  # dyadic Ricker frame reinterprets the 2K control coeffs
        wl_centers = jnp.asarray(wavelet[0], dtype=jnp.float32)
        wl_sigmas = jnp.asarray(wavelet[1], dtype=jnp.float32)
        wl_freqs = jnp.asarray(wavelet[2], dtype=jnp.float32)
    l_ids, r_ids, base_stiff, base_lenspring = tendons
    n_tend = int(l_ids.shape[0])  # tuned structural-tendon groups (48 params = 2*n_tend)
    total_steps = warmup_steps + sample_steps
    n_full = max(1, total_steps // steps_per_cycle)
    n_avg = max(1, min(int(avg_periods), n_full))
    window_start = (n_full - n_avg) * steps_per_cycle
    window_end = n_full * steps_per_cycle
    record = (jnp.arange(total_steps) >= window_start) & (jnp.arange(total_steps) < window_end)
    qdim = int(dx0_base.qpos.shape[0])
    if tune_strokeplane:
        sp_qadr = int(strokeplane["qadr"])
        sp_body_ids = strokeplane["body_ids"]
        sp_tilt_signs = strokeplane["tilt_signs"]
        sp_tilt0, sp_pitch0 = float(strokeplane["tilt0"]), float(strokeplane["pitch0"])
        sp_max = float(strokeplane["max_rad"])

    def split(theta):
        i = 0
        if tune_control:
            ctrl = theta[i : i + 2 * K + 1]
            i += 2 * K + 1
        else:
            ctrl = control_fixed
        f_log = theta[i] if tune_frequency else 0.0
        if tune_frequency:
            i += 1
        stiff_log = pre_log = pitch_log = motor_log = None
        if tune_tendons:
            stiff_log = theta[i : i + n_tend]
            pre_log = theta[i + n_tend : i + 2 * n_tend]
            i += 2 * n_tend
        if tune_pitch:  # 4: up-deadband, down-deadband, up-stiffness, down-stiffness
            pitch_log = theta[i : i + 4]
            i += 4
        if tune_motor:  # 2: shared motor-tendon stiffness, pretension
            motor_log = theta[i : i + 2]
            i += 2
        sp = theta[i : i + 10] if tune_strokeplane else None
        return ctrl, f_log, stiff_log, pre_log, pitch_log, motor_log, sp

    def perturbed_model(mxb, stiff_log, pre_log, motor_log):
        # Log-factors (seeded exp(0)=1) scale the design base; exp keeps values >0.
        # The base comes off `mxb`, not the captured arrays, so a per-genome model
        # (evolve_codesign's geometry genes) scales ITS OWN derived rest lengths --
        # tendon `length = derived` moves when the struts do.
        stiff, lenspring = mxb.tendon_stiffness, mxb.tendon_lengthspring
        base_stiff, base_lenspring = stiff, lenspring
        if tune_tendons:  # per-pair structural, mirrored L=R
            sp = base_stiff[l_ids] * jnp.exp(stiff_log)
            stiff = stiff.at[l_ids].set(sp).at[r_ids].set(sp)
            lpl = base_lenspring[l_ids] * jnp.exp(pre_log)[:, None]
            lpr = base_lenspring[r_ids] * jnp.exp(pre_log)[:, None]
            lenspring = lenspring.at[l_ids].set(lpl).at[r_ids].set(lpr)
        if tune_motor:  # one shared stiffness + one shared pretension over the 4 motor tendons
            stiff = stiff.at[motor_ids].set(base_stiff[motor_ids] * jnp.exp(motor_log[0]))
            lenspring = lenspring.at[motor_ids].set(base_lenspring[motor_ids] * jnp.exp(motor_log[1]))
        return mxb.replace(tendon_stiffness=stiff, tendon_lengthspring=lenspring)

    def metrics(theta, mx_g=None, dx_g=None):
        """``mx_g``/``dx_g`` override the compiled-in model for THIS call. Both
        default to the model this rollout was built against, so every existing
        caller is unchanged; evolve_codesign passes a per-genome model (same
        array shapes, so one trace serves the whole batched vmap)."""
        mx, dx0 = (mx_base if mx_g is None else mx_g), (dx0_base if dx_g is None else dx_g)
        ctrl, f_log, stiff_log, pre_log, pitch_log, motor_log, sp = split(theta)
        a0 = base_a0 * jnp.exp(ctrl[0])
        a, b = ctrl[1 : 1 + K], ctrl[1 + K :]
        wl_coeff = ctrl[1:]  # wavelet mode: all 2K coeffs are atom amplitudes
        omega_eff = omega * jnp.exp(f_log)
        mx_eff = (perturbed_model(mx, stiff_log, pre_log, motor_log)
                  if (tune_tendons or tune_motor) else mx)
        # Pitch deadband+stiffness as log-factors of the design base (resonant feathering).
        pc = pitch
        if tune_pitch:
            pc = pitch._replace(
                upstroke_pitch=pitch.upstroke_pitch * jnp.exp(pitch_log[0]),
                downstroke_pitch=pitch.downstroke_pitch * jnp.exp(pitch_log[1]),
                upstroke_stiffness=pitch.upstroke_stiffness * jnp.exp(pitch_log[2]),
                downstroke_stiffness=pitch.downstroke_stiffness * jnp.exp(pitch_log[3]),
            )

        def step(carry, rec):
            dx, state, cq, cnfw, ca, cu, i = carry
            mx_step = mx_eff
            if tune_strokeplane:
                # Stroke-plane signal over crank phase (L_t1 hinge angle): per-DOF
                # mean + 2 harmonics, tanh-bounded to +/-sp_max rad around the
                # compiled mount angles. body_quat is re-derived every mjx.step,
                # so the whole flap arc follows the new plane, differentiably.
                th = dx.qpos[sp_qadr]
                basis = jnp.stack([jnp.ones_like(th), jnp.cos(th), jnp.sin(th),
                                   jnp.cos(2.0 * th), jnp.sin(2.0 * th)])
                tilt = sp_tilt0 + sp_max * jnp.tanh(jnp.dot(sp[:5], basis) / sp_max)
                pitv = sp_pitch0 + sp_max * jnp.tanh(jnp.dot(sp[5:], basis) / sp_max)
                # euler "xyz" with z=0 -> quat = qx(tilt) * qy(pitch), wxyz; tilt
                # sign-mirrored on R (matches compile.py _mirror_shoulder).
                hx, hy = 0.5 * tilt * sp_tilt_signs, 0.5 * pitv
                cx, sx, cy, sy = jnp.cos(hx), jnp.sin(hx), jnp.cos(hy), jnp.sin(hy)
                quats = jnp.stack([cx * cy, sx * cy, cx * sy, sx * sy], axis=1)
                mx_step = mx_eff.replace(body_quat=mx_eff.body_quat.at[sp_body_ids].set(
                    quats.astype(mx_eff.body_quat.dtype)))
            phase = omega_eff * dx.time
            # a,b are modulation depths (fractions of a0) so one LR moves them with
            # real authority; absolute harmonic n is a0*a_n. cmd = a0*(1 + sum(...)).
            if wavelet is not None:
                mod = jnp.sum(wl_coeff * ricker_atoms(phase, wl_centers, wl_sigmas, jnp))
            else:
                mod = jnp.sum(a * jnp.cos(harmonics * phase) + b * jnp.sin(harmonics * phase))
            cmd = a0 * (1.0 + mod)
            ctrl_vec = jnp.zeros_like(dx.ctrl).at[flap_ids].set(cmd)
            dx = dx.replace(ctrl=ctrl_vec, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
            dx = wing_pitch.apply(dx, pc)  # pitch every step (matches CPU)
            # Aero recompute-and-hold every `aero_substep` steps; aero_state (FD prev
            # velocity) only advances on recompute, so the window spans substep*dt.
            def recompute(_):
                q, nfw, ns, alpha, speed = aero_runtime.aero_wrench_with_diagnostics(mx_step, dx, aero, state)
                return q, nfw, ns, alpha, speed
            def hold(_):
                return cq, cnfw, state, ca, cu
            qfrc, net_force_world, state, alpha, speed = jax.lax.cond(
                (i % aero_substep) == 0, recompute, hold, operand=None)
            dx = dx.replace(qfrc_applied=dx.qfrc_applied + qfrc)
            dx = mjx.step(mx_step, dx)
            carry = _match_dtypes((dx, state, qfrc, net_force_world, alpha, speed, i + 1), carry)
            tau = jnp.max(jnp.abs(dx.actuator_force[flap_ids]))
            # Motor mechanical power per crank actuator: P = force*velocity.
            # P>0 motor drives (supplies), P<0 motor brakes (absorbs = inactive).
            pm = dx.actuator_force[flap_ids] * dx.actuator_velocity[flap_ids]
            power = jnp.sum(pm)                              # net (for the braking fraction)
            p_sup = jnp.sum(jnp.maximum(0.0, pm))            # supplied/motoring power [W]
            tips = dx.site_xpos[tip_ids].reshape(-1)  # [Lx,Ly,Lz, Rx,Ry,Rz] -- full 3D stroke
            # Per-motor |torque|: group A = t1 ("blue"/upstroke), B = t4 ("orange"/
            # downstroke). Used for the anti-overlap penalty below.
            absf = jnp.abs(dx.actuator_force[flap_ids])
            tauA = jnp.sum(absf * flap_t1_mask)
            tauB = jnp.sum(absf * (1.0 - flap_t1_mask))
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
        # Per-cycle exact torque peaks over the sample window. tau_peak (max over
        # cycles) is the honest metric CPU certification measures; tau_drift
        # (last-minus-first cycle peak) exposes slow torque transients/beats that a
        # short window would otherwise sample as fake gains. tau_hat (the gradient
        # surrogate) is the soft-max over per-cycle Lp: unlike a whole-window Lp it
        # cannot be lowered by narrowing the spike duty cycle across the window or
        # by parking one good cycle in it.
        whole_cycles = (window_end <= total_steps
                        and (window_end - window_start) == n_avg * steps_per_cycle)
        if whole_cycles:
            tau_cyc = tau[window_start:window_end].reshape(n_avg, steps_per_cycle)
            pc_exact = jnp.max(tau_cyc, axis=1)
            tau_peak = jnp.max(pc_exact)
            tau_drift = pc_exact[-1] - pc_exact[0]
            pc_lp = jnp.mean(tau_cyc ** lp_power, axis=1) ** (1.0 / lp_power)
            tau_hat = _soft_extreme(pc_lp, 300.0)
        else:  # window shorter than one cycle (smoke tests): whole-window fallback
            tau_hat = recorded_lp(tau, record, lp_power)
            tau_peak = recorded_peak(tau, record)
            tau_drift = jnp.array(0.0)
        # 3D stroke amplitude: the wingtip sweeps a tilted arc, so measure the full
        # extent, not just z. Per wing, the bounding-box diagonal of the tip path
        # over the sample window (sqrt of summed per-axis peak-to-peak). Take the MIN
        # over the two wings so the floor preserves BOTH (tendons are un-mirrored, so
        # one wing could otherwise collapse while the other compensates).
        tip_w = tips[window_start:window_end]                    # (sample, wing, xyz)
        ptp = jnp.max(tip_w, axis=0) - jnp.min(tip_w, axis=0)    # (wing, xyz)
        A_stroke = jnp.min(jnp.sqrt(jnp.sum(ptp**2, axis=1)))    # weaker wing's 3D extent [m]
        # Energy regularizer: penalize fine scales (high freq) harder, exactly as
        # harmonic order n does for Fourier -- that is where dt-tricks would hide.
        if wavelet is not None:
            S = jnp.sum(wl_freqs**2 * wl_coeff**2)
        else:
            S = jnp.sum(harmonics**2 * (a**2 + b**2))

        # Limit-cycle stability: cross-cycle variance of the lift trace. Reshape
        # the sample window to (n_full cycles, steps_per_cycle); var across cycles
        # at each phase is 0 iff every cycle traces the same orbit (stable limit
        # cycle). Needs >=2 whole cycles, else disabled.
        if n_full >= 2:
            lift_cyc = lift_z[window_start:window_end].reshape(n_avg, steps_per_cycle)
            # Relative cross-cycle variance: dimensionless, scale-free, ~0 for a
            # stable orbit, ~O(1) when cycles disagree. (Normalizing by L_target^2
            # over-weights it, since instantaneous lift swings >> mean lift.)
            C_lc = jnp.mean(jnp.var(lift_cyc, axis=0)) / (jnp.mean(lift_cyc**2) + 1e-12)
        else:
            C_lc = jnp.array(0.0)

        # Inactive (reactive/braking) motor power over the sample window: the part
        # the motor must ABSORB (P<0). Minimizing it pushes elastic tendon storage
        # to handle deceleration instead of the motor (resonance). Normalized by
        # the apparent power scale so it's ~O(1).
        p_s = jnp.where(record, power, 0.0)
        P_brake = jnp.sum(jnp.maximum(0.0, -p_s)) / n_win          # mean absorbed power [W]
        P_scale = jnp.sum(jnp.abs(p_s)) / n_win + 1e-9             # mean |power|
        P_inactive = P_brake / P_scale                            # fraction of power that's braking
        # Mean SUPPLIED (motoring) motor power over the sample window [W] -- the real
        # energy cost. The primary efficiency objective now (subsumes braking power: a
        # gait that brakes then re-supplies pays for the re-supply here).
        P_supplied = jnp.sum(jnp.where(record, p_sup, 0.0)) / n_win

        # Anti-overlap of the two motors (t1 "blue"/upstroke vs t4 "orange"/
        # downstroke): uncentered cosine similarity of their |torque| traces over
        # the sample window. 1 => both peak together (wasted simultaneous effort,
        # high inactive power); 0 => each peaks in the other's trough. Minimizing
        # it staggers the two motors. Signals are >=0 so it's in [0,1] and smooth.
        a_s = jnp.where(record, tauA, 0.0)
        b_s = jnp.where(record, tauB, 0.0)
        C_overlap = jnp.sum(a_s * b_s) / (jnp.sqrt(jnp.sum(a_s**2) * jnp.sum(b_s**2)) + 1e-9)

        # Torque swing over the converged window, not peak-to-peak.
        tau_win = tau[window_start:window_end]
        if swing_mode == "slew":
            # RMS torque slew rate, normalized by the drive frequency so it lands in
            # torque units (mean |dtau/dphase|): penalizes jerk, not just spread --
            # a flat-but-jerky profile and a smooth one have equal std but not equal slew.
            dt_s = (2.0 * math.pi / omega) / steps_per_cycle
            swing = jnp.sqrt(jnp.mean((jnp.diff(tau_win) / dt_s) ** 2)) / omega
        else:
            swing = jnp.sqrt(jnp.mean((tau_win - jnp.mean(tau_win)) ** 2))

        alpha_pk = recorded_lp(alpha_trace, record, lp_power)
        Ubar = jnp.sum(speed_trace * mask) / n_win + 1e-9
        k_eff = omega_eff * jnp.mean(aero.chords) / (2.0 * Ubar)
        C_valid = ((_soft_hinge(alpha_pk - alpha_stall, alpha_smooth) ** 2) / (alpha_stall ** 2)
                   + (_soft_hinge(k_eff - k_max, k_smooth) ** 2) / (k_max ** 2))
        if n_full >= 2:
            qcyc = qpos[:n_full * steps_per_cycle].reshape(n_full, steps_per_cycle, qdim)
            conv_err = jnp.sqrt(jnp.mean((qcyc[-1] - qcyc[-2]) ** 2))
        else:
            conv_err = jnp.array(0.0)
        return (Lbar, tau_hat, A_stroke, S, C_lc, P_inactive, C_overlap,
                swing, P_supplied, C_valid, alpha_pk, k_eff, conv_err,
                tau_peak, tau_drift)

    return metrics


def cpu_validate_design(scene_path, ctrl_params, tendon_arrays, integrator,
                        *, pitch_params=None, strokeplane_params=None, periods=3.0,
                        warmup=2.0, phase_lag=0.0, timestep_ms=None, aero_substep=None,
                        cycle_stats=False, wavelet=None):
    """Run the design on MuJoCo CPU (numpy aero) -> (mean_lift, peak_torque, mean_supplied_power).

    Uses the SAME compiled scene and the SAME tendon arrays the MJX optimizer
    used (no .tg round-trip), so any MJX<->CPU difference is purely the physics
    rollout, not a remodeling artifact. ``ctrl_params`` = (a0, a, b, omega) with
    a,b modulation depths (cmd = a0*(1 + sum a_n cos + b_n sin)).
    """
    from runtime.aero_runtime import AeroBladeRuntime
    from runtime.viewer_runtime import get_actuator_id
    from runtime.wing_pitch_limit import WingPitchLimiter

    a0, a, b, omega = ctrl_params
    a = np.asarray(a); b = np.asarray(b); harm = np.arange(1, len(a) + 1)
    if wavelet is not None:  # a = all wavelet atom amplitudes; b unused
        wl_centers, wl_sigmas = np.asarray(wavelet[0]), np.asarray(wavelet[1])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = _INTEGRATORS[integrator]
    if timestep_ms is not None:
        model.opt.timestep = timestep_ms * 1e-3
    if tendon_arrays is not None:
        stiff, lenspring = tendon_arrays
        model.tendon_stiffness[:] = np.asarray(stiff)
        model.tendon_lengthspring[:] = np.asarray(lenspring)
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, phase_lag)
    aero = AeroBladeRuntime(model, substep=aero_substep)
    pitch = WingPitchLimiter(model, data)
    if pitch_params is not None:  # apply the optimized feathering params (up/down deadband, stiffness)
        pitch.upstroke_pitch, pitch.downstroke_pitch, pitch.upstroke_stiffness, pitch.downstroke_stiffness = (
            float(x) for x in pitch_params)
    aids = [get_actuator_id(model, n) for n in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    disk_orient = None
    if strokeplane_params is not None:  # same signal the MJX rollout applied
        from runtime.disk_orientation import DiskOrientation
        sp_coef, sp_tilt0, sp_pitch0, sp_max = strokeplane_params
        sp_coef = np.asarray(sp_coef, dtype=float)
        disk_orient = DiskOrientation(model)
        sp_qadr = int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_t1_disk_hinge")])

    def step():
        if disk_orient is not None:
            th = float(data.qpos[sp_qadr])
            basis = np.array([1.0, math.cos(th), math.sin(th),
                              math.cos(2.0 * th), math.sin(2.0 * th)])
            disk_orient.set(
                sp_tilt0 + sp_max * math.tanh(float(sp_coef[:5] @ basis) / sp_max),
                sp_pitch0 + sp_max * math.tanh(float(sp_coef[5:] @ basis) / sp_max))
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        phase = omega * data.time
        if wavelet is not None:
            mod = float(np.sum(a * ricker_atoms(phase, wl_centers, wl_sigmas, np)))
        else:
            mod = float(np.sum(a * np.cos(harm * phase) + b * np.sin(harm * phase)))
        cmd = a0 * (1.0 + mod)
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = cmd
        lift = sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings)
        mujoco.mj_step(model, data)
        pm = data.actuator_force[aids] * data.actuator_velocity[aids]
        return (lift, float(np.max(np.abs(data.actuator_force[aids]))),
                float(np.sum(np.maximum(0.0, pm))))          # mean supplied power [W]

    period = math.tau / omega
    for _ in range(int(warmup * period / model.opt.timestep)):
        step()
    rec = [step() for _ in range(int(periods * period / model.opt.timestep))]
    L = np.array([r[0] for r in rec]); T = np.array([r[1] for r in rec]); P = np.array([r[2] for r in rec])
    if cycle_stats:
        # Per-cycle exact peaks over the sample window -- lets callers gate on
        # stationarity (a slow beat shows as a drifting per-cycle peak series).
        spc = max(1, int(round(period / model.opt.timestep)))
        ncyc = max(1, len(T) // spc)
        pc = [float(np.max(T[i * spc:min((i + 1) * spc, len(T))])) for i in range(ncyc)]
        return float(L.mean()), float(T.max()), float(P.mean()), pc
    return float(L.mean()), float(T.max()), float(P.mean())


def _soft_hinge(x, s):
    """Smooth max(0, x): softplus(x/s)*s. C-infinity, no kink at 0."""
    return jax.nn.softplus(x / s) * s


def _soft_extreme(values, beta, sign=1.0):
    """Smooth max/min with softmax weights; log-mean avoids a batch-size offset."""
    v = jnp.asarray(values)
    b = jnp.asarray(beta, dtype=v.dtype)
    n = jnp.log(jnp.asarray(v.size, dtype=v.dtype))
    return sign * (jax.nn.logsumexp(sign * b * v) - n) / b


def make_loss(metrics, *, lift_floor, lift_cap, lift_smooth, lift_above_weight,
              A_target, tau_max, weights, w_lc, w_power, w_overlap, w_swing,
              lift0=None, tradeoff=None, metrics_impl=None, metrics_rk4=None, w_gap=0.0,
              amp_band_pct=0.02, amp_floor=None, extra_metrics=(), w_valid=0.0,
              w_drift=0.0):
    """Scalar loss = -R, normalized per the design spec, with metrics as aux.

    Objective: hold lift above a floor, preserve the flap amplitude (``wK``, the
    kicker), and minimize peak motor torque AND torque swing. Integrator-hacking is
    fought in the SELECTION loop by the cross-integrator transfer gate (--cpu-every /
    --cpu-gap): the euler gradient can still find euler-fragile designs, but only ones
    whose euler and implicitfast peaks agree are eligible to win.
      * Lift: one-sided floor only (penalize below ``lift_floor``). Overshoot is free.
      * Stroke amplitude: STRICT two-sided band around baseline A_target, +/-amp_band_pct
        (``wK``) -- both shrinking AND overshooting the flap are penalized.
      * Peak torque: ``(tau/tau_max)^2`` + 4x extra above the ceiling (``wtau``).
      * Torque swing: ``(swing/tau_max)^2`` (``w_swing``) -- a flat low-torque profile,
        not just a low peak.
      * Inactive power: ``w_power`` on the braking-power fraction; harmonics: ``ws``.
    (Supplied power is still measured and logged, just not optimized here.) All terms
    are smooth (no hard max(0,.) walls) so gradients flow.
    """
    wL, wtau, wK, ws = weights
    if amp_floor is not None:
        # One-sided amplitude FLOOR [m], absolute (not anchored to the seed's A*,
        # which may itself be collapsed): shrink below it is walled off, growth is
        # free. Tolerance-normalized like the lift floor: penalty reaches wK at one
        # smoothing width (2% of the floor) short.
        amp_lo, amp_hi = amp_floor, 1e9
        A_smooth = max(1e-9, 0.02 * amp_floor)
    else:
        amp_lo = A_target * (1.0 - amp_band_pct)
        amp_hi = A_target * (1.0 + amp_band_pct)
        A_smooth = max(1e-9, 0.2 * amp_band_pct * A_target)  # smoothing scale inside the band

    def loss(theta, tau_max, robust_beta):
        base_metrics = metrics(theta)
        (Lbar, tau_hat, A_stroke, S, C_lc, P_inactive, C_overlap, swing, P_supplied,
         C_valid, alpha_pk, k_eff, conv_err, tau_peak, tau_drift) = base_metrics
        # Roll the SAME design under the validation variants and optimize the robust
        # envelope. Timestep variants used to affect torque only, which still allowed
        # the optimizer to hide lift/amplitude/swing failures off the trained dt.
        impl_metrics = metrics_impl(theta) if metrics_impl is not None else base_metrics
        rk4_metrics = metrics_rk4(theta) if metrics_rk4 is not None else base_metrics
        variant_metrics = (base_metrics, impl_metrics, rk4_metrics) + tuple(m(theta) for m in extra_metrics)
        tau_i = impl_metrics[1]
        tau_r = rk4_metrics[1]
        taus = jnp.stack([m[1] for m in variant_metrics])
        lifts = jnp.stack([m[0] for m in variant_metrics])
        amps = jnp.stack([m[2] for m in variant_metrics])
        swings = jnp.stack([m[7] for m in variant_metrics])
        validities = jnp.stack([m[9] for m in variant_metrics])
        tau_hard = jnp.max(taus)
        tau_used = _soft_extreme(taus, robust_beta)
        tau_spread = tau_hard - jnp.min(taus)
        lift_used = _soft_extreme(lifts, robust_beta, sign=-1.0)
        lift_high = _soft_extreme(lifts, robust_beta)
        amp_low = _soft_extreme(amps, robust_beta, sign=-1.0)
        amp_high = _soft_extreme(amps, robust_beta)
        swing_used = _soft_extreme(swings, robust_beta)
        C_valid_used = _soft_extreme(validities, robust_beta)
        Pgap = w_gap * (tau_spread / tau_max) ** 2   # cross-integrator disagreement penalty
        # Torque swing (normalized by the stage ceiling): a flat low-torque profile
        # beats a low-peak-but-violently-oscillating one.
        Pswing = (swing_used / tau_max) ** 2
        # Strict band: penalize A_stroke outside [amp_lo, amp_hi] on EITHER side.
        amp_short = _soft_hinge(amp_lo - amp_low, A_smooth)
        amp_over = _soft_hinge(amp_high - amp_hi, A_smooth)
        if amp_floor is not None:
            Ckin = amp_short**2 / A_smooth**2  # wall, not the weak /A_target**2 slope
        else:
            Ckin = (amp_short**2 + amp_over**2) / A_target**2
        Clc = C_lc  # already dimensionless (relative cross-cycle lift variance)
        # Exact worst-case peak + worst |per-cycle drift| across the batch variants.
        # The drift penalty prices beat/transient-riding in the GRADIENT itself:
        # short training windows show only a few mNm of drift even on a strong
        # beat, so it is linear in |drift| (not squared) to keep real authority
        # at that scale, while honest designs (~0.2 mNm) pay ~nothing.
        tau_peak_hard = jnp.max(jnp.stack([m[13] for m in variant_metrics]))
        drift_hard = jnp.max(jnp.stack([jnp.abs(m[14]) for m in variant_metrics]))
        P_drift = w_drift * drift_hard / tau_max
        if tradeoff is not None:
            # Controlled lift<->torque trade: a `tradeoff`% peak-torque cut is worth a
            # 1% lift drop below baseline; the reverse is gentler: a `tradeoff`%
            # lift gain pays for a 1% torque increase.
            Ptau = tau_used / tau_max                                  # ~1 at baseline (robust)
            l_smooth = max(1e-9, 0.02 * abs(lift0))
            Pdrop = _soft_hinge(lift0 - lift_used, l_smooth) / (abs(lift0) + 1e-9)
            Pgain = jnp.maximum(0.0, lift_used - lift0) / (abs(lift0) + 1e-9)
            R = (-wtau * Ptau - wtau * tradeoff * Pdrop + wtau * Pgain / tradeoff
                 - w_swing * Pswing - wK * Ckin - ws * S
                 - w_lc * Clc - w_power * P_inactive - w_overlap * C_overlap
                 - w_valid * C_valid_used - Pgap - P_drift)
        else:
            # Lift: strict floor only (optional moderate above-cap if lift_cap is set).
            below = _soft_hinge(lift_floor - lift_used, lift_smooth)
            above = _soft_hinge(lift_high - lift_cap, lift_smooth) if lift_cap is not None else 0.0
            # Tolerance-normalized (the run-9 hinge lesson): /floor**2 made a 16%
            # lift shortfall cost ~0.03*wL while the tau quadratic paid ~0.6*wtau,
            # so the optimizer dove 60 mN under the floor and got gate-rejected.
            # Normalized by lift_smooth, the penalty hits wL at ONE smoothing width
            # (~15 mN) short -- a wall with a thin soft boundary layer.
            PhiL = -(below**2 + lift_above_weight * above**2) / max(lift_smooth, 1e-9)**2
            # Peak torque (ROBUST), normalized by the stage ceiling, +4x above it (soft hinge).
            over = _soft_hinge(tau_used - tau_max, 0.02 * tau_max)
            Ptau = (tau_used**2 + 4.0 * over**2) / tau_max**2
            R = (wL * PhiL - wtau * Ptau - w_swing * Pswing - wK * Ckin - ws * S
                 - w_lc * Clc - w_power * P_inactive - w_overlap * C_overlap
                 - w_valid * C_valid_used - Pgap - P_drift)
        aux = (Lbar, tau_hat, A_stroke, S, Clc, P_inactive, C_overlap, swing, P_supplied, -R,
               tau_i, tau_r, tau_hard, tau_spread, C_valid, alpha_pk, k_eff, conv_err,
               tau_peak_hard, drift_hard)
        return -R, aux

    return jax.jit(jax.jacfwd(loss, argnums=0, has_aux=True))  # (grad, aux) in one fwd sweep


def load_control_seed(path: Path | None, K: int, drive, base_a0: float,
                      wavelet: bool = False) -> list[float]:
    """Control seed [a0_log, a_1..a_K, b_1..b_K]; a0 stored as log-factor vs base_a0.

    In wavelet mode the 2K coeffs are atom amplitudes seeded to 0 (pure sinusoid =
    the fair starting waveform), so warm-starting off a Fourier design keeps a0 and
    drops its harmonics.
    """
    if path is None:
        a, b, a0 = list(drive.a), list(drive.b), float(drive.a0)
    else:
        d = json.loads(Path(path).read_text())
        a, b, a0 = list(d["a"]), list(d["b"]), float(d["a0"])
    if wavelet:
        return [math.log(a0 / base_a0)] + [0.0] * (2 * K)
    a = (a + [0.0] * K)[:K]
    b = (b + [0.0] * K)[:K]
    # harmonics stored as modulation depths (fraction of a0); design's a_n=0 -> 0.
    a = [x / base_a0 for x in a]
    b = [x / base_a0 for x in b]
    return [math.log(a0 / base_a0)] + a + b  # same sign as base_a0 -> ratio > 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path, default=ROOT / "design" / "baseline_shoulder.tg")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--harmonics", type=int, default=5, help="K Fourier harmonics")
    p.add_argument("--wavelet", action="store_true",
                   help="reinterpret the 2K control coeffs as a dyadic Ricker WAVELET frame "
                        "(localized crank-speed atoms) instead of K Fourier harmonics. Same "
                        "genome width (2K+1), same robustness/drift machinery; warm-starts "
                        "clean from a Fourier sinusoid (all atom amplitudes -> 0).")
    p.add_argument("--lift-floor", type=float, default=0.300,
                   help="soft floor: penalize mean lift below this [N] (smooth softplus hinge)")
    p.add_argument("--lift-cap", type=float, default=None,
                   help="optional upper cap [N]; OFF by default (overshooting lift is free). "
                        "Set it to re-enable a moderate above penalty at --lift-above-weight")
    p.add_argument("--lift-above-weight", type=float, default=0.25,
                   help="relative weight on the above-cap lift penalty (only used if --lift-cap set)")
    p.add_argument("--lift-smooth", type=float, default=0.015,
                   help="softplus smoothing scale [N] for the lift floor (larger = softer, more gradient)")
    p.add_argument("--warmup-periods", type=float, default=8.0,
                   help="settling periods before sampling (the rig keeps building lift for ~8 "
                        "cycles; 5 samples an unsettled transient, well below the true steady state)")
    p.add_argument("--sample-periods", type=float, default=3.0)
    p.add_argument("--aero-substep", type=int, default=4,
                   help="recompute aero every N steps, hold between (CPU parity; 1 = noisy every-step)")
    p.add_argument("--integrator", choices=list(_INTEGRATORS), default="euler",
                   help="physics integrator; euler is the MJX<->CPU-matching choice (default)")
    p.add_argument("--timestep", type=float, default=None,
                   help="override physics timestep [ms] (default: the tg's, 0.13 ms). Shorter "
                        "dt improves euler stability -- pairs with --integrator euler")
    p.add_argument("--tau-min", type=float, default=0.100,
                   help="tightest torque-budget stage [Nm]; curriculum 0.30->0.25->0.20->0.15->0.10 "
                        "clamped to this. Stage 5 (100 mNm) is the demo target")
    p.add_argument("--added-mass", action="store_true",
                   help="enable the unsteady added-mass aero term. Now transfers MJX<->CPU "
                        "(ratio ~1.01) after the mj_jacSite velocity fix; off by default")
    p.add_argument("--iters-per-stage", type=int, default=40)
    p.add_argument("--cpu-every", type=int, default=0,
                   help="every N iters, re-run the CURRENT design on MuJoCo CPU (numpy aero) under "
                        "BOTH euler and implicitfast. 0 = off. Tracks a CPU-selected best scored on "
                        "the ROBUST (worse-integrator) peak torque among designs that TRANSFER "
                        "(see --cpu-gap), saved to <out>/cpu_best/ -- this is the integrator gate")
    p.add_argument("--cpu-gap", type=float, default=0.15,
                   help="transfer gate for --cpu-every: reject a design as cpu_best if euler and "
                        "implicitfast peak torque disagree by more than this fraction (default 0.15). "
                        "A large gap means the gait is exploiting the euler integrator")
    p.add_argument("--tri-integrator", action="store_true",
                   help="INTEGRATOR-AGNOSTIC reward: run euler + implicitfast + rk4 MJX rollouts "
                        "(all differentiable) each step and optimize the ROBUST peak torque "
                        "max(tau_e,tau_i,tau_r) plus a spread penalty (--w-gap). Makes integrator-"
                        "hacking unrewarding in the gradient itself, not just at selection. ~3x cost")
    p.add_argument("--w-gap", type=float, default=15.0,
                   help="weight on the cross-integrator torque spread penalty ((max-min)/tau_max)^2 "
                        "when --tri-integrator is set")
    p.add_argument("--tri-timestep", action="store_true",
                   help="TIMESTEP-AGNOSTIC reward (Fix 3, EXPERIMENT_LOG.org): also roll out euler "
                        "at dt/2 and 2*dt (differentiable in MJX) and fold their peak torque into the "
                        "same robust max + spread penalty as --tri-integrator. Confirmed need: seed 1 "
                        "transferred across integrators at dt=0.06ms but degraded 10-30%% off that exact "
                        "dt. Stacks with --tri-integrator (5-way robust max: 3 integrators x baseline dt, "
                        "plus euler @ dt/2, euler @ 2dt). ~2x extra cost on top of --tri-integrator")
    p.add_argument("--tri-timestep-factors", type=float, nargs="*", default=None,
                   help="dt factors used by --tri-timestep in the differentiable loss. Default: 0.5 2.0")
    p.add_argument("--rotate-batch", type=int, default=0, metavar="K",
                   help="ROTATING robust mini-batches: each iteration draws one of --n-batches "
                        "precompiled K-variant batches (variants = 3 integrators x --dt-bank) and "
                        "optimizes the robust envelope over just that batch (same make_loss math as "
                        "--tri-integrator, which this replaces). Stochastic cover of the full "
                        "integrator x dt matrix at ~K rollouts/iter instead of the whole matrix")
    p.add_argument("--dt-bank", type=float, nargs="*", default=None, metavar="MS",
                   help="training timestep grid [ms] for --rotate-batch (default: the run dt only)")
    p.add_argument("--n-batches", type=int, default=6,
                   help="number of precompiled rotating mini-batches for --rotate-batch")
    p.add_argument("--batch-seed", type=int, default=0,
                   help="RNG seed for mini-batch composition and rotation order")
    p.add_argument("--dt-jitter", type=float, default=0.0,
                   help="rotate mode: pre-sample each training batch dt as center*(1+u), "
                        "u~Uniform(-jitter,+jitter). Full-matrix checks stay on centers")
    p.add_argument("--precision-perturb", type=float, default=0.0,
                   help="rotate mode: add one differentiable noisy-state rollout per batch "
                        "with qvel noise sigma; cheap MJX/CPU sensitivity proxy")
    p.add_argument("--softmax-beta-start", type=float, default=50.0,
                   help="initial beta [1/unit] for smooth robust max/min aggregation")
    p.add_argument("--softmax-beta-end", type=float, default=300.0,
                   help="final beta [1/unit] for smooth robust max/min aggregation")
    p.add_argument("--full-check-every", type=int, default=25,
                   help="rotate mode: every N iters evaluate the FULL integrator x dt-bank matrix "
                        "on MJX; track a transfer-gated robust best (saved to <out>/robust_best)")
    p.add_argument("--full-gap", type=float, default=0.15,
                   help="transfer gate for the full-matrix check: max fractional peak-torque "
                        "spread across the whole matrix for a design to be eligible")
    p.add_argument("--full-sample-periods", type=float, default=None,
                   help="sample window [periods] for full-matrix checks and the MJX holdout "
                        "(default: --sample-periods). Set >= one torque-beat period (~24) so "
                        "slow transients can't hide from the gate")
    p.add_argument("--cpu-sample-periods", type=float, default=None,
                   help="sample window [periods] for the CPU gates and CPU holdout "
                        "(default: --sample-periods)")
    p.add_argument("--score-swing", type=float, default=1.0,
                   help="weight of the swing term in candidate SELECTION scores (loss keeps "
                        "--w-swing). Set 0 to rank promotions purely on certified worst-case "
                        "peak torque -- the right KPI for a current-limited motor")
    p.add_argument("--w-drift", type=float, default=0.0,
                   help="weight on the per-cycle peak-torque drift penalty in the LOSS "
                        "(linear in worst |drift|/tau_max across batch variants). Makes "
                        "beat-riding expensive in the gradient itself instead of only at "
                        "the selection gates; ~20 gives it real authority")
    p.add_argument("--drift-gate", type=float, default=0.03,
                   help="stationarity gate: reject a candidate when any gate variant's "
                        "per-cycle peak-torque drift (last minus first sampled cycle) exceeds "
                        "this fraction of the robust peak -- catches beat-transient parking, "
                        "where a 3-cycle window reports the minimum of a ~40-cycle beat")
    p.add_argument("--warmup-bank", type=float, nargs="*", default=None, metavar="P",
                   help="rotate mode: per-batch-member warmup periods sampled from this set "
                        "(e.g. 8 16 24 32). Randomizing where the sample window lands prices "
                        "slow torque beats into the robust envelope; default: --warmup-periods")
    p.add_argument("--holdout-dts", type=float, nargs="*", default=(), metavar="MS",
                   help="UNSEEN timesteps [ms] evaluated only at the END on the winner (3 "
                        "integrators on MJX + euler/implicitfast on CPU). Never trained on -- "
                        "passing these means the design learned physics, not a dt trick")
    p.add_argument("--control-after", type=int, default=0,
                   help="zero the control-harmonic gradient for the first N iters of each stage: "
                        "tendon stiffness/pretension + motor preload first (they generalize), "
                        "harmonics after (they are where timestep hacks live)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weights", type=float, nargs=4, default=(10.0, 10.0, 8.0, 0.01),
                   metavar=("wL", "wtau", "wK", "ws"))
    p.add_argument("--w-power", type=float, default=5.0,
                   help="weight on inactive (reactive/braking) motor power -- pushes elastic "
                        "tendon storage to handle decel instead of the motor (resonance). A "
                        "primary objective now; dial after seeing Pin in the log")
    p.add_argument("--w-overlap", type=float, default=5.0,
                   help="weight on the two-motor torque-overlap penalty (t1 'blue' vs t4 "
                        "'orange'): cosine similarity of their |torque| traces. Drives each "
                        "motor to peak in the other's trough (anti-phase) to cut wasted work")
    p.add_argument("--lift-torque-tradeoff", type=float, default=None, metavar="N",
                   help="put lift back in the objective as a controlled trade: an N%% peak-torque "
                        "cut is worth a 1%% lift drop below baseline (N=5 -> 5:1). Linear marginal "
                        "rate; also gates the --single-stage winner. Off by default (lift ignored, "
                        "as with --weights wL=0)")
    p.add_argument("--amp-band-pct", type=float, default=0.02,
                   help="fractional +/- band around the baseline stroke amplitude A* (wK bites "
                        "on EITHER side, not just below). Default 2%% (was floor-only before)")
    p.add_argument("--amp-floor", type=float, default=None, metavar="M",
                   help="ABSOLUTE stroke-amplitude floor [m] replacing the +/- band: a hard "
                        "one-sided wall (weight wK) + feasibility gate below it; growth is free")
    p.add_argument("--swing-mode", choices=("std", "slew"), default="std",
                   help="swing metric: 'std' = temporal std of torque (spread); 'slew' = RMS "
                        "dtau/dt normalized by drive omega (penalizes jerk, idea-queue item)")
    p.add_argument("--w-swing", type=float, default=0.0,
                   help="weight on the torque-SWING penalty (temporal std of the two drive-group "
                        "torque envelopes, normalized by the budget). The v3 demo objective: a "
                        "flat low-torque profile, not just a low peak. Set >0 to enable")
    p.add_argument("--w-valid", type=float, default=0.0,
                   help="weight on aero validity: soft AoA stall + reduced-frequency hinges")
    p.add_argument("--lp-power", type=float, default=10.0,
                   help="Lp power for smooth peak aggregation over the measured window")
    p.add_argument("--avg-periods", type=int, default=3,
                   help="number of final cycles used for downstream quantities")
    p.add_argument("--conv-eps", type=float, default=1e-4,
                   help="cycle-to-cycle qpos RMS convergence tolerance; excess is added to C_lc")
    p.add_argument("--alpha-stall-deg", type=float, default=45.0)
    p.add_argument("--alpha-smooth-deg", type=float, default=3.0)
    p.add_argument("--k-max", type=float, default=0.10)
    p.add_argument("--k-smooth", type=float, default=0.01)
    p.add_argument("--cpu-timestep-factors", type=float, nargs="*", default=(),
                   help="extra timestep factors for --cpu-every transfer checks, e.g. 0.5 2.0")
    p.add_argument("--single-stage", action="store_true",
                   help="v3: skip the torque curriculum -- one stage, normalize peak torque by the "
                        "baseline, and SELECT the design with the lowest peak+swing that keeps the "
                        "baseline flap amplitude (instead of lowest-tau clearing the lift floor)")
    p.add_argument("--w-lc", type=float, default=0.0,
                   help="weight on limit-cycle stability (cross-cycle lift variance); logged either "
                        "way. Default 0: gradient descent doesn't reduce it (chaos-corrupted variance "
                        "gradient), confirmed empirically")
    p.add_argument("--tune-frequency", action="store_true",
                   help="optimize the drive frequency (log-factor) -- the resonance lever for torque")
    # Tendon co-design: optimize per-pair (L=R) structural-tendon stiffness + pretension.
    p.add_argument("--tune-tendons", action="store_true",
                   help="optimize the 24 structural-tendon params (12 stiffness + 12 pretension)")
    p.add_argument("--tendon-params", choices=("both", "stiffness", "pretension"), default="both",
                   help="restrict --tune-tendons to one half of the structural-tendon block; the "
                        "frozen half stays at its seed (--init-tendons/--init-codesign, else 1x)")
    p.add_argument("--tune-pitch", action="store_true",
                   help="optimize the 4 wing-pitch params (up/down deadband + stiffness) -- feathering")
    p.add_argument("--tune-motor-tendons", action="store_true",
                   help="optimize the 2 motor-tendon params (shared stiffness + pretension)")
    p.add_argument("--tune-strokeplane", action="store_true",
                   help="learn the crank-disk mount orientation over the wingbeat: per DOF "
                        "(tilt, pitch) a mean + 2 Fourier harmonics of the L_t1 crank angle "
                        "(10 params), tanh-bounded by --sp-max. Kinematic stroke-plane control")
    p.add_argument("--sp-max", type=float, default=0.3,
                   help="stroke-plane bound [rad]: tilt/pitch deviation from the compiled "
                        "mount angles is squashed to +/- this via tanh")
    p.add_argument("--sp-lr-mult", type=float, default=1.0,
                   help="learning-rate multiplier for the 10 stroke-plane params. Adam steps "
                        "each param ~lr/iter regardless of gradient size, and lr is tuned for "
                        "log-factors; radians need a bigger stride or they ramp invisibly "
                        "(the run-7 a0/harmonics slow-ramp trap)")
    p.add_argument("--freeze-control", action="store_true",
                   help="hold the control signal fixed (use with --init-control)")
    p.add_argument("--init-control", type=Path, default=None,
                   help="warm-start control from a phase-1 theta_best.json")
    p.add_argument("--init-tendons", type=Path, default=None,
                   help="seed the 2*n_tend structural-tendon log-factors from a JSON "
                        "{\"tendon_theta\": [...]} (e.g. an evolutionary-search seed) instead of 1x")
    p.add_argument("--init-codesign", type=Path, default=None,
                   help="seed full [control, structural tendon, motor tendon] genome from evolve_codesign")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "runs")
    args = p.parse_args()
    if args.dt_jitter < 0.0:
        p.error("--dt-jitter must be >= 0")
    if args.precision_perturb < 0.0:
        p.error("--precision-perturb must be >= 0")
    if args.softmax_beta_start <= 0.0 or args.softmax_beta_end <= 0.0:
        p.error("--softmax-beta-start/end must be > 0")
    if args.lift_torque_tradeoff is not None and args.lift_torque_tradeoff <= 0.0:
        p.error("--lift-torque-tradeoff must be > 0")

    set_unsteady_aero(args.added_mass)  # default OFF -> MJX==CPU (transfer fix)
    tune_control = not args.freeze_control
    tune_tendons = args.tune_tendons
    tune_pitch = args.tune_pitch
    tune_motor = args.tune_motor_tendons
    tune_sp = args.tune_strokeplane
    if not (tune_control or tune_tendons or tune_pitch or tune_motor or tune_sp):
        p.error("nothing to optimize")

    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator=args.integrator,
        timestep_ms=args.timestep)
    K = args.harmonics
    wavelet_dict = wavelet_dictionary(2 * K) if args.wavelet else None
    omega = float(control.omega)
    dt = model.opt.timestep
    period = control.period
    steps_per_cycle = max(1, round(period / dt))
    warmup_steps = math.ceil(args.warmup_periods * period / dt)
    sample_steps = math.ceil(args.sample_periods * period / dt)
    motor_dir = load_morphology(args.tg).motors.motor_direction

    # Half-tendon freeze: theta packs [control][f_log][stiff(n)][pre(n)][...], so the
    # frozen half is a contiguous slice whose gradient we zero every step.
    tendon_freeze = None
    if tune_tendons and args.tendon_params != "both":
        _off = (2 * K + 1 if tune_control else 0) + (1 if args.tune_frequency else 0)
        _n = int(np.asarray(tendons[0]).shape[0])
        tendon_freeze = ((_off + _n, _off + 2 * _n) if args.tendon_params == "stiffness"
                         else (_off, _off + _n))

    base_a0 = float(drive.a0)
    control_seed = jnp.asarray(load_control_seed(args.init_control, K, drive, base_a0,
                                                 wavelet=args.wavelet),
                               dtype=jnp.float32)
    # Split the flap actuators into the two 180-deg-offset motors for the
    # anti-overlap penalty: t1 = "blue"/upstroke (mask 1), t4 = "orange"/downstroke.
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    flap_t1_mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names],
                               dtype=jnp.float32)
    sp_info = None
    if tune_sp:
        _disk_bodies = ("L_t1_disk", "L_t4_disk", "R_t1_disk", "R_t4_disk")
        _ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _disk_bodies]
        _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_t1_disk_hinge")
        assert min(_ids) >= 0 and _jid >= 0, "disk bodies / crank hinge missing from model"
        _spec = load_morphology(args.tg).motors.disk
        sp_info = dict(qadr=int(model.jnt_qposadr[_jid]),
                       body_ids=jnp.asarray(_ids, dtype=jnp.int32),
                       tilt_signs=jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=jnp.float32),
                       tilt0=float(_spec.tilt_rad), pitch0=float(_spec.pitch_rad),
                       max_rad=args.sp_max)
    metric_kwargs = dict(
        lp_power=args.lp_power,
        avg_periods=args.avg_periods,
        conv_eps=args.conv_eps,
        alpha_stall=math.radians(args.alpha_stall_deg),
        alpha_smooth=math.radians(args.alpha_smooth_deg),
        k_max=args.k_max,
        k_smooth=args.k_smooth,
        swing_mode=args.swing_mode,
        tune_strokeplane=tune_sp,
        strokeplane=sp_info,
        wavelet=wavelet_dict,
    )
    metrics = build_rollout(mx, dx0, drive, pitch, aero, tip_ids, omega, K,
                            warmup_steps, sample_steps, steps_per_cycle=steps_per_cycle,
                            aero_substep=args.aero_substep, base_a0=base_a0,
                            tune_control=tune_control, control_fixed=control_seed,
                            tune_frequency=args.tune_frequency, tune_tendons=tune_tendons,
                            tendons=tendons, tune_pitch=tune_pitch, tune_motor=tune_motor,
                            motor_ids=motor_ids, flap_t1_mask=flap_t1_mask, **metric_kwargs)

    # Integrator-agnostic reward: build implicitfast + rk4 rollouts of the SAME design
    # (both differentiable in MJX) so the loss can optimize the robust cross-integrator
    # peak torque. Same builder, only the baked mx.opt.integrator differs.
    metrics_impl = metrics_rk4 = None
    if args.tri_integrator:
        def _mx_with(integ):
            model.opt.integrator = _INTEGRATORS[integ]
            return mjx.put_model(model)
        mx_impl, mx_rk4 = _mx_with("implicitfast"), _mx_with("rk4")
        model.opt.integrator = _INTEGRATORS[args.integrator]  # restore euler for CPU checks
        _ro = lambda mxx: build_rollout(mxx, dx0, drive, pitch, aero, tip_ids, omega, K,
                            warmup_steps, sample_steps, steps_per_cycle=steps_per_cycle,
                            aero_substep=args.aero_substep, base_a0=base_a0,
                            tune_control=tune_control, control_fixed=control_seed,
                            tune_frequency=args.tune_frequency, tune_tendons=tune_tendons,
                            tendons=tendons, tune_pitch=tune_pitch, tune_motor=tune_motor,
                            motor_ids=motor_ids, flap_t1_mask=flap_t1_mask, **metric_kwargs)
        metrics_impl, metrics_rk4 = _ro(mx_impl), _ro(mx_rk4)

    # Fix 3: also roll out euler at dt/2 and 2*dt (differentiable in MJX) so a design
    # that only holds up at the exact trained timestep is punished in the gradient too,
    # not just the integrator swap above. Each dt gets its own steps_per_cycle/warmup/
    # sample_steps (period is fixed, dt changes -> step counts rescale).
    extra_metrics = []
    if args.tri_timestep:
        dt_factors = tuple(args.tri_timestep_factors) if args.tri_timestep_factors is not None else (0.5, 2.0)
        for dt_scale in dt_factors:
            if dt_scale <= 0:
                p.error("--tri-timestep-factors must be positive")
            if abs(dt_scale - 1.0) < 1e-9:
                continue
            model.opt.timestep = dt * dt_scale
            model.opt.integrator = _INTEGRATORS[args.integrator]
            mx_dt = mjx.put_model(model)
            aero_dt = aero_runtime.build_aero_config(model, substep=args.aero_substep)
            spc = max(1, round(period / (dt * dt_scale)))
            wu = math.ceil(args.warmup_periods * period / (dt * dt_scale))
            sm = math.ceil(args.sample_periods * period / (dt * dt_scale))
            extra_metrics.append(build_rollout(
                mx_dt, dx0, drive, pitch, aero_dt, tip_ids, omega, K, wu, sm,
                steps_per_cycle=spc, aero_substep=args.aero_substep, base_a0=base_a0,
                tune_control=tune_control, control_fixed=control_seed,
                tune_frequency=args.tune_frequency, tune_tendons=tune_tendons,
                tendons=tendons, tune_pitch=tune_pitch, tune_motor=tune_motor,
                motor_ids=motor_ids, flap_t1_mask=flap_t1_mask, **metric_kwargs))
        model.opt.timestep = dt  # restore baseline dt for CPU checks

    def _variant_metrics(integ, dt_ms, *, qvel_noise=None, warmup_periods=None,
                         sample_periods=None, avg_periods=None):
        """Forward rollout of the SAME design under (integrator, dt) -- the building
        block for the rotating mini-batches, the full-matrix check and the holdout.
        warmup/sample/avg overrides support warmup-jittered training members and the
        long-window certification rollouts."""
        model.opt.integrator = _INTEGRATORS[integ]
        model.opt.timestep = dt_ms * 1e-3
        mx_v = mjx.put_model(model)
        aero_v = aero_runtime.build_aero_config(model, substep=args.aero_substep)
        wu_p = args.warmup_periods if warmup_periods is None else warmup_periods
        sm_p = args.sample_periods if sample_periods is None else sample_periods
        spc = max(1, round(period / (dt_ms * 1e-3)))
        wu = math.ceil(wu_p * period / (dt_ms * 1e-3))
        sm = math.ceil(sm_p * period / (dt_ms * 1e-3))
        mk = dict(metric_kwargs)
        if avg_periods is not None:
            mk["avg_periods"] = avg_periods
        dx_start = dx0 if qvel_noise is None else dx0.replace(
            qvel=dx0.qvel + jnp.asarray(qvel_noise, dtype=dx0.qvel.dtype)
        )
        f = build_rollout(mx_v, dx_start, drive, pitch, aero_v, tip_ids, omega, K, wu, sm,
                          steps_per_cycle=spc, aero_substep=args.aero_substep,
                          base_a0=base_a0, tune_control=tune_control,
                          control_fixed=control_seed, tune_frequency=args.tune_frequency,
                          tune_tendons=tune_tendons, tendons=tendons, tune_pitch=tune_pitch,
                          tune_motor=tune_motor, motor_ids=motor_ids,
                          flap_t1_mask=flap_t1_mask, **mk)
        model.opt.integrator = _INTEGRATORS[args.integrator]
        model.opt.timestep = dt  # restore for CPU checks
        return f

    # Rotating robust mini-batches: a BANK of (integrator, dt) rollouts; each
    # iteration optimizes the robust envelope over one small random batch. The full
    # matrix is enforced every --full-check-every iters instead of every step.
    bank = bank_jit = None
    if args.rotate_batch:
        if args.tri_integrator or args.tri_timestep:
            p.error("--rotate-batch replaces --tri-integrator/--tri-timestep")
        if args.rotate_batch < 2:
            p.error("--rotate-batch needs >=2 variants per batch")
        dt_bank = tuple(args.dt_bank) if args.dt_bank else (dt * 1e3,)
        # Certification rollouts sample a LONG window (>= one beat period) so slow
        # torque transients show up as per-cycle drift instead of hiding outside a
        # 3-cycle window; training batches keep the short (cheap) window and rely
        # on warmup jitter instead.
        full_sp = args.full_sample_periods if args.full_sample_periods else args.sample_periods
        bank = {(ig, dm): _variant_metrics(ig, dm, sample_periods=full_sp,
                                           avg_periods=max(1, int(full_sp)))
                for ig in ("euler", "implicitfast", "rk4") for dm in dt_bank}
        bank_jit = {k: jax.jit(f) for k, f in bank.items()}

    # theta packs active groups in order:
    # [control(2K+1)] [f_log(1)] [tendon(24)] [pitch(4)] [motor(2)]
    parts = []
    codesign = None
    n_tend = int(np.asarray(tendons[0]).shape[0])  # structural tendons tuned (individual: 24)
    if args.init_codesign is not None:
        d = json.loads(args.init_codesign.read_text())
        codesign = jnp.asarray(d.get("genome", d.get("theta")), dtype=jnp.float32)
        want = 2 * K + 1 + 2 * n_tend + 2
        assert codesign.shape[0] >= want, f"init-codesign len {codesign.shape[0]} < {want}"
    if tune_control:
        if codesign is not None:
            # Keep the warm-start a0; in wavelet mode zero the basis coeffs so a
            # Fourier design's harmonics don't get reinterpreted as atom amplitudes
            # (sinusoid start -- structure inherited, control reset to explore).
            c = codesign[:2 * K + 1]
            parts.append(c.at[1:].set(0.0) if args.wavelet else c)
        else:
            parts.append(control_seed)
    if args.tune_frequency:
        parts.append(jnp.zeros(1, dtype=jnp.float32))  # f_log, seeded at base frequency
    if tune_tendons:
        if codesign is not None:
            s = 2 * K + 1
            parts.append(codesign[s:s + 2 * n_tend])
        elif args.init_tendons is not None:  # seed from an evolutionary-search config
            seed_t = jnp.asarray(json.loads(args.init_tendons.read_text())["tendon_theta"],
                                 dtype=jnp.float32)
            assert seed_t.shape[0] == 2 * n_tend, f"init-tendons len {seed_t.shape[0]} != {2*n_tend}"
            parts.append(seed_t)
        else:
            parts.append(jnp.zeros(2 * n_tend, dtype=jnp.float32))  # stiff+pre log-factors, seeded 1x
    if tune_pitch:
        parts.append(jnp.zeros(4, dtype=jnp.float32))   # pitch deadband/stiffness log-factors
    if tune_motor:
        parts.append(codesign[2 * K + 1 + 2 * n_tend:2 * K + 1 + 2 * n_tend + 2]
                     if codesign is not None else jnp.zeros(2, dtype=jnp.float32))
    if tune_sp:
        # sp is always the LAST group; a full-genome seed (theta_full.json) carries it.
        if codesign is not None and codesign.shape[0] >= 2 * K + 1 + 2 * n_tend + 2 + 10:
            parts.append(codesign[-10:])
        else:
            parts.append(jnp.zeros(10, dtype=jnp.float32))  # seeded at the baseline plane
    theta = jnp.concatenate(parts)
    lr_scale = jnp.ones_like(theta)
    if tune_sp and args.sp_lr_mult != 1.0:
        lr_scale = lr_scale.at[-10:].set(args.sp_lr_mult)

    # Baseline A* = stroke amplitude of the seed (prevents collapse); P0 = baseline
    # supplied power (normalizes the power term so it starts at ~1).
    (Lbar0, tau0, A_target, S0, Clc0, Pin0, Ovl0, swing0, P0, Cvalid0, alpha0, k0, conv0,
     taupk0, drift0) = (float(x) for x in metrics(theta))
    print(f"[setup] tune_control={tune_control} tune_tendons={tune_tendons} "
          f"n_tend={n_tend} params={theta.shape[0]} steps={warmup_steps + sample_steps} "
          f"steps/cycle={steps_per_cycle} w_lc={args.w_lc} wK={args.weights[2]}")
    print(f"[baseline] Lbar={Lbar0*1e3:.1f} mN  tau_hat={tau0*1e3:.1f} mNm  "
          f"tau_peak={taupk0*1e3:.1f} mNm  drift={drift0*1e3:+.1f} mNm  "
          f"A_stroke={A_target*1e3:.2f} mm  power={P0*1e3:.0f} mW  S={S0:.3g}  "
          f"C_lc={Clc0:.3g}  C_valid={Cvalid0:.3g}  alpha={math.degrees(alpha0):.0f}deg "
          f"k={k0:.3g} conv={conv0:.2e}  Pin={Pin0:.3g}  overlap={Ovl0:.3g}")
    cap_str = f"{args.lift_cap*1e3:.0f}" if args.lift_cap is not None else "none"
    print(f"[target]   lift_floor={args.lift_floor*1e3:.0f} mN  cap={cap_str}  "
          f"A*={A_target*1e3:.2f} mm (wK={args.weights[2]})  w_power={args.w_power} (mean supplied)")

    # Lift reference for the tradeoff: the FLOOR, not the baseline. Anchoring at
    # baseline lift made every mN below it cost 5:1 -- the optimizer hoarded lift
    # and provably could not reach the soft-tendon corridor (fixed-codesign lesson).
    lift_ref = args.lift_floor if args.lift_torque_tradeoff else Lbar0
    loss_kwargs = dict(lift_floor=args.lift_floor, lift_cap=args.lift_cap,
                       lift_smooth=args.lift_smooth, lift_above_weight=args.lift_above_weight,
                       A_target=A_target, tau_max=None,
                       weights=tuple(args.weights), w_lc=args.w_lc, w_power=args.w_power,
                       w_overlap=args.w_overlap, w_swing=args.w_swing,
                       lift0=lift_ref, tradeoff=args.lift_torque_tradeoff, w_gap=args.w_gap,
                       amp_band_pct=args.amp_band_pct, amp_floor=args.amp_floor,
                       w_valid=args.w_valid, w_drift=args.w_drift)
    grad_fn = make_loss(metrics, metrics_impl=metrics_impl, metrics_rk4=metrics_rk4,
                        extra_metrics=tuple(extra_metrics), **loss_kwargs)

    # Compose the rotating mini-batches: random K-subsets of the bank, each with >=2
    # integrators and >=2 dts (so every batch exercises both robustness axes). Each
    # batch gets its own jitted robust grad_fn; the loop rotates among them.
    rotate_fns = None
    rng = random.Random(args.batch_seed)
    if args.rotate_batch:
        keys = list(bank)
        kbatch = min(args.rotate_batch, len(keys))

        def _ok(bt):
            return (len({v[0] for v in bt}) >= min(2, kbatch)
                    and (len({v[1] for v in bt}) >= min(2, kbatch) or len(dt_bank) == 1))

        batches = []
        for _ in range(1000):
            if len(batches) >= args.n_batches:
                break
            bt = tuple(sorted(rng.sample(keys, kbatch)))
            if _ok(bt) and bt not in batches:
                batches.append(bt)
        if not batches:
            p.error("--rotate-batch: no valid batch composition from the bank")
        rotate_fns = []
        # Warmup jitter: each member also draws its warmup length, so the (short)
        # sample window lands at a different point of any slow torque beat per
        # member -- the robust envelope then prices the whole beat, and parking a
        # transient minimum at one window position stops paying.
        wu_bank = tuple(args.warmup_bank) if args.warmup_bank else (args.warmup_periods,)
        for centers in batches:
            # Floor jittered dts at the smallest bank center: sub-0.04 draws cost
            # ~15% more steps/rollout and CPU factor 0.5 already certifies dt 0.03.
            bt = tuple(
                (ig, max(dm * (1.0 + rng.uniform(-args.dt_jitter, args.dt_jitter)),
                         min(dt_bank)),
                 rng.choice(wu_bank))
                for ig, dm in centers
            )
            ms = [_variant_metrics(ig, dm, warmup_periods=wu) for ig, dm, wu in bt]
            labels = [f"{ig}@{dm:g}/w{wu:g}" for ig, dm, wu in bt]
            if args.precision_perturb:
                ig, dm, wu = bt[0]
                noise = np.asarray([rng.gauss(0.0, args.precision_perturb)
                                    for _ in range(int(dx0.qvel.shape[0]))])
                ms.append(_variant_metrics(ig, dm, warmup_periods=wu, qvel_noise=noise))
                labels.append(f"{ig}@{dm:g}/w{wu:g}+eps")
            rotate_fns.append((tuple(labels), make_loss(
                ms[0], metrics_impl=ms[1] if len(ms) > 1 else None,
                metrics_rk4=ms[2] if len(ms) > 2 else None,
                extra_metrics=tuple(ms[3:]), **loss_kwargs)))
        print(f"[rotate] bank={len(bank)} center variants (3 integrators x dts {dt_bank} ms), "
              f"{len(batches)} batches of {kbatch}"
              f" jitter={args.dt_jitter:g} precision_perturb={args.precision_perturb:g}:")
        for labels, _ in rotate_fns:
            print("  " + " + ".join(labels))

    # Hand-rolled Adam (no optax dependency).
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0

    args.out.mkdir(parents=True, exist_ok=True)
    tnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(k))
              for k in np.asarray(tendons[0])]  # full name (individual tendons aren't unique sans L_/R_)
    base_stiff = np.asarray(tendons[2])
    base_lenspring = np.asarray(tendons[3])
    l_ids = np.asarray(tendons[0])

    base_freq = control.fundamental_frequency_hz
    control_src = Path(load_morphology(args.tg).motors.control_file)
    if not control_src.is_absolute():
        control_src = args.tg.parent / control_src

    def dump(design, log, out_dir=args.out):
        """Checkpoint the selected design to disk (interrupted run isn't lost)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out = np.asarray(design["theta"])
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))
        # Raw packed theta: --init-codesign can warm-start straight from this file
        # (control+tendons+motor sliced by position, sp tail picked up when present).
        (out_dir / "theta_full.json").write_text(json.dumps(
            {"theta": [float(x) for x in out]}))
        shoulder = args.tg.read_text()
        shoulder = re.sub(r"(?m)^(\s*control\s*=\s*).*$", r"\1control.tg", shoulder)
        (out_dir / "shoulder.tg").write_text(shoulder)
        i = 0
        freq_hz = base_freq
        if tune_control:
            ctrl = out[i : i + 2 * K + 1]; i += 2 * K + 1
        else:
            ctrl = np.asarray(control_seed)
        a0 = base_a0 * math.exp(ctrl[0])  # a0 log-factor -> absolute (actuator frame)
        a_out, b_out = a0 * ctrl[1 : 1 + K], a0 * ctrl[1 + K :]  # depth -> absolute a_n
        if args.tune_frequency:
            freq_hz = base_freq * math.exp(float(out[i])); i += 1
        if (tune_control or args.tune_frequency) and args.wavelet:
            # Wavelet basis: coeffs are dimensionless atom depths (frame-neutral).
            # theta_full.json is the authoritative warm-start; this carries the
            # dictionary so a wavelet-aware certifier can rebuild the waveform.
            coeffs = [float(x) for x in ctrl[1:]]
            centers, sigmas, freqs = wavelet_dictionary(2 * K)
            lines = [f"basis = wavelet", f"n_atoms = {2 * K}",
                     f"fundamental_frequency = {freq_hz:g} Hz", f"a0 = {a0 / motor_dir:.6g}"]
            lines += [f"w_{j} = {coeffs[j]:.6g}" for j in range(2 * K)]
            control_text = "\n".join(lines) + "\n"
            (out_dir / "control.tg").write_text(control_text)
            (out_dir / "control_optimized.tg").write_text(control_text)
            (out_dir / "theta_best.json").write_text(json.dumps(
                {"a0": float(a0), "basis": "wavelet", "n_atoms": 2 * K,
                 "coeffs": coeffs, "centers": [float(x) for x in centers],
                 "sigmas": [float(x) for x in sigmas],
                 "frequency_hz": float(freq_hz), "omega": float(math.tau * freq_hz),
                 "motor_direction": motor_dir, "Lbar": design["Lbar"],
                 "tau_hat": design["tau"]}, indent=2))
        elif tune_control or args.tune_frequency:
            lines = [f"harmonics = {K}", f"fundamental_frequency = {freq_hz:g} Hz",
                     f"a0 = {a0 / motor_dir:.6g}"]
            for j in range(K):
                lines.append(f"a_{j+1} = {a_out[j] / motor_dir:.6g}")
                lines.append(f"b_{j+1} = {b_out[j] / motor_dir:.6g}")
            control_text = "\n".join(lines) + "\n"
            (out_dir / "control.tg").write_text(control_text)
            (out_dir / "control_optimized.tg").write_text(control_text)
            (out_dir / "theta_best.json").write_text(json.dumps(
                {"a0": float(a0), "a": [float(x) for x in a_out], "b": [float(x) for x in b_out],
                 "frequency_hz": float(freq_hz), "omega": float(math.tau * freq_hz),
                 "motor_direction": motor_dir, "Lbar": design["Lbar"],
                 "tau_hat": design["tau"]}, indent=2))
        else:
            (out_dir / "control.tg").write_text(control_src.read_text())
        if tune_tendons:
            stiff_log, pre_log = out[i : i + n_tend], out[i + n_tend : i + 2 * n_tend]; i += 2 * n_tend
            tendon_out = {
                tnames[k]: {
                    "stiffness": float(base_stiff[l_ids[k]] * math.exp(stiff_log[k])),
                    "stiffness_factor": float(math.exp(stiff_log[k])),
                    "pretension_factor": float(math.exp(pre_log[k])),
                    "rest_length": float(base_lenspring[l_ids[k]][0] * math.exp(pre_log[k])),
                }
                for k in range(n_tend)
            }
            (out_dir / "tendon_optimized.json").write_text(json.dumps(
                {"tau_hat": design["tau"], "Lbar": design["Lbar"], "pairs": tendon_out}, indent=2))
        if tune_pitch:
            p4 = out[i : i + 4]; i += 4
            (out_dir / "pitch_optimized.json").write_text(json.dumps({
                "upstroke_pitch_rad": float(pitch.upstroke_pitch) * math.exp(p4[0]),
                "downstroke_pitch_rad": float(pitch.downstroke_pitch) * math.exp(p4[1]),
                "upstroke_stiffness": float(pitch.upstroke_stiffness) * math.exp(p4[2]),
                "downstroke_stiffness": float(pitch.downstroke_stiffness) * math.exp(p4[3]),
            }, indent=2))
        if tune_motor:
            ml = out[i : i + 2]
            mids = np.asarray(motor_ids)
            (out_dir / "motor_tendon_optimized.json").write_text(json.dumps({
                "stiffness": float(base_stiff[mids[0]] * math.exp(ml[0])),
                "stiffness_factor": float(math.exp(ml[0])),
                "pretension_factor": float(math.exp(ml[1])),
            }, indent=2))
        if tune_sp:
            spv = out[-10:]  # always the LAST theta group
            (out_dir / "strokeplane_optimized.json").write_text(json.dumps({
                "phase_ref_joint": "L_t1_disk_hinge",
                "tilt0_rad": sp_info["tilt0"], "pitch0_rad": sp_info["pitch0"],
                "max_delta_rad": args.sp_max,
                "tilt_coeffs": [float(x) for x in spv[:5]],
                "pitch_coeffs": [float(x) for x in spv[5:]],
                "model": "angle = angle0 + max*tanh((c0 + c1*cos(th) + c2*sin(th) "
                         "+ c3*cos(2th) + c4*sin(2th))/max), th = L_t1 crank angle",
            }, indent=2))

    phase_lag = load_morphology(args.tg).motors.crank_phase_lag_rad

    def theta_to_cpu(out):
        """Decode a packed theta -> (ctrl_params, tendon_arrays, pitch_params) for
        cpu_validate_design (same arrays MJX used, no .tg round-trip)."""
        out = np.asarray(out); j = 0
        if tune_control:
            ctrl = out[j : j + 2 * K + 1]; j += 2 * K + 1
        else:
            ctrl = np.asarray(control_seed)  # frozen control = the loaded seed
        a0 = base_a0 * math.exp(ctrl[0])
        # wavelet mode: all 2K coeffs are atom amplitudes (b unused); else a/b halves.
        a_dep, b_dep = (ctrl[1:], ctrl[:0]) if args.wavelet else (ctrl[1 : 1 + K], ctrl[1 + K :])
        f_log = out[j] if args.tune_frequency else 0.0
        if args.tune_frequency:
            j += 1
        tendon_arrays = None
        if tune_tendons or tune_motor:
            bs = np.asarray(tendons[2]).copy(); bl = np.asarray(tendons[3]).copy()
        if tune_tendons:
            li = np.asarray(tendons[0]); ri = np.asarray(tendons[1])
            sl, pl = out[j : j + n_tend], out[j + n_tend : j + 2 * n_tend]; j += 2 * n_tend
            bs[li] = np.asarray(tendons[2])[li] * np.exp(sl); bs[ri] = bs[li]
            bl[li] = np.asarray(tendons[3])[li] * np.exp(pl)[:, None]; bl[ri] = bl[li]
        pitch_params = None
        if tune_pitch:
            p4 = out[j : j + 4]; j += 4
            pitch_params = (float(pitch.upstroke_pitch) * math.exp(p4[0]),
                            float(pitch.downstroke_pitch) * math.exp(p4[1]),
                            float(pitch.upstroke_stiffness) * math.exp(p4[2]),
                            float(pitch.downstroke_stiffness) * math.exp(p4[3]))
        if tune_motor:
            ml = out[j : j + 2]; mids = np.asarray(motor_ids)
            bs[mids] = np.asarray(tendons[2])[mids] * math.exp(ml[0])
            bl[mids] = np.asarray(tendons[3])[mids] * math.exp(ml[1])
            j += 2
        if tune_tendons or tune_motor:
            tendon_arrays = (bs, bl)
        sp_params = ((out[j : j + 10], sp_info["tilt0"], sp_info["pitch0"], args.sp_max)
                     if tune_sp else None)
        return (a0, a_dep, b_dep, omega * math.exp(f_log)), tendon_arrays, pitch_params, sp_params

    cpu_sp = args.cpu_sample_periods if args.cpu_sample_periods else args.sample_periods

    def cpu_check(theta_v, integrator="euler"):
        cp, ta, pp, spp = theta_to_cpu(theta_v)
        return cpu_validate_design(args.scene, cp, ta, integrator, pitch_params=pp,
                                   strokeplane_params=spp,
                                   warmup=args.warmup_periods, periods=cpu_sp,
                                   phase_lag=phase_lag, timestep_ms=args.timestep,
                                   aero_substep=args.aero_substep, wavelet=wavelet_dict)

    def cpu_transfer(theta_v):
        variants = [(args.integrator, args.timestep)]
        if args.tri_integrator or args.rotate_batch:
            variants += [("implicitfast", args.timestep), ("rk4", args.timestep)]
        variants += [(args.integrator, None if args.timestep is None else args.timestep * f)
                     for f in args.cpu_timestep_factors]
        out = []
        cp, ta, pp, spp = theta_to_cpu(theta_v)
        for integ, ts in variants:
            L, T, P, pc = cpu_validate_design(args.scene, cp, ta, integ, pitch_params=pp,
                                              strokeplane_params=spp,
                                              warmup=args.warmup_periods, periods=cpu_sp,
                                              phase_lag=phase_lag, timestep_ms=ts,
                                              aero_substep=args.aero_substep, cycle_stats=True,
                                              wavelet=wavelet_dict)
            drift = (pc[-1] - pc[0]) if len(pc) >= 2 else 0.0
            out.append({"integrator": integ, "timestep_ms": ts, "L": L, "T": T, "P": P,
                        "drift": drift})
        robust = max(x["T"] for x in out)
        gap = (robust - min(x["T"] for x in out)) / max(robust, 1e-9)
        lift = min(x["L"] for x in out)
        driftmax = max(abs(x["drift"]) for x in out)
        return out, robust, gap, lift, driftmax

    # CPU reference for the baseline: periodic checks score designs on the ROBUST
    # (worse of euler/implicitfast) peak torque, so a euler-hacked gait can't look cheap.
    cpuL0 = cpuT0 = cpuT0i = cpuT0r = float("nan")
    cpu_best = None
    if args.cpu_every:
        # Startup CPU baseline REMOVED (~45 min of CPU rollouts before training even
        # started, on a design whose numbers are already known). The first in-loop
        # cpu check seeds cpuT0*/cpuL0 instead.
        cpu_best = {"score": math.inf, "tau": math.inf, "swing": float("nan"),
                    "theta": theta, "Lbar": float("nan"), "Pin": float("nan"), "mjx_tau": float("nan")}

    # Full-matrix check (rotate mode): forward-evaluate EVERY bank variant. The
    # rotating batches only ever see subsets; this is the in-loop 12-case referee.
    robust_best = None
    rob0 = liftmin0 = float("nan")
    if args.rotate_batch:
        def full_matrix(theta_v):
            # Scored on the EXACT per-cycle peak (m[13]), not the Lp proxy (m[1]):
            # Lp under-reads narrow torque spikes by 15-25% and is duty-cycle
            # hackable, which previously masqueraded as an "MJX<->CPU gap".
            rows = []
            for (ig, dm), f in bank_jit.items():
                m = [float(x) for x in f(theta_v)]
                rows.append({"integrator": ig, "timestep_ms": dm, "L": m[0], "T": m[13],
                             "T_lp": m[1], "drift": m[14], "A": m[2], "swing": m[7]})
            rob = max(r["T"] for r in rows)
            gap = (rob - min(r["T"] for r in rows)) / max(rob, 1e-9)
            driftmax = max(abs(r["drift"]) for r in rows)
            return rows, rob, gap, min(r["L"] for r in rows), driftmax

        # Startup full-matrix baseline REMOVED (12 jit compiles + rollouts before
        # training started). The first in-loop full check seeds rob0/liftmin0.
        robust_best = {"score": math.inf, "tau": math.inf, "swing": float("nan"),
                       "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}

    log = []
    # Selection (--single-stage): lowest peak torque + swing keeping the baseline flap
    # amplitude. Otherwise (curriculum): lowest peak torque clearing the lift floor.
    if args.amp_floor is not None:
        amp_gate_lo, amp_gate_hi = args.amp_floor, float("inf")
    else:
        amp_gate_lo = A_target * (1.0 - args.amp_band_pct)
        amp_gate_hi = A_target * (1.0 + args.amp_band_pct)
    def score_of(tau_hat, swing, Lbar):
        if not args.single_stage:
            return tau_hat
        s = tau_hat + args.score_swing * swing
        if args.lift_torque_tradeoff:  # convert lift drop into a torque-equivalent at N:1
            # Same floor-anchored reference as the loss (see lift_ref above).
            drop = max(0.0, float(lift_ref) - Lbar) / (abs(float(lift_ref)) + 1e-9)
            gain = max(0.0, Lbar - float(lift_ref)) / (abs(float(lift_ref)) + 1e-9)
            s += args.lift_torque_tradeoff * float(tau0) * drop
            s -= float(tau0) * gain / args.lift_torque_tradeoff
        return s
    def is_feasible(Lbar, A_stroke):
        if args.single_stage:
            ok = amp_gate_lo <= A_stroke <= amp_gate_hi
            if args.lift_torque_tradeoff is None:
                # No trade to guard lift and the amp band may be vacuous: the
                # explicit floor is the selection contract (min tau AT >= floor).
                ok = ok and Lbar >= args.lift_floor
            return ok
        return Lbar >= args.lift_floor
    best = {"score": math.inf, "tau": math.inf, "power": math.inf, "swing": float("nan"),
            "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
    if args.single_stage:
        # One stage, peak torque normalized by the baseline (so baseline scores ~1).
        curriculum = (tau0,)
    else:
        curriculum = tuple(sorted({max(args.tau_min, s) for s in TAU_CURRICULUM_NM}, reverse=True))
    print(f"[curriculum] tau_max stages (mNm): {[round(x*1e3) for x in curriculum]}")
    total_train_iters = max(1, len(curriculum) * args.iters_per_stage - 1)
    for stage_i, tau_max in enumerate(curriculum):
        print(f"\n=== stage tau_max = {tau_max*1e3:.0f} mNm ===")
        # Per-stage winner: lowest peak-torque design that clears the lift floor AT
        # THIS budget. Saved to its own subdir so the demo has one design per stage.
        stage_best = {"score": math.inf, "tau": math.inf, "power": math.inf, "swing": float("nan"),
                      "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
        lr_cur = args.lr          # adaptive step (halved on each divergence)
        last_good = theta         # last theta that produced a finite gradient
        for it in range(args.iters_per_stage):
            train_i = stage_i * args.iters_per_stage + it
            beta_frac = min(1.0, train_i / total_train_iters)
            robust_beta = args.softmax_beta_start + (
                args.softmax_beta_end - args.softmax_beta_start
            ) * beta_frac
            # Phase boundary: snapshot the certified winners of the tendon/motor-only
            # phase before the harmonics start moving, so "what did harmonics buy"
            # is answerable by diffing <out>/*_preharmonics against the final bests.
            if args.control_after and it == args.control_after:
                if robust_best is not None and math.isfinite(robust_best["tau"]):
                    dump(robust_best, log, out_dir=args.out / "robust_best_preharmonics")
                if cpu_best is not None and math.isfinite(cpu_best["tau"]):
                    dump(cpu_best, log, out_dir=args.out / "cpu_best_preharmonics")
                print(f"  [phase] it{it:03d}: harmonics UNLOCKED; pre-harmonics bests "
                      f"snapshotted to *_preharmonics", flush=True)
            theta_eval = theta  # metrics below describe THIS theta (pre-update)
            batch_keys = None
            if rotate_fns:
                batch_keys, batch_fn = rotate_fns[rng.randrange(len(rotate_fns))]
            else:
                batch_fn = grad_fn
            grad, aux = batch_fn(theta, tau_max, robust_beta)
            # NaN guard: a tendon step can walk the rollout into an unstable
            # (euler-divergent) config -> NaN grad poisons Adam for the whole run.
            # Instead revert to the last finite design, clear momentum, halve the
            # step, and retry -- converges to the best STABLE design at the edge.
            if not bool(jnp.isfinite(grad).all()):
                theta, lr_cur = last_good, lr_cur * 0.5
                m = jnp.zeros_like(theta); v = jnp.zeros_like(theta); t = 0
                print(f"  it{it:03d} NaN/diverged -> revert, lr={lr_cur:.2e}", flush=True)
                continue
            last_good = theta
            if args.control_after and it < args.control_after:
                if tune_control:
                    grad = grad.at[:2 * K + 1].set(0.0)  # safer knobs first: tendons/motor only
                if tune_sp:
                    grad = grad.at[-10:].set(0.0)  # stroke-plane unlocks with the harmonics
            if tendon_freeze is not None:
                grad = grad.at[tendon_freeze[0]:tendon_freeze[1]].set(0.0)
            (Lbar, tau_hat, A_stroke, S, Clc, Pin, Ovl, swing, P_sup, loss,
             tau_i, tau_r, tau_rob, tau_spread, Cvalid, alpha_pk, k_eff, conv_err,
             tau_pk, drift_w) = (float(x) for x in aux)
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad**2
            mhat = m / (1 - b1**t)
            vhat = v / (1 - b2**t)
            theta = theta - lr_cur * lr_scale * mhat / (jnp.sqrt(vhat) + eps)

            log.append({"stage_tau_max": tau_max, "iter": it, "loss": loss, "Lbar": Lbar,
                        "tau_hat": tau_hat, "tau_impl": tau_i, "tau_rk4": tau_r,
                        "tau_peak": tau_pk, "tau_drift": drift_w,
                        "tau_robust": tau_rob, "mjx_gap": tau_spread / max(tau_rob, 1e-9),
                        "A_stroke": A_stroke, "power": P_sup, "swing": swing,
                        "S": S, "C_lc": Clc, "P_inactive": Pin, "overlap": Ovl,
                        "C_valid": Cvalid, "alpha_pk": alpha_pk, "k_eff": k_eff,
                        "conv_err": conv_err, "softmax_beta": robust_beta})
            # Select on the ROBUST peak so the winner is integrator-agnostic by construction.
            # Rotate mode: per-iter metrics only cover the drawn batch (an optimistic
            # subset), so selection happens at the full-matrix checks instead.
            score = score_of(tau_rob, swing, Lbar)
            sel = {"score": score, "tau": tau_rob, "power": P_sup, "swing": swing,
                   "theta": theta_eval, "Lbar": Lbar, "Pin": Pin}
            if not rotate_fns and is_feasible(Lbar, A_stroke) and score < best["score"]:
                best = sel
            if not rotate_fns and is_feasible(Lbar, A_stroke) and score < stage_best["score"]:
                stage_best = sel
            if rotate_fns:
                tri = (f" [{'/'.join(batch_keys)} "
                       f"rob={tau_rob*1e3:.1f} gap={tau_spread/max(tau_rob,1e-9)*100:.0f}%]")
            else:
                tri = (f" [e{tau_hat*1e3:.0f}/i{tau_i*1e3:.0f}/r{tau_r*1e3:.0f} "
                       f"rob={tau_rob*1e3:.1f} gap={tau_spread/max(tau_rob,1e-9)*100:.0f}%]"
                       if (args.tri_integrator or args.tri_timestep) else "")
            print(f"  it{it:03d} loss={loss:.4f}  L={Lbar*1e3:6.1f}mN  tau={tau_hat*1e3:6.1f}mNm  "
                  f"pk={tau_pk*1e3:6.1f}mNm dr={drift_w*1e3:+5.1f}  "
                  f"swing={swing*1e3:5.1f}mNm  A={A_stroke*1e3:5.2f}mm  P={P_sup*1e3:5.0f}mW  "
                  f"S={S:.3g}  Cv={Cvalid:.2g} a={math.degrees(alpha_pk):.0f} k={k_eff:.2g} "
                  f"conv={conv_err:.1e} beta={robust_beta:.0f} Pin={Pin:.3g}{tri}", flush=True)
            if it % 3 == 0:
                dump(sel, log, out_dir=args.out / "checkpoints" / f"it{it:03d}")
            if (rotate_fns and args.full_check_every
                    and it % args.full_check_every == 0 and train_i > 0):
                rows, rob, gapf, liftmin, driftf = full_matrix(theta_eval)
                if not math.isfinite(liftmin0):  # first check seeds the baseline (startup eval removed)
                    rob0, liftmin0 = rob, liftmin
                amps = [r["A"] for r in rows]
                swmax = max(r["swing"] for r in rows)
                feas = amp_gate_lo <= min(amps) and max(amps) <= amp_gate_hi
                if args.single_stage and args.lift_torque_tradeoff is None:
                    feas = feas and liftmin >= args.lift_floor  # robust-min lift clears the floor
                stationary = driftf <= args.drift_gate * max(rob, 1e-9)
                transfers = gapf <= args.full_gap and liftmin >= 0.9 * liftmin0 and stationary
                scoref = score_of(rob, swmax, liftmin)
                tag = ("TRANSFERS" if feas else "LIFT/AMP-OUT") if transfers \
                    else (f"HACK(gap {gapf*100:.0f}%)" if gapf > args.full_gap
                          else f"DRIFT({driftf/max(rob,1e-9)*100:.0f}%)")
                print(f"    [full it{it:03d}] robust={rob*1e3:6.1f} mNm "
                      f"({(rob-rob0)/rob0*100:+.0f}% vs base)  gap={gapf*100:.0f}%  "
                      f"liftmin={liftmin*1e3:6.1f} mN  drift={driftf*1e3:.1f} mNm  [{tag}]",
                      flush=True)
                log[-1].update(full_robust=rob, full_gap=gapf, full_liftmin=liftmin,
                               full_drift=driftf, full_rows=rows)
                sel_f = {"score": scoref, "tau": rob, "power": P_sup, "swing": swmax,
                         "theta": theta_eval, "Lbar": liftmin, "Pin": Pin}
                if feas and scoref < best["score"]:
                    best = sel_f
                if feas and scoref < stage_best["score"]:
                    stage_best = sel_f
                if feas and transfers and scoref < robust_best["score"]:
                    robust_best = dict(sel_f)
                    dump(robust_best, log, out_dir=args.out / "robust_best")
            if args.cpu_every and it % args.cpu_every == 0 and train_i > 0:
                cpu_vars, robust, gap, cpuL, cdrift = cpu_transfer(theta_eval)
                cpuT = cpu_vars[0]["T"]
                cpuTi = next((x["T"] for x in cpu_vars if x["integrator"] == "implicitfast"), cpuT)
                cpuTr = next((x["T"] for x in cpu_vars if x["integrator"] == "rk4"), cpuTi)
                if not math.isfinite(cpuT0r):  # first check seeds the baseline (startup eval removed)
                    cpuT0, cpuT0i, cpuT0r, cpuL0 = cpuT, cpuTi, robust, cpuL
                stationary = cdrift <= args.drift_gate * max(robust, 1e-9)
                transfers = gap <= args.cpu_gap and stationary
                dpk = (robust - cpuT0r) / cpuT0r * 100 if cpuT0r else float("nan")
                tag = "TRANSFERS" if transfers else (
                    f"HACK(gap {gap*100:.0f}%)" if gap > args.cpu_gap
                    else f"DRIFT({cdrift/max(robust,1e-9)*100:.0f}%)")
                desc = " ".join(f"{x['integrator']}@{(x['timestep_ms'] or 0):g}={x['T']*1e3:.0f}"
                                for x in cpu_vars)
                print(f"    [cpu it{it:03d}] mjx={tau_hat*1e3:6.1f}  {desc}  "
                      f"robust={robust*1e3:6.1f} mNm ({dpk:+.0f}% vs base)  "
                      f"Lmin={cpuL*1e3:6.1f} mN  drift={cdrift*1e3:.1f} mNm  [{tag}]", flush=True)
                log[-1].update(cpu_tau=cpuT, cpu_tau_impl=cpuTi, cpu_tau_rk4=cpuTr,
                               cpu_L=cpuL, cpu_gap=gap, cpu_drift=cdrift,
                               cpu_variants=cpu_vars)
                # CPU-selected best: lowest ROBUST peak among MJX-feasible designs that keep
                # lift AND transfer between integrators (small euler<->implicit gap). The
                # transfer gate is what stops the optimizer shipping an integrator-hacked gait.
                cpu_lift_ok = (cpuL >= args.lift_floor
                               if (args.single_stage and args.lift_torque_tradeoff is None)
                               else cpuL >= 0.9 * cpuL0)
                if (is_feasible(Lbar, A_stroke) and cpu_lift_ok and transfers
                        and robust < cpu_best["tau"]):
                    cpu_best = {"score": robust, "tau": robust, "swing": swing, "theta": theta_eval,
                                "Lbar": Lbar, "Pin": Pin, "mjx_tau": tau_hat, "cpu_L": cpuL,
                                "cpu_euler": cpuT, "cpu_impl": cpuTi, "cpu_rk4": cpuTr, "cpu_gap": gap}
        stage_dir = args.out / f"stage_{round(tau_max*1e3)}mNm"
        dump(stage_best, log, out_dir=stage_dir)  # per-stage design for the demo
        dump(best, log)                            # running global best (interrupt-safe)
        print(f"  [stage {round(tau_max*1e3)} mNm] best tau={stage_best['tau']*1e3:.1f} mNm "
              f"swing={stage_best['swing']*1e3:.1f} mNm L={stage_best['Lbar']*1e3:.1f} mN "
              f"-> {stage_dir}")

    dump(best, log)
    print(f"\nbest design: tau={best['tau']*1e3:.1f} mNm  swing={best['swing']*1e3:.1f} mNm  "
          f"L={best['Lbar']*1e3:.1f} mN  (from {tau0*1e3:.1f} mNm / {swing0*1e3:.1f} mNm "
          f"baseline)  ->  {args.out}")

    # CPU-validate the winner(s) on the SAME scene (euler matches MJX; implicitfast
    # is the nominal physics). Records transfer so the autonomous tuner can gate on it.
    def validate_design(theta_v, label):
        cp, ta, pp, spp = theta_to_cpu(theta_v)
        cpu = {}
        integs = (("euler", "implicitfast", "rk4")
                  if (args.tri_integrator or args.rotate_batch) else ("euler", "implicitfast"))
        for integ in integs:
            try:
                L, T, P = cpu_validate_design(args.scene, cp, ta, integ, pitch_params=pp,
                                              strokeplane_params=spp,
                                              warmup=args.warmup_periods, periods=cpu_sp,
                                              phase_lag=phase_lag, timestep_ms=args.timestep,
                                              aero_substep=args.aero_substep, wavelet=wavelet_dict)
                cpu[integ] = {"mean_lift_N": L, "peak_torque_Nm": T, "supplied_power_W": P}
                print(f"[cpu/{integ}/{label}] lift={L*1e3:.1f} mN  peak_torque={T*1e3:.1f} mNm  "
                      f"power={P*1e3:.0f} mW")
            except Exception as e:
                cpu[integ] = {"error": str(e)[:200]}
                print(f"[cpu/{integ}/{label}] FAILED: {e}")
        return cpu

    report = {"baseline": {"cpu_euler_lift_N": cpuL0, "cpu_euler_peak_Nm": cpuT0,
                           "cpu_robust_peak_Nm": cpuT0r},
              "mjx_best": {"mjx": {"tau_hat": best["tau"], "swing": best["swing"],
                                   "Lbar": best["Lbar"]},
                           "cpu": validate_design(best["theta"], "mjx_best")}}
    # The CPU-selected best (--cpu-every) is the honest, integrator-robust peak-torque
    # winner -- save it separately so the demo doesn't ship a euler-hacked design.
    if cpu_best is not None and math.isfinite(cpu_best["tau"]):
        dump(cpu_best, log, out_dir=args.out / "cpu_best")
        _rk4 = cpu_best.get("cpu_rk4", cpu_best["cpu_impl"])
        print(f"\ncpu-selected best: robust tau={cpu_best['tau']*1e3:.1f} mNm "
              f"(euler {cpu_best['cpu_euler']*1e3:.1f}/impl {cpu_best['cpu_impl']*1e3:.1f}/"
              f"rk4 {_rk4*1e3:.1f})  L={cpu_best['Lbar']*1e3:.1f} mN  "
              f"(from cpu baseline {cpuT0r*1e3:.1f} mNm)  ->  {args.out / 'cpu_best'}")
        report["cpu_best"] = {"robust_peak_Nm": cpu_best["tau"], "cpu_gap": cpu_best["cpu_gap"],
                              "cpu": validate_design(cpu_best["theta"], "cpu_best")}

    # Full-matrix winner (rotate mode): best robust design that passed the in-loop
    # 12-case transfer gate. This is the deliverable of the rotating-batch run.
    if robust_best is not None and math.isfinite(robust_best["tau"]):
        dump(robust_best, log, out_dir=args.out / "robust_best")
        print(f"\nfull-matrix best (transfer-gated): robust tau={robust_best['tau']*1e3:.1f} mNm "
              f"liftmin={robust_best['Lbar']*1e3:.1f} mN  ->  {args.out / 'robust_best'}")
        report["robust_best"] = {"robust_peak_Nm": robust_best["tau"],
                                 "cpu": validate_design(robust_best["theta"], "robust_best")}

    # Holdout generalization: UNSEEN dts, never part of any training batch or the
    # full-check matrix. A design that holds up here learned physics, not a dt trick.
    if args.holdout_dts:
        winner = wname = None
        for cand, name in ((robust_best, "robust_best"), (cpu_best, "cpu_best"),
                           (best, "mjx_best")):
            if cand is not None and math.isfinite(cand["tau"]):
                winner, wname = cand, name
                break
        if winner is not None:
            print(f"\n[holdout] evaluating {wname} on unseen dts {tuple(args.holdout_dts)} ms")
            hold = {"design": wname, "mjx": [], "cpu": []}
            hold_sp = args.full_sample_periods if args.full_sample_periods else args.sample_periods
            for dm in args.holdout_dts:
                for ig in ("euler", "implicitfast", "rk4"):
                    f = _variant_metrics(ig, dm, sample_periods=hold_sp,
                                         avg_periods=max(1, int(hold_sp)))
                    m = [float(x) for x in jax.jit(f)(winner["theta"])]
                    hold["mjx"].append({"integrator": ig, "timestep_ms": dm,
                                        "L": m[0], "T": m[13], "T_lp": m[1], "drift": m[14]})
                    print(f"[holdout mjx {ig}@{dm:g}] lift={m[0]*1e3:.1f} mN "
                          f"peak={m[13]*1e3:.1f} mNm drift={m[14]*1e3:+.1f} mNm", flush=True)
            cp, ta, pp, spp = theta_to_cpu(winner["theta"])
            for dm in args.holdout_dts:
                for ig in ("euler", "implicitfast"):
                    try:
                        L, T, P = cpu_validate_design(
                            args.scene, cp, ta, ig, pitch_params=pp,
                            strokeplane_params=spp,
                            warmup=args.warmup_periods, periods=cpu_sp,
                            phase_lag=phase_lag, timestep_ms=dm,
                            aero_substep=args.aero_substep, wavelet=wavelet_dict)
                        hold["cpu"].append({"integrator": ig, "timestep_ms": dm, "L": L, "T": T})
                        print(f"[holdout cpu {ig}@{dm:g}] lift={L*1e3:.1f} mN "
                              f"peak={T*1e3:.1f} mNm", flush=True)
                    except Exception as e:
                        hold["cpu"].append({"integrator": ig, "timestep_ms": dm,
                                            "error": str(e)[:200]})
                        print(f"[holdout cpu {ig}@{dm:g}] FAILED: {e}")
            ts = [r["T"] for r in hold["mjx"] + hold["cpu"]
                  if "T" in r and math.isfinite(r["T"])]
            if ts:
                hold["robust_peak_Nm"] = max(ts)
                hold["spread"] = (max(ts) - min(ts)) / max(ts)
                print(f"[holdout] robust peak={max(ts)*1e3:.1f} mNm "
                      f"spread={hold['spread']*100:.0f}% over {len(ts)} unseen cases")
            report["holdout"] = hold
    (args.out / "cpu_validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
