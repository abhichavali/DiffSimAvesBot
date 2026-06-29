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
from jax.scipy.special import logsumexp

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.mjx.rollout import _match_dtypes
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest

WINGTIP_SITE = "L_wing_aero_blade_04"  # stroke-amplitude proxy (z excursion of tip)
G = 9.81
TAU_CURRICULUM_NM = (0.300, 0.250, 0.200, 0.150, 0.100)  # 300 -> 250 -> 200 -> 150 -> 100 mNm


_INTEGRATORS = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}


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


def setup(tg_path: Path, scene_path: Path, *, aero_substep: int = 4, integrator: str = "euler"):
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
    compile_tg(tg_path, scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = _INTEGRATORS[integrator]
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    control = load_control_for_morphology(tg_path)
    drive = rollout.build_drive_config(model, ramp_duration=0.0)
    pitch = wing_pitch.config_from_runtime(model, centers=rollout._pitch_centers(model, data))
    aero = aero_runtime.build_aero_config(model, substep=aero_substep)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)

    mx = mjx.put_model(model)
    dx0 = mjx.put_data(model, data)
    tendons = structural_tendon_pairs(model)
    motor_ids = motor_tendon_ids(model)
    return model, mx, dx0, control, drive, pitch, aero, tip_id, tendons, motor_ids


def structural_tendon_pairs(model):
    """L/R-mirrored structural tendon indices + base stiffness/rest-length arrays.

    Returns (l_ids, r_ids, base_stiffness(ntendon,), base_lengthspring(ntendon,2)).
    Each (l_ids[i], r_ids[i]) is one mirror pair sharing a tuned stiffness and
    pretension (the design enforces R = mirror of L), so 12 pairs -> 24 params.
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


def build_rollout(mx, dx0, drive, pitch, aero, tip_id, omega, K, warmup_steps, sample_steps,
                  *, steps_per_cycle, aero_substep, base_a0, tune_control, control_fixed,
                  tune_frequency, tune_tendons, tendons, tune_pitch=False, tune_motor=False,
                  motor_ids=None):
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
    l_ids, r_ids, base_stiff, base_lenspring = tendons
    record = jnp.concatenate(
        [jnp.zeros(warmup_steps, dtype=bool), jnp.ones(sample_steps, dtype=bool)]
    )
    n_full = sample_steps // steps_per_cycle  # whole cycles in the sample window (static)

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
            stiff_log = theta[i : i + 12]
            pre_log = theta[i + 12 : i + 24]
            i += 24
        if tune_pitch:  # 4: up-deadband, down-deadband, up-stiffness, down-stiffness
            pitch_log = theta[i : i + 4]
            i += 4
        if tune_motor:  # 2: shared motor-tendon stiffness, pretension
            motor_log = theta[i : i + 2]
        return ctrl, f_log, stiff_log, pre_log, pitch_log, motor_log

    def perturbed_model(stiff_log, pre_log, motor_log):
        # Log-factors (seeded exp(0)=1) scale the design base; exp keeps values >0.
        stiff, lenspring = base_stiff, base_lenspring
        if tune_tendons:  # per-pair structural, mirrored L=R
            sp = base_stiff[l_ids] * jnp.exp(stiff_log)
            stiff = stiff.at[l_ids].set(sp).at[r_ids].set(sp)
            lpl = base_lenspring[l_ids] * jnp.exp(pre_log)[:, None]
            lpr = base_lenspring[r_ids] * jnp.exp(pre_log)[:, None]
            lenspring = lenspring.at[l_ids].set(lpl).at[r_ids].set(lpr)
        if tune_motor:  # one shared stiffness + one shared pretension over the 4 motor tendons
            stiff = stiff.at[motor_ids].set(base_stiff[motor_ids] * jnp.exp(motor_log[0]))
            lenspring = lenspring.at[motor_ids].set(base_lenspring[motor_ids] * jnp.exp(motor_log[1]))
        return mx.replace(tendon_stiffness=stiff, tendon_lengthspring=lenspring)

    def metrics(theta):
        ctrl, f_log, stiff_log, pre_log, pitch_log, motor_log = split(theta)
        a0 = base_a0 * jnp.exp(ctrl[0])
        a, b = ctrl[1 : 1 + K], ctrl[1 + K :]
        omega_eff = omega * jnp.exp(f_log)
        mx_eff = perturbed_model(stiff_log, pre_log, motor_log) if (tune_tendons or tune_motor) else mx
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
            dx, state, cq, cnfw, i = carry
            phase = omega_eff * dx.time
            # a,b are modulation depths (fractions of a0) so one LR moves them with
            # real authority; absolute harmonic n is a0*a_n. cmd = a0*(1 + sum(...)).
            cmd = a0 * (1.0 + jnp.sum(a * jnp.cos(harmonics * phase) + b * jnp.sin(harmonics * phase)))
            ctrl_vec = jnp.zeros_like(dx.ctrl).at[flap_ids].set(cmd)
            dx = dx.replace(ctrl=ctrl_vec, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
            dx = wing_pitch.apply(dx, pc)  # pitch every step (matches CPU)
            # Aero recompute-and-hold every `aero_substep` steps; aero_state (FD prev
            # velocity) only advances on recompute, so the window spans substep*dt.
            def recompute(_):
                q, nfw, ns = aero_runtime.aero_wrench(mx_eff, dx, aero, state)
                return q, nfw, ns
            def hold(_):
                return cq, cnfw, state
            qfrc, net_force_world, state = jax.lax.cond(
                (i % aero_substep) == 0, recompute, hold, operand=None)
            dx = dx.replace(qfrc_applied=dx.qfrc_applied + qfrc)
            dx = mjx.step(mx_eff, dx)
            carry = _match_dtypes((dx, state, qfrc, net_force_world, i + 1), carry)
            tau = jnp.max(jnp.abs(dx.actuator_force[flap_ids]))
            # Motor mechanical power per crank actuator, summed: P = force*velocity.
            # P>0 motor drives (supplies), P<0 motor brakes (absorbs = inactive).
            power = jnp.sum(dx.actuator_force[flap_ids] * dx.actuator_velocity[flap_ids])
            tip_z = dx.site_xpos[tip_id, 2]
            keep = rec.astype(net_force_world.dtype)
            return carry, jnp.stack([net_force_world[2] * keep, tau, tip_z, power])

        state0 = aero_runtime.init_aero_state(aero)
        carry0 = (dx0, state0, jnp.zeros_like(dx0.qvel), jnp.zeros(3), jnp.int32(0))
        _, rec = jax.lax.scan(step, carry0, record)
        lift_z, tau, tip_z, power = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]

        Lbar = jnp.sum(lift_z) / sample_steps
        # soft-max (LSE) over the sample window approximates max_t |tau|, beta large.
        beta = 100.0
        tau_s = jnp.where(record, tau, -jnp.inf)
        tau_hat = (logsumexp(beta * tau_s) - math.log(sample_steps)) / beta
        tip_s = jnp.where(record, tip_z, jnp.nan)
        A_stroke = jnp.nanmax(tip_s) - jnp.nanmin(tip_s)
        S = jnp.sum(harmonics**2 * (a**2 + b**2))

        # Limit-cycle stability: cross-cycle variance of the lift trace. Reshape
        # the sample window to (n_full cycles, steps_per_cycle); var across cycles
        # at each phase is 0 iff every cycle traces the same orbit (stable limit
        # cycle). Needs >=2 whole cycles, else disabled.
        if n_full >= 2:
            lift_cyc = lift_z[warmup_steps : warmup_steps + n_full * steps_per_cycle]
            lift_cyc = lift_cyc.reshape(n_full, steps_per_cycle)
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
        P_brake = jnp.sum(jnp.maximum(0.0, -p_s)) / sample_steps   # mean absorbed power [W]
        P_scale = jnp.sum(jnp.abs(p_s)) / sample_steps + 1e-9      # mean |power|
        P_inactive = P_brake / P_scale                            # fraction of power that's braking
        return Lbar, tau_hat, A_stroke, S, C_lc, P_inactive

    return metrics


def cpu_validate_design(scene_path, ctrl_params, tendon_arrays, integrator,
                        *, pitch_params=None, periods=3.0, warmup=2.0):
    """Run the design on MuJoCo CPU (numpy aero) -> (mean_lift, peak_torque).

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
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.integrator = _INTEGRATORS[integrator]
    if tendon_arrays is not None:
        stiff, lenspring = tendon_arrays
        model.tendon_stiffness[:] = np.asarray(stiff)
        model.tendon_lengthspring[:] = np.asarray(lenspring)
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    if pitch_params is not None:  # apply the optimized feathering params (up/down deadband, stiffness)
        pitch.upstroke_pitch, pitch.downstroke_pitch, pitch.upstroke_stiffness, pitch.downstroke_stiffness = (
            float(x) for x in pitch_params)
    aids = [get_actuator_id(model, n) for n in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]

    def step():
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        phase = omega * data.time
        cmd = a0 * (1.0 + float(np.sum(a * np.cos(harm * phase) + b * np.sin(harm * phase))))
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = cmd
        lift = sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings)
        mujoco.mj_step(model, data)
        return lift, float(np.max(np.abs(data.actuator_force[aids])))

    period = math.tau / omega
    for _ in range(int(warmup * period / model.opt.timestep)):
        step()
    rec = [step() for _ in range(int(periods * period / model.opt.timestep))]
    L = np.array([r[0] for r in rec]); T = np.array([r[1] for r in rec])
    return float(L.mean()), float(T.max())


def _soft_hinge(x, s):
    """Smooth max(0, x): softplus(x/s)*s. C-infinity, no kink at 0."""
    return jax.nn.softplus(x / s) * s


def make_loss(metrics, *, lift_floor, lift_cap, lift_smooth, lift_above_weight,
              A_target, tau_max, weights, w_lc, w_power):
    """Scalar loss = -R, normalized per the design spec, with metrics as aux.

    Asymmetries (per the demo objective):
      * Lift band [lift_floor, lift_cap] (190-200 mN): below is penalized at full
        weight (strict floor), above only at ``lift_above_weight`` (moderate) so
        the optimizer is free to overshoot lift but never starve it.
      * Stroke amplitude: penalized ONLY when it drops below the baseline A_target
        (one-sided soft hinge). Flapping bigger than baseline is free; shrinking
        the flap is heavily penalized (that is the whole point of the demo gait).
    """
    wL, wtau, wK, ws = weights
    A_smooth = max(1e-9, 0.02 * A_target)  # ~2% band tolerance before the amp floor bites

    def loss(theta, tau_max):
        Lbar, tau_hat, A_stroke, S, C_lc, P_inactive = metrics(theta)
        # Lift band: strict below the floor, moderate above the cap.
        below = _soft_hinge(lift_floor - Lbar, lift_smooth)
        above = _soft_hinge(Lbar - lift_cap, lift_smooth) if lift_cap is not None else 0.0
        PhiL = -(below**2 + lift_above_weight * above**2) / lift_floor**2
        Ptau = (jnp.maximum(0.0, tau_hat - tau_max) ** 2) / tau_max**2
        # One-sided: penalize only A_stroke < A_target (shrinking the flap).
        amp_short = _soft_hinge(A_target - A_stroke, A_smooth)
        Ckin = amp_short**2 / A_target**2
        Clc = C_lc  # already dimensionless (relative cross-cycle lift variance)
        R = wL * PhiL - wtau * Ptau - wK * Ckin - ws * S - w_lc * Clc - w_power * P_inactive
        aux = (Lbar, tau_hat, A_stroke, S, Clc, P_inactive, -R)
        return -R, aux

    return jax.jit(jax.jacfwd(loss, has_aux=True))  # (grad, aux) in one fwd sweep


def load_control_seed(path: Path | None, K: int, drive, base_a0: float) -> list[float]:
    """Control seed [a0_log, a_1..a_K, b_1..b_K]; a0 stored as log-factor vs base_a0."""
    if path is None:
        a, b, a0 = list(drive.a), list(drive.b), float(drive.a0)
    else:
        d = json.loads(Path(path).read_text())
        a, b, a0 = list(d["a"]), list(d["b"]), float(d["a0"])
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
    p.add_argument("--lift-floor", type=float, default=0.190,
                   help="STRICT floor: penalize mean lift below this [N] (smooth softplus hinge)")
    p.add_argument("--lift-cap", type=float, default=0.200,
                   help="upper cap [N]; lift above it is penalized only at --lift-above-weight")
    p.add_argument("--lift-above-weight", type=float, default=0.25,
                   help="relative weight on the above-cap lift penalty (moderate vs the strict floor)")
    p.add_argument("--lift-smooth", type=float, default=0.005,
                   help="softplus smoothing scale [N] for the lift hinge")
    p.add_argument("--warmup-periods", type=float, default=2.0)
    p.add_argument("--sample-periods", type=float, default=2.0)
    p.add_argument("--aero-substep", type=int, default=4,
                   help="recompute aero every N steps, hold between (CPU parity; 1 = noisy every-step)")
    p.add_argument("--integrator", choices=list(_INTEGRATORS), default="euler",
                   help="physics integrator; euler is the MJX<->CPU-matching choice (default)")
    p.add_argument("--tau-min", type=float, default=0.100,
                   help="tightest torque-budget stage [Nm]; curriculum 0.30->0.25->0.20->0.15->0.10 "
                        "clamped to this. Stage 5 (100 mNm) is the demo target")
    p.add_argument("--added-mass", action="store_true",
                   help="enable the unsteady added-mass aero term. OFF by default: its MJX/CPU "
                        "implementations diverge ~4x and break transfer (see set_unsteady_aero)")
    p.add_argument("--iters-per-stage", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weights", type=float, nargs=4, default=(10.0, 10.0, 8.0, 0.01),
                   metavar=("wL", "wtau", "wK", "ws"))
    p.add_argument("--w-power", type=float, default=0.0,
                   help="weight on inactive (reactive/braking) motor power -- pushes elastic "
                        "tendon storage to handle decel instead of the motor (resonance)")
    p.add_argument("--w-lc", type=float, default=0.0,
                   help="weight on limit-cycle stability (cross-cycle lift variance); logged either "
                        "way. Default 0: gradient descent doesn't reduce it (chaos-corrupted variance "
                        "gradient), confirmed empirically")
    p.add_argument("--tune-frequency", action="store_true",
                   help="optimize the drive frequency (log-factor) -- the resonance lever for torque")
    # Tendon co-design: optimize per-pair (L=R) structural-tendon stiffness + pretension.
    p.add_argument("--tune-tendons", action="store_true",
                   help="optimize the 24 structural-tendon params (12 stiffness + 12 pretension)")
    p.add_argument("--tune-pitch", action="store_true",
                   help="optimize the 4 wing-pitch params (up/down deadband + stiffness) -- feathering")
    p.add_argument("--tune-motor-tendons", action="store_true",
                   help="optimize the 2 motor-tendon params (shared stiffness + pretension)")
    p.add_argument("--freeze-control", action="store_true",
                   help="hold the control signal fixed (use with --init-control)")
    p.add_argument("--init-control", type=Path, default=None,
                   help="warm-start control from a phase-1 theta_best.json")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "runs")
    args = p.parse_args()

    set_unsteady_aero(args.added_mass)  # default OFF -> MJX==CPU (transfer fix)
    tune_control = not args.freeze_control
    tune_tendons = args.tune_tendons
    tune_pitch = args.tune_pitch
    tune_motor = args.tune_motor_tendons
    if not (tune_control or tune_tendons or tune_pitch or tune_motor):
        p.error("nothing to optimize")

    model, mx, dx0, control, drive, pitch, aero, tip_id, tendons, motor_ids = setup(
        args.tg, args.scene, aero_substep=args.aero_substep, integrator=args.integrator)
    K = args.harmonics
    omega = float(control.omega)
    dt = model.opt.timestep
    period = control.period
    steps_per_cycle = max(1, round(period / dt))
    warmup_steps = math.ceil(args.warmup_periods * period / dt)
    sample_steps = math.ceil(args.sample_periods * period / dt)
    motor_dir = load_morphology(args.tg).motors.motor_direction

    base_a0 = float(drive.a0)
    control_seed = jnp.asarray(load_control_seed(args.init_control, K, drive, base_a0),
                               dtype=jnp.float32)
    metrics = build_rollout(mx, dx0, drive, pitch, aero, tip_id, omega, K,
                            warmup_steps, sample_steps, steps_per_cycle=steps_per_cycle,
                            aero_substep=args.aero_substep, base_a0=base_a0,
                            tune_control=tune_control, control_fixed=control_seed,
                            tune_frequency=args.tune_frequency, tune_tendons=tune_tendons,
                            tendons=tendons, tune_pitch=tune_pitch, tune_motor=tune_motor,
                            motor_ids=motor_ids)

    # theta packs active groups in order:
    # [control(2K+1)] [f_log(1)] [tendon(24)] [pitch(4)] [motor(2)]
    parts = []
    if tune_control:
        parts.append(control_seed)
    if args.tune_frequency:
        parts.append(jnp.zeros(1, dtype=jnp.float32))  # f_log, seeded at base frequency
    if tune_tendons:
        parts.append(jnp.zeros(24, dtype=jnp.float32))  # log-factors, seeded at 1x
    if tune_pitch:
        parts.append(jnp.zeros(4, dtype=jnp.float32))   # pitch deadband/stiffness log-factors
    if tune_motor:
        parts.append(jnp.zeros(2, dtype=jnp.float32))   # motor stiffness/pretension log-factors
    theta = jnp.concatenate(parts)

    # Baseline A* = stroke amplitude of the seed (prevents collapse).
    Lbar0, tau0, A_target, S0, Clc0, Pin0 = (float(x) for x in metrics(theta))
    print(f"[setup] tune_control={tune_control} tune_tendons={tune_tendons} "
          f"params={theta.shape[0]} steps={warmup_steps + sample_steps} "
          f"steps/cycle={steps_per_cycle} w_lc={args.w_lc}")
    print(f"[baseline] Lbar={Lbar0*1e3:.1f} mN  tau_hat={tau0*1e3:.1f} mNm  "
          f"A_stroke={A_target*1e3:.2f} mm  S={S0:.3g}  C_lc={Clc0:.3g}")
    cap_str = f"{args.lift_cap*1e3:.0f}" if args.lift_cap is not None else "none"
    print(f"[target]   lift_floor={args.lift_floor*1e3:.0f} mN  cap={cap_str}  "
          f"A*={A_target*1e3:.2f} mm  w_lc={args.w_lc}")

    grad_fn = make_loss(metrics, lift_floor=args.lift_floor, lift_cap=args.lift_cap,
                        lift_smooth=args.lift_smooth, lift_above_weight=args.lift_above_weight,
                        A_target=A_target, tau_max=None,
                        weights=tuple(args.weights), w_lc=args.w_lc, w_power=args.w_power)

    # Hand-rolled Adam (no optax dependency).
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0

    args.out.mkdir(parents=True, exist_ok=True)
    tnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(k))[2:]
              for k in np.asarray(tendons[0])]
    base_stiff = np.asarray(tendons[2])
    base_lenspring = np.asarray(tendons[3])
    l_ids = np.asarray(tendons[0])

    base_freq = control.fundamental_frequency_hz

    def dump(design, log, out_dir=args.out):
        """Checkpoint the selected design to disk (interrupted run isn't lost)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out = np.asarray(design["theta"])
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))
        i = 0
        freq_hz = base_freq
        if tune_control:
            ctrl = out[i : i + 2 * K + 1]; i += 2 * K + 1
            a0 = base_a0 * math.exp(ctrl[0])  # a0 log-factor -> absolute (actuator frame)
            a_out, b_out = a0 * ctrl[1 : 1 + K], a0 * ctrl[1 + K :]  # depth -> absolute a_n
        else:
            a0, a_out, b_out = base_a0, np.zeros(K), np.zeros(K)
        if args.tune_frequency:
            freq_hz = base_freq * math.exp(float(out[i])); i += 1
        if tune_control:
            lines = [f"harmonics = {K}", f"fundamental_frequency = {freq_hz:g} Hz",
                     f"a0 = {a0 / motor_dir:.6g}"]
            for j in range(K):
                lines.append(f"a_{j+1} = {a_out[j] / motor_dir:.6g}")
                lines.append(f"b_{j+1} = {b_out[j] / motor_dir:.6g}")
            (out_dir / "control_optimized.tg").write_text("\n".join(lines) + "\n")
            (out_dir / "theta_best.json").write_text(json.dumps(
                {"a0": float(a0), "a": [float(x) for x in a_out], "b": [float(x) for x in b_out],
                 "frequency_hz": float(freq_hz), "omega": float(math.tau * freq_hz),
                 "motor_direction": motor_dir, "Lbar": design["Lbar"],
                 "tau_hat": design["tau"]}, indent=2))
        if tune_tendons:
            stiff_log, pre_log = out[i : i + 12], out[i + 12 : i + 24]; i += 24
            tendon_out = {
                tnames[k]: {
                    "stiffness": float(base_stiff[l_ids[k]] * math.exp(stiff_log[k])),
                    "stiffness_factor": float(math.exp(stiff_log[k])),
                    "pretension_factor": float(math.exp(pre_log[k])),
                    "rest_length": float(base_lenspring[l_ids[k]][0] * math.exp(pre_log[k])),
                }
                for k in range(12)
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

    log = []
    # Select the lowest peak-torque design that still clears the lift floor (the
    # actual goal), NOT global-best-loss -- the curriculum makes the loose stage
    # win on loss while doing the least torque reduction.
    best = {"tau": math.inf, "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
    curriculum = tuple(sorted({max(args.tau_min, s) for s in TAU_CURRICULUM_NM}, reverse=True))
    print(f"[curriculum] tau_max stages (mNm): {[round(x*1e3) for x in curriculum]}")
    for tau_max in curriculum:
        print(f"\n=== stage tau_max = {tau_max*1e3:.0f} mNm ===")
        # Per-stage winner: lowest peak-torque design that clears the lift floor AT
        # THIS budget. Saved to its own subdir so the demo has one design per stage.
        stage_best = {"tau": math.inf, "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
        for it in range(args.iters_per_stage):
            theta_eval = theta  # metrics below describe THIS theta (pre-update)
            grad, aux = grad_fn(theta, tau_max)
            Lbar, tau_hat, A_stroke, S, Clc, Pin, loss = (float(x) for x in aux)
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad**2
            mhat = m / (1 - b1**t)
            vhat = v / (1 - b2**t)
            theta = theta - args.lr * mhat / (jnp.sqrt(vhat) + eps)

            log.append({"stage_tau_max": tau_max, "iter": it, "loss": loss, "Lbar": Lbar,
                        "tau_hat": tau_hat, "A_stroke": A_stroke, "S": S, "C_lc": Clc, "P_inactive": Pin})
            if Lbar >= args.lift_floor and tau_hat < best["tau"]:
                best = {"tau": tau_hat, "theta": theta_eval, "Lbar": Lbar, "Pin": Pin}
            if Lbar >= args.lift_floor and tau_hat < stage_best["tau"]:
                stage_best = {"tau": tau_hat, "theta": theta_eval, "Lbar": Lbar, "Pin": Pin}
            print(f"  it{it:03d} loss={loss:.4f}  L={Lbar*1e3:6.1f}mN  "
                  f"tau={tau_hat*1e3:6.1f}mNm  A={A_stroke*1e3:5.2f}mm  S={S:.3g}  "
                  f"Clc={Clc:.3g}  Pin={Pin:.3g}", flush=True)
        stage_dir = args.out / f"stage_{round(tau_max*1e3)}mNm"
        dump(stage_best, log, out_dir=stage_dir)  # per-stage design for the demo
        dump(best, log)                            # running global best (interrupt-safe)
        feasible = "OK" if stage_best["tau"] <= tau_max else "MISS"
        print(f"  [stage {round(tau_max*1e3)} mNm] best tau={stage_best['tau']*1e3:.1f} mNm "
              f"L={stage_best['Lbar']*1e3:.1f} mN [{feasible}] -> {stage_dir}")

    dump(best, log)
    print(f"\nbest design: tau={best['tau']*1e3:.1f} mNm  L={best['Lbar']*1e3:.1f} mN "
          f"(from {tau0*1e3:.1f} mNm baseline)  ->  {args.out}")

    # CPU-validate the best design on the SAME scene (euler matches MJX; implicitfast
    # is the nominal physics). Records transfer so the autonomous tuner can gate on it.
    out = np.asarray(best["theta"]); j = 0
    if tune_control:
        ctrl = out[j : j + 2 * K + 1]; j += 2 * K + 1
        a0 = base_a0 * math.exp(ctrl[0]); a_dep, b_dep = ctrl[1 : 1 + K], ctrl[1 + K :]
    else:
        a0, a_dep, b_dep = base_a0, np.zeros(K), np.zeros(K)
    f_log = out[j] if args.tune_frequency else 0.0
    if args.tune_frequency:
        j += 1
    tendon_arrays = None
    if tune_tendons or tune_motor:
        bs = np.asarray(tendons[2]).copy(); bl = np.asarray(tendons[3]).copy()
    if tune_tendons:
        li = np.asarray(tendons[0]); ri = np.asarray(tendons[1])
        sl, pl = out[j : j + 12], out[j + 12 : j + 24]; j += 24
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
    if tune_tendons or tune_motor:
        tendon_arrays = (bs, bl)
    cpu = {}
    for integ in ("euler", "implicitfast"):
        try:
            L, T = cpu_validate_design(args.scene, (a0, a_dep, b_dep, omega * math.exp(f_log)),
                                       tendon_arrays, integ, pitch_params=pitch_params)
            cpu[integ] = {"mean_lift_N": L, "peak_torque_Nm": T}
            print(f"[cpu/{integ}] lift={L*1e3:.1f} mN  peak_torque={T*1e3:.1f} mNm")
        except Exception as e:
            cpu[integ] = {"error": str(e)[:200]}
            print(f"[cpu/{integ}] FAILED: {e}")
    (args.out / "cpu_validation.json").write_text(json.dumps(
        {"mjx": {"tau_hat": best["tau"], "Lbar": best["Lbar"], "P_inactive": best.get("Pin")}, "cpu": cpu}, indent=2))


if __name__ == "__main__":
    main()
