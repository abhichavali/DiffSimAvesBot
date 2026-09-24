"""The gradient co-design run behind opt/optimize_control.py.

run(cfg) takes the parsed CLI namespace and:
  1. builds the rig, the genome and the primary MJX rollout, plus its robustness variants:
     --tri-integrator / --tri-timestep rollouts every step, or a --rotate-batch bank of
     (integrator, dt, warmup) rollouts drawn as random mini-batches;
  2. scores the seed (the baseline every objective is normalized by);
  3. runs hand-rolled Adam on jacfwd of the robust loss over a tau_max curriculum (or one
     --single-stage), guarding against NaN steps and freezing gene groups on request;
  4. selects candidates per iteration or at the full-matrix MJX check (robust_best), and
     certifies them on CPU MuJoCo in the transfer gate (cpu_best);
  5. writes every winner as a design dir, then CPU-validates the winners and evaluates
     the unseen --holdout-dts into cpu_validation.json.
"""

from __future__ import annotations

import json
import math
import random

import jax
import jax.numpy as jnp
import numpy as np

from codesign.artifacts import write_design
from codesign.cpu_check import cpu_gate, cpu_rollout, to_cpu
from codesign.genome import Genome, initial_theta, load_control_seed, wavelet_dictionary
from codesign.loss import LossAux, make_loss, objective_norm, objective_value, variant_set
from codesign.metrics import Metrics, build_rollout
from codesign.rig import (flap_t1_mask, set_unsteady_aero, setup, step_counts, strokeplane_info,
                          variant)
from morphology.tg_parser import load_morphology

TAU_CURRICULUM_NM = (0.300, 0.250, 0.200, 0.150, 0.100)  # 300 -> 250 -> 200 -> 150 -> 100 mNm
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def full_matrix(bank_jit, theta):
    """Forward-evaluate EVERY bank cell, scored on the EXACT per-cycle peak (tau_peak):
    the Lp proxy under-reads narrow torque spikes by 15-25% and is duty-cycle hackable.
    Returns (rows, robust peak, gap, min lift, max |drift|)."""
    rows = []
    for (ig, dm), f in bank_jit.items():
        m = Metrics(*(float(x) for x in f(theta)))
        rows.append({"integrator": ig, "timestep_ms": dm, "L": m.Lbar, "T": m.tau_peak,
                     "T_lp": m.tau_hat, "drift": m.tau_drift, "A": m.A_stroke, "swing": m.swing,
                     "P": m.P_supplied, "overlap": m.C_overlap, "tau_rms": m.tau_rms,
                     "power_crest": m.power_crest, "lift_ripple": m.lift_ripple})
    rob = max(r["T"] for r in rows)
    gap = (rob - min(r["T"] for r in rows)) / max(rob, 1e-9)
    driftmax = max(abs(r["drift"]) for r in rows)
    return rows, rob, gap, min(r["L"] for r in rows), driftmax


def robust_row(rows, rob, liftmin) -> dict:
    """The worst cell of the matrix for whatever is minimized (lift: the MIN)."""
    return {"tau": rob, "power": max(r["P"] for r in rows), "lift": liftmin,
            "swing": max(r["swing"] for r in rows), "overlap": max(r["overlap"] for r in rows),
            "tau_rms": max(r["tau_rms"] for r in rows),
            "power_crest": max(r["power_crest"] for r in rows),
            "lift_ripple": max(r["lift_ripple"] for r in rows)}


def run(cfg) -> None:
    set_unsteady_aero(cfg.added_mass)  # default OFF -> MJX==CPU
    rig = setup(cfg.tg, cfg.scene, aero_substep=cfg.aero_substep, integrator=cfg.integrator,
                timestep_ms=cfg.timestep, lock_wings=cfg.lock_wings)
    model, control, drive = rig.model, rig.control, rig.drive
    morph = load_morphology(cfg.tg)
    genome = Genome(K=cfg.harmonics, n_tend=int(np.asarray(rig.tendons[0]).shape[0]),
                    control=not cfg.freeze_control, frequency=cfg.tune_frequency,
                    tendons=cfg.tune_tendons, pitch=cfg.tune_pitch, motor=cfg.tune_motor_tendons,
                    strokeplane=cfg.tune_strokeplane, wavelet=cfg.wavelet)
    K, slices = genome.K, genome.slices()
    wavelet_dict = wavelet_dictionary(2 * K) if cfg.wavelet else None
    omega = float(control.omega)
    dt = model.opt.timestep
    period = control.period
    steps_per_cycle, warmup_steps, sample_steps = step_counts(
        period, dt, cfg.warmup_periods, cfg.sample_periods)
    tendon_freeze = None  # --tendon-params: the tendon half whose gradient is zeroed
    if genome.tendons and cfg.tendon_params != "both":
        tendon_freeze = slices["pre" if cfg.tendon_params == "stiffness" else "stiff"]
    control_seed = jnp.asarray(load_control_seed(cfg.init_control, K, drive, float(drive.a0),
                                                 wavelet=cfg.wavelet), dtype=jnp.float32)
    t1_mask = flap_t1_mask(model, drive)
    sp_info = strokeplane_info(model, morph, cfg.sp_max) if genome.strokeplane else None

    def rollout(r, spc, wu, sm, avg_periods=3):
        return build_rollout(r, genome, omega, warmup_steps=wu, sample_steps=sm,
                             steps_per_cycle=spc, aero_substep=cfg.aero_substep,
                             control_fixed=control_seed, flap_t1_mask=t1_mask,
                             strokeplane=sp_info, avg_periods=avg_periods)

    def variant_metrics(integ, dt_s, *, warmup_periods=cfg.warmup_periods,
                        sample_periods=cfg.sample_periods, avg_periods=3, qvel_noise=None):
        """Rollout of the SAME design under (integrator, timestep dt_s [s]): the robustness
        variants, rotating batch members, full-matrix cells and holdout cells."""
        mx_v, aero_v = variant(model, integ, dt_s, cfg.aero_substep)
        dx = rig.dx0 if qvel_noise is None else rig.dx0.replace(
            qvel=rig.dx0.qvel + jnp.asarray(qvel_noise, dtype=rig.dx0.qvel.dtype))
        return rollout(rig._replace(mx=mx_v, aero=aero_v, dx0=dx),
                       *step_counts(period, dt_s, warmup_periods, sample_periods), avg_periods)

    metrics = rollout(rig, steps_per_cycle, warmup_steps, sample_steps)
    # --tri-integrator: the same design under implicitfast + rk4 in every loss evaluation;
    # --tri-timestep: under euler at dt*factor, so a dt-specific trick is priced in the
    # gradient itself.
    impl = rk4 = None
    if cfg.tri_integrator:
        impl, rk4 = variant_metrics("implicitfast", dt), variant_metrics("rk4", dt)
    extra = []
    if cfg.tri_timestep:
        factors = tuple(cfg.tri_timestep_factors) if cfg.tri_timestep_factors is not None else (0.5, 2.0)
        extra = [variant_metrics(cfg.integrator, dt * f) for f in factors if abs(f - 1.0) >= 1e-9]

    # --rotate-batch: a BANK of (integrator, dt) cells, certified on a LONG window (>= one
    # torque-beat period) so slow transients show as drift; training batches keep the short
    # window and rely on warmup jitter instead.
    bank_jit = None
    if cfg.rotate_batch:
        dt_bank = tuple(cfg.dt_bank) if cfg.dt_bank else (dt * 1e3,)
        full_sp = cfg.full_sample_periods if cfg.full_sample_periods else cfg.sample_periods
        bank = {(ig, dm): variant_metrics(ig, dm * 1e-3, sample_periods=full_sp,
                                          avg_periods=max(1, int(full_sp)))
                for ig in cfg.bank_integrators for dm in dt_bank}
        bank_jit = {k: jax.jit(f) for k, f in bank.items()}

    theta = initial_theta(genome, control_seed, cfg.init_codesign, cfg.lock_wings)
    lr_scale = jnp.ones_like(theta)
    if genome.strokeplane and cfg.sp_lr_mult != 1.0:
        lr_scale = lr_scale.at[slices["sp"]].set(cfg.sp_lr_mult)

    # Baseline: A* (the stroke amplitude to keep) and every objective's normalizer.
    m0 = Metrics(*(float(x) for x in metrics(theta)))
    A_target = m0.A_stroke
    print(f"[setup] tune_control={genome.control} tune_tendons={genome.tendons} "
          f"n_tend={genome.n_tend} params={theta.shape[0]} steps={warmup_steps + sample_steps} "
          f"steps/cycle={steps_per_cycle} w_lc={cfg.w_lc} wK={cfg.weights[2]}")
    print(f"[baseline] Lbar={m0.Lbar*1e3:.1f} mN  tau_hat={m0.tau_hat*1e3:.1f} mNm  "
          f"tau_peak={m0.tau_peak*1e3:.1f} mNm  drift={m0.tau_drift*1e3:+.1f} mNm  "
          f"A_stroke={A_target*1e3:.2f} mm  power={m0.P_supplied*1e3:.0f} mW  S={m0.S:.3g}  "
          f"C_lc={m0.C_lc:.3g}  C_valid={m0.C_valid:.3g}  alpha={math.degrees(m0.alpha_pk):.0f}deg "
          f"k={m0.k_eff:.3g} conv={m0.conv_err:.2e}  Pin={m0.P_inactive:.3g}  overlap={m0.C_overlap:.3g}")
    cap_str = f"{cfg.lift_cap*1e3:.0f}" if cfg.lift_cap is not None else "none"
    print(f"[target]   lift_floor={cfg.lift_floor*1e3:.0f} mN  cap={cap_str}  "
          f"A*={A_target*1e3:.2f} mm (wK={cfg.weights[2]})  w_power={cfg.w_power} (mean supplied)")

    obj_norm = objective_norm(cfg.objective, m0, cfg.lift_floor)

    def obj_of(row):
        return objective_value(cfg.objective, row, cfg.lift_floor, obj_norm)

    # Lift reference for the tradeoff: the FLOOR, not the baseline (anchoring at baseline
    # made every mN below it cost 5:1, so the optimizer hoarded lift).
    lift_ref = cfg.lift_floor if cfg.lift_torque_tradeoff else m0.Lbar
    loss_kwargs = dict(lift_floor=cfg.lift_floor, lift_cap=cfg.lift_cap,
                       lift_smooth=cfg.lift_smooth, A_target=A_target,
                       weights=tuple(cfg.weights), w_lc=cfg.w_lc, w_power=cfg.w_power,
                       w_overlap=cfg.w_overlap, w_swing=cfg.w_swing, lift0=lift_ref,
                       tradeoff=cfg.lift_torque_tradeoff, w_gap=cfg.w_gap,
                       amp_band_pct=cfg.amp_band_pct, amp_floor=cfg.amp_floor,
                       w_drift=cfg.w_drift, objective=cfg.objective, obj_norm=obj_norm,
                       tau_guard=cfg.tau_guard)
    grad_fn = make_loss(variant_set(metrics, impl, rk4, extra), **loss_kwargs)

    # Rotating mini-batches: random K-subsets of the bank, each with >= 2 integrators and
    # >= 2 dts (so every batch exercises both robustness axes), each its own jitted loss.
    rotate_fns = None
    rng = random.Random(cfg.batch_seed)
    if cfg.rotate_batch:
        keys = list(bank)
        kbatch = min(cfg.rotate_batch, len(keys))

        def _ok(bt):
            return (len({v[0] for v in bt}) >= min(2, kbatch)
                    and (len({v[1] for v in bt}) >= min(2, kbatch) or len(dt_bank) == 1))

        batches = []
        for _ in range(1000):
            if len(batches) >= cfg.n_batches:
                break
            bt = tuple(sorted(rng.sample(keys, kbatch)))
            if _ok(bt) and bt not in batches:
                batches.append(bt)
        if not batches:
            raise SystemExit("--rotate-batch: no valid batch composition from the bank")
        rotate_fns = []
        # Warmup jitter: each member also draws its warmup length, so the short sample
        # window lands at a different point of any slow torque beat per member.
        wu_bank = tuple(cfg.warmup_bank) if cfg.warmup_bank else (cfg.warmup_periods,)
        for centers in batches:
            # Floor jittered dts at the smallest bank center.
            bt = tuple(
                (ig, max(dm * (1.0 + rng.uniform(-cfg.dt_jitter, cfg.dt_jitter)), min(dt_bank)),
                 rng.choice(wu_bank))
                for ig, dm in centers
            )
            ms = [variant_metrics(ig, dm * 1e-3, warmup_periods=wu) for ig, dm, wu in bt]
            labels = [f"{ig}@{dm:g}/w{wu:g}" for ig, dm, wu in bt]
            if cfg.precision_perturb:
                ig, dm, wu = bt[0]
                noise = np.asarray([rng.gauss(0.0, cfg.precision_perturb)
                                    for _ in range(int(rig.dx0.qvel.shape[0]))])
                ms.append(variant_metrics(ig, dm * 1e-3, warmup_periods=wu, qvel_noise=noise))
                labels.append(f"{ig}@{dm:g}/w{wu:g}+eps")
            rotate_fns.append((tuple(labels), make_loss(variant_set(*ms[:3], extra=ms[3:]),
                                                        **loss_kwargs)))
        print(f"[rotate] bank={len(bank)} center variants (3 integrators x dts {dt_bank} ms), "
              f"{len(batches)} batches of {kbatch}"
              f" jitter={cfg.dt_jitter:g} precision_perturb={cfg.precision_perturb:g}:")
        for labels, _ in rotate_fns:
            print("  " + " + ".join(labels))

    adam_m = jnp.zeros_like(theta)
    adam_v = jnp.zeros_like(theta)
    t = 0

    cfg.out.mkdir(parents=True, exist_ok=True)

    def dump(design, log, out_dir=cfg.out):
        write_design(out_dir, design, log, genome=genome, control_seed=control_seed, rig=rig,
                     tg=cfg.tg, morph=morph, strokeplane=sp_info)

    def cpu_design(theta_v):
        return to_cpu(genome, theta_v, control_seed, rig, omega, sp_info)

    three_integrators = cfg.tri_integrator or cfg.rotate_batch
    cpu_sp = cfg.cpu_sample_periods if cfg.cpu_sample_periods else cfg.sample_periods
    cpu_kw = dict(warmup=cfg.warmup_periods, periods=cpu_sp,
                  phase_lag=morph.motors.crank_phase_lag_rad, aero_substep=cfg.aero_substep,
                  wavelet=wavelet_dict)
    cpu_cells = [(cfg.integrator, cfg.timestep)]
    if three_integrators:
        cpu_cells += [("implicitfast", cfg.timestep), ("rk4", cfg.timestep)]
    cpu_cells += [(cfg.integrator, None if cfg.timestep is None else cfg.timestep * f)
                  for f in cfg.cpu_timestep_factors]

    # CPU gate baselines: the first in-loop check seeds them (a startup CPU eval cost
    # ~45 min on a design whose numbers are already known).
    cpuL0 = cpuT0 = cpuT0r = cpu_obj0 = float("nan")
    cpu_best = None
    if cfg.cpu_every:
        cpu_best = {"score": math.inf, "tau": math.inf, "swing": float("nan"),
                    "theta": theta, "Lbar": float("nan"), "Pin": float("nan"), "mjx_tau": float("nan")}
    # Full-matrix gate (rotate mode): the first in-loop check seeds rob0/liftmin0.
    robust_best = None
    rob0 = liftmin0 = float("nan")
    if cfg.rotate_batch:
        robust_best = {"score": math.inf, "tau": math.inf, "swing": float("nan"),
                       "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}

    log = []
    # Selection (--single-stage): lowest objective (+ swing) keeping the baseline flap
    # amplitude. Otherwise (curriculum): lowest objective clearing the lift floor.
    if cfg.amp_floor is not None:
        amp_gate_lo, amp_gate_hi = cfg.amp_floor, float("inf")
    else:
        amp_gate_lo = A_target * (1.0 - cfg.amp_band_pct)
        amp_gate_hi = A_target * (1.0 + cfg.amp_band_pct)
    # obj0: the baseline in the objective's own units, so the lift trade converts right.
    obj0 = obj_of({"tau": m0.tau_hat, "power": m0.P_supplied, "lift": m0.Lbar, "swing": m0.swing,
                   "overlap": m0.C_overlap, "tau_rms": m0.tau_rms, "power_crest": m0.power_crest,
                   "lift_ripple": m0.lift_ripple})

    def score_of(metric, swing, Lbar):
        if not cfg.single_stage:
            return metric
        s = metric + cfg.score_swing * swing
        if cfg.lift_torque_tradeoff:  # lift drop -> objective-equivalent at N:1
            drop = max(0.0, float(lift_ref) - Lbar) / (abs(float(lift_ref)) + 1e-9)
            gain = max(0.0, Lbar - float(lift_ref)) / (abs(float(lift_ref)) + 1e-9)
            s += cfg.lift_torque_tradeoff * obj0 * drop
            if cfg.objective == "torque":  # power/energy: no linear credit (see make_loss)
                s -= obj0 * gain / cfg.lift_torque_tradeoff
        return s

    def is_feasible(Lbar, A_stroke):
        if cfg.single_stage:
            ok = amp_gate_lo <= A_stroke <= amp_gate_hi
            if cfg.lift_torque_tradeoff is None:
                # No trade guards lift and the amp band may be vacuous: the floor is the
                # selection contract.
                ok = ok and Lbar >= cfg.lift_floor
            return ok
        return Lbar >= cfg.lift_floor

    best = {"score": math.inf, "tau": math.inf, "power": math.inf, "swing": float("nan"),
            "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
    if cfg.single_stage:
        curriculum = (m0.tau_hat,)  # peak torque normalized by the baseline
    else:
        curriculum = tuple(sorted({max(cfg.tau_min, s) for s in TAU_CURRICULUM_NM}, reverse=True))
    print(f"[curriculum] tau_max stages (mNm): {[round(x*1e3) for x in curriculum]}")
    total_train_iters = max(1, len(curriculum) * cfg.iters_per_stage - 1)
    for stage_i, tau_max in enumerate(curriculum):
        print(f"\n=== stage tau_max = {tau_max*1e3:.0f} mNm ===")
        # Per-stage winner: saved to its own subdir so the demo has one design per stage.
        stage_best = {"score": math.inf, "tau": math.inf, "power": math.inf, "swing": float("nan"),
                      "theta": theta, "Lbar": float("nan"), "Pin": float("nan")}
        lr_cur = cfg.lr  # halved on each divergence
        last_good = theta  # last theta that produced a finite gradient
        for it in range(cfg.iters_per_stage):
            train_i = stage_i * cfg.iters_per_stage + it
            beta_frac = min(1.0, train_i / total_train_iters)
            robust_beta = cfg.softmax_beta_start + (
                cfg.softmax_beta_end - cfg.softmax_beta_start
            ) * beta_frac
            # Phase boundary: snapshot the certified winners of the tendon/motor-only phase
            # before the harmonics move ("what did harmonics buy" = diff vs *_preharmonics).
            if cfg.control_after and it == cfg.control_after:
                if robust_best is not None and math.isfinite(robust_best["tau"]):
                    dump(robust_best, log, out_dir=cfg.out / "robust_best_preharmonics")
                if cpu_best is not None and math.isfinite(cpu_best["tau"]):
                    dump(cpu_best, log, out_dir=cfg.out / "cpu_best_preharmonics")
                print(f"  [phase] it{it:03d}: harmonics UNLOCKED; pre-harmonics bests "
                      f"snapshotted to *_preharmonics", flush=True)
            theta_eval = theta  # the metrics below describe THIS theta (pre-update)
            batch_keys = None
            if rotate_fns:
                batch_keys, batch_fn = rotate_fns[rng.randrange(len(rotate_fns))]
            else:
                batch_fn = grad_fn
            grad, aux = batch_fn(theta, tau_max, robust_beta)
            # NaN guard: an unstable (euler-divergent) step would poison Adam for the rest
            # of the run. Revert to the last finite design, clear momentum, halve the step.
            if not bool(jnp.isfinite(grad).all()):
                theta, lr_cur = last_good, lr_cur * 0.5
                adam_m = jnp.zeros_like(theta); adam_v = jnp.zeros_like(theta); t = 0
                print(f"  it{it:03d} NaN/diverged -> revert, lr={lr_cur:.2e}", flush=True)
                continue
            last_good = theta
            if cfg.control_after and it < cfg.control_after:
                if genome.control:  # tendons/motor first; --a0-free keeps the frequency live
                    ctrl = slices["control"]
                    grad = grad.at[ctrl.start + (1 if cfg.a0_free else 0):ctrl.stop].set(0.0)
                if genome.strokeplane:
                    grad = grad.at[slices["sp"]].set(0.0)  # unlocks with the harmonics
            if tendon_freeze is not None:
                grad = grad.at[tendon_freeze].set(0.0)
            a = LossAux(*(float(x) for x in aux))
            t += 1
            adam_m = ADAM_B1 * adam_m + (1 - ADAM_B1) * grad
            adam_v = ADAM_B2 * adam_v + (1 - ADAM_B2) * grad**2
            mhat = adam_m / (1 - ADAM_B1**t)
            vhat = adam_v / (1 - ADAM_B2**t)
            theta = theta - lr_cur * lr_scale * mhat / (jnp.sqrt(vhat) + ADAM_EPS)

            log.append({"stage_tau_max": tau_max, "iter": it, "loss": a.loss, "Lbar": a.Lbar,
                        "tau_hat": a.tau_hat, "tau_impl": a.tau_impl, "tau_rk4": a.tau_rk4,
                        "tau_peak": a.tau_peak, "tau_drift": a.tau_drift,
                        "tau_robust": a.tau_robust, "mjx_gap": a.tau_spread / max(a.tau_robust, 1e-9),
                        "A_stroke": a.A_stroke, "power": a.P_supplied, "swing": a.swing,
                        "S": a.S, "C_lc": a.C_lc, "P_inactive": a.P_inactive, "overlap": a.C_overlap,
                        "C_valid": a.C_valid, "alpha_pk": a.alpha_pk, "k_eff": a.k_eff,
                        "conv_err": a.conv_err, "softmax_beta": robust_beta})
            # Select on the ROBUST peak so the winner is integrator-agnostic by construction.
            row = {"tau": a.tau_robust, "power": a.P_supplied, "lift": a.Lbar, "swing": a.swing,
                   "overlap": a.C_overlap, "tau_rms": a.tau_rms, "power_crest": a.power_crest,
                   "lift_ripple": a.lift_ripple}
            score = score_of(obj_of(row), a.swing, a.Lbar)
            sel = {"score": score, "tau": a.tau_robust, "power": a.P_supplied, "swing": a.swing,
                   "theta": theta_eval, "Lbar": a.Lbar, "Pin": a.P_inactive}
            # Rotate mode selects at the full-matrix check (a batch is an optimistic
            # subset) -- unless full checks are off, when nothing else would select.
            select_here = (not rotate_fns) or not cfg.full_check_every
            if select_here and is_feasible(a.Lbar, a.A_stroke) and score < best["score"]:
                best = sel
            if select_here and is_feasible(a.Lbar, a.A_stroke) and score < stage_best["score"]:
                stage_best = sel
            gap_pct = a.tau_spread / max(a.tau_robust, 1e-9) * 100
            if rotate_fns:
                tri = f" [{'/'.join(batch_keys)} rob={a.tau_robust*1e3:.1f} gap={gap_pct:.0f}%]"
            else:
                tri = (f" [e{a.tau_hat*1e3:.0f}/i{a.tau_impl*1e3:.0f}/r{a.tau_rk4*1e3:.0f} "
                       f"rob={a.tau_robust*1e3:.1f} gap={gap_pct:.0f}%]"
                       if (cfg.tri_integrator or cfg.tri_timestep) else "")
            print(f"  it{it:03d} loss={a.loss:.4f}  L={a.Lbar*1e3:6.1f}mN  tau={a.tau_hat*1e3:6.1f}mNm  "
                  f"pk={a.tau_peak*1e3:6.1f}mNm dr={a.tau_drift*1e3:+5.1f}  "
                  f"swing={a.swing*1e3:5.1f}mNm  rms={a.tau_rms*1e3:5.1f}  pc={a.power_crest:5.3f}  "
                  f"lr={a.lift_ripple:5.3f}  ovl={a.C_overlap:5.3f}  obj={obj_of(row)/max(obj0,1e-12):6.4f}  "
                  f"A={a.A_stroke*1e3:5.2f}mm  P={a.P_supplied*1e3:5.0f}mW  "
                  f"S={a.S:.3g}  Cv={a.C_valid:.2g} a={math.degrees(a.alpha_pk):.0f} k={a.k_eff:.2g} "
                  f"conv={a.conv_err:.1e} beta={robust_beta:.0f} Pin={a.P_inactive:.3g}{tri}", flush=True)
            if it % 3 == 0:
                dump(sel, log, out_dir=cfg.out / "checkpoints" / f"it{it:03d}")
            if cfg.control_after and it == cfg.control_after:
                # The waveform goes live on the NEXT step, so this is the a0-only design:
                # the reference every "did harmonics help?" claim is measured against.
                pre = dict(sel)
                if rotate_fns and cfg.full_check_every:
                    pr, prob, _, plift, _ = full_matrix(bank_jit, theta_eval)
                    pre.update(full_rows=pr, full_robust=prob, full_liftmin=plift,
                               full_obj=obj_of(robust_row(pr, prob, plift)))
                    print(f"    [preharmonics it{it:03d}] robust obj="
                          f"{pre['full_obj']/max(obj0,1e-12):.4f} x baseline  "
                          f"liftmin={plift*1e3:.1f} mN", flush=True)
                dump(pre, log, out_dir=cfg.out / "preharmonics")
            if (rotate_fns and cfg.full_check_every
                    and it % cfg.full_check_every == 0 and train_i > 0):
                rows, rob, gapf, liftmin, driftf = full_matrix(bank_jit, theta_eval)
                if not math.isfinite(liftmin0):
                    rob0, liftmin0 = rob, liftmin
                amps = [r["A"] for r in rows]
                swmax = max(r["swing"] for r in rows)
                feas = amp_gate_lo <= min(amps) and max(amps) <= amp_gate_hi
                if cfg.single_stage and cfg.lift_torque_tradeoff is None:
                    feas = feas and liftmin >= cfg.lift_floor  # robust-min lift clears the floor
                stationary = driftf <= cfg.drift_gate * max(rob, 1e-9)
                transfers = gapf <= cfg.full_gap and liftmin >= 0.9 * liftmin0 and stationary
                powf = max(r["P"] for r in rows)
                objf = obj_of(robust_row(rows, rob, liftmin))
                if cfg.objective != "torque":
                    # A power-hacked cell rejects like a torque-hacked one.
                    pgap = (powf - min(r["P"] for r in rows)) / max(powf, 1e-9)
                    transfers = transfers and pgap <= cfg.full_gap
                scoref = score_of(objf, swmax, liftmin)
                tag = ("TRANSFERS" if feas else "LIFT/AMP-OUT") if transfers \
                    else (f"HACK(gap {gapf*100:.0f}%)" if gapf > cfg.full_gap
                          else f"DRIFT({driftf/max(rob,1e-9)*100:.0f}%)")
                print(f"    [full it{it:03d}] robust={rob*1e3:6.1f} mNm "
                      f"({(rob-rob0)/rob0*100:+.0f}% vs base)  gap={gapf*100:.0f}%  "
                      f"liftmin={liftmin*1e3:6.1f} mN  P={powf*1e3:5.0f} mW  "
                      f"P/L={powf/max(liftmin,1e-9):5.1f} W/N ({(objf-obj0)/obj0*100:+.0f}% obj)  "
                      f"drift={driftf*1e3:.1f} mNm  [{tag}]", flush=True)
                log[-1].update(full_robust=rob, full_gap=gapf, full_liftmin=liftmin,
                               full_drift=driftf, full_power=powf, full_obj=objf,
                               full_rows=rows)
                sel_f = {"score": scoref, "tau": rob, "power": powf, "swing": swmax,
                         "theta": theta_eval, "Lbar": liftmin, "Pin": a.P_inactive}
                if feas and scoref < best["score"]:
                    best = sel_f
                if feas and scoref < stage_best["score"]:
                    stage_best = sel_f
                if feas and transfers and scoref < robust_best["score"]:
                    robust_best = dict(sel_f)
                    dump(robust_best, log, out_dir=cfg.out / "robust_best")
            if cfg.cpu_every and it % cfg.cpu_every == 0 and train_i > 0:
                cpu_vars, robust, gap, cpuL, cdrift = cpu_gate(
                    cfg.scene, cpu_design(theta_eval), cpu_cells, **cpu_kw)
                cpuT = cpu_vars[0]["T"]
                cpuTi = next((x["T"] for x in cpu_vars if x["integrator"] == "implicitfast"), cpuT)
                cpuTr = next((x["T"] for x in cpu_vars if x["integrator"] == "rk4"), cpuTi)
                if not math.isfinite(cpuT0r):
                    cpuT0, cpuT0r, cpuL0 = cpuT, robust, cpuL
                stationary = cdrift <= cfg.drift_gate * max(robust, 1e-9)
                transfers = gap <= cfg.cpu_gap and stationary
                cpuP = max(x["P"] for x in cpu_vars)
                # The CPU rollout has no shape metrics, so for those objectives the CPU side
                # keeps selecting on robust PEAK torque as a transfer guard; the shape winner
                # comes from the full-matrix check, which sees every metric.
                cpu_obj = (obj_of({"tau": robust, "power": cpuP, "lift": cpuL})
                           if cfg.objective in ("torque", "power", "energy") else robust)
                if not math.isfinite(cpu_obj0):
                    cpu_obj0 = cpu_obj
                if cfg.objective != "torque":
                    pgap = (cpuP - min(x["P"] for x in cpu_vars)) / max(cpuP, 1e-9)
                    transfers = transfers and pgap <= cfg.cpu_gap
                dpk = (robust - cpuT0r) / cpuT0r * 100 if cpuT0r else float("nan")
                tag = "TRANSFERS" if transfers else (
                    f"HACK(gap {gap*100:.0f}%)" if gap > cfg.cpu_gap
                    else f"DRIFT({cdrift/max(robust,1e-9)*100:.0f}%)")
                desc = " ".join(f"{x['integrator']}@{(x['timestep_ms'] or 0):g}={x['T']*1e3:.0f}"
                                for x in cpu_vars)
                print(f"    [cpu it{it:03d}] mjx={a.tau_hat*1e3:6.1f}  {desc}  "
                      f"robust={robust*1e3:6.1f} mNm ({dpk:+.0f}% vs base)  "
                      f"Lmin={cpuL*1e3:6.1f} mN  P={cpuP*1e3:5.0f} mW  "
                      f"P/L={cpuP/max(cpuL,1e-9):5.1f} W/N "
                      f"({(cpu_obj-cpu_obj0)/max(cpu_obj0,1e-9)*100:+.0f}% obj)  "
                      f"drift={cdrift*1e3:.1f} mNm  [{tag}]", flush=True)
                log[-1].update(cpu_tau=cpuT, cpu_tau_impl=cpuTi, cpu_tau_rk4=cpuTr,
                               cpu_L=cpuL, cpu_gap=gap, cpu_drift=cdrift,
                               cpu_power=cpuP, cpu_obj=cpu_obj, cpu_variants=cpu_vars)
                # CPU-selected best: lowest robust score among MJX-feasible designs that keep
                # lift AND transfer between integrators -- the gate against integrator hacks.
                cpu_lift_ok = (cpuL >= cfg.lift_floor
                               if (cfg.single_stage and cfg.lift_torque_tradeoff is None)
                               else cpuL >= 0.9 * cpuL0)
                if (is_feasible(a.Lbar, a.A_stroke) and cpu_lift_ok and transfers
                        and cpu_obj < cpu_best["score"]):
                    cpu_best = {"score": cpu_obj, "tau": robust, "swing": a.swing, "theta": theta_eval,
                                "Lbar": a.Lbar, "Pin": a.P_inactive, "mjx_tau": a.tau_hat, "cpu_L": cpuL,
                                "cpu_P": cpuP, "cpu_obj": cpu_obj,
                                "cpu_euler": cpuT, "cpu_impl": cpuTi, "cpu_rk4": cpuTr, "cpu_gap": gap}
        stage_dir = cfg.out / f"stage_{round(tau_max*1e3)}mNm"
        dump(stage_best, log, out_dir=stage_dir)
        dump(best, log)  # running global best (interrupt-safe)
        print(f"  [stage {round(tau_max*1e3)} mNm] best tau={stage_best['tau']*1e3:.1f} mNm "
              f"swing={stage_best['swing']*1e3:.1f} mNm L={stage_best['Lbar']*1e3:.1f} mN "
              f"-> {stage_dir}")

    dump(best, log)
    print(f"\nbest design: tau={best['tau']*1e3:.1f} mNm  swing={best['swing']*1e3:.1f} mNm  "
          f"L={best['Lbar']*1e3:.1f} mN  (from {m0.tau_hat*1e3:.1f} mNm / {m0.swing*1e3:.1f} mNm "
          f"baseline)  ->  {cfg.out}")

    def validate_design(theta_v, label):
        """CPU-validate a winner on the SAME scene (euler matches MJX; implicitfast is the
        nominal physics)."""
        design = cpu_design(theta_v)
        cpu = {}
        for integ in (("euler", "implicitfast", "rk4") if three_integrators else ("euler", "implicitfast")):
            try:
                r = cpu_rollout(cfg.scene, design, integ, timestep_ms=cfg.timestep, **cpu_kw)
                cpu[integ] = {"mean_lift_N": r.lift, "peak_torque_Nm": r.peak, "supplied_power_W": r.power}
                print(f"[cpu/{integ}/{label}] lift={r.lift*1e3:.1f} mN  peak_torque={r.peak*1e3:.1f} mNm  "
                      f"power={r.power*1e3:.0f} mW")
            except Exception as e:
                cpu[integ] = {"error": str(e)[:200]}
                print(f"[cpu/{integ}/{label}] FAILED: {e}")
        return cpu

    report = {"baseline": {"cpu_euler_lift_N": cpuL0, "cpu_euler_peak_Nm": cpuT0,
                           "cpu_robust_peak_Nm": cpuT0r},
              "mjx_best": {"mjx": {"tau_hat": best["tau"], "swing": best["swing"],
                                   "Lbar": best["Lbar"]},
                           "cpu": validate_design(best["theta"], "mjx_best")}}
    # The CPU-selected best is the honest, integrator-robust winner.
    if cpu_best is not None and math.isfinite(cpu_best["tau"]):
        dump(cpu_best, log, out_dir=cfg.out / "cpu_best")
        _rk4 = cpu_best.get("cpu_rk4", cpu_best["cpu_impl"])
        print(f"\ncpu-selected best: robust tau={cpu_best['tau']*1e3:.1f} mNm "
              f"(euler {cpu_best['cpu_euler']*1e3:.1f}/impl {cpu_best['cpu_impl']*1e3:.1f}/"
              f"rk4 {_rk4*1e3:.1f})  L={cpu_best['Lbar']*1e3:.1f} mN  "
              f"(from cpu baseline {cpuT0r*1e3:.1f} mNm)  ->  {cfg.out / 'cpu_best'}")
        report["cpu_best"] = {"robust_peak_Nm": cpu_best["tau"], "cpu_gap": cpu_best["cpu_gap"],
                              "cpu": validate_design(cpu_best["theta"], "cpu_best")}
    # The full-matrix winner (rotate mode): the deliverable of a rotating-batch run.
    if robust_best is not None and math.isfinite(robust_best["tau"]):
        dump(robust_best, log, out_dir=cfg.out / "robust_best")
        print(f"\nfull-matrix best (transfer-gated): robust tau={robust_best['tau']*1e3:.1f} mNm "
              f"liftmin={robust_best['Lbar']*1e3:.1f} mN  ->  {cfg.out / 'robust_best'}")
        report["robust_best"] = {"robust_peak_Nm": robust_best["tau"],
                                 "cpu": validate_design(robust_best["theta"], "robust_best")}
    if cfg.holdout_dts:
        hold = holdout(cfg, [(robust_best, "robust_best"), (cpu_best, "cpu_best"),
                             (best, "mjx_best")], variant_metrics, cpu_design, cpu_kw)
        if hold is not None:
            report["holdout"] = hold
    (cfg.out / "cpu_validation.json").write_text(json.dumps(report, indent=2))


def holdout(cfg, candidates, variant_metrics, cpu_design, cpu_kw):
    """Evaluate the first finite winner on UNSEEN timesteps (never trained on or in the
    full-check matrix): 3 integrators on MJX, euler + implicitfast on CPU. A design that
    holds up here learned physics, not a dt trick."""
    winner = wname = None
    for cand, name in candidates:
        if cand is not None and math.isfinite(cand["tau"]):
            winner, wname = cand, name
            break
    if winner is None:
        return None
    print(f"\n[holdout] evaluating {wname} on unseen dts {tuple(cfg.holdout_dts)} ms")
    hold = {"design": wname, "mjx": [], "cpu": []}
    hold_sp = cfg.full_sample_periods if cfg.full_sample_periods else cfg.sample_periods
    for dm in cfg.holdout_dts:
        for ig in ("euler", "implicitfast", "rk4"):
            f = variant_metrics(ig, dm * 1e-3, sample_periods=hold_sp, avg_periods=max(1, int(hold_sp)))
            m = Metrics(*(float(x) for x in jax.jit(f)(winner["theta"])))
            hold["mjx"].append({"integrator": ig, "timestep_ms": dm,
                                "L": m.Lbar, "T": m.tau_peak, "T_lp": m.tau_hat, "drift": m.tau_drift})
            print(f"[holdout mjx {ig}@{dm:g}] lift={m.Lbar*1e3:.1f} mN "
                  f"peak={m.tau_peak*1e3:.1f} mNm drift={m.tau_drift*1e3:+.1f} mNm", flush=True)
    design = cpu_design(winner["theta"])
    for dm in cfg.holdout_dts:
        for ig in ("euler", "implicitfast"):
            try:
                r = cpu_rollout(cfg.scene, design, ig, timestep_ms=dm, **cpu_kw)
                hold["cpu"].append({"integrator": ig, "timestep_ms": dm, "L": r.lift, "T": r.peak})
                print(f"[holdout cpu {ig}@{dm:g}] lift={r.lift*1e3:.1f} mN "
                      f"peak={r.peak*1e3:.1f} mNm", flush=True)
            except Exception as e:
                hold["cpu"].append({"integrator": ig, "timestep_ms": dm, "error": str(e)[:200]})
                print(f"[holdout cpu {ig}@{dm:g}] FAILED: {e}")
    ts = [r["T"] for r in hold["mjx"] + hold["cpu"] if "T" in r and math.isfinite(r["T"])]
    if ts:
        hold["robust_peak_Nm"] = max(ts)
        hold["spread"] = (max(ts) - min(ts)) / max(ts)
        print(f"[holdout] robust peak={max(ts)*1e3:.1f} mNm "
              f"spread={hold['spread']*100:.0f}% over {len(ts)} unseen cases")
    return hold
