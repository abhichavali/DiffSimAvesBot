"""Self-check for the demo reward (run: uv run pytest -q tests/test_reward.py -s).

Verifies the reward shape:
  1. Stroke amplitude is STRICTLY banded around the baseline (+/-amp_band_pct):
     both shrinking AND growing the flap outside the band is penalized.
  2. Lift is a one-sided SOFT floor: below it is penalized, overshooting it is
     free (no cap), and right at the floor a soft-hinge penalty remains (so the
     constraint is soft and gradients flow, not a hard wall).
  3. In lift/torque tradeoff mode, a 5% lift gain can pay for a 1% torque rise.
make_loss returns jacfwd(loss, has_aux); aux[9] = -R = the scalar loss -- a
constant stub metrics ignores theta, so
the gradient is irrelevant, only the loss value is checked.
"""
import jax.numpy as jnp

from opt.optimize_control import make_loss, recorded_peak

A_TARGET = 0.020        # 20 mm baseline stroke
FLOOR = 0.300           # 300 mN lift floor (no cap by default)


def loss_for(*, Lbar, A_stroke, lift_cap=None, swing=0.0, w_swing=0.0,
             tau_hat=0.0, tau_max=0.300, lift0=None, tradeoff=None,
             P_sup=0.0, objective="torque", obj_norm=None, L0=None, tau_guard=0.0,
             overlap=0.0, tau_rms=0.0, power_crest=0.0, lift_ripple=0.0):
    """Scalar loss for a design with the given lift / amplitude / torque-swing
    (other metrics neutral). metrics() = (Lbar, tau_hat, A_stroke, S, C_lc,
    P_inactive, C_overlap, swing, P_supplied, C_valid, alpha_pk, k_eff,
    conv_err, tau_peak, tau_drift) -- make_loss unpacks all 15, so this stub
    has to track build_rollout's tuple exactly."""
    def metrics(theta):
        zeros = [jnp.asarray(0.0)] * 18
        out = list(zeros)
        out[0], out[1], out[2] = jnp.asarray(Lbar), jnp.asarray(tau_hat), jnp.asarray(A_stroke)
        out[7] = jnp.asarray(swing)
        out[8] = jnp.asarray(P_sup)
        out[6] = jnp.asarray(overlap)
        out[15], out[16], out[17] = (jnp.asarray(tau_rms), jnp.asarray(power_crest),
                                     jnp.asarray(lift_ripple))
        return tuple(out)
    grad_fn = make_loss(metrics, lift_floor=FLOOR, lift_cap=lift_cap, lift_smooth=0.015,
                        lift_above_weight=0.25, A_target=A_TARGET, tau_max=None,
                        weights=(1.0, 1.0, 1.0, 0.0), w_lc=0.0, w_power=0.0, w_overlap=0.0,
                        w_swing=w_swing, lift0=lift0, tradeoff=tradeoff,
                        objective=objective, obj_norm=obj_norm, L0=L0, tau_guard=tau_guard)
    _, aux = grad_fn(jnp.zeros(1), tau_max, 300.0)
    return float(aux[9])


def test_reward_shape():
    peak = recorded_peak(
        jnp.asarray([999.0, 0.10, 0.42, 0.11]),
        jnp.asarray([False, True, True, True]),
    )
    assert abs(float(peak) - 0.42) < 1e-6, f"recorded peak must match CPU max, got {peak}"

    nominal = loss_for(Lbar=0.450, A_stroke=A_TARGET)        # well above floor, amp at baseline
    shrunk = loss_for(Lbar=0.450, A_stroke=0.8 * A_TARGET)   # 20% smaller flap (outside band)
    grown = loss_for(Lbar=0.450, A_stroke=1.2 * A_TARGET)    # 20% bigger flap (outside band)
    in_band = loss_for(Lbar=0.450, A_stroke=1.01 * A_TARGET)  # 1% over, inside the 2% band
    assert shrunk > nominal + 1e-4, f"shrinking flap must hurt: {shrunk} !> {nominal}"
    assert grown > nominal + 1e-4, f"growing flap outside the band must hurt: {grown} !> {nominal}"
    assert in_band <= nominal + 1e-3, f"small deviation inside the band should be ~free: {in_band} !<= {nominal}"

    below = loss_for(Lbar=FLOOR - 0.030, A_stroke=A_TARGET)  # 30 mN under floor
    at = loss_for(Lbar=FLOOR, A_stroke=A_TARGET)             # exactly at floor
    way_above = loss_for(Lbar=0.600, A_stroke=A_TARGET)      # 2x the floor
    assert below > at + 1e-3, f"below floor must be penalized: {below} !> {at}"
    assert way_above <= nominal + 1e-9, f"overshooting lift must never hurt: {way_above} !<= {nominal}"
    assert at > nominal + 1e-6, f"soft floor must carry a penalty at the floor: {at} !> {nominal}"

    # A set cap re-enables a moderate above penalty (kept as an option).
    capped_above = loss_for(Lbar=0.450, A_stroke=A_TARGET, lift_cap=0.350)
    assert capped_above > nominal + 1e-4, "with a cap, overshoot is penalized again"

    # Torque swing: with w_swing>0, more swing must hurt; with w_swing=0 it's free.
    no_swing = loss_for(Lbar=0.450, A_stroke=A_TARGET, swing=0.150, w_swing=0.0)
    swung = loss_for(Lbar=0.450, A_stroke=A_TARGET, swing=0.150, w_swing=10.0)
    assert abs(no_swing - nominal) < 1e-9, f"swing must be free when w_swing=0: {no_swing} != {nominal}"
    assert swung > nominal + 1e-3, f"torque swing must be penalized when w_swing>0: {swung} !> {nominal}"

    trade_base = loss_for(Lbar=1.0, A_stroke=A_TARGET, tau_hat=1.0,
                          tau_max=1.0, lift0=1.0, tradeoff=5.0)
    lift_for_torque = loss_for(Lbar=1.05, A_stroke=A_TARGET, tau_hat=1.01,
                               tau_max=1.0, lift0=1.0, tradeoff=5.0)
    assert lift_for_torque < trade_base, (
        f"5% lift gain must improve despite 1% torque rise: {lift_for_torque} !< {trade_base}"
    )

    # --- objective=energy: specific power P/L, lift credited divisively ---------
    P_BASE, L_BASE = 0.500, 0.400        # 500 mW at 400 mN = 1.25 W/N baseline
    def energy(P, L, tau_hat=0.0):
        return loss_for(Lbar=L, A_stroke=A_TARGET, P_sup=P, tau_hat=tau_hat, tau_max=0.300,
                        objective="energy", obj_norm=(P_BASE / L_BASE,), L0=L_BASE)
    base = energy(P_BASE, L_BASE)
    assert abs(base - 1.0) < 0.02, f"baseline must normalize to ~1 objective unit, got {base}"
    # 1. Fewer watts at the same lift wins; more watts loses. (The point of the run.)
    assert energy(0.9 * P_BASE, L_BASE) < base - 1e-3, "less power at same lift must win"
    assert energy(1.1 * P_BASE, L_BASE) > base + 1e-3, "more power at same lift must lose"
    # 2. Lift is credited -- but ONLY when bought below the current average P/L.
    #    Baseline is 1.25 W/N: +100 mN for +100 mW (1.0 W/N marginal) is a real gain;
    #    the SAME lift for +200 mW (2.0 W/N) is not. A linear lambda-credit (run 7's
    #    fountain) would have rewarded both.
    cheap = energy(P_BASE + 0.100, L_BASE + 0.100)   # 1.0 W/N marginal < 1.25 average
    dear = energy(P_BASE + 0.200, L_BASE + 0.100)    # 2.0 W/N marginal > 1.25 average
    assert cheap < base - 1e-3, f"lift bought under the average W/N must win: {cheap} !< {base}"
    assert dear > base + 1e-3, f"lift bought over the average W/N must lose: {dear} !> {base}"
    # 3. Not flapping is not efficiency: the denominator is floored at lift_floor, so
    #    collapsing lift below it can never divide the cost down.
    collapse = energy(0.05 * P_BASE, 0.02)           # 25 mW at 20 mN = 1.25 W/N "ratio"
    assert collapse > base - 1e-3, f"sub-floor lift must not buy the ratio: {collapse} !> {base}"
    assert energy(0.05 * P_BASE, 0.02) > energy(0.05 * P_BASE, FLOOR) - 1e-9, \
        "floored denominator: below the floor, extra lift loss gives no further credit"
    # 4. Torque is a guard, not a target: free under the ceiling, priced above it.
    guarded = loss_for(Lbar=L_BASE, A_stroke=A_TARGET, P_sup=P_BASE, tau_hat=0.400,
                       tau_max=0.300, objective="energy", obj_norm=(P_BASE / L_BASE,),
                       L0=L_BASE, tau_guard=4.0)
    assert abs(energy(P_BASE, L_BASE, tau_hat=0.290) - base) < 1e-6, "torque under the ceiling is free"
    assert guarded > base + 0.1, f"torque over the ceiling must be guarded: {guarded} !> {base}"

    # --- objective=swing: the Sep-10 objective screen's pick ------------
    SW0 = 0.030          # 30 mNm baseline torque swing
    def swing_obj(sw, Lbar=L_BASE):
        return loss_for(Lbar=Lbar, A_stroke=A_TARGET, swing=sw, tau_max=0.300,
                        objective="swing", obj_norm=(SW0,), L0=L_BASE)
    sw_base = swing_obj(SW0)
    assert abs(sw_base - 1.0) < 0.02, f"swing baseline must normalize to ~1, got {sw_base}"
    assert swing_obj(0.8 * SW0) < sw_base - 1e-3, "less torque swing must win"
    assert swing_obj(1.2 * SW0) > sw_base + 1e-3, "more torque swing must lose"
    # Not gameable by dropping lift out from under the machine: the floor wall bites.
    assert swing_obj(0.5 * SW0, Lbar=FLOOR - 0.050) > sw_base, \
        "halving swing by losing 50 mN of lift must still lose"

    # --- every screen-derived objective: normalized to 1, monotone in ITS metric ---
    # The whole point of the generic path is that a candidate is trained on exactly
    # the quantity it is later scored on, so one loop covers all of them.
    from opt.optimize_control import OBJ_METRIC_IDX
    BASE = {"power": 0.500, "swing": 0.030, "overlap": 0.820, "tau_rms": 0.110,
            "power_crest": 1.024, "lift_ripple": 0.777}
    for obj, b0 in BASE.items():
        kw = dict(Lbar=L_BASE, A_stroke=A_TARGET, objective=obj, obj_norm=(b0,), L0=L_BASE)
        arg = {"power": "P_sup", "swing": "swing"}.get(obj, obj)
        o_at = loss_for(**kw, **{arg: b0})
        better = loss_for(**kw, **{arg: 0.8 * b0})
        worse = loss_for(**kw, **{arg: 1.2 * b0})
        assert abs(o_at - 1.0) < 0.02, f"{obj}: baseline must score ~1, got {o_at}"
        assert better < o_at - 1e-3, f"{obj}: less must win ({better} !< {o_at})"
        assert worse > o_at + 1e-3, f"{obj}: more must lose ({worse} !> {o_at})"
        assert obj in OBJ_METRIC_IDX, f"{obj} missing from OBJ_METRIC_IDX"
    # Composite: 50/50 of swing and tau_rms, so improving EITHER half improves it.
    comp = dict(Lbar=L_BASE, A_stroke=A_TARGET, objective="swing_rms",
                obj_norm=(0.030, 0.110), L0=L_BASE)
    c_at = loss_for(**comp, swing=0.030, tau_rms=0.110)
    assert abs(c_at - 1.0) < 0.02, f"composite baseline must score ~1, got {c_at}"
    assert loss_for(**comp, swing=0.024, tau_rms=0.110) < c_at - 1e-3, "composite: swing half"
    assert loss_for(**comp, swing=0.030, tau_rms=0.088) < c_at - 1e-3, "composite: rms half"

    print(f"ok: shrink={shrunk:.4f}>nom={nominal:.4f}; grown={grown:.4f}>nom (band enforced); "
          f"below={below:.4f}>at={at:.4f}>nom; wayabove==nom; cap re-penalizes; "
          f"swing={swung:.4f}>nom (w_swing=0 free); lift/torque trade ok; "
          f"energy: base={base:.4f}~1 cheap-lift={cheap:.4f}< dear-lift={dear:.4f}> "
          f"collapse={collapse:.4f}> guard={guarded:.4f}>; "
          f"swing: base={sw_base:.4f}~1 less={swing_obj(0.8*SW0):.4f}< more={swing_obj(1.2*SW0):.4f}>; "
          f"all {len(BASE)} screen objectives normalize to 1 and are monotone; composite ok")

