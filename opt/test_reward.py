"""Self-check for the demo reward (run: uv run python opt/test_reward.py).

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp

from opt.optimize_control import make_loss, recorded_peak

A_TARGET = 0.020        # 20 mm baseline stroke
FLOOR = 0.300           # 300 mN lift floor (no cap by default)


def loss_for(*, Lbar, A_stroke, lift_cap=None, swing=0.0, w_swing=0.0,
             tau_hat=0.0, tau_max=0.300, lift0=None, tradeoff=None):
    """Scalar loss for a design with the given lift / amplitude / torque-swing
    (other metrics neutral). metrics() = (Lbar, tau_hat, A_stroke, S, C_lc,
    P_inactive, C_overlap, swing, P_supplied, C_valid, alpha_pk, k_eff,
    conv_err, tau_peak, tau_drift) -- make_loss unpacks all 15, so this stub
    has to track build_rollout's tuple exactly."""
    def metrics(theta):
        zeros = [jnp.asarray(0.0)] * 15
        out = list(zeros)
        out[0], out[1], out[2] = jnp.asarray(Lbar), jnp.asarray(tau_hat), jnp.asarray(A_stroke)
        out[7] = jnp.asarray(swing)
        return tuple(out)
    grad_fn = make_loss(metrics, lift_floor=FLOOR, lift_cap=lift_cap, lift_smooth=0.015,
                        lift_above_weight=0.25, A_target=A_TARGET, tau_max=None,
                        weights=(1.0, 1.0, 1.0, 0.0), w_lc=0.0, w_power=0.0, w_overlap=0.0,
                        w_swing=w_swing, lift0=lift0, tradeoff=tradeoff)
    _, aux = grad_fn(jnp.zeros(1), tau_max, 300.0)
    return float(aux[9])


def main():
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

    print(f"ok: shrink={shrunk:.4f}>nom={nominal:.4f}; grown={grown:.4f}>nom (band enforced); "
          f"below={below:.4f}>at={at:.4f}>nom; wayabove==nom; cap re-penalizes; "
          f"swing={swung:.4f}>nom (w_swing=0 free); lift/torque trade ok")


if __name__ == "__main__":
    main()
