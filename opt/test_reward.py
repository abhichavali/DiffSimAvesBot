"""Self-check for the demo reward (run: uv run python opt/test_reward.py).

Verifies the reward shape:
  1. Stroke amplitude is penalized ONLY below the baseline (shrinking the flap),
     never above (bigger flap is free).
  2. Lift is a one-sided SOFT floor: below it is penalized, overshooting it is
     free (no cap), and right at the floor a soft-hinge penalty remains (so the
     constraint is soft and gradients flow, not a hard wall).
make_loss returns jacfwd(loss, has_aux); aux[-1] = -R = the scalar loss, which is
all we need (a constant stub metrics ignores theta, so the gradient is irrelevant).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp

from opt.optimize_control import make_loss

A_TARGET = 0.020        # 20 mm baseline stroke
FLOOR = 0.300           # 300 mN lift floor (no cap by default)


def loss_for(*, Lbar, A_stroke, lift_cap=None):
    """Scalar loss for a design with the given lift / amplitude (other metrics neutral)."""
    def metrics(theta):
        return (jnp.asarray(Lbar), jnp.asarray(0.0), jnp.asarray(A_stroke),
                jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
    grad_fn = make_loss(metrics, lift_floor=FLOOR, lift_cap=lift_cap, lift_smooth=0.015,
                        lift_above_weight=0.25, A_target=A_TARGET, tau_max=None,
                        weights=(1.0, 1.0, 1.0, 0.0), w_lc=0.0, w_power=0.0, w_overlap=0.0)
    _, aux = grad_fn(jnp.zeros(1), 0.300)
    return float(aux[-1])


def main():
    nominal = loss_for(Lbar=0.450, A_stroke=A_TARGET)        # well above floor, amp at baseline
    shrunk = loss_for(Lbar=0.450, A_stroke=0.8 * A_TARGET)   # 20% smaller flap
    grown = loss_for(Lbar=0.450, A_stroke=1.2 * A_TARGET)    # 20% bigger flap
    assert shrunk > nominal + 1e-4, f"shrinking flap must hurt: {shrunk} !> {nominal}"
    assert grown <= nominal + 1e-9, f"bigger flap must not be penalized: {grown} !<= {nominal}"

    below = loss_for(Lbar=FLOOR - 0.030, A_stroke=A_TARGET)  # 30 mN under floor
    at = loss_for(Lbar=FLOOR, A_stroke=A_TARGET)             # exactly at floor
    way_above = loss_for(Lbar=0.600, A_stroke=A_TARGET)      # 2x the floor
    assert below > at + 1e-3, f"below floor must be penalized: {below} !> {at}"
    assert way_above <= nominal + 1e-9, f"overshooting lift must never hurt: {way_above} !<= {nominal}"
    assert at > nominal + 1e-6, f"soft floor must carry a penalty at the floor: {at} !> {nominal}"

    # A set cap re-enables a moderate above penalty (kept as an option).
    capped_above = loss_for(Lbar=0.450, A_stroke=A_TARGET, lift_cap=0.350)
    assert capped_above > nominal + 1e-4, "with a cap, overshoot is penalized again"

    print(f"ok: shrink={shrunk:.4f}>nom={nominal:.4f}; grown<=nom; "
          f"below={below:.4f}>at={at:.4f}>nom; wayabove==nom; cap re-penalizes")


if __name__ == "__main__":
    main()
