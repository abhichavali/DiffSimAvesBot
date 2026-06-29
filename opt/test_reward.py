"""Self-check for the asymmetric demo reward (run: uv run python opt/test_reward.py).

Verifies the two asymmetries the demo objective requires:
  1. Stroke amplitude is penalized ONLY below the baseline (shrinking the flap),
     not above (flapping bigger than baseline is free).
  2. The lift band [floor, cap] penalizes BELOW the floor much harder than ABOVE
     the cap (strict floor, moderate ceiling).
make_loss returns jacfwd(loss, has_aux); aux[-1] = -R = the scalar loss, which is
all we need (a constant stub metrics ignores theta, so the gradient is irrelevant).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jnp

from opt.optimize_control import make_loss

A_TARGET = 0.020        # 20 mm baseline stroke
FLOOR, CAP = 0.190, 0.200
ABOVE_W = 0.25


def loss_for(*, Lbar, A_stroke):
    """Scalar loss for a design with the given lift / amplitude (other metrics neutral)."""
    def metrics(theta):
        return (jnp.asarray(Lbar), jnp.asarray(0.0), jnp.asarray(A_stroke),
                jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
    grad_fn = make_loss(metrics, lift_floor=FLOOR, lift_cap=CAP, lift_smooth=0.005,
                        lift_above_weight=ABOVE_W, A_target=A_TARGET, tau_max=None,
                        weights=(1.0, 1.0, 1.0, 0.0), w_lc=0.0, w_power=0.0)
    _, aux = grad_fn(jnp.zeros(1), 0.300)
    return float(aux[-1])


def main():
    in_band = loss_for(Lbar=0.195, A_stroke=A_TARGET)        # everything nominal
    shrunk = loss_for(Lbar=0.195, A_stroke=0.8 * A_TARGET)   # 20% smaller flap
    grown = loss_for(Lbar=0.195, A_stroke=1.2 * A_TARGET)    # 20% bigger flap
    assert shrunk > in_band + 1e-4, f"shrinking flap must hurt: {shrunk} !> {in_band}"
    assert grown <= in_band + 1e-9, f"bigger flap must not be penalized: {grown} !<= {in_band}"

    below = loss_for(Lbar=FLOOR - 0.010, A_stroke=A_TARGET)  # 10 mN under floor
    above = loss_for(Lbar=CAP + 0.010, A_stroke=A_TARGET)    # 10 mN over cap
    assert below > above + 1e-4, f"below-floor must hurt more than above-cap: {below} !> {above}"
    # moderate-above factor should be ~ABOVE_W of the strict-below penalty
    ratio = (above - in_band) / (below - in_band)
    assert 0.1 < ratio < 0.4, f"above/below penalty ratio {ratio:.3f} not ~{ABOVE_W}"

    print(f"ok: shrink={shrunk:.4f} > inband={in_band:.4f}; grown=={in_band:.4f}; "
          f"below={below:.4f} > above={above:.4f} (ratio {ratio:.2f})")


if __name__ == "__main__":
    main()
