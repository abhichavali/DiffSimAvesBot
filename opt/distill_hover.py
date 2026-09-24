#!/usr/bin/env python3
"""Distill the per-wingbeat LQR into the hover MLP (the starting point for GPU fine-tuning).

Fits hover.env.policy(params, s) to u = clip(-K s, +-U_MAX) on states spanning push-sized disturbances.

    PYTHONPATH=src uv run python opt/distill_hover.py --lqr designs/hummer/free-hover/lqr.npz --out mlp.npz
"""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from hover import env as E
from hover.policy import save_mlp

STATE_STD = np.array([0.03] * 3 + [0.05] * 3 + [0.15] * 3 + [0.7] * 3, np.float32)   # m, rad, m/s, rad/s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lqr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    K = jnp.asarray(np.load(a.lqr)["K"], jnp.float32)
    rng = np.random.default_rng(0)
    S = jnp.asarray(rng.standard_normal((20000, 12)).astype(np.float32) * STATE_STD * rng.uniform(0.1, 2.0, (20000, 1)).astype(np.float32))
    U = jnp.clip(-(S @ K.T), -E.U_MAX, E.U_MAX)
    params = E.mlp_init(jax.random.PRNGKey(0), [E.N_OBS, a.hidden, a.hidden, K.shape[0]], out_scale=0.1)

    def loss(p):
        pred = jax.vmap(lambda s: E.policy(p, s))(S)
        return jnp.mean(jnp.sum((pred - U) ** 2, -1)) / jnp.mean(jnp.sum(U ** 2, -1))

    g = jax.jit(jax.value_and_grad(loss))
    m = jax.tree_util.tree_map(jnp.zeros_like, params); v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for it in range(1, a.steps + 1):
        l, gr = g(params)
        m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, gr)
        v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, gr)
        lr = 3e-3 * (0.1 ** (it / a.steps))
        params = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * (m_ / (1 - 0.9 ** it)) / (jnp.sqrt(v_ / (1 - 0.999 ** it)) + 1e-8),
                                        params, m, v)
        if it % 1000 == 0 or it == 1:
            print(f"step {it}: relative MSE vs LQR {float(l):.5f}", flush=True)
    save_mlp(a.out, jax.tree_util.tree_map(np.asarray, params), E.OBS_SCALE, E.U_MAX)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
