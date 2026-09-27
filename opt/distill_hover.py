#!/usr/bin/env python3
"""Distill the per-wingbeat LQR into the hover MLP (the starting point for GPU fine-tuning).

Fits hover.env.policy(params, s) to u = clip(-K s, +-U_MAX) on states spanning push-sized disturbances.

    PYTHONPATH=src uv run python opt/distill_hover.py --lqr designs/hummer/free-hover/lqr.npz --out mlp.npz

Reduced-observation controllers (hover.reduced, --ctrl ctl.npz) distill onto what they read:
  sof: MLP(y~) ~ clip(-Ky y~), y~ = the selected channels minus the acc feedthrough J u_prev, sampled as
       y~ = Cz z with z = [x, x_prev] ~ STATE_STD x U(0.1, 2);
  lqg: MLP(x_hat) ~ clip(-K x_hat) on the design states; the Kalman filter stays in the loop at runtime.
--noise-aug s adds input noise (s x the sensor std for sof, s x the filter's posterior std for lqg) while the
target is computed from the clean input. The output npz is the controller npz plus the MLP (fly it with
verify_hover/push_sweep --ctrl, or as a flight control's closed_loop).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from hover import env as E
from hover import reduced as Rd
from hover import sensors as Sn
from hover.policy import save_mlp

STATE_STD = np.array([0.03] * 3 + [0.05] * 3 + [0.15] * 3 + [0.7] * 3, np.float32)   # m, rad, m/s, rad/s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lqr", type=Path, help="full-state LQR npz (hover.lqr): the state MLP, as before")
    src.add_argument("--ctrl", type=Path, help="reduced controller npz (hover.reduced): distill onto its inputs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--noise-aug", type=float, default=0.0, help="input-noise augmentation (x the input's noise std)")
    a = ap.parse_args()
    if a.ctrl:
        return distill_reduced(a)
    K = jnp.asarray(np.load(a.lqr)["K"], jnp.float32)
    rng = np.random.default_rng(0)
    S = jnp.asarray(rng.standard_normal((20000, 12)).astype(np.float32) * STATE_STD * rng.uniform(0.1, 2.0, (20000, 1)).astype(np.float32))
    U = jnp.clip(-(S @ K.T), -E.U_MAX, E.U_MAX)
    params = E.mlp_init(jax.random.PRNGKey(0), [E.N_OBS, a.hidden, a.hidden, K.shape[0]], out_scale=0.1)

    def loss(p):
        pred = jax.vmap(lambda s: E.policy(p, s))(S)
        return jnp.mean(jnp.sum((pred - U) ** 2, -1)) / jnp.mean(jnp.sum(U ** 2, -1))

    params = fit(loss, params, a.steps)
    save_mlp(a.out, jax.tree_util.tree_map(np.asarray, params), E.OBS_SCALE, E.U_MAX)
    print(f"-> {a.out}")


def distill_reduced(a):
    ctl = Rd.load_controller(a.ctrl)
    kind = ctl["kind"]
    if kind not in ("sof", "lqg"):
        raise SystemExit(f"--ctrl kind {kind!r}: distill a full-state LQR with --lqr")
    xs, obs = np.asarray(ctl["xs"]), np.asarray(ctl["obs"])
    rng = np.random.default_rng(0)
    n = 20000
    amp = rng.uniform(0.1, 2.0, (n, 1))
    if kind == "sof":
        z = np.hstack([rng.standard_normal((n, len(xs))) * Rd.STATE_STD[xs], rng.standard_normal((n, len(xs))) * Rd.STATE_STD[xs]]) * amp
        X = z @ ctl["Cz"].T
        U = np.clip(-X @ ctl["Ky"].T, -Rd.U_MAX, Rd.U_MAX)
        std = np.asarray(ctl["V_std"])
        scale = Sn.CHANNEL_SCALE[obs]
    else:                                               # input x_hat (+ the dead-reckoned position axes)
        xi = np.concatenate([xs, ctl["dr"]]) if "dr" in ctl else xs
        K = np.hstack([ctl["K"], ctl["K_dr"]]) if "dr" in ctl else ctl["K"]
        X = rng.standard_normal((n, len(xi))) * Rd.STATE_STD[xi] * amp
        U = np.clip(-X @ K.T, -Rd.U_MAX, Rd.U_MAX)
        std = np.concatenate([ctl["P_post_std"], np.zeros(len(xi) - len(xs))])
        scale = E.OBS_SCALE[xi].astype(float)
    Xn = X + a.noise_aug * std * rng.standard_normal(X.shape)
    Xj, Uj = jnp.asarray(Xn, jnp.float32), jnp.asarray(U, jnp.float32)
    sc = jnp.asarray(scale, jnp.float32)

    def pol(p, x):
        h = x * sc
        for w, b in p[:-1]:
            h = jnp.tanh(w @ h + b)
        w, b = p[-1]
        return Rd.U_MAX * jnp.tanh(w @ h + b)

    params = E.mlp_init(jax.random.PRNGKey(0), [X.shape[1], a.hidden, a.hidden, U.shape[1]], out_scale=0.1)

    def loss(p):
        pred = jax.vmap(lambda x: pol(p, x))(Xj)
        return jnp.mean(jnp.sum((pred - Uj) ** 2, -1)) / jnp.mean(jnp.sum(Uj ** 2, -1))

    params = fit(loss, params, a.steps)
    arrays = {k: np.asarray(v) for k, v in ctl.items() if k not in ("info", "mlp")}
    for i, (W, b) in enumerate(params):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(W), np.asarray(b)
    info = dict(ctl["info"], distilled={"from": str(a.ctrl), "input": "y~" if kind == "sof" else "x_hat",
                                        "noise_aug": a.noise_aug, "hidden": a.hidden, "steps": a.steps})
    np.savez(a.out, obs_scale=np.asarray(scale), info=np.array(json.dumps(info, default=float)), **arrays)
    print(f"-> {a.out} (fly with --ctrl)")


def fit(loss, params, steps):
    g = jax.jit(jax.value_and_grad(loss))
    m = jax.tree_util.tree_map(jnp.zeros_like, params); v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for it in range(1, steps + 1):
        l, gr = g(params)
        m = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, gr)
        v = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, gr)
        lr = 3e-3 * (0.1 ** (it / steps))
        params = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * (m_ / (1 - 0.9 ** it)) / (jnp.sqrt(v_ / (1 - 0.999 ** it)) + 1e-8),
                                        params, m, v)
        if it % 1000 == 0 or it == 1:
            print(f"step {it}: relative MSE vs LQR {float(l):.5f}", flush=True)
    return params


if __name__ == "__main__":
    main()
