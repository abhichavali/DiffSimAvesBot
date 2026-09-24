#!/usr/bin/env python3
"""Train the closed-loop hover MLP by backprop through MJX (GPU; diffsim).

The open-loop policy (a phase-locked trim from ``python -m hover.trim``) flies the free-standing bird; the
MLP adds per-wingbeat corrections to the 24 harmonic depths from the strobed state. Episodes start from the
cradle release with random rigid kicks; the horizon grows as the loss drops.

    PYTHONPATH=src .venv/bin/python opt/train_hover.py --trim trim_locked.json --out opt/runs_hover_cl
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import jax
import jax.numpy as jnp
import numpy as np

from design_paths import FREE_HOVER_TG
from hover import env as E
from hover import launch as L
from hover import trim as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trim", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("opt/runs_hover_cl"))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--beats", type=int, default=4, help="starting horizon (wingbeats)")
    ap.add_argument("--max-beats", type=int, default=24)
    ap.add_argument("--grow-below", type=float, default=0.6, help="lengthen the horizon when the loss is below this")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--kick-v", type=float, default=0.1, help="max |kick| per axis, COM velocity [m/s]")
    ap.add_argument("--kick-w", type=float, default=0.5, help="max |kick| per axis, angular velocity [rad/s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, default=None, help="warm-start params.pkl")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    t = json.load(open(a.trim)); x, att0 = np.array(t["x"], float), np.array(t["att0"], float)
    bird = L.load_bird(FREE_HOVER_TG, theta_g=x[0])
    t0 = time.time()
    rel = L.release(bird, x[:T.NT], T.attitude(att0, x[T.NT:]), T.K)
    env = E.make(bird, rel, x[:T.NT], T.K)
    print(f"[hover] released in {time.time()-t0:.1f}s; {env.n_step} steps/wingbeat, dt {env.dt*1e3:.5f} ms, "
          f"device {jax.devices()[0]}", flush=True)

    params = (pickle.load(open(a.init, "rb")) if a.init else
              E.mlp_init(jax.random.PRNGKey(a.seed), [E.N_OBS, a.hidden, a.hidden, 8 * T.K]))
    params = jax.tree_util.tree_map(jnp.asarray, params)
    rng = np.random.default_rng(a.seed)

    def kicks(n):
        k = np.concatenate([rng.uniform(-a.kick_v, a.kick_v, (n, 3)), rng.uniform(-a.kick_w, a.kick_w, (n, 3))], 1)
        k[0] = 0.0                                             # always keep the clean release
        return jnp.asarray(k, jnp.float32)

    compiled = {}

    def grad_fn(n_beats):
        if n_beats not in compiled:
            f = lambda p, k: E.loss(p, env, k, n_beats)
            compiled[n_beats] = jax.jit(jax.value_and_grad(f, has_aux=True))
        return compiled[n_beats]

    m = jax.tree_util.tree_map(jnp.zeros_like, params); v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps, clip = 0.9, 0.999, 1e-8, 1.0
    beats, best = a.beats, math.inf
    for it in range(1, a.iters + 1):
        t0 = time.time()
        (l, (S, U)), g = grad_fn(beats)(params, kicks(a.batch))
        l = float(l)
        if not math.isfinite(l):
            print(f"[hover] it {it}: non-finite loss, skipping", flush=True); continue
        gn = float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g))))
        g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, clip / (gn + 1e-12)), g)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        params = jax.tree_util.tree_map(lambda p, m_, v_: p - a.lr * (m_ / (1 - b1 ** it)) / (jnp.sqrt(v_ / (1 - b2 ** it)) + eps),
                                        params, m, v)
        S = np.asarray(S)
        tilt = np.degrees(np.linalg.norm(S[:, -1, 3:5], axis=1)); pos = np.linalg.norm(S[:, -1, :3], axis=1) * 1e3
        print(f"[hover] it {it:4d} beats {beats:2d}  loss {l:.4f}  |g| {gn:9.3g}  final tilt mean {tilt.mean():5.2f} max {tilt.max():5.2f} deg  "
              f"pos mean {pos.mean():5.1f} mm  |u| max {float(np.abs(np.asarray(U)).max()):.3f}  {time.time()-t0:5.1f}s", flush=True)
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(a.out / "params_last.pkl", "wb"))
        if l < best:
            best = l
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(a.out / f"params_best_b{beats}.pkl", "wb"))
        if l < a.grow_below and beats < a.max_beats:
            beats = min(a.max_beats, beats * 2); best = math.inf
            print(f"[hover] horizon -> {beats} wingbeats", flush=True)
    json.dump({"trim": str(a.trim), "hidden": a.hidden, "u_max": E.U_MAX, "beats": beats, "args": vars(a) | {"trim": str(a.trim), "out": str(a.out), "init": str(a.init)}},
              open(a.out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
