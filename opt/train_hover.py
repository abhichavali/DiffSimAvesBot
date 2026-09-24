#!/usr/bin/env python3
"""Train the closed-loop hover MLP by backprop through MJX (GPU; diffsim).

The open-loop policy of a flight control .tg (hover.policy; e.g. from ``python -m hover.trim``) flies the bird; the
MLP adds per-wingbeat corrections to the 24 harmonic depths from the strobed state. Episodes start from the
cradle release with random rigid kicks; the horizon grows as the loss drops.

    PYTHONPATH=src .venv/bin/python opt/train_hover.py [--control designs/hummer/free-hover/control.tg] --out opt/runs_hover_cl
"""
import argparse
import json
import math
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
from hover.policy import load_flight_policy, load_mlp, save_mlp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--out", type=Path, default=Path("opt/runs_hover_cl"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--schedule", type=str, default="4:40,8:60,16:100",
                    help="horizon curriculum 'beats:iters,...' (overrides --beats/--iters/--max-beats)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--kick-v", type=float, default=0.1, help="max |kick| per axis, COM velocity [m/s]")
    ap.add_argument("--kick-w", type=float, default=0.5, help="max |kick| per axis, angular velocity [rad/s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, default=None, help="warm-start MLP (npz, hover.policy format); default: the control's closed loop, if any")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fp = load_flight_policy(a.control)
    bird = L.load_bird(a.tg, control=a.control)
    t0 = time.time()
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock)
    env = E.make(bird, rel, fp.theta, fp.K)
    print(f"[hover] released in {time.time()-t0:.1f}s; {env.n_step} steps/wingbeat, dt {env.dt*1e3:.5f} ms, "
          f"device {jax.devices()[0]}", flush=True)

    init = load_mlp(a.init)["layers"] if a.init else (fp.mlp["layers"] if fp.mlp else None)
    params = init or E.mlp_init(jax.random.PRNGKey(a.seed), [E.N_OBS, a.hidden, a.hidden, 8 * fp.K])
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
    plan = [b for part in a.schedule.split(",") for b in [int(part.split(":")[0])] * int(part.split(":")[1])]
    best, beats = math.inf, plan[0]
    for it, beats_it in enumerate(plan, 1):
        if beats_it != beats:
            beats, best = beats_it, math.inf
            print(f"[hover] horizon -> {beats} wingbeats", flush=True)
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
        save_mlp(a.out / "mlp_last.npz", jax.tree_util.tree_map(np.asarray, params), E.OBS_SCALE, E.U_MAX)
        if l < best:
            best = l
            save_mlp(a.out / f"mlp_best_b{beats}.npz", jax.tree_util.tree_map(np.asarray, params), E.OBS_SCALE, E.U_MAX)
    json.dump({"control": str(a.control), "hidden": a.hidden, "u_max": E.U_MAX, "beats": beats,
               "args": {k: str(v) for k, v in vars(a).items()}},
              open(a.out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
