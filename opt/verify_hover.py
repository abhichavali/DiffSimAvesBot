#!/usr/bin/env python3
"""Fly an open-loop trim (+ optional trained MLP) on CPU MuJoCo (full solver, float64) with pushes.

    PYTHONPATH=src uv run python opt/verify_hover.py --trim designs/hummer/free-hover/open_loop.json \
        [--lqr designs/hummer/free-hover/lqr.npz | --params params.pkl] [--beats 120]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover import trim as T

U_MAX = 0.15
OBS_SCALE = np.array([20.0] * 3 + [10.0] * 3 + [5.0] * 3 + [1.0] * 3)   # = hover.env.OBS_SCALE


def mlp_policy(params):
    def pol(s):
        x = s * OBS_SCALE
        for w, b in params[:-1]:
            x = np.tanh(w @ x + b)
        w, b = params[-1]
        return np.concatenate([[0.0], U_MAX * np.tanh(w @ x + b)])
    return pol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trim", type=Path, required=True)
    ap.add_argument("--params", type=Path, default=None, help="trained MLP params.pkl")
    ap.add_argument("--lqr", type=Path, default=None, help="npz with the per-wingbeat LQR gain K (hover.lqr)")
    ap.add_argument("--beats", type=int, default=120)
    a = ap.parse_args()
    t = json.load(open(a.trim)); x, att0 = np.array(t["x"]), np.array(t["att0"])
    bird = L.load_bird(FREE_HOVER_TG, theta_g=x[0])
    rel = L.release(bird, x[:T.NT], T.attitude(att0, x[T.NT:]), T.K)
    pol = mlp_policy(pickle.load(open(a.params, "rb"))) if a.params else None
    if a.lqr:
        Kg = np.load(a.lqr)["K"]
        pol = lambda s: np.concatenate([[0.0], np.clip(-Kg @ s, -U_MAX, U_MAX)])
    pushes = {30: np.r_[0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],         # 0.2 m/s forward
              60: np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],         # 1 rad/s roll rate
              90: np.r_[0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0.5, 0]}      # sideways + up + pitch rate
    for name, kicks in (("no pushes", None), ("pushes at wb 30/60/90", pushes)):
        tr = L.fly(bird, rel, x[:T.NT], a.beats, policy=pol, kicks=kicks)
        tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
        spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
        ok = len(tr) == a.beats + 1
        print(f"{'closed' if pol else 'open'} loop, {name}: {'flew all %d wingbeats (%.1f s)' % (a.beats, a.beats / 8.53) if ok else 'LOST at wb %d' % (len(tr) - 1)}"
              f" | tilt max {tilt.max():5.1f} deg, last-20 mean {tilt[-20:].mean():4.1f} | pos max {pos.max():6.0f} mm, final {pos[-1]:6.0f} | "
              f"speed last-20 {spd[-20:].mean():5.0f} mm/s", flush=True)
        print("   tilt every 10 wb:", np.round(tilt[::10], 1).tolist())
        print("   pos mm every 10 wb:", np.round(pos[::10]).tolist())


if __name__ == "__main__":
    main()
