"""Open-loop trim of the free bird under the phase-locked drive: Gauss-Newton on the launch cradle's mean
wrench over x = [theta (1+8K), extra tilt (rotation vector x, y)] until the cradle carries nothing.

    python -m hover.trim --init trim.json --out trim_locked.json [--iters 6]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import mujoco
import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L

K = 3
NT = 1 + 8 * K
SCALE = np.array([1 / 0.226] * 3 + [1 / 1e-3] * 3)       # residual units: weight, mNm


def attitude(att0, tilt):
    rv = np.array([tilt[0], tilt[1], 0.0]); a = np.linalg.norm(rv); qe = np.array([1.0, 0, 0, 0])
    if a > 1e-12:
        mujoco.mju_axisAngle2Quat(qe, rv / a, a)
    q = np.zeros(4); mujoco.mju_mulQuat(q, qe, np.asarray(att0, float))
    return q


def cradle_wrench(args):
    x, att0, g0 = args
    try:
        bird = L.load_bird(FREE_HOVER_TG, theta_g=g0)
        return L.release(bird, x[:NT], attitude(att0, x[NT:]), K, n_cradle=16, n_avg=8).cradle_wrench
    except FloatingPointError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="json with x (theta + tilt) and att0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    t = json.load(open(a.init)); x, att0 = np.array(t["x"], float), np.array(t["att0"], float)
    g0 = x[0]            # dt is snapped to this flap rate; g stays close to it
    steps = np.concatenate([[0.01], np.full(NT - 1, 0.02), np.full(2, 0.02)])
    with Pool(os.cpu_count()) as pool:
        for it in range(a.iters):
            out = pool.map(cradle_wrench, [(x, att0, g0)] + [(x + s * np.eye(len(x))[j] * sg, att0, g0)
                                                              for j, s in enumerate(steps) for sg in (1, -1)])
            w = out[0]
            if w is None:
                raise SystemExit("base rollout unstable")
            print(f"it{it}: cradle F {np.round(w[:3] * 1e3, 2)} mN  M {np.round(w[3:] * 1e3, 3)} mNm  g {x[0]:+.4f} "
                  f"(a0 {65 * math.exp(x[0]):.1f})  tilt+ {np.round(np.degrees(x[NT:]), 2)} deg", flush=True)
            if np.abs(w[:3]).max() < 0.5e-3 and np.abs(w[3:]).max() < 0.02e-3:
                break
            J = np.stack([(out[1 + 2 * j] - out[2 + 2 * j]) * SCALE / (2 * steps[j])
                          if out[1 + 2 * j] is not None and out[2 + 2 * j] is not None else np.zeros(6)
                          for j in range(len(x))], 1)
            x = x - np.linalg.pinv(J) @ (w * SCALE)
            json.dump({"x": x.tolist(), "att0": att0.tolist(), "K": K, "J": J.tolist()}, open(a.out, "w"))
    json.dump({"x": x.tolist(), "att0": att0.tolist(), "K": K, "J": J.tolist() if it else None}, open(a.out, "w"))


if __name__ == "__main__":
    main()
