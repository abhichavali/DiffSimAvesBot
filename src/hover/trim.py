"""Open-loop trim of the free bird under the phase-locked drive: Gauss-Newton on the launch cradle's mean
wrench over x = [theta (1+8K), extra tilt (rotation vector x, y)] until the cradle carries nothing.

Reads a flight control .tg (hover.policy) as the starting point and writes the trimmed one (crank speed,
per-crank harmonics, hover attitude; its closed loop, if any, is dropped - re-derive it for the new trim).

    python -m hover.trim [--control designs/hummer/free-hover/control.tg] --out trimmed.tg [--jacobian J.npz]
"""
from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover.policy import load_flight_policy, write_flight_control

SCALE = np.array([1 / 0.226] * 3 + [1 / 1e-3] * 3)       # residual units: weight, mNm


def attitude(att0, tilt):
    """``att0`` (wxyz) rotated by the world-frame rotation vector (tilt_x, tilt_y, 0)."""
    rv = np.array([tilt[0], tilt[1], 0.0]); a = np.linalg.norm(rv); qe = np.array([1.0, 0, 0, 0])
    if a > 1e-12:
        mujoco.mju_axisAngle2Quat(qe, rv / a, a)
    q = np.zeros(4); mujoco.mju_mulQuat(q, qe, np.asarray(att0, float))
    return q


def cradle_wrench(args):
    x, att0, K, tg, control, k_phi = args
    try:
        bird = L.load_bird(tg, control=control)
        nt = 1 + 8 * K
        return L.release(bird, x[:nt], attitude(att0, x[nt:]), K, n_cradle=16, n_avg=8, k_phi=k_phi).cradle_wrench
    except FloatingPointError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jacobian", type=Path, default=None, help="also save the last trim Jacobian (npz)")
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    fp = load_flight_policy(a.control)
    K, att0 = fp.K, fp.attitude
    nt = 1 + 8 * K
    x = np.concatenate([fp.theta, [0.0, 0.0]])
    steps = np.concatenate([[0.01], np.full(nt - 1, 0.02), np.full(2, 0.02)])
    J = None
    with Pool(os.cpu_count()) as pool:
        for it in range(a.iters):
            common = (att0, K, a.tg, a.control, fp.phase_lock)
            out = pool.map(cradle_wrench, [(x, *common)] + [(x + s * np.eye(len(x))[j] * sg, *common)
                                                          for j, s in enumerate(steps) for sg in (1, -1)])
            w = out[0]
            if w is None:
                raise SystemExit("base rollout unstable")
            print(f"it{it}: cradle F {np.round(w[:3] * 1e3, 2)} mN  M {np.round(w[3:] * 1e3, 3)} mNm  "
                  f"a0 {fp.control.a0 * math.exp(x[0]):.3f}  tilt+ {np.round(np.degrees(x[nt:]), 2)} deg", flush=True)
            if np.abs(w[:3]).max() < 0.5e-3 and np.abs(w[3:]).max() < 0.02e-3:
                break
            J = np.stack([(out[1 + 2 * j] - out[2 + 2 * j]) * SCALE / (2 * steps[j])
                          if out[1 + 2 * j] is not None and out[2 + 2 * j] is not None else np.zeros(6)
                          for j in range(len(x))], 1)
            x = x - np.linalg.pinv(J) @ (w * SCALE)
    write_flight_control(a.out, a0=fp.control.a0 * math.exp(x[0]), depth_a=x[1:1 + 4 * K].reshape(4, K),
                         depth_b=x[1 + 4 * K:nt].reshape(4, K), attitude=attitude(att0, x[nt:]),
                         phase_lock=fp.phase_lock, bird=fp.control.bird,
                         header=f"Trimmed by hover.trim from {a.control.name}: the launch cradle carries "
                                f"{np.abs(w[:3]).max() * 1e3:.2f} mN / {np.abs(w[3:]).max() * 1e3:.3f} mNm.")
    if a.jacobian is not None and J is not None:
        np.savez(a.jacobian, J=J)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
