#!/usr/bin/env python3
"""Fly a flight control .tg (open loop + optional closed loop) on CPU MuJoCo (full solver, float64) with pushes.

    PYTHONPATH=src uv run python opt/verify_hover.py [--control designs/hummer/free-hover/control.tg]
        [--open-loop] [--lqr designs/hummer/free-hover/lqr.npz] [--beats 120]
"""
import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover.policy import load_flight_policy

PUSHES = {30: np.r_[0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],         # 0.2 m/s forward
          60: np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],         # 1 rad/s roll rate
          90: np.r_[0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0, 0.5, 0]}      # sideways + up + pitch rate


def load(tg, control, dt_ms=0.13):
    """(bird, released state, policy) for a design + flight control .tg."""
    fp = load_flight_policy(control)
    bird = L.load_bird(tg, control=control, dt_ms=dt_ms)
    return bird, L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock), fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--open-loop", action="store_true", help="ignore the control's closed loop")
    ap.add_argument("--lqr", type=Path, default=None, help="fly the LQR gain K from this npz instead of the MLP")
    ap.add_argument("--beats", type=int, default=120)
    ap.add_argument("--dt-ms", type=float, default=0.13, help="timestep (snapped to an integer number per wingbeat)")
    a = ap.parse_args()
    bird, rel, fp = load(a.tg, a.control, a.dt_ms)
    pol = None if a.open_loop or fp.mlp is None else fp.correction
    if a.lqr:
        Kg, u_max = np.load(a.lqr)["K"], 0.15
        pol = lambda s: np.concatenate([[0.0], np.clip(-Kg @ s, -u_max, u_max)])
    for name, kicks in (("no pushes", None), ("pushes at wb 30/60/90", PUSHES)):
        tr = L.fly(bird, rel, fp.theta, a.beats, policy=pol, kicks=kicks)
        tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
        spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
        ok = len(tr) == a.beats + 1
        print(f"{'closed' if pol else 'open'} loop, {name}: {'flew all %d wingbeats (%.1f s)' % (a.beats, a.beats * 2 * np.pi / fp.control.omega) if ok else 'LOST at wb %d' % (len(tr) - 1)}"
              f" | tilt max {tilt.max():5.1f} deg, last-20 mean {tilt[-20:].mean():4.1f} | pos max {pos.max():6.0f} mm, final {pos[-1]:6.0f} | "
              f"speed last-20 {spd[-20:].mean():5.0f} mm/s", flush=True)
        print("   tilt every 10 wb:", np.round(tilt[::10], 1).tolist())


if __name__ == "__main__":
    main()
