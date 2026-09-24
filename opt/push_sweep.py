#!/usr/bin/env python3
"""How hard a push can the hover controller take? One push at wingbeat 20, then 60 more wingbeats, on CPU.

    PYTHONPATH=src python opt/push_sweep.py --trim designs/hummer/free-hover/open_loop.json --params mlp.pkl
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover import trim as T
from verify_hover import mlp_policy

PUSHES = [(f"{axis} {mag:g} {unit}", np.eye(12)[idx] * mag)
          for axis, idx, unit, mags in (("side vel", 7, "m/s", (0.2, 0.4, 0.6, 0.8)), ("fwd vel", 6, "m/s", (0.4, 0.8)),
                                        ("up vel", 8, "m/s", (0.3, 0.6)), ("roll rate", 9, "rad/s", (1, 2, 3, 4)),
                                        ("pitch rate", 10, "rad/s", (1, 2, 3, 4)), ("yaw rate", 11, "rad/s", (2, 4, 8)))
          for mag in mags]
_W = {}


def _init(trim, params):
    t = json.load(open(trim)); x, att0 = np.array(t["x"]), np.array(t["att0"])
    bird = L.load_bird(FREE_HOVER_TG, theta_g=x[0])
    _W.update(bird=bird, x=x, rel=L.release(bird, x[:T.NT], T.attitude(att0, x[T.NT:]), T.K),
              pol=mlp_policy(pickle.load(open(params, "rb"))))


def run(push):
    name, d = push
    tr = L.fly(_W["bird"], _W["rel"], _W["x"][:T.NT], 80, policy=_W["pol"], kicks={20: d})
    tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
    ok = len(tr) == 81 and tilt[-10:].mean() < 2.0
    return name, ok, len(tr) - 1, tilt.max(), pos.max(), tilt[-10:].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trim", required=True)
    ap.add_argument("--params", required=True)
    a = ap.parse_args()
    with Pool(initializer=_init, initargs=(a.trim, a.params)) as p:
        for name, ok, n, tmax, pmax, tend in p.map(run, PUSHES):
            print(f"{name:22s}: {'RECOVERED' if ok else 'LOST at wb %d' % n if n < 80 else 'not settled'} | peak tilt {tmax:5.1f} deg, "
                  f"peak excursion {pmax:5.0f} mm, final tilt {tend:4.1f} deg", flush=True)


if __name__ == "__main__":
    main()
