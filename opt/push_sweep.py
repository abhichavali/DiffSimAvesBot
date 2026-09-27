#!/usr/bin/env python3
"""How hard a push can a flight control take? One push at wingbeat 20, then 60 more wingbeats, on CPU.

    uv run python opt/push_sweep.py [--control designs/hummer/free-hover/control.tg] [--dt-ms 0.10 0.13 0.16]
        [--lqr lqr.npz [--obs att,w --mode lqg] | --ctrl ctl.npz] [--noise-scale 1 --seed 0] [--json out.json]

Pass per push = all 81 strobes flown and mean tilt over the last 10 wingbeats < 2 deg. Policy and sensor
flags: hover.evaluate.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import evaluate as E

PUSHES = E.SWEEP_PUSHES
_W = {}


def _init(tg, control, spec, seed):
    _W.update(tg=tg, control=control, spec=spec, seed=seed)


def run(job):
    dt_ms, (name, d) = job
    tr, sen, _ = E.fly(_W["spec"], _W["tg"], _W["control"], dt_ms, 80, kicks={20: d}, seed=_W["seed"])
    tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
    ok = len(tr) == 81 and tilt[-10:].mean() < 2.0
    return dt_ms, name, ok, len(tr) - 1, tilt.max(), pos.max(), tilt[-10:].mean(), E.summarize(tr, 80, 0.0, sen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    E.add_args(ap)
    a = ap.parse_args()
    spec = E.spec_from_args(a, a.control)
    jobs = [(dt, p) for dt in a.dt_ms for p in PUSHES]
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.tg, a.control, spec, a.seed)) as p:
        for dt_ms, name, ok, n, tmax, pmax, tend, summ in p.imap(run, jobs):
            if name == PUSHES[0][0] and (len(a.dt_ms) > 1 or spec["sensor"] is not None or spec["policy"] == "linear"):
                print(f"== dt {dt_ms} ms | {spec['label']}" + (f" | sensors noise x{spec['sensor']['noise'].scale:g}, seed {a.seed}"
                                                              if spec["sensor"] else ""), flush=True)
            print(f"{name:22s}: {'RECOVERED' if ok else 'LOST at wb %d' % n if n < 80 else 'not settled'} | peak tilt {tmax:5.1f} deg, "
                  f"peak excursion {pmax:5.0f} mm, final tilt {tend:4.1f} deg", flush=True)
            rows.append({"dt_ms": dt_ms, "push": name, "recovered": bool(ok), "beats": n, "peak_tilt_deg": float(tmax),
                         "peak_excursion_mm": float(pmax), "final_tilt_deg": float(tend),
                         **{k: v for k, v in summ.items() if k.startswith("att_est") or k.startswith("yaw_est")}})
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        per_dt = {str(dt): sum(r["recovered"] for r in rows if r["dt_ms"] == dt) for dt in a.dt_ms}
        a.json.write_text(json.dumps({"tool": "push_sweep", "config": E.config(a, spec), "n_pushes": len(PUSHES),
                                      "recovered_per_dt": per_dt, "runs": rows}, indent=1, default=float))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
