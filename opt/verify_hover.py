#!/usr/bin/env python3
"""Fly a flight control .tg (open loop + optional closed loop) on CPU MuJoCo (full solver, float64) with pushes.

    PYTHONPATH=src uv run python opt/verify_hover.py [--control designs/hummer/free-hover/control.tg]
        [--open-loop] [--lqr designs/hummer/free-hover/lqr.npz] [--beats 120] [--dt-ms 0.10 0.13 0.16]
        [--obs att,w --mode lqg|sof | --ctrl ctl.npz] [--noise-scale 1 --seed 0 --att-mode filter] [--json out.json]

Policy and sensor flags: hover.evaluate. Without sensor flags the policy reads the true strobed state.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import evaluate as E

PUSHES = E.VERIFY_PUSHES


load = E.load        # (bird, released state, flight policy) for a design + flight control .tg, cached per dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--beats", type=int, default=120)
    E.add_args(ap)
    a = ap.parse_args()
    spec = E.spec_from_args(a, a.control)
    runs = []
    for dt_ms in a.dt_ms:
        bird, rel, fp = load(a.tg, a.control, dt_ms)
        beat_s = 2 * np.pi / fp.control.omega
        if len(a.dt_ms) > 1 or spec["sensor"] is not None or spec["policy"] == "linear" and spec["ctl"]["kind"] != "state":
            print(f"== dt {dt_ms} ms (snapped {bird.model.opt.timestep * 1e3:.5f}) | {spec['label']}"
                  + (f" | sensors noise x{spec['sensor']['noise'].scale:g}, att {spec['sensor']['att_mode']}, seed {a.seed}"
                     if spec["sensor"] else ""), flush=True)
        for name, kicks in (("no pushes", None), ("pushes at wb 30/60/90", PUSHES)):
            tr, sen, _ = E.fly(spec, a.tg, a.control, dt_ms, a.beats, kicks=kicks, seed=a.seed)
            tilt = np.degrees(np.linalg.norm(tr[:, 3:5], axis=1)); pos = np.linalg.norm(tr[:, :3], axis=1) * 1e3
            spd = np.linalg.norm(tr[:, 6:9], axis=1) * 1e3
            ok = len(tr) == a.beats + 1
            print(f"{'open' if spec['policy'] == 'open' else 'closed'} loop, {name}: {'flew all %d wingbeats (%.1f s)' % (a.beats, a.beats * beat_s) if ok else 'LOST at wb %d' % (len(tr) - 1)}"
                  f" | tilt max {tilt.max():5.1f} deg, last-20 mean {tilt[-20:].mean():4.1f} | pos max {pos.max():6.0f} mm, final {pos[-1]:6.0f} | "
                  f"speed last-20 {spd[-20:].mean():5.0f} mm/s", flush=True)
            print("   tilt every 10 wb:", np.round(tilt[::10], 1).tolist())
            runs.append({"dt_ms": dt_ms, "timestep_ms": bird.model.opt.timestep * 1e3, "scenario": "no_push" if kicks is None else "verify_pushes",
                         **E.summarize(tr, a.beats, beat_s, sen)})
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"tool": "verify_hover", "beats": a.beats, "config": E.config(a, spec),
                                      "pass": all(r["flew_all"] for r in runs), "runs": runs}, indent=1, default=float))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
