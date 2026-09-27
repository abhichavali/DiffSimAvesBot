#!/usr/bin/env python3
"""Noise Monte Carlo of a hover policy on CPU: seeds x dt x noise_scale x scenario, over all cores.

    PYTHONPATH=src .venv/bin/python opt/hover_mc.py --lqr lqr.npz --obs att,w --mode lqg \\
        --noise-scales 0 0.5 1 2 --seeds 8 --dt-ms 0.10 0.13 0.16 --scenarios none verify --json mc.json

Each job flies one flight (hover.evaluate.fly: CPU MuJoCo, float64, the same code verify_hover / push_sweep
run) with its own sensor noise seed. Scenarios: none (--beats wingbeats, no push), verify (verify_hover's
pushes at 30/60/90), sweep (each push_sweep push at wingbeat 20, 80 wingbeats). Pass = flew every wingbeat
(sweep: and mean tilt over the last 10 < 2 deg). With --lqr --obs, an LQG controller is designed per
noise scale (its V follows the flight's noise) unless --design-noise-scale fixes it; --ctrl flies one
saved controller for every noise scale.

Seeds are common random numbers: seed k draws the same biases in every dt / scenario / noise-scale cell, so a
rate rests on --seeds independent draws, not on the number of flights (the accel bias dominates the outcome).
pass does not gate tilt: read tilt_max_deg of the no-push runs (a steady accel-bias tilt of 1-2 deg is the
usual failure; finding 2026-09-26-hover-imu-noise).

CPU, not the MJX env (hover.env): the sensor model and the observers exist only on CPU, the MJX closed loop
is float32 with iterations=1 and has no free-flight parity check, and the CPU run is the reference the
verify tools already use. One flight of 120 wingbeats is ~25-30 s on one core.
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

import numpy as np

from design_paths import FREE_HOVER_TG
from hover import evaluate as E
from hover import sensors as S
from hover.policy import load_flight_policy

_W = {}


def _init(tg, control):
    _W.update(tg=tg, control=control)


def run(job):
    t0 = time.time()
    spec, dt_ms, scale, seed, scen, kicks, beats = job
    tr, sen, _ = E.fly(spec, _W["tg"], _W["control"], dt_ms, beats, kicks=kicks, seed=seed)
    fp = E.load(_W["tg"], _W["control"], dt_ms)[2]
    s = E.summarize(tr, beats, 2 * math.pi / fp.control.omega, sen)
    ok = s["flew_all"] and (not scen.startswith("sweep") or s["tilt_last10_deg"] < 2.0)
    return {"dt_ms": dt_ms, "noise_scale": scale, "seed": seed, "scenario": scen, "pass": bool(ok),
            "wall_s": time.time() - t0, **s}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--beats", type=int, default=120)
    ap.add_argument("--noise-scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--design-noise-scale", type=float, default=None, help="fix the LQG design noise (default: = flight noise)")
    ap.add_argument("--seeds", type=int, default=4, help="noise seeds per cell (seed, seed+1, ...)")
    ap.add_argument("--scenarios", nargs="+", default=["none", "verify"], choices=("none", "verify", "sweep"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    E.add_args(ap, dt_default=(0.10, 0.13, 0.16))
    a = ap.parse_args()
    fp = load_flight_policy(a.control)
    specs = {}
    for ns in a.noise_scales:
        a.noise_scale = ns
        if a.lqr and a.obs and a.design_noise_scale is not None:
            a.noise_scale = a.design_noise_scale
            spec = E.spec_from_args(a, a.control)
            spec["sensor"] = {"noise": S.SensorNoise(scale=ns, pos_scale=a.pos_noise_scale, pos_latency_ms=a.pos_latency_ms),
                              "att_mode": a.att_mode or "filter"}
        else:
            spec = E.spec_from_args(a, a.control)
        specs[ns] = spec
    scen = []
    for s in a.scenarios:
        if s == "none":
            scen.append(("none", None, a.beats))
        elif s == "verify":
            scen.append(("verify", E.VERIFY_PUSHES, a.beats))
        else:
            scen += [(f"sweep:{name}", {20: d}, 80) for name, d in E.SWEEP_PUSHES]
    jobs = [(specs[ns], dt, ns, seed, name, kicks, beats) for dt in a.dt_ms for ns in a.noise_scales
            for seed in range(a.seed, a.seed + a.seeds) for name, kicks, beats in scen]
    print(f"{len(jobs)} flights on {a.workers} workers | {specs[a.noise_scales[0]]['label']}", flush=True)
    t0, rows = time.time(), []
    with Pool(a.workers, initializer=_init, initargs=(a.tg, a.control)) as p:
        for r in p.imap_unordered(run, jobs):
            rows.append(r)
            if len(rows) % max(1, len(jobs) // 20) == 0:
                print(f"  {len(rows)}/{len(jobs)} done, {time.time() - t0:.0f} s", flush=True)
    agg = []
    for dt in a.dt_ms:
        for ns in a.noise_scales:
            for sname in a.scenarios:
                rs = [r for r in rows if r["dt_ms"] == dt and r["noise_scale"] == ns and r["scenario"].split(":")[0] == sname]
                if not rs:
                    continue
                g = {"dt_ms": dt, "noise_scale": ns, "scenario": sname, "n": len(rs),
                     "pass_rate": float(np.mean([r["pass"] for r in rs])), "beats_mean": float(np.mean([r["beats"] for r in rs])),
                     "beats_min": int(min(r["beats"] for r in rs)),
                     "tilt_max_deg_median": float(np.median([r["tilt_max_deg"] for r in rs])),
                     "tilt_last20_deg_median": float(np.median([r["tilt_last20_deg"] for r in rs])),
                     "drift_final_mm_median": float(np.median([r["drift_final_mm"] for r in rs]))}
                if "att_est_err_rms_deg" in rs[0]:
                    g["att_est_err_rms_deg_median"] = float(np.median([r["att_est_err_rms_deg"] for r in rs]))
                agg.append(g)
                print(f"dt {dt:5.3f} noise x{ns:<5g} {sname:7s}: pass {g['pass_rate'] * 100:5.1f}% of {g['n']:3d} | beats mean "
                      f"{g['beats_mean']:6.1f} min {g['beats_min']:4d} | tilt max med {g['tilt_max_deg_median']:5.2f} deg | "
                      f"drift med {g['drift_final_mm_median']:6.1f} mm", flush=True)
    wall = time.time() - t0
    print(f"{len(jobs)} flights in {wall:.0f} s ({wall * a.workers / max(len(jobs), 1):.1f} core-s per flight)")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        cfg = E.config(a, specs[a.noise_scales[0]])
        cfg.update(noise_scales=a.noise_scales, design_noise_scale=a.design_noise_scale, seeds=a.seeds,
                   scenarios=a.scenarios, beats=a.beats, wall_s=wall, workers=a.workers)
        a.json.write_text(json.dumps({"tool": "hover_mc", "config": cfg, "aggregate": agg,
                                      "runs": sorted(rows, key=lambda r: (r["dt_ms"], r["noise_scale"], r["scenario"], r["seed"]))},
                                     indent=1, default=float))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
