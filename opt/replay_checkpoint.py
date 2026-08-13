#!/usr/bin/env python3
"""Replay an optimizer checkpoint on MuJoCo CPU with its tuned tendons applied.

Thin CLI over design_io.reconstruct_design -- the .tg carries only baseline
tendons, so the per-tendon tuning is reapplied from the JSON artifacts and rolled
out via the SAME cpu_validate_design the optimizer used. Printed lift / peak
torque match the run's cpu_validation.json.

    uv run python opt/replay_checkpoint.py <design_dir> [--integrator euler implicitfast] [--timestep 0.06]
    uv run python opt/replay_checkpoint.py --selfcheck   # asserts it000 + cpu_best reproduce known peaks
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "opt") not in sys.path:
    sys.path.insert(0, str(ROOT / "opt"))

from design_io import validate, _demo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir", type=Path, nargs="?")
    ap.add_argument("--integrator", nargs="+", default=["euler", "implicitfast"])
    ap.add_argument("--timestep", type=float, default=0.06, help="ms (match the run)")
    ap.add_argument("--warmup", type=float, default=8.0, help="settling periods before sampling")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _demo()
        return
    if args.design_dir is None:
        ap.error("design_dir required (or --selfcheck)")
    res = validate(args.design_dir, args.integrator, timestep_ms=args.timestep, warmup=args.warmup)
    for integ, (L, T, P) in res.items():
        print(f"[{args.design_dir.name}/{integ:12s}] lift={L*1e3:7.1f} mN  "
              f"peak_torque={T*1e3:7.1f} mNm  power={P*1e3:6.0f} mW")


if __name__ == "__main__":
    main()
