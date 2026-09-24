#!/usr/bin/env python3
"""Replay an optimizer design dir on MuJoCo CPU with its tuned tendons applied.

Uses codesign.artifacts.validate: the same cpu_rollout the optimizer's CPU validation
used, so the printed lift / peak torque match the run's cpu_validation.json.

    uv run python opt/design_io.py <design_dir> [--integrator euler implicitfast] [--timestep 0.06]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codesign.artifacts import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design_dir", type=Path)
    ap.add_argument("--integrator", nargs="+", default=["euler", "implicitfast"])
    ap.add_argument("--timestep", type=float, default=0.06, help="ms (match the run)")
    ap.add_argument("--warmup", type=float, default=8.0, help="settling periods before sampling")
    args = ap.parse_args()
    res = validate(args.design_dir, args.integrator, timestep_ms=args.timestep, warmup=args.warmup)
    for integ, r in res.items():
        print(f"[{args.design_dir.name}/{integ:12s}] lift={r.lift*1e3:7.1f} mN  "
              f"peak_torque={r.peak*1e3:7.1f} mNm  power={r.power*1e3:6.0f} mW")


if __name__ == "__main__":
    main()
