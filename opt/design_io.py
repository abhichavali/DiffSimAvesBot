#!/usr/bin/env python3
"""Replay a design dir on MuJoCo CPU: an optimizer output dir with its tuned tendons
applied, or a committed designs/ dir with its own control.tg.

Uses codesign.artifacts.validate: the same cpu_rollout the optimizer's CPU validation
used, so the printed lift / peak torque match the run's cpu_validation.json.

    uv run python opt/design_io.py <design_dir> [--integrator euler implicitfast] [--timestep 0.06]

Certify a design (docs/ROBUST_CODESIGN_METHODOLOGY.md section 12: 8 warmup + 24 sample
cycles; the worst peak over both commands is the certified number):

    uv run python opt/design_io.py designs/hummer/hover --integrator euler implicitfast rk4 --periods 24
    uv run python opt/design_io.py designs/hummer/hover --timestep 0.03 --periods 24
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
    ap.add_argument("--periods", type=float, default=3.0, help="sampled periods (certify: 24)")
    ap.add_argument("--pre-clock-fix-dt", type=float, default=None, metavar="MS",
                    help="the run's --timestep, for harmonics written before 649cab1 (2026-09-24): "
                         "rotates a_k/b_k by k*omega*2000*dt")
    args = ap.parse_args()
    res = validate(args.design_dir, args.integrator, timestep_ms=args.timestep, warmup=args.warmup,
                   periods=args.periods, pre_clock_fix_dt_ms=args.pre_clock_fix_dt)
    for integ, r in res.items():
        print(f"[{args.design_dir.name}/{integ:12s}] lift={r.lift*1e3:7.1f} mN  "
              f"peak_torque={r.peak*1e3:7.1f} mNm  power={r.power*1e3:6.0f} mW")


if __name__ == "__main__":
    main()
