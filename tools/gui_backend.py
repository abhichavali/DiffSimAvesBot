#!/usr/bin/env python3
"""JSON backend for the Rust morphology comparison GUI.

The Rust UI owns tabs, threading, and plotting. This script keeps MuJoCo,
morphology parsing, and rollout math in the existing Python code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_fixed_frequency_gui import (  # noqa: E402
    data_by_phase,
    structural_tendon_stiffness_rows,
    coupled_control_path,
    wing_path_correlation,
    wing_path_data,
    work_loop_data,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("coupled-control")
    p.add_argument("--tg", type=Path, required=True)

    p = sub.add_parser("tendons")
    p.add_argument("--tg", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)

    p = sub.add_parser("phase")
    p.add_argument("--tg", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--plot-kind", required=True)
    p.add_argument("--periods", type=float, required=True)
    p.add_argument("--settle", type=float, required=True)

    p = sub.add_parser("work")
    p.add_argument("--tg", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--periods", type=float, required=True)
    p.add_argument("--settle", type=float, required=True)

    p = sub.add_parser("wing")
    p.add_argument("--tg", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--periods", type=float, required=True)
    p.add_argument("--settle", type=float, required=True)

    p = sub.add_parser("wing-correlation")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "coupled-control":
        out = {"control": str(coupled_control_path(args.tg))}
    elif args.cmd == "tendons":
        out = {
            "rows": [
                {"color": color, "name": name, "stiffness": stiffness}
                for color, name, stiffness in structural_tendon_stiffness_rows(args.tg, args.scene)
            ]
        }
    elif args.cmd == "phase":
        x, series, title, xlabel, ylabel, colors = data_by_phase(
            args.tg, args.scene, args.periods, args.plot_kind, args.control, args.settle
        )
        out = {"x": x, "series": series, "title": title, "xlabel": xlabel, "ylabel": ylabel, "colors": colors}
    elif args.cmd == "work":
        out = work_loop_data(args.tg, args.scene, args.periods, args.control, args.settle)
    elif args.cmd == "wing":
        out = wing_path_data(args.tg, args.scene, args.periods, args.control, args.settle)
    else:
        a = json.loads(args.a.read_text())
        b = json.loads(args.b.read_text())
        overall, per_axis = wing_path_correlation(a, b)
        out = {"overall": overall, "per_axis": per_axis}

    print(json.dumps(_jsonable(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
