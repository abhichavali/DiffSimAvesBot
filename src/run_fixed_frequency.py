"""Headless CPU lift statistics for a plate-wing design under its own control signal.

    uv run python src/run_fixed_frequency.py [--tg designs/hummer/legacy-baseline/shoulder.tg] [--periods 5]

Compiles the .tg, rests and crank-syncs it, ramps the drive in over the warmup
periods, then prints the mean/RMS/min/max summed aerodynamic lift (N) over the
sampled periods. ``run_mjx_fixed_frequency.py`` prints the same lines from MJX.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from compile_tg import compile_tg
from morphology.model import require_bird
from morphology.tg_parser import load_morphology
from runtime.control_signal import drive_command, load_control_for_morphology
from runtime.morphology_runtime import LEGACY_BASELINE_TG
from runtime.sim import PlateWingSim

ROOT = Path(__file__).resolve().parent.parent


def lift_trace(tg_path: Path, scene_path: Path, periods: float, warmup_periods: float):
    """The CPU drive loop: (control, per-step summed lift in N over the sampled periods)."""
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    require_bird(morph, "hummer", tool="run_fixed_frequency")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    sim = PlateWingSim(model, morph)
    data = sim.data

    control = load_control_for_morphology(tg_path)
    period = control.period
    drive_start = data.time
    ramp_seconds = warmup_periods * period

    def step() -> float:
        return sim.step(drive_command(control, morph.motors.motor_direction, data.time, drive_start, ramp_seconds))

    for _ in range(math.ceil(warmup_periods * period / model.opt.timestep)):
        step()

    lifts = np.array([step() for _ in range(math.ceil(periods * period / model.opt.timestep))])
    return control, lifts


def print_lift_stats(control, periods: float, lifts: np.ndarray) -> None:
    print(f"control_fundamental_frequency_hz {control.fundamental_frequency_hz:.6g}")
    print(f"control_harmonics {control.harmonics}")
    print(f"periods {periods:.6g}")
    print(f"samples {lifts.size}")
    print(f"mean_lift_N {float(lifts.mean()):.9g}")
    print(f"rms_lift_N {float(np.sqrt(np.mean(lifts * lifts))):.9g}")
    print(f"min_lift_N {float(lifts.min()):.9g}")
    print(f"max_lift_N {float(lifts.max()):.9g}")


def run(tg_path: Path, scene_path: Path, periods: float, warmup_periods: float) -> None:
    control, lifts = lift_trace(tg_path, scene_path, periods, warmup_periods)
    print_lift_stats(control, periods, lifts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compiled morphology with its design control signal.")
    parser.add_argument("--tg", type=Path, default=LEGACY_BASELINE_TG)
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--periods", type=float, default=5.0)
    parser.add_argument("--warmup-periods", type=float, default=2.0)
    args = parser.parse_args()
    run(args.tg, args.scene, args.periods, args.warmup_periods)


if __name__ == "__main__":
    main()
