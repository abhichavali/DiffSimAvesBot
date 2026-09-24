"""Headless MJX lift statistics: run_fixed_frequency.py's printout from a jitted lax.scan.

    uv run python src/run_mjx_fixed_frequency.py [--tg designs/hummer/hover/shoulder.tg] [--periods 5]

The drive starts as a step (ramp_duration=0), not ramped over the warmup like the
CPU runner, and the default design differs, so the two printouts are not the same
experiment.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from compile_tg import compile_tg
from morphology.model import PLATE_BIRDS, require_bird
from morphology.tg_parser import load_morphology
from run_fixed_frequency import print_lift_stats
from runtime.control_signal import load_control_for_morphology
from runtime.crank import sync_crank_disks_at_rest
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.morphology_runtime import DEFAULT_HUMMER_TG
from runtime.wing_pitch import calibrate_wing_pitch_at_rest

ROOT = Path(__file__).resolve().parent.parent


def run(tg_path: Path, scene_path: Path, periods: float, warmup_periods: float) -> None:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)
    require_bird(load_morphology(tg_path), *PLATE_BIRDS, tool="run_mjx_fixed_frequency")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    control = load_control_for_morphology(tg_path)
    period = control.period
    warmup_steps = math.ceil(warmup_periods * period / model.opt.timestep)
    sample_steps = math.ceil(periods * period / model.opt.timestep)

    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)
    drive = rollout.build_drive_config(model, ramp_duration=0.0)
    pitch = wing_pitch.config_from_runtime(model, centers=rollout._pitch_centers(model, data))
    aero = aero_runtime.build_aero_config(model)

    def step(carry, record):
        d, state = carry
        ctrl = rollout.drive_ctrl(d.ctrl, d.time, drive)
        d = d.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(d.qfrc_applied))
        d = wing_pitch.apply(d, pitch)
        qfrc, net_force_world, state = aero_runtime.aero_wrench(mx, d, aero, state)
        d = d.replace(qfrc_applied=d.qfrc_applied + qfrc)
        d = mjx.step(mx, d)
        return (d, state), jnp.where(record, net_force_world[2], 0.0)

    def scan(carry):
        records = jnp.concatenate([jnp.zeros(warmup_steps, dtype=bool), jnp.ones(sample_steps, dtype=bool)])
        return jax.lax.scan(step, carry, records)

    carry0 = (dx, aero_runtime.init_aero_state(aero))
    (_final, _state), lifts_all = jax.jit(scan)(carry0)
    print_lift_stats(control, periods, np.asarray(lifts_all[warmup_steps:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compiled morphology in MJX with its design control signal.")
    parser.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--periods", type=float, default=5.0)
    parser.add_argument("--warmup-periods", type=float, default=2.0)
    args = parser.parse_args()
    run(args.tg, args.scene, args.periods, args.warmup_periods)


if __name__ == "__main__":
    main()
