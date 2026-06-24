from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from compile_tg import compile_tg
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.mjx import aero_runtime, rollout, wing_pitch
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest


def run(tg_path: Path, scene_path: Path, frequency_hz: float, periods: float, warmup_periods: float) -> None:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    omega = math.tau * frequency_hz
    period = 1.0 / frequency_hz
    warmup_steps = math.ceil(warmup_periods * period / model.opt.timestep)
    sample_steps = math.ceil(periods * period / model.opt.timestep)

    mx = mjx.put_model(model)
    dx = mjx.put_data(model, data)
    drive = rollout.build_drive_config(model, motor_speed=omega, ramp_duration=0.0)
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
    lifts = np.asarray(lifts_all[warmup_steps:])
    print(f"frequency_hz {frequency_hz:.6g}")
    print(f"periods {periods:.6g}")
    print(f"samples {lifts.size}")
    print(f"mean_lift_N {float(lifts.mean()):.9g}")
    print(f"rms_lift_N {float(np.sqrt(np.mean(lifts * lifts))):.9g}")
    print(f"min_lift_N {float(lifts.min()):.9g}")
    print(f"max_lift_N {float(lifts.max()):.9g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compiled morphology in MJX at fixed crank frequency.")
    parser.add_argument("--tg", type=Path, default=ROOT / "design" / "shoulder.tg")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--frequency", type=float, default=10.0, help="Crank frequency in Hz")
    parser.add_argument("--periods", type=float, default=5.0)
    parser.add_argument("--warmup-periods", type=float, default=2.0)
    args = parser.parse_args()
    run(args.tg, args.scene, args.frequency, args.periods, args.warmup_periods)


if __name__ == "__main__":
    main()
