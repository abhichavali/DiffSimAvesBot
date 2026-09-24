"""Measure the motor tendons' length envelope on a probe model of the compiled rig.

The compiler emits the scene once without motor-tendon ranges, loads it here,
lets the structure settle under gravity, then turns each crank disk through a
full revolution and records the tendon's shortest, longest and rest length. The
final emit uses those numbers for the tendon range and spring rest length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import mujoco
import numpy as np

from morphology.model import Morphology

SETTLE_STEPS = 2000
SWEEP_STEPS = 72
MARGIN = 0.002  # 2 mm headroom so limits are not hit at the sweep extrema


@dataclass(frozen=True)
class MotorTendonRange:
    l_min: float
    l_max: float
    l_rest: float
    springlength: float


def measure_motor_tendon_ranges(xml: str, morph: Morphology) -> Dict[str, MotorTendonRange]:
    """Full-rotation envelope per motor tendon, keyed by tendon name."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    # Each motor tendon runs from the strut to its own disk's pin, so all the
    # disks can turn together without changing each other's tendon lengths.
    feet = [(side, foot) for side in ("L", "R") for foot in morph.motors.feet]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_t{foot.station}_disk_hinge")
            for side, foot in feet]
    tids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"{side}_motor_t{foot.station}")
            for side, foot in feet]
    assert min(jids + tids) >= 0, "disk hinge or motor tendon missing from the probe scene"
    qadr = [model.jnt_qposadr[jid] for jid in jids]
    mujoco.mj_forward(model, data)
    rest = data.ten_length[tids]
    samples = []
    for angle in np.linspace(0.0, 2.0 * np.pi, SWEEP_STEPS, endpoint=False):
        data.qpos[qadr] = angle
        mujoco.mj_forward(model, data)
        samples.append(data.ten_length[tids])
    lo, hi = np.min(samples, axis=0), np.max(samples, axis=0)

    ranges: Dict[str, MotorTendonRange] = {}
    for (side, foot), l_min, l_max, l_rest in zip(feet, lo.tolist(), hi.tolist(), rest.tolist()):
        springlength = foot.tendon_rest_length
        if springlength is None:
            springlength = morph.motors.tendon_rest_length
        if springlength is None:
            springlength = l_rest
        ranges[f"{side}_motor_t{foot.station}"] = MotorTendonRange(
            max(0.0, l_min - MARGIN), l_max + MARGIN, l_rest, springlength)
    return ranges
