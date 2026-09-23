"""Motor tendon length measurement at the simulator rest pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from morphology.model import Morphology

_DEFAULT_SCENE_DIR = Path(__file__).resolve().parents[2] / "templates" / "generated-joint"


@dataclass(frozen=True)
class MotorTendonRange:
    l_min: float
    l_max: float
    l_rest: float
    springlength: float


def _load_probe_model(xml: str, scene_dir: Path) -> mujoco.MjModel:
    """Load probe MJCF; write to disk so meshdir-relative assets resolve."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    probe_path = scene_dir / "_motor_tendon_probe.xml"
    probe_path.write_text(xml)
    try:
        return mujoco.MjModel.from_xml_path(str(probe_path))
    finally:
        probe_path.unlink(missing_ok=True)


def _sweep_tendon_length(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    disk_joint: str,
    tendon_name: str,
    steps: int = 72,
) -> tuple[float, float, float]:
    """Return (L_min, L_max, L_rest) while rotating one disk hinge through 2π."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, disk_joint)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    if jid < 0 or tid < 0:
        raise ValueError(f"Missing joint/tendon: {disk_joint!r} / {tendon_name!r}")

    adr = model.jnt_qposadr[jid]
    saved = float(data.qpos[adr])
    mujoco.mj_forward(model, data)
    l_rest = float(data.ten_length[tid])

    samples = []
    for angle in np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False):
        data.qpos[adr] = angle
        mujoco.mj_forward(model, data)
        samples.append(float(data.ten_length[tid]))

    data.qpos[adr] = saved
    mujoco.mj_forward(model, data)
    return min(samples), max(samples), l_rest


def measure_motor_tendon_ranges(
    xml: str,
    morph: Morphology,
    scene_dir: Path | None = None,
    *,
    settle_steps: int = 2000,
    sweep_steps: int = 72,
) -> Dict[str, MotorTendonRange]:
    """Full rotation envelope per motor tendon, with shared spring rest length."""
    from morphology.mujoco_memory import release_mujoco

    scene_dir = scene_dir or _DEFAULT_SCENE_DIR
    model = _load_probe_model(xml, scene_dir)
    data = mujoco.MjData(model)

    try:
        from runtime.wing_pitch import calibrate_wing_pitch_at_rest

        calibrate_wing_pitch_at_rest(model, data, settle_steps=settle_steps)

        ranges: Dict[str, MotorTendonRange] = {}
        for shoulder in ("L", "R"):
            for foot in morph.motors.feet:
                st = foot.station
                tendon_name = f"{shoulder}_motor_t{st}"
                disk_joint = f"{shoulder}_t{st}_disk_hinge"
                l_min, l_max, l_rest = _sweep_tendon_length(
                    model, data, disk_joint, tendon_name, steps=sweep_steps
                )
                margin = 0.002  # 2 mm headroom so limits are not hit at sweep extrema
                ranges[tendon_name] = MotorTendonRange(
                    l_min=max(0.0, l_min - margin),
                    l_max=l_max + margin,
                    l_rest=l_rest,
                    springlength=(
                        foot.tendon_rest_length
                        if foot.tendon_rest_length is not None
                        else morph.motors.tendon_rest_length
                        if morph.motors.tendon_rest_length is not None
                        else l_rest
                    ),
                )
        return ranges
    finally:
        release_mujoco(model, data)
