"""Kinematic stroke-plane control: re-orient the crank-disk mounts at runtime.

The disk mount orientation is compiled into MJCF as a fixed euler
[±tilt, pitch, 0] on each ``*_disk`` body (see morphology/compile.py).
MuJoCo re-derives world kinematics from ``model.body_quat`` every step, so
rewriting those quats steers the stroke plane live — no extra joints or
actuators needed. The crank hinge keeps spinning about the reoriented local
z axis, so the whole flap arc follows the new plane.

ponytail: kinematic override only — the plane change exerts no reaction
torque on the pillar; add real gimbal joints if that dynamics ever matters.
"""

from __future__ import annotations

import mujoco
import numpy as np

_DISKS = ("L_t1", "L_t4", "R_t1", "R_t4")


class DiskOrientation:
    """Sets the (tilt, pitch) mount angles of all four crank disks.

    tilt is sign-mirrored on the right side, matching the compiled morphology
    (compile.py _mirror_shoulder: R euler = [-tilt, pitch, 0]).
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        # (body_id, tilt_sign) per disk
        self.bodies: list[tuple[int, float]] = []
        for name in _DISKS:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_disk")
            if body_id >= 0:
                self.bodies.append((body_id, 1.0 if name.startswith("L") else -1.0))
        self._quat = np.zeros(4)

    def set(self, tilt_rad: float, pitch_rad: float) -> None:
        for body_id, sign in self.bodies:
            # "xyz" matches the MJCF compiler's default eulerseq.
            mujoco.mju_euler2Quat(self._quat, [sign * tilt_rad, pitch_rad, 0.0], "xyz")
            self.model.body_quat[body_id] = self._quat
