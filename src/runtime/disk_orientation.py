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

import math

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


class StrokePlaneSignal:
    """The learned disk-mount (tilt, pitch) over crank phase.

    angle = angle0 + max_delta*tanh(coef . [1, cos th, sin th, cos 2th, sin 2th] / max_delta),
    th = the phase-reference crank hinge angle; coef[:5] drive tilt, coef[5:] pitch.
    optimize_control --tune-strokeplane saves it beside a design as
    strokeplane_optimized.json (read with from_json).
    """

    def __init__(self, model: mujoco.MjModel, coef, tilt0: float, pitch0: float, max_delta: float,
                 phase_joint: str = "L_t1_disk_hinge") -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, phase_joint)
        self.orient = DiskOrientation(model)
        self.qadr = int(model.jnt_qposadr[jid])
        self.coef = np.asarray(coef, dtype=float)
        self.tilt0 = float(tilt0)
        self.pitch0 = float(pitch0)
        self.max_delta = float(max_delta)

    @classmethod
    def from_json(cls, model: mujoco.MjModel, spec: dict) -> "StrokePlaneSignal":
        return cls(model, spec["tilt_coeffs"] + spec["pitch_coeffs"], spec["tilt0_rad"],
                   spec["pitch0_rad"], spec["max_delta_rad"], spec["phase_ref_joint"])

    def angles(self, data: mujoco.MjData) -> tuple[float, float]:
        th = float(data.qpos[self.qadr])
        basis = np.array([1.0, math.cos(th), math.sin(th), math.cos(2.0 * th), math.sin(2.0 * th)])
        mx = self.max_delta
        return (self.tilt0 + mx * math.tanh(float(self.coef[:5] @ basis) / mx),
                self.pitch0 + mx * math.tanh(float(self.coef[5:] @ basis) / mx))

    def apply(self, data: mujoco.MjData) -> None:
        self.orient.set(*self.angles(data))
