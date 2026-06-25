"""Dead-zone restorative spring for the wing pitch joints.

The pitch hinges are emitted passively free (zero stiffness/damping). This
controller always applies velocity ``damping`` (an intrinsic property of the
joint), and additionally applies a restorative spring once the pitch exceeds
its one-sided pitch deadband from leveled-neutral rest. Positive pitch deviation
uses the upstroke pitch/stiffness; negative deviation uses downstroke.
Call :meth:`apply` once per physics step, before ``mujoco.mj_step`` (it
accumulates into ``data.qfrc_applied``).
"""

from dataclasses import dataclass

import mujoco

from runtime.runtime_config import load_runtime_config


@dataclass
class _PitchJoint:
    qpos_adr: int
    dof_adr: int
    center: float


class WingPitchLimiter:
    def __init__(self, model, data):
        params = (load_runtime_config() or {}).get("wing_pitch_parameters", {})
        fallback_free_angle = float(params.get("free_angle", 0.0))
        fallback_stiffness = float(params.get("limit_stiffness", 0.5))
        self.upstroke_pitch = float(params.get("upstroke_pitch", fallback_free_angle))
        self.downstroke_pitch = float(params.get("downstroke_pitch", fallback_free_angle))
        self.upstroke_stiffness = float(params.get("upstroke_stiffness", fallback_stiffness))
        self.downstroke_stiffness = float(params.get("downstroke_stiffness", fallback_stiffness))
        self.damping = float(params.get("limit_damping", 0.0005))

        self.joints: list[_PitchJoint] = []
        for joint_name in ("L_wing_pitch", "R_wing_pitch"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            self.joints.append(
                _PitchJoint(
                    qpos_adr=qpos_adr,
                    dof_adr=int(model.jnt_dofadr[joint_id]),
                    center=float(data.qpos[qpos_adr]),
                )
            )

    def apply(self, model, data) -> None:
        for joint in self.joints:
            vel = float(data.qvel[joint.dof_adr])
            torque = -self.damping * vel
            dev = float(data.qpos[joint.qpos_adr]) - joint.center
            free_angle = self.upstroke_pitch if dev >= 0.0 else self.downstroke_pitch
            stiffness = self.upstroke_stiffness if dev >= 0.0 else self.downstroke_stiffness
            excess = abs(dev) - free_angle
            if excess > 0.0:
                sign = 1.0 if dev > 0.0 else -1.0
                torque += -stiffness * excess * sign
            data.qfrc_applied[joint.dof_adr] += torque
