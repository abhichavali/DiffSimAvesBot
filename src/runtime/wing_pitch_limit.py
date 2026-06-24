"""Dead-zone restorative spring for the wing pitch joints.

The pitch hinges are emitted passively free (zero stiffness/damping). This
controller always applies velocity ``damping`` (an intrinsic property of the
joint), and additionally applies a restorative spring once the pitch exceeds
``free_angle`` of its leveled-neutral rest on either side. Call :meth:`apply`
once per physics step, before ``mujoco.mj_step`` (it accumulates into
``data.qfrc_applied``).
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
        self.free_angle = float(params.get("free_angle", 0.0))
        self.stiffness = float(params.get("limit_stiffness", 0.5))
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
            excess = abs(dev) - self.free_angle
            if excess > 0.0:
                sign = 1.0 if dev > 0.0 else -1.0
                torque += -self.stiffness * excess * sign
            data.qfrc_applied[joint.dof_adr] += torque
