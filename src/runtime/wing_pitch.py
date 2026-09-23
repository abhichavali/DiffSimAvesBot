"""Wing pitch hinges on the CPU: the rest-state neutral and the dead-zone limiter.

``calibrate_wing_pitch_at_rest``: initialize the wing pitch DOFs to a symmetric
neutral angle at rest.

The pitch hinge spins the wing about the orange pitch rod -- but that rod lies
*along* the hinge axis, so the rod's world direction is invariant to the pitch
angle. The previous implementation tried to "level the rod" by searching for the
pitch that made it horizontal; because the objective is degenerate (every pitch
gives the same, already-horizontal rod) the search returned an essentially
arbitrary angle, and it landed differently on the left and right wings -- one
wing started visibly twisted the wrong way while the other looked fine.

Instead we set both pitch DOFs to the model's configured neutral angle
(``qpos0`` = the joint ``ref``, which the compiler emits from
``wing.pitch.initial``). This is deterministic and left/right symmetric: the two
hinges have mirrored axes (+z vs -z), so the *same* qpos produces mirror-image
wings. ``WingPitchLimiter`` then applies its free-zone spring centered on this
neutral.

``WingPitchLimiter``: dead-zone restorative spring for the wing pitch joints.

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


def _settle(model, data, steps=500):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _reset_pitch_to_neutral(model, data, wing_name: str) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{wing_name}_pitch")
    if joint_id < 0:
        return
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    # Configured neutral (joint `ref`); identical for L and R, so symmetric.
    data.qpos[qpos_adr] = float(model.qpos0[qpos_adr])
    if dof_adr >= 0:
        data.qvel[dof_adr] = 0.0
    # No built-in spring: the joint is passively free. WingPitchLimiter applies
    # the restorative torque at runtime, centered on this neutral angle.
    model.jnt_stiffness[joint_id] = 0.0


def calibrate_wing_pitch_at_rest(model, data, *, settle_steps: int = 2000) -> None:
    """Settle the structure, then set both wing pitches to the neutral angle."""
    if settle_steps > 0:
        _settle(model, data, settle_steps)

    for wing_name in ("L_wing", "R_wing"):
        _reset_pitch_to_neutral(model, data, wing_name)
    mujoco.mj_forward(model, data)


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
