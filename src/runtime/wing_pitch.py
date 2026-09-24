"""Wing pitch hinges on the CPU: the rest-state neutral and the dead-zone limiter.

``calibrate_wing_pitch_at_rest`` settles the structure with no command, then sets
both pitch DOFs to the model's configured neutral angle (``qpos0`` = the joint
``ref``, which the compiler emits from ``wing.pitch.initial``). That is
deterministic and left/right symmetric: the two hinges have mirrored axes (+z vs
-z), so the *same* qpos produces mirror-image wings. (An earlier version searched
for the pitch that "levels the pitch rod"; the rod lies along the hinge axis, so
every pitch was a solution and L and R landed on different ones.) The compiler
also runs it, in ``morphology.motor_tendons``, so changing it changes compiled
scenes.

``WingPitchLimiter`` is the dead-zone restorative spring on those hinges, which
are emitted passively free (zero stiffness/damping). It always applies velocity
``damping``, and adds a spring once the pitch leaves its one-sided deadband
around the neutral captured at construction: positive deviation uses the
upstroke pitch/stiffness, negative the downstroke. Build it after the rest
calibration and call :meth:`apply` once per physics step, before
``mujoco.mj_step`` (it accumulates into ``data.qfrc_applied``).
"""

from dataclasses import dataclass

import mujoco

from runtime.morphology_runtime import load_design_morphology


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
    """Settle the structure, then set both wing pitches to the neutral angle.

    ``data.time`` is left at the end of the settle, not reset.
    """
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    for wing_name in ("L_wing", "R_wing"):
        _reset_pitch_to_neutral(model, data, wing_name)
    mujoco.mj_forward(model, data)


@dataclass
class _PitchJoint:
    qpos_adr: int
    dof_adr: int
    center: float


class WingPitchLimiter:
    def __init__(self, model, data, morph=None):
        """``morph`` supplies ``wing.pitch``; without it the design at ``MORPHOLOGY_PATH`` does."""
        pitch = (load_design_morphology() if morph is None else morph).wing.pitch
        self.upstroke_pitch = float(pitch.upstroke_pitch_rad)
        self.downstroke_pitch = float(pitch.downstroke_pitch_rad)
        self.upstroke_stiffness = float(pitch.upstroke_stiffness)
        self.downstroke_stiffness = float(pitch.downstroke_stiffness)
        self.damping = float(pitch.damping)

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
