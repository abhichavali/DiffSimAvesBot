"""Initialize the wing pitch DOFs to a symmetric neutral angle at rest.

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
"""

import mujoco

from runtime.viewer_runtime import settle


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
        settle(model, data, settle_steps)

    for wing_name in ("L_wing", "R_wing"):
        _reset_pitch_to_neutral(model, data, wing_name)
    mujoco.mj_forward(model, data)
