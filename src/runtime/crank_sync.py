"""Synchronize the left/right crank disks to a common rest phase at startup.

The settle pass in ``load_model`` steps the model for ~2000 frames with no motor
command. The velocity-driven disks creep a few degrees under tendon/gravity load
and end up at slightly different angles *and* at nonzero speed, asymmetrically
between the two shoulders. The result: when the flap ramp begins the L/R cranks
are already out of phase and mid-spin, so the wings appear to start at different
times. This resets every crank disk to a matched, zero-velocity phase so both
shoulders begin together.
"""

import mujoco

from morphology.tg_parser import load_morphology
from runtime.morphology_runtime import resolve_morphology_path


def sync_crank_disks_at_rest(model, data) -> None:
    feet = load_morphology(resolve_morphology_path()).motors.feet

    def joint_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    changed = False
    # Phasing is created by counter-rotation (the actuator gear), not the initial
    # angle, so every crank starts from a matched zero-velocity rest. The sigma
    # mirror swaps stations 1<->4, so the right crank mirroring L_t{a} is R_t{b}
    # (the other foot): bring it to L's angle so both wings begin from one phase.
    for left_foot, right_foot in zip(feet, reversed(feet)):
        left = joint_id(f"L_t{left_foot.station}_disk_hinge")
        right = joint_id(f"R_t{right_foot.station}_disk_hinge")
        if left < 0 or right < 0:
            continue
        data.qpos[model.jnt_qposadr[right]] = float(data.qpos[model.jnt_qposadr[left]])
        data.qvel[model.jnt_dofadr[left]] = 0.0
        data.qvel[model.jnt_dofadr[right]] = 0.0
        changed = True

    if changed:
        mujoco.mj_forward(model, data)
