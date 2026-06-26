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
    motors = load_morphology(resolve_morphology_path()).motors
    feet = motors.feet
    offset = float(motors.drive_phase_offset_rad)

    def joint_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    changed = False
    # Right wing matches the left exactly; its inversion is the actuator gear,
    # not the initial angle. Both cranks per shoulder sit `offset` apart.
    phases = {
        ("L", feet[0].station): 0.0,
        ("L", feet[1].station): offset,
        ("R", feet[0].station): 0.0,
        ("R", feet[1].station): offset,
    }
    for (side, station), phase in phases.items():
        joint = joint_id(f"{side}_t{station}_disk_hinge")
        if joint < 0:
            continue
        data.qpos[model.jnt_qposadr[joint]] = phase
        data.qvel[model.jnt_dofadr[joint]] = 0.0
        changed = True

    if changed:
        mujoco.mj_forward(model, data)
