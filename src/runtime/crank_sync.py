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


def sync_crank_disks_at_rest(model, data, lag_rad: float | None = None) -> None:
    morph = load_morphology(resolve_morphology_path())
    feet = morph.motors.feet

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
    apply_crank_phase_lag(
        model,
        data,
        morph.motors.crank_phase_lag_rad if lag_rad is None else lag_rad,
    )


def apply_crank_phase_lag(model, data, lag_rad: float) -> None:
    """Phase-lag the non-reference (t4 'orange') cranks by ``lag_rad`` vs t1.

    The two cranks ALWAYS counter-rotate (actuator gear -1); this adds an initial
    crank-angle offset on top, a genuine phase lag that persists under the shared
    velocity command (both integrate the same speed, so the offset is constant).
    Call AFTER sync_crank_disks_at_rest. ``crank_phase_lag`` in the .tg sets it;
    sweep it to stagger the two motors' torque peaks (cuts overlap / inactive power).
    """
    if not lag_rad:
        return
    feet = load_morphology(resolve_morphology_path()).motors.feet
    ref = feet[0].station
    changed = False
    for foot in feet:
        if foot.station == ref:
            continue
        for side in ("L", "R"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_t{foot.station}_disk_hinge")
            if jid < 0:
                continue
            # R shoulder is the Y-mirror of L: its t4 crank runs the opposite gear
            # (see emit_mjcf._add_actuators). Adding the same qpos offset to both
            # sides would advance them in opposite world directions, mirror-breaking
            # the orange disks. Flip the sign on R so the offset is a true mirror and
            # both wings see the same t4-vs-t1 lag.
            sign = 1.0 if side == "L" else -1.0
            data.qpos[model.jnt_qposadr[jid]] += sign * float(lag_rad)
            changed = True
    if changed:
        mujoco.mj_forward(model, data)
