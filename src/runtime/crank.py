"""Synchronize the left/right crank disks to a common rest phase at startup.

The rest settle (``wing_pitch.calibrate_wing_pitch_at_rest``) steps the model for
~2000 frames with no motor command. The velocity-driven disks creep a few degrees under tendon/gravity load
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
    # R_t{s}_disk is the Y-mirror of L_t{s}_disk (same station -- verified: every
    # L_/R_ body and site pair sits at mirrored xpos), and its hinge axis mirrors
    # with it, so the mirrored angle is the NEGATED one. The settle pass already
    # leaves them that way (+0.330 / -0.330 deg); this just cleans up the creep.
    # This used to pair L_t{a} with R_t{b} across the sigma station swap (1<->4)
    # and copy the angle unnegated, which took a mirror-correct rest state and
    # un-mirrored it: the two wings then traced paths 0.6-0.8% of stroke apart
    # for the whole run, on top of whatever the .tg baked in. Per-station and
    # negated, the wingtips agree to solver noise (tests/test_wing_lock.py).
    for foot in feet:
        left = joint_id(f"L_t{foot.station}_disk_hinge")
        right = joint_id(f"R_t{foot.station}_disk_hinge")
        if left < 0 or right < 0:
            continue
        data.qpos[model.jnt_qposadr[right]] = -float(data.qpos[model.jnt_qposadr[left]])
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
