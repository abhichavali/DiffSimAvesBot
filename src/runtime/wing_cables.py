"""Tensioning for the wing cables — the wing's own tensegrity.

The compiler emits two antagonist cables per wing, each threaded through an
anchor on the humerus, the ulna and the manus. What it cannot emit is their rest
lengths, because those only mean anything relative to the pose the wing is
actually in: a cable is tensioned when its spring is *shorter* than the distance
its anchors currently span.

So the fractions in the ``.tg`` are fractions of the compiled extended pose,
measured here at load time. ``extend = [1.00, 0.90]`` means "leave the cranial
cable slack at its natural length and pull the caudal one to 90% of its".

Zeroing the elbow servo is offered alongside, because a search over cable
tensions is meaningless while a stiff position servo is holding the joint
wherever it likes.
"""

from __future__ import annotations

from typing import Dict, Sequence

import mujoco
import numpy as np

CABLE_TAGS = ("cranial", "caudal")


def cable_names(model) -> list[str]:
    return [
        name
        for i in range(model.ntendon)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)) and "_cable_" in name
    ]


def measure_extended_lengths(model) -> Dict[str, float]:
    """Each cable's span at the compiled rest pose, which is the extended wing."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out: Dict[str, float] = {}
    for name in cable_names(model):
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        out[name] = float(data.ten_length[tid])
    return out


def apply_tension(model, fractions: Sequence[float], extended: Dict[str, float] | None = None) -> None:
    """Set both cables' rest lengths to ``fractions`` of their extended span.

    ``fractions`` is (cranial, caudal). Mutates the model in place — the same
    tensioning every runner and the search share, so a pose found by the search
    is reproduced exactly by the viewer.
    """
    if extended is None:
        extended = measure_extended_lengths(model)
    by_tag = dict(zip(CABLE_TAGS, (float(f) for f in fractions)))
    for name, length in extended.items():
        tag = next((t for t in CABLE_TAGS if name.endswith(t)), None)
        if tag is None:
            continue
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        model.tendon_lengthspring[tid] = length * by_tag[tag]


def release_fold_servos(model) -> None:
    """Zero the elbow servos' gain so the cables alone decide the wing's pose.

    A position actuator keeps pulling toward its setpoint however the cables are
    tensioned, so leaving it live would measure the servo, not the tensegrity.
    """
    for side in ("L", "R"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_elbow_servo")
        if aid < 0:
            continue
        model.actuator_gainprm[aid, 0] = 0.0
        model.actuator_biasprm[aid, 1] = 0.0
        model.actuator_biasprm[aid, 2] = 0.0


def wing_pose(model, data, side: str = "L") -> Dict[str, float]:
    """Span, elbow angle and wingtip offset — what a cable setting is judged on.

    Span is the wingtip's distance from the shoulder, which is the number that
    says "extended" or "contracted" regardless of how the two joints divided the
    work between them.
    """
    humerus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_humerus")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_wingtip")
    rot = np.asarray(data.xmat[humerus]).reshape(3, 3)
    local = rot.T @ (np.asarray(data.site_xpos[tip]) - np.asarray(data.xpos[humerus]))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_elbow")
    elbow = float(data.qpos[model.jnt_qposadr[jid]]) if jid >= 0 else float("nan")
    return {
        "span": float(np.linalg.norm(local)),
        "reach": float(local[2]),  # along the humerus' own span axis
        "cranial": float(-local[0]),  # how far the tip has swung ahead of the bone line
        "elbow": elbow,
    }
