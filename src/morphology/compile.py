"""The morphology compiler: Morphology -> CompiledMorphology -> MJCF string.

compile_morphology derives the left shoulder (strut pair, rest poses, motor
anchors, the 12 posed structural tendons, crank disks and, for the eagle, the
wrist tensegrity) and y-mirrors it into the right one. compile_to_mjcf emits a
probe scene, measures the motor tendons on it, and emits the final scene.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Dict

import numpy as np

from morphology.compiled import CompiledDisk, CompiledMorphology, CompiledShoulder, CompiledWrist
from morphology.emit_mjcf import emit_mjcf
from morphology.geometry import (
    StrutGeometry,
    compose,
    compute_body_poses,
    derive_disk_pillar_pos,
    derive_disk_rim_local,
    mirror_quat,
    mirror_y,
    motor_anchor_node,
    quat_from_axis_angle,
    quat_mul,
    quat_rotate,
    strut_pair,
    strut_quat,
    world_node,
)
from morphology.model import Morphology
from morphology.motor_tendons import measure_motor_tendon_ranges
from morphology.tendon_net import derive_tendon_length, expand_sigma_tendons

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def compile_morphology(morph: Morphology) -> CompiledMorphology:
    morph.validate()
    arc = morph.arc
    mobile_geom, fixed_geom = strut_pair(arc.radius, arc.width, arc.stations_deg, morph.arrange.inplane_offset,
                                         morph.world.scale, arc.arc_rod, arc.rung_rod)
    mobile_pos, mobile_quat, fixed_pos, fixed_quat = compute_body_poses(morph)

    for foot in morph.motors.feet:
        name, point, edge_point = motor_anchor_node(
            foot.station, foot.attach_along_edge_rad, mobile_geom, foot.attach_radius
        )
        mobile_geom.nodes[name] = point
        mobile_geom.site_quats[name] = mobile_geom.site_quats[f"t{foot.station}"]
        if edge_point is not None:
            edge_name = f"mrod_t{foot.station}"
            mobile_geom.nodes[edge_name] = edge_point
            mobile_geom.motor_rod_pairs.append((edge_name, name))

    edges = expand_sigma_tendons(morph.tendons)
    left = CompiledShoulder(
        side="L",
        mobile_pos=mobile_pos,
        mobile_quat=mobile_quat,
        fixed_pos=fixed_pos,
        fixed_quat=fixed_quat,
        mobile_geom=mobile_geom,
        fixed_geom=fixed_geom,
        structural_edges=edges,
        tendon_lengths=[
            derive_tendon_length(e, mobile_geom, fixed_geom, mobile_pos, mobile_quat, fixed_pos, fixed_quat)
            for e in edges
        ],
        disks=_compile_disks(morph, mobile_geom, mobile_pos, mobile_quat),
        wrist=_compile_wrist(morph, mobile_geom, mobile_pos, mobile_quat),
    )
    return CompiledMorphology(morph=morph, shoulders=(left, _mirror_shoulder(left, morph)))


def compile_to_mjcf(morph: Morphology, *, color_structural_tendons: bool = False) -> str:
    compiled = compile_morphology(morph)
    motor_ranges = measure_motor_tendon_ranges(emit_mjcf(compiled), morph)
    return emit_mjcf(compiled, motor_tendon_ranges=motor_ranges,
                     color_structural_tendons=color_structural_tendons)


def _compile_wrist(morph: Morphology, mobile_geom, mobile_pos, mobile_quat):
    """Build the wrist tensegrity pair, if the design declares one.

    The pair is constructed in the shoulder's own canonical frame — same arc
    builder, same six sigma generators, just a smaller radius — and then carried
    onto the ulna's distal end as a rigid unit. That keeps the tendon net's shape
    identical to the shoulder's, which is the whole point of "like the shoulder,
    but smaller".

    Getting there needs forward kinematics: the distal strut is a worldbody child
    (free joints must be top level), so its rest pose has to be resolved all the
    way out through strut -> humerus mount -> humerus -> ulna at qpos zero.
    """
    bones = morph.wing.bones
    if bones is None or bones.wrist is None:
        return None
    w = bones.wrist
    # Same proportion of the arc radius the shoulder uses for the in-plane
    # offset, so the two struts nest the same way at any size.
    distal_geom, proximal_geom = strut_pair(
        morph.arc.radius * w.scale, w.width, w.stations_deg, morph.arrange.inplane_offset * w.scale,
        morph.world.scale, w.arc_rod, w.rung_rod,
    )

    # Canonical placement, mirroring compute_body_poses: the two struts are
    # separated along +y, distal (mobile) inboard of proximal (fixed) by `gap`.
    # Carry the canonical frame onto the ulna's span: canonical +y becomes the
    # ulna's -z, so the distal strut lands OUTBOARD of the proximal one.
    carry = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -math.pi / 2.0)

    prox_quat = quat_mul(carry, strut_quat())
    dist_quat_rel = quat_mul(carry, strut_quat())

    # Forward kinematics to the ulna's distal end at qpos = 0.
    hum_pos, hum_quat = compose(mobile_pos, mobile_quat,
                                mobile_geom.midpoint, np.array(bones.mount_quat()))
    ulna_pos, ulna_quat = compose(hum_pos, hum_quat,
                                  np.array([0.0, 0.0, bones.humerus]), _IDENTITY_QUAT)
    tip_pos, tip_quat = compose(ulna_pos, ulna_quat,
                                np.array([0.0, 0.0, bones.ulna]), _IDENTITY_QUAT)

    # One strut on each bone, both centred on the bone line, `gap` apart along the
    # span. Each is placed by its OWN centroid: an arc's body origin sits off to
    # one side of the arc it describes, so placing by origin leaves the cage
    # hanging below the bones. Centring the pair as a unit instead keeps the
    # shoulder's interleaving, but it puts both cages in the same place along the
    # span — which is not what a wrist is. The ulna's strut sits on the ulna's
    # tip; the manus' strut sits one gap outboard, where the hand starts.
    prox_centroid = np.array([quat_rotate(prox_quat, n)
                              for n in proximal_geom.nodes.values()]).mean(axis=0)
    dist_centroid = np.array([quat_rotate(dist_quat_rel, n)
                              for n in distal_geom.nodes.values()]).mean(axis=0)
    prox_body = -prox_centroid
    dist_body = np.array([0.0, 0.0, w.gap]) - dist_centroid

    proximal_pos_world, proximal_quat_world = compose(tip_pos, tip_quat,
                                                      prox_body, prox_quat)
    distal_pos_world, distal_quat_world = compose(tip_pos, tip_quat,
                                                  dist_body, dist_quat_rel)

    edges = expand_sigma_tendons(morph.tendons)
    lengths = [
        derive_tendon_length(e, distal_geom, proximal_geom, distal_pos_world, distal_quat_world,
                             proximal_pos_world, proximal_quat_world)
        for e in edges
    ]

    # The hand starts exactly one gap outboard of the forearm's tip, so the joint
    # bridges a real gap: ulna ends, tensegrity, hand begins. Hanging the manus
    # back at the forearm's tip buries the strut pair inside the hand, and hanging
    # it off the distal strut's arc once put the hand INBOARD of its own joint.
    inv_distal = np.array([distal_quat_world[0], *(-distal_quat_world[1:])])
    hand_world = tip_pos + quat_rotate(tip_quat, np.array([0.0, 0.0, w.gap]))
    manus_pos = quat_rotate(inv_distal, hand_world - distal_pos_world)
    manus_quat = quat_mul(inv_distal, tip_quat)

    return CompiledWrist(
        # In the ULNA's frame: placed so the arc's centre lands on the tip.
        proximal_pos=np.array([0.0, 0.0, bones.ulna]) + prox_body,
        proximal_quat=prox_quat,
        distal_pos=distal_pos_world,
        distal_quat=distal_quat_world,
        proximal_geom=proximal_geom,
        distal_geom=distal_geom,
        structural_edges=edges,
        tendon_lengths=lengths,
        manus_pos=manus_pos,
        manus_quat=manus_quat,
    )


def _compile_disks(morph: Morphology, mobile_geom, mobile_pos, mobile_quat) -> Dict[str, CompiledDisk]:
    feet = morph.motors.feet
    foot_worlds = {f.station: world_node(mobile_geom.nodes[f"t{f.station}"], mobile_pos, mobile_quat)
                   for f in feet}
    disks: Dict[str, CompiledDisk] = {}
    for foot in feet:
        spec = morph.motors.disk
        if foot.crank_throw is not None:
            spec = dataclasses.replace(spec, crank_throw=foot.crank_throw)
        if foot.foot_outset is not None:
            spec = dataclasses.replace(spec, foot_outset=foot.foot_outset)
        sibling = 4 if foot.station == 1 else 1
        disks[f"L_t{foot.station}"] = CompiledDisk(
            pillar_pos=derive_disk_pillar_pos(foot_worlds[foot.station], foot_worlds[sibling], spec),
            euler=[morph.motors.disk.tilt_rad, morph.motors.disk.pitch_rad, 0.0],
            rim_pos=derive_disk_rim_local(spec),
            rim_rgba="0.2 0.8 0.2 1",
        )
    return disks


def _mirror_strut(geom: StrutGeometry, *, quats: bool = True) -> StrutGeometry:
    """The y-mirror of a strut; `quats=False` keeps the left side's site frames."""
    out = copy.deepcopy(geom)
    out.nodes = {k: mirror_y(v) for k, v in geom.nodes.items()}
    if quats:
        out.site_quats = {k: mirror_quat(v) for k, v in geom.site_quats.items()}
    return out


def _mirror_shoulder(left: CompiledShoulder, morph: Morphology) -> CompiledShoulder:
    """The right shoulder: the exact y-mirror of the left."""
    disks: Dict[str, CompiledDisk] = {}
    for name, disk in left.disks.items():
        # The R disk body is tilted -tilt where L is +tilt about its local X, so a
        # y-mirror of a disk-local point flips only its y. Negating x too puts the
        # motor-tendon pin on the wrong side of the rim and the right wing flaps
        # out of mirror.
        rim = disk.rim_pos.copy()
        rim[1] = -rim[1]
        disks[name.replace("L_", "R_")] = CompiledDisk(
            pillar_pos=mirror_y(disk.pillar_pos),
            euler=[-morph.motors.disk.tilt_rad, morph.motors.disk.pitch_rad, 0.0],
            rim_pos=rim,
            rim_rgba="0.2 0.2 0.8 1",
        )

    wrist = None
    if left.wrist is not None:
        lw = left.wrist
        # A relative transform mirrors the same way a world one does: for
        # world_R = M world_L M to hold at every link, each link's own rel must
        # satisfy rel_R = M rel_L M. So both the ulna-local proximal pose and the
        # distal-local manus pose mirror, not just the world-frame distal pose.
        wrist = CompiledWrist(
            proximal_pos=mirror_y(lw.proximal_pos),
            proximal_quat=mirror_quat(lw.proximal_quat),
            distal_pos=mirror_y(lw.distal_pos),
            distal_quat=mirror_quat(lw.distal_quat),
            proximal_geom=_mirror_strut(lw.proximal_geom),
            distal_geom=_mirror_strut(lw.distal_geom),
            structural_edges=lw.structural_edges,
            tendon_lengths=lw.tendon_lengths,
            manus_pos=mirror_y(lw.manus_pos),
            manus_quat=mirror_quat(lw.manus_quat),
        )

    return CompiledShoulder(
        side="R",
        mobile_pos=mirror_y(left.mobile_pos),
        mobile_quat=mirror_quat(left.mobile_quat),
        fixed_pos=mirror_y(left.fixed_pos),
        fixed_quat=mirror_quat(left.fixed_quat),
        mobile_geom=_mirror_strut(left.mobile_geom),
        # The fixed strut keeps the left side's site frames (cosmetic: sites do
        # not enter the dynamics).
        fixed_geom=_mirror_strut(left.fixed_geom, quats=False),
        structural_edges=left.structural_edges,
        tendon_lengths=left.tendon_lengths,
        disks=disks,
        wrist=wrist,
    )
