"""Main morphology compiler pipeline."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Tuple

import numpy as np

from morphology.compiled import CompiledMorphology, CompiledShoulder, CompiledWrist
from morphology.geometry import (
    ArcPlane,
    apply_scale_to_strut,
    build_strut_geometry,
    compose,
    compute_body_poses,
    derive_arc_center,
    derive_center_of_pressure,
    derive_disk_pillar_pos,
    derive_disk_rim_local,
    derive_fixed_width_axis_offset,
    fixed_body_orientation,
    mobile_body_orientation,
    motor_anchor_node,
    quat_from_axis_angle,
    quat_mul,
    quat_rotate,
    scaled_rod,
    WingSpec,
    world_node,
)
from morphology.model import Arrange, Face, Morphology, TendonGenerator
from morphology.symmetry import StructuralEdge, expand_sigma_tendons
from morphology.tendons import (
    SHOULDER_GENERATORS,
    anchor_node_name,
    derive_tendon_length,
    edge_topology_key,
)


def compile_morphology(morph: Morphology) -> CompiledMorphology:
    morph.validate()
    scale = morph.world.scale

    radius = morph.arc.radius
    width = morph.arc.width
    stations = morph.arc.stations_deg

    mobile_center = derive_arc_center(radius, stations, ArcPlane.XZ)
    fixed_center = derive_arc_center(
        radius,
        stations,
        ArcPlane.YZ,
        mobile_center_x=mobile_center[0],
    )
    fixed_x = derive_fixed_width_axis_offset(mobile_center[0], morph.arrange)

    mobile_geom = build_strut_geometry(
        radius, width, stations, ArcPlane.XZ, center=mobile_center
    )
    fixed_geom = build_strut_geometry(
        radius,
        width,
        stations,
        ArcPlane.YZ,
        center=fixed_center,
        width_axis_offset=fixed_x,
    )

    mobile_geom = apply_scale_to_strut(mobile_geom, scale)
    fixed_geom = apply_scale_to_strut(fixed_geom, scale)

    arc_r, arc_m = scaled_rod(morph.arc.arc_rod.radius, morph.arc.arc_rod.mass, scale)
    rung_r, rung_m = scaled_rod(morph.arc.rung_rod.radius, morph.arc.rung_rod.mass, scale)
    mobile_geom.arc_rod = (arc_r, arc_m)
    mobile_geom.rung_rod = (rung_r, rung_m)
    fixed_geom.arc_rod = (arc_r, arc_m)
    fixed_geom.rung_rod = (rung_r, rung_m)

    mobile_pos, mobile_quat, fixed_pos, fixed_quat = compute_body_poses(morph)

    motor_nodes: Dict[str, np.ndarray] = {}
    for foot in morph.motors.feet:
        name, point, edge_point = motor_anchor_node(
            foot.station, foot.attach_along_edge_rad, mobile_geom, foot.attach_radius
        )
        motor_nodes[name] = point
        mobile_geom.nodes[name] = point
        if name in mobile_geom.site_quats:
            del mobile_geom.site_quats[name]
        src = mobile_geom.site_quats.get(f"t{foot.station}", np.array([0.707107, 0.707107, 0.0, 0.0]))
        mobile_geom.site_quats[name] = src
        if edge_point is not None:
            edge_name = f"mrod_t{foot.station}"
            mobile_geom.nodes[edge_name] = edge_point
            mobile_geom.motor_rod_pairs.append((edge_name, name))

    edges = expand_sigma_tendons(morph.tendons)
    tendon_lengths: Dict[str, float] = {}
    for edge in edges:
        length = derive_tendon_length(
            edge, mobile_geom, fixed_geom, mobile_pos, mobile_quat, fixed_pos, fixed_quat
        )
        key = edge_topology_key(edge)
        tendon_lengths[key] = length

    disks = _compile_disks(morph, mobile_geom, mobile_pos, mobile_quat)

    wrist = _compile_wrist(morph, mobile_geom, mobile_pos, mobile_quat)

    left = CompiledShoulder(
        side="L",
        mobile_body="L_part_1",
        fixed_body="L_part_2",
        mobile_pos=mobile_pos,
        mobile_quat=mobile_quat,
        fixed_pos=fixed_pos,
        fixed_quat=fixed_quat,
        mobile_geom=mobile_geom,
        fixed_geom=fixed_geom,
        motor_nodes=motor_nodes,
        structural_edges=edges,
        tendon_lengths=tendon_lengths,
        disks=disks,
        wrist=wrist,
    )
    right = _mirror_shoulder(left, morph)
    wing_cop = derive_center_of_pressure(
        WingSpec(rod_length=morph.wing.rod_length * scale, mount_arm=morph.wing.mount_arm * scale)
    )

    return CompiledMorphology(
        morph=morph,
        shoulders=(left, right),
        wing_cop_local=wing_cop,
    )


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
    scale = morph.world.scale

    radius = morph.arc.radius * w.scale
    arrange = Arrange(
        gap=w.gap / scale,  # apply_scale_to_strut shrinks the pair afterwards
        angle_deg=0.0,
        mobile_dx=0.0,
        # Same proportion of the arc radius the shoulder uses, so the two struts
        # nest the same way at any size.
        inplane_offset=morph.arrange.inplane_offset * w.scale,
    )
    mob_center = derive_arc_center(radius, w.stations_deg, ArcPlane.XZ)
    fix_center = derive_arc_center(radius, w.stations_deg, ArcPlane.YZ,
                                   mobile_center_x=mob_center[0])
    fix_x = derive_fixed_width_axis_offset(mob_center[0], arrange)

    distal_geom = build_strut_geometry(radius, w.width, w.stations_deg, ArcPlane.XZ,
                                       center=mob_center)
    proximal_geom = build_strut_geometry(radius, w.width, w.stations_deg, ArcPlane.YZ,
                                         center=fix_center, width_axis_offset=fix_x)
    distal_geom = apply_scale_to_strut(distal_geom, scale)
    proximal_geom = apply_scale_to_strut(proximal_geom, scale)
    for geom in (distal_geom, proximal_geom):
        geom.arc_rod = scaled_rod(w.arc_rod.radius, w.arc_rod.mass, scale)
        geom.rung_rod = scaled_rod(w.rung_rod.radius, w.rung_rod.mass, scale)

    # Canonical placement, mirroring compute_body_poses: the two struts are
    # separated along +y, distal (mobile) inboard of proximal (fixed) by `gap`.
    # Carry the canonical frame onto the ulna's span: canonical +y becomes the
    # ulna's -z, so the distal strut lands OUTBOARD of the proximal one.
    carry = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -math.pi / 2.0)

    prox_quat = quat_mul(carry, fixed_body_orientation())
    dist_quat_rel = quat_mul(carry, mobile_body_orientation())

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
    lengths: Dict[str, float] = {}
    for edge in edges:
        lengths[edge_topology_key(edge)] = derive_tendon_length(
            edge, distal_geom, proximal_geom,
            distal_pos_world, distal_quat_world,
            proximal_pos_world, proximal_quat_world,
        )

    # The hand starts AT the distal strut, so the joint bridges a real gap: ulna
    # ends, tensegrity, hand begins. Hanging the manus back at the forearm's tip
    # instead (which keeps the planform identical) buries the strut pair inside
    # the hand, and then the joint reads as a cage lying on the feathers rather
    # than as the thing holding them on.
    # The hand starts exactly one gap outboard of the forearm's tip, stated
    # directly rather than inferred from the distal strut's arc — centring the
    # pair moves that arc, and hanging the manus off it once put the hand INBOARD
    # of the joint it is supposed to be distal to.
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


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _compile_disks(
    morph: Morphology,
    mobile_geom,
    mobile_pos: np.ndarray,
    mobile_quat: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    disks: Dict[str, Dict[str, Any]] = {}
    foot_worlds: Dict[int, np.ndarray] = {}
    for foot in morph.motors.feet:
        node = mobile_geom.nodes[f"t{foot.station}"]
        foot_worlds[foot.station] = world_node(node, mobile_pos, mobile_quat)

    for foot in morph.motors.feet:
        st = foot.station
        sibling = 4 if st == 1 else 1
        foot_world = foot_worlds[st]
        name = f"L_t{st}"
        disk_spec = morph.motors.disk
        if foot.crank_throw is not None:
            disk_spec = dataclasses.replace(disk_spec, crank_throw=foot.crank_throw)
        if foot.foot_outset is not None:
            disk_spec = dataclasses.replace(disk_spec, foot_outset=foot.foot_outset)
        disks[name] = {
            "pillar_pos": derive_disk_pillar_pos(
                foot_world,
                foot_worlds[sibling],
                disk_spec,
                morph.placement.mount_height,
            ),
            "euler": [morph.motors.disk.tilt_rad, morph.motors.disk.pitch_rad, 0.0],
            "rim_pos": derive_disk_rim_local(disk_spec),
            "rim_rgba": "0.2 0.8 0.2 1",
            "foot_station": foot.station,
        }
    return disks


def _mirror_shoulder(left: CompiledShoulder, morph: Morphology) -> CompiledShoulder:
    from morphology.symmetry import mirror_quat_wxyz, mirror_vec3

    def mirror_point(p: np.ndarray) -> np.ndarray:
        return np.array(mirror_vec3(tuple(p)))

    def mirror_nodes(nodes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: mirror_point(v) for k, v in nodes.items()}

    def mirror_quat(q: np.ndarray) -> np.ndarray:
        return np.array(mirror_quat_wxyz(tuple(q)))

    mobile_pos = mirror_point(left.mobile_pos)
    fixed_pos = mirror_point(left.fixed_pos)
    mobile_quat = mirror_quat(left.mobile_quat)
    fixed_quat = mirror_quat(left.fixed_quat)

    mobile_geom = left.mobile_geom
    fixed_geom = left.fixed_geom

    import copy

    mobile_geom_r = copy.deepcopy(mobile_geom)
    fixed_geom_r = copy.deepcopy(fixed_geom)
    mobile_geom_r.nodes = mirror_nodes(mobile_geom.nodes)
    fixed_geom_r.nodes = mirror_nodes(fixed_geom.nodes)
    mobile_geom_r.midpoint = mirror_point(mobile_geom.midpoint)
    mobile_geom_r.site_quats = {k: mirror_quat(v) for k, v in mobile_geom.site_quats.items()}

    motor_nodes = {k: mirror_point(v) for k, v in left.motor_nodes.items()}
    for k, v in motor_nodes.items():
        mobile_geom_r.nodes[k] = v

    edges = left.structural_edges
    tendon_lengths = dict(left.tendon_lengths)

    disks: Dict[str, Dict[str, Any]] = {}
    for name, spec in left.disks.items():
        rname = name.replace("L_", "R_")
        pillar = mirror_point(np.array(spec["pillar_pos"]))
        # Disk-local rim mirror across Y-plane: the disk body is tilted by
        # +tilt on L and -tilt on R about its local X axis. With X preserved
        # under that mirror, only the local Y component of rim_pos flips —
        # negating X too puts the motor-tendon anchor on the wrong side of the
        # rim and forces the right wing into a non-mirrored flap.
        rim = list(spec["rim_pos"])
        rim[1] = -float(rim[1])
        disks[rname] = {
            **spec,
            "pillar_pos": pillar,
            "euler": [-morph.motors.disk.tilt_rad, morph.motors.disk.pitch_rad, 0.0],
            "rim_pos": rim,
            "rim_rgba": "0.2 0.2 0.8 1",
        }

    wrist = None
    if left.wrist is not None:
        lw = left.wrist
        prox_r = copy.deepcopy(lw.proximal_geom)
        dist_r = copy.deepcopy(lw.distal_geom)
        prox_r.nodes = mirror_nodes(lw.proximal_geom.nodes)
        dist_r.nodes = mirror_nodes(lw.distal_geom.nodes)
        prox_r.midpoint = mirror_point(lw.proximal_geom.midpoint)
        dist_r.midpoint = mirror_point(lw.distal_geom.midpoint)
        prox_r.site_quats = {k: mirror_quat(v) for k, v in lw.proximal_geom.site_quats.items()}
        dist_r.site_quats = {k: mirror_quat(v) for k, v in lw.distal_geom.site_quats.items()}
        # A relative transform mirrors the same way a world one does: for
        # world_R = M world_L M to hold at every link, each link's own rel must
        # satisfy rel_R = M rel_L M. So both the ulna-local proximal pose and the
        # distal-local manus pose mirror, not just the world-frame distal pose.
        wrist = CompiledWrist(
            proximal_pos=mirror_point(lw.proximal_pos),
            proximal_quat=mirror_quat(lw.proximal_quat),
            distal_pos=mirror_point(lw.distal_pos),
            distal_quat=mirror_quat(lw.distal_quat),
            proximal_geom=prox_r,
            distal_geom=dist_r,
            structural_edges=lw.structural_edges,
            tendon_lengths=dict(lw.tendon_lengths),
            manus_pos=mirror_point(lw.manus_pos),
            manus_quat=mirror_quat(lw.manus_quat),
        )

    return CompiledShoulder(
        side="R",
        mobile_body="R_part_1",
        fixed_body="R_part_2",
        mobile_pos=mobile_pos,
        mobile_quat=mobile_quat,
        fixed_pos=fixed_pos,
        fixed_quat=fixed_quat,
        mobile_geom=mobile_geom_r,
        fixed_geom=fixed_geom_r,
        motor_nodes=motor_nodes,
        structural_edges=edges,
        tendon_lengths=tendon_lengths,
        disks=disks,
        wrist=wrist,
    )


def compile_to_mjcf(
    morph: Morphology,
    *,
    settle_steps: int = 2000,
    tendon_sweep_steps: int = 72,
    color_structural_tendons: bool = False,
) -> str:
    from morphology.emit_mjcf import emit_mjcf
    from morphology.motor_tendons import measure_motor_tendon_ranges

    compiled = compile_morphology(morph)
    probe_xml = emit_mjcf(compiled)
    motor_ranges = measure_motor_tendon_ranges(
        probe_xml,
        morph,
        settle_steps=settle_steps,
        sweep_steps=tendon_sweep_steps,
    )
    del probe_xml
    xml = emit_mjcf(
        compiled,
        motor_tendon_ranges=motor_ranges,
        color_structural_tendons=color_structural_tendons,
    )
    del compiled, motor_ranges
    return xml


def default_generators() -> Tuple[TendonGenerator, ...]:
    return tuple(TendonGenerator(**g) for g in SHOULDER_GENERATORS)
