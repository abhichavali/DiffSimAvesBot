"""Strut geometry and rest-pose placement for the compiler.

An arc strut is four stations on a circular arc, each with an A and a B face
node a strut-width apart. The mobile strut's arc lies in its body x-z plane, the
fixed strut's in y-z. This module builds those nodes (and their site frames),
places the pair, finds the motor anchors and crank-disk positions, and holds the
w-first quaternion helpers and the y-mirror that turns the left shoulder into
the right one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from morphology.model import ArcPlane, DiskSpec, Face, Morphology, RodSpec


@dataclass
class StrutGeometry:
    nodes: Dict[str, np.ndarray]
    site_quats: Dict[str, np.ndarray]
    arc_rod_pairs: List[Tuple[str, str]]
    rung_rod_pairs: List[Tuple[str, str]]
    arc_rod: Tuple[float, float]  # (radius, mass), scaled
    rung_rod: Tuple[float, float]
    # Rigid rods bridging a motor foot edge to an inset motor anchor (radius<1).
    motor_rod_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def midpoint(self) -> np.ndarray:
        """Where the wing (or the eagle's humerus) mounts."""
        return 0.5 * (self.nodes["c2"] + self.nodes["c3"])


def _unit_direction(angle_rad: float, plane: ArcPlane) -> np.ndarray:
    if plane == ArcPlane.XZ:
        return np.array([math.cos(angle_rad), 0.0, math.sin(angle_rad)], dtype=float)
    return np.array([0.0, math.cos(angle_rad), -math.sin(angle_rad)], dtype=float)


def _face_offset(face: Face, width: float, plane: ArcPlane) -> np.ndarray:
    half = width / 2.0
    if plane == ArcPlane.XZ:
        return np.array([0.0, -half if face == Face.A else half, 0.0], dtype=float)
    return np.array([-half if face == Face.A else half, 0.0, 0.0], dtype=float)


def _tangent_along_arc(angle_rad: float, plane: ArcPlane) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if plane == ArcPlane.XZ:
        return np.array([-s, 0.0, c], dtype=float)
    return np.array([0.0, -s, c], dtype=float)


def _site_quat_from_axes(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    t = tangent / (np.linalg.norm(tangent) + 1e-12)
    n = normal / (np.linalg.norm(normal) + 1e-12)
    b = np.cross(n, t)
    b = b / (np.linalg.norm(b) + 1e-12)
    t = np.cross(b, n)
    rot = np.column_stack([t, b, n])
    trace = float(np.trace(rot))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def derive_arc_center(
    radius: float,
    stations_deg: Tuple[float, ...],
    plane: ArcPlane,
    mobile_center_x: float | None = None,
) -> np.ndarray:
    """Derive arc center in strut body frame from radius + station angles only.

    Mobile (XZ): apex–foot bias from the station-1→2 opening and supplementary
    foot symmetry.  Fixed (YZ, needs ``mobile_center_x``): orthogonal complement;
    ``cz`` scales with the derived mobile ``cx`` (arrange coupling).
    """
    t1 = math.radians(stations_deg[0])
    t2 = math.radians(stations_deg[1])
    t3 = math.radians(stations_deg[2])
    cx_r = math.cos(t1) * math.cos((t2 - t1) / 2.0) + math.sin(t1) / 3.0
    if plane == ArcPlane.XZ:
        cos_t1 = max(abs(math.cos(t1)), 1e-9)
        sin_t2 = max(abs(math.sin(t2)), 1e-9)
        cz_r = -math.sin(t1) * math.sin(t2) / (2.0 * cos_t1) * (1.0 - math.sin(t1) / (2.0 * sin_t2))
        return np.array([radius * cx_r, 0.0, radius * cz_r], dtype=float)
    cy_r = -math.sin(t1) / 4.0
    cz_r = mobile_center_x * (math.sin(t2) + math.sin(t3)) / (8.0 * radius * cx_r)
    return np.array([0.0, radius * cy_r, radius * cz_r], dtype=float)


def scaled_rod(rod: RodSpec, scale: float) -> Tuple[float, float]:
    return rod.radius * scale, rod.mass * (scale**3)


def build_strut(
    plane: ArcPlane,
    radius: float,
    width: float,
    stations_deg: Tuple[float, ...],
    center: np.ndarray,
    scale: float,
    arc_rod: RodSpec,
    rung_rod: RodSpec,
    width_axis_offset: float = 0.0,
) -> StrutGeometry:
    """One arc strut: nodes and site frames, scaled by `scale` about the node centroid.

    Nodes: cN on the arc centreline, tN on face A. Face B is tN_2 on the mobile
    (XZ) strut but t{5-N}_2 on the fixed (YZ) one, so the σ map lines up.
    """
    nodes: Dict[str, np.ndarray] = {}
    site_quats: Dict[str, np.ndarray] = {}
    for idx, angle_deg in enumerate(stations_deg, start=1):
        angle = math.radians(angle_deg)
        cl = center + radius * _unit_direction(angle, plane)
        if plane == ArcPlane.YZ:
            cl = cl + np.array([width_axis_offset, 0.0, 0.0])
        nodes[f"c{idx}"] = cl.copy()
        b_name = f"t{idx}_2" if plane == ArcPlane.XZ else f"t{5 - idx}_2"
        for name, face in ((f"t{idx}", Face.A), (b_name, Face.B)):
            nodes[name] = cl + _face_offset(face, width, plane)
            site_quats[name] = _site_quat_from_axes(
                _tangent_along_arc(angle, plane), _face_offset(face, 1.0, plane)
            )

    if scale != 1.0:
        centroid = np.stack(list(nodes.values())).mean(axis=0)
        nodes = {k: centroid + scale * (v - centroid) for k, v in nodes.items()}

    if plane == ArcPlane.XZ:
        rung_pairs = [(f"t{i}", f"t{i}_2") for i in range(1, 5)]
    else:
        rung_pairs = [(f"t{i}", f"t{5 - i}_2") for i in range(1, 5)]
    return StrutGeometry(
        nodes=nodes,
        site_quats=site_quats,
        arc_rod_pairs=[(f"c{i}", f"c{i + 1}") for i in range(1, 4)],
        rung_rod_pairs=rung_pairs,
        arc_rod=scaled_rod(arc_rod, scale),
        rung_rod=scaled_rod(rung_rod, scale),
    )


def strut_pair(
    radius: float,
    width: float,
    stations_deg: Tuple[float, ...],
    inplane_offset: float,
    scale: float,
    arc_rod: RodSpec,
    rung_rod: RodSpec,
) -> Tuple[StrutGeometry, StrutGeometry]:
    """(mobile, fixed) struts of one tensegrity joint, each in its own body frame.

    The fixed strut's centreline sits `inplane_offset` off the mobile arc centre's x.
    """
    mobile_center = derive_arc_center(radius, stations_deg, ArcPlane.XZ)
    fixed_center = derive_arc_center(radius, stations_deg, ArcPlane.YZ, mobile_center_x=mobile_center[0])
    mobile = build_strut(ArcPlane.XZ, radius, width, stations_deg, mobile_center, scale, arc_rod, rung_rod)
    fixed = build_strut(ArcPlane.YZ, radius, width, stations_deg, fixed_center, scale, arc_rod, rung_rod,
                        width_axis_offset=mobile_center[0] + inplane_offset)
    return mobile, fixed


def strut_quat() -> np.ndarray:
    """Body orientation shared by both struts of a pair."""
    return np.array([0.707107, -0.707107, 0.0, 0.0], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two w-first quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])


def compose(pos_a: np.ndarray, quat_a: np.ndarray,
            pos_b: np.ndarray, quat_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Frame B expressed in A's parent: A then B."""
    return pos_a + quat_rotate(quat_a, pos_b), quat_mul(quat_a, quat_b)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rot @ np.asarray(v, dtype=float)


def world_node(local: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray) -> np.ndarray:
    return quat_rotate(body_quat, local) + body_pos


def mirror_y(v) -> np.ndarray:
    """Reflect a point across the y = 0 plane (left shoulder -> right)."""
    return np.array([v[0], -v[1], v[2]], dtype=float)


def mirror_quat(q) -> np.ndarray:
    """The y-reflection of a w-first orientation."""
    return np.array([q[0], -q[1], q[2], -q[3]], dtype=float)


def compute_body_poses(morph: Morphology) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left-shoulder (mobile_pos, mobile_quat, fixed_pos, fixed_quat) from placement + arrange."""
    z = morph.placement.mount_height
    y_mobile = morph.placement.lateral_offset
    y_fixed = y_mobile + morph.arrange.gap

    mobile_pos = np.array([morph.arrange.mobile_dx, y_mobile, z], dtype=float)
    fixed_pos = np.array([0.0, y_fixed, z], dtype=float)

    if morph.arrange.angle_deg != 0.0:
        angle = math.radians(morph.arrange.angle_deg)
        rel = mobile_pos - fixed_pos
        c, s = math.cos(angle), math.sin(angle)
        rotated = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        mobile_pos = fixed_pos + rotated

    return mobile_pos, strut_quat(), fixed_pos, strut_quat()


def motor_anchor_node(
    foot_station: int,
    attach_rad: float,
    geom: StrutGeometry,
    attach_radius: float = 1.0,
) -> Tuple[str, np.ndarray, np.ndarray | None]:
    """Resolve the motor-tendon anchor as a polar point on the mobile strut.

    ``attach_rad`` is the angular coordinate (slides along the foot edge between
    adjacent stations). ``attach_radius`` is a 0..1 fraction from the strut
    midpoint (0) out to the foot edge (1). At 1.0 the anchor lies on the edge
    (legacy). Below 1.0 the anchor moves toward the midpoint and the second
    return value is the edge point that a rigid bridging rod connects to.
    """
    if foot_station == 1:
        a, b = geom.nodes["t1"], geom.nodes["t2"]
    else:
        a, b = geom.nodes["t4"], geom.nodes["t3"]
    t = max(0.0, min(1.0, attach_rad / (math.pi / 2.0)))
    edge_point = (1.0 - t) * a + t * b
    name = f"motor_t{foot_station}"

    r = max(0.0, min(1.0, attach_radius))
    if r >= 1.0:
        return name, edge_point, None

    # Pole = centroid of the foot corners. These all lie on the same strut
    # face plane as the edge point, so the bridging rod stays in that plane
    # (horizontal, like the strut rods) instead of sloping toward the
    # centerline c-nodes.
    pole = 0.25 * (
        geom.nodes["t1"] + geom.nodes["t2"] + geom.nodes["t3"] + geom.nodes["t4"]
    )
    anchor = pole + r * (edge_point - pole)
    return name, anchor, edge_point


def derive_disk_pillar_pos(
    foot_world: np.ndarray,
    sibling_foot_world: np.ndarray,
    disk: DiskSpec,
) -> np.ndarray:
    """Place crank pillar base from foot world xy, optionally outset from sibling foot."""
    xy = np.array([foot_world[0], foot_world[1]], dtype=float)
    if disk.foot_outset != 0.0:
        # Signed offset along the line joining the two feet: positive spreads the
        # disk away from its sibling (pair moves apart), negative pulls it toward
        # the sibling (pair moves together). Symmetric — both disks shift equally.
        away = xy - np.array([sibling_foot_world[0], sibling_foot_world[1]], dtype=float)
        norm = float(np.linalg.norm(away))
        if norm > 1e-9:
            xy = xy + (disk.foot_outset / norm) * away
    return np.array([xy[0] + disk.disk_dx, xy[1] + disk.disk_dy, 0.0], dtype=float)


def derive_disk_rim_local(disk: DiskSpec) -> np.ndarray:
    # Tendon pin radius = crank_throw (a rigid arm past the rim) if set, else the
    # disk radius (pin on the rim). Decoupling throw from the visible disk size is
    # the gear-ratio lever for flap swing without growing the disk.
    throw = disk.radius if disk.crank_throw is None else disk.crank_throw
    # Parked pin phase sets where the no-drive rest sits within the driven arc
    # (the lever to re-center a one-sided stroke). Default ~atan2(0.2,1) keeps the
    # legacy [throw, 0.2*throw, 0] placement.
    phase = disk.crank_phase_rad
    return throw * np.array([math.cos(phase), math.sin(phase), 0.0], dtype=float)
