"""Tendon anchor resolution and length derivation."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from morphology.geometry import StrutGeometry, anchor_to_node_name, world_node
from morphology.model import Face
from morphology.symmetry import SemanticAnchor, StructuralEdge


def anchor_node_name(anchor: SemanticAnchor) -> str:
    return anchor_to_node_name(anchor.side, anchor.station, anchor.face)


def edge_topology_key(edge: StructuralEdge) -> Tuple[str, str, str, str]:
    """Return sorted (mobile_body, mobile_node, fixed_body, fixed_node) for parity tests."""
    ma = edge.anchor_a if edge.anchor_a.side == "M" else edge.anchor_b
    fa = edge.anchor_b if edge.anchor_a.side == "M" else edge.anchor_a
    mb = "L_part_1"
    fb = "L_part_2"
    return (mb, anchor_node_name(ma), fb, anchor_node_name(fa))


def derive_tendon_length(
    edge: StructuralEdge,
    mobile_geom: StrutGeometry,
    fixed_geom: StrutGeometry,
    mobile_pos: np.ndarray,
    mobile_quat: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_quat: np.ndarray,
) -> float:
    ma = edge.anchor_a if edge.anchor_a.side == "M" else edge.anchor_b
    fa = edge.anchor_b if edge.anchor_a.side == "M" else edge.anchor_a
    m_node = mobile_geom.nodes[anchor_node_name(ma)]
    f_node = fixed_geom.nodes[anchor_node_name(fa)]
    w1 = world_node(m_node, mobile_pos, mobile_quat)
    w2 = world_node(f_node, fixed_pos, fixed_quat)
    return float(np.linalg.norm(w2 - w1))


def legacy_edge_name(prefix: str, mobile_node: str, fixed_node: str) -> str:
    return f"{prefix}_{mobile_node}_to_{fixed_node}"


def build_legacy_topology_map() -> Dict[Tuple[str, str, str, str], str]:
    """Map topology keys to legacy tendon names (left shoulder)."""
    pairs = [
        ("t1", "t1_2"),
        ("t1", "t2_2"),
        ("t1_2", "t3_2"),
        ("t1_2", "t4_2"),
        ("t2", "t1_2"),
        ("t2_2", "t4_2"),
        ("t3_2", "t1"),
        ("t4_2", "t1"),
        ("t3", "t4"),
        ("t4", "t4"),
        ("t4_2", "t2"),
        ("t4", "t3"),
    ]
    return {
        ("L_part_1", m, "L_part_2", f): f"L_{m}_to_{f}"
        for m, f in pairs
    }


# Six σ-basis generators (declared topology — S-class)
SHOULDER_GENERATORS = (
    # M[1A] — F[4B]
    dict(mobile_station=1, mobile_face=Face.A, fixed_station=4, fixed_face=Face.B),
    # M[1A] — F[3B]
    dict(mobile_station=1, mobile_face=Face.A, fixed_station=3, fixed_face=Face.B),
    # M[1B] — F[2B]
    dict(mobile_station=1, mobile_face=Face.B, fixed_station=2, fixed_face=Face.B),
    # M[1B] — F[1B]
    dict(mobile_station=1, mobile_face=Face.B, fixed_station=1, fixed_face=Face.B),
    # M[2A] — F[4B]
    dict(mobile_station=2, mobile_face=Face.A, fixed_station=4, fixed_face=Face.B),
    # M[2B] — F[1B]
    dict(mobile_station=2, mobile_face=Face.B, fixed_station=1, fixed_face=Face.B),
)
