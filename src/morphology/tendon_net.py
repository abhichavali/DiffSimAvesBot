"""The structural tendon net: σ-expansion, materials, anchor node names, posed lengths.

A .tg declares six generators, each joining a mobile-strut anchor M[station face]
to a fixed-strut anchor F[station face]. σ (station n -> 5-n, face A <-> B) maps
each generator onto a partner, giving the 12 structural tendons of a shoulder
(and of the eagle's wrist, which reuses the same six).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from morphology.geometry import StrutGeometry, world_node
from morphology.model import Face, TendonMaterial, TendonNet


@dataclass(frozen=True)
class SemanticAnchor:
    side: str  # "M" or "F"
    station: int
    face: Face

    def edge_id(self) -> str:
        return f"{self.side}{self.station}{self.face.value}"


@dataclass
class StructuralEdge:
    """One structural tendon after σ expansion, from the mobile to the fixed strut."""

    mobile: SemanticAnchor
    fixed: SemanticAnchor
    material: TendonMaterial  # unset fields fall back to the net defaults


def _sigma(anchor: SemanticAnchor) -> SemanticAnchor:
    return SemanticAnchor(anchor.side, 5 - anchor.station, Face.B if anchor.face == Face.A else Face.A)


def canonical_edge_id(a: SemanticAnchor, b: SemanticAnchor) -> str:
    left, right = sorted([a.edge_id(), b.edge_id()])
    return f"{left}-{right}"


def expand_sigma_tendons(net: TendonNet) -> List[StructuralEdge]:
    """Expand the 6 generators to 12 structural edges via σ."""
    edges: List[StructuralEdge] = []
    seen: set[str] = set()
    for gen in net.generators:
        m = SemanticAnchor("M", gen.mobile_station, gen.mobile_face)
        f = SemanticAnchor("F", gen.fixed_station, gen.fixed_face)
        for mobile, fixed in ((m, f), (_sigma(m), _sigma(f))):
            eid = canonical_edge_id(mobile, fixed)
            if eid in seen:
                continue
            seen.add(eid)
            edges.append(StructuralEdge(mobile, fixed, net.edge_overrides.get(eid, gen.material)))
    if len(edges) != 12:
        raise ValueError(f"σ expansion produced {len(edges)} edges, expected 12")
    return edges


def resolve_material_props(net: TendonNet, edge: StructuralEdge) -> Tuple[float, float, float, str]:
    """(stiffness, damping, pretension, color) of an edge, net defaults filling the gaps."""
    mat, d = edge.material, net.defaults
    return (
        mat.stiffness if mat.stiffness is not None else d.stiffness,
        mat.damping if mat.damping is not None else d.damping,
        mat.pretension if mat.pretension is not None else d.pretension,
        mat.color if mat.color is not None else d.color,
    )


def anchor_node_name(anchor: SemanticAnchor) -> str:
    """MJCF node of an anchor: face A at station n is tn; face B is tn_2 on the
    mobile strut but t{5-n}_2 on the fixed one (its B face runs the other way)."""
    if anchor.face == Face.A:
        return f"t{anchor.station}"
    return f"t{anchor.station}_2" if anchor.side == "M" else f"t{5 - anchor.station}_2"


def derive_tendon_length(
    edge: StructuralEdge,
    mobile_geom: StrutGeometry,
    fixed_geom: StrutGeometry,
    mobile_pos: np.ndarray,
    mobile_quat: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_quat: np.ndarray,
) -> float:
    """Anchor-to-anchor distance in the rest pose."""
    w1 = world_node(mobile_geom.nodes[anchor_node_name(edge.mobile)], mobile_pos, mobile_quat)
    w2 = world_node(fixed_geom.nodes[anchor_node_name(edge.fixed)], fixed_pos, fixed_quat)
    return float(np.linalg.norm(w2 - w1))
