"""σ (2-fold) and bilateral mirror symmetry over tendon / morphology graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from morphology.model import Face, TendonGenerator, TendonMaterial, TendonNet


@dataclass(frozen=True)
class SemanticAnchor:
    side: str  # "M" or "F"
    station: int
    face: Face

    def edge_id(self) -> str:
        return f"{self.side}{self.station}{self.face.value}"


@dataclass
class StructuralEdge:
    """One structural tendon after σ expansion."""

    anchor_a: SemanticAnchor
    anchor_b: SemanticAnchor
    generator_id: str  # canonical id of declaring generator
    material: TendonMaterial


def sigma_map_station(station: int) -> int:
    return 5 - station


def sigma_map_face(face: Face) -> Face:
    return Face.B if face == Face.A else Face.A


def sigma_map_anchor(anchor: SemanticAnchor) -> SemanticAnchor:
    return SemanticAnchor(
        side=anchor.side,
        station=sigma_map_station(anchor.station),
        face=sigma_map_face(anchor.face),
    )


def generator_to_anchors(gen: TendonGenerator) -> Tuple[SemanticAnchor, SemanticAnchor]:
    return (
        SemanticAnchor("M", gen.mobile_station, gen.mobile_face),
        SemanticAnchor("F", gen.fixed_station, gen.fixed_face),
    )


def canonical_edge_id(a: SemanticAnchor, b: SemanticAnchor) -> str:
    left, right = sorted([a.edge_id(), b.edge_id()])
    return f"{left}-{right}"


def expand_sigma_tendons(net: TendonNet) -> List[StructuralEdge]:
    """Expand 6 generators to 12 structural edges via σ."""
    edges: List[StructuralEdge] = []
    seen: set[str] = set()

    for gen in net.generators:
        a, b = generator_to_anchors(gen)
        for ga, gb in ((a, b), (sigma_map_anchor(a), sigma_map_anchor(b))):
            eid = canonical_edge_id(ga, gb)
            if eid in seen:
                continue
            seen.add(eid)
            mat = _resolve_material(net, gen, eid)
            edges.append(
                StructuralEdge(
                    anchor_a=ga,
                    anchor_b=gb,
                    generator_id=f"M{gen.mobile_station}{gen.mobile_face.value}-F{gen.fixed_station}{gen.fixed_face.value}",
                    material=mat,
                )
            )
    if len(edges) != 12:
        raise ValueError(f"σ expansion produced {len(edges)} edges, expected 12")
    return edges


def _resolve_material(net: TendonNet, gen: TendonGenerator, edge_id: str) -> TendonMaterial:
    if edge_id in net.edge_overrides:
        return net.edge_overrides[edge_id]
    m = gen.material
    if any(v is not None for v in (m.stiffness, m.damping, m.pretension, m.color)):
        return TendonMaterial(
            stiffness=m.stiffness if m.stiffness is not None else net.defaults.stiffness,
            damping=m.damping if m.damping is not None else net.defaults.damping,
            pretension=m.pretension if m.pretension is not None else net.defaults.pretension,
            color=m.color if m.color is not None else net.defaults.color,
        )
    return TendonMaterial(
        stiffness=net.defaults.stiffness,
        damping=net.defaults.damping,
        pretension=net.defaults.pretension,
        color=net.defaults.color,
    )


def resolve_material_props(net: TendonNet, edge: StructuralEdge) -> Tuple[float, float, float, str]:
    mat = edge.material
    return (
        mat.stiffness if mat.stiffness is not None else net.defaults.stiffness,
        mat.damping if mat.damping is not None else net.defaults.damping,
        mat.pretension if mat.pretension is not None else net.defaults.pretension,
        mat.color if mat.color is not None else net.defaults.color,
    )


def mirror_y(value: float, axis: str = "y") -> float:
    if axis == "y":
        return -value
    raise ValueError(f"Unsupported mirror axis {axis!r}")


def mirror_vec3(vec: Tuple[float, float, float], axis: str = "y") -> Tuple[float, float, float]:
    x, y, z = vec
    if axis == "y":
        return (x, -y, z)
    raise ValueError(f"Unsupported mirror axis {axis!r}")


def mirror_quat_wxyz(quat: Tuple[float, float, float, float], axis: str = "y") -> Tuple[float, float, float, float]:
    w, x, y, z = quat
    if axis == "y":
        return (w, -x, y, -z)
    raise ValueError(f"Unsupported mirror axis {axis!r}")
