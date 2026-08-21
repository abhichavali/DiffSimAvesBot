"""Compiled morphology intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from morphology.model import Morphology
from morphology.symmetry import StructuralEdge


@dataclass
class CompiledWrist:
    """A scaled-down tensegrity joint standing in for the wrist hinge.

    ``proximal_*`` is expressed in the ULNA's frame (it is a rigid child of the
    forearm). ``distal_*`` is in WORLD, because it carries a free joint and MuJoCo
    only allows those at the top level — so the manus hangs off a worldbody child
    and is joined to the forearm by tendons alone.
    """

    proximal_pos: np.ndarray
    proximal_quat: np.ndarray
    distal_pos: np.ndarray
    distal_quat: np.ndarray
    proximal_geom: Any
    distal_geom: Any
    structural_edges: List[StructuralEdge]
    tendon_lengths: Dict[str, float]
    #: Manus pose in the distal strut's frame, chosen so the hand keeps the
    #: position and heading the hinge wrist gave it.
    manus_pos: np.ndarray
    manus_quat: np.ndarray


@dataclass
class CompiledShoulder:
    side: str
    mobile_body: str
    fixed_body: str
    mobile_pos: np.ndarray
    mobile_quat: np.ndarray
    fixed_pos: np.ndarray
    fixed_quat: np.ndarray
    mobile_geom: Any
    fixed_geom: Any
    motor_nodes: Dict[str, np.ndarray]
    structural_edges: List[StructuralEdge]
    tendon_lengths: Dict[str, float]
    disks: Dict[str, Dict[str, Any]]
    wrist: "CompiledWrist | None" = None


@dataclass
class CompiledMorphology:
    morph: Morphology
    shoulders: Tuple[CompiledShoulder, CompiledShoulder]
    wing_cop_local: np.ndarray
    aero_blade_sites: List[Dict[str, Any]] = field(default_factory=list)
