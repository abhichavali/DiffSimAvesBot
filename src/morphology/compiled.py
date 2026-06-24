"""Compiled morphology intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from morphology.model import Morphology
from morphology.symmetry import StructuralEdge


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


@dataclass
class CompiledMorphology:
    morph: Morphology
    shoulders: Tuple[CompiledShoulder, CompiledShoulder]
    wing_cop_local: np.ndarray
    aero_blade_sites: List[Dict[str, Any]] = field(default_factory=list)
