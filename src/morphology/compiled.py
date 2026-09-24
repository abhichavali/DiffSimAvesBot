"""The compiler's output: every derived pose and length, ready for the MJCF emitter.

compile.compile_morphology fills these from a Morphology; emit_mjcf turns them
into XML. Positions are metres, quaternions w-first, all in the parent frame the
emitter places them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from morphology.geometry import StrutGeometry
from morphology.model import Morphology
from morphology.tendon_net import StructuralEdge


@dataclass
class CompiledDisk:
    """One crank disk: pillar base (world), orientation, and the tendon pin (disk-local)."""

    pillar_pos: np.ndarray
    euler: List[float]  # disk body euler (tilt, pitch, 0)
    rim_pos: np.ndarray
    rim_rgba: str


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
    proximal_geom: StrutGeometry
    distal_geom: StrutGeometry
    structural_edges: List[StructuralEdge]
    tendon_lengths: List[float]  # posed length of each structural edge
    #: Manus pose in the distal strut's frame, chosen so the hand keeps the
    #: position and heading the hinge wrist gave it.
    manus_pos: np.ndarray
    manus_quat: np.ndarray


@dataclass
class CompiledShoulder:
    side: str  # "L" or "R"
    mobile_pos: np.ndarray
    mobile_quat: np.ndarray
    fixed_pos: np.ndarray
    fixed_quat: np.ndarray
    mobile_geom: StrutGeometry
    fixed_geom: StrutGeometry
    structural_edges: List[StructuralEdge]
    tendon_lengths: List[float]  # posed length of each structural edge
    disks: Dict[str, CompiledDisk]  # keyed "L_t1", "L_t4", ...
    wrist: CompiledWrist | None = None

    @property
    def mobile_body(self) -> str:
        return f"{self.side}_part_1"

    @property
    def fixed_body(self) -> str:
        return f"{self.side}_part_2"


@dataclass
class CompiledMorphology:
    morph: Morphology
    shoulders: Tuple[CompiledShoulder, CompiledShoulder]
