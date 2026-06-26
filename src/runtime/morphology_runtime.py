"""Load the active morphology design for simulation runtime.

Runtime code (aero, wing geometry, pitch calibration) reads physical knobs from
the same `.tg` file the compiler uses — not from legacy JSON configs.

Override the design file with ``MORPHOLOGY_PATH`` (absolute or repo-relative).
"""

from __future__ import annotations

import os
from functools import lru_cache

from morphology.model import Morphology
from morphology.tg_parser import load_morphology

# Repo root: src/runtime/morphology_runtime.py → ../../
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOULDER_TG = os.path.join(ROOT, "design", "shoulder.tg")


def resolve_morphology_path() -> str:
    return os.getenv("MORPHOLOGY_PATH", SHOULDER_TG)


@lru_cache(maxsize=1)
def load_design_morphology() -> Morphology:
    """Parse and validate the morphology design (cached for the process lifetime)."""
    return load_morphology(resolve_morphology_path())
