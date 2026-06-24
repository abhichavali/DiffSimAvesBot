"""Load the active morphology design for simulation runtime.

Runtime code (aero, wing geometry, pitch calibration) reads physical knobs from
the same `.tg` file the compiler uses — not from legacy JSON configs.

Override the design file with ``MORPHOLOGY_PATH`` (absolute or repo-relative).
"""

from __future__ import annotations

import os
from functools import lru_cache

from morphology.instances.shoulder import SHOULDER_TG, shoulder_morphology
from morphology.model import Morphology


def resolve_morphology_path() -> str:
    return os.getenv("MORPHOLOGY_PATH", SHOULDER_TG)


@lru_cache(maxsize=1)
def load_design_morphology() -> Morphology:
    """Parse and validate the morphology design (cached for the process lifetime)."""
    return shoulder_morphology(resolve_morphology_path())
