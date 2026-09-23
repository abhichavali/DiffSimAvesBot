"""Load the active morphology design for simulation runtime.

Runtime code (aero, wing geometry, pitch calibration) reads physical knobs from
the same `.tg` file the compiler uses — not from legacy JSON configs.

Override the design file with ``MORPHOLOGY_PATH`` (absolute or repo-relative).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from morphology.model import Morphology
from morphology.tg_parser import load_morphology

# Repo root: src/runtime/morphology_runtime.py → ../../
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The committed designs (see designs/README.md) and the tool defaults that point at them.
DESIGNS_DIR = Path(ROOT) / "designs"
DEFAULT_HUMMER_TG = DESIGNS_DIR / "hummer" / "hover" / "shoulder.tg"  # runtime, viewers, GUI, renders
HOVER_BASE_TG = DESIGNS_DIR / "hummer" / "hover-base" / "shoulder.tg"  # EA + run_gui
LEGACY_BASELINE_TG = DESIGNS_DIR / "hummer" / "legacy-baseline" / "shoulder.tg"  # optimizer, compile_tg, run_fixed_frequency
EAGLE_TG = DESIGNS_DIR / "eagle" / "shoulder.tg"


def resolve_morphology_path() -> str:
    return os.getenv("MORPHOLOGY_PATH", str(DEFAULT_HUMMER_TG))


@lru_cache(maxsize=1)
def load_design_morphology() -> Morphology:
    """Parse and validate the morphology design (cached for the process lifetime)."""
    return load_morphology(resolve_morphology_path())
