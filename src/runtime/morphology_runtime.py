"""Where the designs live, and the design each tool opens when not given ``--tg``."""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: src/runtime/morphology_runtime.py → ../../
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The committed designs (see designs/README.md) and the tool defaults that point at them.
DESIGNS_DIR = Path(ROOT) / "designs"
DEFAULT_HUMMER_TG = DESIGNS_DIR / "hummer" / "hover" / "shoulder.tg"  # runtime, viewers, GUI, renders
HOVER_BASE_TG = DESIGNS_DIR / "hummer" / "hover-base" / "shoulder.tg"  # EA + run_gui
LEGACY_BASELINE_TG = DESIGNS_DIR / "hummer" / "legacy-baseline" / "shoulder.tg"  # optimizer, compile_tg, run_fixed_frequency
EAGLE_TG = DESIGNS_DIR / "eagle" / "shoulder.tg"
