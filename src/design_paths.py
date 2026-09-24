"""Where the designs live, and the design each tool opens when not given ``--tg``."""

from pathlib import Path

# The committed designs (see designs/README.md) and the tool defaults that point at them.
DESIGNS_DIR = Path(__file__).resolve().parents[1] / "designs"
DEFAULT_HUMMER_TG = DESIGNS_DIR / "hummer" / "hover" / "shoulder.tg"  # optimizer, viewers, GUI, renders, run_fixed_frequency
HOVER_BASE_TG = DESIGNS_DIR / "hummer" / "hover-base" / "shoulder.tg"  # EA + run_gui
LEGACY_BASELINE_TG = DESIGNS_DIR / "hummer" / "legacy-baseline" / "shoulder.tg"  # compile_tg
FREE_HOVER_TG = DESIGNS_DIR / "hummer" / "free-hover" / "shoulder.tg"  # src/hover (free-flight control)
EAGLE_TG = DESIGNS_DIR / "eagle" / "shoulder.tg"
