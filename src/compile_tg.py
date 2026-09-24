"""Compile a .tg design to a MuJoCo scene file.

    uv run python src/compile_tg.py --tg designs/hummer/hover/shoulder.tg --out scene.xml

compile_tg(tg, out) is what every tool calls; it writes the requested file and
nothing else (morphology.compile.compile_file returns the XML without writing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from morphology.compile import compile_file
from runtime.morphology_runtime import LEGACY_BASELINE_TG

ROOT = Path(__file__).resolve().parent.parent


def compile_tg(tg_path: Path, output_path: Path, *, color_structural_tendons: bool = False) -> Path:
    xml = compile_file(tg_path, color_structural_tendons=color_structural_tendons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a .tg morphology to MuJoCo XML.")
    parser.add_argument("--tg", type=Path, default=LEGACY_BASELINE_TG)
    parser.add_argument("--out", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--color", action="store_true", help="Color structural tendons with plot colors")
    args = parser.parse_args()
    print(compile_tg(args.tg, args.out, color_structural_tendons=args.color))


if __name__ == "__main__":
    main()
