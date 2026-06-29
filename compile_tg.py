from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphology.compile import compile_to_mjcf
from morphology.tg_parser import load_morphology


def compile_tg(tg_path: Path, output_path: Path, *, color_structural_tendons: bool = False) -> Path:
    morph = load_morphology(tg_path)
    if not Path(morph.world.meshdir).is_absolute():
        morph.world.meshdir = str((tg_path.parent.parent / morph.world.meshdir).resolve())
    xml = compile_to_mjcf(morph, color_structural_tendons=color_structural_tendons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a .tg morphology to MuJoCo XML.")
    parser.add_argument("--tg", type=Path, default=ROOT / "design" / "baseline_shoulder.tg")
    parser.add_argument("--out", type=Path, default=ROOT / "scene.xml")
    parser.add_argument("--color", action="store_true", help="Color structural tendons with plot colors")
    args = parser.parse_args()
    print(compile_tg(args.tg, args.out, color_structural_tendons=args.color))


if __name__ == "__main__":
    main()
