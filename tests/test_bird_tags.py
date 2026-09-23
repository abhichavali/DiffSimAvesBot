"""Every .tg is tagged with its bird, and the tools refuse the wrong one.

Run: uv run --with pytest pytest -q tests/test_bird_tags.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from morphology.model import BIRDS, PLATE_BIRDS, require_bird  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.control_signal import load_control, load_control_for_morphology  # noqa: E402

SHOULDER = ROOT / "design" / "shoulder.tg"
EAGLE = ROOT / "design" / "eagle.tg"


def tracked_tg() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.tg"], cwd=ROOT, capture_output=True, check=True)
    return [ROOT / f for f in out.stdout.decode().split("\0") if f]


def is_morphology(path: Path) -> bool:
    return re.search(r"(?m)^world:", path.read_text()) is not None


def test_every_tracked_tg_is_tagged_and_loads():
    files = tracked_tg()
    assert len(files) > 100  # the optimizer artefacts count too
    for path in files:
        assert not path.is_symlink(), path
        text = path.read_text()
        m = re.search(r"(?m)^bird\s*=\s*(\w+)", text)
        assert m and m.group(1) in BIRDS, f"{path} has no bird tag"
        if is_morphology(path):
            assert load_morphology(path).bird == m.group(1)
        elif re.search(r"(?m)^harmonics\s*=", text):  # wavelet controls have their own reader
            assert load_control(path).bird == m.group(1)


def test_the_main_designs_wear_the_expected_birds():
    assert load_morphology(SHOULDER).bird == "hummer"
    assert load_morphology(EAGLE).bird == "eagle"
    assert load_control_for_morphology(SHOULDER).bird == "hummer"
    assert load_control_for_morphology(EAGLE).bird == "eagle"


def test_untagged_or_mistagged_files_are_refused(tmp_path):
    untagged = tmp_path / "untagged.tg"
    untagged.write_text(re.sub(r"(?m)^bird\s*=.*\n", "", SHOULDER.read_text()))
    with pytest.raises(ValueError, match="bird"):
        load_morphology(untagged)

    mistagged = tmp_path / "eagle_as_hummer.tg"
    mistagged.write_text(EAGLE.read_text().replace("bird = eagle", "bird = hummer", 1))
    with pytest.raises(ValueError, match="bones"):
        load_morphology(mistagged)

    control = tmp_path / "control.tg"
    control.write_text(re.sub(r"(?m)^bird\s*=.*\n", "", (ROOT / "design" / "control.tg").read_text()))
    with pytest.raises(ValueError, match="bird"):
        load_control(control)


def test_a_control_of_another_bird_cannot_be_paired(tmp_path):
    (tmp_path / "shoulder.tg").write_text(SHOULDER.read_text())
    (tmp_path / "control.tg").write_text(
        (ROOT / "design" / "control.tg").read_text().replace("bird = hummer", "bird = eagle", 1)
    )
    with pytest.raises(ValueError, match="tagged bird = hummer .* tagged bird = eagle"):
        load_control_for_morphology(tmp_path / "shoulder.tg")


def test_require_bird_keeps_the_tools_apart():
    hummer, eagle = load_morphology(SHOULDER), load_morphology(EAGLE)
    require_bird(hummer, *PLATE_BIRDS, tool="plate tool")
    require_bird(eagle, "eagle", tool="eagle tool")
    with pytest.raises(ValueError, match="plate tool drives hummer / swift"):
        require_bird(eagle, *PLATE_BIRDS, tool="plate tool")
    with pytest.raises(ValueError, match="eagle tool drives eagle"):
        require_bird(hummer, "eagle", tool="eagle tool")
