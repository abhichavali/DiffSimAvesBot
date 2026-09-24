"""Per-tendon material overrides: color / pretension / stiffness.

Run: uv run pytest -q tests/test_tendon_overrides.py
Covers the new logic units without a full mesh-dependent compile.
"""

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from morphology.mjcf_parts import color_to_rgba
from morphology.compile import compile_morphology
from morphology.emit_mjcf import emit_mjcf
from morphology.model import TendonDefaults, TendonMaterial, TendonNet
from morphology.tendon_net import resolve_material_props
from morphology.tg_parser import _parse_brace_dict, _parse_material_dict, load_morphology


def testcolor_to_rgba():
    assert color_to_rgba("1 0 0 1") == "1 0 0 1"        # rgba passthrough
    assert color_to_rgba("0.5 0.5 0.5") == "0.5 0.5 0.5 1"  # alpha padded
    assert color_to_rgba("00ff00") == "0.0 1.0 0.0 1"   # 6-digit hex (no '#')


def test_parse_material_dict():
    mat = _parse_material_dict(_parse_brace_dict("{ stiffness: 333, pretension: 0.5, color: 1 0 0 1 }"))
    assert mat.stiffness == 333.0 and mat.pretension == 0.5 and mat.color == "1 0 0 1"
    bare = _parse_material_dict(_parse_brace_dict("{ color: 00ff00 }"))
    assert bare.stiffness is None and bare.color == "00ff00"  # only color overridden


def test_resolve_falls_back_to_defaults():
    net = TendonNet(defaults=TendonDefaults(stiffness=200, damping=0.5, pretension=0.8,
                                            color="0.8 0.2 0.2 1"), generators=())

    class _E:  # minimal stand-in for StructuralEdge
        material = TendonMaterial(stiffness=333, color="1 0 0 1")  # pretension unset -> default

    stiff, damp, pret, color = resolve_material_props(net, _E())
    assert (stiff, damp, pret, color) == (333, 0.5, 0.8, "1 0 0 1")


def test_motor_tendon_rest_length():
    morph = load_morphology(Path(__file__).resolve().parents[1] / "designs" / "hummer" / "legacy-baseline" / "shoulder.tg")
    assert morph.motors.tendon_rest_length == 0.053
    assert morph.motors.crank_phase_lag_rad == math.radians(180)
    assert [foot.station for foot in morph.motors.feet] == [1, 4]


def test_motor_tendon_rest_length_foot_override(tmp_path=None):
    baseline = Path(__file__).resolve().parents[1] / "designs" / "hummer" / "legacy-baseline" / "shoulder.tg"
    text = baseline.read_text()
    if "tendon_rest_length_t1" in text:
        text = "\n".join(
            "    tendon_rest_length_t1 = 47.7 mm" if "tendon_rest_length_t1" in line else line
            for line in text.splitlines()
        ) + "\n"
    else:
        text = text.replace(
            "    tendon_rest_length = 53 mm\n",
            "    tendon_rest_length = 53 mm\n    tendon_rest_length_t1 = 47.7 mm\n",
        )
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shoulder.tg"
            path.write_text(text)
            morph = load_morphology(path)
    else:
        path = tmp_path / "shoulder.tg"
        path.write_text(text)
        morph = load_morphology(path)

    assert morph.motors.tendon_rest_length == 0.053
    assert math.isclose(morph.motors.feet[0].tendon_rest_length, 0.0477)
    assert morph.motors.feet[1].tendon_rest_length is None


def test_motor_tendons_are_crossed():
    morph = load_morphology(Path(__file__).resolve().parents[1] / "designs" / "hummer" / "legacy-baseline" / "shoulder.tg")
    root = ET.fromstring(emit_mjcf(compile_morphology(morph)))
    tendons = {el.attrib["name"]: [site.attrib["site"] for site in el] for el in root.find("tendon")}
    assert tendons["L_motor_t1"] == ["L_part_1_motor_t4", "crank_disks_L_t1"]
    assert tendons["L_motor_t4"] == ["L_part_1_motor_t1", "crank_disks_L_t4"]


def test_individual_structural_tendon_override(tmp_path=None):
    baseline = Path(__file__).resolve().parents[1] / "designs" / "hummer" / "legacy-baseline" / "shoulder.tg"
    text = baseline.read_text().replace(
        "motors:\n",
        "    overrides:\n"
        "        L_t1_to_t1_2 { stiffness: 123, pretension: 0.7 }\n"
        "        R_t1_to_t1_2 { stiffness: 456, pretension: 0.9 }\n"
        "motors:\n",
        1,
    )
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shoulder.tg"
            path.write_text(text)
            morph = load_morphology(path)
    else:
        path = tmp_path / "shoulder.tg"
        path.write_text(text)
        morph = load_morphology(path)
    root = ET.fromstring(emit_mjcf(compile_morphology(morph)))
    tendons = {el.attrib["name"]: el.attrib for el in root.find("tendon")}
    assert tendons["L_t1_to_t1_2"]["stiffness"] == "123.0"
    assert tendons["R_t1_to_t1_2"]["stiffness"] == "456.0"

