"""Per-tendon material overrides: color / pretension / stiffness.

Runnable directly (`uv run python tests/test_tendon_overrides.py`) or via pytest.
Covers the new logic units without a full mesh-dependent compile.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morphology.emit_mjcf import _color_to_rgba
from morphology.model import TendonDefaults, TendonMaterial, TendonNet
from morphology.symmetry import resolve_material_props
from morphology.tg_parser import _parse_brace_dict, _parse_material_dict, load_morphology


def test_color_to_rgba():
    assert _color_to_rgba("1 0 0 1") == "1 0 0 1"        # rgba passthrough
    assert _color_to_rgba("0.5 0.5 0.5") == "0.5 0.5 0.5 1"  # alpha padded
    assert _color_to_rgba("00ff00") == "0.0 1.0 0.0 1"   # 6-digit hex (no '#')


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


def test_motor_foot_pretension_override():
    morph = load_morphology(Path(__file__).resolve().parents[1] / "design" / "baseline_shoulder.tg")
    assert morph.motors.tendon_pretension == 0.82
    assert morph.motors.foot_tendon_pretension == {1: 0.67}


if __name__ == "__main__":
    for fn in (
        test_color_to_rgba,
        test_parse_material_dict,
        test_resolve_falls_back_to_defaults,
        test_motor_foot_pretension_override,
    ):
        fn()
        print(f"ok  {fn.__name__}")
    print("all passed")
