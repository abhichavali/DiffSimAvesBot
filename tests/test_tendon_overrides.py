"""Tendon materials and overrides, the motor tendons' wiring, and the inset motor anchor.

Run: uv run pytest -q tests/test_tendon_overrides.py
These stop at emit_mjcf (no probe), so they are quick.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from design_paths import DEFAULT_HUMMER_TG, EAGLE_TG
from morphology.compile import compile_morphology
from morphology.emit_mjcf import emit_mjcf
from morphology.mjcf_parts import color_to_rgba
from morphology.model import TendonDefaults, TendonMaterial, TendonNet
from morphology.tendon_net import resolve_material_props
from morphology.tg_parser import _parse_brace_dict, _parse_material_dict, load_morphology

BASELINE = Path(__file__).resolve().parents[1] / "designs" / "hummer" / "legacy-baseline" / "shoulder.tg"


def emitted(tg: Path) -> ET.Element:
    return ET.fromstring(emit_mjcf(compile_morphology(load_morphology(tg))))


def test_color_to_rgba():
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
    morph = load_morphology(BASELINE)
    assert morph.motors.tendon_rest_length == 0.053
    assert morph.motors.crank_phase_lag_rad == math.radians(180)
    assert [foot.station for foot in morph.motors.feet] == [1, 4]


def test_motor_tendon_rest_length_foot_override(tmp_path):
    path = tmp_path / "shoulder.tg"
    path.write_text("\n".join(
        "    tendon_rest_length_t1 = 47.7 mm" if "tendon_rest_length_t1" in line else line
        for line in BASELINE.read_text().splitlines()
    ) + "\n")
    morph = load_morphology(path)
    assert morph.motors.tendon_rest_length == 0.053
    assert math.isclose(morph.motors.feet[0].tendon_rest_length, 0.0477)
    assert morph.motors.feet[1].tendon_rest_length is None


def test_motor_tendons_are_crossed():
    tendons = {el.attrib["name"]: [site.attrib["site"] for site in el] for el in emitted(BASELINE).find("tendon")}
    assert tendons["L_motor_t1"] == ["L_part_1_motor_t4", "crank_disks_L_t1"]
    assert tendons["L_motor_t4"] == ["L_part_1_motor_t1", "crank_disks_L_t4"]


def test_probe_limit_is_the_parked_motor_tendon_length():
    """The probe's motor-tendon limit is the qpos0 length, with the disk's tilt and pitch."""
    for tg in (BASELINE, DEFAULT_HUMMER_TG, EAGLE_TG):
        model = mujoco.MjModel.from_xml_string(emit_mjcf(compile_morphology(load_morphology(tg))))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        motor = [i for i in range(model.ntendon) if "_motor_t" in model.tendon(i).name]
        assert len(motor) == 4
        for i in motor:
            assert abs(model.tendon_range[i][1] - data.ten_length[i]) < 1e-8, (tg, model.tendon(i).name)


def test_individual_structural_tendon_override(tmp_path):
    path = tmp_path / "shoulder.tg"
    path.write_text(BASELINE.read_text().replace(
        "motors:\n",
        "    overrides:\n"
        "        L_t1_to_t1_2 { stiffness: 123, pretension: 0.7 }\n"
        "        R_t1_to_t1_2 { stiffness: 456, pretension: 0.9 }\n"
        "motors:\n",
        1,
    ))
    tendons = {el.attrib["name"]: el.attrib for el in emitted(path).find("tendon")}
    assert tendons["L_t1_to_t1_2"]["stiffness"] == "123.0"
    assert tendons["R_t1_to_t1_2"]["stiffness"] == "456.0"


def test_inset_motor_anchor_is_bridged_by_a_rod(tmp_path):
    path = tmp_path / "shoulder.tg"
    path.write_text(BASELINE.read_text().replace("attach_radius = 1.0", "attach_radius = 0.5", 1))
    root = emitted(path)
    for side in "LR":
        strut = root.find(f".//body[@name='{side}_part_1']")
        assert [g.get("name") for g in strut.iter("geom") if "motor_rod" in g.get("name", "")] == [
            f"{side}_part_1_motor_rod_0", f"{side}_part_1_motor_rod_1"]
        assert strut.find(f"site[@name='{side}_part_1_motor_t1']") is not None
