"""Parse shoulder.tg DSL → Morphology object model.

The `.tg` format is the authoring surface for the morphology compiler. It
describes free parameters and topology; the compiler derives coordinates,
tendon rest lengths, wing CoP, and bilateral mirror geometry.

Unit suffixes on literals (mm, deg, ms, g, N/m, rad) are converted to SI
(base units) when building the typed ``Morphology`` dataclass tree.
"""

from __future__ import annotations

import math
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from morphology.model import (
    ArcPlane,
    Arrange,
    DiskSpec,
    Face,
    Morphology,
    MotorFoot,
    Motors,
    Placement,
    RodSpec,
    StrutArc,
    StrutInstance,
    StrutRole,
    TendonDefaults,
    TendonGenerator,
    TendonMaterial,
    TendonNet,
    Wing,
    WingAero,
    WingBones,
    FeatherVanes,
    WingCables,
    WristTensegrity,
    WingPitch,
    World,
)
from morphology.tendon_net import SemanticAnchor, canonical_edge_id

_ANCHOR_RE = re.compile(r"^([MF])\[(\d+)([AB])\]$")
_EDGE_RE = re.compile(
    r"^([MF])\[(\d+)([AB])\]\s*--\s*([MF])\[(\d+)([AB])\]"
    r"(?:\s*\{([^}]*)\})?\s*$"
)
_TENDON_OVERRIDE_RE = re.compile(r"^([LR]_[A-Za-z0-9_]+_to_[A-Za-z0-9_]+)\s*\{([^}]*)\}\s*$")
_DRIVE_RE = re.compile(r"^drive\s+M\[(\d+)\]\s+and\s+M\[(\d+)\]\s*$", re.I)
_ASSIGN_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")
_BOOL = {"on": True, "off": False, "true": True, "false": False}


def _strip_comment(line: str) -> str:
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.rstrip()


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValueError("empty scalar")
    lower = text.lower()
    if lower in _BOOL:
        return _BOOL[lower]
    if lower == "derived":
        return None
    array_unit = re.match(
        r"^(\[[^\]]+\])\s+(mm|cm|m|ms|s|deg|rad|g|kg|N/m|N\s*/\s*m)$",
        text,
        re.I,
    )
    if array_unit:
        text = array_unit.group(1)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return ()
        return tuple(_parse_scalar(part.strip()) for part in inner.split(","))
    if text.startswith("{") and text.endswith("}"):
        return _parse_brace_dict(text)
    return _parse_quantity(text)


def _parse_quantity(text: str) -> float | str:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]

    m = re.match(
        r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
        r"(mm|cm|m|ms|s|deg|rad|g|kg|N/m|N\s*/\s*m)?$",
        text,
        re.I,
    )
    if not m:
        return text
    value = float(m.group(1))
    unit = (m.group(2) or "").lower().replace(" ", "")
    if unit == "mm":
        return value / 1000.0
    if unit == "cm":
        return value / 100.0
    if unit in ("m", ""):
        return value
    if unit == "ms":
        return value / 1000.0
    if unit == "s":
        return value
    if unit == "deg":
        return value
    if unit == "rad":
        return value
    if unit == "g":
        return value / 1000.0
    if unit == "kg":
        return value
    if unit in ("n/m",):
        return value
    return value


def _parse_brace_dict(text: str) -> Dict[str, Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    parts: List[str] = []
    depth = 0
    bracket = 0
    current: List[str] = []
    for ch in inner:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        if ch == "," and depth == 0 and bracket == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    out: Dict[str, Any] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        out[key.strip()] = _scalar_or_str(val.strip())
    return out


def _scalar_or_str(text: str) -> Any:
    """Parse as a scalar, else keep the raw string (e.g. an rgba/hex color)."""
    try:
        return _parse_scalar(text)
    except ValueError:
        return text


def _parse_material_dict(raw: Dict[str, Any]) -> TendonMaterial:
    color = raw.get("color")
    return TendonMaterial(
        stiffness=_optional_float(raw.get("stiffness")),
        damping=_optional_float(raw.get("damping")),
        pretension=_optional_float(raw.get("pretension")),
        color=str(color) if color is not None else None,
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _parse_anchor(token: str) -> SemanticAnchor:
    m = _ANCHOR_RE.match(token.strip())
    if not m:
        raise ValueError(f"Invalid anchor {token!r}")
    return SemanticAnchor(m.group(1), int(m.group(2)), Face(m.group(3)))


def _canonical_override_id(m_token: str, f_token: str) -> str:
    a = _parse_anchor(m_token)
    b = _parse_anchor(f_token)
    return canonical_edge_id(a, b)


class _TgParser:
    def __init__(self, text: str) -> None:
        self.lines = [_strip_comment(line) for line in text.splitlines()]
        self.name = "bi-shoulder"
        self.bird = ""
        self.world: Dict[str, Any] = {}
        self.archetype: Dict[str, Any] = {}
        self.struts: Dict[str, Dict[str, Any]] = {}
        self.arrange: Dict[str, Any] = {}
        self.placement: Dict[str, Any] = {}
        self.tendon_defaults: Dict[str, Any] = {}
        self.generators: List[TendonGenerator] = []
        self.edge_overrides: Dict[str, TendonMaterial] = {}
        self.individual_overrides: Dict[str, TendonMaterial] = {}
        self.disk: Dict[str, Any] = {}
        self.motor_feet: Tuple[int, int] = (1, 4)
        self.attach_along_edge = 1.2
        self.attach_radius = 1.0
        self.foot_crank_throw: Dict[int, float] = {}
        self.foot_outset: Dict[int, float] = {}
        self.foot_tendon_rest_length: Dict[int, float] = {}
        self.tendon_rest_length: Optional[float] = None
        self.tendon_stiffness = 500.0
        self.tendon_damping = 1.0
        self.motor_direction = 1.0
        self.control_file = "control.tg"
        self.crank_phase_lag_rad = 0.0
        self.wing: Dict[str, Any] = {}
        self.wing_pitch: Dict[str, Any] = {}
        self.wing_aero: Dict[str, Any] = {}
        self.wing_bones: Dict[str, Any] = {}
        self.wing_bones_declared = False
        self.wing_wrist: Dict[str, Any] = {}
        self.wing_wrist_declared = False
        self.wing_cables: Dict[str, Any] = {}
        self.wing_cables_declared = False
        self.wing_feathers: Dict[str, Any] = {}
        self.wing_feathers_declared = False

    def parse(self) -> Morphology:
        i = 0
        while i < len(self.lines):
            line = self.lines[i].strip()
            i += 1
            if not line:
                continue

            if line.startswith("name ="):
                self.name = line.split("=", 1)[1].strip()
                continue

            if line.startswith("bird ="):
                self.bird = line.split("=", 1)[1].split("#", 1)[0].strip()
                continue

            if line == "world:":
                i = self._parse_block(i, self.world)
                continue

            if line.startswith("archetype strut "):
                i = self._parse_block(i, self.archetype)
                continue

            if line.startswith("strut "):
                self._parse_strut_line(line)
                continue

            if line.startswith("arrange "):
                i = self._parse_block(i, self.arrange)
                continue

            if line == "placement:":
                i = self._parse_block(i, self.placement)
                continue

            if line == "tendons:":
                i = self._parse_tendons(i)
                continue

            if line == "motors:":
                i = self._parse_motors(i)
                continue

            if line.startswith("wing on "):
                i = self._parse_wing(i)
                continue

            raise ValueError(f"Unrecognized line: {line}")

        return self._build()

    def _parse_block(self, start: int, target: Dict[str, Any]) -> int:
        i = start
        while i < len(self.lines):
            raw = self.lines[i]
            if not raw.strip():
                i += 1
                continue
            if not raw.startswith("    "):
                break
            line = raw.strip()
            i += 1
            if line.endswith(":") and "{" not in line and "=" not in line:
                continue
            if line.startswith("defaults "):
                brace = line[len("defaults ") :].strip()
                target.update(_parse_brace_dict(brace))
                continue
            if " = " in line:
                key, val = line.split("=", 1)
                target[key.strip()] = _parse_scalar(val.strip())
                continue
            m = _ASSIGN_RE.match(line)
            if m:
                target[m.group(1)] = _parse_scalar(m.group(2))
        return i

    def _parse_strut_line(self, line: str) -> None:
        # strut F = C { role: fixed, plane: YZ }
        m = re.match(r"strut\s+(\w+)\s*=\s*(\w+)\s*\{([^}]*)\}", line, re.I)
        if not m:
            raise ValueError(f"Invalid strut line: {line}")
        side = m.group(1).upper()
        props = _parse_brace_dict("{" + m.group(3) + "}")
        self.struts[side] = props

    def _parse_tendons(self, start: int) -> int:
        i = start
        in_generators = False
        in_overrides = False
        while i < len(self.lines):
            raw = self.lines[i]
            if not raw.strip():
                i += 1
                continue
            if not raw.startswith("    ") and not raw.startswith("\t"):
                break
            line = raw.strip()
            i += 1

            if line.startswith("defaults "):
                self.tendon_defaults.update(_parse_brace_dict(line[len("defaults ") :].strip()))
                continue
            if line.startswith("length"):
                continue
            if line.startswith("generators under sigma:"):
                in_generators = True
                in_overrides = False
                continue
            if line == "overrides:":
                in_generators = False
                in_overrides = True
                continue

            edge = _EDGE_RE.match(line)
            if edge:
                m_side, m_st, m_face, f_side, f_st, f_face, mat_text = edge.groups()
                if m_side != "M" or f_side != "F":
                    raise ValueError(f"Generator must connect M to F, got: {line}")
                material = TendonMaterial()
                if mat_text:
                    material = _parse_material_dict(_parse_brace_dict("{" + mat_text + "}"))
                gen = TendonGenerator(
                    mobile_station=int(m_st),
                    mobile_face=Face(m_face),
                    fixed_station=int(f_st),
                    fixed_face=Face(f_face),
                    material=material,
                )
                if in_overrides:
                    edge_id = _canonical_override_id(
                        f"M[{m_st}{m_face}]", f"F[{f_st}{f_face}]"
                    )
                    self.edge_overrides[edge_id] = material
                else:
                    self.generators.append(gen)
                continue

            tendon_override = _TENDON_OVERRIDE_RE.match(line)
            if tendon_override and in_overrides:
                name, mat_text = tendon_override.groups()
                self.individual_overrides[name] = _parse_material_dict(
                    _parse_brace_dict("{" + mat_text + "}")
                )
                continue

            raise ValueError(f"Unrecognized tendon line: {line}")
        return i

    def _parse_motors(self, start: int) -> int:
        i = start
        while i < len(self.lines):
            raw = self.lines[i]
            if not raw.strip():
                i += 1
                continue
            if not raw.startswith("    "):
                break
            line = raw.strip()
            i += 1

            if line.startswith("disk "):
                brace_text = line[len("disk ") :].strip()
                while not brace_text.endswith("}"):
                    if i >= len(self.lines):
                        break
                    brace_text += " " + self.lines[i].strip()
                    i += 1
                self.disk.update(_parse_brace_dict(brace_text))
                continue
            drive = _DRIVE_RE.match(line)
            if drive:
                self.motor_feet = (int(drive.group(1)), int(drive.group(2)))
                continue
            if line.startswith("attach_along_edge"):
                self.attach_along_edge = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("attach_radius"):
                self.attach_radius = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("crank_throw_t"):
                key, value = line.split("=", 1)
                station = int(key.strip()[len("crank_throw_t"):])
                self.foot_crank_throw[station] = float(_parse_quantity(value.strip()))
                continue
            if line.startswith("foot_outset_t"):
                key, value = line.split("=", 1)
                station = int(key.strip()[len("foot_outset_t"):])
                self.foot_outset[station] = float(_parse_quantity(value.strip()))
                continue
            if line.startswith("tendon_rest_length_t"):
                key, value = line.split("=", 1)
                station = int(key.strip()[len("tendon_rest_length_t"):])
                self.foot_tendon_rest_length[station] = float(_parse_quantity(value.strip()))
                continue
            if line.startswith(("crank_phase_lag", "drive_phase_offset")):
                raw = line.split("=", 1)[1].strip()
                val = float(_parse_scalar(raw))
                if re.search(r"\bdeg\b", line, re.I):
                    val = math.radians(val)
                self.crank_phase_lag_rad = val
                continue
            if line.startswith("tendon_rest_length"):
                self.tendon_rest_length = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("tendon_stiffness"):
                self.tendon_stiffness = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("tendon_damping"):
                self.tendon_damping = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("motor_direction"):
                self.motor_direction = float(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("control"):
                self.control_file = str(_parse_scalar(line.split("=", 1)[1].strip()))
                continue
            if line.startswith("phase ="):
                continue
            raise ValueError(f"Unrecognized motor line: {line}")
        return i

    def _parse_wing(self, start: int) -> int:
        i = start
        section: Optional[str] = None
        while i < len(self.lines):
            raw = self.lines[i]
            if not raw.strip():
                i += 1
                continue
            if not raw.startswith("    "):
                break
            line = raw.strip()
            indent = len(raw) - len(raw.lstrip())
            i += 1

            if line == "pitch:":
                section = "pitch"
                continue
            if line == "aero:":
                section = "aero"
                continue
            if line == "bones:":
                section = "bones"
                self.wing_bones_declared = True
                continue
            if line == "wrist:":
                section = "wrist"
                self.wing_wrist_declared = True
                continue
            if line == "cables:":
                section = "cables"
                self.wing_cables_declared = True
                continue
            if line == "feathers:":
                section = "feathers"
                self.wing_feathers_declared = True
                continue
            if " = " in line:
                key, val = line.split("=", 1)
                key = key.strip()
                bucket = self.wing
                if section == "pitch":
                    bucket = self.wing_pitch
                elif section == "aero":
                    bucket = self.wing_aero
                elif section == "bones":
                    bucket = self.wing_bones
                elif section == "wrist":
                    bucket = self.wing_wrist
                elif section == "cables":
                    bucket = self.wing_cables
                elif section == "feathers":
                    bucket = self.wing_feathers
                parsed = _parse_scalar(val.strip())
                # _parse_scalar keeps angle literals numeric; convert deg->rad here.
                if (
                    section == "pitch"
                    and key in ("free_angle", "upstroke_pitch", "downstroke_pitch")
                    and re.search(r"\bdeg\b", val, re.I)
                ):
                    parsed = math.radians(float(parsed))
                elif (
                    section == "wrist"
                    and key != "stations"  # stations stay in degrees, like archetype's
                    and re.search(r"\bdeg\b", val, re.I)
                ) or (section in ("bones", "cables", "feathers")
                      and re.search(r"\bdeg\b", val, re.I)):
                    parsed = (
                        tuple(math.radians(float(v)) for v in parsed)
                        if isinstance(parsed, tuple)
                        else math.radians(float(parsed))
                    )
                bucket[key] = parsed
                continue
            if indent >= 8 and ":" in line:
                continue
        return i

    def _build(self) -> Morphology:
        stations = self.archetype.get("stations")
        if not isinstance(stations, tuple) or len(stations) != 4:
            raise ValueError("archetype.stations must be four angles in degrees")

        arc_rod = self.archetype.get("arc_rod", {})
        rung_rod = self.archetype.get("rung_rod", {})
        if not isinstance(arc_rod, dict) or not isinstance(rung_rod, dict):
            raise ValueError("archetype arc_rod and rung_rod must be brace dicts")

        f_props = self.struts.get("F", {})
        m_props = self.struts.get("M", {})

        angle_deg = float(self.arrange.get("angle", 0.0))

        tilt = self.disk.get("tilt", 0.26)
        if isinstance(tilt, (int, float)) and tilt > 3.0:
            tilt = math.radians(tilt)

        pitch = self.disk.get("pitch", 0.0)
        if isinstance(pitch, (int, float)) and pitch > 3.0:
            pitch = math.radians(pitch)

        if len(self.generators) != 6:
            raise ValueError(f"Expected 6 tendon generators, got {len(self.generators)}")

        fallback_free_angle = float(self.wing_pitch.get("free_angle", math.radians(30))) / 2.0
        fallback_stiffness = float(self.wing_pitch.get("stiffness", 0.5))

        return Morphology(
            name=self.name,
            bird=self.bird,
            world=World(
                gravity=tuple(float(x) for x in self.world.get("gravity", (0.0, 0.0, -9.81))),
                timestep=float(self.world.get("timestep", 0.000375)),
                integrator=str(self.world.get("integrator", "Euler")),
                scale=float(self.world.get("scale", 1.0)),
                meshdir=str(self.world.get("meshdir", "assets")),
            ),
            arc=StrutArc(
                radius=float(self.archetype["radius"]),
                stations_deg=tuple(float(x) for x in stations),
                width=float(self.archetype["width"]),
                arc_rod=RodSpec(
                    radius=float(arc_rod["radius"]),
                    mass=float(arc_rod["mass"]),
                ),
                rung_rod=RodSpec(
                    radius=float(rung_rod["radius"]),
                    mass=float(rung_rod["mass"]),
                ),
            ),
            mobile=StrutInstance(
                role=StrutRole.MOBILE,
                plane=ArcPlane(str(m_props.get("plane", "XZ")).upper()),
            ),
            fixed=StrutInstance(
                role=StrutRole.FIXED,
                plane=ArcPlane(str(f_props.get("plane", "YZ")).upper()),
            ),
            arrange=Arrange(
                gap=float(self.arrange["gap"]),
                angle_deg=angle_deg,
                mobile_dx=float(self.arrange.get("mobile_dx", 0.005)),
                inplane_offset=float(self.arrange.get("inplane_offset", 0.004649)),
            ),
            placement=Placement(
                mount_height=float(self.placement["mount_height"]),
                lateral_offset=float(self.placement["lateral_offset"]),
            ),
            tendons=TendonNet(
                defaults=TendonDefaults(
                    stiffness=float(self.tendon_defaults.get("stiffness", 300.0)),
                    damping=float(self.tendon_defaults.get("damping", 0.5)),
                    pretension=float(self.tendon_defaults.get("pretension", 0.85)),
                    width=float(self.tendon_defaults.get("width", 0.00075)),
                    color=str(self.tendon_defaults.get("color", "0.8 0.2 0.2 1")),
                ),
                generators=tuple(self.generators),
                edge_overrides=dict(self.edge_overrides),
                individual_overrides=dict(self.individual_overrides),
            ),
            motors=Motors(
                disk=DiskSpec(
                    radius=float(self.disk.get("radius", 0.015)),
                    tilt_rad=float(tilt),
                    pitch_rad=float(pitch),
                    kv=float(self.disk.get("kv", 0.25)),
                    ctrlrange=tuple(float(x) for x in self.disk.get("ctrl", (-10.0, 10.0))),
                    solreflimit=tuple(float(x) for x in self.disk.get("solref", (0.02857142857142857, 1.0))),
                    outboard=float(self.disk.get("outboard", 0.01875)),
                    crank_throw=(float(self.disk["crank_throw"]) if "crank_throw" in self.disk else None),
                    **({"crank_phase_rad": float(self.disk["crank_phase"])} if "crank_phase" in self.disk else {}),
                    foot_outset=float(self.disk.get("foot_outset", 0.0)),
                    disk_dx=float(self.disk.get("disk_dx", 0.0)),
                    disk_dy=float(self.disk.get("disk_dy", 0.0)),
                ),
                feet=(
                    MotorFoot(
                        station=self.motor_feet[0],
                        attach_along_edge_rad=self.attach_along_edge,
                        attach_radius=self.attach_radius,
                        crank_throw=self.foot_crank_throw.get(self.motor_feet[0]),
                        foot_outset=self.foot_outset.get(self.motor_feet[0]),
                        tendon_rest_length=self.foot_tendon_rest_length.get(self.motor_feet[0]),
                    ),
                    MotorFoot(
                        station=self.motor_feet[1],
                        attach_along_edge_rad=self.attach_along_edge,
                        attach_radius=self.attach_radius,
                        crank_throw=self.foot_crank_throw.get(self.motor_feet[1]),
                        foot_outset=self.foot_outset.get(self.motor_feet[1]),
                        tendon_rest_length=self.foot_tendon_rest_length.get(self.motor_feet[1]),
                    ),
                ),
                tendon_rest_length=self.tendon_rest_length,
                tendon_stiffness=float(self.tendon_stiffness),
                tendon_damping=float(self.tendon_damping),
                crank_phase_lag_rad=float(self.crank_phase_lag_rad),
                motor_direction=float(self.motor_direction),
                control_file=self.control_file,
            ),
            wing=Wing(
                rod_length=float(self.wing.get("rod_length", 0.018)),
                mount_arm=float(self.wing.get("mount_arm", 0.055)),
                mesh_file=str(self.wing.get("mesh", "wing.stl")),
                pitch=WingPitch(
                    upstroke_stiffness=float(
                        self.wing_pitch.get("upstroke_stiffness", fallback_stiffness)
                    ),
                    downstroke_stiffness=float(
                        self.wing_pitch.get("downstroke_stiffness", fallback_stiffness)
                    ),
                    damping=float(self.wing_pitch.get("damping", 0.0005)),
                    initial_rad=float(self.wing_pitch.get("initial", 0.35)),
                    range_rad=tuple(float(x) for x in self.wing_pitch.get("range", (-1.0, 1.0))),
                    armature=float(self.wing_pitch.get("armature", 1e-5)),
                    upstroke_pitch_rad=float(
                        self.wing_pitch.get("upstroke_pitch", fallback_free_angle)
                    ),
                    downstroke_pitch_rad=float(
                        self.wing_pitch.get("downstroke_pitch", fallback_free_angle)
                    ),
                ),
                aero=WingAero(
                    blades=int(self.wing_aero.get("blades", 4)),
                    force_scale=float(self.wing_aero.get("force_scale", 1_000_000.0)),
                    include_body_torque=bool(self.wing_aero.get("body_torque", False)),
                ),
                bones=self._build_bones(),
            ),
        )

    def _build_bones(self) -> Optional[WingBones]:
        # Strict, unlike a plain dict.get chain: a typo'd key here silently
        # substitutes a dataclass default, and the mount especially is a searched
        # value where a dropped component is a different wing, not a rounder
        # number. Fail on the .tg line instead.
        b = self.wing_bones
        if not b:
            if self.wing_bones_declared:
                raise ValueError("wing bones: section is empty — remove it, or give it keys")
            return None
        d = WingBones()
        known = {
            "mount", "humerus", "ulna", "manus", "chord", "primary", "mass",
            "elbow_range", "wrist_range", "twist_range", "servo", "damping", "armature",
        }
        unknown = sorted(set(b) - known)
        if unknown:
            raise ValueError(
                f"unknown wing bones key(s) {', '.join(unknown)}; expected one of "
                + ", ".join(sorted(known))
            )

        def vec(key, default, want):
            value = b.get(key, default)
            if not isinstance(value, (tuple, list)) or len(value) != want:
                raise ValueError(f"wing bones {key} must be {want} numbers, got {value!r}")
            return tuple(float(x) for x in value)

        pair = lambda k, dflt: vec(k, dflt, 2)  # noqa: E731
        return WingBones(
            mount_euler_rad=vec("mount", d.mount_euler_rad, 3),
            humerus=float(b.get("humerus", d.humerus)),
            ulna=float(b.get("ulna", d.ulna)),
            manus=float(b.get("manus", d.manus)),
            chord=float(b.get("chord", d.chord)),
            primary=float(b.get("primary", d.primary)),
            mass=float(b.get("mass", d.mass)),
            elbow_range_rad=pair("elbow_range", d.elbow_range_rad),
            wrist_range_rad=pair("wrist_range", d.wrist_range_rad),
            twist_range_rad=pair("twist_range", d.twist_range_rad),
            servo=float(b.get("servo", d.servo)),
            damping=float(b.get("damping", d.damping)),
            armature=float(b.get("armature", d.armature)),
            wrist=self._build_wrist(),
            cables=self._build_cables(),
            feathers=self._build_feathers(),
        )

    def _build_feathers(self) -> Optional[FeatherVanes]:
        f = self.wing_feathers
        if not f:
            if self.wing_feathers_declared:
                raise ValueError("wing feathers: section is empty — remove it, or give it keys")
            return None
        d = FeatherVanes()
        known = {"segments", "stiffness", "zeta", "armature", "range"}
        unknown = sorted(set(f) - known)
        if unknown:
            raise ValueError(
                f"unknown wing feathers key(s) {', '.join(unknown)}; expected one of "
                + ", ".join(sorted(known))
            )
        rng = f.get("range", d.range_rad)
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            raise ValueError(f"wing feathers range must be two angles, got {rng!r}")
        segments = int(f.get("segments", d.segments))
        if segments < 1:
            raise ValueError(f"wing feathers segments must be at least 1, got {segments}")
        return FeatherVanes(
            segments=segments,
            stiffness=float(f.get("stiffness", d.stiffness)),
            zeta=float(f.get("zeta", d.zeta)),
            armature=float(f.get("armature", d.armature)),
            range_rad=(float(rng[0]), float(rng[1])),
        )

    def _build_cables(self) -> Optional[WingCables]:
        c = self.wing_cables
        if not c:
            if self.wing_cables_declared:
                raise ValueError("wing cables: section is empty — remove it, or give it keys")
            return None
        d = WingCables()
        known = {"humerus_frac", "ulna_frac", "manus_frac", "offset", "disks",
                 "humerus_guide", "ulna_guide",
                 "pin_radius", "pin_phase", "stiffness", "damping", "width",
                 "extend", "contract"}
        unknown = sorted(set(c) - known)
        if unknown:
            raise ValueError(
                f"unknown wing cables key(s) {', '.join(unknown)}; expected one of "
                + ", ".join(sorted(known))
            )

        def triple(key, default):
            v = c.get(key, default)
            if not isinstance(v, (tuple, list)) or len(v) != 3:
                raise ValueError(f"wing cables {key} must be 3 values (humerus, ulna, manus), got {v!r}")
            return v

        disks = tuple(str(x).strip() for x in triple("disks", d.disks))
        for name in disks:
            if name not in ("t1", "t4"):
                raise ValueError(f"wing cables disks must each be t1 or t4, got {name!r}")
        return WingCables(
            humerus_frac=float(c.get("humerus_frac", d.humerus_frac)),
            ulna_frac=float(c.get("ulna_frac", d.ulna_frac)),
            manus_frac=float(c.get("manus_frac", d.manus_frac)),
            offset=float(c.get("offset", d.offset)),
            humerus_guide=float(c.get("humerus_guide", d.humerus_guide)),
            ulna_guide=float(c.get("ulna_guide", d.ulna_guide)),
            disks=disks,
            pin_radius=float(c.get("pin_radius", d.pin_radius)),
            pin_phase=float(c.get("pin_phase", d.pin_phase)),
            stiffness=float(c.get("stiffness", d.stiffness)),
            damping=float(c.get("damping", d.damping)),
            width=float(c.get("width", d.width)),
            extend=tuple(float(x) for x in triple("extend", d.extend)),
            contract=tuple(float(x) for x in triple("contract", d.contract)),
        )

    def _build_wrist(self) -> Optional[WristTensegrity]:
        w = self.wing_wrist
        if not w:
            if self.wing_wrist_declared:
                raise ValueError("wing wrist: section is empty — remove it, or give it keys")
            return None
        d = WristTensegrity()
        known = {"scale", "stations", "gap", "width", "stiffness", "damping",
                 "pretension", "tendon_width", "arc_rod", "rung_rod"}
        unknown = sorted(set(w) - known)
        if unknown:
            raise ValueError(
                f"unknown wing wrist key(s) {', '.join(unknown)}; expected one of "
                + ", ".join(sorted(known))
            )
        stations = w.get("stations", d.stations_deg)
        if not isinstance(stations, (tuple, list)) or len(stations) != 4:
            raise ValueError(f"wing wrist stations must be four angles, got {stations!r}")

        def rod(key, default):
            raw = w.get(key)
            if raw is None:
                return default
            if not isinstance(raw, dict):
                raise ValueError(f"wing wrist {key} must be a brace dict, got {raw!r}")
            return RodSpec(float(raw.get("radius", default.radius)),
                           float(raw.get("mass", default.mass)))

        return WristTensegrity(
            scale=float(w.get("scale", d.scale)),
            stations_deg=tuple(float(x) for x in stations),
            gap=float(w.get("gap", d.gap)),
            width=float(w.get("width", d.width)),
            stiffness=float(w.get("stiffness", d.stiffness)),
            damping=float(w.get("damping", d.damping)),
            pretension=float(w.get("pretension", d.pretension)),
            tendon_width=float(w.get("tendon_width", d.tendon_width)),
            arc_rod=rod("arc_rod", d.arc_rod),
            rung_rod=rod("rung_rod", d.rung_rod),
        )


def load_morphology(path: str | Path) -> Morphology:
    """Load a .tg morphology file into a validated Morphology."""
    text = Path(path).read_text()
    morph = _TgParser(text).parse()
    morph.validate()
    return morph
