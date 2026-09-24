"""Parse a .tg morphology file into the typed Morphology tree (model.py).

A .tg is line-oriented. Top-level statements (`name =`, `bird =`, `strut ...`)
sit beside sections (`world:`, `archetype strut C:`, `arrange ...`,
`placement:`, `tendons:`, `motors:`, `wing on ...`) whose body is the indented
lines below them. `#` starts a comment.

Literals carry units: mm, cm, g and ms are scaled to SI here; m, s, kg, N/m and
rad are taken as written. `deg` is NOT converted by the lexer -- each key
decides: every angle converts when its literal says deg (disk tilt/pitch/
crank_phase, crank_phase_lag, attach_along_edge, the wing pitch angles and the
eagle's bones/cables/feathers/wrist), and station angles and arrange.angle stay
in degrees by design. Missing keys take the model's defaults. Unknown keys and
malformed lines are errors: a typo would otherwise fall back to a default and
build a different design. The few keys old files carry that nothing reads
(archetype shape/symmetry, disk outboard, aero force_scale/body_torque/cop) are
accepted and ignored.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from morphology.model import (
    Arrange,
    DiskSpec,
    Face,
    FeatherVanes,
    Morphology,
    MotorFoot,
    Motors,
    Placement,
    RodSpec,
    StrutArc,
    TendonDefaults,
    TendonGenerator,
    TendonMaterial,
    TendonNet,
    Wing,
    WingAero,
    WingBones,
    WingCables,
    WingPitch,
    World,
    WristTensegrity,
)
from morphology.tendon_net import SemanticAnchor, canonical_edge_id

_UNITS = r"(mm|cm|m|ms|s|deg|rad|g|kg|N/m|N\s*/\s*m)"
_QUANTITY_RE = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*" + _UNITS + "?$", re.I)
_ARRAY_UNIT_RE = re.compile(r"^(\[[^\]]+\])\s+" + _UNITS + "$", re.I)
#: Units scaled to SI by division (dividing, not multiplying by 1e-3, keeps parses bit-stable).
_DIVISORS = {"mm": 1000.0, "cm": 100.0, "ms": 1000.0, "g": 1000.0}
_BOOL = {"on": True, "off": False, "true": True, "false": False}

_EDGE_RE = re.compile(
    r"^([MF])\[(\d+)([AB])\]\s*--\s*([MF])\[(\d+)([AB])\]"
    r"(?:\s*\{([^}]*)\})?\s*$"
)
_TENDON_OVERRIDE_RE = re.compile(r"^([LR]_[A-Za-z0-9_]+_to_[A-Za-z0-9_]+)\s*\{([^}]*)\}\s*$")
_DRIVE_RE = re.compile(r"^drive\s+M\[(\d+)\]\s+and\s+M\[(\d+)\]\s*$", re.I)
_ASSIGN_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")
_DEG_RE = re.compile(r"\bdeg\b", re.I)

#: `motors:` keys besides the per-foot `<key>_t<N>` ones. drive_phase_offset is the
#: old crank_phase_lag.
_MOTOR_KEYS = (
    "attach_along_edge", "attach_radius", "crank_phase_lag", "drive_phase_offset", "tendon_rest_length",
    "tendon_stiffness", "tendon_damping", "motor_direction", "control",
)
_WING_SECTIONS = ("pitch", "aero", "bones", "wrist", "cables", "feathers")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValueError("empty scalar")
    lower = text.lower()
    if lower in _BOOL:
        return _BOOL[lower]
    if lower == "derived":
        return None
    array_unit = _ARRAY_UNIT_RE.match(text)
    if array_unit:  # a unit after an array is dropped, not applied
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
    """A number with an optional unit, scaled to SI; anything else comes back as a string."""
    text = text.strip()
    if text[:1] in ('"', "'") and text.endswith(text[:1]):
        return text[1:-1]
    m = _QUANTITY_RE.match(text)
    if not m:
        return text
    value = float(m.group(1))
    unit = (m.group(2) or "").lower().replace(" ", "")
    return value / _DIVISORS[unit] if unit in _DIVISORS else value


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
            raise ValueError(f"expected key: value, got {part!r} in {text}")
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
    def opt(key):
        return None if raw.get(key) is None else float(raw[key])

    color = raw.get("color")
    return TendonMaterial(
        stiffness=opt("stiffness"),
        damping=opt("damping"),
        pretension=opt("pretension"),
        color=str(color) if color is not None else None,
    )


def _radians(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(math.radians(float(v)) for v in value)
    return math.radians(float(value))


def _assignment(line: str) -> tuple[str, str]:
    """`key = value` (or `key=value`) as (key, raw value)."""
    if " = " in line:
        key, val = line.split("=", 1)
        return key.strip(), val.strip()
    m = _ASSIGN_RE.match(line)
    if not m:
        raise ValueError(f"Unrecognized line: {line}")
    return m.group(1), m.group(2)


def _reject_unknown(where: str, raw: Dict[str, Any], known: str, ignored: tuple = ()) -> None:
    """Raise on a key outside `known` (space-separated); `ignored` keys are accepted and dropped."""
    known_keys = set(known.split())
    unknown = sorted(set(raw) - known_keys - set(ignored))
    if unknown:
        raise ValueError(f"unknown {where} key(s) {', '.join(unknown)}; expected one of "
                         + ", ".join(sorted(known_keys)))


def _floats(raw: Dict[str, Any], default: Any, names: str) -> Dict[str, float]:
    """{name: float(raw[name])} for each space-separated name, else `default`'s value."""
    return {n: float(raw.get(n, getattr(default, n))) for n in names.split()}


class _TgParser:
    def __init__(self, text: str) -> None:
        self.lines = [_strip_comment(line) for line in text.splitlines()]
        self.i = 0  # cursor into self.lines
        self.name = Morphology.name
        self.bird = ""
        self.world: Dict[str, Any] = {}
        self.archetype: Dict[str, Any] = {}
        self.arrange: Dict[str, Any] = {}
        self.placement: Dict[str, Any] = {}
        self.tendon_defaults: Dict[str, Any] = {}
        self.generators: List[TendonGenerator] = []
        self.edge_overrides: Dict[str, TendonMaterial] = {}
        self.individual_overrides: Dict[str, TendonMaterial] = {}
        self.disk: Dict[str, Any] = {}
        self.motor_feet = (1, 4)
        self.motors: Dict[str, Any] = {}
        self.per_foot: Dict[str, Dict[int, float]] = {"crank_throw_t": {}, "foot_outset_t": {},
                                                      "tendon_rest_length_t": {}}
        self.wing: Dict[str, Any] = {}
        self.wing_sections: Dict[str, Dict[str, Any]] = {s: {} for s in _WING_SECTIONS}
        self.declared: set[str] = set()

    def parse(self) -> Morphology:
        while self.i < len(self.lines):
            line = self.lines[self.i].strip()
            self.i += 1
            if not line:
                continue
            if line.startswith("name ="):
                self.name = line.split("=", 1)[1].strip()
            elif line.startswith("bird ="):
                self.bird = line.split("=", 1)[1].strip()
            elif line == "world:":
                self._assignments(self.world)
            elif line.startswith("archetype strut "):
                self._assignments(self.archetype)
            elif line.startswith("strut "):  # `strut F = C { role: fixed, plane: YZ }`: informational
                continue
            elif line.startswith("arrange "):
                self._assignments(self.arrange)
            elif line == "placement:":
                self._assignments(self.placement)
            elif line == "tendons:":
                self._tendons()
            elif line == "motors:":
                self._motors()
            elif line.startswith("wing on "):
                self._wing()
            else:
                raise ValueError(f"Unrecognized line: {line}")
        return self._build()

    def _block(self):
        """Yield the stripped lines of the indented block at the cursor, advancing past it."""
        while self.i < len(self.lines):
            raw = self.lines[self.i]
            if raw.strip() and not raw.startswith(("    ", "\t")):
                return
            self.i += 1
            if raw.strip():
                yield raw.strip()

    def _assignments(self, target: Dict[str, Any]) -> None:
        """`key = value` lines (and a `defaults { ... }` dict) into `target`; bare `name:` headers are skipped."""
        for line in self._block():
            if line.endswith(":") and "{" not in line and "=" not in line:
                continue
            if line.startswith("defaults "):
                target.update(_parse_brace_dict(line[len("defaults "):].strip()))
            else:
                key, val = _assignment(line)
                target[key] = _parse_scalar(val)

    def _tendons(self) -> None:
        in_overrides = False
        for line in self._block():
            if line.startswith("defaults "):
                self.tendon_defaults.update(_parse_brace_dict(line[len("defaults "):].strip()))
                continue
            if line.startswith("length"):
                continue
            if line.startswith("generators under sigma:"):
                in_overrides = False
                continue
            if line == "overrides:":
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
                if in_overrides:
                    edge_id = canonical_edge_id(SemanticAnchor("M", int(m_st), Face(m_face)),
                                                SemanticAnchor("F", int(f_st), Face(f_face)))
                    self.edge_overrides[edge_id] = material
                else:
                    self.generators.append(TendonGenerator(int(m_st), Face(m_face), int(f_st), Face(f_face), material))
                continue

            tendon_override = _TENDON_OVERRIDE_RE.match(line)
            if tendon_override and in_overrides:
                name, mat_text = tendon_override.groups()
                self.individual_overrides[name] = _parse_material_dict(_parse_brace_dict("{" + mat_text + "}"))
                continue

            raise ValueError(f"Unrecognized tendon line: {line}")

    def _motors(self) -> None:
        for line in self._block():
            if line.startswith("disk "):
                text = line[len("disk "):].strip()
                while not text.endswith("}") and self.i < len(self.lines):  # a dict may span lines
                    text += " " + self.lines[self.i].strip()
                    self.i += 1
                self.disk.update(_parse_brace_dict(text))
                for k in ("tilt", "pitch", "crank_phase"):
                    if re.search(rf"\b{k}\s*:[^,}}]*\bdeg\b", text, re.I):
                        self.disk[k] = math.radians(self.disk[k])
                continue
            drive = _DRIVE_RE.match(line)
            if drive:
                self.motor_feet = (int(drive.group(1)), int(drive.group(2)))
                continue
            if line.startswith("phase ="):
                continue
            key, eq, value = (part.strip() for part in line.partition("="))
            foot = re.fullmatch(r"(\w+_t)(\d+)", key)
            if eq and foot and foot.group(1) in self.per_foot:  # e.g. crank_throw_t4 = 23 mm
                self.per_foot[foot.group(1)][int(foot.group(2))] = float(_parse_quantity(value))
                continue
            if not eq or key not in _MOTOR_KEYS:
                raise ValueError(f"Unrecognized motor line: {line}")
            if key in ("crank_phase_lag", "drive_phase_offset", "attach_along_edge"):
                angle = float(_parse_scalar(value))
                key = "crank_phase_lag" if key == "drive_phase_offset" else key
                self.motors[key] = math.radians(angle) if _DEG_RE.search(line) else angle
            elif key == "control":
                self.motors[key] = str(_parse_scalar(value))
            else:
                self.motors[key] = float(_parse_scalar(value))

    def _wing(self) -> None:
        section: Optional[str] = None
        for line in self._block():
            if line.endswith(":") and line[:-1] in self.wing_sections:
                section = line[:-1]
                self.declared.add(section)
                continue
            key, val = _assignment(line)
            parsed = _parse_scalar(val)
            if _DEG_RE.search(val) and (
                section in ("bones", "cables", "feathers")
                or (section == "wrist" and key != "stations")  # stations stay in degrees
                or (section == "pitch" and key in ("free_angle", "upstroke_pitch", "downstroke_pitch",
                                                   "initial", "range"))
            ):
                parsed = _radians(parsed)
            (self.wing if section is None else self.wing_sections[section])[key] = parsed

    def _build(self) -> Morphology:
        arch = self.archetype
        stations = arch.get("stations")
        if not isinstance(stations, tuple) or len(stations) != 4:
            raise ValueError("archetype.stations must be four angles in degrees")
        arc_rod = arch.get("arc_rod", {})
        rung_rod = arch.get("rung_rod", {})
        if not isinstance(arc_rod, dict) or not isinstance(rung_rod, dict):
            raise ValueError("archetype arc_rod and rung_rod must be brace dicts")
        angle_deg = float(self.arrange.get("angle", 0.0))
        if len(self.generators) != 6:
            raise ValueError(f"Expected 6 tendon generators, got {len(self.generators)}")

        w, disk, m = self.world, self.disk, self.motors
        wing_pitch, aero = self.wing_sections["pitch"], self.wing_sections["aero"]
        _reject_unknown("world", w, "gravity timestep integrator scale meshdir")
        _reject_unknown("archetype", arch, "radius stations width arc_rod rung_rod",
                        ignored=("shape", "symmetry sigma"))
        _reject_unknown("arrange", self.arrange, "gap angle mobile_dx inplane_offset")
        _reject_unknown("placement", self.placement, "mount_height lateral_offset")
        _reject_unknown("tendon defaults", self.tendon_defaults, "color stiffness damping pretension width")
        _reject_unknown("disk", disk, "tilt pitch ctrl solref crank_throw crank_phase radius kv foot_outset "
                                      "disk_dx disk_dy", ignored=("outboard",))
        _reject_unknown("wing", self.wing, "rod_length mount_arm mesh")
        _reject_unknown("wing pitch", wing_pitch, "free_angle stiffness upstroke_stiffness downstroke_stiffness "
                                                  "damping initial range armature upstroke_pitch downstroke_pitch")
        _reject_unknown("wing aero", aero, "blades", ignored=("force_scale", "body_torque", "cop"))
        # Older files give one `free_angle` (the whole dead band) and one `stiffness`.
        free_angle = float(wing_pitch.get("free_angle", math.radians(30))) / 2.0
        pitch_stiffness = float(wing_pitch.get("stiffness", 0.5))

        return Morphology(
            name=self.name,
            bird=self.bird,
            world=World(
                gravity=tuple(float(x) for x in w.get("gravity", World.gravity)),
                timestep=float(w.get("timestep", World.timestep)),
                integrator=str(w.get("integrator", World.integrator)),
                scale=float(w.get("scale", World.scale)),
                meshdir=str(w.get("meshdir", World.meshdir)),
            ),
            arc=StrutArc(
                radius=float(arch["radius"]),
                stations_deg=tuple(float(x) for x in stations),
                width=float(arch["width"]),
                arc_rod=RodSpec(radius=float(arc_rod["radius"]), mass=float(arc_rod["mass"])),
                rung_rod=RodSpec(radius=float(rung_rod["radius"]), mass=float(rung_rod["mass"])),
            ),
            arrange=Arrange(
                gap=float(self.arrange["gap"]),
                angle_deg=angle_deg,
                **_floats(self.arrange, Arrange, "mobile_dx inplane_offset"),
            ),
            placement=Placement(
                mount_height=float(self.placement["mount_height"]),
                lateral_offset=float(self.placement["lateral_offset"]),
            ),
            tendons=TendonNet(
                defaults=TendonDefaults(
                    color=str(self.tendon_defaults.get("color", TendonDefaults.color)),
                    **_floats(self.tendon_defaults, TendonDefaults, "stiffness damping pretension width"),
                ),
                generators=tuple(self.generators),
                edge_overrides=dict(self.edge_overrides),
                individual_overrides=dict(self.individual_overrides),
            ),
            motors=Motors(
                disk=DiskSpec(
                    tilt_rad=float(disk.get("tilt", DiskSpec.tilt_rad)),
                    pitch_rad=float(disk.get("pitch", DiskSpec.pitch_rad)),
                    ctrlrange=tuple(float(x) for x in disk.get("ctrl", DiskSpec.ctrlrange)),
                    solreflimit=tuple(float(x) for x in disk.get("solref", DiskSpec.solreflimit)),
                    crank_throw=float(disk["crank_throw"]) if "crank_throw" in disk else None,
                    crank_phase_rad=float(disk.get("crank_phase", DiskSpec.crank_phase_rad)),
                    **_floats(disk, DiskSpec, "radius kv foot_outset disk_dx disk_dy"),
                ),
                feet=tuple(
                    MotorFoot(
                        station=station,
                        attach_along_edge_rad=m.get("attach_along_edge", MotorFoot.attach_along_edge_rad),
                        attach_radius=m.get("attach_radius", 1.0),
                        crank_throw=self.per_foot["crank_throw_t"].get(station),
                        foot_outset=self.per_foot["foot_outset_t"].get(station),
                        tendon_rest_length=self.per_foot["tendon_rest_length_t"].get(station),
                    )
                    for station in self.motor_feet
                ),
                tendon_rest_length=m.get("tendon_rest_length"),
                tendon_stiffness=m.get("tendon_stiffness", Motors.tendon_stiffness),
                tendon_damping=m.get("tendon_damping", Motors.tendon_damping),
                crank_phase_lag_rad=m.get("crank_phase_lag", Motors.crank_phase_lag_rad),
                motor_direction=m.get("motor_direction", Motors.motor_direction),
                control_file=m.get("control", Motors.control_file),
            ),
            wing=Wing(
                rod_length=float(self.wing.get("rod_length", Wing.rod_length)),
                mount_arm=float(self.wing.get("mount_arm", Wing.mount_arm)),
                mesh_file=str(self.wing.get("mesh", Wing.mesh_file)),
                pitch=WingPitch(
                    upstroke_stiffness=float(wing_pitch.get("upstroke_stiffness", pitch_stiffness)),
                    downstroke_stiffness=float(wing_pitch.get("downstroke_stiffness", pitch_stiffness)),
                    damping=float(wing_pitch.get("damping", WingPitch.damping)),
                    initial_rad=float(wing_pitch.get("initial", WingPitch.initial_rad)),
                    range_rad=tuple(float(x) for x in wing_pitch.get("range", WingPitch.range_rad)),
                    armature=float(wing_pitch.get("armature", WingPitch.armature)),
                    upstroke_pitch_rad=float(wing_pitch.get("upstroke_pitch", free_angle)),
                    downstroke_pitch_rad=float(wing_pitch.get("downstroke_pitch", free_angle)),
                ),
                aero=WingAero(blades=int(aero.get("blades", WingAero.blades))),
                bones=self._bones(),
            ),
        )

    def _strict(self, section: str, known: str) -> Optional[Dict[str, Any]]:
        """A strict wing section's keys, or None if it is absent. A typo'd key here
        would silently become a different wing (the bone mount especially is a
        searched value), so unknown keys and empty sections are errors."""
        raw = self.wing_sections[section]
        if not raw:
            if section in self.declared:
                raise ValueError(f"wing {section}: section is empty — remove it, or give it keys")
            return None
        _reject_unknown(f"wing {section}", raw, known)
        return raw

    def _bones(self) -> Optional[WingBones]:
        b = self._strict("bones", "mount humerus ulna manus chord primary mass elbow_range wrist_range "
                                  "twist_range servo damping armature")
        if b is None:
            return None
        d = WingBones()

        def vec(key, default, want):
            value = b.get(key, default)
            if not isinstance(value, (tuple, list)) or len(value) != want:
                raise ValueError(f"wing bones {key} must be {want} numbers, got {value!r}")
            return tuple(float(x) for x in value)

        return WingBones(
            mount_euler_rad=vec("mount", d.mount_euler_rad, 3),
            elbow_range_rad=vec("elbow_range", d.elbow_range_rad, 2),
            wrist_range_rad=vec("wrist_range", d.wrist_range_rad, 2),
            twist_range_rad=vec("twist_range", d.twist_range_rad, 2),
            **_floats(b, d, "humerus ulna manus chord primary mass servo damping armature"),
            wrist=self._wrist(),
            cables=self._cables(),
            feathers=self._feathers(),
        )

    def _feathers(self) -> Optional[FeatherVanes]:
        f = self._strict("feathers", "segments stiffness zeta armature range")
        if f is None:
            return None
        d = FeatherVanes()
        rng = f.get("range", d.range_rad)
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            raise ValueError(f"wing feathers range must be two angles, got {rng!r}")
        segments = int(f.get("segments", d.segments))
        if segments < 1:
            raise ValueError(f"wing feathers segments must be at least 1, got {segments}")
        return FeatherVanes(
            segments=segments,
            range_rad=(float(rng[0]), float(rng[1])),
            **_floats(f, d, "stiffness zeta armature"),
        )

    def _cables(self) -> Optional[WingCables]:
        c = self._strict("cables", "humerus_frac ulna_frac manus_frac offset disks humerus_guide ulna_guide "
                                   "pin_radius pin_phase stiffness damping width extend contract")
        if c is None:
            return None
        d = WingCables()

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
            disks=disks,
            extend=tuple(float(x) for x in triple("extend", d.extend)),
            contract=tuple(float(x) for x in triple("contract", d.contract)),
            **_floats(c, d, "humerus_frac ulna_frac manus_frac offset humerus_guide ulna_guide "
                            "pin_radius pin_phase stiffness damping width"),
        )

    def _wrist(self) -> Optional[WristTensegrity]:
        w = self._strict("wrist", "scale stations gap width stiffness damping pretension tendon_width "
                                  "arc_rod rung_rod")
        if w is None:
            return None
        d = WristTensegrity()
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
            stations_deg=tuple(float(x) for x in stations),
            arc_rod=rod("arc_rod", d.arc_rod),
            rung_rod=rod("rung_rod", d.rung_rod),
            **_floats(w, d, "scale gap width stiffness damping pretension tendon_width"),
        )


def load_morphology(path: str | Path) -> Morphology:
    """Load a .tg morphology file into a validated Morphology."""
    morph = _TgParser(Path(path).read_text()).parse()
    morph.validate()
    return morph
