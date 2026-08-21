"""Typed morphology object model (Option B — dataclasses)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class StrutRole(str, Enum):
    FIXED = "FIXED"
    MOBILE = "MOBILE"


class ArcPlane(str, Enum):
    XZ = "XZ"  # mobile strut: arc in x–z, width along y
    YZ = "YZ"  # fixed strut: arc in y–z, width along x


class StrutSide(str, Enum):
    MOBILE = "M"
    FIXED = "F"


class Face(str, Enum):
    A = "A"
    B = "B"


@dataclass
class RodSpec:
    radius: float
    mass: float


@dataclass
class StrutArc:
    """Shared strut archetype geometry."""

    radius: float  # arc radius (m)
    stations_deg: Tuple[float, float, float, float]
    width: float  # A↔B face gap (m)
    arc_rod: RodSpec
    rung_rod: RodSpec


@dataclass
class StrutInstance:
    role: StrutRole
    plane: ArcPlane


@dataclass
class Arrange:
    """Mobile strut pose relative to fixed (rest pose)."""

    gap: float  # m, lateral separation along y
    angle_deg: float = 0.0  # rotation about z at fixed origin
    mobile_dx: float = 0.005  # world x offset of mobile vs fixed
    inplane_offset: float = 0.004649  # orthogonal arc centerline shift (fixed x − mobile cx)


@dataclass
class Placement:
    mount_height: float  # world z of strut assembly anchor (m)
    lateral_offset: float  # |y| for left shoulder; right = mirror (m)


@dataclass
class TendonMaterial:
    stiffness: Optional[float] = None
    damping: Optional[float] = None
    pretension: Optional[float] = None
    color: Optional[str] = None  # hex "#rrggbb" or MJCF "r g b a"; None -> default


@dataclass
class TendonDefaults:
    stiffness: float = 300.0
    damping: float = 0.5
    pretension: float = 0.85
    width: float = 0.00075
    color: str = "0.8 0.2 0.2 1"  # legacy structural-tendon rgba


@dataclass
class TendonGenerator:
    """One of six σ-basis structural edges (semantic anchors)."""

    mobile_station: int
    mobile_face: Face
    fixed_station: int
    fixed_face: Face
    material: TendonMaterial = field(default_factory=TendonMaterial)


@dataclass
class TendonNet:
    defaults: TendonDefaults = field(default_factory=TendonDefaults)
    generators: Tuple[TendonGenerator, ...] = ()
    # per-edge overrides keyed by canonical edge id, e.g. "M1A-F4B"
    edge_overrides: Dict[str, TendonMaterial] = field(default_factory=dict)
    # per-emitted tendon overrides keyed by legacy MJCF name, e.g. "L_t1_to_t1_2"
    individual_overrides: Dict[str, TendonMaterial] = field(default_factory=dict)


@dataclass
class DiskSpec:
    radius: float = 0.015
    tilt_rad: float = 0.26
    # Disk-orbit PITCH about world Y (rad). tilt_rad rolls the orbit plane about
    # world X (throws the stroke sideways/lateral); pitch_rad pitches it about Y so
    # the crank pulls the wing UP/DOWN + fore-aft — a vertical x–z figure-8 stroke
    # instead of a horizontal row. Mirror-preserved across L/R (same sign on both).
    pitch_rad: float = 0.0
    kv: float = 0.25
    ctrlrange: Tuple[float, float] = (-10.0, 10.0)
    solreflimit: Tuple[float, float] = (0.02857142857142857, 1.0)
    outboard: float = 0.01875  # rim offset from disk center (m)
    # Crank THROW (m) = radius of the tendon pin from the hinge axis. Defaults to
    # the disk radius (pin on the rim). Setting it LARGER than `radius` mounts the
    # pin on a rigid crank ARM extending past the disk, so throw (the dominant
    # flap-swing lever) is decoupled from the visible disk size — a gear change
    # that buys more swing without growing the spinning disk. None ⇒ use radius.
    crank_throw: float | None = None
    # Parked crank phase (rad): angular position of the tendon pin at rest (disk
    # hinge q=0). The driven arc is phase-invariant in space, but the no-drive REST
    # is pulled toward the parked pin, so this sets WHERE in the arc the wing rests
    # — the knob to re-center a one-sided stroke. Default reproduces the legacy pin
    # at [radius, 0.2*radius, 0] (phase = atan2(0.2, 1)).
    crank_phase_rad: float = math.atan2(0.2, 1.0)
    foot_outset: float = 0.0  # offset pillar base away from sibling foot along arc (m)
    # Direct world-xy translation of the disk pillar base (m), applied to the
    # left disk before the y-mirror. dx shifts both disks the same world-x
    # (fore/aft); dy moves the left disk +y and the right disk −y (both medial
    # or both lateral together), preserving left/right mirror symmetry.
    disk_dx: float = 0.0
    disk_dy: float = 0.0
    pillar_height: float = 0.22
    pillar_radius: float = 0.004
    disk_half_height: float = 0.002


@dataclass
class MotorFoot:
    """One crank disk driving a mobile foot station."""

    station: int  # 1 or 4 (feet)
    attach_along_edge_rad: float = 0.0  # slide anchor along foot edge toward next station
    # Radial fraction of the anchor from the strut midpoint (0) to the foot
    # edge (1). At 1.0 the anchor sits on the edge (legacy). Below 1.0 the
    # anchor moves inward and a rigid rod bridges the edge to the anchor.
    attach_radius: float = 1.0
    # Per-disk crank throw (m) override. None ⇒ use the shared DiskSpec.crank_throw.
    # Lets the two cranks on a shoulder run different throws (asymmetric stroke).
    crank_throw: float | None = None
    # Per-disk pillar outset override. None ⇒ use the shared DiskSpec.foot_outset.
    foot_outset: float | None = None
    # Per-motor tendon spring rest length override. None ⇒ use Motors.tendon_rest_length.
    tendon_rest_length: float | None = None


@dataclass
class Motors:
    disk: DiskSpec = field(default_factory=DiskSpec)
    feet: Tuple[MotorFoot, MotorFoot] = (
        MotorFoot(station=1),
        MotorFoot(station=4),
    )
    # Shared motor-tendon spring rest length. None uses each tendon's measured
    # rest-pose length; explicit values make crank geometry/phase set preload.
    tendon_rest_length: Optional[float] = None
    tendon_stiffness: float = 500.0
    tendon_damping: float = 1.0
    # Phase lag (rad): initial crank-angle offset of the t4 crank vs t1. The two
    # cranks always counter-rotate (gear -1); this is an added timing offset, held
    # under the shared velocity command (see crank_sync.apply_crank_phase_lag).
    crank_phase_lag_rad: float = 0.0
    motor_direction: float = 1.0
    control_file: str = "control.tg"


@dataclass
class WingPitch:
    upstroke_stiffness: float = 0.5
    downstroke_stiffness: float = 0.5
    damping: float = 0.0005
    initial_rad: float = 0.35
    range_rad: Tuple[float, float] = (-1.0, 1.0)
    armature: float = 1e-5
    upstroke_pitch_rad: float = math.radians(15)
    downstroke_pitch_rad: float = math.radians(15)


@dataclass
class WingAero:
    blades: int = 4
    force_scale: float = 1_000_000.0
    include_body_torque: bool = False


@dataclass
class WristTensegrity:
    """A second, smaller tensegrity joint in place of the wrist hinge.

    Same topology as the shoulder — two arc struts held apart by the sigma-expanded
    tendon net, reusing the shoulder's six generators — scaled down and given a
    wrist's proportions. The proximal strut is rigid on the ulna; the distal one
    carries the manus.

    That distal strut has to be a *free* body, and MuJoCo only allows a free joint
    at the top level, so it is emitted as a worldbody child and the manus leaves
    the kinematic chain with it. Nothing but tendons then connects the hand to the
    forearm — which is exactly what makes this a tensegrity joint rather than a
    compliant hinge, and the same arrangement the humerus already has at the
    shoulder.

    ``scale`` is relative to the shoulder archetype's arc. ``stations_deg`` is a
    shallower arc than the shoulder's: a wrist wants less range and more stiffness
    out of the wing plane than a shoulder does.
    """

    scale: float = 0.42
    stations_deg: Tuple[float, float, float, float] = (28.0, 74.0, 106.0, 152.0)
    #: Distance between the two struts' centres along the span (m). The ulna's
    #: strut sits on the forearm's tip and the manus' one this far outboard,
    #: where the hand's bone begins — so this is the whole joint's length.
    gap: float = 0.014
    width: float = 0.0026  # A<->B face separation (m)
    stiffness: float = 400.0
    damping: float = 0.6
    pretension: float = 0.88
    tendon_width: float = 0.0004
    arc_rod: RodSpec = field(default_factory=lambda: RodSpec(0.0013, 2.0e-5))
    rung_rod: RodSpec = field(default_factory=lambda: RodSpec(0.0009, 1.0e-5))


@dataclass
class WingCables:
    """Cables from each wing bone down to a crank disk — the wing as a tensegrity.

    One cable per bone: an anchor on the humerus, the ulna and the manus, each
    running in to a pin on a crank disk. The pin is off the disk's hinge axis, so
    the cable's length is a function of crank angle and the motors drive the
    wing's shape directly, the same way they already drive the shoulder. Nothing
    routes bone-to-bone; every cable ends on a motor.

    ``disks`` names which disk each of the three cables goes to, in bone order.
    ``pin_radius`` and ``pin_phase`` place the pin on that disk — radius zero puts
    it on the hinge axis, where the cable no longer changes length and the wing
    stops being driven at all.
    """

    humerus_frac: float = 0.55  # fraction along each bone, from its proximal end
    ulna_frac: float = 0.50
    manus_frac: float = 0.45
    offset: float = 0.0  # chordwise stand-off of the bone anchor (m)
    disks: Tuple[str, str, str] = ("t1", "t4", "t4")
    pin_radius: float = 0.012  # m from the disk hinge axis
    pin_phase: float = math.radians(180.0)
    stiffness: float = 120.0
    damping: float = 0.4
    width: float = 0.0005
    #: Rest-length fractions of each cable's compiled length, in bone order.
    extend: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    contract: Tuple[float, float, float] = (0.92, 0.92, 0.92)


@dataclass
class WingBones:
    """Three-linkage avian wing (humerus / ulna / manus) in place of the flat plate.

    Bone frame: +z = span (shoulder -> wingtip), +x = chord (toward the trailing
    edge), +y = wing-plane normal. The humerus is a rigid child of the mobile
    strut — the tensegrity shoulder IS its joint, exactly as for the plate wing —
    plus one twist hinge about the span for feathering. The elbow and wrist are
    ordinary hinges about the wing normal, so the wing folds in its own plane.

    ``mount_euler_rad`` reorients that chain on the strut. It is load-bearing: the
    strut's rocking is a compound whirl, and only a narrow band of mount
    orientations turns it into a dominant up/down flap instead of the plate wing's
    fore-aft hummingbird sweep. See ``design/eagle.tg`` for how the default was found.
    """

    mount_euler_rad: Tuple[float, float, float] = (
        math.radians(175.0),
        math.radians(143.0),
        math.radians(-15.0),
    )
    humerus: float = 0.045
    ulna: float = 0.060
    manus: float = 0.075
    chord: float = 0.055  # secondary-feather length (wing chord at the forearm)
    primary: float = 0.062  # primary-feather length at the wingtip
    mass: float = 1.0e-4  # per bone
    elbow_range_rad: Tuple[float, float] = (-1.25, 0.09)
    wrist_range_rad: Tuple[float, float] = (-1.20, 0.09)
    twist_range_rad: Tuple[float, float] = (-0.60, 0.60)
    # Position-servo stiffness for the three driven hinges. Damping is derived
    # critical (2*sqrt(kp*armature)) so the fold can never ring the tensegrity.
    servo: float = 0.1
    damping: float = 2.0e-4
    armature: float = 2.0e-5
    # None -> the wrist is a plain hinge. Set -> a scaled-down tensegrity joint,
    # and the wrist hinge and its servo disappear.
    wrist: Optional[WristTensegrity] = None
    # None -> no wing cables. Set -> the wing is itself a tensegrity.
    cables: Optional[WingCables] = None

    @property
    def span(self) -> float:
        """Shoulder to wingtip, feathers included."""
        return self.humerus + self.ulna + self.manus + self.primary

    def mount_quat(self) -> Tuple[float, float, float, float]:
        """Intrinsic x-y-z Euler (MJCF ``eulerseq`` default) as a w-first quaternion."""
        half = [0.5 * a for a in self.mount_euler_rad]
        q = (1.0, 0.0, 0.0, 0.0)
        for axis, a in enumerate(half):
            r = [math.cos(a), 0.0, 0.0, 0.0]
            r[axis + 1] = math.sin(a)
            w1, x1, y1, z1 = q
            w2, x2, y2, z2 = r
            q = (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        return q


@dataclass
class Wing:
    rod_length: float = 0.018
    mount_arm: float = 0.055
    mesh_file: str = "wing.stl"
    pitch: WingPitch = field(default_factory=WingPitch)
    aero: WingAero = field(default_factory=WingAero)
    # None -> the legacy flat plate + aero blade sites (hummingbird).
    bones: Optional[WingBones] = None


@dataclass
class World:
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.000375
    # MuJoCo integrator. RK4 has a far larger explicit-stability region than the
    # default Euler, so it tolerates realistically stiff tendons at this timestep
    # (Euler caps structural stiffness ~30 N/m before it diverges — a numerical,
    # not physical, limit). One of: Euler, RK4, implicit, implicitfast.
    integrator: str = "Euler"
    scale: float = 1.0
    meshdir: str = "assets"


@dataclass
class Morphology:
    """Complete machine description — parsed from ``*.tg``, compiled to MJCF."""

    name: str = "bi-shoulder"
    world: World = field(default_factory=World)
    arc: StrutArc = field(default_factory=StrutArc)
    mobile: StrutInstance = field(
        default_factory=lambda: StrutInstance(role=StrutRole.MOBILE, plane=ArcPlane.XZ)
    )
    fixed: StrutInstance = field(
        default_factory=lambda: StrutInstance(role=StrutRole.FIXED, plane=ArcPlane.YZ)
    )
    arrange: Arrange = field(default_factory=Arrange)
    placement: Placement = field(default_factory=Placement)
    tendons: TendonNet = field(default_factory=TendonNet)
    motors: Motors = field(default_factory=Motors)
    wing: Wing = field(default_factory=Wing)

    def validate(self) -> None:
        if len(self.arc.stations_deg) != 4:
            raise ValueError("arc.stations_deg must have exactly 4 angles")
        if len(self.tendons.generators) != 6:
            raise ValueError("tendons.generators must have exactly 6 σ-basis edges")
