"""The eagle's MJCF: the three-bone wing, feather vanes, wrist tensegrity, wing cables, bone servos.

emit_mjcf calls in here at four points, only for a design with a ``bones:`` wing
(bird = eagle): the bone wing in place of the plate wing, the cable pin on each
crank disk, the wrist net and cables after each shoulder's tendons, and the
bone servos after the crank motors.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from morphology.compiled import CompiledMorphology, CompiledShoulder
from morphology.geometry import mirror_quat
from morphology.mjcf_parts import (
    STRUCTURAL_TENDON_COLORS,
    hex_to_rgba,
    strut_body,
    structural_tendon,
    vec2,
    vec3,
    vec4,
)
from morphology.tendon_net import anchor_node_name

#: Spacing between successive feather vanes along the wing normal (m).
_SHINGLE = 0.0005


def _emit_vane(bone, name: str, root_z: float, rake: float, length: float,
               lift: float, width: float, mass: float, rgba: str, f) -> None:
    """Emit one vane as a chain of sprung links, so the shaft curves under load.

    Every hinge turns about its link's local x, perpendicular to both the shaft
    (local z) and the wing normal (local y, which the rake rotation leaves alone),
    so the vane bends out of the wing plane rather than sweeping in it.

    ``f.stiffness`` is the whole vane's, so each of the N nodes gets N times it —
    the discretisation of a uniform beam. Damping and armature are derived from
    each link's own inertia rather than given absolutely, because that inertia
    falls as 1/N^3 and a fixed number stops meaning anything the moment the
    segment count moves.

    Both class defaults must be overridden here, not inherited: the bones sit
    under ``childclass="shoulder-joint"``, whose frictionloss of 0.1 N m is ~1e5
    times anything a milligram vane exerts and whose armature of 0.005 is orders
    over a link's inertia. Either one alone welds the feather solid.
    """
    n = f.segments
    seg, seg_mass = length / n, mass / n
    stiffness = n * f.stiffness
    # A thin box about the end of its own long axis. The width and thickness
    # terms are ~1e-3 of this on any vane here, so they are left out.
    link_inertia = seg_mass * seg * seg / 3.0
    armature = f.armature * link_inertia
    damping = 2.0 * f.zeta * math.sqrt(stiffness * (link_inertia + armature))

    parent, pos, euler = bone, f"0 0 {root_z}", f"0 {rake} 0"
    for j in range(n):
        attrs = {"pos": pos} if euler is None else {"pos": pos, "euler": euler}
        link = ET.SubElement(parent, "body", name=f"{name}_{j + 1:02d}_quill", **attrs)
        ET.SubElement(
            link,
            "joint",
            name=f"{name}_{j + 1:02d}_bend",
            type="hinge",
            axis="1 0 0",
            limited="true",
            range=vec2(f.range_rad),
            stiffness=str(stiffness),
            damping=str(damping),
            armature=str(armature),
            frictionloss="0",
        )
        ET.SubElement(
            link,
            "geom",
            name=f"{name}_{j + 1:02d}",
            type="box",
            pos=f"0 {lift} {0.5 * seg}",
            size=f"{0.5 * width} 0.0004 {0.5 * seg}",
            mass=str(seg_mass),
            rgba=rgba,
            contype="0",
            conaffinity="0",
            group="2",
        )
        parent, pos, euler = link, f"0 0 {seg}", None


def _emit_cable_anchors(prefix, b, humerus, ulna) -> None:
    """One cable anchor on the humerus and one on the ulna, plus the two guides.

    The manus anchor is emitted with the hand, since the hand is a free body and
    does not exist yet at this point in the tree.

    The guides are the posts the cables ride over on their way in: one at the
    distal end of the humerus (the elbow) and one at the distal end of the ulna
    (the wrist), each standing off the bone axis. That stand-off is the whole
    point of them — routed through the joint centre a cable has no moment arm
    there and cannot fold that joint -- so it is a bird's retinaculum, not a
    decoration.
    """
    c = b.cables
    for body, name, length, frac in (
        (humerus, "humerus", b.humerus, c.humerus_frac),
        (ulna, "ulna", b.ulna, c.ulna_frac),
    ):
        _cable_anchor_site(body, prefix, name, c.offset, frac * length)
    for body, name, span, offset in (
        (humerus, "guide_humerus", b.humerus, c.humerus_guide),
        (ulna, "guide_ulna", b.ulna, c.ulna_guide),
    ):
        _cable_anchor_site(body, prefix, name, offset, span)
        ET.SubElement(
            body,
            "geom",
            name=f"{prefix}_cable_{name}_post",
            type="capsule",
            fromto=f"0 0 {span} {offset} 0 {span}",
            size="0.0011",
            mass="0",
            rgba="0.15 0.85 0.4 1",
            contype="0",
            conaffinity="0",
            group="2",
        )


def _cable_anchor_site(body, prefix, name, offset, span) -> None:
    ET.SubElement(
        body,
        "site",
        name=f"{prefix}_cable_{name}",
        type="sphere",
        pos=f"{offset} 0 {span}",
        size="0.0024",
        rgba="0.15 0.85 0.4 1",
        group="3",
    )


BONE_RGBA = "0.93 0.90 0.82 1"
FEATHER_RGBA = ("0.30 0.22 0.16 1", "0.20 0.15 0.11 1")  # coverts, flight feathers


def _emit_wrist_tensegrity(worldbody, ulna, shoulder: CompiledShoulder) -> ET.Element:
    """Emit the wrist's strut pair and return the body the manus should hang on.

    Proximal strut is rigid on the ulna. Distal strut is a worldbody child with a
    free joint — MuJoCo permits those only at the top level, which is precisely
    why this is a tensegrity joint and not a compliant hinge: nothing structural
    reaches the hand, only tendons.
    """
    wrist = shoulder.wrist
    prefix = shoulder.side
    strut_body(ulna, f"{prefix}_wrist_proximal", wrist.proximal_geom,
               wrist.proximal_pos, wrist.proximal_quat, free=False)
    return strut_body(worldbody, f"{prefix}_wrist_distal", wrist.distal_geom,
                      wrist.distal_pos, wrist.distal_quat, free=True)


def emit_bone_wing(parent: ET.Element, shoulder: CompiledShoulder, morph, worldbody: ET.Element) -> None:
    """Humerus / ulna / manus chain hanging off the mobile strut.

    The humerus is a rigid child of the strut — the tensegrity network is its
    shoulder joint, same as the plate wing — with one twist hinge about the span
    for feathering. Elbow and wrist are plain hinges about the wing-plane normal,
    so the wing folds inside its own plane the way a bird's does.

    Bone-local frame: +z spans out to the tip, +x is chord (toward the trailing
    edge, so -x is cranial), +y is the wing normal. Elbow and wrist hinge about
    opposite normals so they counter-rotate: in the reference plate the elbow is
    the caudal apex and the wrist the cranial one, and same-sign hinges spiral
    the chain into a C instead of folding it into the avian Z.

    The R side is the exact y-mirror of L: every offset here is y-free, so only
    the mount quaternion mirrors and the twist axis flips sign (a_R = -M a_L,
    which leaves the two wing-normal hinges unchanged — the same rule the plate
    wing's pitch hinge follows).
    """
    b = morph.wing.bones
    prefix = shoulder.side
    quat = b.mount_quat()
    if prefix == "R":
        quat = mirror_quat(quat)
    twist_axis = "0 0 1" if prefix == "L" else "0 0 -1"
    # Feather vanes are the one thing here with an offset off the y=0 plane, so
    # they are also the one thing that has to mirror explicitly (p_R = M p_L).
    y_sign = 1.0 if prefix == "L" else -1.0

    def hinge(body, name, axis, rng):
        ET.SubElement(
            body,
            "joint",
            name=f"{prefix}_{name}",
            type="hinge",
            axis=axis,
            limited="true",
            range=vec2(rng),
            damping=str(b.damping),
            armature=str(b.armature),
            # The shoulder-joint class default frictionloss=0.1 N m is ~1000x the
            # torque these gram-scale bones see; it would stiction-lock the fold.
            frictionloss="0",
        )

    def bone(body, name, z0, z1, radius, mass, dx=0.0):
        ET.SubElement(
            body,
            "geom",
            name=f"{prefix}_{name}",
            type="capsule",
            fromto=f"{dx} 0 {z0} {dx} 0 {z1}",
            size=str(radius),
            mass=str(mass),
            rgba=BONE_RGBA,
            contype="0",
            conaffinity="0",
            group="2",
        )

    def feathers(body, tag, count, z0, z1, len0, len1, rake0, rake1, width, rgba, group=0):
        """Fan ``count`` flat vanes off a bone, rooted at successive span stations.

        ``rake`` is the angle of the shaft off the span axis toward the trailing
        edge: ~24 deg on the outermost primary (it continues the wingtip) up to
        ~95 deg on a secondary (straight aft). Sweeping the rake across the manus
        is what gives an eagle its separated, fingered tip rather than one slab.
        Keep it under 90: past that the vane's own axis points back inboard, and
        the root feather reaches through the strut and across the midline.

        Vanes are shingled along the wing normal by ``_SHINGLE`` per index, the
        outer lying over the next one in, as a bird's do. Emitting them all at
        y=0 makes them exactly coplanar, which z-fights into streaks and a
        staircased trailing edge — the whole fan is only 0.8 mm thick.

        Returns the bone-local tip of the last vane.
        """
        # Each fan restarts its shingle, so offset the groups by a third of a step
        # to keep every vane on its own plane -- two vanes at the same y anywhere
        # on the wing z-fight wherever they overlap.
        base = _SHINGLE * group / 3.0
        tip = (0.0, y_sign * (base + _SHINGLE * max(count - 1, 0)), z1)
        for i in range(count):
            t = 0.0 if count == 1 else i / (count - 1)
            rake = rake0 + t * (rake1 - rake0)
            length = len0 + t * (len1 - len0)
            root_z = z0 + t * (z1 - z0)
            lift = y_sign * (base + _SHINGLE * i)
            # Rotating +z about +y by `rake` gives (sin, 0, cos): out along the
            # span and aft. The vane's long axis is its own local z.
            d = (math.sin(rake), 0.0, math.cos(rake))
            name = f"{prefix}_{tag}_{i + 1:02d}"
            if b.feathers is not None:
                # Each link is a body turned so its own +z runs out along the
                # shaft, which leaves `lift` as the geom's only offset — still
                # the shingle, and still one plane per feather.
                _emit_vane(body, name, root_z, rake, length, lift, width,
                           0.02 * b.mass, rgba, b.feathers)
            else:
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="box",
                    pos=f"{0.5 * length * d[0]} {lift} {root_z + 0.5 * length * d[2]}",
                    euler=f"0 {rake} 0",
                    size=f"{0.5 * width} 0.0004 {0.5 * length}",
                    mass=str(0.02 * b.mass),
                    rgba=rgba,
                    contype="0",
                    conaffinity="0",
                    group="2",
                )
            tip = (length * d[0], lift, root_z + length * d[2])
        return tip

    humerus = ET.SubElement(
        parent,
        "body",
        name=f"{prefix}_humerus",
        pos=vec3(shoulder.mobile_geom.midpoint),
        childclass="shoulder-joint",
        quat=vec4(quat),
    )
    hinge(humerus, "humerus_twist", twist_axis, b.twist_range_rad)
    bone(humerus, "humerus", 0.0, b.humerus, 0.0030, 0.40 * b.mass)
    ET.SubElement(
        humerus,
        "geom",
        name=f"{prefix}_humerus_head",
        type="sphere",
        pos="0 0 0",
        size="0.0042",
        mass=str(0.05 * b.mass),
        rgba=BONE_RGBA,
        contype="0",
        conaffinity="0",
        group="2",
    )
    # Tertials: the innermost flight feathers, filling the gap in to the body.
    # Rake stays under 90 deg — past it the vane points back inboard, and the
    # root one then reaches through the strut and across the midline.
    feathers(humerus, "tertial", 3, 0.46 * b.humerus, 0.98 * b.humerus,
             0.95 * b.chord, 1.0 * b.chord, math.radians(88), math.radians(80),
             0.36 * b.chord, FEATHER_RGBA[0], group=0)

    ulna = ET.SubElement(humerus, "body", name=f"{prefix}_ulna", pos=f"0 0 {b.humerus}")
    hinge(ulna, "elbow", "0 1 0", b.elbow_range_rad)
    bone(ulna, "ulna", 0.0, b.ulna, 0.0024, 0.35 * b.mass)
    # The radius: the thin second forearm bone in the reference figure. Visual
    # only, but it is what makes the forearm read as a forearm and not a stick.
    bone(ulna, "radius", 0.0, b.ulna, 0.0013, 0.05 * b.mass, dx=-0.0052)
    feathers(ulna, "secondary", 7, 0.06 * b.ulna, 0.96 * b.ulna,
             1.0 * b.chord, 0.84 * b.chord, math.radians(95), math.radians(78),
             0.30 * b.chord, FEATHER_RGBA[0], group=1)

    if b.cables is not None:
        _emit_cable_anchors(prefix, b, humerus, ulna)

    if b.wrist is None:
        manus = ET.SubElement(ulna, "body", name=f"{prefix}_manus", pos=f"0 0 {b.ulna}")
        hinge(manus, "wrist", "0 -1 0", b.wrist_range_rad)
    else:
        # Tensegrity wrist: the hand leaves the kinematic chain. The proximal
        # strut goes on the ulna and the free distal strut in the worldbody; the
        # manus rides the latter, joined to the forearm by tendons alone. No
        # wrist joint exists to hinge or to servo.
        manus = ET.SubElement(
            _emit_wrist_tensegrity(worldbody, ulna, shoulder),
            "body",
            name=f"{prefix}_manus",
            pos=vec3(shoulder.wrist.manus_pos),
            quat=vec4(shoulder.wrist.manus_quat),
            childclass="shoulder-joint",
        )
    bone(manus, "carpometacarpus", 0.0, 0.62 * b.manus, 0.0019, 0.14 * b.mass)
    bone(manus, "phalanx", 0.62 * b.manus, b.manus, 0.0013, 0.06 * b.mass)
    # Primaries: long, separated, raked progressively forward toward the tip —
    # the eagle's fingered wingtip. The outermost one carries the wingtip marker.
    # rake1 stays well off the span: at 16 deg the outermost primary carried only
    # 33 mm of chord and the hand converged to a spike. An eagle's is blunt and
    # deeply slotted, so the outer feather keeps ~40 mm of chord on it.
    tip = feathers(manus, "primary", 8, 0.05 * b.manus, b.manus,
                   0.72 * b.primary, 1.0 * b.primary, math.radians(72), math.radians(24),
                   0.20 * b.chord, FEATHER_RGBA[1], group=2)
    if b.cables is not None:
        _cable_anchor_site(manus, prefix, "manus", b.cables.offset,
                           b.cables.manus_frac * b.manus)

    # Wingtip marker: the reference point every kinematic check measures.
    ET.SubElement(
        manus,
        "site",
        name=f"{prefix}_wingtip",
        type="sphere",
        pos=vec3(tip),
        size="0.004",
        rgba="1 0.85 0.1 1",
        group="3",
    )



def add_cable_pin(disk_body: ET.Element, disk_name: str, cables) -> None:
    """The wing cables' own pin on a crank disk, clear of the motor tendon's."""
    # Off the hinge axis on purpose: at radius zero the cable length stops depending on
    # crank angle and the motor stops driving the wing's shape at all.
    #
    # Mirrored exactly as the motor tendon's rim is (compile.py, where the R
    # disks are built): the R disk body is tilted -tilt where L is +tilt, so a
    # y-mirror of a disk-local point flips its y and leaves x alone. Giving
    # both sides the same local pin puts the R one on the wrong side of its
    # disk — 19 mm off its mirror position, which leaves the right cables
    # 17 mm short and flaps the right wing out of phase with the left.
    y_mirror = -1.0 if disk_name.startswith("R_") else 1.0
    pin = np.array([
        cables.pin_radius * math.cos(cables.pin_phase),
        y_mirror * cables.pin_radius * math.sin(cables.pin_phase),
        0.0,
    ])
    ET.SubElement(
        disk_body,
        "geom",
        name=f"{disk_name}_cable_arm",
        type="capsule",
        fromto=f"0 0 0 {pin[0]} {pin[1]} {pin[2]}",
        size="0.0016",
        rgba="0.15 0.85 0.4 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        disk_body,
        "site",
        name=f"{disk_name}_cable_pin",
        type="sphere",
        pos=vec3(pin),
        size="0.0024",
        rgba="0.15 0.85 0.4 1",
        group="3",
    )


def _add_wing_cables(tendon_el, prefix: str, morph) -> None:
    """One cable per bone, each running from the bone in to a pin on a crank disk.

    That pin is off the disk's hinge axis, so every cable's length is a function
    of crank angle: the motors drive the wing's shape the same way they already
    drive the shoulder, instead of the wing being posed by a servo.

    They are emitted INERT — springlength as a wide range, which MuJoCo reads as
    "no force between these" — because a cable is only tensioned relative to a
    pose and the compiler has not got one. ``runtime.wing_cables.apply_tension``
    measures the compiled pose and writes the real rest length. A single-valued
    springlength here would be a two-sided spring against a placeholder number,
    which is ~110 N of nonsense shoving the wing off the rig.
    """
    c = morph.wing.bones.cables
    bones = (("humerus", c.disks[0]), ("ulna", c.disks[1]), ("manus", c.disks[2]))
    # Each cable rides in over the guide at the distal end of every bone inboard
    # of its own before it drops to the disk. Straight from the hand to the body
    # a cable hangs across open air and hauls the hand off the wrist cage; over
    # the forearm and the humerus it stays on the skeleton, and it gains a moment
    # arm at every joint it crosses instead of only at the shoulder.
    via = {
        "humerus": (),
        "ulna": ("guide_humerus",),
        "manus": ("guide_ulna", "guide_humerus"),
    }
    for bone, station in bones:
        spatial = ET.SubElement(
            tendon_el,
            "spatial",
            name=f"{prefix}_cable_{bone}",
            stiffness=str(c.stiffness),
            damping=str(c.damping),
            width=str(c.width),
            rgba="0.15 0.85 0.4 1",
            springlength="0 1e9",
        )
        ET.SubElement(spatial, "site", site=f"{prefix}_cable_{bone}")
        for guide in via[bone]:
            ET.SubElement(spatial, "site", site=f"{prefix}_cable_{guide}")
        ET.SubElement(spatial, "site", site=f"{prefix}_{station}_cable_pin")


def add_tendons(tendon_el, shoulder: CompiledShoulder, morph, color_structural_tendons: bool) -> None:
    """A shoulder's wrist net (if it has a tensegrity wrist) then its wing cables."""
    if shoulder.wrist is not None:
        _add_wrist_tendons(tendon_el, shoulder, morph, color_structural_tendons)
    if morph.wing.bones.cables is not None:
        _add_wing_cables(tendon_el, shoulder.side, morph)


def _add_wrist_tendons(tendon_el, shoulder: CompiledShoulder, morph,
                       color_structural_tendons: bool) -> None:
    """The wrist's own sigma-expanded net, same topology as the shoulder's.

    Materials come from the ``wrist:`` block rather than the shoulder's tendon
    defaults: this net is an order of magnitude smaller and holds the hand, not
    the whole wing, so it wants its own stiffness and pretension.
    """
    w = morph.wing.bones.wrist
    prefix = shoulder.side
    wrist = shoulder.wrist
    for index, (edge, length) in enumerate(zip(wrist.structural_edges, wrist.tendon_lengths)):
        m_node = anchor_node_name(edge.mobile)
        f_node = anchor_node_name(edge.fixed)
        rgba = hex_to_rgba(STRUCTURAL_TENDON_COLORS[index]) if color_structural_tendons else "0.25 0.55 0.9 1"
        structural_tendon(tendon_el, f"{prefix}_wrist_{m_node}_to_{f_node}", length, w.stiffness, w.damping,
                          w.pretension, rgba, f"{prefix}_wrist_distal_{m_node}",
                          f"{prefix}_wrist_proximal_{f_node}", width=w.tendon_width)


def add_bone_servos(act: ET.Element, compiled: CompiledMorphology) -> None:
    """Position servos for the avian wing's elbow, wrist and humerus twist.

    The tensegrity shoulder supplies the flap; these three hold the fold and the
    feathering against it. Servos rather than springs because the fold has to
    track the stroke phase, and because a stiff, critically damped setpoint is
    the cheapest way to keep a long wing from ringing the tendon network.
    """
    b = compiled.morph.wing.bones
    # Critically damped: overshoot on these hinges shakes the whole tendon
    # network, and a folding wing has no reason to ring.
    kv = 2.0 * math.sqrt(max(b.servo, 0.0) * max(b.armature, 1e-12))
    joints = [
        ("humerus_twist", b.twist_range_rad),
        # 1 DOF, deliberately: the elbow is the one plain hinge left in the wing.
        ("elbow", b.elbow_range_rad),
    ]
    if b.wrist is None:
        joints.append(("wrist", b.wrist_range_rad))
    for shoulder in compiled.shoulders:
        for name, rng in joints:
            ET.SubElement(
                act,
                "position",
                name=f"{shoulder.side}_{name}_servo",
                joint=f"{shoulder.side}_{name}",
                kp=f"{b.servo:.6g}",
                kv=f"{kv:.6g}",
                ctrlrange=vec2(rng),
            )
