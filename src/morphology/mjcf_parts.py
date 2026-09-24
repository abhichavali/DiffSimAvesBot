"""MJCF building blocks shared by the hummer and eagle emitters.

Number formatting, strut bodies (capsule rods plus node sites), structural
tendons and the tendon colour palette. Both emit_mjcf and emit_eagle build on
these; keeping them here keeps the two emitters from importing each other.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from morphology.geometry import StrutGeometry

#: Per-tendon debug palette (matplotlib tab10 + two), indexed by structural edge.
STRUCTURAL_TENDON_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
)


def vec2(v) -> str:
    return f"{v[0]} {v[1]}"


def vec3(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    return f"{v[0]} {v[1]} {v[2]}"


def vec4(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    return " ".join(str(x) for x in v)


def vec6(p1, p2) -> str:
    return f"{p1[0]} {p1[1]} {p1[2]} {p2[0]} {p2[1]} {p2[2]}"


def hex_to_rgba(color: str) -> str:
    color = color.lstrip("#")
    return " ".join(str(int(color[i : i + 2], 16) / 255.0) for i in (0, 2, 4)) + " 1"


def color_to_rgba(color: str) -> str:
    """Normalize a tendon color to MJCF rgba. Accepts 'r g b [a]' or 'rrggbb' hex.

    ('#' can't be used in .tg -- it's a comment marker -- so hex has no leading #.)
    """
    text = str(color).strip()
    if " " in text:  # already space-separated rgba (pad alpha if omitted)
        parts = text.split()
        return " ".join(parts) if len(parts) >= 4 else " ".join(parts + ["1"])
    return hex_to_rgba(text)  # 6-digit hex


def capsule(parent, name, p1, p2, radius, mass) -> None:
    # contype=2/conaffinity=1 is the standard self-collision filter: strut-vs-strut
    # is masked (2&1 == 0 both ways) so the interleaved tensegrity capsules don't
    # spawn hundreds of spurious MJX candidate pairs, while strut-vs-floor
    # (contype 1) still collides. The struts never actually touch in this rig, so
    # this only removes wasted broad-phase work (and gradient noise).
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="capsule",
        fromto=vec6(p1, p2),
        size=str(radius),
        mass=str(mass),
        rgba="0.5 0.5 0.5 1",
        contype="2",
        conaffinity="1",
    )


def site(parent, body_name, node_name, pos, quat=None) -> None:
    attrs = {
        "name": f"{body_name}_{node_name}",
        "pos": vec3(pos),
        "size": "0.002",
        "rgba": "0.2 0.8 0.2 1",
        "group": "3",
    }
    if quat is not None:
        attrs["quat"] = vec4(quat)
    ET.SubElement(parent, "site", **attrs)


def strut_body(
    parent: ET.Element,
    body_name: str,
    geom: StrutGeometry,
    pos: np.ndarray,
    quat: np.ndarray,
    free: bool,
) -> ET.Element:
    """A strut body: rung rods, arc rods, any motor bridging rods, and a site per node."""
    body = ET.SubElement(
        parent,
        "body",
        name=body_name,
        pos=vec3(pos),
        childclass="shoulder-joint",
        quat=vec4(quat),
    )
    if free:
        ET.SubElement(body, "freejoint", name=f"{body_name}_freejoint")

    arc_r, arc_m = geom.arc_rod
    rung_r, rung_m = geom.rung_rod

    for i, (a, b) in enumerate(geom.rung_rod_pairs):
        capsule(body, f"{body_name}_rod_{i}", geom.nodes[a], geom.nodes[b], rung_r, rung_m)

    base = len(geom.rung_rod_pairs)
    for i, (a, b) in enumerate(geom.arc_rod_pairs):
        capsule(body, f"{body_name}_rod_{base + i}", geom.nodes[a], geom.nodes[b], arc_r, arc_m)

    # Rigid bridging rods from a motor foot edge to an inset motor anchor.
    # Mass tracks the arc-rod linear density so the inset strut is physical.
    if geom.motor_rod_pairs:
        ref_a, ref_b = geom.arc_rod_pairs[0]
        ref_len = float(np.linalg.norm(geom.nodes[ref_a] - geom.nodes[ref_b]))
        lin_density = arc_m / ref_len if ref_len > 1e-9 else 0.0
        for i, (a, b) in enumerate(geom.motor_rod_pairs):
            length = float(np.linalg.norm(geom.nodes[a] - geom.nodes[b]))
            capsule(body, f"{body_name}_motor_rod_{i}", geom.nodes[a], geom.nodes[b], arc_r, lin_density * length)

    attachment_nodes = {n for pair in geom.rung_rod_pairs + geom.arc_rod_pairs for n in pair}
    attachment_nodes.update(n for n in geom.nodes if n.startswith("motor_"))
    for node_name, node_pos in geom.nodes.items():
        is_centreline = node_name.startswith("c") and node_name[1:].isdigit()
        if is_centreline or node_name in attachment_nodes:
            site(body, body_name, node_name, node_pos, geom.site_quats.get(node_name))
    return body


def structural_tendon(tendon_el, name, length, stiffness, damping, pretension, rgba,
                      mobile_site, fixed_site, width=None) -> None:
    """One structural tendon: range [0, posed length], spring rest = posed length * pretension
    (pretension < 1 pre-tensions it)."""
    attrs = {
        "name": name,
        "class": "string",
        "range": f"0.0 {length}",
        "stiffness": str(stiffness),
        "damping": str(damping),
        "springlength": str(length * pretension),
    }
    if width is not None:
        attrs["width"] = str(width)
    attrs["rgba"] = rgba
    spatial = ET.SubElement(tendon_el, "spatial", **attrs)
    ET.SubElement(spatial, "site", site=mobile_site)
    ET.SubElement(spatial, "site", site=fixed_site)
