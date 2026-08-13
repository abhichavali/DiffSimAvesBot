"""Turn the grounded bench rig into a free-flying airframe (+ demo obstacles).

The compiled scene is a *test stand*: the fixed struts and the motor pillars are
welded to the world, and only the mobile struts (wings) float on freejoints,
held by the tendon net. To "fly", everything that was world-welded is collected
under ONE new root body carrying a single freejoint. The mobile struts stay
where they are and remain coupled to the airframe only through the tendons — the
tensegrity shoulder joint — so the whole robot becomes one self-contained free
body with the flapping mechanism onboard.

Positions are absolute in `<worldbody>`; the airframe root sits at the origin
with identity orientation, so reparenting is a pure move — no geometry shifts.

Pure string->string XML surgery (ElementTree only); no MuJoCo import here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

AIRFRAME_BODY = "airframe"
AIRFRAME_FREEJOINT = "airframe_freejoint"

# The pillars were ground anchors (0.1 kg each) — irrelevant when welded, absurd
# for a flapping micro-robot once it must fly. Lighten them so the airframe (a
# few tens of grams) actually reacts to aero and impacts.
# ponytail: single knob; raise for a more sluggish/stable airframe, lower for a
# twitchier one. Weight is cancelled at runtime, so this only sets reactivity.
PILLAR_MASS = 0.01
PILLAR_INERTIA = "1e-5 1e-5 1e-5 0 0 0"


def to_airframe(xml: str, *, pillar_mass: float = PILLAR_MASS) -> str:
    """Wrap all world-welded bodies into one free `airframe` body."""
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene has no <worldbody>")

    # A direct child <body> with no top-level joint is welded to the world.
    welded = [
        b
        for b in worldbody.findall("body")
        if b.find("freejoint") is None and b.find("joint") is None
    ]
    if not welded:
        raise ValueError("no world-welded bodies found to build an airframe")

    airframe = ET.Element("body", name=AIRFRAME_BODY, pos="0 0 0", quat="1 0 0 0")
    ET.SubElement(airframe, "freejoint", name=AIRFRAME_FREEJOINT)
    for body in welded:
        worldbody.remove(body)
        _lighten_pillar(body, pillar_mass)
        airframe.append(body)
    worldbody.append(airframe)
    return _tostr(root)


def _lighten_pillar(body: ET.Element, mass: float) -> None:
    if not body.get("name", "").endswith("_pillar"):
        return
    inertial = body.find("inertial")
    if inertial is not None:
        inertial.set("mass", str(mass))
        inertial.set("fullinertia", PILLAR_INERTIA)


def shrink_pillar_visual(xml: str, *, stub: float = 0.05) -> str:
    """Trim the motor-pillar geoms to a short stub just below the disk.

    In the bench rig the pillars are 0.22 m posts down to the floor; free-flying
    they dangle like stilts. The disk body sits at the pillar top and the pillar
    MASS lives in a separate <inertial>, so shortening only the geom is purely
    cosmetic — it keeps the crank standoff (and every tendon length) intact.
    """
    root = ET.fromstring(xml)
    for body in root.iter("body"):
        if not body.get("name", "").endswith("_pillar"):
            continue
        geom = body.find("geom")
        if geom is None or geom.get("type") != "cylinder":
            continue
        radius, halfh = (float(v) for v in geom.get("size").split())
        top = 2.0 * halfh  # disk height (geom is centered at halfh, spans 0..2*halfh)
        geom.set("size", f"{radius} {stub / 2.0}")
        geom.set("pos", f"0 0 {top - stub / 2.0}")
    return _tostr(root)


def add_obstacle(
    xml: str,
    pos,
    *,
    radius: float = 0.02,
    z_lo: float = 0.10,
    z_hi: float = 0.48,
    rgba: str = "0.85 0.2 0.2 1",
) -> str:
    """Add a fixed vertical pole into the flight corridor.

    A pole (not a ball) so the collidable surface spans the whole flight-height
    band — a thin, fast, flapping wing can't miss it in z the way it slips past a
    small sphere. contype/conaffinity 1 collides with the wings and struts (which
    carry bit 1); a ball cap on top just makes it read as a deliberate object.
    ``pos`` is (x, y, _); the third component is ignored (the pole spans z).
    """
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    x, y, _ = pos
    obstacle = ET.SubElement(worldbody, "body", name="obstacle", pos=f"{x} {y} 0")
    ET.SubElement(
        obstacle,
        "geom",
        name="obstacle_pole",
        type="capsule",
        fromto=f"0 0 {z_lo} 0 0 {z_hi}",
        size=str(radius),
        rgba=rgba,
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        obstacle,
        "geom",
        name="obstacle_cap",
        type="sphere",
        pos=f"0 0 {z_hi}",
        size=str(radius * 1.8),
        rgba=rgba,
        contype="0",
        conaffinity="0",
        group="2",
    )
    return _tostr(root)


CEILING_THICKNESS = 0.012


def add_ceiling(
    xml: str,
    *,
    half_x: float = 6.0,
    half_y: float = 3.0,      # wide enough that its edge isn't cutting across the shot
    thickness: float = CEILING_THICKNESS,
    rgba: str = "1 1 1 1",
    rgb1: str = "0.50 0.53 0.59",
    rgb2: str = "0.36 0.39 0.45",
    markrgb: str = "0.85 0.86 0.9",
    shadows: bool = True,
    park=(0.0, 0.0, 5.0),
) -> str:
    """A movable overhead slab the wingtip can strike at the top of the upstroke.

    Mocap (kinematic, infinite mass) rather than welded so the renderer can flap
    once, measure where the tip actually peaks, and drop the ceiling into that
    arc — a hardcoded height quietly stops being a collision the moment the
    design, the flap amplitude or ``--lift`` changes. Semi-transparent so it
    never hides the bird. Starts parked well overhead.

    ``half_y`` and ``thickness`` are what make it READ as a ceiling rather than
    as a horizon line. A slab wide enough to run off both sides of the frame has
    no visible edges, so from a camera that is only a little below it there is
    nothing to say it is overhead — which is exactly how the first ceiling clip
    failed. Narrow it to a soffit (both long edges in shot, converging with
    perspective) and give it enough thickness that those edges are a visible
    band, and the same camera reads it instantly.

    contype 4 / conaffinity 0 keys off conaffinity 4, the ONE bit unique to the
    wing capsules. The obvious mirror of ``add_mocap_obstacle`` (contype 4 /
    conaffinity 2) would also catch every tensegrity strut — they carry contype 2
    as well — and the bird would hang off the ceiling by its shoulder instead of
    clipping a wingtip.
    """
    root = ET.fromstring(xml)
    # Checkered like the floor, at the same texture scale. A flat untextured slab
    # is featureless from below, so the wing meeting it reads as nothing at all —
    # you cannot see where the surface is, let alone that something hit it.
    #
    # The DEFAULT greys are a trap for anything seen from below. The only
    # directional light in the scene points straight down, so a ceiling's
    # underside is lit by the headlight and the ambient term alone — call it 0.9
    # of albedo — and a mid-grey texture lands somewhere around 0.4, which
    # renders as a murky brown ceiling. Pass bright rgb1/rgb2 for a slab that is
    # meant to be looked UP at.
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="2d", name="ceiling_tex", builtin="checker",
                  mark="edge", rgb1=rgb1, rgb2=rgb2,
                  markrgb=markrgb, width="300", height="300")
    ET.SubElement(asset, "material", name="ceiling_mat", texture="ceiling_tex",
                  texuniform="true", texrepeat="5 5", rgba=rgba, reflectance="0.05")
    worldbody = root.find("worldbody")
    px, py, pz = park
    body = ET.SubElement(worldbody, "body", name="ceiling", mocap="true",
                         pos=f"{px} {py} {pz}")
    ET.SubElement(
        body, "geom", name="ceiling_slab", type="box",
        size=f"{half_x} {half_y} {thickness / 2.0}",
        material="ceiling_mat", contype="4", conaffinity="0",
        solref=RIGID_SOLREF, solimp=RIGID_SOLIMP,
    )
    if not shadows:
        # ponytail: shadow casting is a LIGHT attribute in MuJoCo, not a geom
        # one, so this is all-or-nothing for the scene rather than just for the
        # ceiling. Worth it for a slab this size: it runs 12 m down the corridor,
        # far past the extent of the shadow map, and what lands on the floor is
        # not a ceiling's shadow but a hard-edged dark quadrilateral that reads as
        # a rendering glitch. Nothing else in this scene casts a shadow worth
        # keeping — the bird flies a metre up. Upgrade path if a shadow is ever
        # wanted (it would be a strong overhead cue): widen the scene's shadow map
        # with <visual><map shadowclip/shadowscale> and turn this back on.
        for light in root.iter("light"):
            light.set("castshadow", "false")
    return _tostr(root)


# Wing planform in the wing body frame, measured from the wing mesh bounds
# (y = chord, z = span; the plate is only ~3.4 mm thick in x), plus a small margin.
# The self-check below asserts these still cover the mesh.
WING_PLATE_HALF = (0.004, 0.039, 0.118)
WING_PLATE_CENTRE = (0.0, 0.0198, 0.1224)   # y is mirrored on the R wing

# Contact stiffness for the wing plate and anything it strikes. MuJoCo's default
# solref (0.02 s) is soft enough that a wing driven up into a ceiling sinks
# visibly into it before the constraint bites. 0.002 s is still ~15x the 0.13 ms
# timestep, which is well inside the stable range.
RIGID_SOLREF = "0.002 1"
RIGID_SOLIMP = "0.95 0.99 0.001 0.5 2"


def enable_wing_collision(xml: str) -> str:
    """Give each wing a collision box covering its PLANFORM so it can hit things.

    This used to be a thin capsule down the span — 6 mm of radius about the span
    axis, while the wing mesh reaches 57 mm aft of it. So the entire chord had no
    collision geometry at all: the rod on the axis would stop against a ceiling
    while the visible wing swept straight through it. The box is sized from the
    mesh bounds instead. Still not a mesh collider (those force MJX onto its slow
    convex path) — a flat plate IS a box to within a millimetre.

    Asymmetric collision masks so the wing hits ONLY the obstacle/ceiling: wing =
    contype 2 / conaffinity 4. That makes wing<->obstacle collide, but NOT
    wing<->body (bit 1, the struts/pillars sit right by the wing root), NOT
    wing<->floor, and crucially NOT wing<->wing (both wings share the same masks,
    so a symmetric bit would make the two plates collide with each other at the
    top of the stroke and blow up the sim).
    """
    root = ET.fromstring(xml)
    added = 0
    for body in root.iter("body"):
        name = body.get("name", "")
        if not name.endswith("_wing"):
            continue
        cx, cy, cz = WING_PLATE_CENTRE
        cy = cy if name.startswith("L") else -cy   # R shoulder is the Y-mirror of L
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_collision",
            type="box",
            pos=f"{cx} {cy} {cz}",
            size=" ".join(str(h) for h in WING_PLATE_HALF),
            density="0",     # collision-only: must not add mass (a solid box at
                             # default density would sink the 73 g bird)
            rgba="0 0 0 0",  # invisible; collision only
            contype="2",
            conaffinity="4",
            group="3",
            solref=RIGID_SOLREF,
            solimp=RIGID_SOLIMP,
        )
        added += 1
    if added == 0:
        raise ValueError("no *_wing bodies found to add collision to")
    return _tostr(root)


def add_poles(
    xml: str,
    names,
    *,
    radius: float = 0.025,
    z_lo: float = 0.0,
    z_hi: float = 2.4,
    rgba: str = "0.86 0.30 0.14 1",
    park=(30.0, 0.0, 0.0),
) -> str:
    """Vertical collidable poles the wings strike in level forward flight.

    Mocap (kinematic, infinite mass) for exactly the reason ``add_ceiling`` is:
    the renderer flaps once, MEASURES how far the wing plate actually reaches
    sideways, and only then plants the poles at a lateral offset the wing cannot
    miss. A hardcoded y stops being a collision the moment the design, the flap
    amplitude or the stroke-plane tilt changes.

    contype 4 / conaffinity 0, the same pairing as the ceiling and for the same
    reason: conaffinity 4 is the ONE bit unique to the wing plates, so a pole
    catches a wing and nothing else — not the tensegrity struts (they carry
    contype 2 as well, and the bird would hang off a pole by its shoulder), not
    the floor. The bird's BODY is therefore a ghost to the poles; the render
    keeps it clear of them geometrically instead (see ``pole_axis_gap``).

    Capsule, not cylinder: capsule<->box is a native MuJoCo primitive pair, and
    the rounded ends of a 2.4 m pole are not visible anyway.
    """
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    px, py, pz = park
    for name in names:
        body = ET.SubElement(worldbody, "body", name=name, mocap="true",
                             pos=f"{px} {py} {pz}")
        ET.SubElement(
            body, "geom", name=f"{name}_geom", type="capsule",
            fromto=f"0 0 {z_lo} 0 0 {z_hi}", size=str(radius), rgba=rgba,
            contype="4", conaffinity="0",
            solref=RIGID_SOLREF, solimp=RIGID_SOLIMP,
        )
    return _tostr(root)


def add_mocap_obstacle(
    xml: str,
    *,
    radius: float = 0.018,
    half_height: float = 0.16,
    rgba: str = "0.85 0.2 0.2 1",
    park=(5.0, 0.0, 0.3),
) -> str:
    """A MOVABLE collidable pole (a mocap body) that can be teleported in/out of the
    wing's arc at runtime via ``data.mocap_pos``. Unlike ``add_obstacle`` (a welded,
    fixed pole) a mocap body stays kinematic — infinite mass, driven by mocap_pos —
    yet still collides, so the collision demo can bring an obstacle in front of the
    wing and then remove it. Spans ``±half_height`` in its own z about the mocap
    origin; contype/conaffinity 1 to hit the wing collision capsule. Starts parked
    at ``park`` (well out of the way)."""
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    px, py, pz = park
    body = ET.SubElement(worldbody, "body", name="obstacle", mocap="true", pos=f"{px} {py} {pz}")
    ET.SubElement(
        body, "geom", name="obstacle_pole", type="capsule",
        fromto=f"0 0 {-half_height} 0 0 {half_height}", size=str(radius),
        rgba=rgba, contype="4", conaffinity="2",   # pairs with the wing capsule only
    )
    return _tostr(root)


# --------------------------------------------------------------------------- #
# Visual hummingbird skin (ported from Tensegrity-Playground/build_visual_bird)
# --------------------------------------------------------------------------- #
# The rufous-hummingbird meshes live in the sibling Tensegrity-Playground repo
# (~29 MB); reference them in place rather than duplicating. Override with the
# --bird-assets flag if that checkout is elsewhere.
BIRD_ASSETS_DIR = (
    "/Users/abhinavchavali/Code/Tensegrity-Playground/"
    "templates/generated-joint/assets/bird"
)

# ── skin TUNABLES (start = the TP values; retune for this rig from stills) ──
BODY_SCALE = 0.0058          # body/head/tail base mesh scale
BODY_WIDEN = 1.55            # extra lateral scale on the BODY so it wraps the struts
WING_SCALE = 0.0085          # rufous wing mesh scale
BODY_ALPHA = 0.20            # torso opacity (lower = more see-through to the joint)
TORSO_POS = (0.05, 0.0, 0.255)
TORSO_QUAT = (0.69412, 0.69412, 0.13492, -0.13492)
WING_QUAT_L = (0.5, 0.5, -0.5, 0.5)
WING_QUAT_R = (-0.5, 0.5, 0.5, 0.5)
WING_RAISE = -0.118          # world-z offset of the wing root (see _wing_pos)
WING_OUT = 0.0
WING_FWD = 0.0


def _wing_pos(side: str) -> str:
    raise_local = -WING_RAISE if side == "L" else WING_RAISE
    return f"{WING_FWD} {raise_local} {WING_OUT}"


def add_visual_bird(xml: str, *, assets_dir: str = BIRD_ASSETS_DIR) -> str:
    """Layer the rufous-hummingbird skin over the flying rig.

    A static, semi-transparent torso (body/head/tail) is parented to the
    ``airframe`` body so it flies with the mechanism while the joint shows
    through; each flapping ``wing.stl`` geom is swapped for the real L/R wing
    mesh. Must run AFTER to_airframe. Meshes are referenced by absolute path.
    """
    root = ET.fromstring(xml)
    asset = root.find("asset")
    s_body = f"{BODY_SCALE} {BODY_SCALE} {BODY_SCALE * BODY_WIDEN}"
    s_ht = f"{BODY_SCALE} {BODY_SCALE} {BODY_SCALE}"
    s_w = f"{WING_SCALE} {WING_SCALE} {WING_SCALE}"
    ET.SubElement(asset, "texture", name="rufous_tex", type="2d",
                  file=f"{assets_dir}/rufous_diffuse.png")
    for name, fname, scale in (
        ("bird_body", "rufous_body.obj", s_body),
        ("bird_head", "rufous_head.obj", s_ht),
        ("bird_tail", "rufous_tail.obj", s_ht),
        ("bird_wing", "rufous_lw_fd.obj", s_w),   # left, feathers-down (pre-generated)
        ("bird_wing_r", "rufous_rw_c.obj", s_w),  # right, canonicalized (pre-generated)
    ):
        ET.SubElement(asset, "mesh", name=name, file=f"{assets_dir}/{fname}", scale=scale)
    ET.SubElement(asset, "material", name="bird_body_mat", texture="rufous_tex",
                  rgba=f"1 1 1 {BODY_ALPHA}", specular="0.2", shininess="0.3")
    ET.SubElement(asset, "material", name="bird_mat", texture="rufous_tex",
                  rgba="1 1 1 1", specular="0.2", shininess="0.3")
    ET.SubElement(asset, "material", name="bird_wing_mat", texture="rufous_tex",
                  rgba="1 1 1 1", specular="0.2", shininess="0.3")

    airframe = next((b for b in root.iter("body") if b.get("name") == AIRFRAME_BODY), None)
    if airframe is None:
        raise ValueError("add_visual_bird must run after to_airframe (no airframe body)")
    torso = ET.SubElement(airframe, "body", name="visual_torso",
                          pos=" ".join(map(str, TORSO_POS)),
                          quat=" ".join(map(str, TORSO_QUAT)))
    # The skin is COSMETIC: an explicit tiny inertial makes MuJoCo ignore the
    # mesh-derived mass (the body/head/tail meshes at default density weigh ~2.3 kg,
    # which would sink the 73 g bird). Without this the skinned rig is dead weight.
    ET.SubElement(torso, "inertial", pos="0 0 0", mass="1e-6", diaginertia="1e-9 1e-9 1e-9")
    for gname, mesh, mat in (
        ("bird_body_geom", "bird_body", "bird_body_mat"),
        ("bird_head_geom", "bird_head", "bird_mat"),
        ("bird_tail_geom", "bird_tail", "bird_mat"),
    ):
        ET.SubElement(torso, "geom", name=gname, type="mesh", mesh=mesh, material=mat,
                      contype="0", conaffinity="0", group="2")

    # Swap the two placeholder wing.stl geoms (L then R in document order).
    wing_geoms = [g for g in root.iter("geom") if g.get("mesh") == "wing"]
    if len(wing_geoms) != 2:
        raise ValueError(f"expected 2 wing.stl geoms, found {len(wing_geoms)}")
    for g, side, quat, mesh in zip(
        wing_geoms, ("L", "R"), (WING_QUAT_L, WING_QUAT_R), ("bird_wing", "bird_wing_r")
    ):
        g.set("mesh", mesh)
        g.set("material", "bird_wing_mat")
        g.set("pos", _wing_pos(side))
        g.set("quat", " ".join(map(str, quat)))
    return _tostr(root)


def _tostr(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    # Self-check: the transforms produce a scene MuJoCo can load, with a single
    # airframe freejoint that swallowed the welded bodies (mobile struts stay free).
    from pathlib import Path

    import mujoco
    import numpy as np

    base = Path(__file__).resolve().parent.parent / "scene.xml"
    xml = base.read_text()

    flying = to_airframe(xml)
    m = mujoco.MjModel.from_xml_string(flying)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, AIRFRAME_BODY)
    assert bid > 0, "airframe body missing"
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, AIRFRAME_FREEJOINT)
    assert jid >= 0 and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    # Airframe subtree must carry the pillars + fixed struts (i.e. real mass).
    assert m.body_subtreemass[bid] > 0.01, m.body_subtreemass[bid]
    # Mobile struts still float independently: 3 freejoints total (airframe + L,R).
    n_free = int((m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())
    assert n_free == 3, n_free

    full = enable_wing_collision(add_obstacle(flying, (-0.26, 0.02, 0.0)))
    m2 = mujoco.MjModel.from_xml_string(full)
    assert mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_pole") >= 0
    assert mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "L_wing_collision") >= 0
    print("flying.py self-check OK: airframe + obstacle + wing collision load")

    # Ceiling: mocap (so it can be positioned at runtime) and it must pair with
    # the wing capsule ONLY -- if the masks let the body through, the bird would
    # belly-flop into it instead of clipping a wingtip.
    ceil = enable_wing_collision(add_ceiling(flying))
    m4 = mujoco.MjModel.from_xml_string(ceil)
    cg = mujoco.mj_name2id(m4, mujoco.mjtObj.mjOBJ_GEOM, "ceiling_slab")
    assert cg >= 0 and m4.body_mocapid[m4.geom_bodyid[cg]] >= 0, "ceiling must be mocap"
    hits = sorted(
        mujoco.mj_id2name(m4, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(m4.ngeom)
        if (m4.geom_contype[g] & m4.geom_conaffinity[cg])
        or (m4.geom_contype[cg] & m4.geom_conaffinity[g])
    )
    assert hits == ["L_wing_collision", "R_wing_collision"], hits
    print("  + ceiling loads, collides with the wing plates only")

    # Poles: same mask contract as the ceiling, checked the same way. A pole that
    # quietly stopped pairing with the wing plates would render as the wing
    # sailing straight through it, which is the one failure this demo cannot have.
    m5 = mujoco.MjModel.from_xml_string(enable_wing_collision(add_poles(flying, ("p1", "p2"))))
    for name in ("p1", "p2"):
        pg = mujoco.mj_name2id(m5, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        assert pg >= 0 and m5.body_mocapid[m5.geom_bodyid[pg]] >= 0, "poles must be mocap"
        assert sorted(
            mujoco.mj_id2name(m5, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(m5.ngeom)
            if (m5.geom_contype[g] & m5.geom_conaffinity[pg])
            or (m5.geom_contype[pg] & m5.geom_conaffinity[g])
        ) == ["L_wing_collision", "R_wing_collision"], f"{name} pairs with the wrong geoms"
    print("  + poles load, collide with the wing plates only")

    # The collision box must COVER the visual wing. When it didn't, the wing mesh
    # sailed straight through the ceiling while the collision geom stopped short.
    for side in "LR":
        wb = mujoco.mj_name2id(m4, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wing")
        cg2 = mujoco.mj_name2id(m4, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_wing_collision")
        lo = m4.geom_pos[cg2] - m4.geom_size[cg2]
        hi = m4.geom_pos[cg2] + m4.geom_size[cg2]
        meshes = [g for g in range(m4.ngeom)
                  if m4.geom_bodyid[g] == wb and m4.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        assert meshes, f"no {side} wing mesh to check against"
        for g in meshes:
            mid = m4.geom_dataid[g]
            v = m4.mesh_vert[m4.mesh_vertadr[mid]: m4.mesh_vertadr[mid] + m4.mesh_vertnum[mid]]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m4.geom_quat[g])
            pts = v @ rot.reshape(3, 3).T + m4.geom_pos[g]
            assert (pts.min(axis=0) >= lo - 1e-6).all() and (pts.max(axis=0) <= hi + 1e-6).all(), (
                f"{side} wing mesh sticks out of its collision box: "
                f"mesh {np.round(pts.min(axis=0), 4)}..{np.round(pts.max(axis=0), 4)} "
                f"vs box {np.round(lo, 4)}..{np.round(hi, 4)}")
    print("  + wing collision boxes cover the wing meshes")

    if Path(BIRD_ASSETS_DIR).is_dir():
        m3 = mujoco.MjModel.from_xml_string(add_visual_bird(flying))
        assert mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_BODY, "visual_torso") >= 0
        assert mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_MATERIAL, "bird_body_mat") >= 0
        print("  + visual bird skin loads")
    else:
        print(f"  (skin skipped: {BIRD_ASSETS_DIR} not present)")
