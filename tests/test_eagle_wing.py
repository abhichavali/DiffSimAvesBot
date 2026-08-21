"""The AvesEagle wing strokes like the hummingbird, and the plate wing still works.

One rollout of `design/eagle.tg` covers the whole chain: the .tg parses, the
compiler emits the three-linkage wing, the drive's calibration converges, and the
resulting motion is the hummingbird's — a wide side-to-side sweep in a
near-horizontal stroke plane, mirror-symmetric, tucking a little at each reversal,
flipping between half-strokes, plane held level, stable for the whole run.

    uv run pytest -q tests/test_eagle_wing.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402

from compile_tg import compile_tg  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.control_signal import load_control_for_morphology  # noqa: E402
from runtime.crank_sync import sync_crank_disks_at_rest  # noqa: E402
from runtime.eagle_drive import (  # noqa: E402
    CrankDrive,
    calibrate_stroke,
    calibrated_drive,
    check_stable,
    wing_plane_roll,
)  # noqa: E402

EAGLE_TG = ROOT / "design" / "eagle.tg"

CYCLES = 8.0
SETTLE = 5.0


@pytest.fixture(scope="module")
def stroke(tmp_path_factory):
    """Rollout of the eagle design; returns per-sample kinematics of the last cycles."""
    scene = tmp_path_factory.mktemp("eagle") / "scene.xml"
    os.environ["MORPHOLOGY_PATH"] = str(EAGLE_TG)
    os.environ["MUJOCO_XML_PATH"] = str(scene)
    compile_tg(EAGLE_TG, scene)
    morph = load_morphology(EAGLE_TG)
    control = load_control_for_morphology(EAGLE_TG)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    sync_crank_disks_at_rest(model, data)
    cranks = CrankDrive(model, morph, control)
    drive = calibrated_drive(model, morph, control)

    body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_humerus") for s in "LR"}
    tip = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_wingtip") for s in "LR"}
    elbow = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_elbow")]
    wrist = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_wrist")]

    out = {k: [] for k in
           ("elev_L", "elev_R", "azim_L", "roll_L", "roll_R", "tip_L", "tip_R",
            "elbow", "wrist", "camber")}
    # Step-counted: MuJoCo rewinds data.time when it resets a diverged model, so
    # a `while data.time < end` loop spins forever instead of failing.
    total = round(CYCLES * control.period / model.opt.timestep)
    settle = round(SETTLE * control.period / model.opt.timestep)
    for step in range(total):
        data.ctrl[:] = 0.0
        cranks.apply(data)
        drive.apply(data)
        mujoco.mj_step(model, data)
        check_stable(model, data, "eagle rollout")
        if step < settle:
            continue
        for side in "LR":
            rot = np.asarray(data.xmat[body[side]]).reshape(3, 3)
            span = rot[:, 2]
            out[f"elev_{side}"].append(math.degrees(math.asin(np.clip(span[2], -1, 1))))
            out[f"tip_{side}"].append(np.array(data.site_xpos[tip[side]]))
            out[f"roll_{side}"].append(math.degrees(wing_plane_roll(rot, side)))
            if side == "L":
                out["azim_L"].append(math.degrees(math.atan2(span[0], span[1])))
                # World height of the trailing edge minus the leading edge (-x is
                # cranial): positive is nose-down, i.e. pronated.
                origin = np.array(data.xpos[body[side]])
                edge = [origin + rot @ np.array([s * 0.035, 0.0, 0.11]) for s in (-1.0, 1.0)]
                out["camber"].append(float(edge[1][2] - edge[0][2]))
        out["elbow"].append(math.degrees(float(data.qpos[elbow])))
        out["wrist"].append(math.degrees(float(data.qpos[wrist])))
    return {k: np.asarray(v) for k, v in out.items()}, morph, drive


def test_topology_is_humerus_ulna_manus(tmp_path):
    """Humerus rides the tensegrity strut; ulna and manus hang off plain hinges."""
    scene = tmp_path / "topology.xml"
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))

    def parent_of(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"missing body {name}"
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[bid])

    # L_part_1 is the mobile strut: the humerus' only connection to the world is
    # through the tendon network, exactly like the hummingbird's plate wing.
    assert parent_of("L_humerus") == "L_part_1"
    assert parent_of("L_ulna") == "L_humerus"
    for joint in ("L_elbow", "L_humerus_twist"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert jid >= 0, f"missing joint {joint}"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    # The humerus carries no joint of its own beyond the feathering twist — no
    # shoulder hinge sneaking in to do the tensegrity's job.
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    assert model.body_jntnum[hid] == 1


def test_elbow_is_exactly_one_dof(tmp_path):
    """The elbow is the one plain hinge left in the wing, and it stays 1 DOF."""
    scene = tmp_path / "elbow.xml"
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    for side in "LR":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ulna")
        assert model.body_jntnum[bid] == 1, f"{side}_ulna carries more than the elbow"
        jid = model.body_jntadr[bid]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) == f"{side}_elbow"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        assert model.jnt_limited[jid], "the elbow hinge is unbounded"


def test_wrist_is_a_tensegrity_joint(tmp_path):
    """The wrist is a scaled-down copy of the shoulder, not a hinge.

    A free joint has to be top level in MuJoCo, so the only way to hold the hand
    on tendons alone is to lift it out of the kinematic chain: the distal strut is
    a worldbody child and the manus rides it. If the manus is still a descendant
    of the ulna, whatever is there is a compliant hinge, not a tensegrity joint.
    """
    scene = tmp_path / "wrist.xml"
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    morph = load_morphology(EAGLE_TG)
    assert morph.wing.bones.wrist is not None

    def parent(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"missing body {name}"
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[bid])

    for side in "LR":
        assert parent(f"{side}_wrist_proximal") == f"{side}_ulna"
        assert parent(f"{side}_wrist_distal") == "world"
        assert parent(f"{side}_manus") == f"{side}_wrist_distal"
        # No wrist hinge and no wrist servo survive.
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist") < 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_wrist_servo") < 0
        # The distal strut's one joint is free — that is the joint.
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_distal")
        assert model.body_jntnum[bid] == 1
        assert model.jnt_type[model.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE

    # Same six sigma generators as the shoulder, so twelve tendons a side.
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(model.ntendon)]
    assert sum(1 for n in names if n and n.startswith("L_wrist_")) == 12

    # It has to sit ON the joint and bridge it: forearm ends, tensegrity, hand
    # begins. Centred on the bone line, straddling the ulna's tip, with the manus
    # starting at the far strut. Placed by body origin instead of by arc midpoint
    # it hangs below the bones and reads as a cage lying on the feathers; started
    # at the ulna's tip instead of the far strut it is buried inside the hand.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bones = morph.wing.bones
    humerus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    rot = np.asarray(data.xmat[humerus]).reshape(3, 3)
    origin = np.asarray(data.xpos[humerus])

    def in_wing_frame(point):
        return rot.T @ (np.asarray(point) - origin)

    joint = bones.humerus + bones.ulna
    cage = np.array([
        in_wing_frame(data.site_xpos[i])
        for i in range(model.nsite)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").startswith("L_wrist")
    ])
    assert cage[:, 2].min() < joint < cage[:, 2].max(), "the joint does not straddle the ulna's tip"
    # Centred on the bones, not hanging off one side of them.
    assert abs(cage[:, 0].mean()) < 0.002 and abs(cage[:, 1].mean()) < 0.002

    manus = in_wing_frame(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_manus")])
    assert manus[2] > joint + 0.5 * bones.wrist.gap, "the hand starts inside the joint"
    assert abs(manus[0]) < 0.002 and abs(manus[1]) < 0.002, "the hand is off the bone line"
    # Scaled down, not a second shoulder.
    assert (cage[:, 2].max() - cage[:, 2].min()) < 0.5 * bones.ulna


def test_stroke_is_a_hummingbird_sweep(stroke):
    """Side to side in a near-horizontal stroke plane, not an up-and-down flap."""
    rec, _, drive = stroke
    azim, elev = rec["azim_L"], rec["elev_L"]
    sweep = azim.max() - azim.min()
    rise = elev.max() - elev.min()
    assert 80.0 <= sweep <= 160.0, f"sweep {sweep:.1f} deg is not a hummingbird stroke"
    assert sweep > 2.0 * rise, f"elevation {rise:.1f} deg rivals the sweep {sweep:.1f} deg"
    # Centred: the wing rows about the lateral direction and stays near level,
    # rather than sweeping only forward or riding high on one side.
    assert abs(0.5 * (azim.max() + azim.min())) < 20.0, "sweep is not centred on lateral"
    assert abs(0.5 * (elev.max() + elev.min())) < 15.0, "stroke plane is not near level"
    # And the tip must actually travel further fore-aft than up-down.
    travel = rec["tip_L"].max(axis=0) - rec["tip_L"].min(axis=0)
    assert travel[0] > 2.0 * travel[2], f"wingtip travel {travel * 1000} mm is not a sweep"


def test_wing_tucks_at_each_reversal(stroke):
    """Spread across the sweep, tucked a little at both ends — twice a cycle.

    A hummingbird's wing is nearly rigid, so the joints only take up a shallow
    flex, but it has to be phase-locked to the stroke: most flexed where the sweep
    turns around, most extended where it is fastest.
    """
    rec, _, drive = stroke
    travel = rec["elbow"].max() - rec["elbow"].min()
    assert 8.0 < travel < 45.0, f"elbow moves {travel:.1f} deg — not a shallow tuck"
    # Flexion (elbow toward its low stop) must track distance from mid-sweep.
    corr = float(np.corrcoef(rec["elbow"], np.abs(rec["azim_L"]))[0, 1])
    assert corr < -0.5, f"tuck is not phase-locked to the stroke (corr {corr:+.2f})"


def test_folding_shortens_the_wing(tmp_path):
    """Driving the elbow in must pull the hand in, not fling it forward.

    With a tensegrity wrist the hand is a free body, so this has to be settled
    rather than posed: setting a joint angle and calling mj_forward moves the
    forearm and leaves the manus where it was. Measured in the humerus frame,
    where +z is span and -x is cranial.
    """
    scene = tmp_path / "fold.xml"
    os.environ["MORPHOLOGY_PATH"] = str(EAGLE_TG)
    compile_tg(EAGLE_TG, scene)
    morph = load_morphology(EAGLE_TG)
    control = load_control_for_morphology(EAGLE_TG)
    model = mujoco.MjModel.from_xml_path(str(scene))
    humerus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_wingtip")
    elbow_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_elbow_servo")
    lo, hi = (float(v) for v in model.actuator_ctrlrange[elbow_act])

    def settle(fold):
        """Hold the elbow at `fold` of its range with the shoulder still, then look."""
        data = mujoco.MjData(model)
        sync_crank_disks_at_rest(model, data, lag_rad=morph.motors.crank_phase_lag_rad)
        target = hi + (lo - hi) * fold
        for _ in range(round(2.0 * control.period / model.opt.timestep)):
            data.ctrl[:] = 0.0
            for side in "LR":
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_elbow_servo")
                data.ctrl[aid] = target
            mujoco.mj_step(model, data)
            check_stable(model, data, "fold settle")
        rot = np.asarray(data.xmat[humerus]).reshape(3, 3)
        return rot.T @ (np.asarray(data.site_xpos[tip]) - np.asarray(data.xpos[humerus]))

    # Measured at half fold, the deepest the drive goes (it uses 0.35). Beyond
    # that the hand swings cranially — with a tensegrity wrist nothing actuates
    # the carpus, so the distal wing hinges at the elbow as one rigid arm and
    # there is no counter-rotation to keep the tip back. In a bird that comes from
    # the ulna-radius linkage, not from a muscle at the wrist, which is what the
    # wing cables are for; at full fold the tip is still 167 mm cranial.
    spread, folded = settle(0.0), settle(0.5)
    assert folded[2] < spread[2] - 0.003, "folding the elbow must shorten the span"
    assert folded[2] > 0.8 * spread[2], "half fold should barely shorten an eagle's wing"
    assert folded[0] > -0.4 * spread[2], f"tip swings {-folded[0] * 1000:.0f} mm cranial at half fold"


def test_feathers_clear_the_rig_and_each_other(tmp_path):
    """No vane may reach inboard through the strut, across the midline, or lie
    exactly on top of a neighbour.

    A tertial raked past 90 deg points back inboard: the root one used to pass
    through the strut rod and into the other wing's tertials. And every vane
    emitted at bone-frame y=0 is exactly coplanar with its neighbours, which
    z-fights into streaks. Both are unguarded constants otherwise.
    """
    scene = tmp_path / "clearance.xml"
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    vanes = {s: [n for n in names if n and n.startswith(f"{s}_") and
                 any(k in n for k in ("tertial", "secondary", "primary"))] for s in "LR"}
    assert len(vanes["L"]) == len(vanes["R"]) > 12
    rods = [n for n in names if n and n.startswith("L_part_1_rod")]

    joints = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"L_{n}")
              for n in ("elbow", "wrist", "humerus_twist")}
    mirrored = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"R_{n}")
                for n in ("elbow", "wrist", "humerus_twist")}
    worst_rig = worst_cross = 1e9
    for fold in (0.0, 0.5, 1.0):
        for twist in (-1.0, 0.0, 1.0):
            for side_joints in (joints, mirrored):
                for name, jid in side_joints.items():
                    lo, hi = model.jnt_range[jid]
                    frac = twist * 0.5 + 0.5 if name == "humerus_twist" else fold
                    data.qpos[model.jnt_qposadr[jid]] = hi + (lo - hi) * frac
            mujoco.mj_forward(model, data)
            for vane in vanes["L"]:
                for rod in rods:
                    worst_rig = min(worst_rig,
                                    mujoco.mj_geomDistance(model, data, gid(vane), gid(rod), 0.05, None))
                for other in vanes["R"]:
                    worst_cross = min(worst_cross,
                                      mujoco.mj_geomDistance(model, data, gid(vane), gid(other), 0.2, None))
    assert worst_rig > 0.0, f"a feather passes through the strut ({worst_rig * 1000:.1f} mm)"
    assert worst_cross > 0.0, f"the wings' feathers pass through each other ({worst_cross * 1000:.1f} mm)"

    # Shingled, not coplanar: no two vanes anywhere on the wing may share an
    # offset along the wing normal, or they z-fight wherever they overlap.
    ys = {round(float(model.geom_pos[gid(n)][1]), 7) for n in vanes["L"]}
    assert len(ys) == len(vanes["L"]), f"vanes are coplanar ({len(ys)} planes for {len(vanes['L'])})"
    # ...but the whole fan stays thin: this is a wing, not a venetian blind.
    assert max(ys) - min(ys) < 0.012


def test_wingtip_clears_the_floor(stroke):
    rec, _, drive = stroke
    assert rec["tip_L"][:, 2].min() > 0.02, "wingtip sweeps through the floor"


@pytest.mark.parametrize(
    "pattern,replacement",
    [
        # Matched by shape, not by literal value: the mount is a searched number
        # and pinning it here means this test breaks every time it is retuned.
        (r"mount(\s*)= \[[^\]]*\] deg", r"mount\1= [-11.4, 8.1] deg"),
        (r"mount(\s*)= \[[^\]]*\] deg", r"mount\1= 30 deg"),
        (r"humerus(\s*)= ", r"humerous\1= "),
        (r"elbow_range(\s*)= \[[^\]]*\] deg", r"elbow_range\1= [-70, 4, 9] deg"),
    ],
)
def test_malformed_bones_section_is_rejected(tmp_path, pattern, replacement):
    """A dropped mount component or a typo'd key is a different wing, not a rounder
    number — the parser must say so instead of substituting a default."""
    text, count = re.subn(pattern, replacement, EAGLE_TG.read_text(), count=1)
    assert count == 1, f"{pattern} no longer matches design/eagle.tg"
    tg = tmp_path / "bad.tg"
    tg.write_text(text)
    with pytest.raises(ValueError):
        load_morphology(tg)


def test_crank_lag_holds_over_a_long_run(tmp_path):
    """The lag that selects the flap mode must not walk off.

    The two crank disks are velocity servos at different points of the tendon
    load cycle, so their achieved rates differ ~0.1% and the offset drifted
    -0.31 deg/beat: the stroke left the eagle regime by about cycle 200. The pair
    counter-rotates, so q_t1 + q_t4 is the invariant that exposes it.
    """
    scene = tmp_path / "lag.xml"
    os.environ["MORPHOLOGY_PATH"] = str(EAGLE_TG)
    compile_tg(EAGLE_TG, scene)
    morph = load_morphology(EAGLE_TG)
    control = load_control_for_morphology(EAGLE_TG)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    sync_crank_disks_at_rest(model, data, lag_rad=morph.motors.crank_phase_lag_rad)
    cranks = CrankDrive(model, morph, control)

    cycles = 25
    for _ in range(round(cycles * control.period / model.opt.timestep)):
        data.ctrl[:] = 0.0
        cranks.apply(data)
        mujoco.mj_step(model, data)
        check_stable(model, data, "crank lag run")
    drift = math.degrees(cranks.lag_error(data))
    # Unlocked this is ~0.31 deg/cycle, i.e. ~8 deg over this window.
    assert abs(drift) < 2.0, f"crank lag drifted {drift:+.2f} deg over {cycles} cycles"


def test_divergence_raises_instead_of_hanging(tmp_path):
    """MuJoCo resets MjData on a bad qacc, which rewinds data.time — so a
    clock-bounded loop restarts forever and the finite check never runs. The
    warning counter is the only thing that remembers."""
    tg = tmp_path / "toobig.tg"
    tg.write_text(
        EAGLE_TG.read_text()
        .replace("timestep   = 0.13 ms", "timestep   = 0.40 ms")
        .replace("meshdir    = assets", f"meshdir    = {ROOT / 'assets'}")
        .replace("control = eagle_control.tg", f"control = {ROOT / 'design' / 'eagle_control.tg'}")
    )
    scene = tmp_path / "toobig.xml"
    os.environ["MORPHOLOGY_PATH"] = str(tg)
    compile_tg(tg, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    with pytest.raises(RuntimeError, match="diverged"):
        calibrated_drive(model, load_morphology(tg), load_control_for_morphology(tg))
    os.environ["MORPHOLOGY_PATH"] = str(EAGLE_TG)


def test_calibration_does_not_read_the_environment(tmp_path, monkeypatch):
    """`calibrate_stroke` is handed a morphology; it must use that one.

    The crank sync underneath re-reads MORPHOLOGY_PATH, which falls back to
    design/shoulder.tg. Leaking through it applies that file's crank lag instead
    of this one's, which silently moves the clock every schedule is written
    against.
    """
    scene = tmp_path / "noenv.xml"
    monkeypatch.setenv("MORPHOLOGY_PATH", str(EAGLE_TG))
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    morph = load_morphology(EAGLE_TG)
    control = load_control_for_morphology(EAGLE_TG)
    monkeypatch.delenv("MORPHOLOGY_PATH")
    calib = calibrate_stroke(model, morph, control)
    monkeypatch.setenv("MORPHOLOGY_PATH", str(EAGLE_TG))
    reference = calibrate_stroke(model, morph, control)
    assert abs(calib.reversal_phase - reference.reversal_phase) < 1e-9
    assert calib.axis == reference.axis
    assert np.allclose(calib.bank_coeffs, reference.bank_coeffs)


def test_plate_wing_still_compiles(tmp_path):
    """The hummingbird path is untouched: no `bones:` -> the old plate wing."""
    tg = ROOT / "design" / "shoulder.tg"
    assert load_morphology(tg).wing.bones is None
    scene = tmp_path / "plate.xml"
    compile_tg(tg, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_wing") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus") < 0
