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
    assert parent_of("L_manus") == "L_ulna"
    for joint, expect in (("L_elbow", "hinge"), ("L_wrist", "hinge"), ("L_humerus_twist", "hinge")):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert jid >= 0, f"missing joint {joint}"
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, expect
    # The humerus carries no joint of its own beyond the feathering twist — no
    # shoulder hinge sneaking in to do the tensegrity's job.
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    assert model.body_jntnum[hid] == 1


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
    for joint in ("elbow", "wrist"):
        travel = rec[joint].max() - rec[joint].min()
        assert 8.0 < travel < 45.0, f"{joint} moves {travel:.1f} deg — not a shallow tuck"
    # Flexion (elbow toward its low stop) must track distance from mid-sweep.
    corr = float(np.corrcoef(rec["elbow"], np.abs(rec["azim_L"]))[0, 1])
    assert corr < -0.5, f"tuck is not phase-locked to the stroke (corr {corr:+.2f})"


def test_fold_is_a_zigzag_not_a_spiral(tmp_path):
    """Elbow and wrist must counter-rotate, the way the reference plate shows.

    With both hinges on the same normal the chain spirals into a C: at full
    flexion the wingtip ended up 160 mm AHEAD of the leading edge and inboard of
    the shoulder. A bird's fold shortens the span and keeps the tip behind the
    leading edge. Measured in the humerus frame, where -x is cranial.
    """
    scene = tmp_path / "fold.xml"
    compile_tg(EAGLE_TG, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    humerus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_wingtip")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"L_{n}") for n in ("elbow", "wrist")]

    def tip_in_bone_frame(fold):
        for jid in joints:
            lo, hi = model.jnt_range[jid]
            data.qpos[model.jnt_qposadr[jid]] = hi + (lo - hi) * fold
        mujoco.mj_forward(model, data)
        rot = np.asarray(data.xmat[humerus]).reshape(3, 3)
        return rot.T @ (np.asarray(data.site_xpos[tip]) - np.asarray(data.xpos[humerus]))

    spread, folded = tip_in_bone_frame(0.0), tip_in_bone_frame(1.0)
    assert folded[2] < spread[2], "folding must shorten the span"
    assert folded[2] > 0.6 * spread[2], "the wing collapses; an eagle flexes, it does not tuck"
    # -x is cranial. The tip may creep forward of the humerus axis, but nowhere
    # near the leading edge -- a spiral fold put it 160 mm out there.
    assert folded[0] > -0.5 * spread[2], f"tip swings {-folded[0] * 1000:.0f} mm past the leading edge"


def test_wing_plane_stays_level(tmp_path):
    """The twist servo cancels the roll the tendon network imposes.

    Uncancelled that roll is ~41 deg peak-to-peak and the wing turns edge-on twice
    a beat. Measured with the deliberate flip switched off, so what is left is
    purely what the cancellation failed to remove.
    """
    scene = tmp_path / "level.xml"
    os.environ["MORPHOLOGY_PATH"] = str(EAGLE_TG)
    compile_tg(EAGLE_TG, scene)
    morph = load_morphology(EAGLE_TG)
    control = load_control_for_morphology(EAGLE_TG)
    model = mujoco.MjModel.from_xml_path(str(scene))
    humerus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus")

    def roll_range(**kwargs):
        data = mujoco.MjData(model)
        sync_crank_disks_at_rest(model, data, lag_rad=morph.motors.crank_phase_lag_rad)
        cranks = CrankDrive(model, morph, control)
        drive = calibrated_drive(model, morph, control, twist=0.0, **kwargs)
        rolls = []
        settle = round(SETTLE * control.period / model.opt.timestep)
        for step in range(round(CYCLES * control.period / model.opt.timestep)):
            data.ctrl[:] = 0.0
            cranks.apply(data)
            drive.apply(data)
            mujoco.mj_step(model, data)
            check_stable(model, data, "levelling rollout")
            if step >= settle:
                rot = np.asarray(data.xmat[humerus]).reshape(3, 3)
                rolls.append(math.degrees(wing_plane_roll(rot, "L")))
        return float(np.ptp(np.asarray(rolls)))

    uncancelled = roll_range(level=0.0)
    residual = roll_range(level=1.0)
    assert uncancelled > 25.0, "nothing to cancel — has the stroke changed?"
    assert residual < 0.4 * uncancelled, (
        f"levelling only got {uncancelled:.1f} deg down to {residual:.1f}"
    )
    assert residual < 15.0, f"wing plane still rolls {residual:.1f} deg"


def test_wing_paddles_rather_than_slices(stroke):
    """The wing plane stands on edge, near-perpendicular to the sweep.

    Flat (roll ~0) the wing slices through a horizontal stroke edge-first, which
    is the wrong way round for a wing that is meant to push against the air. The
    servo holds whatever roll the mount was built with; this asserts the whole
    assembly really is rolled up, not laid down.
    """
    rec, _, drive = stroke
    held = abs(math.degrees(drive.attack))
    assert 60.0 < held < 120.0, f"mount holds the wing at {held:.0f} deg — that is not on edge"
    assert abs(np.abs(rec["roll_L"]).mean() - held) < 20.0, "the wing is not tracking its attack angle"


def test_wing_flips_between_half_strokes(stroke):
    """Leaning one way going forward, the other coming back — the hummingbird flip.

    Measured as the wing plane's departure from the angle the servo holds, so it
    reads the same whether the wing is laid flat or stood on edge.
    """
    rec, _, drive = stroke
    deviation = rec["roll_L"] - math.degrees(drive.attack)
    forward = np.gradient(rec["azim_L"]) > 0
    assert forward.any() and (~forward).any()
    a, b = deviation[forward].mean(), deviation[~forward].mean()
    assert a * b < 0, f"wing does not flip between half-strokes ({a:+.1f} / {b:+.1f} deg)"
    assert min(abs(a), abs(b)) > 3.0, "the flip is too shallow to read"


def test_left_and_right_mirror(stroke):
    rec, _, drive = stroke
    assert np.abs(rec["elev_L"] - rec["elev_R"]).max() < 0.5
    # Side-correct roll metric: mirrored wings must report the same number, not
    # opposite or 180 deg apart.
    assert np.abs(rec["roll_L"] - rec["roll_R"]).max() < 0.5
    mirror = np.abs(rec["tip_L"] - rec["tip_R"] * np.array([1.0, -1.0, 1.0])).max()
    assert mirror < 1e-4, f"wingtips are {mirror * 1000:.2f} mm from mirrored"


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
