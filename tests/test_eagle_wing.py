"""The AvesEagle wing actually flaps like an eagle, and the plate wing still works.

One rollout of `design/eagle.tg` covers the whole chain: the .tg parses, the
compiler emits the three-linkage wing, the drive's calibration converges, and the
resulting motion is a bird's — dominated by elevation, mirror-symmetric, folded on
the upstroke, plane held level, and stable for the whole run.

    uv run pytest -q tests/test_eagle_wing.py
"""

from __future__ import annotations

import math
import os
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
from runtime.eagle_drive import calibrated_drive, drive_cranks, wing_plane_roll  # noqa: E402

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
    drive = calibrated_drive(model, morph, control)

    body = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_humerus") for s in "LR"}
    tip = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_wingtip") for s in "LR"}
    elbow = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_elbow")]
    wrist = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_wrist")]

    out = {k: [] for k in ("elev_L", "elev_R", "azim_L", "roll_L", "tip_L", "tip_R", "elbow", "wrist")}
    end, settle = CYCLES * control.period, SETTLE * control.period
    while data.time < end:
        data.ctrl[:] = 0.0
        drive_cranks(model, data, morph, control, 0.0, 2.0 * control.period)
        drive.apply(data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), f"simulation diverged at t={data.time:.3f}s"
        if data.time < settle:
            continue
        for side in "LR":
            rot = np.asarray(data.xmat[body[side]]).reshape(3, 3)
            span = rot[:, 2]
            out[f"elev_{side}"].append(math.degrees(math.asin(np.clip(span[2], -1, 1))))
            out[f"tip_{side}"].append(np.array(data.site_xpos[tip[side]]))
            if side == "L":
                out["azim_L"].append(math.degrees(math.atan2(span[0], span[1])))
                out["roll_L"].append(math.degrees(wing_plane_roll(rot)))
        out["elbow"].append(math.degrees(float(data.qpos[elbow])))
        out["wrist"].append(math.degrees(float(data.qpos[wrist])))
    return {k: np.asarray(v) for k, v in out.items()}, morph


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


def test_flap_is_eagle_like(stroke):
    rec, morph = stroke
    elev = rec["elev_L"]
    amplitude = elev.max() - elev.min()
    # A golden eagle's cruise stroke spans roughly -30..+40 deg of elevation.
    assert 45.0 <= amplitude <= 110.0, f"flap amplitude {amplitude:.1f} deg is not eagle-like"
    assert abs(0.5 * (elev.max() + elev.min())) < 15.0, "stroke is not centred near level"
    # Elevation must dominate the fore-aft sweep, or it is the hummingbird's
    # horizontal stroke plane wearing feathers.
    sweep = rec["azim_L"].max() - rec["azim_L"].min()
    assert amplitude > 1.4 * sweep, f"sweep {sweep:.1f} deg rivals the flap {amplitude:.1f} deg"


def test_wing_folds_on_the_upstroke(stroke):
    rec, _ = stroke
    fold = rec["elbow"].max() - rec["elbow"].min()
    assert fold > 25.0, f"elbow barely moves ({fold:.1f} deg) — the wing is not folding"
    assert rec["wrist"].max() - rec["wrist"].min() > 25.0, "wrist is not folding"
    # Folded at the top of the upstroke, extended at the bottom: the elbow angle
    # and the wing's elevation must run together, not independently.
    corr = float(np.corrcoef(rec["elbow"], rec["elev_L"])[0, 1])
    assert corr < -0.5, f"fold is not phase-locked to the flap (corr {corr:+.2f})"


def test_wing_plane_stays_level(stroke):
    """The twist servo cancels the roll the tendon network imposes.

    Uncancelled it is ~50 deg peak-to-peak and the wing turns edge-on twice a
    beat; what is left should be the deliberate pronation and little else.
    """
    rec, _ = stroke
    roll = rec["roll_L"].max() - rec["roll_L"].min()
    assert roll < 40.0, f"wing plane rolls {roll:.1f} deg — leveling is not holding"


def test_left_and_right_mirror(stroke):
    rec, _ = stroke
    assert np.abs(rec["elev_L"] - rec["elev_R"]).max() < 0.5
    mirror = np.abs(rec["tip_L"] - rec["tip_R"] * np.array([1.0, -1.0, 1.0])).max()
    assert mirror < 1e-4, f"wingtips are {mirror * 1000:.2f} mm from mirrored"


def test_wingtip_clears_the_floor(stroke):
    rec, _ = stroke
    assert rec["tip_L"][:, 2].min() > 0.02, "wingtip sweeps through the floor"


def test_plate_wing_still_compiles(tmp_path):
    """The hummingbird path is untouched: no `bones:` -> the old plate wing."""
    tg = ROOT / "design" / "shoulder.tg"
    assert load_morphology(tg).wing.bones is None
    scene = tmp_path / "plate.xml"
    compile_tg(tg, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_wing") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_humerus") < 0
