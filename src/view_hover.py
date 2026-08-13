"""Fly the stabilized hover design in the MuJoCo viewer, with demo modes.

The base is the trained hover controller (6-DOF stabilizing wrench on the airframe
while the flapping carries the gravity-matched lift). Each ``--mode`` layers one
demo on top of that stable hover:

  hover       just hover (default) — upright, on-station, recovers from nudges
  vector      force vectoring: flap speed + stroke-plane scripted wingbeat-by-
              wingbeat (tilt fore/aft, slow to a full STOP, resume) with live
              aerodynamic-force arrows so you see the force track the wing motion
  multimodal  up/down-stroke asymmetry via a precessing stroke plane + aero arrows
  flying      the visual hummingbird skin over the tensegrity joint + motors, flying
  collision   from hover, a collidable obstacle is moved in front of a wing and then
              removed (wing collision mesh on; contact forces shown)

    mjpython src/view_hover.py --mode vector
    mjpython src/view_hover.py --mode collision
    mjpython src/view_hover.py --mode flying
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco
import mujoco.viewer
import numpy as np

import flying
from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.disk_orientation import DiskOrientation
from runtime.viewer_runtime import RealtimeSync, get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter
from hover_policy import HoverCfg, hover_features, policy_commands, level_stroke_plane

ACT = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")
MODES = ("hover", "vector", "multimodal", "flying", "collision")
WING_COLORS = {"L": np.array([0.10, 0.55, 1.0, 0.95]), "R": np.array([1.0, 0.45, 0.10, 0.95])}
# modes that hold the body extra-stiffly so it stays a clean platform while the
# wings do something drastic (e.g. stop) — larger force bound = stiffer hold.
HOLD_FMAX = {"vector": 0.9, "multimodal": 0.6, "collision": 0.7}


# --------------------------------------------------------------------------- #
# live aero-force arrows (drawn into the viewer's user scene)
# --------------------------------------------------------------------------- #
def _arrow(scn, a, b, rgba, width):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        rgba.astype(np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, a.astype(np.float64), b.astype(np.float64))
    scn.ngeom += 1


def draw_aero_arrows(scn, aero, scale):
    scn.ngeom = 0
    for wing in aero.wings:
        if wing.cached_forces is None:
            continue
        col = WING_COLORS[wing.side]
        thin = col.copy(); thin[3] = 0.5
        net_f = np.zeros(3); net_p = np.zeros(3); n = 0
        for f, p in zip(wing.cached_forces, wing.cached_points):
            f = np.asarray(f); p = np.asarray(p)
            _arrow(scn, p, p + f * scale, thin, 0.0013)
            net_f += f; net_p += p; n += 1
        if n:
            c = net_p / n
            _arrow(scn, c, c + net_f * scale, col, 0.0034)


# --------------------------------------------------------------------------- #
# per-mode schedules (t = seconds since sim start, p = phase within the period)
# --------------------------------------------------------------------------- #
def vector_schedule(t, period):
    """(crank_factor, sp_tilt, sp_pitch): script the wing motion beat-by-beat."""
    p = (t % period) / period
    if p < 0.22:      return 1.0, 0.0, 0.0             # normal hover flap
    if p < 0.40:      return 1.0, 0.0, +0.35           # vector forward (tilt stroke plane)
    if p < 0.58:      return 1.0, 0.0, -0.35           # vector aft
    if p < 0.72:      return max(0.0, 1.0 - (p - 0.58) / 0.14), 0.0, 0.0   # slow to a stop
    if p < 0.84:      return 0.0, 0.0, 0.0             # wings STOPPED (force -> 0)
    return min(1.0, (p - 0.84) / 0.16), 0.0, 0.0        # resume


def multimodal_sp(t, period, amp=0.28):
    """Precess the stroke-plane normal so up/down strokes are asymmetric."""
    ph = 2.0 * math.pi * t / period
    return amp * math.sin(ph), amp * math.sin(ph + math.pi / 2.0)


def collision_obstacle(t, period, park, hit):
    """Where the mocap obstacle is this frame: parked, then in the wing's arc, then gone.
    Enters early (12% into the cycle) and stays for ~40% of it."""
    p = (t % period) / period
    return hit if 0.12 <= p < 0.52 else park


# --------------------------------------------------------------------------- #
def build_scene(args, meta):
    tg = Path(meta["tg"])
    scene = args.run / "hover_scene.xml"
    os.environ["MORPHOLOGY_PATH"] = str(tg)
    os.environ["MUJOCO_XML_PATH"] = str(scene)
    compile_tg(tg, scene.with_suffix(".bench.xml"), color_structural_tendons=True)
    xml = flying.to_airframe(scene.with_suffix(".bench.xml").read_text())
    if not args.keep_pillars:
        xml = flying.shrink_pillar_visual(xml)
    if args.mode == "collision":
        xml = flying.enable_wing_collision(xml)
        xml = flying.add_mocap_obstacle(xml, radius=args.obstacle_r, rgba="0.9 0.15 0.15 1")
    if args.mode == "flying":
        xml = flying.add_visual_bird(xml, assets_dir=args.bird_assets)
    scene.write_text(xml)
    return tg, scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="hover")
    ap.add_argument("--run", type=Path, default=ROOT / "opt" / "runs_hover")
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--speed", type=float, default=0.15, help="viewer realtime factor (0.15 = ~7x slow-mo)")
    ap.add_argument("--openloop", action="store_true", help="ignore the policy (K=0) for comparison")
    ap.add_argument("--hover-height", type=float, default=0.25, help="metres of ground clearance to hold")
    ap.add_argument("--keep-pillars", action="store_true", help="don't trim the pillar visuals")
    ap.add_argument("--arrows", choices=["auto", "on", "off"], default="auto",
                    help="aero-force arrows (auto = on for vector/multimodal)")
    ap.add_argument("--force-scale", type=float, default=1.5, help="aero-arrow length per newton")
    ap.add_argument("--period", type=float, default=8.0, help="seconds per demo cycle (vector/multimodal/collision)")
    ap.add_argument("--obstacle-r", type=float, default=0.05)
    ap.add_argument("--obstacle-dx", type=float, default=0.0, help="obstacle x offset from hover point")
    ap.add_argument("--obstacle-dy", type=float, default=0.04, help="obstacle y offset (smaller = deeper into the wing arc = harder)")
    ap.add_argument("--obstacle-dz", type=float, default=0.16, help="obstacle z offset from hover point")
    ap.add_argument("--bird-assets", default=flying.BIRD_ASSETS_DIR)
    args = ap.parse_args()

    meta = json.loads((args.run / "hover_meta.json").read_text())
    pol = np.load(args.run / f"policy_{args.policy}.npz")
    K, b = pol["K"], pol["b"]
    if args.openloop:
        K, b = np.zeros_like(K), np.zeros_like(b)

    tg, scene = build_scene(args, meta)
    morph = load_morphology(tg)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.integrator = 0
    model.opt.timestep = meta["dt_ms"] * 1e-3
    model.opt.gravity[2] = -meta["gravity"]
    level_stroke_plane(model)  # leveled hover base; multimodal/vector re-tilt it live
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, morph.motors.crank_phase_lag_rad)

    f_max = HOLD_FMAX.get(args.mode, meta["f_max"])
    cfg = HoverCfg(a0=meta["a0"], f_max=f_max, tau_max=meta["tau_max"],
                   qadr=meta["qadr"], dofadr=meta["dofadr"])
    com0 = np.asarray(meta["com0"])
    dof = cfg.dofadr

    if args.hover_height:  # lift all free bodies together + raise the set-point
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                data.qpos[model.jnt_qposadr[j] + 2] += args.hover_height
        com0 = com0.copy(); com0[2] += args.hover_height
        mujoco.mj_forward(model, data)

    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    disk = DiskOrientation(model)
    aids = [get_actuator_id(model, n) for n in ACT]
    alpha = float(meta.get("filter_alpha", 1.0))
    ema = hover_features(data.qpos, data.qvel, cfg, com0, np)

    # collision obstacle placement (world) relative to the hover point
    park = np.array([5.0, 0.0, 0.3])
    hit = np.array([com0[0] + args.obstacle_dx, com0[1] + args.obstacle_dy, com0[2] + args.obstacle_dz])
    mocap_id = -1
    if args.mode == "collision":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        mocap_id = int(model.body_mocapid[bid])
        data.mocap_pos[mocap_id] = park

    t0 = data.time
    draw_arrows = args.arrows == "on" or (args.arrows == "auto" and args.mode in ("vector", "multimodal"))

    def step_forces(_m, _d) -> None:
        nonlocal ema
        t = data.time - t0
        crank_factor = 1.0
        if args.mode == "vector":
            crank_factor, spt, spp = vector_schedule(t, args.period)
            disk.set(spt, spp)
        elif args.mode == "multimodal":
            spt, spp = multimodal_sp(t, args.period)
            disk.set(spt, spp)
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = collision_obstacle(t, args.period, park, hit)

        feats = hover_features(data.qpos, data.qvel, cfg, com0, np)
        ema = ema + alpha * (feats - ema)
        force, tau, _u = policy_commands(K, b, ema, cfg, np)
        data.qfrc_applied[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        data.qfrc_applied[dof : dof + 3] += np.asarray(force, float)
        data.qfrc_applied[dof + 3 : dof + 6] += np.asarray(tau, float)
        data.ctrl[:] = 0.0
        for aid in aids:
            if aid >= 0:
                data.ctrl[aid] = float(cfg.a0) * crank_factor

    print(f"[{args.mode}] gravity={meta['gravity']:.3f} dt={meta['dt_ms']:.3f}ms "
          f"f_max={f_max} arrows={draw_arrows} height={args.hover_height}", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.mode == "collision":
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        sync = RealtimeSync(data, realtime_factor=args.speed)
        while viewer.is_running():
            sync.step_and_sync(model, data, viewer, pre_mj_step_hook=step_forces)
            if draw_arrows:
                draw_aero_arrows(viewer.user_scn, aero, args.force_scale)


if __name__ == "__main__":
    main()
