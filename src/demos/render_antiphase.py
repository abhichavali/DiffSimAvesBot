"""Render the left/right wings driven out of phase, sweeping the phase as it goes.

The two shoulders normally start from one matched crank angle (runtime.crank), so
they beat together. Every disk integrates the same velocity command, so an
offset on the right cranks is a persistent L/R phase difference -- 180 deg =
wings in opposite phase.

The clip changes that offset EVERY WINGBEAT -- one waypoint per beat, held for
most of the beat and slewed over the rest -- by adding extra velocity to the
right motors: with a velocity drive, a ctrl difference integrates straight into
phase, so the pattern can be re-commanded as fast as the servo will take it.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import numpy as np

from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from render_configs import TIP_SITES, WING_COLORS, draw_aero_arrows, draw_trail, smoothstep
from runtime.aero import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology, ramp_factor
from runtime.crank import sync_crank_disks_at_rest
from runtime.morphology_runtime import DEFAULT_HUMMER_TG
from runtime.realtime import get_actuator_id
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest

ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")

# One waypoint per wingbeat. Step size is the constraint, not the list: the slew
# happens in half a beat, so a step of d deg costs the right motors an extra
# 1.5*d/360 of nominal ctrl at its peak. Sized against the tracking assert.
PHASES = [0, 90, 180, 90, 0, -90, -180, -90, 0, 45, 135, 180, 90, 0, -45, -135,
          -180, -90, 0, 60, 120, 180, 120, 60, 0, 0]


def _crank_gears(model):
    """Each crank hinge's rotation direction, from its actuator gear.

    The right shoulder is the Y-mirror of the left, so its two cranks run the
    opposite gear (see emit_mjcf._add_actuators). A phase offset has to follow
    each crank's OWN direction or it breaks the t1-vs-t4 relationship inside the
    shoulder -- which is invisible at 180 deg (+-180 are the same angle) and
    wrong everywhere else.
    """
    gears = {}
    for name in ACTUATORS:
        side, station = name.split("_")[0], name.split("_")[2]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{station}_disk_hinge")
        aid = get_actuator_id(model, name)
        if jid >= 0 and aid >= 0:
            gears[(side, station)] = (jid, float(np.sign(model.actuator_gear[aid][0])))
    return gears


def offset_right_cranks(model, data, phase_rad: float) -> None:
    """Advance the right shoulder by phase_rad (0 = the usual in-phase beat)."""
    if not phase_rad:
        return
    for (side, _), (jid, gear) in _crank_gears(model).items():
        if side == "R":
            data.qpos[model.jnt_qposadr[jid]] += gear * phase_rad
    mujoco.mj_forward(model, data)


def relative_phase(model, data, gears) -> float:
    """Right-vs-left crank phase in radians, each side read in its own direction."""
    jl, gl = gears[("L", "t1")]
    jr, gr = gears[("R", "t1")]
    return gr * float(data.qpos[model.jnt_qposadr[jr]]) - gl * float(data.qpos[model.jnt_qposadr[jl]])


def phase_schedule(phases_deg, dwell_beats, ramp_beats, period):
    """Piecewise dwell/smoothstep-ramp through phases_deg, in seconds -> radians.

    One waypoint per segment, and a segment is one beat by default: the pattern
    holds for half a beat and slews to the next over the other half. A velocity
    drive makes that a ctrl offset, not a position command, so how fast the
    pattern can change is limited only by how much extra ctrl the right motors
    will take -- see the per-beat tracking assert in self_check.
    """
    seg = (dwell_beats + ramp_beats) * period
    dwell = dwell_beats * period

    def at(t: float) -> float:
        i = min(int(t / seg), len(phases_deg) - 1)
        u = smoothstep((t - i * seg - dwell) / (ramp_beats * period)) if ramp_beats else 1.0
        nxt = phases_deg[min(i + 1, len(phases_deg) - 1)]
        return math.radians(phases_deg[i] + (nxt - phases_deg[i]) * u)

    return at, seg * len(phases_deg)


def setup(tg: Path, scene: Path, phase_deg: float, warmup_periods: float):
    """Compile, place the cranks phase_deg apart, return the pieces to run it."""
    os.environ["MORPHOLOGY_PATH"] = str(tg)
    os.environ["MUJOCO_XML_PATH"] = str(scene)
    compile_tg(tg, scene, color_structural_tendons=True)

    morph = load_morphology(tg)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    offset_right_cranks(model, data, math.radians(phase_deg))

    control = load_control_for_morphology(tg)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = [get_actuator_id(model, name) for name in ACTUATORS]
    drive_start = data.time
    ramp_seconds = warmup_periods * control.period

    # The nominal drive turns each crank exactly one revolution per control
    # period, so an extra ctrl of nominal*r on the right motors adds r rev of
    # relative phase per period. That is the whole phase-rate calibration.
    ts = np.arange(0.0, control.period, model.opt.timestep)
    nominal = float(np.mean([morph.motors.motor_direction * eval_control(control, t) for t in ts]))
    ctrl_per_phase_rate = nominal * control.period / (2.0 * math.pi)

    def step(phase_rate: float = 0.0) -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ramp = ramp_factor(data.time, drive_start, ramp_seconds)
        base = ramp * morph.motors.motor_direction * eval_control(control, data.time)
        extra = phase_rate * ctrl_per_phase_rate
        for name, aid in zip(ACTUATORS, actuator_ids):
            if aid >= 0:
                data.ctrl[aid] = base + (extra if name[0] == "R" else 0.0)
        mujoco.mj_step(model, data)

    return model, data, control, aero, step


def flap_trace(tg: Path, scene: Path, phase_deg: float, beats: float, warmup: float):
    """Elevation of each wing's flap axis over `beats` beats, after `warmup`."""
    model, data, control, _, step = setup(tg, scene, phase_deg, warmup)
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wing") for s in ("L", "R")]
    for _ in range(math.ceil(warmup * control.period / model.opt.timestep)):
        step()
    trace = []
    for _ in range(math.ceil(beats * control.period / model.opt.timestep)):
        step()
        # Row 2 col 0 of xmat: world z of the wing frame's x axis == flap elevation.
        # (The frame's y/z axes carry the 2/rev feathering, so they are NOT the beat.)
        trace.append([float(np.asarray(data.xmat[b]).reshape(3, 3)[2, 0]) for b in bids])
    return np.array(trace)


def self_check(args) -> None:
    """Two things here can silently be wrong: does the offset actually put the
    wings out of step, and does the live sweep land where it was commanded?"""
    for phase, want in ((0.0, +1.0), (90.0, 0.0), (180.0, -1.0)):
        t = flap_trace(args.tg, args.scene, phase, 2.0, args.warmup_periods)
        t -= t.mean(axis=0)
        corr = float(np.corrcoef(t[:, 0], t[:, 1])[0, 1])
        print(f"static phase {phase:5.1f} deg -> L/R flap correlation {corr:+.3f}")
        if want:
            assert corr * want > 0.9, f"phase {phase}: wanted correlation sign {want:+.0f}, got {corr:+.3f}"
        else:
            assert abs(corr) < 0.3, f"phase {phase}: wanted quadrature, got {corr:+.3f}"

    # The clip changes the pattern every beat, so the thing that silently fails is
    # the servo not keeping up -- the caption would claim a phase the wings never
    # reached. Check the ACTUAL relative phase against the command over the whole
    # schedule, but only inside the back half of each hold: the ramps are supposed
    # to lag, the holds are what the viewer reads.
    model, data, control, _, step = setup(args.tg, args.scene, args.phases[0], args.warmup_periods)
    gears = _crank_gears(model)
    dt = model.opt.timestep
    for _ in range(math.ceil(args.warmup_periods * control.period / dt)):
        step()
    phase_at, seconds = phase_schedule(args.phases, args.dwell_beats, args.ramp_beats, control.period)
    seg = (args.dwell_beats + args.ramp_beats) * control.period
    dwell = args.dwell_beats * control.period
    t0 = data.time
    bias, worst, worst_t = relative_phase(model, data, gears) - phase_at(0.0), 0.0, 0.0
    for _ in range(math.ceil(seconds / dt)):
        t = data.time - t0
        step((phase_at(t + dt) - phase_at(t)) / dt)
        t = data.time - t0
        if 0.5 * dwell <= t % seg <= dwell:
            err = abs(relative_phase(model, data, gears) - bias - phase_at(t))
            worst, worst_t = (err, t) if err > worst else (worst, worst_t)
    print(f"per-beat phase tracking over {seconds / control.period:.0f} beats: "
          f"worst hold error {math.degrees(worst):.1f} deg at t={worst_t:.3f}s")
    assert worst < math.radians(15.0), "the servo cannot keep up with a per-beat pattern change"

    # The strip chart is drawn from pass one and the playhead indexes pass two, so
    # the two runs have to be the same trajectory or the plot silently lies.
    worst = 0.0
    for (_, ta, _, da, _, _), (_, tb, _, db, _, _) in zip(play(args), play(args)):
        worst = max(worst, abs(ta - tb), float(np.abs(da.qpos - db.qpos).max()))
        if ta > 5.0 * control.period:
            break
    print(f"replay divergence over 5 beats: {worst:.2e}")
    assert worst == 0.0, "the two passes diverged; the speed plot would not line up"
    print("self-check ok")


def play(args):
    """Run the whole clip, yielding (frame, time, model, data, aero, trail) per video frame.

    Deterministic, so it can be replayed: pass one collects the motor-speed trace,
    pass two draws it under the render. Cheaper than buffering 500 RGB frames.
    """
    model, data, control, aero, step = setup(args.tg, args.scene, args.phases[0], args.warmup_periods)
    phase_at, seconds = phase_schedule(args.phases, args.dwell_beats, args.ramp_beats, control.period)
    dt = model.opt.timestep

    for _ in range(math.ceil(args.warmup_periods * control.period / dt)):
        step()

    tip_ids = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for s, n in TIP_SITES.items()}
    trail = {s: deque(maxlen=int(args.trail_pts * args.trail_cycles)) for s in tip_ids}
    trail_every = max(1, round(control.period / dt / args.trail_pts))

    sim_per_frame = args.realtime_factor / args.fps
    t0, n_steps = data.time, 0
    for frame in range(math.ceil(seconds / sim_per_frame)):
        t_end = data.time + sim_per_frame
        while data.time < t_end:
            t = data.time - t0
            # Rate, not position: the drive is a velocity servo, so the schedule
            # is differentiated and fed in as an extra right-side ctrl.
            step((phase_at(t + dt) - phase_at(t)) / dt)
            n_steps += 1
            if n_steps % trail_every == 0:
                for s, sid in tip_ids.items():
                    trail[s].append(np.asarray(data.site_xpos[sid]).copy())
        yield frame, data.time - t0, model, data, aero, trail


def n_frames_of(args, period) -> tuple[int, float]:
    _, seconds = phase_schedule(args.phases, args.dwell_beats, args.ramp_beats, period)
    return math.ceil(seconds / (args.realtime_factor / args.fps)), seconds


def speed_strip(times, speeds, width, height, png_path):
    """Render the motor-speed trace once, as an RGB strip to sit under the video.

    Returns (image, x0, x1) -- the pixel columns the time axis spans, so the
    playhead can be a plain column write instead of a matplotlib redraw per frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, (full, zoom) = plt.subplots(2, 1, sharex=True, figsize=(width / dpi, height / dpi), dpi=dpi)
    for ax in (full, zoom):
        for (side, station), v in speeds.items():
            if ax is zoom and side != "L":
                continue  # the swept right traces leave the zoom's y range; they are just clutter here
            ax.plot(times, v, lw=1.0, color=WING_COLORS[side][:3],
                    ls="-" if station == "t1" else "--", label=f"{side} {station}")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    full.set_xlim(times[0], times[-1])
    full.set_ylabel("rev/s", fontsize=8)
    full.legend(fontsize=7, ncol=4, loc="lower right", framealpha=0.6)
    # The sweep dips are ~1 rev/s and the per-beat load ripple is ~0.02, so on one
    # pair of axes the ripple is a flat line. Zoom the second panel onto the left
    # side (never swept, so it is the clean load signature).
    left = np.concatenate([v for (side, _), v in speeds.items() if side == "L"])
    pad = 0.15 * (left.max() - left.min())
    zoom.set_ylim(left.min() - pad, left.max() + pad)
    zoom.set_ylabel("rev/s (zoom)", fontsize=8)
    zoom.set_xlabel("sim time (s)", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    bb = full.get_window_extent()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    assert img.shape == (height, width, 3), f"strip is {img.shape}, wanted {(height, width, 3)}"
    return img, float(bb.x0), float(bb.x1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "out" / "antiphase.mp4")
    p.add_argument("--phases", type=lambda s: [float(x) for x in s.split(",")],
                   default=PHASES,
                   help="L/R phase waypoints in deg, unwrapped, ONE PER BEAT (dwell at each, ramp between)")
    # 0.6/0.4 is the measured edge for 90 deg per beat: it holds to 4.2 deg, and
    # 0.7/0.3 misses a whole step (91.5 deg) because the slew needs more extra
    # ctrl than the right motors have left. Shorten the ramp only if the steps
    # shrink too.
    p.add_argument("--dwell-beats", type=float, default=0.6, help="beats held at each waypoint")
    p.add_argument("--ramp-beats", type=float, default=0.4, help="beats spent moving between waypoints")
    p.add_argument("--warmup-periods", type=float, default=2.0, help="unrecorded ramp-in beats")
    p.add_argument("--fps", type=int, default=30)
    # 10x slow-mo puts one wingbeat on ~1 s of video; at 5x the pattern changes
    # faster than the eye separates them, which defeats the point of the clip.
    p.add_argument("--realtime-factor", type=float, default=0.1, help="sim s per wall s (0.1 = 10x slow-mo)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720, help="3D view height; the plot strip is added below")
    p.add_argument("--plot-height", type=int, default=280, help="motor-speed strip height")
    # Peak net force per wing is ~1 N here, so 1.5 m/N (the render_configs value)
    # throws the arrow a metre and a half off frame. 0.17 puts the peak at about
    # one half-span, which is readable next to the wing that made it.
    p.add_argument("--force-scale", type=float, default=0.17, help="aero arrow length per N")
    p.add_argument("--trail-pts", type=int, default=90, help="wingtip trail samples per beat")
    p.add_argument("--trail-cycles", type=float, default=1.0, help="beats of trail kept")
    p.add_argument("--trail-width", type=float, default=0.0011)
    p.add_argument("--self-check", action="store_true",
                   help="assert the static phases and the live sweep, no render")
    args = p.parse_args()

    if args.self_check:
        self_check(args)
        return

    control = load_control_for_morphology(args.tg)
    phase_at, seconds = phase_schedule(args.phases, args.dwell_beats, args.ramp_beats, control.period)
    n_frames, _ = n_frames_of(args, control.period)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: the speed trace, so the strip chart exists before frame 0 is drawn.
    print(f"pass 1/2: motor speeds over {n_frames} frames", flush=True)
    times, speeds, lookat = [], None, None
    for frame, t, model, data, _, _ in play(args):
        gears = _crank_gears(model)
        if speeds is None:
            speeds, lookat = {k: [] for k in gears}, data.xpos[1:].mean(axis=0).copy()
        times.append(t)
        for k, (jid, gear) in gears.items():
            # Read each crank in its own rotation direction, in rev/s.
            speeds[k].append(gear * float(data.qvel[model.jnt_dofadr[jid]]) / (2.0 * math.pi))
    strip, px0, px1 = speed_strip(np.array(times), {k: np.array(v) for k, v in speeds.items()},
                                  args.width, args.plot_height, args.out.with_name(args.out.stem + "_motor_speed.png"))

    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = 150.0, -12.0, 1.05

    opt = mujoco.MjvOption()
    for gi in range(len(opt.geomgroup)):
        opt.geomgroup[gi] = 1 if gi <= 2 else 0
    for si in range(len(opt.sitegroup)):
        opt.sitegroup[si] = 0

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height + args.plot_height}", "-r", str(args.fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.out)],
        stdin=subprocess.PIPE,
    )
    print(f"pass 2/2: rendering {n_frames} frames, {seconds / control.period:.0f} beats, "
          f"phases {args.phases} -> {args.out}", flush=True)

    renderer = None
    for frame, t, model, data, aero, trail in play(args):
        if renderer is None:  # pass 2 builds its own model; bind the renderer to that one
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        draw_aero_arrows(renderer.scene, aero, args.force_scale)
        # The rig is world-fixed here, so the trail is already in world frame.
        draw_trail(renderer.scene, trail, np.zeros(3), np.eye(3), args.trail_width)
        playhead = strip.copy()
        col = int(px0 + (px1 - px0) * t / seconds)
        playhead[:, max(0, col - 1):col + 2] = 30
        proc.stdin.write(np.vstack([renderer.render(), playhead]).tobytes())
        if frame % 60 == 0:
            print(f"  frame {frame:4d}/{n_frames}  t={t:5.2f}s  "
                  f"phase cmd={math.degrees(phase_at(t)):+7.1f} "
                  f"actual={math.degrees(relative_phase(model, data, _crank_gears(model))):+7.1f} deg",
                  flush=True)

    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"done: {args.out}  ({n_frames} frames, {seconds:.2f}s sim)")


if __name__ == "__main__":
    main()
