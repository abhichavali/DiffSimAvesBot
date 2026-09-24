"""One clip, one unchanged machine: which control axis each wing modulation buys.

Nothing about the morphology changes. Three knobs move, each settable per
shoulder, all of them held mild:

  * STROKE PLANE pitch -- swings the plane fore/aft.
  * STROKE PLANE tilt  -- rolls the plane (compiled mirror convention, so equal
    values on both sides IS the bilaterally symmetric case).
  * STROKE WEIGHTING   -- the within-wing knob: the crank is sped up around
    mid-downstroke and slowed on the upstroke, ctrl *= 1 + w cos(theta - theta_down).
    Zero mean, so the flap frequency is untouched; only the shape of one beat is.
  * L/R PHASE          -- a mild timing offset between the two shoulders.

The clip runs bilaterally SYMMETRIC settings first, then bilaterally
ASYMMETRIC ones. The strip chart under the render is the point: the rig is
world-fixed, so the payoff is not motion but the cycle-averaged aero force and
moment it would fly on. Symmetric settings move fore/aft force and pitch
moment and leave roll/yaw/lateral at zero; asymmetric settings do the reverse.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import numpy as np

from common import (
    TIP_SITES,
    WING_COLORS,
    captioner,
    crank_gears,
    ctrl_per_phase_rate,
    draw_aero_arrows,
    draw_trail,
    ffmpeg_pipe,
    relative_phase,
    scene_option,
    smoothstep,
)
from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.control_signal import drive_command, load_control_for_morphology
from runtime.morphology_runtime import DEFAULT_HUMMER_TG
from runtime.sim import PlateWingSim

# label, group, plane pitch L/R (deg), plane tilt L/R (deg), stroke weight L/R, L-R phase (deg)
# Mild throughout -- these are trim inputs, not the 180 deg party trick. Each row
# changes exactly one thing against neutral.
NEUTRAL = ("neutral", "base", 0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.0)
INPUTS = [
    ("symmetric: stroke plane pitched forward", "sym",  14.0,  14.0,  0.0,   0.0,  0.00,  0.00,  0.0),
    ("symmetric: stroke plane pitched aft",     "sym", -14.0, -14.0,  0.0,   0.0,  0.00,  0.00,  0.0),
    ("symmetric: downstroke-weighted beat",     "sym",   0.0,   0.0,  0.0,   0.0,  0.30,  0.30,  0.0),
    ("asymmetric: differential plane pitch",   "asym",  10.0, -10.0,  0.0,   0.0,  0.00,  0.00,  0.0),
    ("asymmetric: differential plane tilt",    "asym",   0.0,   0.0, 10.0, -10.0,  0.00,  0.00,  0.0),
    ("asymmetric: differential beat weighting", "asym",  0.0,   0.0,  0.0,   0.0,  0.30, -0.30,  0.0),
    ("asymmetric: mild L/R phase offset",      "asym",   0.0,   0.0,  0.0,   0.0,  0.00,  0.00, 30.0),
]
# Every input is bracketed by neutral. The rig's stroke amplitude decays for tens
# of seconds of sim (68.4 deg at beat 10, still falling through 64.5 at beat 160),
# which drags fore/aft force and pitch moment with it. No warmup outruns that, so
# each input is read against the neutrals either side of it instead.
SEGMENTS = [NEUTRAL] + [s for inp in INPUTS for s in (inp, NEUTRAL)]
KNOBS = ("pitch_L", "pitch_R", "tilt_L", "tilt_R", "weight_L", "weight_R", "phase")


class Sequence:
    """Which setting is commanded at time t, held flat then blended into the next."""

    def __init__(self, hold_beats, blend_beats, period, segments=SEGMENTS):
        self.seg = segments
        self.period = period
        self.hold = hold_beats * period
        self.blend = blend_beats * period
        self.step = self.hold + self.blend
        self.total = self.step * len(self.seg)

    def _pair(self, t):
        i = min(int(t / self.step), len(self.seg) - 1)
        j = min(i + 1, len(self.seg) - 1)
        u = smoothstep((t - i * self.step - self.hold) / self.blend) if self.blend else 1.0
        return i, j, u

    def at(self, t) -> dict:
        """Knob values in degrees / dimensionless, blended."""
        i, j, u = self._pair(t)
        a, b = self.seg[i][2:], self.seg[j][2:]
        return {k: a[n] + (b[n] - a[n]) * u for n, k in enumerate(KNOBS)}

    def label(self, t) -> str:
        """Segment name plus only the knobs actually off zero -- dumping all four
        every frame overruns the frame width and gets clipped at both ends."""
        i, j, u = self._pair(t)
        k = j if u > 0.5 else i
        v = self.at(t)
        bits = []
        for name, keys, fmt, eps in (("plane pitch", ("pitch_L", "pitch_R"), "{:+.0f}deg", 0.5),
                                     ("tilt", ("tilt_L", "tilt_R"), "{:+.0f}deg", 0.5),
                                     ("weight", ("weight_L", "weight_R"), "{:+.2f}", 0.02),
                                     ("phase", ("phase",), "{:+.0f}deg", 0.5)):
            if max(abs(v[k_]) for k_ in keys) > eps:
                bits.append(f"{name} " + "/".join(fmt.format(v[k_]) for k_ in keys))
        return self.seg[k][0] + ("    " + "    ".join(bits) if bits else "")

    def windows(self, skip_beats=0.0):
        """(label, group, t_start, t_end) per segment -- the hold window only.

        Anything reading the one-beat running mean must skip the first beat: at
        t_a the window still reaches back into the blend, so the value is part
        ramp. Shading the plot wants the full hold, hence the argument.
        """
        # A hold shorter than the skip would collapse the window to nothing (and
        # hand detrend an empty slice -> NaN). Keep the back half instead: at a
        # one-beat hold nothing IS settled, so the least-contaminated sample is
        # the best available, not a clean one.
        skip = min(skip_beats * self.period, 0.5 * self.hold)
        return [(s[0], s[1], i * self.step + skip, i * self.step + self.hold)
                for i, s in enumerate(self.seg)]


class SideDisks:
    """Per-shoulder crank-disk mount angles, as offsets from the compiled pose.

    runtime.disk_orientation.DiskOrientation drives all four disks from one
    (tilt, pitch) -- that IS the bilaterally symmetric case and is all the
    stroke-plane clip needs. The asymmetric half here needs the two shoulders
    steered independently, so this keeps the same euler convention and splits
    them. Sign mirroring is unchanged: R euler = [-tilt, pitch, 0].
    """

    def __init__(self, model, tilt0_rad, pitch0_rad):
        self.model, self.tilt0, self.pitch0 = model, tilt0_rad, pitch0_rad
        self.bodies = {"L": [], "R": []}
        for name in ("L_t1", "L_t4", "R_t1", "R_t4"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_disk")
            if bid >= 0:
                self.bodies[name[0]].append(bid)
        self._q = np.zeros(4)

    def set(self, side: str, d_tilt_rad: float, d_pitch_rad: float) -> None:
        sign = 1.0 if side == "L" else -1.0
        for bid in self.bodies[side]:
            mujoco.mju_euler2Quat(
                self._q, [sign * (self.tilt0 + d_tilt_rad), self.pitch0 + d_pitch_rad, 0.0], "xyz")
            self.model.body_quat[bid] = self._q

    def all_body_ids(self):
        return self.bodies["L"] + self.bodies["R"]


def setup(tg: Path, scene: Path, warmup_periods: float):
    """Compile and wire the rig; return the pieces plus a knob-driven step()."""
    os.environ["MORPHOLOGY_PATH"] = str(tg)
    os.environ["MUJOCO_XML_PATH"] = str(scene)
    compile_tg(tg, scene, color_structural_tendons=True)

    morph = load_morphology(tg)
    sim = PlateWingSim(mujoco.MjModel.from_xml_path(str(scene)), morph)
    model, data = sim.model, sim.data

    control = load_control_for_morphology(tg)
    gears = crank_gears(model)
    sides = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)[0] for i in sim.crank_ids]

    disks = SideDisks(model, morph.motors.disk.tilt_rad, morph.motors.disk.pitch_rad)
    # Round-trip: at zero offset the compiled body_quat must come back exactly,
    # or the euler-seq / mirror convention here disagrees with the compiler's.
    compiled = model.body_quat[disks.all_body_ids()].copy()
    for side in ("L", "R"):
        disks.set(side, 0.0, 0.0)
    assert np.allclose(model.body_quat[disks.all_body_ids()], compiled, atol=1e-9)

    drive_start = data.time
    ramp_seconds = warmup_periods * control.period
    per_phase_rate = ctrl_per_phase_rate(model, morph, control)

    def crank_angle(side: str) -> float:
        jid, gear = gears[(side, "t1")]
        return gear * float(data.qpos[model.jnt_qposadr[jid]])

    def step(knobs=None, phase_rate: float = 0.0, theta_down: float = 0.0) -> None:
        knobs = knobs or {}
        for side in ("L", "R"):
            disks.set(side, math.radians(knobs.get(f"tilt_{side}", 0.0)),
                      math.radians(knobs.get(f"pitch_{side}", 0.0)))
        base = drive_command(control, morph.motors.motor_direction, data.time, drive_start, ramp_seconds)
        extra = phase_rate * per_phase_rate
        ctrl = []
        for side in sides:
            # Within-wing variation: speed the crank up around mid-downstroke and
            # slow it on the upstroke. cos has zero mean over a revolution, so the
            # beat frequency is unchanged -- only the shape of the beat moves.
            w = knobs.get(f"weight_{side}", 0.0)
            shape = 1.0 + w * math.cos(crank_angle(side) - theta_down) if w else 1.0
            ctrl.append(base * shape + (extra if side == "R" else 0.0))
        sim.step(ctrl)

    return model, data, control, sim.aero, gears, step


def downstroke_phase(model, data, step, control, beats=2.0) -> float:
    """Crank angle at mid-downstroke, so the weighting knob means what it says.

    Wing elevation is very nearly one sinusoid in crank angle, so fit it and
    take the angle where d(elevation)/d(theta) is most negative.
    """
    jid, gear = crank_gears(model)[("L", "t1")]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_wing")
    th, el = [], []
    for _ in range(math.ceil(beats * control.period / model.opt.timestep)):
        step()
        th.append(gear * float(data.qpos[model.jnt_qposadr[jid]]))
        # Row 2 col 0 of xmat: world z of the wing frame's x axis == flap elevation.
        el.append(float(np.asarray(data.xmat[bid]).reshape(3, 3)[2, 0]))
    th = np.array(th)
    basis = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    _, c1, c2 = np.linalg.lstsq(basis, np.array(el), rcond=None)[0]
    # el = c1 cos(th) + c2 sin(th) -> d/dth = c2 cos(th) - c1 sin(th) = R cos(th + psi)
    return float((math.pi - math.atan2(c1, c2)) % (2.0 * math.pi))


def play(args):
    """Run the clip, yielding one record per video frame.

    Deterministic, so it replays: pass one collects the traces, pass two draws
    them under the render. Cheaper than buffering hundreds of RGB frames.
    """
    model, data, control, aero, gears, step = setup(args.tg, args.scene, args.warmup_periods)
    seq = Sequence(args.hold_beats, args.blend_beats, control.period)
    dt = model.opt.timestep

    theta_down = downstroke_phase(model, data, step, control)
    for _ in range(math.ceil(args.warmup_periods * control.period / dt)):
        step()

    tip_ids = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for s, n in TIP_SITES.items()}
    trail = {s: deque(maxlen=int(args.trail_pts * args.trail_cycles)) for s in tip_ids}
    trail_every = max(1, round(control.period / dt / args.trail_pts))
    centre = data.xpos[1:].mean(axis=0).copy()

    # Instantaneous aero swings +-500 mN; the trims being demonstrated are tens of
    # mN. Averaging over EXACTLY one beat kills the swing -- a window that misses
    # by 8% leaves 8% of the fundamental, which is the same size as the signal.
    steps_per_beat = max(1, round(control.period / dt))
    hist = deque(maxlen=steps_per_beat)

    def load():
        force, moment = np.zeros(3), np.zeros(3)
        for wing in aero.wings:
            if wing.cached_forces is None:
                continue
            for f, p in zip(wing.cached_forces, wing.cached_points):
                force += f
                moment += np.cross(np.asarray(p) - centre, f)
        return np.concatenate([force, moment])

    for _ in range(steps_per_beat):  # prime the window so frame 0 is a full beat
        step()
        hist.append(load())

    # Phase was commanded open loop -- a pure integration -- so ANY other L/R ctrl
    # difference integrated into phase as well and never washed out. The
    # differential weighting knob is exactly such a difference, and it left ~50 deg
    # of uncommanded phase standing through every segment after it, including the
    # one whose whole point is a 30 deg phase offset. Feedforward plus PI against
    # the measured phase, so each knob only does what its caption claims.
    #
    # Deliberately SLOW: one beat of time constant, so the loop cannot respond at
    # the beat frequency. Opposite weighting is supposed to swing the L/R phase
    # within the beat (+-17 deg at w=0.3) -- that IS the demonstration -- and only
    # the standing offset is the artifact. A low-gain P is that split for free; it
    # passes ~85% of the per-beat swing and still zeroes the DC. Averaging the
    # error first instead put half a beat of lag in the loop and it rang (neutral
    # residual 0.16 -> 1.95 mN); adding I did not help, because the disturbance is
    # a one-off kick from the ramp, not a standing rate.
    phase_bias = relative_phase(model, data, gears)
    kp = 1.0 / control.period
    err_hist = deque(maxlen=steps_per_beat)   # for reporting only, never fed back

    sim_per_frame = args.realtime_factor / args.fps
    t0, n_steps = data.time, 0
    for frame in range(math.ceil(seq.total / sim_per_frame)):
        t_end = data.time + sim_per_frame
        while data.time < t_end:
            t = data.time - t0
            knobs = seq.at(t)
            # Phase is commanded as a RATE: the drive is a velocity servo, so the
            # schedule has to be differentiated before it goes in.
            err = math.radians(knobs["phase"]) - (relative_phase(model, data, gears) - phase_bias)
            err_hist.append(err)
            rate = math.radians(seq.at(t + dt)["phase"] - knobs["phase"]) / dt + kp * err
            step(knobs, rate, theta_down)
            hist.append(load())
            n_steps += 1
            if n_steps % trail_every == 0:
                for s, sid in tip_ids.items():
                    trail[s].append(np.asarray(data.site_xpos[sid]).copy())
        cycle_mean = np.mean(hist, axis=0)
        yield {
            "frame": frame, "t": data.time - t0, "model": model, "data": data,
            "aero": aero, "trail": trail, "seq": seq, "gears": gears,
            "force": cycle_mean[:3], "moment": cycle_mean[3:],
            # Beat-MEAN phase. The instantaneous value legitimately swings within
            # the beat whenever weighting is on, so only its mean can be checked
            # against the schedule.
            "phase": math.degrees(math.radians(seq.at(data.time - t0)["phase"])
                                  - sum(err_hist) / len(err_hist)),
            "speed": {k: gear * float(data.qvel[model.jnt_dofadr[jid]]) / (2.0 * math.pi)
                      for k, (jid, gear) in gears.items()},
        }


def detrend(times, values, seq):
    """Subtract the neutral level, interpolated between the neutral segments.

    Absolute force and moment creep through the clip as the stroke decays, and
    the creep is not linear, so a two-point correction leaves curvature. Reading
    each input against the neutrals bracketing it removes both.
    """
    ts, vs = [], []
    for _, group, t_a, t_b in seq.windows(skip_beats=1.0):
        if group != "base":
            continue
        m = (times >= t_a) & (times <= t_b)
        ts.append(0.5 * (t_a + t_b))
        vs.append(values[m].mean(axis=0))
    ts, vs = np.array(ts), np.array(vs)
    trend = np.column_stack([np.interp(times, ts, vs[:, i]) for i in range(values.shape[1])])
    return values - trend


def control_strip(times, speeds, force, moment, seq, width, height, png_path):
    """Three stacked panels, rendered once; the playhead is a column write per frame.

    Returns (image, x0, x1) -- the pixel columns the shared time axis spans.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(width / dpi, height / dpi), dpi=dpi)
    speed_ax, force_ax, moment_ax = axes

    for (side, station), v in speeds.items():
        speed_ax.plot(times, v, lw=0.9, color=WING_COLORS[side][:3],
                      ls="-" if station == "t1" else "--", label=f"{side} {station}")
    speed_ax.set_ylabel("crank\nrev/s", fontsize=7)
    speed_ax.legend(fontsize=6, ncol=4, loc="lower left", framealpha=0.6)

    # Shown as a CHANGE from the neutral level: absolute lift (~385 mN) and pitch
    # moment are large and would flatten the trims (~10 mN) into a straight line.
    for ax, data_, names, unit, colors in (
        (force_ax, force, ("fore/aft", "lateral", "vertical"), "mN",
         ("#1b7f5a", "#c2185b", "#5b6bbf")),
        (moment_ax, moment, ("roll", "pitch", "yaw"), "mN·m",
         ("#c2185b", "#1b7f5a", "#e08a00")),
    ):
        d = 1000.0 * detrend(times, data_, seq)   # already a one-beat mean, from play()
        for i, (nm, c) in enumerate(zip(names, colors)):
            ax.plot(times, d[:, i], lw=1.2, color=c, label=nm)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_ylabel(f"delta\n{unit}", fontsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.6)

    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    # Shade the symmetric half so the two acts of the clip are one glance apart.
    shade = {"base": None, "sym": "0.35", "asym": "#e08a00"}
    for _, group, t_a, t_b in seq.windows():
        if shade[group] is None:
            continue
        for ax in axes:
            ax.axvspan(t_a, t_b, color=shade[group], alpha=0.13)
    speed_ax.set_xlim(times[0], times[-1])
    moment_ax.set_xlabel("sim time (s)   [unshaded = neutral, grey = bilaterally symmetric input, "
                         "amber = asymmetric input]", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    bb = speed_ax.get_window_extent()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    assert img.shape == (height, width, 3), f"strip is {img.shape}, wanted {(height, width, 3)}"
    return img, float(bb.x0), float(bb.x1)


def self_check(args) -> None:
    """The claims this clip makes are the thing worth testing: symmetric settings
    must leave the asymmetric axes alone, and asymmetric ones must move them."""
    period = load_control_for_morphology(args.tg).period
    seq = Sequence(args.hold_beats, args.blend_beats, period)
    for i, (name, _, *vals) in enumerate(SEGMENTS):
        got = seq.at(i * seq.step + seq.hold / 2)
        assert all(abs(got[k] - v) < 1e-9 for k, v in zip(KNOBS, vals)), f"{name}: not held flat"

    # The trim table needs a SETTLED cycle-average and the clip's fast holds
    # cannot give one: the mean spans exactly one beat, so at a one-beat hold it
    # is still half made of the blends either side. Measure the claims at a
    # settled hold; the render is the same knobs at a watchable pace, and the
    # determinism check below still runs at whatever the render is set to.
    table = argparse.Namespace(**vars(args))
    table.hold_beats, table.blend_beats = max(args.hold_beats, 4.0), max(args.blend_beats, 2.0)
    seq = Sequence(table.hold_beats, table.blend_beats, period)
    print(f"trim table measured at hold={table.hold_beats:g} blend={table.blend_beats:g} beats "
          f"(render runs hold={args.hold_beats:g} blend={args.blend_beats:g})")

    times, force, moment, speeds = [], [], [], None
    for rec in play(table):
        if speeds is None:
            speeds = {k: [] for k in rec["speed"]}
        times.append(rec["t"])
        force.append(rec["force"])
        moment.append(rec["moment"])
        for k, v in rec["speed"].items():
            speeds[k].append(v)
    times, force, moment = np.array(times), np.array(force), np.array(moment)

    signals = 1000.0 * detrend(times, np.column_stack([force, moment]), seq)
    print(f"{'segment':44s} {'fore/aft':>9s} {'lateral':>9s} {'vert':>9s} "
          f"{'roll':>9s} {'pitch':>9s} {'yaw':>9s}   (mN, mN·m, vs neutral)")
    rows = []
    for name, group, t_a, t_b in seq.windows(skip_beats=1.0):
        m = (times >= t_a) & (times <= t_b)
        d = signals[m].mean(axis=0)
        rows.append((name, group, d))
        print(f"{name:44s} " + " ".join(f"{v:+9.2f}" for v in d))

    # Symmetric axes are fore/aft force, vertical force, pitch; asymmetric are
    # lateral force, roll, yaw. An absolute mN threshold would be arbitrary, so
    # assert SEPARATION against the noise floor the neutral segments leave behind.
    SYM, ASYM = [0, 2, 4], [1, 3, 5]
    floor = max(np.abs(d).max() for _, g, d in rows if g == "base")
    sym_leak = max(np.abs(d[ASYM]).max() for _, g, d in rows if g == "sym")
    print(f"neutral residual {floor:.2f}, worst symmetric leak into roll/yaw/lateral {sym_leak:.2f}")
    assert floor < 2.0, f"neutral segments are not returning to zero ({floor:.2f})"
    for name, group, d in rows:
        if group == "base":
            continue
        own, other = (SYM, ASYM) if group == "sym" else (ASYM, SYM)
        moved, leaked = np.abs(d[own]).max(), np.abs(d[other]).max()
        if name.endswith("phase offset"):
            weakest = min(np.abs(r[ASYM]).max() for n_, g_, r in rows
                          if g_ == "asym" and not n_.endswith("phase offset"))
            # Expected null, and worth keeping as a claim: shifting one wing in
            # time leaves that wing's own cycle-average untouched, so the pair's
            # mean cannot move on any axis. Phase changes the WITHIN-cycle
            # balance, which a cycle mean is blind to by construction.
            assert moved < weakest / 3.0, f"{name}: expected ~no mean effect, got {moved:.2f}"
            continue
        assert moved > 4.0 * floor, f"{name}: barely moved its own axis ({moved:.2f})"
        if group == "sym":
            assert leaked < 0.25 * moved, f"{name}: leaked {leaked:.2f} into roll/yaw/lateral"

    # Every knob must be the only thing it moves. Phase is the one that can drift
    # uncommanded (the drive integrates any L/R ctrl difference), so check the
    # measured phase against the schedule across the WHOLE clip, at the render's
    # own pacing -- that is where the segments come fastest.
    seq_r = Sequence(args.hold_beats, args.blend_beats, period)
    pt, ph, worst_ph = [], [], 0.0
    for rec in play(args):
        pt.append(rec["t"])
        ph.append(rec["phase"])
        worst_ph = max(worst_ph, abs(rec["phase"] - seq_r.at(rec["t"])["phase"]))
    pt, ph = np.array(pt), np.array(ph)
    held = {}   # "neutral" repeats, so this keeps the LAST one -- the end of the clip
    for name, _, t_a, t_b in seq_r.windows(skip_beats=1.0):
        held[name] = float(ph[(pt >= t_a) & (pt <= t_b)].mean())
    print(f"phase: peak excursion {worst_ph:.1f} deg, held at the offset segment "
          f"{held['asymmetric: mild L/R phase offset']:+.1f} deg (commanded +30), "
          f"final neutral {held['neutral']:+.1f} deg")
    # The peak is a genuine transient: switching a differential weighting on and
    # off inside one beat kicks the phase, and one beat is not long enough to
    # settle. What must be true is that each caption is honest where it is read,
    # and that the kick does not accumulate down the clip.
    assert abs(held["asymmetric: mild L/R phase offset"] - 30.0) < 8.0, "phase segment misses its own command"
    assert abs(held["neutral"]) < 8.0, "uncommanded phase is accumulating across the clip"

    # The strip is drawn from pass one and the playhead indexes pass two, so the
    # two runs have to be the same trajectory or the plot silently lies.
    worst = 0.0
    for a, b in zip(play(args), play(args)):
        worst = max(worst, abs(a["t"] - b["t"]), float(np.abs(a["data"].qpos - b["data"].qpos).max()))
        if a["t"] > 5.0 * period:
            break
    print(f"replay divergence over 5 beats: {worst:.2e}")
    assert worst == 0.0, "the two passes diverged; the strip chart would not line up"
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "out" / "bilateral_control.mp4")
    p.add_argument("--hold-beats", type=float, default=1.0, help="beats each setting is held")
    p.add_argument("--blend-beats", type=float, default=0.5, help="beats spent moving between settings")
    p.add_argument("--warmup-periods", type=float, default=20.0,
                   help="unrecorded beats before the clip; the stroke needs ~10 to stop growing")
    p.add_argument("--fps", type=int, default=30)
    # 5x slow-mo, matching the other renders in the repo: a beat is 0.5 s of video
    # and a setting 0.75 s. Playback only -- the sim runs at its own dt either way,
    # so this changes nothing the strip chart or the trim table measure.
    p.add_argument("--realtime-factor", type=float, default=0.2, help="sim s per wall s (0.2 = 5x slow-mo)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=620, help="3D view height; the strip is added below")
    p.add_argument("--plot-height", type=int, default=420, help="strip-chart height")
    # Peak net force per wing is ~1 N here, so render_configs' 1.5 m/N throws the
    # arrow a metre and a half off frame. Halve whatever you reason about first:
    # mjv_connector draws an ARROW at HALF the length it is handed (a 0.4 m
    # connector renders 196 mm), so on screen an arrow is |F| * scale / 2.
    # They are lift-dominated, so the TOP of the 3D view is what bounds them --
    # never the caption band (top y=535 of 620; no arrow reaches below y~270) nor
    # the strip under it. Projecting every arrow of the clip through this fixed
    # camera, the tallest touches the top edge at 0.63 m/N. 0.30 draws the clip's
    # largest instantaneous net force (1.19 N; p95 0.93 N) 178 mm long, ~1.1
    # half-spans (span/2 = 160 mm); scanning the encoded clip, nothing drawn in
    # the wing colours gets within 100 px of that edge.
    p.add_argument("--force-scale", type=float, default=0.30, help="aero arrow length per N")
    p.add_argument("--trail-pts", type=int, default=90, help="wingtip trail samples per beat")
    p.add_argument("--trail-cycles", type=float, default=1.0, help="beats of trail kept")
    p.add_argument("--trail-width", type=float, default=0.0011)
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--self-check", action="store_true",
                   help="assert each setting hits the control axis it claims, no render")
    args = p.parse_args()

    if args.self_check:
        self_check(args)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("pass 1/2: force, moment and speed traces", flush=True)
    times, force, moment, speeds, lookat, seq = [], [], [], None, None, None
    for rec in play(args):
        if speeds is None:
            speeds = {k: [] for k in rec["speed"]}
            lookat, seq = rec["data"].xpos[1:].mean(axis=0).copy(), rec["seq"]
        times.append(rec["t"])
        force.append(rec["force"])
        moment.append(rec["moment"])
        for k, v in rec["speed"].items():
            speeds[k].append(v)
    times = np.array(times)
    strip, px0, px1 = control_strip(
        times, {k: np.array(v) for k, v in speeds.items()}, np.array(force), np.array(moment),
        seq, args.width, args.plot_height,
        args.out.with_name(args.out.stem + "_traces.png"))

    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = 150.0, -12.0, 1.05

    opt = scene_option()
    caption = captioner(args.height, args.no_captions)
    proc = ffmpeg_pipe(args.out, args.width, args.height + args.plot_height, args.fps)
    n_frames = len(times)
    print(f"pass 2/2: rendering {n_frames} frames, {seq.total:.2f}s sim -> {args.out}", flush=True)

    renderer = None
    for rec in play(args):
        if renderer is None:  # pass 2 builds its own model; bind the renderer to that one
            renderer = mujoco.Renderer(rec["model"], height=args.height, width=args.width)
        renderer.update_scene(rec["data"], camera=cam, scene_option=opt)
        draw_aero_arrows(renderer.scene, rec["aero"], args.force_scale)
        # The rig is world-fixed here, so the trail is already in world frame.
        draw_trail(renderer.scene, rec["trail"], np.zeros(3), np.eye(3), args.trail_width)
        playhead = strip.copy()
        col = int(px0 + (px1 - px0) * rec["t"] / seq.total)
        playhead[:, max(0, col - 1):col + 2] = 30
        px = caption(renderer.render(), seq.label(rec["t"]))
        proc.stdin.write(np.vstack([px, playhead]).tobytes())
        if rec["frame"] % 90 == 0:
            print(f"  frame {rec['frame']:4d}/{n_frames}  t={rec['t']:5.2f}s  "
                  f"phase={math.degrees(relative_phase(rec['model'], rec['data'], rec['gears'])):+6.1f} deg  "
                  f"{seq.label(rec['t']).split('   ')[0]}", flush=True)

    proc.stdin.close()
    proc.wait()
    renderer.close()
    print(f"done: {args.out}  ({n_frames} frames, {seq.total:.2f}s sim)")


if __name__ == "__main__":
    main()
