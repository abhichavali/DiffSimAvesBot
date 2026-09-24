"""Render the three flight-demo configs to MP4 (headless, no mjpython needed).

  1. flying     — the whole robot as one free airframe (tensegrity joint, wings,
                  and body-mounted motors), flapping and holding altitude on a
                  soft vertical weight-assist while pitch/roll/aero stay physical.
  2. multimodal — same airframe with a live stroke-plane schedule that tilts the
                  disks, producing up/down-stroke asymmetry; per-blade aero force
                  arrows are drawn so the asymmetry is visible.
  3. collision  — the bird cruises level down a line of poles and hits them TWICE,
                  and each hit plays the same beat: the wing FEELS the pole (real
                  MuJoCo contact, nothing scripted), sweeps back into a tuck, and
                  the bird glides on out of the obstacle with its wings folded
                  before the motors spool back up and it flies the sag off. Then
                  it flies into a GAP between two poles narrower than its own
                  wingspan, so both wings are caught at once — same event, bigger.
                  Attitude authority is reduced through each knock but never cut:
                  this flapper is open-loop unstable, so cutting it outright shows
                  the airframe diverging rather than the poles hitting it (see
                  --upset-floor).
  4. ceiling    — the same beat against an overhead slab instead of poles. The
                  bird cruises under a soffit, tilts forward and CLIMBS into it
                  until a wing plate physically strikes the underside (again a
                  real MuJoCo contact, not a timestamp), then folds, glides for
                  a second losing height, and flies on underneath with the whole
                  stroke clearing the slab. Shares every envelope with 3.

Frames are piped straight to ffmpeg (already on PATH) — no imageio/opencv dep.
Reuses the existing runtime (control, aero, crank-sync, pitch, disk-orientation).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco  # noqa: E402

import flying  # noqa: E402
from common import (  # noqa: E402
    TIP_SITES,
    airframe_xml,
    attitude_pd,
    draw_aero_arrows,
    draw_trail,
    ffmpeg_pipe,
    lift_free_bodies,
    save_still,
    scene_option,
    smoothstep,
)
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.control_signal import (  # noqa: E402
    eval_control,
    load_control_for_morphology,
    ramp_factor,
)
from runtime.disk_orientation import DiskOrientation  # noqa: E402
from runtime.morphology_runtime import DEFAULT_HUMMER_TG  # noqa: E402
from runtime.sim import PlateWingSim  # noqa: E402

# Vertical weight-assist: F = m g (cancel) + k(z0 - z) - c vz. Horizontal and
# rotational DOFs are untouched, so translation/pitch/roll stay physical while
# the airframe holds frame. Soft for the free-flight showcase (visible bob);
# stiff and near-critically damped for collision, where the cruise has to be
# genuinely level: at c=0.9 (zeta 0.37) the altitude mode rang +-0.05 m, enough
# to throw the wingtip into the ceiling a second after the clip opened, long
# before the scripted climb. c=2.0 is zeta 0.83 against 2*sqrt(k m)=2.42.
# ponytail: k = altitude-hold stiffness, c = vertical damping.
Z_ASSIST = {
    "flying": (4.0, 0.08),
    "multimodal": (4.0, 0.08),
    "collision": (22.0, 2.0),
    # The ceiling clip needs this even more than the poles do: the whole event is
    # a 45 mm bite out of the wing arc, so a ringing altitude mode would either
    # miss the slab or hit it on frame one.
    "ceiling": (22.0, 2.0),
}

# The collision course: one lone pole, then a gap made of two. Grouped by strike
# so a pole hammered over several beats is ONE event, and so the two poles of the
# gap are also one event (see CollisionPlan.note_hit).
POLE_GROUPS = {1: ("pole_a",), 2: ("pole_b", "pole_c")}
POLE_NAMES = [n for g in POLE_GROUPS.values() for n in g]

# The ceiling course: one obstacle, one strike. Same shape as POLE_GROUPS so the
# contact ledger, the strike detection and the beat sheet are shared verbatim.
CEILING_GEOM = "ceiling_slab"
CEILING_GROUPS = {1: (CEILING_GEOM,)}


class WingSweep:
    """Sweep the wing plates aft about the airframe's up axis, kinematically.

    This is the fold. Each wing plate hangs off its strut on the feather hinge
    and carries the mesh, the collision box and all four aero blades, with no
    tendon attached to it, so rewriting ``model.body_quat`` swings the whole
    visible wing — and what the poles collide with, and where the lift is made —
    in one write, the same trick ``runtime/disk_orientation.py`` uses on the
    crank disks.

    It replaces folding via the stroke-plane disks. That route was indirect —
    swing the crank mount and hope the tendons drag the wing round after it — and
    it overcorrected every single time, flipping the wings clean over. Nothing
    about it was tunable; the fold has to be *stated*, not asked for.

    The rotation is applied in the WORLD frame and pushed back through the
    parent, not applied as a fixed local axis: the strut a wing hangs off tumbles
    through the whole stroke, so a fixed local axis sweeps the wing somewhere
    different at every beat phase. About the airframe's own up axis, "back" is
    always back.

    ponytail: kinematic fold, no reaction torque — the plates are placed where
    the shot wants them and the tensegrity shoulder underneath is never asked to
    agree. This clip is a visual demonstration and the fold is the point of the
    shot, not a claim about the mechanism. Upgrade path if that stops being true:
    a real sweep DOF at the shoulder, driven by an actuator.
    """

    def __init__(self, model):
        self.model = model
        self.wings = []
        for side in "LR":
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wing")
            if bid < 0:
                raise ValueError(f"{side}_wing body missing — cannot fold the wings")
            r0 = np.zeros(9)
            mujoco.mju_quat2Mat(r0, model.body_quat[bid])
            # A +y wingtip sweeps aft about +up, a -y wingtip about -up.
            self.wings.append((bid, int(model.body_parentid[bid]), r0.reshape(3, 3).copy(),
                               1.0 if side == "L" else -1.0))
        self._q, self._m = np.zeros(4), np.zeros(9)

    def set(self, data, up, angle):
        for bid, pid, r0, sign in self.wings:
            mujoco.mju_axisAngle2Quat(self._q, up, sign * angle)
            mujoco.mju_quat2Mat(self._m, self._q)
            rp = np.asarray(data.xmat[pid]).reshape(3, 3)
            world = rp.T @ self._m.reshape(3, 3) @ rp @ r0
            mujoco.mju_mat2Quat(self._q, np.ascontiguousarray(world).ravel())
            self.model.body_quat[bid] = self._q


def plate_gap(model, data, gid_a, gid_b):
    """Closest approach of the two wing plates, in metres.

    Corner-to-corner on the two collision boxes. ponytail: an underestimate of
    how close they really are (an edge-edge crossing beats every corner pair),
    which is the safe direction for a "did they overlap" test, and the plates are
    thin enough that corners dominate. Exact box-box distance if this ever needs
    to be tight rather than conservative.
    """
    pts = [plate_corners(model, data, gid) for gid in (gid_a, gid_b)]
    return float(np.min(np.linalg.norm(pts[0][:, None, :] - pts[1][None, :, :], axis=-1)))


def plate_corners(model, data, gid):
    """The 8 world corners of a wing plate's collision box."""
    s = model.geom_size[gid]
    rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
    pos = np.asarray(data.geom_xpos[gid])
    return np.array([pos + rot @ (np.array([sx, sy, sz]) * s)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def plate_top(model, data, gid):
    """World z of the highest corner of a wing plate — what meets the ceiling.

    The corners, not the span centreline `plate_span_pts` uses for the poles: a
    pole is a vertical line the plate sweeps ACROSS, so the centreline is what it
    has to stop, but a slab is a horizontal plane and the plate meets it with
    whichever corner is highest. This is the ceiling's version of
    `pole_axis_gap` — a geometric statement of "did it pass through", computed
    with no help from the contact solver.
    """
    return float(plate_corners(model, data, gid)[:, 2].max())


def wing_up(model, data, gid):
    """How upright a wing plate is: its normal's component along world up.

    The collision box is 4 x 39 x 118 mm, so its thin local x IS the plate
    normal. 1 = level, 0 = knife-edge, negative = upside down. This is the number
    that says the wings never flipped over, which is the failure the old
    stroke-plane fold produced on every run.
    """
    return float(np.asarray(data.geom_xmat[gid]).reshape(3, 3)[2, 0])


# --------------------------------------------------------------------------- #
# Scene building
# --------------------------------------------------------------------------- #
def build_model(config, tg, scene, *, skinned, bird_assets, pole_radius=0.025,
                pole_top=2.4, ceiling_half_y=0.60, ceiling_thickness=0.055,
                ceiling_rgba="1 1 1 1"):
    xml = airframe_xml(tg, scene)
    if config == "collision":
        xml = flying.add_poles(xml, POLE_NAMES, radius=pole_radius, z_hi=pole_top)
        xml = flying.enable_wing_collision(xml)
    if config == "ceiling":
        # A SOFFIT, not a sky-wide slab. The first ceiling clip put a 6 m x 3 m
        # sheet overhead and it rendered as a horizon line: with no edge in shot
        # there was nothing to say the surface was above the bird, and a camera
        # low enough to look up at its underside would have had to be below the
        # floor. Half a metre either side of the wingtips puts both long edges in
        # frame, converging away down the corridor, and 55 mm of thickness makes
        # them a band rather than a hairline. That, plus the checker on the
        # underside, is the whole fix.
        # Bright warm panels with dark seams. The default slab greys came out as
        # a murky brown from below (an underside gets headlight + ambient and no
        # directional light at all), and a ceiling you have to squint at is not
        # much better than one rendered edge-on. castshadow off: the slab is 12 m
        # long, well past the extent of the scene's shadow map, so its shadow
        # landed on the floor as a hard-edged dark quadrilateral that read as a
        # rendering glitch rather than as anything overhead.
        xml = flying.add_ceiling(xml, half_y=ceiling_half_y,
                                 thickness=ceiling_thickness, rgba=ceiling_rgba,
                                 rgb1="0.93 0.87 0.79", rgb2="0.74 0.65 0.56",
                                 markrgb="0.34 0.29 0.25", shadows=False)
        xml = flying.enable_wing_collision(xml)
    if skinned:
        xml = flying.add_visual_bird(xml, assets_dir=bird_assets)
    return mujoco.MjModel.from_xml_string(xml)


# --------------------------------------------------------------------------- #
# Collision beat sheet
# --------------------------------------------------------------------------- #
class CollisionPlan:
    """Two-strike pole clip, in sim seconds since the plan clock started.

      cruise  — level forward flight down the line of poles.
      strike  — a wing catches a pole. The wings keep beating INTO it for
                `strike_seconds`: cutting the drive on the instant of contact
                stopped the wings so fast that the pole never appeared to do
                anything to the wingbeat.
      fold    — the drive fades to the `glide_drive` trickle over
                `stop_seconds` and the wings sweep aft into a tuck (WingSweep).
      glide   — wings folded, motors barely turning, but still FLYING: the speed
                is back to cruise within `glide_pickup` of the release, so the
                glide is what carries the bird clear of the pole it just hit
                rather than parking it in the obstacle. It sags `hit_drop`
                through this, which is what an unpowered glide does.
      recover — after `glide_seconds` the motors spool back up over
                `spool_seconds`, the wings come back out, the sag is flown off
                and the cruise resumes.
      again   — `gap_seconds` of recovered cruise later it flies into a GAP
                between two poles that is narrower than its own wingspan, so
                both wings are caught at once. Same knock; the clip closes on
                the recovery from it.

    The strike times are unscripted — they come from the actual contacts, so
    everything downstream keys off the real hits, not guessed timestamps.

    The VERTICAL assist never lets go. It is not a stabiliser here, it is what
    stands in for gravity: these wings out-lift the cancelled weight by ~0.3 N,
    so a hands-off window doesn't drop the bird, it fires it 0.6 m upward.
    """

    def __init__(self, args):
        self.a = args
        self.hits = {}          # obstacle group -> sim time of its first contact
        self.total = (args.cruise_seconds + args.gap_seconds + args.flap_seconds
                      + 2 * (args.strike_seconds + args.stop_seconds
                             + args.glide_seconds + args.spool_seconds))

    def note_hit(self, t, group):
        """One event per obstacle GROUP.

        Debouncing on time instead would have to separate a pole hammered over
        several beats (many contacts, ~0.3 s apart) from the second strike, and
        that window depends on the flight speed. Grouping by which pole was hit
        needs no window at all: the lone pole is one event and the two poles of
        the gap are one event, by construction.
        """
        self.hits.setdefault(group, t)

    def t_release(self, h):
        """The instant the wing is off the pole — the fold and the glide key off it."""
        return h + self.a.strike_seconds + self.a.stop_seconds

    def t_spool(self, h):
        """End of the unpowered glide; the motors start building here."""
        return self.t_release(h) + self.a.glide_seconds

    def t_powered(self, h):
        """Spool complete: full stroke, cruising again."""
        return self.t_spool(h) + self.a.spool_seconds

    def _knock(self, h, t, floor, t_on=None, ramp=None):
        """One strike's knock-down-and-recover envelope: 1 -> `floor` -> 1.

        `t_on`/`ramp` say when the recovery starts and how long it takes. The
        default is the motor spool, but the SPEED channel comes back on the much
        shorter fold timescale — see speed_target.
        """
        off = 1.0 - smoothstep((t - h - self.a.strike_seconds) / self.a.stop_seconds)
        # Blended from the floor, not max()'d against it: smoothstep is flat at
        # its start, so a decaying trickle plus a rising ramp actually DIPPED
        # below the floor for the first tenth of a second after release.
        on = smoothstep((t - (self.t_spool(h) if t_on is None else t_on))
                        / (self.a.spool_seconds if ramp is None else ramp))
        return max(off, floor + (1.0 - floor) * on)

    def _env(self, t, floor, **kw):
        """The worst knock in force right now. Causal: a strike still in the
        future contributes exactly 1.0, so the whole beat sheet can be built up
        front (which is what lets the poles be PLACED from it, see path_x)."""
        return min([1.0] + [self._knock(h, t, floor, **kw) for h in self.hits.values()])

    def assist(self, t):
        """Attitude hold + station keeping: 1 = flying it, `upset_floor` = knocked.

        The floor is not timidity. The wings are separate free bodies held on by
        tendons alone, so with attitude hold at a hard 0 the airframe thrashes and
        the two wings swing independently — they went 64 deg out of step and beat
        anywhere from 35 to 78 deg, which reads as broken flapping, not as a bird
        taking a knock. A floor caps the airframe's angular excursion.
        """
        f = self.a.upset_floor
        return min([1.0] + [f + (1.0 - f) * smoothstep(
            (t - h - self.a.upset) / self.a.recover) for h in self.hits.values()])

    def speed_target(self, t):
        """Cruise, CHECKED by each strike, then carried on by the glide.

        Speed does not ride the drive envelope. It used to, and that is what made
        the clip read as a knock-then-recover script instead of one movement: the
        bird was held at `stall_frac` for the whole unpowered window, so it sat in
        the obstacle it had just hit — 1.4 s of sim wedged in the gap, grinding —
        and only crept clear once the motors were already back.

        A glide is not a stall. The speed dips at the impact, which is what sells
        the pole actually checking the bird, and is back by the time the fold has
        completed (`glide_pickup`, the fold timescale — NOT `spool_seconds`), so
        the glide is what carries it clear of the poles. Hit, fold, glide out,
        fly on: the wings are still unpowered right through it.
        """
        return self.a.cruise_speed * min(
            [1.0] + [self._knock(h, t, self.a.stall_frac,
                                 t_on=self.t_release(h), ramp=self.a.glide_pickup)
                     for h in self.hits.values()])

    def drive(self, t):
        """Full, knocked down by each strike, then spooling straight back up.

        Once it lets go the drive does not sit at zero waiting to be declared
        clear: it starts turning again at once and builds, so the wings flap
        slowly and get quickly faster.

        It never reaches zero. Letting the cranks stop dead drops the stroke into
        a 20 deg limit cycle it never climbs out of — the mechanism has two stable
        cycles — so `glide_drive` holds the phase alive through the slow part.
        """
        return self._env(t, self.a.glide_drive)

    def wing_sweep(self, t):
        """How far the wings are swept aft right now (rad): 0 cruising, `fold_sweep`
        while crossing the poles.

        Out the moment the wing lets go of the pole, and back to nothing exactly as
        the spool starts — so the stroke picks up the instant the wings are out of
        the tuck, with no dead glide in between. That the two coincide is a safety
        property as much as a timing one: the stroke must never be driven while the
        wings are tucked. The whole tuck is therefore `glide_seconds` long, which is
        the one knob that sets how long the bird crosses with its wings away.
        """
        folds = []
        for h in self.hits.values():
            out = smoothstep((t - self.t_release(h)) / self.a.fold_seconds)
            back = smoothstep(
                (t - (self.t_spool(h) - self.a.fold_seconds)) / self.a.fold_seconds)
            folds.append(out - back)
        return self.a.fold_sweep * max([0.0] + folds)

    def z_target(self, t, z0):
        """Level cruise, sagging `hit_drop` under each strike and flying it back off.

        Reuses the drive envelope rather than carrying a schedule of its own: the
        bird loses height exactly while it has lost its stroke, which is both what
        a knocked bird does and one fewer timeline to keep in step with the rest.
        """
        return z0 - self.a.hit_drop * (1.0 - self._env(t, 0.0))


def ceiling_args(args):
    """The pole beat sheet's knobs, overwritten with the ceiling's own.

    A COPY, not a mutation. The pole numbers are tuned and asserted against each
    other (`--glide-seconds < 0.7`, "the bird should be flapping again as soon as
    it is past the pole") and the ceiling wants the opposite — a full second of
    folded descent, because the recovery here IS losing height. Copying means the
    two clips can disagree without either config gaining an `if`.

    Rebinding a handful of fields rather than threading a second args object
    through the render loop also means every downstream read — the plan, the
    stroke-plane tilt, the attitude target, the servos — picks the ceiling values
    up with no extra plumbing.
    """
    a = copy.copy(args)
    a.strike_seconds = args.ceiling_strike_seconds
    a.glide_seconds = args.ceiling_glide_seconds
    # No tuck here. A pole is something to get a wing PAST, so sweeping it aft
    # earns its keep; a ceiling is not, and the swept wings read as folded back
    # and hunched up when what the shot wants is a bird holding its wings out and
    # skimming along underneath. The recovery is the height, not the pose.
    a.fold_sweep = args.ceiling_fold_sweep
    a.cruise_pitch = args.ceiling_pitch
    a.cruise_lean = args.ceiling_lean
    a.hit_drop = 0.0            # unused: CeilingPlan.z_target replaces it wholesale
    return a


class CeilingPlan(CollisionPlan):
    """One-strike overhead-slab beat, in sim seconds since the plan clock started.

    Every envelope is inherited — drive, assist, speed_target, wing_sweep, the
    knock/recover shape, the release and spool times. Only HEIGHT differs, in the
    two ways that make this a ceiling rather than a pole:

      climb   — the approach is not level. `--ceiling-climb` walks the bird up
                into the slab over the last `--ceiling-climb-seconds` of the
                cruise, so the wings arrive at the ceiling driving forward and
                UP (with the stroke plane and the body already tilted forward,
                see ceiling_args) instead of drifting into it. It is also what
                sets WHEN the strike lands: the slab is planted `--ceiling-bite`
                inside the top of the climbed stroke, so contact happens partway
                up the climb — unscripted, off the real geometry.
      drop    — the sag is PERMANENT. The pole clip flies its sag back off; here
                losing height is the entire recovery, so once the bird has
                dropped `--ceiling-drop` it stays there and flies on underneath.
                Coming back up would fly it straight into the slab again.

    ponytail: both are kinematic scripts written onto the altitude servo's
    setpoint — the bird does not climb because its wings made lift, and it does
    not descend because it stopped making any. This clip is a visual; the beat is
    the point, not the mechanism. Upgrade path if that stops being true: drive
    the height through the stroke-plane amplitude and let the aero do it.
    """

    def __init__(self, args):
        super().__init__(args)
        # One strike, so one knock -- not the pole clip's two, and no gap leg.
        self.total = (args.cruise_seconds + args.strike_seconds + args.stop_seconds
                      + args.glide_seconds + args.spool_seconds + args.flap_seconds)

    def climbed(self, t):
        """Scripted climb into the slab, finishing at `--cruise-seconds`.

        Late rather than spread over the whole approach: the opening beat has to
        be level flight with visible clearance under the ceiling, or there is
        nothing for the climb to be a change FROM.
        """
        r = self.a.ceiling_climb_seconds
        return self.a.ceiling_climb * smoothstep((t - (self.a.cruise_seconds - r)) / r)

    def dropped(self, t):
        """The descent, in two moves — and neither of them ever comes back.

        Two moves, not one. A quick `ceiling_graze_drop` the instant the wings come
        off the slab, sized to do exactly one thing — take the trickling stroke out
        of contact with the underside, so the bird skims a few tens of mm clear
        rather than scraping along it. Then the rest of the drop at the end, to
        make room for the full stroke returning. The first is small enough to read
        as pulling clear, not as falling; the second lands with the stroke.
        """
        out = 0.0
        for h in self.hits.values():
            graze = self.a.ceiling_graze_drop * smoothstep(
                (t - self.t_release(h)) / self.a.ceiling_graze_seconds)
            duck = (self.a.ceiling_drop - self.a.ceiling_graze_drop) * smoothstep(
                (t - (self.t_spool(h) - self.a.ceiling_duck_seconds))
                / self.a.ceiling_duck_seconds)
            out = max(out, graze + duck)
        return out

    def z_target(self, t, z0):
        return z0 + self.climbed(t) - self.dropped(t)


def path_x(args, marks, dt=1e-3):
    """Distance flown by each of `marks` (sim seconds), from the plan's own schedule.

    This is what plants the poles. `x_ref` in the render loop is the exact
    integral of `speed_target`, and the station servo tracks it hard, so replaying
    the schedule says where the bird will BE at the moment each strike is wanted.
    Hardcoded distances stop lining up the moment cruise-speed, stall-frac or the
    knockdown timing changes.
    """
    plan = CollisionPlan(args)
    for i, m in enumerate(marks):
        plan.note_hit(m, i + 1)     # causal, see CollisionPlan._env
    ts = np.arange(0.0, max(marks) + dt, dt)
    v = np.array([plan.speed_target(float(t)) for t in ts])
    x = np.concatenate([[0.0], np.cumsum(v[:-1]) * dt])
    return [float(np.interp(m, ts, x)) for m in marks]


def strike_times(args):
    """When each strike is wanted, in plan seconds. Strike 2 is a full knock,
    glide and spool after strike 1, plus `gap_seconds` of recovered cruise."""
    t1 = args.cruise_seconds
    t2 = (t1 + args.strike_seconds + args.stop_seconds + args.glide_seconds
          + args.spool_seconds + args.gap_seconds)
    return [t1, t2]


def couple_brake(progress, gain):
    """Per-drive velocity correction that holds the LEADERS back to the laggard.

    The four cranks are independent <velocity> servos, so a wing jammed on the
    ceiling slows while the other keeps its commanded speed and the two fall out
    of phase. One motor on a shared driveshaft doesn't do that — obstruct one
    crank and the whole train slows together. This reproduces that: whichever
    crank is furthest behind sets the pace and the rest are braked to it.

    Note the sign. Correcting toward the COMMANDED angle instead would have the
    motors fight the ceiling and erase the collision (tried it: peak roll fell
    from 35 deg to 1 deg). This is never positive — a blocked crank is never
    helped along, the free ones are only ever held back.
    """
    slowest = min(progress)
    return [-gain * (p - slowest) for p in progress]


def pole_contacts(data, pole_gids):
    """(pole gid, other gid, penetration depth m) for every pole contact this step.

    `contact.dist` is MuJoCo's own signed gap, negative inside — the most direct
    measure there is of a wing sinking into a pole, and it needs no geometry
    reconstruction to be believed.
    """
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in pole_gids:
            out.append((g1, g2, -float(c.dist)))
        elif g2 in pole_gids:
            out.append((g2, g1, -float(c.dist)))
    return out


def wing_flap_deg(model, data, gid):
    """Elevation of a wing's span axis above horizontal, in degrees."""
    axis_z = float(np.asarray(data.geom_xmat[gid]).reshape(3, 3)[2, 2])
    return math.degrees(math.asin(max(-1.0, min(1.0, axis_z))))


def wing_pitch_deg(model, data, gid):
    """Wing feathering: tilt of the chord axis, in degrees. Elevation being in
    step says nothing about the wings twisting together."""
    chord_z = float(np.asarray(data.geom_xmat[gid]).reshape(3, 3)[2, 0])
    return math.degrees(math.asin(max(-1.0, min(1.0, chord_z))))


def plate_span_pts(model, data, gid, n=25):
    """`n` world points down the wing plate's SPAN centreline, root to tip.

    The centreline, not the corners: a corner sample can straddle a pole while
    the plate's middle sweeps clean through it, which is exactly the failure this
    is here to catch. The line is what a pole has to stop.
    """
    rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
    hz = float(model.geom_size[gid][2])
    return np.asarray(data.geom_xpos[gid]) + np.outer(np.linspace(-hz, hz, n), rot[:, 2])


def pole_axis_gap(pts, axis_xy):
    """Closest horizontal approach of `pts` to a vertical pole axis.

    The poles run floor-to-overhead, well past the bird either way, so the z
    extent never enters into it — horizontal distance IS the distance.

    ponytail: sampling the span centreline is a LINE-vs-pole test, not the whole
    oriented plate, so in principle a pole could slip through near the leading or
    trailing edge without the centreline ever crossing it. It cannot happen for
    the failure this guards (a broken contype/conaffinity mask lets the wing
    sweep clean through, and the centreline crosses the axis on the way), and the
    MuJoCo contact depth covers the rest. Upgrade path if that stops being true:
    exact line-vs-oriented-box distance instead of a point sample.
    """
    d = np.asarray(pts)[:, :2] - np.asarray(axis_xy)
    return float(np.hypot(d[:, 0], d[:, 1]).min())


# --------------------------------------------------------------------------- #
# Per-config schedules
# --------------------------------------------------------------------------- #
def stroke_plane_schedule(config, disk_orient, tilt0, pitch0, t, drive_start):
    """Drive the crank-disk mount orientation for this step."""
    if disk_orient is None:
        return
    dt = t - drive_start
    if config == "multimodal":
        # One slow precession of the stroke plane: the normal sweeps a cone, so
        # the wing spends part of the cycle on a forward-tilted (asymmetric)
        # plane and part symmetric — the aero arrows swell on the strong stroke.
        phase = 2.0 * math.pi * 0.6 * dt
        amp = 0.22
        disk_orient.set(tilt0 + amp * math.sin(phase), pitch0 + amp * math.sin(phase + math.pi / 2))



# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
CEILING_AZIMUTH = 90.0      # side on, square to the flight path


def make_camera(model, airframe_bid, config):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = airframe_bid
    # Collision: rear quarter, from the side the lone pole is on. The old ceiling
    # shot looked UP from below because a slab overhead renders edge-on from
    # anywhere else; poles are vertical, so the opposite is wanted — an in-plane
    # view that watches the bird fly INTO them. Ahead-looking-back (the old
    # azimuth 150) is unusable here: the tracking camera sits 1 m in front of the
    # bird, so the poles would sweep past BEHIND the lens and every strike would
    # happen off-screen.
    # Ceiling: side on, square to the flight path, barely off level. What this
    # clip has to show is a GAP — how much air is between the wingtips and the
    # slab, before the strike and after the recovery — and a gap is a vertical
    # distance, so it reads best from the side at eye level, where it projects at
    # full size. The earlier shot looked steeply up from below (elevation 19) to
    # get the slab's underside into frame; that foreshortens the very distance the
    # clip is about. The slab does not need its underside visible to read as a
    # ceiling: its thickness makes a band overhead, and the bird flying under it
    # says the rest. 6 degrees keeps a sliver of underside for depth without
    # flattening the gap.
    #
    # It is also HEIGHT-LOCKED, not tracking (the render loop drives `lookat`).
    # A tracking camera rises and falls with the bird, so the one thing this clip
    # exists to show — that it recovers by losing height — is the one thing it
    # cannot show: the bird stays dead centre and the ceiling drifts about
    # instead. Pinning the lookat height nails the slab in place and lets the
    # bird climb into it and then sink away from it.
    if config == "ceiling":
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Azimuth is re-driven per frame in the render loop; this is the opening
        # value, before the strike swings it round to the front.
        cam.azimuth, cam.elevation, cam.distance = CEILING_AZIMUTH, 6.0, 1.60
        return cam
    cam.azimuth = 34.0 if config == "collision" else 150.0
    cam.elevation = -8.0 if config == "collision" else -12.0
    cam.distance = 1.35 if config == "collision" else 0.95
    return cam


# --------------------------------------------------------------------------- #
# Main render loop
# --------------------------------------------------------------------------- #
def render_config(config, args):
    # The ceiling clip is the pole clip's beat sheet with the slab's own timings
    # and a steeper forward tilt -- see ceiling_args for why that is a rebind and
    # not a branch.
    if config == "ceiling":
        args = ceiling_args(args)
    # Everything downstream that is about "the bird is flying a beat sheet at an
    # obstacle" is shared; only the obstacle GEOMETRY differs between the two.
    obstacle = config in ("collision", "ceiling")
    tg = args.tg
    scene = args.scene
    skinned = args.skin == "all" or (args.skin == "flying" and config == "flying")
    model = build_model(config, tg, scene, skinned=skinned, bird_assets=args.bird_assets,
                        pole_radius=args.pole_radius, pole_top=args.pole_top,
                        ceiling_half_y=args.ceiling_half_y,
                        ceiling_thickness=args.ceiling_thickness)
    morph = load_morphology(tg)
    control = load_control_for_morphology(tg)

    sim = PlateWingSim(model, morph)
    data = sim.data

    # Lift the whole robot off the floor so it reads as airborne (compiled COM
    # sits ~0.1 m up with the motor pillars dangling to the ground plane). Every
    # free body — the airframe AND the mobile struts it is tendon-coupled to —
    # must rise together, or the tendons stretch and explode.
    # Collision flies high: the poles run floor-to-overhead, and a bird cruising
    # a metre up has them sweeping past it top AND bottom, which is what sells
    # them as poles. At the 0.28 m showcase altitude they read as short stumps.
    # The ceiling clip flies just as high, for a different reason: the camera has
    # to get UNDER the bird to look up at the slab, and at the 0.28 m showcase
    # altitude there is no room between the bird and the floor to put it.
    lift_free_bodies(model, data, args.collision_lift if obstacle else args.lift)

    motor_dir = morph.motors.motor_direction
    ramp_seconds = 2.0 * control.period
    drive_start = data.time

    airframe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flying.AIRFRAME_BODY)
    m_total = float(model.body_subtreemass[airframe_bid])
    g = float(-model.opt.gravity[2])
    z0 = float(data.subtree_com[airframe_bid][2])
    prev_z = [z0]

    disk_orient = None
    tilt0 = pitch0 = 0.0
    if config in ("multimodal", "collision", "ceiling"):
        disk_orient = DiskOrientation(model)
        tilt0 = morph.motors.disk.tilt_rad
        pitch0 = morph.motors.disk.pitch_rad

    air_dofadr = int(model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)])

    plan = None
    tip_gids = []
    # The plan clock starts when the CLIP does, i.e. after the warmup. Leaving it
    # at drive_start ran the whole approach during the unrecorded warmup, so the
    # clip opened with the bird already parked in the obstacle.
    plan_start = [drive_start + (args.warmup if obstacle else 0.0)]
    x_ref = [float(data.subtree_com[airframe_bid][0])]
    phase_ref = []                                # per-drive crank angle at clip start
    flap_lo, flap_hi = [9e9, 9e9], [-9e9, -9e9]   # per-wing beat extremes since the last print
    flap_gap = [0.0]                              # worst |L-R| elevation, i.e. how out of step
    pitch_gap = [0.0]                             # worst |L-R| feathering asymmetry
    obstacle_gids = {}                            # geom id -> strike group
    pole_xy = {}                                  # geom id -> planted (x, y)
    reach = [0.0, 9e9]                            # wing plate lateral reach: max, min-over-beat
    peak = [0.0]                                  # wing plate reach ABOVE the COM (ceiling clip)
    cam_hold = [0.0]                              # locked camera framing height (ceiling clip)
    # The ceiling's own ledger, alongside the pole one below. `pierce` is the
    # slab's answer to `clear`: how far the highest plate corner ever got PAST the
    # underside (negative = under it, where it belongs). `after` is the highest a
    # plate ever got once the glide was over, i.e. the direct statement that the
    # bird ended up low enough for its whole stroke to clear.
    # `wings` is per-PLATE, unlike `touch`, which is per strike group: the brief
    # says "its wings strike the ceiling", plural, and a slab that only ever
    # caught one wingtip would read as a clip, not as flying into a ceiling.
    slab = {"bottom": 9e9, "gid": -1, "pierce": -9e9, "after": -9e9, "wings": {}}
    # The contact ledger. This is the whole "did the wings actually HIT the poles,
    # or slide through them" question, measured rather than assumed:
    #   touch  — steps with a wing plate <-> pole contact pair, per group
    #   sink   — deepest MuJoCo contact penetration seen (m)
    #   clear  — closest the plate CENTRELINE ever came to a pole axis (m). If a
    #            mask were wrong this collapses toward zero as the wing sweeps
    #            straight through; contact holds it at roughly the pole radius.
    #   bodygap— closest the airframe COM came to a pole axis, i.e. proof the
    #            fuselage (a ghost to these poles) never flew through one.
    #   last   — sim time of the LAST wing<->pole contact in each group, so
    #            hit -> last is how long the wing was ON the obstacle. That is the
    #            whole "fluid, never wedged" claim as a number, and it is the
    #            measure that caught the bird grinding in the gap for over a
    #            second. (Waiting for the COM to pass the pole AXIS is too kind:
    #            the wing sweeps aft of the body and is still on the pole after
    #            the fuselage is nominally past it.)
    #   upright— least upright either wing plate ever was (normal . world up),
    #            split either side of the first strike. It is measured against the
    #            bird's OWN approach stroke, not an absolute limit, because this
    #            wing feathers: the plate reverses pitch every half stroke, so the
    #            normal legitimately swings +0.99 -> -0.62 with nothing wrong. An
    #            absolute floor cannot tell that apart from lying on its back --
    #            it either fails every healthy stroke or permits a real flip. What
    #            IS meaningful is whether the fold takes the wing anywhere the
    #            untouched approach does not already go.
    groups = CEILING_GROUPS if config == "ceiling" else POLE_GROUPS
    touch = {gi: 0 for gi in groups}
    sink, clear, bodygap = [0.0], [9e9], [9e9]
    plategap = {}        # per strike group: the two wings' closest approach while tucked
    upright, up_free = [9e9, -9e9], [9e9, -9e9]   # [least, most] upright, per stretch
    last = {}
    sweep = None

    # Wingtip trail, collision clip only: the point of the clip is what the poles
    # do to the stroke, and the truncated loop shows that far better than the
    # wings alone. Sampled off the PHYSICS clock -- one flap is ~control.period and
    # a 30 fps frame is a tenth of that, so a per-frame trail is a hexagon.
    trail = tip_ids = None
    trail_every = 1
    step_n = [0]
    if obstacle:
        tip_ids = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                   for s, n in TIP_SITES.items()}
        missing = [s for s, i in tip_ids.items() if i < 0]
        if missing:
            raise ValueError(f"missing wingtip sites: {missing}")
        trail_every = max(1, round(control.period / (float(model.opt.timestep) * args.trail_pts)))
        trail = {s: deque(maxlen=int(args.trail_pts * args.trail_cycles)) for s in TIP_SITES}
        plan = CeilingPlan(args) if config == "ceiling" else CollisionPlan(args)
        tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_wing_collision")
                    for s in "LR"]
        # Poles are named bodies with a `_geom` suffix, the slab is one geom; past
        # this point they are just "obstacle geom -> which strike it belongs to".
        obstacle_gids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              n if config == "ceiling" else f"{n}_geom"): gi
            for gi, names in groups.items() for n in names}
        if min(obstacle_gids) < 0 or min(tip_gids) < 0:
            raise ValueError("collision geoms missing — "
                             "add_poles/add_ceiling/enable_wing_collision failed")
        sweep = WingSweep(model)
        if config == "ceiling":
            slab["gid"] = min(obstacle_gids)

    # Cruise feathering, to blend away from and back to through the strike.
    pitch0_free = (sim.pitch.upstroke_pitch, sim.pitch.downstroke_pitch)

    z_k, z_c = Z_ASSIST[config]
    dt = float(model.opt.timestep)
    steps_per_frame = max(1, round((args.realtime_factor / args.fps) / dt))

    def assist(_m, _d):
        com = np.asarray(data.subtree_com[airframe_bid])
        com_z = float(com[2])
        vz = (com_z - prev_z[0]) / dt
        prev_z[0] = com_z
        if not obstacle:
            # Vertical weight-assist, applied at the airframe subtree COM so it
            # induces no spurious torque.
            fz = m_total * g + z_k * (z0 - com_z) - z_c * vz
            mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, fz]), np.zeros(3),
                              com, airframe_bid, data.qfrc_applied)
        else:
            t = data.time - plan_start[0]
            # Beat amplitude has to be sampled every STEP: the flap period and the
            # frame interval are both ~0.1 s, so per-frame samples alias.
            flaps = [wing_flap_deg(model, data, g_) for g_ in tip_gids]
            for k, fd in enumerate(flaps):
                flap_lo[k], flap_hi[k] = min(flap_lo[k], fd), max(flap_hi[k], fd)
            flap_gap[0] = max(flap_gap[0], abs(flaps[0] - flaps[1]))
            pitches = [wing_pitch_deg(model, data, g_) for g_ in tip_gids]
            pitch_gap[0] = max(pitch_gap[0], abs(abs(pitches[0]) - abs(pitches[1])))
            # Before the first strike nothing has touched the wings and no fold is
            # applied, so that stretch IS the free-flapping baseline.
            acc = upright if plan.hits else up_free
            ups = [wing_up(model, data, g_) for g_ in tip_gids]
            acc[0] = min(acc[0], min(ups))
            acc[1] = max(acc[1], min(ups))   # min over the two wings, max over time
            # Contact ledger. Every step, not every frame: contact with a pole
            # lasts a few ms and a 30 fps sample would miss most of the strike.
            for pg, other, depth in pole_contacts(data, obstacle_gids):
                if other not in tip_gids:
                    continue        # not a wing plate -> not a strike (should not happen)
                grp = obstacle_gids[pg]
                touch[grp] += 1
                last[grp] = t
                sink[0] = max(sink[0], depth)
                if slab["gid"] >= 0:
                    side = "L" if other == tip_gids[0] else "R"
                    slab["wings"][side] = slab["wings"].get(side, 0) + 1
                if grp not in plan.hits:
                    plan.note_hit(t, grp)
                    print(f"  [strike {grp}] t={t:.2f}s "
                          f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pg)} x "
                          f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)} "
                          f"depth={depth * 1000:.2f}mm", flush=True)
            # Per strike, because the two do not behave alike: the lone pole hits
            # ONE wing, so that wing gets driven across while the other does not,
            # and the closest approach there is worse than at the symmetric gap.
            if plan.wing_sweep(t) > 0.0:
                grp = max(plan.hits, key=plan.hits.get)      # the strike now in force
                plategap[grp] = min(plategap.get(grp, 9e9), plate_gap(model, data, *tip_gids))
            if pole_xy:
                for g_ in tip_gids:
                    pts = plate_span_pts(model, data, g_)
                    for pg, xy in pole_xy.items():
                        clear[0] = min(clear[0], pole_axis_gap(pts, xy))
                for xy in pole_xy.values():
                    bodygap[0] = min(bodygap[0], float(np.hypot(com[0] - xy[0], com[1] - xy[1])))
            if slab["bottom"] < 9e9:
                # The slab's "did it pass through" measure, and its "is it low
                # enough now" one. Both off the plate corners, both every step:
                # the strike is a few milliseconds long and a per-frame sample
                # would step straight over it.
                top = max(plate_top(model, data, g_) for g_ in tip_gids)
                slab["pierce"] = max(slab["pierce"], top - slab["bottom"])
                if plan.hits and t >= plan.t_spool(min(plan.hits.values())):
                    slab["after"] = max(slab["after"], top)
            a = plan.assist(t)
            # `a` gates ONLY attitude + station keeping (see CollisionPlan): the
            # vertical channel stands in for gravity and never lets go.
            #
            # The HORIZONTAL channel gets its own floor from the release onward.
            # Gating it by `a` too meant that after the strike the bird merely
            # coasted, bleeding 0.96 m/s down to 0.5 over the glide; a glide is
            # supposed to hold its speed, and this is the channel that stands in
            # for the airspeed it would be trading altitude for.
            gliding = any(t >= plan.t_release(h) for h in plan.hits.values())
            ax = max(a, args.glide_hold) if gliding else a
            if a > 0.999 or gliding:
                x_ref[0] += plan.speed_target(t) * dt
            else:
                x_ref[0] = float(com[0])  # no control, no reference -> no snap back
            # Anti-windup, and it is the difference between fluid and a lurch.
            # While a pole is physically holding the bird up the reference keeps
            # integrating at cruise speed, so the station error banks — it reached
            # 0.17 m — and the moment the wing came free the servo fired the bird
            # forward at 1.03 m/s against a 0.6 m/s cruise. That whip is also what
            # threw the free wings through a 78 deg swing and dropped the stroke
            # into its low limit cycle, from which it never recovered (the beat
            # sat at 18 deg for the whole approach to the gap instead of 51).
            # Leashing the reference to the body lets the speed servo do the work.
            x_ref[0] = min(x_ref[0], float(com[0]) + args.ref_leash)
            vel = data.qvel[air_dofadr : air_dofadr + 3]
            f = np.array([
                ax * m_total * (args.pos_gain * (x_ref[0] - float(com[0]))
                                + args.speed_gain * (plan.speed_target(t) - float(vel[0]))),
                ax * m_total * (-args.pos_gain * float(com[1])
                               - args.speed_gain * float(vel[1])),
                m_total * g + z_k * (plan.z_target(t, z0) - com_z) - z_c * vz,
            ])
            # Attitude PD toward the cruise lean.
            tau = a * attitude_pd(data, airframe_bid, air_dofadr, args.cruise_lean,
                                  args.level_kp, args.level_kd)
            mujoco.mj_applyFT(model, data, f, tau, com, airframe_bid, data.qfrc_applied)

    def drive_ctrl():
        ramp = ramp_factor(data.time, drive_start, ramp_seconds)
        if plan is None:
            return ramp * motor_dir * eval_control(control, data.time)
        # The strike knocks the motors out: `env` takes the drive AND the
        # drivetrain coupling to zero together, leaving the servos as pure
        # dampers so the cranks coast down into the glide.
        env = ramp * plan.drive(data.time - plan_start[0])
        base = env * motor_dir * eval_control(control, data.time)
        # Drivetrain coupling: the struck wing drags the other one with it.
        if not phase_ref:
            phase_ref.extend(float(data.actuator_length[aid]) for aid in sim.crank_ids)
        progress = [float(data.actuator_length[aid]) - p0
                    for aid, p0 in zip(sim.crank_ids, phase_ref)]
        ctrl = [base + env * brake for brake in couple_brake(progress, args.drive_couple)]
        step_n[0] += 1
        if trail is not None and step_n[0] % trail_every == 0:
            # Body-local, so it is the path the bird sees, not a smear of where
            # the bird has flown.
            com = np.asarray(data.subtree_com[airframe_bid])
            rot_t = np.asarray(data.xmat[airframe_bid]).reshape(3, 3).T
            for side, sid in tip_ids.items():
                trail[side].append(rot_t @ (np.asarray(data.site_xpos[sid]) - com))
        return ctrl

    def physics_step():
        if plan is None:
            stroke_plane_schedule(config, disk_orient, tilt0, pitch0, data.time, drive_start)
        else:
            # The stroke plane is FIXED at the cruise tilt for the whole clip. It
            # used to carry the fold, and it is why the wings kept flipping over:
            # swinging the crank mount and waiting for the tendons to drag the
            # wing round always overshot. The fold is now written straight onto
            # the wing plates instead — see WingSweep.
            disk_orient.set(0.0, args.cruise_pitch)
            sweep.set(data, np.ascontiguousarray(
                np.asarray(data.xmat[airframe_bid]).reshape(3, 3)[:, 2]),
                plan.wing_sweep(data.time - plan_start[0]))
        if plan is not None and plan.hits:
            # Feather flat through the glide: the pitch hinge is nearly free, so
            # the unloaded wings otherwise sit at their full cruise angle of
            # attack while gliding. Only the free ANGLE is narrowed -- stiffening
            # the spring as well drove the two wings 105 deg out of step.
            # Rides the drive envelope so it comes back with the stroke, and so
            # the SECOND strike gets the same treatment as the first for free.
            flat = 1.0 - (1.0 - args.glide_feather) * (
                1.0 - plan.drive(data.time - plan_start[0]))
            sim.pitch.upstroke_pitch = pitch0_free[0] * flat
            sim.pitch.downstroke_pitch = pitch0_free[1] * flat
        # drive_ctrl reads plan.drive before assist's contact ledger notes a strike;
        # a strike noted this step leaves plan.drive at exactly 1.0, so the order is moot.
        sim.step(drive_ctrl(), assist)

    if config == "collision":
        # Flap unrecorded for a moment with the poles parked out of the way and
        # watch how far the wing plate actually reaches SIDEWAYS — then plant them
        # inside that reach. Measuring beats a hardcoded y, which stops being a
        # collision the moment the design, the flap amplitude or the stroke-plane
        # tilt changes.
        # Only the tail of the warmup counts: the drive ramp is over in 2 periods
        # but the tendon-driven stroke keeps growing for several more, so sampling
        # through the build-up reads the reach low.
        while data.time - drive_start < args.warmup:
            physics_step()
            if data.time - drive_start > args.warmup - args.peak_window:
                cy = float(data.subtree_com[airframe_bid][1])
                r = max(float(np.abs(plate_span_pts(model, data, g_)[:, 1] - cy).max())
                        for g_ in tip_gids)
                reach[0], reach[1] = max(reach[0], r), min(reach[1], r)
        com0 = np.asarray(data.subtree_com[airframe_bid])
        x_ref[0] = float(com0[0])
        # reach[1] is the reach at the TOP of the stroke, where the wing is most
        # upright and least far out. A pole inside it cannot be missed whatever
        # the beat phase happens to be when the bird arrives; a pole between
        # reach[1] and reach[0] is only caught over part of the cycle.
        y1 = args.pole1_frac * reach[0]
        y2 = args.pole23_frac * reach[0]
        stations = path_x(args, strike_times(args))
        # The lone pole sits on the CAMERA side (-y, see make_camera) so the
        # strike happens in the foreground rather than behind the airframe.
        plant = [("pole_a", stations[0], -y1),
                 ("pole_b", stations[1], -y2), ("pole_c", stations[1], +y2)]
        for name, dx, dy in plant:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            xy = (float(com0[0]) + dx + args.pole_lead, float(com0[1]) + dy)
            data.mocap_pos[int(model.body_mocapid[model.geom_bodyid[gid]])] = [xy[0], xy[1], 0.0]
            pole_xy[gid] = xy
        print(f"[collision] wing plate reaches {reach[0] * 1000:.0f} mm out "
              f"({reach[1] * 1000:.0f} mm at the top of the stroke); "
              f"pole radius {args.pole_radius * 1000:.0f} mm; "
              f"lone pole at y={-y1 * 1000:+.0f} mm (bites "
              f"{(reach[0] - y1) * 1000:.0f} mm of wing), gap poles at "
              f"y={y2 * 1000:+.0f} mm -> clear gap {(2 * y2 - 2 * args.pole_radius) * 1000:.0f} "
              f"mm vs {2 * reach[0] * 1000:.0f} mm of span", flush=True)
        print(f"[collision] strikes wanted at t={strike_times(args)} s, "
              f"dx={[round(s, 3) for s in stations]} m", flush=True)

    if config == "ceiling":
        # Same trick as the poles, turned through 90 degrees: flap unrecorded with
        # the slab parked overhead, watch how high the wing plate actually gets
        # ABOVE the COM, and only then drop the ceiling into that arc. A hardcoded
        # height stops being a collision the moment the design, the flap amplitude
        # or the stroke-plane tilt changes — and this clip runs a steeper tilt than
        # the pole one does, which moves the peak on its own.
        while data.time - drive_start < args.warmup:
            physics_step()
            if data.time - drive_start > args.warmup - args.peak_window:
                cz = float(data.subtree_com[airframe_bid][2])
                peak[0] = max(peak[0], max(plate_top(model, data, g_)
                                           for g_ in tip_gids) - cz)
        com0 = np.asarray(data.subtree_com[airframe_bid])
        x_ref[0] = float(com0[0])
        # The whole event in one line. The slab sits `--ceiling-bite` INSIDE the
        # top of the climbed stroke, so:
        #   * on the level approach the wings miss it by (climb - bite),
        #   * at the top of the climb they are driven `bite` into it and stopped,
        #   * once the bird has dropped `--ceiling-drop` they clear it by
        #     (drop - bite) with the stroke back at full amplitude.
        # Every one of those three is measured back out of the render below.
        slab["bottom"] = float(com0[2]) + args.ceiling_climb + peak[0] - args.ceiling_bite
        # Midway between the two EXTREMES of the flight (the top of the climb and
        # whichever is lower of the start and the end of the drop), so the bird
        # has room to move both ways inside a frame that never moves.
        cam_hold[0] = float(com0[2]) + 0.5 * (
            args.ceiling_climb + min(0.0, args.ceiling_climb - args.ceiling_drop))
        gid = slab["gid"]
        # Reaching well behind the bird and far ahead of it: the soffit has to be
        # overhead from frame one (an edge sliding into shot mid-clip reads as the
        # ceiling moving, not the bird) and must outlast the whole flight.
        data.mocap_pos[int(model.body_mocapid[model.geom_bodyid[gid]])] = [
            float(com0[0]) + args.ceiling_ahead, float(com0[1]),
            slab["bottom"] + args.ceiling_thickness / 2.0]
        print(f"[ceiling] wing plate peaks {peak[0] * 1000:.0f} mm above the COM; "
              f"slab underside at z={slab['bottom']:.3f} m "
              f"(bird z={com0[2]:.3f}) -> approach clearance "
              f"{(args.ceiling_climb - args.ceiling_bite) * 1000:.0f} mm, "
              f"bite {args.ceiling_bite * 1000:.0f} mm at the top of the "
              f"{args.ceiling_climb * 1000:.0f} mm climb, "
              f"clearance after the {args.ceiling_drop * 1000:.0f} mm drop "
              f"{(args.ceiling_drop - args.ceiling_bite) * 1000:.0f} mm", flush=True)

    sim_seconds = plan.total if obstacle else args.seconds * args.realtime_factor
    n_frames = int(sim_seconds / args.realtime_factor * args.fps)

    # Skin opacity fade: torso starts opaque and fades to its resting alpha over
    # the first FADE_FRAC of the clip, revealing the mechanism inside.
    body_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "bird_body_mat")
    fade_frames = int(args.fade_frac * n_frames) if skinned else 0

    # ponytail: --dry-run skips the renderer and the encoder and runs the physics
    # and the ledger alone. Offscreen rendering is most of the wall clock, so this
    # is what makes "sweep a parameter and read the contact numbers back" a thing
    # you can actually do in a loop instead of guessing between ten-minute renders.
    renderer = None if args.dry_run else mujoco.Renderer(
        model, height=args.height, width=args.width)
    cam = make_camera(model, airframe_bid, config)
    scn_opt = scene_option()

    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / f"config_{config}.mp4"
    proc = None if args.dry_run else ffmpeg_pipe(mp4, args.width, args.height, args.fps)
    still_frames = set() if args.dry_run else {int(n_frames * f) for f in args.still_fracs}

    print(f"[{config}] rendering {n_frames} frames "
          f"({steps_per_frame} steps/frame, rt={args.realtime_factor}) -> {mp4}", flush=True)
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            physics_step()
        if fade_frames and body_mat >= 0:
            f = min(1.0, frame / fade_frames)
            model.mat_rgba[body_mat][3] = 1.0 + (flying.BODY_ALPHA - 1.0) * f
        if renderer is None:
            continue
        if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            # Follow the bird downrange, hold the framing height (see make_camera).
            com = np.asarray(data.subtree_com[airframe_bid])
            cam.lookat[:] = [float(com[0]), float(com[1]), cam_hold[0]]
            # Swing round to the front on the strike. The side view is what reads
            # the approach — the gap closing is a vertical distance and side-on is
            # where it projects at full size — but once the wings are folded the
            # interesting thing is that they are folded SYMMETRICALLY, and that is
            # only visible head on. Driven off the real strike time, so the move
            # lands with the hit rather than on a frame count.
            if plan is not None and plan.hits:
                u = smoothstep((data.time - plan_start[0]
                                - min(plan.hits.values())) / args.ceiling_pan_seconds)
                cam.azimuth = args.ceiling_azimuth + u * (
                    args.ceiling_front_azimuth - args.ceiling_azimuth)
        renderer.update_scene(data, camera=cam, scene_option=scn_opt)
        if config != "flying":
            # The obstacle clips get them at a shorter scale: the point there is
            # that the arrows COLLAPSE when the drive dies and swell back as it
            # spools, and at the multimodal scale they are longer than the bird
            # and hide it.
            draw_aero_arrows(renderer.scene, sim.aero,
                             args.force_scale * (args.collision_arrow if obstacle else 1.0))
        if trail is not None:
            draw_trail(renderer.scene, trail,
                       np.asarray(data.subtree_com[airframe_bid]),
                       np.asarray(data.xmat[airframe_bid]).reshape(3, 3),
                       args.trail_width)
        px = renderer.render()
        proc.stdin.write(px.tobytes())
        if frame in still_frames:
            save_still(args.out / f"config_{config}_f{frame:04d}.png", px)
        if plan is not None and frame % 15 == 0:
            t = data.time - plan_start[0]
            roll = math.degrees(math.asin(max(-1.0, min(1.0, data.xmat[airframe_bid][7]))))
            print(f"  t={t:4.2f}s assist={plan.assist(t):.2f} "
                  f"z={data.subtree_com[airframe_bid][2]:.3f}"
                  f"(tgt {plan.z_target(t, z0):.3f}) "
                  f"x={data.subtree_com[airframe_bid][0]:+.3f} "
                  f"vx={data.qvel[air_dofadr]:+.2f} roll={roll:+5.1f}deg "
                  f"beatL={flap_hi[0] - flap_lo[0]:5.1f} beatR={flap_hi[1] - flap_lo[1]:5.1f} "
                  f"LRgap={flap_gap[0]:5.1f} sink={sink[0] * 1000:4.1f}mm "
                  # The pole clip's "how near a pole axis" and the ceiling's "how
                  # far past the slab underside" are the same question about two
                  # different obstacles, so they share the column.
                  f"{'pierce' if slab['gid'] >= 0 else 'clear'}="
                  f"{(slab['pierce'] if slab['gid'] >= 0 else clear[0]) * 1000:+6.1f}mm "
                  f"drive={plan.drive(t):.2f} "
                  f"hits={sorted(plan.hits)}", flush=True)
            flap_lo[:], flap_hi[:] = [9e9, 9e9], [-9e9, -9e9]
            flap_gap[0] = pitch_gap[0] = 0.0
    if proc is not None:
        proc.stdin.close()
        proc.wait()
        renderer.close()
    if config == "collision":
        contact_report(args, plan, touch, sink[0], clear[0], bodygap[0], last,
                       upright, up_free, plategap, float(model.opt.timestep))
    if config == "ceiling":
        ceiling_report(args, plan, touch, sink[0], slab, last, upright, up_free,
                       plategap, float(model.opt.timestep))
    print(f"[{config}] done: {mp4}", flush=True)


def contact_report(args, plan, touch, sink, clear, bodygap, last, upright, up_free,
                   plategap, dt):
    """The one runnable check the pole clip leaves behind: did the wings HIT?

    A wing sliding through a pole because of a contype/conaffinity mask, a geom
    group or a missing pair is the failure mode this whole config is exposed to,
    and it looks like a perfectly good render right up until you watch it. So it
    is asserted, not eyeballed:

      * every pole group must actually register wing<->pole contacts,
      * the deepest MuJoCo penetration must stay inside `--max-sink`,
      * the wing plate's span CENTRELINE must never get closer to a pole axis
        than the pole's radius (less a tolerance) — the direct statement of "it
        did not pass through", and the one that would collapse to ~0 if contact
        silently stopped happening,
      * the airframe COM must clear every pole axis, since the poles are masked
        to ignore the fuselage and only geometry keeps it out of them,
      * and a wing may spend no more than `--clear-seconds` actually TOUCHING any
        one obstacle. That is the "fluid" requirement made checkable: striking the
        poles was never the hard part, not getting stuck in them was. The bird used
        to grind in the gap for well over a second waiting for its motors, and
        nothing in the render objected,
      * and the strikes and the fold may not roll a wing further over than the
        bird's own untouched approach stroke already does. The fold used to be
        driven through the stroke-plane disks, which overcorrected every run and
        flipped the wings clean over; that is now written directly (see
        WingSweep), and this is what stops it coming back silently. Measured
        against the approach rather than an absolute floor -- see the note on
        `upright` in render_config for why a fixed limit cannot work on a wing
        that feathers.
    """
    dwell = {gi: last.get(gi, float("inf")) - h for gi, h in plan.hits.items()}
    print(f"[collision] contact ledger: touch={touch} deepest_sink={sink * 1000:.2f}mm "
          f"closest_centreline={clear * 1000:.1f}mm (pole radius "
          f"{args.pole_radius * 1000:.0f}mm) closest_body={bodygap * 1000:.0f}mm "
          f"wing-on-pole={ {k: round(touch.get(k, 0) * dt, 3) for k in sorted(plan.hits)} }s "
          f"span={ {k: round(v, 2) for k, v in sorted(dwell.items())} }s "
          f"wing_up struck=[{upright[0]:+.2f},{upright[1]:+.2f}] "
          f"approach=[{up_free[0]:+.2f},{up_free[1]:+.2f}] "
          f"tucked_plate_gap={ {k: round(v * 1000) for k, v in sorted(plategap.items())} }mm",
          flush=True)
    missed = [gi for gi in POLE_GROUPS if touch.get(gi, 0) == 0]
    assert not missed, f"wings never touched pole group(s) {missed} — they phased through"
    assert set(plan.hits) == set(POLE_GROUPS), \
        f"only strikes {sorted(plan.hits)} fired, wanted {sorted(POLE_GROUPS)}"
    assert sink <= args.max_sink, \
        f"wing sank {sink * 1000:.1f} mm into a pole (limit {args.max_sink * 1000:.0f} mm)"
    assert clear >= args.pole_radius - args.max_sink, \
        (f"wing centreline came within {clear * 1000:.1f} mm of a pole axis — that is "
         f"inside the {args.pole_radius * 1000:.0f} mm pole, i.e. it went THROUGH it")
    assert bodygap >= args.pole_radius + args.body_clearance, \
        (f"airframe passed {bodygap * 1000:.0f} mm from a pole axis; the poles do not "
         f"collide with the body, so it would render as flying through one")
    # Time actually TOUCHING, not first-to-last. The span includes the gaps, so a
    # wing that brushes a pole, comes off, flaps and clips it again on the way past
    # scores the same as one wedged against it the whole time -- and now that the
    # fold is only a gesture (see fold_sweep: the wings CANNOT tuck inside the
    # poles without the plates meeting each other), brushing more than once is the
    # normal way through. Contact seconds tell wedged from brushing; the span
    # cannot.
    held = {gi: touch.get(gi, 0) * dt for gi in plan.hits}
    stuck = {gi: round(d, 3) for gi, d in held.items() if d > args.clear_seconds}
    assert not stuck, \
        (f"wing was in contact with obstacle(s) for {stuck} s (limit "
         f"{args.clear_seconds} s) — that is the wedged-and-grinding beat, not a crossing")
    # RECOVERY, not depth of roll, is what separates "knocked" from "flipped".
    # A feathering wing reverses its plate every half stroke and comes back level
    # every beat; the strike legitimately rolls it further over still (measured
    # -0.85 against the approach's own -0.69). A wing that has actually gone over
    # on its back is the one that never comes back up -- so the test is that after
    # the strikes the plates still return as level as they did on the approach.
    assert upright[1] >= up_free[1] - args.wing_up_margin, \
        (f"after being struck the wing plates only came back to {upright[1]:+.2f} upright, "
         f"against {up_free[1]:+.2f} on the approach (margin {args.wing_up_margin:.2f}) "
         f"— they are not recovering, i.e. a wing is over on its back")
    print("[collision] contact verified: wings struck both pole groups, nothing passed "
          "through, and it flew clear of both without wedging", flush=True)


def ceiling_report(args, plan, touch, sink, slab, last, upright, up_free, plategap, dt):
    """The ceiling clip's runnable check: did the wings HIT the slab, and did
    dropping actually get them out from under it?

    Same failure mode as the poles — a mask, a geom group or a missing pair, and
    the wing sails through the ceiling in a render that otherwise looks perfect —
    plus one the poles do not have: this clip's whole recovery is a scripted
    descent, and a descent that is too shallow leaves the bird flapping into the
    slab for the rest of the clip. Both are asserted rather than eyeballed:

      * the wings must register real wing<->slab contacts (the strike is not a
        timestamp; every downstream beat keys off this),
      * they must strike AFTER a real approach, not on frame one — the first
        ceiling clip's altitude mode did exactly that,
      * the highest plate corner must never get more than `--max-sink` past the
        slab underside. That is the geometric "it did not pass through", computed
        from the plate corners with no help from the contact solver, and it is
        what collapses to a large positive number if contact silently stops,
      * a wing may spend no more than `--clear-seconds` touching the slab,
      * once the glide is over, the whole stroke must clear the underside by
        `--ceiling-clearance` and never touch it again. This is requirements 3
        and 4 of the brief as two numbers: low enough, and stays flying,
      * and, as for the poles, the strike and the fold may not roll a wing any
        further over than the bird's own untouched approach stroke does.
    """
    clearance = slab["bottom"] - slab["after"]
    print(f"[ceiling] contact ledger: touch={touch} per_wing={slab['wings']} "
          f"deepest_sink={sink * 1000:.2f}mm "
          f"deepest_pierce={slab['pierce'] * 1000:+.2f}mm (past the underside; "
          f"negative is under it) wing-on-slab="
          f"{ {k: round(touch.get(k, 0) * dt, 3) for k in sorted(plan.hits)} }s "
          f"struck at t={ {k: round(v, 2) for k, v in sorted(plan.hits.items())} }s "
          f"last touch={ {k: round(v, 2) for k, v in sorted(last.items())} }s "
          f"post-glide clearance={clearance * 1000:.0f}mm "
          f"wing_up struck=[{upright[0]:+.2f},{upright[1]:+.2f}] "
          f"approach=[{up_free[0]:+.2f},{up_free[1]:+.2f}] "
          f"tucked_plate_gap={ {k: round(v * 1000) for k, v in sorted(plategap.items())} }mm",
          flush=True)
    assert touch.get(1, 0) > 0 and 1 in plan.hits, \
        "the wings never touched the ceiling — they phased straight through it"
    hit = plan.hits[1]
    assert hit >= args.ceiling_min_approach, \
        (f"the wing hit the ceiling at t={hit:.2f} s, before the {args.ceiling_min_approach} s "
         f"of level approach the clip needs — there is no forward-tilted run-in to see")
    assert slab["pierce"] <= args.max_sink, \
        (f"a wing plate corner got {slab['pierce'] * 1000:.1f} mm past the slab underside "
         f"(limit {args.max_sink * 1000:.0f} mm) — it went THROUGH the ceiling")
    assert sink <= args.max_sink, \
        f"wing sank {sink * 1000:.1f} mm into the slab (limit {args.max_sink * 1000:.0f} mm)"
    held = touch.get(1, 0) * dt
    assert held <= args.clear_seconds, \
        (f"a wing was in contact with the ceiling for {held:.2f} s (limit "
         f"{args.clear_seconds} s) — that is scraping along it, not striking it")
    # The point of the whole clip: it got low enough. Measured off the plate
    # corners at full stroke amplitude AFTER the glide, so it is the real arc and
    # not the commanded altitude.
    assert clearance >= args.ceiling_clearance, \
        (f"after the glide the stroke still came within {clearance * 1000:.0f} mm of the "
         f"slab (want {args.ceiling_clearance * 1000:.0f} mm) — the bird did not drop far "
         f"enough to fly on underneath; raise --ceiling-drop")
    # Contact belongs to the STRIKE and nothing else. Once the graze drop has
    # pulled the wings clear they must skim without touching for the whole glide —
    # not merely be clear by the time the glide ends. Checked against the graze,
    # not the spool, because the whole glide is the shot and a wing scraping along
    # the underside for a second of it is the thing this is here to catch. It is a
    # real edge: a 75 mm graze drop still dragged to t=2.73 s, 85 mm clears.
    graze_done = plan.t_release(hit) + args.ceiling_graze_seconds
    assert last.get(1, -9e9) <= graze_done, \
        (f"the wings were still on the ceiling at t={last.get(1):.2f} s, after the graze "
         f"drop finished at t={graze_done:.2f} s — they are scraping along it, not "
         f"skimming under it; raise --ceiling-graze-drop")
    assert upright[1] >= up_free[1] - args.wing_up_margin, \
        (f"after being struck the wing plates only came back to {upright[1]:+.2f} upright, "
         f"against {up_free[1]:+.2f} on the approach (margin {args.wing_up_margin:.2f}) "
         f"— they are not recovering, i.e. a wing is over on its back")
    print("[ceiling] contact verified: the wings struck the slab, nothing passed through "
          "it, and the glide dropped the bird far enough to fly on underneath", flush=True)


def ceiling_self_check(args):
    """The ceiling beat sheet: only the HEIGHT story, since everything else in it
    is CollisionPlan and is already asserted by `self_check` above.

    (The other half is `ceiling_report`, which runs at the end of every ceiling
    render and asserts the wings actually hit the slab and then got clear of it.
    That one needs the sim, so it lives there.)
    """
    a = ceiling_args(args)
    plan, z0 = CeilingPlan(a), 1.0
    assert a.cruise_pitch > args.cruise_pitch and a.cruise_lean > args.cruise_lean, \
        "the ceiling approach must be tilted further forward than the pole cruise"
    assert a.glide_seconds >= 0.8, \
        "the brief asks for about a second of glide -- that is --ceiling-glide-seconds"

    # The approach: level, with clearance, and only THEN climbing into the slab.
    assert plan.z_target(0.0, z0) == z0, "opens in level cruise under the ceiling"
    assert plan.climbed(a.cruise_seconds - a.ceiling_climb_seconds) == 0.0, \
        "still level when the climb is due to start"
    assert abs(plan.climbed(a.cruise_seconds) - a.ceiling_climb) < 1e-9, \
        "fully climbed by the end of the approach"
    assert 0.0 < args.ceiling_bite < args.ceiling_climb, \
        ("the slab has to sit inside the CLIMBED stroke but outside the level one, or "
         "the bird either misses it entirely or is already in it on frame one")

    # The strike, then the glide DOWN. This is the whole ask: it recovers by
    # losing height, and the height never comes back.
    t1 = a.cruise_seconds
    plan.note_hit(t1, 1)
    rel, sp = plan.t_release(t1), plan.t_spool(t1)
    assert plan.dropped(rel) == 0.0, "the descent starts as the wings come off the slab"
    assert abs(plan.dropped(sp) - args.ceiling_drop) < 1e-9, \
        ("the whole drop is spent inside the glide -- that is what 'glide down until "
         "it is low enough' means")
    assert plan.dropped(sp + 10.0) == plan.dropped(sp), \
        ("the sag must be PERMANENT: the pole clip flies its sag back off, but here "
         "climbing back up would fly straight into the ceiling again")
    assert args.ceiling_drop > args.ceiling_bite + args.ceiling_clearance, \
        (f"a {args.ceiling_drop * 1000:.0f} mm drop cannot clear a {args.ceiling_bite * 1000:.0f} mm "
         f"bite by {args.ceiling_clearance * 1000:.0f} mm — the wings would still be in the slab")
    # Same statement, arrived at through z_target rather than through the knobs.
    assert (plan.z_target(sp, z0) + args.ceiling_bite
            < plan.z_target(t1, z0) + args.ceiling_climb - args.ceiling_clearance), \
        "the post-glide stroke must sit clear of where the slab was planted"
    assert plan.wing_sweep(sp) == 0.0 and plan.drive(sp) == args.glide_drive, \
        "wings out of the tuck and the stroke not yet driven at the end of the glide"
    assert plan.total > plan.t_powered(t1) + 0.5, \
        "the clip must outlast the recovery, so it closes on normal flight under the slab"
    print(f"render_configs ceiling self-check OK (strike due ~t={t1:.2f} s, "
          f"climb {args.ceiling_climb * 1000:.0f} mm, bite {args.ceiling_bite * 1000:.0f} mm, "
          f"glide {a.glide_seconds:.1f} s, drop {args.ceiling_drop * 1000:.0f} mm, "
          f"clip {plan.total:.2f} s)")


def self_check(args):
    """The beat sheet is the only real logic here — assert each beat is what it says.

    (The other half of the checking is `contact_report`, which runs at the end of
    every collision render and asserts the wings actually HIT the poles. That one
    needs the sim, so it lives there rather than here.)
    """
    plan, z0 = CollisionPlan(args), 1.0
    assert plan.assist(0.0) == 1.0, "cruise: full assist"
    assert plan.z_target(0.0, z0) == z0, "cruise: holds altitude"
    assert plan.drive(0.0) == 1.0 and plan.wing_sweep(0.0) == 0.0, \
        "cruise: full stroke, wings out"

    t1, t2 = strike_times(args)
    plan.note_hit(t1, 1)
    plan.note_hit(t1 + 0.3, 1)
    assert plan.hits == {1: t1}, "only the FIRST contact with a pole counts as its strike"

    assert 0.0 <= args.upset_floor < 1.0, "the upset must actually give something up"
    assert plan.assist(t1) == args.upset_floor, "impact: authority drops to the floor"
    assert plan.assist(t1 + args.upset) == args.upset_floor, "upset: still at the floor"
    assert plan.assist(t1 + args.upset + args.recover) == 1.0, "recovered: full authority back"

    # The strike: full drive INTO the pole, then knocked down to the trickle.
    rel, sp = plan.t_release(t1), plan.t_spool(t1)
    assert plan.drive(t1) == 1.0, "drive is still full AT the instant of contact"
    assert plan.drive(t1 + args.strike_seconds) == 1.0, \
        "the wing must keep beating INTO the pole for the whole strike window"
    assert plan.drive(rel) == args.glide_drive, "knocked down to the trickle on release"
    assert args.glide_drive < plan.drive(rel - args.stop_seconds / 2) < 1.0, \
        "drive fades, no cliff"
    assert args.glide_drive < 0.10, "the slow part of the spool must look like a fold, not a flap"

    # The glide, then the spool: slow at first, then quickly faster.
    assert plan.drive(sp) == args.glide_drive, "still gliding at the end of the glide window"
    assert plan.drive(sp + 0.05) > args.glide_drive, "the stroke builds from there"
    quarter = plan.drive(sp + 0.25 * args.spool_seconds)
    half = plan.drive(sp + 0.5 * args.spool_seconds)
    assert args.glide_drive < quarter < half < 1.0, "slow at first, then quickly faster"
    assert plan.drive(plan.t_powered(t1)) == 1.0, "full stroke by the end of the spool"

    # The fold: wings swept aft at the release, back out before the stroke returns.
    # Floor and ceiling are only 15 deg apart, and that is the honest width of the
    # window: below ~10 deg the tuck does not read on screen at all, above ~25 the
    # knocked wing swings into the other one. Moved down from 30 when the upper
    # guard came down to 25 — they are one constraint and have to travel together.
    assert args.fold_sweep > math.radians(10.0), "a fold has to be big enough to read as one"
    assert plan.wing_sweep(t1) == 0.0, "wings are out and flying right up to the hit"
    assert plan.wing_sweep(rel) == 0.0, \
        "still where the pole stopped them at the release -- the fold is the RESPONSE"
    assert abs(plan.wing_sweep(rel + args.fold_seconds) - args.fold_sweep) < 1e-9, \
        "swept fully back right after the release, promptly, as a direct answer to the hit"
    assert args.glide_seconds >= 2 * args.fold_seconds, \
        "the glide must be long enough to fold and unfold within it"
    assert plan.wing_sweep(sp) == 0.0, \
        "fully unswept BEFORE the spool starts -- the stroke is never driven while tucked"
    assert plan.drive(sp) == args.glide_drive, "and the drive is still at the trickle there"
    assert 0.0 < args.wing_up_margin < 0.5, \
        "the wings-never-inverted margin must be a real constraint, not a rubber stamp"
    # The tuck is now timed to the crossing rather than to the end of the glide,
    # so nothing structural stops it running past the spool any more -- say so.
    # See --fold-sweep for the measured table. 25 deg is the guard, not 40: the
    # bench said the plates were fine to 40, but in the clip the asymmetric knock
    # from the lone pole closes them to 28 mm by 28 deg. Trust the clip.
    assert args.fold_sweep <= math.radians(25.0), \
        (f"fold sweep {math.degrees(args.fold_sweep):.0f} deg swings the wing plates into "
         f"each other behind the body once a strike knocks one of them across")
    # The stroke must come back the moment the wings are out of the tuck: no dead
    # glide with the wings out and the motors still off.
    assert plan.wing_sweep(sp - 1e-9) < math.radians(1.0), \
        "the wings are still tucked where the spool starts"
    assert args.glide_seconds < 0.7, \
        (f"a {args.glide_seconds} s tuck is a glide, not a crossing — the bird should be "
         f"flapping again as soon as it is past the pole")

    # The sag: down under the knock, flown back off with the stroke.
    assert plan.z_target(rel, z0) < z0 - 0.5 * args.hit_drop, "the strike knocks it down"
    assert abs(plan.z_target(sp, z0) - (z0 - args.hit_drop)) < 1e-9, \
        "the whole sag is exactly --hit-drop, and it is still down there at the spool"
    assert abs(plan.z_target(plan.t_powered(t1), z0) - z0) < 1e-9, \
        "back at cruise altitude by the time the stroke is back"

    # The glide CARRIES it. This is the anti-wedge property, and the one thing
    # that separates a glide from parking in the obstacle you just hit.
    assert args.stall_frac > 0.0, "the impact checks the speed, it does not stop the bird"
    assert abs(plan.speed_target(0.0) - args.cruise_speed) < 1e-9, "cruises in"
    assert plan.speed_target(rel) == args.cruise_speed * args.stall_frac, \
        "the strike checks it hard at the release"
    assert args.fold_seconds < args.glide_pickup < args.glide_seconds, \
        "fold first, THEN up to glide speed, and both inside the glide window"
    flying_glide = rel + args.glide_pickup
    assert abs(plan.speed_target(flying_glide) - args.cruise_speed) < 1e-9, \
        "back at full speed once the fold is done -- the glide is what carries it clear"
    assert plan.drive(flying_glide) == args.glide_drive, \
        "and it is still UNPOWERED there: gliding at speed, not flapping again"
    assert abs(plan.wing_sweep(flying_glide) - args.fold_sweep) < 1e-9, \
        "with the wings still swept fully back"
    assert abs(plan.speed_target(plan.t_powered(t1)) - args.cruise_speed) < 1e-9, \
        "and cruising again by the end of the spool -- it is not done yet"

    # The SECOND strike: same shape, and it must land on a fully recovered bird.
    assert t2 > plan.t_powered(t1), "strike 2 must come after strike 1 is fully flown off"
    assert abs(t2 - plan.t_powered(t1) - args.gap_seconds) < 1e-9, \
        "--gap-seconds IS the recovered cruise between the two strikes"
    assert plan.drive(t2) == 1.0 and plan.assist(t2) == 1.0, \
        "at full stroke and full authority when it reaches the gap"
    plan.note_hit(t2, 2)
    assert plan.drive(t2) == 1.0, "and, as at strike 1, still full AT the moment of contact"
    assert plan.assist(t2) == args.upset_floor, "the second knock lands as hard as the first"
    assert plan.drive(plan.t_release(t2)) == args.glide_drive, "and knocked down again"
    rel2 = plan.t_release(t2)
    assert plan.speed_target(rel2) == args.cruise_speed * args.stall_frac \
        and abs(plan.speed_target(rel2 + args.glide_pickup) - args.cruise_speed) < 1e-9, \
        "the gap is the SAME event, just bigger: checked, folded, then glided clear"
    assert plan.drive(t1 + 0.5 * (t2 - t1)) > args.glide_drive, \
        "the two knocks must not merge -- there is a real recovery between them"
    assert plan.total > plan.t_powered(t2), "the clip must outlast the second recovery"

    # Pole geometry: the gap has to be narrower than the span but wider than the body.
    assert 0.0 < args.pole23_frac < 1.0, "the gap poles must sit INSIDE the wingspan"
    assert args.pole1_frac > args.pole23_frac, \
        "the lone pole is a wingtip strike, the gap is a two-wing strike -- so it sits further out"
    assert args.pole1_frac < 1.0, "the lone pole must be inside the wing's reach or it is missed"

    # Drivetrain coupling: the laggard sets the pace, and nothing is ever pushed.
    assert couple_brake([1.0] * 4, 30.0) == [0.0] * 4, "in phase: no braking at all"
    b = couple_brake([0.5, 1.0, 1.0, 1.0], 30.0)
    assert b[0] == 0.0 and all(x < 0.0 for x in b[1:]), \
        "a held-back crank sets the pace; the free ones get braked to it"
    assert max(couple_brake([0.0, 3.0], 10.0)) <= 0.0, \
        "coupling must never speed a drive up — that would fight the pole"

    # The poles are PLACED from this schedule, so the schedule has to be walkable.
    d1, d2 = path_x(args, strike_times(args))
    assert 0.0 < d1 < d2, "the second strike must be further down the corridor than the first"
    assert abs(d1 - args.cruise_speed * args.cruise_seconds) < 1e-3, \
        "nothing has knocked it yet at strike 1, so that leg is a plain cruise"
    print(f"render_configs collision self-check OK "
          f"(strikes at t={t1:.2f}/{t2:.2f} s, {d1:.2f}/{d2:.2f} m downrange)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="all",
                   choices=["flying", "multimodal", "collision", "ceiling", "all"])
    p.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "out")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--realtime-factor", type=float, default=0.2,
                   help="sim seconds per wall second in the clip (0.2 = 5x slow-mo)")
    p.add_argument("--force-scale", type=float, default=1.5, help="aero arrow length per N")
    p.add_argument("--lift", type=float, default=0.28, help="initial airframe height above compiled pose (m)")
    p.add_argument("--collision-lift", type=float, default=1.20,
                   help="same, for the collision clip — high enough that the poles run "
                        "past the bird above AND below, which is what sells them as poles")
    p.add_argument("--collision-arrow", type=float, default=0.45,
                   help="aero-arrow scale for the collision clip, as a fraction of "
                        "--force-scale (the multimodal scale draws arrows longer than "
                        "the bird and they hide the strike)")
    p.add_argument("--skin", choices=["flying", "all", "none"], default="flying",
                   help="apply the rufous-hummingbird skin to: just flying / all configs / none")
    p.add_argument("--bird-assets", default=flying.BIRD_ASSETS_DIR,
                   help="dir holding rufous_*.obj + rufous_diffuse.png")
    p.add_argument("--fade-frac", type=float, default=0.35,
                   help="fraction of the clip over which the skin fades to transparent")
    p.add_argument("--dry-run", action="store_true",
                   help="run the physics and print the contact ledger, skipping the "
                        "renderer and the encoder — for sweeping a knob and reading the "
                        "numbers back without waiting on a full render")
    p.add_argument("--still-fracs", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.1, 0.4, 0.7, 0.95], help="clip fractions to dump as PNG stills")
    # collision: beat sheet, in sim seconds (the clip length follows from these)
    p.add_argument("--self-check", action="store_true",
                   help="assert the collision and ceiling beat sheets, no render")
    # ceiling: the overhead-slab clip. Everything it does not name here it takes
    # from the pole knobs above (see ceiling_args) -- same fold, same spool, same
    # servos. What it overrides is HEIGHT and the forward tilt.
    p.add_argument("--ceiling-climb", type=float, default=0.20,
                   help="how far the bird climbs into the slab over the approach (m); "
                        "must exceed --ceiling-bite or it would already be in the ceiling "
                        "on frame one")
    p.add_argument("--ceiling-climb-seconds", type=float, default=0.5,
                   help="tail of the approach the climb happens over (s) — late, so the "
                        "clip opens on level flight with visible clearance to climb out of")
    # 85 mm out of a measured 311 mm of stroke, i.e. the ceiling takes the top
    # quarter of the beat. 45 mm was the first try and it read as a brush: the
    # truncation is what shows on screen, and a seventh of the arc is not enough
    # of one. Sink stays at 4.5 mm against the 10 mm limit either way.
    p.add_argument("--ceiling-bite", type=float, default=0.135,
                   help="how far INSIDE the top of the climbed stroke the slab underside "
                        "sits (m) — i.e. how much wing the ceiling takes. This is what "
                        "makes the strike a strike rather than a brush")
    # Just clears an 85 mm bite by the 60 mm clearance and stops. 220 mm was the
    # first pass and it dropped the bird well out from under the slab, which threw
    # away the shot: the ceiling has to stay a presence right over the wingtips
    # after the recovery, not become something it left behind.
    p.add_argument("--ceiling-drop", type=float, default=0.20,
                   help="how far the bird ducks after the strike (m). Permanent: this IS "
                        "the recovery, so it has to beat --ceiling-bite by "
                        "--ceiling-clearance — and by little else, or it stops skimming")
    p.add_argument("--ceiling-graze-drop", type=float, default=0.10,
                   help="the small immediate drop at the release (m), sized only to lift "
                        "the trickling stroke off the underside so the bird skims close "
                        "WITHOUT touching. Must stay well under --ceiling-drop or it reads "
                        "as falling rather than as pulling clear")
    p.add_argument("--ceiling-graze-seconds", type=float, default=0.15,
                   help="how fast the graze drop happens (s) — quick, it is the wings "
                        "pulling out of the ceiling, not a descent")
    p.add_argument("--ceiling-duck-seconds", type=float, default=0.3,
                   help="how long the duck takes (s), ending as the stroke returns. Much "
                        "shorter than --ceiling-glide-seconds on purpose: the bird skims "
                        "the ceiling with its wings folded for most of the glide and only "
                        "drops at the end, to make room for the stroke coming back")
    # This is the shot, so it gets the time. The wings are folded and the bird is
    # holding station right under the slab for all of it bar the closing duck.
    p.add_argument("--ceiling-glide-seconds", type=float, default=1.6,
                   help="wings-tucked glide after the strike (s), spent skimming along "
                        "under the slab. The pole clip's 0.5 is a crossing, not a glide")
    p.add_argument("--ceiling-strike-seconds", type=float, default=0.20,
                   help="how long the motors keep driving the wing into the slab after "
                        "contact (s). Shorter than the poles': a slab stops the upstroke "
                        "dead, where a pole is a line the wing has to be driven ACROSS")
    # The forward tilt the brief asks for, and it is a POSE, not a mechanism: both
    # numbers are fixed for the whole clip. Tilting the stroke plane dynamically is
    # what flipped the wings over on every run of the old fold (see WingSweep).
    p.add_argument("--ceiling-pitch", type=float, default=math.radians(30.0),
                   help="forward stroke-plane tilt on the ceiling clip (rad), fixed — "
                        "steeper than the pole cruise so the bird is visibly driving "
                        "forward and up INTO the slab rather than drifting up to it")
    p.add_argument("--ceiling-lean", type=float, default=math.radians(17.0),
                   help="commanded nose-down body pitch on the ceiling clip (rad), fixed")
    p.add_argument("--ceiling-half-y", type=float, default=0.60,
                   help="half-width of the overhead slab (m). This is the framing knob: "
                        "narrow enough that BOTH long edges are in shot and converge away "
                        "down the corridor, which is what reads as a ceiling; a slab wide "
                        "enough to run off the frame renders as a horizon line")
    p.add_argument("--ceiling-thickness", type=float, default=0.055,
                   help="slab thickness (m) — thick enough that its edge is a visible band "
                        "seen from below, not a hairline")
    p.add_argument("--ceiling-ahead", type=float, default=4.0,
                   help="downrange offset of the slab centre from the bird's start (m); "
                        "with the default 6 m half-length it is overhead from frame one "
                        "and outlasts the flight")
    p.add_argument("--ceiling-clearance", type=float, default=0.04,
                   help="how far below the slab the whole stroke must sit once the glide "
                        "is over (m) — the measured form of 'low enough to avoid it'")
    p.add_argument("--ceiling-fold-sweep", type=float, default=0.0,
                   help="how far the wings sweep aft on the ceiling clip (rad). 0: they "
                        "stay out at the side and glide there. The pole clip's tuck is "
                        "for getting a wing past an obstacle, which is not the move here")
    p.add_argument("--ceiling-azimuth", type=float, default=CEILING_AZIMUTH,
                   help="camera azimuth on the approach (deg); 90 is square to the "
                        "flight path, which is where the closing gap reads at full size")
    # 180, not 0. Azimuth is the direction the camera LOOKS, not where it sits, so
    # 0 puts the lens BEHIND the bird looking the way it is going — a chase shot,
    # the opposite of what is wanted. 180 looks back down the corridor from ahead,
    # i.e. head on. (The pole camera's own comment says the same thing: its old
    # azimuth 150 is described as "ahead-looking-back".) Through the move the slab
    # swings from a band across the top to a roof running away from the camera,
    # which is a second read on the same geometry.
    p.add_argument("--ceiling-front-azimuth", type=float, default=180.0,
                   help="camera azimuth it swings to on the strike (deg); 180 is head on")
    p.add_argument("--ceiling-pan-seconds", type=float, default=0.9,
                   help="how long the swing to the front takes (s), starting at the "
                        "strike — over the fold, so the move and the tuck land together")
    p.add_argument("--ceiling-min-approach", type=float, default=0.45,
                   help="earliest the strike may land (s); guards the failure the first "
                        "ceiling clip had, where the altitude mode threw a wingtip into "
                        "the slab a moment after the clip opened")
    # The stroke keeps growing for ~3 s after the drive ramp is over; sampling
    # through the build-up reads the reach ~35% low, which would plant the poles
    # inside the wing arc and have them hit on frame one.
    p.add_argument("--warmup", type=float, default=3.0,
                   help="unrecorded flapping before the clip, to settle the stroke (s)")
    p.add_argument("--peak-window", type=float, default=1.5,
                   help="tail of the warmup the wing's lateral reach is measured over (s); "
                        "must span the slow altitude mode, not just a few flap cycles")
    p.add_argument("--cruise-seconds", type=float, default=1.2,
                   help="level approach to the first pole (s) — this is also what places it")
    # Longer than the ceiling clip's 0.08: a slab stops the upstroke dead, but a
    # pole is a vertical line the wing sweeps ACROSS, so the beat has to be driven
    # into it over more than one stroke for the strike to read as a strike.
    p.add_argument("--strike-seconds", type=float, default=0.18,
                   help="how long the motors keep driving the wing into the pole after "
                        "contact (s) — keep short or the wings look welded to it")
    p.add_argument("--upset", type=float, default=0.40, help="knocked-about window (s)")
    # 0 is NOT "more honest": this flapper is open-loop unstable, so cutting
    # attitude authority outright doesn't show the pole knocking the bird, it
    # shows the airframe diverging (35 deg roll, wings 64 deg out of step, beat
    # amplitude wandering 35-78 deg — the broken flapping). Any authority at all
    # stops the divergence; the transition sits between 0.00 and 0.02.
    p.add_argument("--upset-floor", type=float, default=0.15,
                   help="attitude/station authority retained through the upset")
    p.add_argument("--recover", type=float, default=1.5, help="assist ramp-back (s)")
    # 0.06 is the window. At 0.00 the cranks stop dead and the stroke never
    # recovers past 20 deg; at 0.15 the wings swing through 120 deg and it reads
    # as thrashing. At 0.06 the wings look folded and still, and it comes back.
    p.add_argument("--glide-drive", type=float, default=0.06,
                   help="drive floor through the slowest part of the spool (0-1)")
    # No wait-until-clear gate: holding the motors off until the wings were
    # declared clear left it hanging with the wings off for over a second. The
    # drive starts turning the moment the wing lets go and builds instead, which
    # is both quicker to read and what a bird actually does.
    p.add_argument("--spool-seconds", type=float, default=0.8,
                   help="release to full stroke (s): the wings flap slowly and build")
    p.add_argument("--glide-feather", type=float, default=0.25,
                   help="wing free-pitch angle during the glide, as a fraction of cruise: "
                        "lower = flatter wing = lower angle of attack")
    p.add_argument("--flap-seconds", type=float, default=0.9,
                   help="powered forward flight to close the clip on (s)")
    p.add_argument("--stop-seconds", type=float, default=0.10,
                   help="motor fade-out from the moment of contact (s) — short, so the "
                        "wings stop dead against the pole rather than sweeping on")
    # The fold, written straight onto the wing plates (see WingSweep). This used
    # to be a stroke-plane tilt, i.e. swing the crank mount and hope the tendons
    # dragged the wings round; it overcorrected on every run and flipped the wings
    # over. A stated angle cannot overshoot.
    #
    # 20 deg, measured in the clip itself (--dry-run sweep), not on the bench.
    # Sweeping both plates aft about the airframe's up axis converges them behind a
    # body only ~124 mm across, so past a point they meet and pass THROUGH each
    # other. Closest L-R approach while tucked, per strike:
    #     12 deg -> {1: 83, 2: 84} mm      28 deg -> {1: 28, 2: 54} mm
    #     20 deg -> {1: 83, 2: 83} mm      62 deg -> plates clean through each other
    # The knee is between 20 and 28, and it bites on the FIRST strike first: the
    # lone pole hits one wing only, so that wing is driven across the centreline
    # while the other is not. A steady-flapping bench measurement misses this
    # entirely (it said 63 mm at 35 deg) — the asymmetric knock is the whole effect.
    #
    # Not zero, though: the fold earns its place. Wing-on-pole contact at the gap
    # is 0.647 s unfolded against 0.335 s at 20 deg.
    p.add_argument("--fold-sweep", type=float, default=math.radians(20.0),
                   help="how far the wings sweep aft when they take a hit (rad)")
    p.add_argument("--fold-seconds", type=float, default=0.16,
                   help="how fast the wings sweep back, and back out again (s)")
    # Measured, not guessed: free flapping with no poles and no fold already takes
    # the plate normal to -0.62 (it swings +0.99 -> -0.62 every beat as the wing
    # feathers), so this is how much FURTHER over a strike may roll it.
    p.add_argument("--wing-up-margin", type=float, default=0.15,
                   help="how far past the approach stroke's own least-upright a wing "
                        "plate may roll once struck (normal's component along world up)")
    # Shorter than the ceiling clip's 5 s: that one had a single strike to build a
    # whole glide out of, this one has two strikes and a recovery to fit in.
    # This is the whole tuck: the wings sweep out over --fold-seconds, stay away,
    # and are back exactly as the motors start. 0.9 s of it was a glide with the
    # wings folded well after the bird was past the pole; 0.5 covers the crossing
    # (longest measured wing-on-pole contact is 0.56 s from first touch) and puts
    # the stroke back the moment they are clear.
    p.add_argument("--glide-seconds", type=float, default=0.5,
                   help="how long the wings stay tucked after each strike before the "
                        "motors start building again (s)")
    p.add_argument("--glide-hold", type=float, default=0.85,
                   help="horizontal (speed/station) authority kept from the release on, so "
                        "the glide carries forward instead of coasting to a stop")
    p.add_argument("--gap-seconds", type=float, default=1.3,
                   help="recovered cruise between finishing the first strike and reaching "
                        "the two-pole gap (s) — this is what places the gap")
    p.add_argument("--hit-drop", type=float, default=0.10,
                   help="how far it sags while the stroke is gone (m); flown back off as "
                        "the drive spools up")
    # The momentary CHECK at the impact, not a hold: see speed_target. It is only
    # in force for `--glide-pickup` around the release, so it can be deep enough
    # to read as a real hit without parking the bird in the pole.
    p.add_argument("--stall-frac", type=float, default=0.30,
                   help="fraction of cruise speed at the moment the strike lands (0-1)")
    # Has to sit inside the tuck, and the tuck is now only --glide-seconds long,
    # so 0.35 no longer fits: speed must be back BEFORE the wings start coming out,
    # or the "carrying speed while folded" beat has nowhere to happen.
    p.add_argument("--glide-pickup", type=float, default=0.25,
                   help="release to full glide speed (s). This is the FOLD timescale, not "
                        "the spool: the bird must be carrying speed while its wings are "
                        "still folded, or it sits in the obstacle it just hit")
    # collision: pole geometry
    p.add_argument("--pole-radius", type=float, default=0.025, help="pole radius (m)")
    p.add_argument("--pole-top", type=float, default=2.4,
                   help="pole height (m) — floor to well above the bird, so a wing at any "
                        "point of the stroke meets pole, never air")
    # Both are fractions of the MEASURED lateral reach of the wing plate, so they
    # track the design instead of hardcoding a wingspan.
    #
    # 0.75: the lone pole takes the outer quarter of the wing. Deeper than the
    # reach at the TOP of the stroke (~0.61 of the maximum) and the wing can never
    # get round it — the bird would grind against the pole for the rest of the
    # clip instead of clipping it and flying on.
    p.add_argument("--pole1-frac", type=float, default=0.70,
                   help="lone pole's lateral offset, as a fraction of the wing's measured "
                        "reach; must stay above the top-of-stroke reach or it cannot pass")
    # The gap is 2 x 0.52 of the wing's reach, i.e. a good deal narrower than the
    # span, so each wing loses its outer part into a pole. Note the fold does NOT
    # help it through: sweeping the plates aft does not shorten their lateral reach
    # at all (measured 272 mm unswept, 292 mm at 35 deg — rotating a plate about
    # its own root just swings a different corner outboard). The bird crosses by
    # flying through, not by tucking inside.
    p.add_argument("--pole23-frac", type=float, default=0.52,
                   help="each gap pole's lateral offset, as a fraction of the wing's "
                        "measured reach; < 1 is what makes the gap narrower than the span")
    # ponytail: one hand-fitted constant standing in for "how far ahead of the
    # COM does the wing plate actually reach". 0.15 puts the measured strike at
    # t=1.14 s against a wanted 1.20; 0 put it at 0.94. It only shifts WHEN the
    # strikes land — every downstream beat keys off the real contact — so a
    # measured version (plate x-extent during the warmup) is a nicety, not a fix.
    p.add_argument("--pole-lead", type=float, default=0.15,
                   help="extra downrange offset on every pole (m): the wing's chord reaches "
                        "ahead of the COM, so it arrives slightly before the schedule says")
    p.add_argument("--max-sink", type=float, default=0.010,
                   help="deepest the wing plate may penetrate a pole before the render is "
                        "called a failure (m)")
    p.add_argument("--clear-seconds", type=float, default=0.65,
                   help="longest a wing may spend actually touching one obstacle (s) "
                        "before the render is called wedged")
    p.add_argument("--body-clearance", type=float, default=0.03,
                   help="how far outside a pole's surface the airframe COM must stay (m); "
                        "the poles are masked to ignore the body, so only geometry keeps "
                        "it from rendering as flying through one")
    p.add_argument("--cruise-speed", type=float, default=0.6, help="forward speed (m/s)")
    p.add_argument("--cruise-pitch", type=float, default=math.radians(18.0),
                   help="forward stroke-plane tilt (rad)")
    p.add_argument("--cruise-lean", type=float, default=math.radians(10.0),
                   help="commanded nose-down body pitch (rad)")
    p.add_argument("--speed-gain", type=float, default=20.0, help="speed servo gain (1/s)")
    p.add_argument("--pos-gain", type=float, default=80.0, help="station-keeping gain (1/s^2)")
    p.add_argument("--ref-leash", type=float, default=0.05,
                   help="how far the station reference may run ahead of the airframe (m) "
                        "while an obstacle is holding it up — anti-windup, so the bird "
                        "flies out of a strike instead of being catapulted out of it")
    p.add_argument("--level-kp", type=float, default=0.9, help="attitude spring (N m/rad)")
    p.add_argument("--level-kd", type=float, default=0.05, help="attitude damping (N m s/rad)")
    # wingtip trail (collision clip)
    p.add_argument("--trail-pts", type=int, default=115, help="trail samples per flap cycle")
    p.add_argument("--trail-cycles", type=float, default=1.0, help="cycles of trail kept")
    p.add_argument("--trail-width", type=float, default=0.0011)
    p.add_argument("--drive-couple", type=float, default=30.0,
                   help="drivetrain coupling gain (1/s): brakes the free cranks back to "
                        "whichever one the pole is holding up, so the wings stay in "
                        "phase through the strike. 0 = four independent drives")
    args = p.parse_args()
    if args.self_check:
        self_check(args)
        ceiling_self_check(args)
        return
    configs = (["flying", "multimodal", "collision", "ceiling"]
               if args.config == "all" else [args.config])
    for cfg in configs:
        render_config(cfg, args)


if __name__ == "__main__":
    main()
