"""The GUI's live impulse-response sim: constant-speed cranks plus an open-loop torque pulse.

``ImpulseSim`` steps the design in real time for the Impulse response tab, records
crank phases, servo torques and wing kinematics, and can hold the crank phase with
the eagle's ``CrankDrive`` trim. See the class docstring for the pulse model.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import deque
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from gui.sims import prepare_scene
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control, load_control_for_morphology, ramp_factor
from runtime.crank import CrankDrive
from runtime.disk_orientation import StrokePlaneSignal
from runtime.eagle_drive import wing_plane_roll
from runtime.sim import PlateWingSim

REF_MOTOR = "L_motor_t1"


def wrap_deg(angle, center: float = 180.0):
    """Wrap degrees into [center-180, center+180): center 180 -> [0, 360), 0 -> [-180, 180)."""
    return (angle - center + 180.0) % 360.0 + center - 180.0


def cut_wraps(phase: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where a [0, 360) phase trace wraps, so a line plot doesn't draw the flyback."""
    cut = np.flatnonzero(np.diff(phase) < -180.0) + 1
    return np.insert(phase, cut, np.nan), np.insert(values, cut, np.nan)


class ImpulseSim:
    """Live sim: crank disks at one constant speed, plus a torque pulse through chosen servos.

    Phase is each crank's travel from the synced rest angle in its OWN rotation
    sense (the actuator gear encodes the counter-rotation and the L/R mirror), so
    every motor reads 0 at sync and the .tg crank_phase_lag shows as the t4 motors
    sitting at -lag. Relative phase is against ``REF_MOTOR``.

    The motors are velocity servos, and a velocity loop cancels a torque added at
    its shaft within one step, so an additive disturbance never reaches the wing.
    The pulse therefore opens the loop: for its duration the pulsed servo outputs
    the torque-vs-phase profile it produced over its last cycle (the nominal
    drive) plus the pulse, and the wing feels the difference. Afterwards the
    velocity loop closes again, blended back in over the pulse's own duration so
    the re-lock is a finite-bandwidth recovery rather than a one-step slam.

    Nothing in the plain drive regulates phase: the t1 and t4 cranks carry
    different mean loads, so their servo slip differs and the .tg's crank lag
    walks off (~0.1 deg/cycle on the flagship), and any phase a pulse shifts
    stays shifted. ``hold_phase`` switches in the eagle's ``CrankDrive`` trim,
    which pins q_t1 + q_t4 (and the L/R pair) at the synced values, so the lag
    holds and a pulse's phase step decays back over ~1/GAIN seconds.
    """

    SAMPLE_EVERY = 4  # physics steps per recorded sample (~0.5 ms at the 0.13 ms step)
    MAX_CYCLES = 200  # history kept for the plots
    #: Per-wing kinematics in degrees, the same definitions the eagle's stroke
    #: calibration uses, mirrored so a symmetric pose reads the same on both wings:
    #:   sweep      - span direction in the horizontal plane: 0 straight out, + forward
    #:   elevation  - span direction above the horizontal, + up
    #:   roll       - plate rotation about the span (wing_plane_roll): 0 level, + nose-up
    #:   aoa        - angle between the plate and its own relative wind on the aero strip
    #:                holding 70% span (62.5% on a 4-blade wing; unsigned)
    #: The tip strip's fore-aft and height positions ride along (not plotted) so the
    #: stroke-plane angle can be read off the two stroke reversals, Ellington-style.
    KINEMATICS = ("sweep", "elevation", "roll", "aoa")
    FORWARD = np.array([-1.0, 0.0, 0.0])  # .tg convention: +x is the trailing edge, so -x is cranial

    def __init__(self, tg_path: Path, scene_path: Path, control_path: Path | None,
                 settle_periods: float, speed: float | None = None) -> None:
        prepare_scene(tg_path, scene_path)
        self.morph = load_morphology(tg_path)
        self.sim = PlateWingSim(mujoco.MjModel.from_xml_path(str(scene_path)), self.morph)
        self.model, self.data = model, data = self.sim.model, self.sim.data
        control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
        self.speed = control.a0 if speed is None else float(speed)  # rad/s commanded on every disk
        self.period = math.tau / abs(self.speed) if self.speed else control.period
        self.settle_periods = float(settle_periods)
        sp_file = tg_path.parent / "strokeplane_optimized.json"  # learned stroke plane, as the viewer applies it
        self.sp = StrokePlaneSignal.from_json(model, json.loads(sp_file.read_text())) if sp_file.exists() else None

        self.aid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in self.sim.crank_ids}
        self.motors = tuple(self.aid)
        if not self.motors:
            raise ValueError("no crank motors in this design")
        self.ctrl_sign = math.copysign(1.0, self.morph.motors.motor_direction * self.speed) if self.speed else 1.0
        self.qadr: dict[str, int] = {}
        self.dof: dict[str, int] = {}
        self.direction: dict[str, float] = {}
        self.kv: dict[str, float] = {}
        self.gear: dict[str, float] = {}
        for name in self.motors:
            prefix, station = name.split("_motor_")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{station}_disk_hinge")
            if jid < 0:
                raise ValueError(f"missing disk hinge for {name}")
            aid = self.aid[name]
            self.qadr[name] = int(model.jnt_qposadr[jid])
            self.dof[name] = int(model.jnt_dofadr[jid])
            self.kv[name] = float(model.actuator_gainprm[aid][0]) or 1.0
            self.gear[name] = float(model.actuator_gear[aid][0]) or 1.0
            self.direction[name] = math.copysign(1.0, self.gear[name] * self.ctrl_sign)
        self.ref = REF_MOTOR if REF_MOTOR in self.aid else self.motors[0]
        self.theta_ref = float(data.qpos[self.qadr[self.ref]])
        self.hold_phase = False  # off: the plain drive every hummer runner uses
        self.lock = CrankDrive(model, self.morph, control)
        self.lock.capture(data)  # the synced rest phases are the ones to hold, however late it is switched on

        self.sides: dict[str, tuple[int, int, int]] = {}  # side -> (wing body, AoA blade site, tip site)
        self.outboard: dict[str, np.ndarray] = {}  # world direction the span points at rest
        for wing in self.sim.aero.wings:
            side, blades = wing.side, wing.blade_site_ids
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wing")
            if bid >= 0:
                self.sides[side] = (bid, blades[int(0.7 * len(blades))], blades[-1])
                span = np.asarray(data.xmat[bid]).reshape(3, 3)[:, 2]
                self.outboard[side] = np.array([0.0, math.copysign(1.0, span[1]), 0.0])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self.pulse: dict[str, float] = {}  # motor -> pulse torque (N m) while it is on
        self.nominal: dict[str, Callable[[float], float]] = {}  # motor -> last cycle's torque vs phase (N m)
        self.pulse_end = -math.inf
        self.recovery = 0.0  # seconds after pulse_end over which the velocity loop is blended back in
        self.pulses: list[tuple[float, float, float, tuple[str, ...]]] = []  # (t0, t1, mN m, motors)
        self.applied = {name: 0.0 for name in self.motors}  # actuator-frame pulse torque this step (N m)
        self.reference: dict[str, np.ndarray | float] | None = None  # undisturbed stroke, see capture_reference

        keys = ["t"]
        for name in self.motors:
            keys += [f"phase:{name}", f"torque:{name}", f"applied:{name}"]
        for side in self.sides:
            keys += [f"{key}:{side}" for key in (*self.KINEMATICS, "tipf", "tipz")]
        maxlen = int(self.MAX_CYCLES * self.period / (self.SAMPLE_EVERY * model.opt.timestep)) + 1
        self.hist = {key: deque(maxlen=maxlen) for key in keys}

        self.drive_start = data.time
        self._steps = 0
        self._carry = 0.0
        for _ in range(math.ceil(self.settle_periods * self.period / model.opt.timestep)):
            self.step()
        self.t0 = data.time
        self.record()

    # -- physics ------------------------------------------------------------
    def step(self) -> None:
        model, data = self.model, self.data
        ramp = ramp_factor(data.time, self.drive_start, self.settle_periods * self.period)
        ctrl = ramp * self.morph.motors.motor_direction * self.speed
        t = data.time
        if self.sp is not None:
            self.sp.apply(data)
        self.sim.apply(ctrl)
        if self.hold_phase:
            self.lock.trim(data)
        for name in self.motors:
            aid = self.aid[name]
            tau = 0.0
            if name in self.pulse and t < self.pulse_end + self.recovery:
                # The servo's force is kv*(ctrl - gear*qvel): pick the ctrl that makes it
                # exactly what we want. Pulse: last cycle's profile plus the pulse (loop
                # open). Recovery: blend from that profile back to the velocity loop.
                gear_qvel = self.gear[name] * float(data.qvel[self.dof[name]])
                nominal = self.nominal[name](self.phase_deg(name))
                if t < self.pulse_end:
                    tau = self.pulse[name] * self.ctrl_sign  # actuator frame: + pushes forward
                    want = nominal + tau
                else:
                    s = (t - self.pulse_end) / self.recovery
                    want = (1.0 - s) * nominal + s * self.kv[name] * (ctrl - gear_qvel)
                data.ctrl[aid] = want / self.kv[name] + gear_qvel  # open loop: no trim on this shaft
            self.applied[name] = tau
        mujoco.mj_step(model, data)
        self._steps += 1

    def advance(self, seconds: float) -> None:
        """Step ``seconds`` of sim time, recording a sample every SAMPLE_EVERY steps."""
        dt = self.model.opt.timestep
        n = int((seconds + self._carry) / dt)
        self._carry = seconds + self._carry - n * dt
        for _ in range(n):
            self.step()
            if self._steps % self.SAMPLE_EVERY == 0:
                self.record()

    def inject(self, torque_mnm: float, duration_ms: float, motors) -> tuple[str, ...]:
        """Open the loop on ``motors`` for ``duration_ms`` and add ``torque_mnm`` (+ aids rotation)."""
        motors = tuple(name for name in motors if name in self.aid)
        self.nominal = {name: self.nominal_profile(name) for name in motors}
        self.pulse = {name: 1e-3 * torque_mnm for name in motors}
        self.pulse_end = self.data.time + 1e-3 * duration_ms
        self.recovery = 1e-3 * duration_ms
        self.pulses.append((self.time, self.pulse_end - self.t0, torque_mnm, motors))
        self.capture_reference()
        return motors

    def nominal_profile(self, name: str) -> Callable[[float], float]:
        """This servo's torque vs phase over its last cycle, as a periodic interpolant (N m)."""
        w = self.window(1.0)
        phase, torque = w[f"phase:{name}"], 1e-3 * w[f"torque:{name}"]
        if len(phase) < 8:  # barely started: hold what it is doing now
            hold = float(self.data.actuator_force[self.aid[name]])
            return lambda _phase: hold
        order = np.argsort(phase)
        phase, torque = phase[order], torque[order]
        return lambda p: float(np.interp(p, phase, torque, period=360.0))

    def capture_reference(self) -> None:
        """Keep the last full stroke before the pulse; the plots tile it ahead as the undisturbed motion."""
        t = np.array(self.hist["t"])
        phase = np.array(self.hist[f"phase:{self.ref}"])
        wraps = np.flatnonzero(np.diff(phase) < -180.0) + 1
        if len(wraps) < 2:
            self.reference = None
            return
        a, b = int(wraps[-2]), int(wraps[-1])
        self.reference = {"t0": float(t[a]), "period": float(t[b] - t[a]), "t": t[a:b] - t[a]}
        for side in self.sides:
            for key in self.KINEMATICS:
                self.reference[f"{key}:{side}"] = np.array(self.hist[f"{key}:{side}"])[a:b]

    # -- readings -----------------------------------------------------------
    @property
    def time(self) -> float:
        return self.data.time - self.t0

    def kinematics(self, side: str) -> dict[str, float]:
        model, data = self.model, self.data
        bid, sid, tid = self.sides[side]
        rot = np.asarray(data.xmat[bid]).reshape(3, 3)
        normal, span = rot[:, 1], rot[:, 2]
        out = {
            "sweep": math.degrees(math.atan2(float(span @ self.FORWARD), float(span @ self.outboard[side]))),
            "elevation": math.degrees(math.asin(float(np.clip(span[2], -1.0, 1.0)))),
            "roll": math.degrees(wing_plane_roll(rot, side)),
        }
        # relative wind at the blade, in the section plane perpendicular to the span
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, sid)
        vel = self._jacp @ data.qvel
        vel -= float(vel @ span) * span
        speed = float(np.linalg.norm(vel))
        # Blank around stroke reversal: with the plate nearly stopped its relative wind
        # has no direction and the reading pins at 90 deg (mid-stroke is ~2 m/s).
        out["aoa"] = (
            math.degrees(math.asin(min(1.0, abs(float(vel @ normal)) / speed))) if speed > 0.3 else math.nan
        )
        tip = data.site_xpos[tid]
        out["tipf"] = 1000.0 * float(tip @ self.FORWARD)
        out["tipz"] = 1000.0 * float(tip[2])
        return out

    def stroke_cycle(self, side: str) -> dict[str, float]:
        """Last period's stroke amplitude (deg, peak-to-peak sweep) and stroke-plane angle (deg).

        The stroke plane is the line through the wingtip at the two stroke reversals,
        measured from horizontal: + when the forward reversal is the higher one.
        Both are NaN until a full period is in the history.
        """
        w = self.window(1.0)
        if len(w["t"]) < 0.9 * self.period / (self.SAMPLE_EVERY * self.model.opt.timestep):
            return {"amplitude": math.nan, "plane": math.nan}
        sweep = w[f"sweep:{side}"]
        fwd, aft = int(np.argmax(sweep)), int(np.argmin(sweep))
        rise = w[f"tipz:{side}"][fwd] - w[f"tipz:{side}"][aft]
        run = w[f"tipf:{side}"][fwd] - w[f"tipf:{side}"][aft]
        return {"amplitude": float(sweep[fwd] - sweep[aft]), "plane": math.degrees(math.atan2(rise, run))}

    def phase_deg(self, name: str) -> float:
        travel = self.direction[name] * (float(self.data.qpos[self.qadr[name]]) - self.theta_ref)
        return wrap_deg(math.degrees(travel))

    def phases(self) -> dict[str, float]:
        return {name: self.phase_deg(name) for name in self.motors}

    def relative_deg(self, name: str) -> float:
        return wrap_deg(self.phase_deg(name) - self.phase_deg(self.ref), 0.0)

    def record(self) -> None:
        h, data = self.hist, self.data
        h["t"].append(self.time)
        for name in self.motors:
            h[f"phase:{name}"].append(self.phase_deg(name))
            h[f"torque:{name}"].append(1000.0 * float(data.actuator_force[self.aid[name]]))
            h[f"applied:{name}"].append(1000.0 * self.applied[name])
        for side in self.sides:
            for key, value in self.kinematics(side).items():
                h[f"{key}:{side}"].append(value)

    def window(self, cycles: float) -> dict[str, np.ndarray]:
        """The last ``cycles`` commanded periods of history, oldest first."""
        n = int(cycles * self.period / (self.SAMPLE_EVERY * self.model.opt.timestep)) + 1
        n = max(1, min(n, len(self.hist["t"])))
        return {
            key: np.fromiter(itertools.islice(reversed(dq), n), dtype=float, count=n)[::-1]
            for key, dq in self.hist.items()
        }

    def reference_tiled(self, key: str, t_from: float, t_to: float) -> tuple[np.ndarray, np.ndarray]:
        """The captured stroke repeated over [t_from, t_to] (NaN-separated), for overlaying."""
        ref = self.reference
        if ref is None:
            return np.array([]), np.array([])
        t0, period = ref["t0"], ref["period"]
        k_first = max(1, math.floor((t_from - t0) / period))  # the captured cycle itself is real data
        k_last = math.ceil((t_to - t0) / period)
        xs, ys = [], []
        for k in range(k_first, k_last + 1):
            xs += [t0 + k * period + ref["t"], [np.nan]]
            ys += [ref[key], [np.nan]]
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.array([]), np.array([]))
