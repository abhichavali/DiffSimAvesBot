"""Watch the free-standing bird fly under a flight control .tg: cradle launch, release, real-time flight.

    mjpython src/view_free_hover.py [--tg designs/hummer/free-hover/shoulder.tg]
        [--control designs/hummer/free-hover/control.tg] [--speed 0.25] [--open-loop]
    mjpython src/view_free_hover.py --control designs/hummer/free-hover/command.tg [--sensors] [--stdin]

The control's open loop (per-crank Fourier drive, phase-locked cranks) flies the bird; its closed loop (the
per-wingbeat MLP, if the file names one) corrects the harmonics once per wingbeat. Nothing else acts on the
bird after release. In the viewer window: P = push the bird (random 0.3 m/s + 1.5 rad/s kick),
O = toggle closed / open loop. The GUI's Viewer tab launches this for flight controls.

A command-following control (``command_loop``, hover.command.LiveCommand) is driven like a game character,
seen side-on (screen right = the bird's forward): HOLD Right / Left = fly forward / back, HOLD Up / Down = climb /
descend (two keys combine, e.g. forward + up), release = hover in place; Home = hover, + / - = faster / slower
(clipped to the policy's measured envelope). Held keys need key-release events, which MuJoCo's viewer does not
report: they come from ``pynput`` (a global keyboard listener; on macOS allow Input Monitoring for the terminal
that runs mjpython). Without it (or without that permission) each tap toggles a direction on / off instead.
``--stdin`` also reads one command per line (``FWD``, ``UP 0.2``, ``HOVER``, ``PUSH``, ``LOOP``): the GUI's
Viewer tab drives the bird through it. ``--sensors`` flies the servo on its observer (``command_observer``:
mocap position with 5 ms latency + the IMU attitude filter and gyro, datasheet noise x ``--noise-scale``)
instead of the true state. Command flight is free air, as the policy was evaluated: contacts off and the
floor (0.25 m below the release) moved out of the way.
"""

from __future__ import annotations

import argparse
import copy
import math
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import mujoco.viewer
import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover.policy import load_flight_policy

# GLFW key codes (mujoco.viewer passes key PRESSES only to key_callback)
KEY_ARROWS = {262: "right", 263: "left", 265: "up", 264: "down"}
KEY_HOME, KEY_PLUS, KEY_MINUS, KEY_KP_ADD, KEY_KP_SUB = 268, 61, 45, 334, 333


class Keys:
    """Arrow-key driving: held keys (pynput press / release, if it delivers events) or, as a fallback, toggles
    from the viewer's key presses. ``axes()`` -> (forward -1/0/+1, up -1/0/+1); a latched command (GUI / stdin /
    toggles) applies while no arrow is held."""

    def __init__(self):
        self.held, self.lock = set(), threading.Lock()
        self.hold_mode = False               # True once pynput has delivered an arrow event
        self.latch = [0, 0]

    def start_listener(self) -> str:
        try:
            from pynput import keyboard
        except Exception as e:                       # noqa: BLE001 (not installed / no display backend)
            return f"hold-to-move unavailable ({type(e).__name__}: {e}); arrows toggle a direction instead"
        names = {keyboard.Key.right: "right", keyboard.Key.left: "left", keyboard.Key.up: "up", keyboard.Key.down: "down"}

        def press(k):
            if k in names:
                with self.lock:
                    if not self.hold_mode:
                        self.hold_mode, self.latch = True, [0, 0]
                    self.held.add(names[k])

        def release(k):
            if k in names:
                with self.lock:
                    self.held.discard(names[k])
        try:
            keyboard.Listener(on_press=press, on_release=release, daemon=True).start()
        except Exception as e:                       # noqa: BLE001
            return f"hold-to-move unavailable ({type(e).__name__}: {e}); arrows toggle a direction instead"
        return "hold the arrows to move (release = hover); if nothing moves, arrows toggle instead"

    def viewer_press(self, name: str) -> None:
        """An arrow press from the viewer: ignored when held keys work, else toggles that direction."""
        with self.lock:
            if self.hold_mode:
                return
            i, sgn = {"right": (0, 1), "left": (0, -1), "up": (1, 1), "down": (1, -1)}[name]
            self.latch[i] = 0 if self.latch[i] == sgn else sgn

    def set_latch(self, f: int, z: int) -> None:
        with self.lock:
            self.latch = [f, z]

    def axes(self) -> tuple[int, int]:
        with self.lock:
            h = self.held
            f = ("right" in h) - ("left" in h)
            z = ("up" in h) - ("down" in h)
            return (f, z) if h else tuple(self.latch)


def read_stdin(q: queue.Queue) -> None:
    for line in sys.stdin:
        if line.strip():
            q.put(line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=None, help="flight control .tg (default: the design's)")
    ap.add_argument("--speed", type=float, default=0.25, help="playback speed (1 = real time)")
    ap.add_argument("--open-loop", action="store_true", help="start with the closed loop off")
    ap.add_argument("--sensors", action="store_true", help="command control: fly on the command_observer (sensor model)")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="--sensors: datasheet noise x this")
    ap.add_argument("--stdin", action="store_true", help="command control: also read commands from stdin (the GUI)")
    args = ap.parse_args()
    control = args.control or args.tg.parent / "control.tg"

    fp = load_flight_policy(control)
    live = sensor = None
    if fp.command is not None:
        from hover.command import LiveCommand
        if args.sensors and fp.observer is None:
            raise SystemExit(f"{control} names no command_observer: fly without --sensors")
        live = LiveCommand(fp.command, fp.observer if args.sensors else None)
        if args.sensors:
            from hover import sensors as Sn
            noise = Sn.SensorNoise(pos_latency_ms=float(fp.command.get("sensor_pos_latency_ms", 5.0))).scaled(args.noise_scale)
            sensor = Sn.Sensors(noise, att_mode="filter", seed=int(time.time()))
    has_loop = fp.mlp is not None or live is not None
    bird = L.load_bird(args.tg, control=control)
    if live is not None:                       # free air, as evaluated (the floor is 0.25 m below the release)
        bird.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        planes = bird.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        bird.model.geom_pos[planes, 2] -= 100.0
    print(f"launching {control.name}: settling the bird in the launch cradle...", flush=True)
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock)
    sim = copy.deepcopy(rel.sim)
    drive = copy.copy(rel.drive); drive.sim = sim; drive.L_des = rel.drive.L_des.copy()
    m, d = sim.model, sim.data
    n_beat = round(2 * math.pi / bird.omega / m.opt.timestep)
    if sensor is not None:
        sensor.start(bird, rel, sim, fp.theta)
    state = {"closed": has_loop and not args.open_loop, "push": False}
    cmds: queue.Queue = queue.Queue()
    rng = np.random.default_rng()
    keys = Keys()
    if live is not None:
        drive_speed = [live.default_speed("FWD"), live.default_speed("UP")]     # m/s: forward/back, up/down
        v_top = max(live.envelope.values()) if live.envelope else 1.0

    def key(code: int) -> None:
        if code == ord("P"):
            state["push"] = True
        elif code == ord("O") and has_loop:
            cmds.put("LOOP")
        elif live is not None and code in KEY_ARROWS:
            keys.viewer_press(KEY_ARROWS[code])
        elif live is not None and code == KEY_HOME:
            keys.set_latch(0, 0)
        elif live is not None and code in (KEY_PLUS, KEY_MINUS, KEY_KP_ADD, KEY_KP_SUB):
            d = 0.05 if code in (KEY_PLUS, KEY_KP_ADD) else -0.05
            drive_speed[:] = [float(np.clip(v + d, 0.05, v_top)) for v in drive_speed]
            print(f"drive speed {drive_speed[0]:.2f} m/s (clipped per axis to the envelope)", flush=True)

    def handle(line: str, s) -> None:
        word, *rest = line.split()
        word = word.upper()
        if word == "PUSH":
            state["push"] = True
        elif word == "LOOP" and has_loop:
            state["closed"] = not state["closed"]
            if live is not None and state["closed"]:      # re-engage from where the bird is now
                live.reset()
                live.refs.p = s[:3].copy()
            print(f"closed loop {'ON' if state['closed'] else 'OFF'}", flush=True)
        elif live is not None and word in ("HOVER", "FWD", "REV", "UP", "DOWN"):
            f, z = {"HOVER": (0, 0), "FWD": (1, 0), "REV": (-1, 0), "UP": (0, 1), "DOWN": (0, -1)}[word]
            if rest:
                drive_speed[0 if f else 1] = abs(float(rest[0]))
            keys.set_latch(f, z)
            print(f"command {word}" + (f" {float(rest[0]):.2f} m/s" if rest else ""), flush=True)
        else:
            print(f"ignored: {line!r}", flush=True)

    if args.stdin:
        threading.Thread(target=read_stdin, args=(cmds,), daemon=True).start()
    help_keys = "P = push, O = toggle closed/open loop"
    if live is not None:
        help_keys += ", Right/Left = forward/back, Up/Down = climb/descend, Home = hover, +/- = speed"
        print(f"command control ({'sensors: ' + 'pos,att,w observer' if sensor else 'true state'}); envelope "
              + ", ".join(f"{k} {v:g}" for k, v in live.envelope.items()) + " m/s; " + keys.start_listener(), flush=True)
    print(f"released; {'closed' if state['closed'] else 'open'} loop. Keys: {help_keys}.", flush=True)
    sync_every = max(1, round(args.speed / 60.0 / m.opt.timestep))
    e_f = np.asarray(fp.command["e_f"]) if live is not None else None
    s = np.zeros(12)
    beat_s = n_beat * m.opt.timestep
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = bird.af
        viewer.cam.distance = 0.7
        if live is not None:                 # side-on: +x (the bird's forward) is screen right, as the arrows
            viewer.cam.azimuth, viewer.cam.elevation = 90.0, -12.0
        wall0, sim0, step = time.time(), d.time, 0
        while viewer.is_running():
            if step % n_beat == 0:
                while not cmds.empty():
                    handle(cmds.get(), s)
                if state["push"]:
                    state["push"] = False
                    L.kick(m, d, np.r_[np.zeros(6), rng.uniform(-0.3, 0.3, 3), rng.uniform(-1.5, 1.5, 3)])
                    print("push!", flush=True)
                mujoco.mj_forward(m, d)
                p_prev = s[:3]
                s = L.strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit
                v_mean = (s[:3] - p_prev) / beat_s              # mean motion over the last wingbeat
                obs = s if sensor is None else sensor.read(d, s)
                if live is not None:
                    f, z = keys.axes()
                    k = math.sqrt(0.5) if f and z else 1.0      # a diagonal keeps the single-axis total speed
                    live.set_velocity(k * f * drive_speed[0], k * z * drive_speed[1])
                    corr = live(obs) if state["closed"] else 0.0
                else:
                    corr = fp.correction(s) if state["closed"] else 0.0
                drive.set_theta(fp.theta + corr)
                tilt = math.degrees(np.linalg.norm(s[3:5]))
                if live is not None:
                    text = (f"{live.command} {live.speed:.2f} m/s{'  (sensors)' if sensor else ''}"
                            f"{'' if state['closed'] else '  LOOP OFF'}\n"
                            f"fwd {v_mean @ e_f:+.2f} m/s  up {v_mean[2]:+.2f} m/s  (mean over the last wingbeat)\n"
                            f"pos err {np.linalg.norm(s[:3] - live.x_ref[:3]) * 100 if live.x_ref is not None else 0:.1f} cm"
                            f"  tilt {tilt:.1f} deg  alt {s[2]:+.2f} m")
                    text += ("\nhold arrows: right/left fwd/back, up/down climb/descend" if keys.hold_mode else
                             "\narrows toggle: right/left fwd/back, up/down climb/descend, Home hover")
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Command", text))
                if step % (10 * n_beat) == 0:
                    print(f"t {d.time - sim0:6.2f} s  tilt {tilt:5.1f} deg  "
                          f"offset {np.linalg.norm(s[:3]) * 1e3:6.0f} mm  speed {np.linalg.norm(s[6:9]) * 1e3:5.0f} mm/s"
                          + (f"  [{live.command}]" if live is not None else ""), flush=True)
            drive.step()
            if sensor is not None:
                sensor.step(d)
            step += 1
            if step % sync_every == 0:
                viewer.sync()
                lag = (d.time - sim0) / args.speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
