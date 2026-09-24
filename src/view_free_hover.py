"""Watch the free-standing bird hover under a flight control .tg: cradle launch, release, real-time flight.

    mjpython src/view_free_hover.py [--tg designs/hummer/free-hover/shoulder.tg]
        [--control designs/hummer/free-hover/control.tg] [--speed 0.25] [--open-loop]

The control's open loop (per-crank Fourier drive, phase-locked cranks) flies the bird; its closed loop (the
per-wingbeat MLP, if the file names one) corrects the harmonics once per wingbeat. Nothing else acts on the
bird after release. In the viewer window: P = push the bird (random 0.3 m/s + 1.5 rad/s kick),
O = toggle closed / open loop. The GUI's Viewer tab launches this for flight controls.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import mujoco.viewer
import numpy as np

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover.policy import load_flight_policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=None, help="flight control .tg (default: the design's)")
    ap.add_argument("--speed", type=float, default=0.25, help="playback speed (1 = real time)")
    ap.add_argument("--open-loop", action="store_true", help="start with the closed loop off")
    args = ap.parse_args()
    control = args.control or args.tg.parent / "control.tg"

    fp = load_flight_policy(control)
    bird = L.load_bird(args.tg, control=control)
    print(f"launching {control.name}: settling the bird in the launch cradle...", flush=True)
    rel = L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock)
    sim = copy.deepcopy(rel.sim)
    drive = copy.copy(rel.drive); drive.sim = sim; drive.L_des = rel.drive.L_des.copy()
    m, d = sim.model, sim.data
    n_beat = round(2 * math.pi / bird.omega / m.opt.timestep)
    state = {"closed": fp.mlp is not None and not args.open_loop, "push": False}
    rng = np.random.default_rng()

    def key(code: int) -> None:
        if code == ord("P"):
            state["push"] = True
        elif code == ord("O") and fp.mlp is not None:
            state["closed"] = not state["closed"]
            print(f"closed loop {'ON' if state['closed'] else 'OFF'}", flush=True)

    print(f"released; {'closed' if state['closed'] else 'open'} loop. Keys: P = push, O = toggle closed/open loop.", flush=True)
    sync_every = max(1, round(args.speed / 60.0 / m.opt.timestep))
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = bird.af
        viewer.cam.distance = 0.7
        wall0, sim0, step = time.time(), d.time, 0
        while viewer.is_running():
            if step % n_beat == 0:
                if state["push"]:
                    state["push"] = False
                    L.kick(m, d, np.r_[np.zeros(6), rng.uniform(-0.3, 0.3, 3), rng.uniform(-1.5, 1.5, 3)])
                    print("push!", flush=True)
                mujoco.mj_forward(m, d)
                s = L.strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit
                drive.set_theta(fp.theta + (fp.correction(s) if state["closed"] else 0.0))
                if step % (10 * n_beat) == 0:
                    print(f"t {d.time - sim0:6.2f} s  tilt {math.degrees(np.linalg.norm(s[3:5])):5.1f} deg  "
                          f"offset {np.linalg.norm(s[:3]) * 1e3:6.0f} mm  speed {np.linalg.norm(s[6:9]) * 1e3:5.0f} mm/s", flush=True)
            drive.step()
            step += 1
            if step % sync_every == 0:
                viewer.sync()
                lag = (d.time - sim0) / args.speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
