import time

import mujoco
import numpy as np


def get_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, name)


def get_first_existing_body_id(model, names):
    for name in names:
        body_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            return body_id
    return -1


def get_actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def settle(model, data, steps=500):
    for _ in range(steps):
        mujoco.mj_step(model, data)


class RealtimeSync:
    def __init__(self, data, fps=60.0, realtime_factor=1.0):
        # realtime_factor scales playback speed: 1.0 = realtime, 0.25 = 4x
        # slow-motion (sim advances 4x slower than wall clock). Useful for
        # eyeballing a fast flap (e.g. 94 rad/s ~ 15 Hz is a blur at 1.0).
        self.fps = fps
        self.realtime_factor = max(1e-6, float(realtime_factor))
        self.wall_start = time.time()
        self.sim_start = data.time
        self.step_count = 0

    def step_and_sync(
        self,
        model,
        data,
        viewer,
        pre_mj_step_hook=None,
    ):
        t0 = data.time

        # Advance only `realtime_factor / fps` of SIM time per rendered frame
        # (not 1/fps), so slow-motion comes from less sim-per-frame, NOT from
        # stretching each frame. That keeps the wall-clock framerate at ~fps
        # (smooth) instead of dropping it to fps*realtime_factor. Always step at
        # least once so the sim can't stall at very low factors.
        sim_per_frame = self.realtime_factor / self.fps
        stepped = False
        while not stepped or data.time - t0 < sim_per_frame:
            data.qfrc_applied.fill(0.0)

            if pre_mj_step_hook is not None:
                pre_mj_step_hook(model, data)

            mujoco.mj_step(model, data)
            self.step_count += 1
            stepped = True

        # Cumulative target wall time = sim-elapsed / realtime_factor, so each
        # frame lands ~1/fps wall apart -> steady ~fps rendering.
        sim_elapsed = data.time - self.sim_start
        dt = sim_elapsed / self.realtime_factor - (time.time() - self.wall_start)
        if dt > 0:
            time.sleep(dt)
        viewer.sync()


class Logger:
    def __init__(self, model, body_names, interval=0.5):
        self.ids = {name: get_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
        self.interval = interval
        self.last_time = 0.0

    def log(self, data, **extras):
        if data.time - self.last_time < self.interval:
            return

        parts = [f"t={data.time:5.2f}s"]
        for key, value in extras.items():
            parts.append(f"{key}={value:+.3f}")
        for name, body_id in self.ids.items():
            parts.append(f"{name}={data.xpos[body_id]}")

        print(" | ".join(parts))
        self.last_time = data.time


def get_free_body_names(model):
    names = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and model.body_jntnum[body_id] > 0 and model.jnt_type[model.body_jntadr[body_id]] == mujoco.mjtJoint.mjJNT_FREE:
            names.append(name)
    return names
