"""Viewer pacing: step the sim at wall-clock speed (or slow motion) between viewer syncs."""

import time

import mujoco


def get_actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


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
