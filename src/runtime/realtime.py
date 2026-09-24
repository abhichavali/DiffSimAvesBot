"""Viewer pacing: step the sim at wall-clock speed (or slow motion) between viewer syncs."""

import time

import mujoco


class RealtimeSync:
    """``realtime_factor`` scales playback: 1.0 = realtime, 0.25 = 4x slow motion."""

    def __init__(self, data, fps=60.0, realtime_factor=1.0):
        self.fps = fps
        self.realtime_factor = max(1e-6, float(realtime_factor))
        self.wall_start = time.time()
        self.sim_start = data.time

    def step_and_sync(self, model, data, viewer, pre_mj_step_hook=None):
        """Advance one frame of sim time (at least one step), sleep to pace it, sync the viewer.

        Each step zeroes ``qfrc_applied``, runs ``pre_mj_step_hook(model, data)``, then
        ``mj_step``. Slow motion comes from less sim per frame, not longer frames, so
        the wall-clock framerate stays at ~fps.
        """
        t0 = data.time
        sim_per_frame = self.realtime_factor / self.fps
        stepped = False
        while not stepped or data.time - t0 < sim_per_frame:
            data.qfrc_applied.fill(0.0)
            if pre_mj_step_hook is not None:
                pre_mj_step_hook(model, data)
            mujoco.mj_step(model, data)
            stepped = True

        # Cumulative target wall time = sim-elapsed / realtime_factor, so each
        # frame lands ~1/fps wall apart.
        lag = (data.time - self.sim_start) / self.realtime_factor - (time.time() - self.wall_start)
        if lag > 0:
            time.sleep(lag)
        viewer.sync()
