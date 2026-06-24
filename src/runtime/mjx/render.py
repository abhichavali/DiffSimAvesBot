"""Headless rendering of MJX rollouts for remote (Jupyter) viewing.

MJX runs the physics on the GPU; rendering uses the regular CPU
``mujoco.Renderer`` offscreen. On a headless server set ``MUJOCO_GL=egl`` (GPU
offscreen GL) **before importing mujoco** — e.g. at the top of the notebook:

    import os; os.environ["MUJOCO_GL"] = "egl"

Then replay a qpos trajectory (as returned by
``runtime.mjx.rollout.simulate`` with the default ``measure_fn``) through a CPU
``MjData`` and capture frames to show as inline video (``mediapy``/imageio).
"""

from __future__ import annotations

import numpy as np
import mujoco


def render_trajectory(
    model: mujoco.MjModel,
    qpos_traj,
    *,
    height: int = 480,
    width: int = 640,
    camera: int | str = -1,
    stride: int = 1,
) -> np.ndarray:
    """Render a ``(T, nq)`` qpos trajectory to a ``(frames, H, W, 3)`` uint8 array.

    ``stride`` subsamples timesteps (the sim runs at a tiny timestep, so render
    every Nth frame). ``camera`` is a name or id (-1 = free camera).
    """
    qpos_traj = np.asarray(qpos_traj)
    if qpos_traj.ndim != 2 or qpos_traj.shape[1] != model.nq:
        raise ValueError(f"expected (T, {model.nq}) qpos trajectory, got {qpos_traj.shape}")

    data = mujoco.MjData(model)
    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for t in range(0, qpos_traj.shape[0], stride):
            data.qpos[:] = qpos_traj[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    return np.asarray(frames)
