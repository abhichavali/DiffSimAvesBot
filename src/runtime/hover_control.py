"""The free-flight hover controller: a bounded 6-DOF stabilizing wrench on the airframe.

The flapping carries the (gravity-matched) hover lift; this state feedback supplies the
attitude and position authority the mechanism lacks:

    u = K @ scaled [pos_err(3), linvel(3), att_err(3), angvel(3)] + b     (6,)
    force = f_max*tanh(u[:3]) (world),  tau = tau_max*tanh(u[3:]) (body)

The same functions run on numpy (src/view_hover.py) and jnp (opt/hover_policy.py, the
trainer): pass ``xp``. level_stroke_plane flattens the crank-disk mounts for hover.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from runtime.disk_orientation import DiskOrientation

N_FEAT = 12
N_CTRL = 6  # [Fx, Fy, Fz, tau_x, tau_y, tau_z] airframe stabilizing wrench (world force, body torque)

# Normalize feature groups to ~O(1) so the feedback gains the controller needs are
# also ~O(1) and reachable from K=0 in a few hundred adam steps. Without this,
# holding a 5 cm position error needs a gain ~4 (meters are tiny) that grows only
# ~lr/iter -> position feedback stays untrained. Order: pos, linvel, att, angvel.
FEATURE_SCALE = np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0])


def level_stroke_plane(model) -> None:
    """Flatten the crank-disk mounts (tilt=pitch=0) so the flap is a pure vertical
    stroke with no built-in forward thrust -- hovering in place is then a matter of
    attitude + altitude hold, not fighting the design's 20-deg thrust. Mutates
    model.body_quat in place; call before put_model / calibration."""
    DiskOrientation(model).set(0.0, 0.0)


class HoverCfg(NamedTuple):
    """Static wiring shared by trainer (jnp) and viewer (np)."""
    a0: float
    f_max: float     # N   (per-axis force bound)
    tau_max: float   # Nm  (per-axis torque bound)
    qadr: int        # airframe freejoint qpos address
    dofadr: int      # airframe freejoint dof address


def hover_features(qpos, qvel, cfg: HoverCfg, com0, xp):
    """[pos_err(3), linvel(3), att_err(3), angvel(3)] for the airframe freejoint."""
    q, v = cfg.qadr, cfg.dofadr
    pos = qpos[q : q + 3]
    quat = qpos[q + 3 : q + 7]                     # wxyz
    lin = qvel[v : v + 3]
    ang = qvel[v + 3 : v + 6]
    att = 2.0 * xp.sign(quat[0]) * quat[1:4]       # small-angle rotation vector vs upright
    return xp.concatenate([pos - com0, lin, att, ang]) * FEATURE_SCALE


def policy_commands(K, b, feats, cfg: HoverCfg, xp):
    """feats -> (force[3], tau[3], u[6]).  Bounded wrench, tanh-saturated."""
    u = K @ feats + b
    force = cfg.f_max * xp.tanh(u[:3])
    tau = cfg.tau_max * xp.tanh(u[3:6])
    return force, tau, u


def pd_init_theta(cfg: HoverCfg, *, kp_pos=3.0, kd_pos=1.2, kp_att=0.05, kd_att=0.012):
    """A physically-sized stabilizing PD as the policy starting point (diagonal K):
    position/velocity -> force, attitude/rate -> torque. Training a stabilizer via
    BPTT from K=0 explodes (the plant tumbles -> exponential gradients); starting
    from a controller that already keeps trajectories bounded fixes that. Gains are
    expressed in the SCALED-feature / tanh convention so u stays in the linear range.
    """
    K = np.zeros((N_CTRL, N_FEAT))
    for i in range(3):  # force axes: -kp*pos_i - kd*vel_i
        K[i, i]     = -kp_pos / (cfg.f_max * FEATURE_SCALE[i])
        K[i, 3 + i] = -kd_pos / (cfg.f_max * FEATURE_SCALE[3 + i])
    for i in range(3):  # torque axes: -kp*att_i - kd*angvel_i
        K[3 + i, 6 + i] = -kp_att / (cfg.tau_max * FEATURE_SCALE[6 + i])
        K[3 + i, 9 + i] = -kd_att / (cfg.tau_max * FEATURE_SCALE[9 + i])
    return np.concatenate([K.reshape(-1), np.zeros(N_CTRL)])
