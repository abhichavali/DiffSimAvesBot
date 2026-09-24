"""CPU side of free-flight hover: the phase-locked Fourier drive, the launch cradle, the strobed state.

Drive: each crank tracks its own Fourier velocity command AND the integral of it (encoder phase lock),
    ctrl_i = w_i(phi) + K_PHI (L_des_i - L_i),   L_des_i += w_i(phi) dt,
    w_i(phi) = A0 e^g (1 + sum_k a_ik cos k phi + b_ik sin k phi)        (actuator space)
Without the lock the velocity servos let the t1/t4 crank lag drift (-0.07 / +0.05 deg per wingbeat, L/R
opposite), which slides the trim during flight.

Launch: a mean-only cradle holds the bird's wingbeat-averaged pose (PID on the one-wingbeat boxcar of the
pose error: the wobble stays free) while the flapping settles; it is removed at release. No force of any
kind acts after release.

State (strobed at the start of every wingbeat): COM position (3), attitude rotation vector vs the release
attitude (3), COM velocity (3), angular velocity in world (3) - relative to the orbit's own release state.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

import flying
from morphology.compile import compile_file
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control_for_morphology
from runtime.sim import PlateWingSim

K_PHI = 20.0            # crank phase-lock gain [1/s]
N_MOTORS = 4


def unpack(theta, K):
    return theta[0], np.reshape(theta[1:1 + N_MOTORS * K], (N_MOTORS, K)), np.reshape(theta[1 + N_MOTORS * K:], (N_MOTORS, K))


def rotate_bird(model, data, quat) -> None:
    """Rigidly rotate every free body (airframe + tendon-held struts) about the bird's COM."""
    com = (model.body_mass[:, None] * data.xipos).sum(0) / model.body_mass.sum()
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        a = model.jnt_qposadr[j]
        data.qpos[a:a + 3] = com + R @ (data.qpos[a:a + 3] - com)
        q = np.zeros(4); mujoco.mju_mulQuat(q, quat, data.qpos[a + 3:a + 7].copy()); data.qpos[a + 3:a + 7] = q
    mujoco.mj_forward(model, data)


def kick(model, data, dx) -> None:
    """Rigid velocity kick of the whole bird: dx[6:9] COM velocity, dx[9:12] angular velocity (world)."""
    com = (model.body_mass[:, None] * data.xipos).sum(0) / model.body_mass.sum()
    for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        v, b = model.jnt_dofadr[j], model.jnt_bodyid[j]
        R = data.xmat[b].reshape(3, 3)
        data.qvel[v:v + 3] += dx[6:9] + np.cross(dx[9:12], data.xpos[b] - com)
        data.qvel[v + 3:v + 6] += R.T @ dx[9:12]
    mujoco.mj_forward(model, data)


@dataclass
class Bird:
    model: mujoco.MjModel
    morph: object
    a0: float            # signed design crank speed (actuator space) [rad/s]
    omega: float         # design flap angular frequency [rad/s]
    qadr: int
    vadr: int
    af: int              # airframe body id


def load_bird(tg, *, dt_ms=0.13, theta_g=0.0) -> Bird:
    """The free-standing bird, with dt snapped so one wingbeat at e^g x the design rate is an integer
    number of steps (exact phase locking of the per-wingbeat strobe)."""
    morph = load_morphology(tg)
    ctl = load_control_for_morphology(tg)
    m = mujoco.MjModel.from_xml_string(flying.to_airframe(compile_file(tg)))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    wf = ctl.omega * math.exp(theta_g)
    m.opt.timestep = 2 * math.pi / wf / round(2 * math.pi / wf / (dt_ms * 1e-3))
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)
    return Bird(m, morph, morph.motors.motor_direction * ctl.a0, ctl.omega, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "airframe"))


class Drive:
    """The phase-locked per-crank Fourier drive (CPU)."""

    def __init__(self, bird: Bird, sim: PlateWingSim, theta, K):
        self.bird, self.sim, self.K = bird, sim, K
        self.set_theta(theta)
        self.phi = 0.0
        self.L_des = sim.data.actuator_length[sim.crank_ids].copy()

    def set_theta(self, theta):
        self.g, self.a, self.b = unpack(np.asarray(theta, float), self.K)

    def step(self):
        ks = np.arange(1, self.K + 1)
        w = self.bird.a0 * math.exp(self.g) * (1 + (self.a * np.cos(ks * self.phi) + self.b * np.sin(ks * self.phi)).sum(1))
        d = self.sim.data
        ctrl = w + K_PHI * (self.L_des - d.actuator_length[self.sim.crank_ids])
        lift = self.sim.step(list(ctrl))
        dt = self.bird.model.opt.timestep
        self.L_des += w * dt
        self.phi += self.bird.omega * math.exp(self.g) * dt
        return lift

    def beat(self):
        """Advance exactly one wingbeat (integer steps by construction of dt)."""
        n = round(2 * math.pi / (self.bird.omega * math.exp(self.g)) / self.bird.model.opt.timestep)
        for _ in range(n):
            self.step()


def strobe(bird: Bird, d, com_ref, q_ref):
    q, v, af = bird.qadr, bird.vadr, bird.af
    com = d.subtree_com[0]; R = d.xmat[af].reshape(3, 3); w_world = R @ d.qvel[v + 3:v + 6]
    qc = np.zeros(4); mujoco.mju_negQuat(qc, q_ref); qe = np.zeros(4); mujoco.mju_mulQuat(qe, d.xquat[af], qc)
    vcom = d.qvel[v:v + 3] + np.cross(w_world, com - d.qpos[q:q + 3])
    return np.concatenate([com - com_ref, 2 * np.sign(qe[0]) * qe[1:], vcom, w_world])


@dataclass
class Released:
    sim: PlateWingSim
    drive: Drive
    com_ref: np.ndarray
    q_ref: np.ndarray
    x_orbit: np.ndarray
    cradle_wrench: np.ndarray    # mean cradle [force, moment] (world) over its last n_avg wingbeats = trim error


def release(bird: Bird, theta, attitude, K, *, n_cradle=12, n_avg=6,
            body_inertia=4e-5 + flying.BODY_INERTIA) -> Released:
    """Settle the bird on ``theta`` at ``attitude`` (wxyz) in the mean-only cradle, then release it."""
    m, af = bird.model, bird.af
    m.opt.gravity[:] = 0
    sim = PlateWingSim(m, bird.morph); d = sim.data; d.qvel[:] = 0
    rotate_bird(m, d, np.asarray(attitude, float))
    m.opt.gravity[:] = (0, 0, -9.81); mujoco.mj_forward(m, d)
    M, I, ws = m.body_mass.sum(), body_inertia, 2 * math.pi * 1.5
    kp, kd, ki = M * ws**2, 2 * 0.8 * M * ws, M * ws**3 / 4
    kr, krd, kri = I * ws**2, 2 * 0.8 * I * ws, I * ws**3 / 4
    com0, q0 = d.subtree_com[0].copy(), d.xquat[af].copy()
    drive = Drive(bird, sim, theta, K)
    n = round(2 * math.pi / (bird.omega * math.exp(drive.g)) / m.opt.timestep)
    buf, iE, acc = deque(maxlen=n), np.zeros(6), []
    for i in range(n_cradle * n):
        s = strobe(bird, d, com0, q0)
        buf.append(s); avg = np.mean(buf, 0); iE += avg[:6] * m.opt.timestep
        F = -kp * avg[:3] - kd * avg[6:9] - ki * iE[:3]; T = -kr * avg[3:6] - krd * avg[9:12] - kri * iE[3:]
        d.xfrc_applied[af, :3] = F; d.xfrc_applied[af, 3:] = T + np.cross(d.subtree_com[0] - d.xipos[af], F)
        drive.step()
        if i >= (n_cradle - n_avg) * n:
            acc.append(np.concatenate([F, T]))
        if not np.isfinite(d.qpos).all():
            raise FloatingPointError("unstable in the cradle")
    d.xfrc_applied[af] = 0.0
    mujoco.mj_forward(m, d)
    com_ref, q_ref = d.subtree_com[0].copy(), d.xquat[af].copy()
    return Released(sim, drive, com_ref, q_ref, strobe(bird, d, com_ref, q_ref), np.mean(acc, 0))


def fly(bird: Bird, rel: Released, theta, n_beats, *, policy=None, kicks=None):
    """Fly from a released state (copied, so ``rel`` is reusable). ``policy(s) -> correction to theta``
    once per wingbeat. Returns the strobed deviation states (n_beats+1, 12)."""
    sim = copy.deepcopy(rel.sim); drive = copy.copy(rel.drive); drive.sim = sim; drive.L_des = rel.drive.L_des.copy()
    m, d = sim.model, sim.data
    out = []
    for n in range(n_beats):
        if kicks and n in kicks:
            kick(m, d, kicks[n])
        mujoco.mj_forward(m, d)
        s = strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit
        out.append(s)
        if not np.isfinite(s).all() or np.linalg.norm(s[3:5]) > math.radians(60):
            break
        drive.set_theta(np.asarray(theta) + (policy(s) if policy is not None else 0.0))
        drive.beat()
    else:
        mujoco.mj_forward(m, d)
        out.append(strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit)
    return np.array(out)
