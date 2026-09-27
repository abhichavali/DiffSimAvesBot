"""Per-wingbeat linear model of the free bird around its phase-locked open-loop trim, and an LQR on it.

State = the strobed deviation (hover.launch.strobe - orbit), input = corrections to the 24 harmonic depths
(crank speed fixed). A, B come from one-wingbeat central differences from the released state: rigid
rotations/kicks of the whole bird for A, harmonic perturbations for B (position columns are exact: the
dynamics are translation invariant).

The same rollouts also identify the sensor channels (hover.sensors, noise-free) for hover.reduced:
    y_k = C_now x_k + C_prev x_{k-1} + J u_{k-1}
C_now (18x12): instantaneous channels (pos att vel w gyro) read right after the perturbation, regressed on
the perturbed strobed state (so it is in the flight's own state coordinates: C_now[:12] = I);
C_prev, J: the interval channels (acc, a mean over the last wingbeat) after one wingbeat. Saved next to
A, B, K in the npz (with ``channels``).

    PYTHONPATH=src uv run python -m hover.lqr [--control designs/hummer/free-hover/control.tg] --out lqr.npz
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from multiprocessing import Pool

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover import sensors as S
from hover.policy import load_flight_policy

_W = {}


def _init(tg, control):
    fp = load_flight_policy(control)
    bird = L.load_bird(tg, control=control)
    _W.update(bird=bird, theta=fp.theta, rel=L.release(bird, fp.theta, fp.attitude, fp.K, k_phi=fp.phase_lock))


def one_beat(args):
    """Strobed state one wingbeat after a rigid perturbation d12 (att: rot-vector, vel, w) and input du (24)."""
    return one_beat_obs(args)[0]


def one_beat_obs(args):
    """(strobed state after one wingbeat, noise-free sensor channels right after the perturbation, the same
    after one wingbeat, the strobed state right after the perturbation) for a rigid perturbation d12 + input du."""
    d12, du = args
    bird, theta, rel = _W["bird"], _W["theta"], _W["rel"]
    sim = copy.deepcopy(rel.sim); drive = copy.copy(rel.drive); drive.sim = sim; drive.L_des = rel.drive.L_des.copy()
    m, d = sim.model, sim.data
    if np.linalg.norm(d12[3:6]) > 0:
        a = np.linalg.norm(d12[3:6]); q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, d12[3:6] / a, a); L.rotate_bird(m, d, q)
    L.kick(m, d, d12)
    sen = S.Sensors(S.SensorNoise.zero(), att_mode="truth", reference=(np.zeros(3), np.zeros(3), None))
    sen.start(bird, rel, sim)
    s0 = L.strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit
    y0 = sen.read(d, s0)
    drive.set_theta(theta + np.concatenate([[0.0], du]))
    drive.beat(sen.step)
    mujoco.mj_forward(m, d)
    s1 = L.strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit
    return s1, y0, sen.read(d, s1), s0


def linearize(pool, K):
    nu = 8 * K
    sx = np.array([0.01] * 3 + [0.02] * 3 + [0.1] * 3)            # att, vel, w steps
    su = np.full(nu, 0.01)
    tasks = [(np.zeros(12), np.zeros(nu))]
    for i, s in enumerate(sx):
        for sg in (1, -1):
            d = np.zeros(12); d[3 + i] = sg * s; tasks.append((d, np.zeros(nu)))
    for j, s in enumerate(su):
        for sg in (1, -1):
            du = np.zeros(nu); du[j] = sg * s; tasks.append((np.zeros(12), du))
    res = pool.map(one_beat_obs, tasks)
    out = [r[0] for r in res]
    A = np.zeros((12, 12)); A[:3, :3] = np.eye(3)                  # translation invariance
    for i, s in enumerate(sx):
        A[:, 3 + i] = (out[1 + 2 * i] - out[2 + 2 * i]) / (2 * s)
    o = 1 + 2 * len(sx)
    B = np.stack([(out[o + 2 * j] - out[o + 1 + 2 * j]) / (2 * su[j]) for j in range(nu)], 1)
    # position also integrates the velocity over the beat: already in the pos rows of the velocity columns
    linearize.obs = observation_model(res, sx, su)
    return A, B, out[0]


def observation_model(res, sx, su):
    """C_now, C_prev, J (see the module docstring) from one_beat_obs results laid out as in linearize."""
    n9, nu = len(sx), len(su)
    dS0 = np.stack([res[1 + 2 * i][3] - res[2 + 2 * i][3] for i in range(n9)], 1)[3:]       # (9, 9) true dx0
    dY0 = np.stack([res[1 + 2 * i][1] - res[2 + 2 * i][1] for i in range(n9)], 1)           # (18, 9)
    dY1 = np.stack([res[1 + 2 * i][2] - res[2 + 2 * i][2] for i in range(n9)], 1)
    inst = np.setdiff1d(np.arange(S.N_CH), S.INTERVAL)
    itv = np.asarray(S.INTERVAL)
    C_now, C_prev, J = np.zeros((S.N_CH, 12)), np.zeros((S.N_CH, 12)), np.zeros((S.N_CH, nu))
    C_now[inst, 3:] = (dY0 @ np.linalg.inv(dS0))[inst]
    C_now[0:3, 0:3] = np.eye(3)                                    # translation: only the position channel moves
    C_prev[itv, 3:] = (dY1 @ np.linalg.inv(dS0))[itv]
    o = 1 + 2 * n9
    J[itv] = np.stack([(res[o + 2 * j][2] - res[o + 1 + 2 * j][2]) / (2 * su[j]) for j in range(nu)], 1)[itv]
    return {"C_now": C_now, "C_prev": C_prev, "J": J, "y_drift": res[0][2]}


def cost(nu, *, q_pos=0.02, q_att=0.05, q_vel=0.1, q_w=0.5, r=0.05):
    """LQR weights (Q 12x12, R nu x nu): tolerances 2 cm, 0.05 rad, 0.1 m/s, 0.5 rad/s; input 0.05."""
    Q = np.diag([1 / q_pos**2] * 3 + [1 / q_att**2] * 3 + [1 / q_vel**2] * 3 + [1 / q_w**2] * 3)
    return Q, np.eye(nu) / r**2


def lqr(A, B, *, Q=None, R=None, **weights):
    if Q is None or R is None:
        Q, R = cost(B.shape[1], **weights)
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=FREE_HOVER_TG)
    ap.add_argument("--control", type=Path, default=FREE_HOVER_TG.parent / "control.tg")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    K = load_flight_policy(a.control).K
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.tg, a.control)) as pool:
        A, B, drift = linearize(pool, K)
    K = lqr(A, B)
    ol = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]; cl = np.sort(np.abs(np.linalg.eigvals(A - B @ K)))[::-1]
    print("open-loop |eig| per wingbeat:", np.round(ol, 3))
    print("closed-loop |eig| (LQR):     ", np.round(cl, 3))
    print("one-beat drift of the unperturbed orbit:", np.round(drift, 4))
    obs = linearize.obs
    print("sensor model: C_now[:12] - I max", np.abs(obs["C_now"][:12] - np.eye(12)).max().round(6),
          "| acc rows |C_prev| max", np.abs(obs["C_prev"]).max().round(3), "|J| max", np.abs(obs["J"]).max().round(3))
    np.savez(a.out, A=A, B=B, K=K, drift=drift, channels=np.array(S.CHANNELS), **obs)


if __name__ == "__main__":
    main()
