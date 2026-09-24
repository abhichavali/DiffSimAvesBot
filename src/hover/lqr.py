"""Per-wingbeat linear model of the free bird around its phase-locked open-loop trim, and an LQR on it.

State = the strobed deviation (hover.launch.strobe - orbit), input = corrections to the 24 harmonic depths
(crank speed fixed). A, B come from one-wingbeat central differences from the released state: rigid
rotations/kicks of the whole bird for A, harmonic perturbations for B (position columns are exact: the
dynamics are translation invariant).

    python -m hover.lqr --trim trim_locked.json --out lqr.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from multiprocessing import Pool

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from design_paths import FREE_HOVER_TG
from hover import launch as L
from hover import trim as T

_W = {}


def _init(trim_path):
    t = json.load(open(trim_path)); x, att0 = np.array(t["x"]), np.array(t["att0"])
    bird = L.load_bird(FREE_HOVER_TG, theta_g=x[0])
    _W.update(bird=bird, x=x, rel=L.release(bird, x[:T.NT], T.attitude(att0, x[T.NT:]), T.K))


def one_beat(args):
    """Strobed state one wingbeat after a rigid perturbation d12 (att: rot-vector, vel, w) and input du (24)."""
    d12, du = args
    bird, x, rel = _W["bird"], _W["x"], _W["rel"]
    sim = copy.deepcopy(rel.sim); drive = copy.copy(rel.drive); drive.sim = sim; drive.L_des = rel.drive.L_des.copy()
    m, d = sim.model, sim.data
    if np.linalg.norm(d12[3:6]) > 0:
        a = np.linalg.norm(d12[3:6]); q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, d12[3:6] / a, a); L.rotate_bird(m, d, q)
    L.kick(m, d, d12)
    drive.set_theta(x[:T.NT] + np.concatenate([[0.0], du]))
    drive.beat()
    mujoco.mj_forward(m, d)
    return L.strobe(bird, d, rel.com_ref, rel.q_ref) - rel.x_orbit


def linearize(trim_path, pool):
    nu = 8 * T.K
    sx = np.array([0.01] * 3 + [0.02] * 3 + [0.1] * 3)            # att, vel, w steps
    su = np.full(nu, 0.01)
    tasks = [(np.zeros(12), np.zeros(nu))]
    for i, s in enumerate(sx):
        for sg in (1, -1):
            d = np.zeros(12); d[3 + i] = sg * s; tasks.append((d, np.zeros(nu)))
    for j, s in enumerate(su):
        for sg in (1, -1):
            du = np.zeros(nu); du[j] = sg * s; tasks.append((np.zeros(12), du))
    out = pool.map(one_beat, tasks)
    A = np.zeros((12, 12)); A[:3, :3] = np.eye(3)                  # translation invariance
    for i, s in enumerate(sx):
        A[:, 3 + i] = (out[1 + 2 * i] - out[2 + 2 * i]) / (2 * s)
    o = 1 + 2 * len(sx)
    B = np.stack([(out[o + 2 * j] - out[o + 1 + 2 * j]) / (2 * su[j]) for j in range(nu)], 1)
    # position also integrates the velocity over the beat: already in the pos rows of the velocity columns
    return A, B, out[0]


def lqr(A, B, *, q_pos=0.02, q_att=0.05, q_vel=0.1, q_w=0.5, r=0.05):
    Q = np.diag([1 / q_pos**2] * 3 + [1 / q_att**2] * 3 + [1 / q_vel**2] * 3 + [1 / q_w**2] * 3)
    R = np.eye(B.shape[1]) / r**2
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trim", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.trim,)) as pool:
        A, B, drift = linearize(a.trim, pool)
    K = lqr(A, B)
    ol = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]; cl = np.sort(np.abs(np.linalg.eigvals(A - B @ K)))[::-1]
    print("open-loop |eig| per wingbeat:", np.round(ol, 3))
    print("closed-loop |eig| (LQR):     ", np.round(cl, 3))
    print("one-beat drift of the unperturbed orbit:", np.round(drift, 4))
    np.savez(a.out, A=A, B=B, K=K, drift=drift)


if __name__ == "__main__":
    main()
