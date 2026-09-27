"""Hover controllers that read only a subset of the sensor channels: static output feedback (SOF) and LQG.

Works on the per-wingbeat linear model hover.lqr saves (A, B, K + the sensor model C_now, C_prev, J):
    x_{k+1} = A x_k + B u_k + drift,        y_k = C_now x_k + C_prev x_{k-1} + J u_{k-1} + v_k
(y = the hover.sensors channels, 18; acc is a mean over the last wingbeat, hence x_{k-1} and u_{k-1}).
Augmented state z_k = [x_k, x_{k-1}]:  z_{k+1} = Az z_k + Bz u_k,  y~_k = y_k - J u_{k-1} = Cz z_k.

Design states: position axes that no selected channel measures are dropped from the design model (they are
pure integrators, A[:, pos] = [I; 0], unobservable and undetectable: nothing can hold position without
measuring it); the controller then regulates attitude, velocity and rates only. Everything else is kept.

- ``sof``: u = -Ky y~, Ky minimizing the same LQR cost E[sum x'Qx + u'Ru] over initial states ~ STATE_STD
  (Anderson-Moore style gradient of the discrete Lyapunov cost, L-BFGS), started from the projection of the
  full-state gain; if that start is unstable, a discount homotopy (Az, Bz scaled by alpha < 1, then alpha -> 1).
- ``lqg``: steady-state Kalman filter on (Az, Cz) with V from the sensor noise model (SensorNoise.channel_std)
  and process noise W (PROCESS_STD x process_scale), plus the LQR gain on the design states.
- ``state``: the full-state LQR on the true-state channels (what verify_hover --lqr flies).
- ``lqg --dead-reckon``: integral action for the dropped position axes. They are predicted open loop from the
  estimate and fed back with the full 12-state LQR gain, so a constant disturbance (the orbit's drift term, model
  error) leaves no steady velocity. Without it a position-free controller drifts at a steady ~15-20 mm/s even
  with the true velocity; with it the drift is only the integrated velocity-estimate error.
- ``lqg --bias att_xy``: the Kalman filter also estimates a constant bias on the named channels (random walk
  BIAS_WALK per wingbeat), e.g. the attitude filter's tilt error from the accelerometer bias. It needs a channel
  that sees the bias's effect (position or velocity); the closed-loop rho leaves out the bias states' unit
  eigenvalues. It did not help at datasheet noise (docs/findings/2026-09-26-hover-noise-tuning-dead-ends.md).

Which subsets hover: docs/findings/2026-09-26-hover-minimal-inputs.md (noise-free) and
2026-09-26-hover-imu-noise.md (datasheet noise; design with --noise-scale 1).

    PYTHONPATH=src python -m hover.reduced --lqr lqr.npz --obs att,w --mode lqg [--noise-scale 1] [--dead-reckon] [--bias att_xy] --out ctl.npz
    PYTHONPATH=src python -m hover.reduced --lqr lqr.npz --report "state; pos,att,w; att,w; att,w,acc; imu"
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from hover import sensors as S
from hover.lqr import cost, lqr

U_MAX = 0.15
STATE_NAMES = S.CHANNELS[:12]
# initial-state spread for the SOF cost and the distillation (m, rad, m/s, rad/s): push-sized
STATE_STD = np.array([0.03] * 3 + [0.05] * 3 + [0.15] * 3 + [0.7] * 3)
# per-wingbeat process noise of the Kalman filter (model error; pushes are much larger and are not in W)
PROCESS_STD = np.array([0.5e-3] * 3 + [2e-3] * 3 + [5e-3] * 3 + [0.03] * 3)
V_FLOOR = 1e-3 / S.CHANNEL_SCALE        # measurement std floor (keeps the Riccati equation regular at noise 0)
# lqg bias states (``bias=``): per-wingbeat random walk of a channel bias, and the white std left on that channel
# once its bias is a state (att: the Mahony filter's beat-to-beat jitter, ~0.1 deg at datasheet noise)
BIAS_WALK = 3e-4
BIAS_WHITE = 2e-3


def load_model(path) -> dict:
    z = np.load(path)
    if "C_now" not in z.files:
        raise ValueError(f"{path} has no sensor model (C_now): re-run python -m hover.lqr with this code")
    return {k: z[k] for k in ("A", "B", "K", "drift", "C_now", "C_prev", "J")}


def design_states(idx) -> np.ndarray:
    """State indices the controller is designed on: every non-position state + the measured position axes."""
    return np.array(sorted({int(i) for i in idx if i < 3} | set(range(3, 12))))


def augment(model, idx, xs=None, pos_latency_s=0.0):
    """(Az, Bz, Cz, J_sel, c_z) of the augmented system on design states xs for channels idx."""
    idx = S.parse_obs(idx)
    xs = design_states(idx) if xs is None else np.asarray(xs)
    A, B = model["A"], model["B"]
    Cn, Cp = model["C_now"].copy(), model["C_prev"]
    Cn[0:3, 6:9] -= pos_latency_s * np.eye(3)        # latency: pos(t - L) = pos - L vel
    drop = np.setdiff1d(np.arange(12), xs)
    if drop.size and (np.abs(A[np.ix_(xs, drop)]).max(initial=0) > 0 or np.abs(Cn[np.ix_(idx, drop)]).max(initial=0) > 0
                      or np.abs(Cp[np.ix_(idx, drop)]).max(initial=0) > 0):
        raise ValueError("dropped states still drive the kept states or the selected channels")
    n = len(xs)
    Az = np.block([[A[np.ix_(xs, xs)], np.zeros((n, n))], [np.eye(n), np.zeros((n, n))]])
    Bz = np.vstack([B[xs], np.zeros((n, B.shape[1]))])
    Cz = np.hstack([Cn[np.ix_(idx, xs)], Cp[np.ix_(idx, xs)]])
    cz = np.concatenate([model["drift"][xs], np.zeros(n)])
    return Az, Bz, Cz, model["J"][idx], cz


def observability(model, idx, pos_latency_s=0.0) -> dict:
    """Observability of the strobed state from channels idx, on the full 12-state model (x_k block of the
    augmented system): observable states, unobservable modes (PBH) and whether they are stable (detectable)."""
    idx = S.parse_obs(idx)
    xs = np.arange(12)
    Az, _, Cz, _, _ = augment(model, idx, xs, pos_latency_s)
    n = Az.shape[0]
    Cs = Cz / np.maximum(np.linalg.norm(Cz, axis=1, keepdims=True), 1e-300) if len(idx) else np.zeros((0, n))
    O = np.vstack([Cs @ np.linalg.matrix_power(Az, k) for k in range(n)]) if len(idx) else np.zeros((1, n))
    sv = np.linalg.svd(O, compute_uv=False)
    tol = 1e-7 * max(sv.max(), 1e-300)
    r = int((sv > tol).sum())
    obs_states = []
    for i in range(12):
        e = np.zeros((1, n)); e[0, i] = 1.0
        s2 = np.linalg.svd(np.vstack([O, e]), compute_uv=False)
        if int((s2 > 1e-7 * s2.max()).sum()) == r:
            obs_states.append(STATE_NAMES[i])
    unobs_abs = _unobservable_modes(Az, Cs)
    xd = design_states(idx)
    Azd, _, Czd, _, _ = augment(model, idx, xd, pos_latency_s)
    Csd = Czd / np.maximum(np.linalg.norm(Czd, axis=1, keepdims=True), 1e-300) if len(idx) else np.zeros((0, Azd.shape[0]))
    unobs_d = _unobservable_modes(Azd, Csd)
    return {"obs": S.obs_names(idx), "n_non_imu": S.n_non_imu(idx), "observable_states": obs_states,
            "unobservable_states": [s for s in STATE_NAMES if s not in obs_states],
            "unobservable_mode_abs": unobs_abs, "detectable": all(a < 1 - 1e-6 for a in unobs_abs),
            "design_states": [STATE_NAMES[i] for i in xd], "design_unobservable_mode_abs": unobs_d,
            "design_detectable": all(a < 1 - 1e-6 for a in unobs_d)}


def _unobservable_modes(Az, Cs):
    """|eigenvalues| of the unobservable modes (PBH test), x_{k-1} copy modes (eig 0) ignored."""
    n, out = Az.shape[0], set()
    for lam in np.linalg.eigvals(Az):
        if abs(lam) < 1e-9:
            continue
        smin = np.linalg.svd(np.vstack([lam * np.eye(n) - Az, Cs]), compute_uv=False)[-1]
        if smin < 1e-6 * max(1.0, abs(lam)):
            out.add(round(float(abs(lam)), 4))
    return sorted(out, reverse=True)


def _sigma0(xs):
    X0 = np.diag(STATE_STD[xs] ** 2)
    return np.block([[X0, np.zeros_like(X0)], [np.zeros_like(X0), X0]])


def _rho(M):
    return float(np.abs(np.linalg.eigvals(M)).max())


def sof_cost(Ky, Az, Bz, Cz, Qz, R, Sig, alpha=1.0):
    """(cost, gradient) of u = -Ky Cz z on (alpha Az, alpha Bz); inf when unstable."""
    Acl = alpha * (Az - Bz @ Ky @ Cz)
    if _rho(Acl) >= 1:
        return math.inf, np.zeros_like(Ky)
    P = solve_discrete_lyapunov(Acl.T, Qz + Cz.T @ Ky.T @ R @ Ky @ Cz)
    X = solve_discrete_lyapunov(Acl, Sig)
    G = 2 * (R @ Ky @ Cz - alpha * Bz.T @ P @ Acl) @ X @ Cz.T
    return float(np.trace(P @ Sig)), G


def sof(model, idx, *, pos_latency_s=0.0, weights=None, max_iter=2000, verbose=False):
    """Static output feedback gain Ky (nu x ny) on channels idx. Returns (Ky, info)."""
    idx = S.parse_obs(idx)
    xs = design_states(idx)
    Az, Bz, Cz, _, _ = augment(model, idx, xs, pos_latency_s)
    Q, R = cost(model["B"].shape[1], **(weights or {}))
    n = len(xs)
    Qz = np.zeros((2 * n, 2 * n)); Qz[:n, :n] = Q[np.ix_(xs, xs)]
    Kx = lqr(model["A"][np.ix_(xs, xs)], model["B"][xs], Q=Q[np.ix_(xs, xs)], R=R)
    Kz = np.hstack([Kx, np.zeros_like(Kx)])
    Sig = _sigma0(xs)
    J_lqr, _ = sof_cost(Kz, Az, Bz, np.eye(2 * n), Qz, R, Sig)
    Ky = Kz @ Sig @ Cz.T @ np.linalg.pinv(Cz @ Sig @ Cz.T)                # projected full-state gain
    info = {"rho_projected": _rho(Az - Bz @ Ky @ Cz), "J_lqr": J_lqr}
    shape = Ky.shape

    ysc = np.sqrt(np.maximum(np.diag(Cz @ Sig @ Cz.T), 1e-300))          # precondition: Ky = Kt / ysc

    def run(K0, alpha):
        """Barzilai-Borwein gradient descent with backtracking (monotone) on the preconditioned gain."""
        f = lambda Kt: sof_cost(Kt / ysc, Az, Bz, Cz, Qz, R, Sig, alpha)
        Kt = K0 * ysc
        J, G = f(Kt); G = G / ysc
        step, it = 1e-3 * np.linalg.norm(Kt) / max(np.linalg.norm(G), 1e-300), 0
        for it in range(1, max_iter + 1):
            while True:
                Kn = Kt - step * G
                Jn, Gn = f(Kn)
                if Jn <= J - 1e-4 * step * np.sum(G * G):
                    break
                step *= 0.3
                if step < 1e-14 * max(np.linalg.norm(Kt), 1.0) / max(np.linalg.norm(G), 1e-300):
                    return Kt / ysc, it
            Gn = Gn / ysc
            s, dg = Kn - Kt, Gn - G
            done = (J - Jn) < 1e-10 * J
            Kt, J, G = Kn, Jn, Gn
            if done:
                break
            sy = np.sum(s * dg)
            step = np.sum(s * s) / sy if sy > 0 else step * 2
        return Kt / ysc, it

    alpha, homotopy = 1.0, []
    rho = info["rho_projected"]
    while rho >= 1:
        alpha_new = min(1.0, 0.99 / rho) if not homotopy else min(1.0, 0.99 / rho)
        if homotopy and alpha_new <= homotopy[-1] + 1e-4:
            info.update(ok=False, reason="discount homotopy stalled: no stabilizing SOF found", homotopy=homotopy)
            return Ky, info
        homotopy.append(alpha_new)
        Ky, _ = run(Ky, alpha_new)
        rho = _rho(Az - Bz @ Ky @ Cz)
        if verbose:
            print(f"  homotopy alpha {alpha_new:.4f}: rho {rho:.4f}")
        if len(homotopy) > 60:
            info.update(ok=False, reason="homotopy did not converge", homotopy=homotopy)
            return Ky, info
    Ky, iters = run(Ky, 1.0)
    J, _ = sof_cost(Ky, Az, Bz, Cz, Qz, R, Sig)
    info.update(ok=bool(math.isfinite(J)), J=J, J_ratio=J / J_lqr, rho=_rho(Az - Bz @ Ky @ Cz), homotopy=homotopy,
                iters=int(iters))
    return Ky, info


def with_bias(Az, Bz, Cz, cz, idx, bias_idx):
    """Append constant-bias states b (b+ = b) on channels bias_idx (a subset of idx): y~ = Cz z + b."""
    idx, bias_idx = list(S.parse_obs(idx)), S.parse_obs(bias_idx)
    if not set(bias_idx) <= set(idx):
        raise ValueError(f"bias channels {S.obs_names(bias_idx)} must be among the observed {S.obs_names(idx)}")
    nb, n = len(bias_idx), Az.shape[0]
    Cb = np.zeros((len(idx), nb))
    for j, c in enumerate(bias_idx):
        Cb[idx.index(c), j] = 1.0
    Az2 = np.block([[Az, np.zeros((n, nb))], [np.zeros((nb, n)), np.eye(nb)]])
    return Az2, np.vstack([Bz, np.zeros((nb, Bz.shape[1]))]), np.hstack([Cz, Cb]), np.concatenate([cz, np.zeros(nb)])


def kalman(model, idx, noise: S.SensorNoise, *, beat_s, process_scale=1.0, pos_latency_s=None, bias=None,
           bias_walk=BIAS_WALK, bias_white=BIAS_WHITE):
    """Steady-state Kalman gain on the augmented system. Returns (L, info) with P_post (posterior covariance).
    ``bias``: channels whose (slow) bias becomes a state; their V is then only the white part ``bias_white``
    (x noise_scale, at least 1x) instead of channel_std's bias-as-white-noise approximation."""
    idx = S.parse_obs(idx)
    lat = noise.pos_latency_ms * 1e-3 if pos_latency_s is None else pos_latency_s
    xs = design_states(idx)
    Az, Bz, Cz, _, cz = augment(model, idx, xs, lat)
    n = len(xs)
    vstd = np.maximum(noise.channel_std(beat_s)[idx], V_FLOOR[idx])
    nb = 0
    if bias is not None:
        bidx = S.parse_obs(bias)
        Az, Bz, Cz, cz = with_bias(Az, Bz, Cz, cz, idx, bidx)
        nb = len(bidx)
        sel = np.isin(idx, bidx)
        vstd[sel] = np.maximum(bias_white * max(1.0, noise.scale), V_FLOOR[idx][sel])
    V = np.diag(vstd ** 2)
    W = np.zeros((2 * n + nb, 2 * n + nb)); W[:n, :n] = np.diag((process_scale * PROCESS_STD[xs]) ** 2)
    W[n:2 * n, n:2 * n] = 1e-14 * np.eye(n)
    if nb:
        W[2 * n:, 2 * n:] = bias_walk ** 2 * np.eye(nb)
    P = solve_discrete_are(Az.T, Cz.T, W, V)                             # prior covariance
    L = P @ Cz.T @ np.linalg.inv(Cz @ P @ Cz.T + V)
    Pp = (np.eye(Az.shape[0]) - L @ Cz) @ P
    return L, {"V_std": np.sqrt(np.diag(V)), "P_post_std": np.sqrt(np.clip(np.diag(Pp)[:n], 0, None)), "P_post": Pp,
               "Az": Az, "Bz": Bz, "Cz": Cz, "cz": cz, "bias_post_std": np.sqrt(np.clip(np.diag(Pp)[2 * n:], 0, None))}


def closed_loop_rho(kind, Az, Bz, Cz, gain, L=None, n_bias=0):
    """Spectral radius of the linear closed loop (plant + controller [+ observer]), no input clipping.
    ``n_bias`` constant-bias states (the last ones of Az) are exogenous: their n_bias unit eigenvalues are left out."""
    if kind == "sof":
        return _rho(Az - Bz @ gain @ Cz)
    if n_bias:
        n = Az.shape[0]
        top = np.hstack([Az, -Bz @ np.hstack([gain, np.zeros((gain.shape[0], n - gain.shape[1]))])])
        zh_pred = np.hstack([np.zeros((n, n)), Az - Bz @ np.hstack([gain, np.zeros((gain.shape[0], n - gain.shape[1]))])])
        ev = np.abs(np.linalg.eigvals(np.vstack([top, (np.eye(n) - L @ Cz) @ zh_pred + L @ Cz @ top])))
        ev = np.sort(ev)
        drop = np.argsort(np.abs(ev - 1.0))[:n_bias]
        return float(np.delete(ev, drop).max())
    n = Az.shape[0]
    Kz = np.hstack([gain, np.zeros((gain.shape[0], n - gain.shape[1]))])
    # z+ = Az z - Bz Kz zh ; zh+ = (I - L Cz)(Az zh - Bz Kz zh) + L Cz z+
    top = np.hstack([Az, -Bz @ Kz])
    zh_pred = np.hstack([np.zeros((n, n)), Az - Bz @ Kz])
    bot = (np.eye(n) - L @ Cz) @ zh_pred + L @ Cz @ top
    return _rho(np.vstack([top, bot]))


def design(model, obs, mode, *, noise=None, beat_s=2 * math.pi / 53.6681584313, process_scale=1.0,
           weights=None, dead_reckon=False, bias=None, bias_walk=BIAS_WALK, verbose=False) -> dict:
    """A controller dict (``save_controller`` / ``LinearController``) for channels ``obs`` and ``mode`` in
    state | sof | lqg. ``noise`` sets the LQG V and the mocap latency (default SensorNoise()).
    ``dead_reckon`` (lqg only): integral action on the unmeasured position axes. The controller predicts them
    open loop from the estimate (p += A[p, xs] x_hat + B[p] u + drift[p], no correction: they are unobservable)
    and applies the full 12-state LQR gain, so a constant disturbance no longer leaves a steady velocity. The
    dead-reckoned position is only as good as the velocity estimate: it drifts with its error.
    ``bias`` (lqg only): channels whose slow bias the Kalman filter estimates as extra states (e.g. "att_xy": the
    attitude filter's tilt error from the accelerometer bias; observable only when a position or velocity channel
    is read, otherwise the design is undetectable and raises)."""
    noise = noise if noise is not None else S.SensorNoise()
    idx = S.parse_obs(obs)
    lat = noise.pos_latency_ms * 1e-3
    if mode == "state":
        if not np.array_equal(idx, np.arange(12)):
            raise ValueError("mode 'state' needs obs = state (pos,att,vel,w)")
        return {"kind": "state", "obs": idx, "K": model["K"], "u_max": U_MAX}
    xs = design_states(idx)
    Az, Bz, Cz, Jsel, cz = augment(model, idx, xs, lat)
    Q, R = cost(model["B"].shape[1], **(weights or {}))
    Kx = lqr(model["A"][np.ix_(xs, xs)], model["B"][xs], Q=Q[np.ix_(xs, xs)], R=R)
    info = {"observability": observability(model, idx, lat), "noise": asdict(noise), "mode": mode}
    V_std = np.maximum(noise.channel_std(beat_s)[idx], V_FLOOR[idx])
    ctl = {"kind": mode, "obs": idx, "xs": xs, "J": Jsel, "u_max": U_MAX, "Az": Az, "Bz": Bz, "Cz": Cz, "V_std": V_std}
    if mode == "sof":
        Ky, si = sof(model, idx, pos_latency_s=lat, weights=weights, verbose=verbose)
        info.update({k: v for k, v in si.items()})
        ctl["Ky"] = Ky
        info["rho_closed_loop"] = closed_loop_rho("sof", Az, Bz, Cz, Ky)
    elif mode == "lqg":
        L, ki = kalman(model, idx, noise, beat_s=beat_s, process_scale=process_scale, pos_latency_s=lat, bias=bias,
                       bias_walk=bias_walk)
        if bias is not None:                              # the observer runs on the bias-augmented system
            ctl.update(Az=ki["Az"], Bz=ki["Bz"], Cz=ki["Cz"], cz=ki["cz"])
            info.update(bias=S.obs_names(S.parse_obs(bias)), bias_post_std=ki["bias_post_std"].tolist())
        ctl.update(K=Kx, L=L, cz=ctl.get("cz", cz), P_post_std=ki["P_post_std"])
        pr = np.setdiff1d(np.arange(3), xs)
        if dead_reckon and pr.size:
            Kf = lqr(model["A"], model["B"], Q=Q, R=R)
            ctl.update(K=Kf[:, xs], K_dr=Kf[:, pr], dr=pr, A_dr=model["A"][np.ix_(pr, xs)], B_dr=model["B"][pr],
                       d_dr=model["drift"][pr])
            info["dead_reckon"] = S.obs_names(pr)
        info.update(V_std=ki["V_std"].tolist(), P_post_std=ki["P_post_std"].tolist(),
                    rho_closed_loop=closed_loop_rho("lqg", ctl["Az"], ctl["Bz"], ctl["Cz"], Kx, L,
                                                    n_bias=len(S.parse_obs(bias)) if bias is not None else 0),
                    rho_lqr=_rho(model["A"][np.ix_(xs, xs)] - model["B"][xs] @ Kx), process_scale=process_scale)
    else:
        raise ValueError(f"unknown mode {mode!r}: state | sof | lqg")
    ctl["info"] = info
    return ctl


def save_controller(path, ctl) -> None:
    arrays = {k: np.asarray(v) for k, v in ctl.items() if k != "info"}
    np.savez(path, info=np.array(json.dumps(ctl.get("info", {}), default=float)), **arrays)


def load_controller(path) -> dict:
    z = np.load(path)
    ctl = {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files if k != "info"}
    ctl["info"] = json.loads(str(z["info"])) if "info" in z.files else {}
    if "W0" in z.files:                                  # a distilled MLP replacing the linear gain
        n = sum(1 for k in z.files if k.startswith("W"))
        ctl["mlp"] = {"layers": [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)], "obs_scale": z["obs_scale"]}
    return ctl


def is_controller(path) -> bool:
    return "kind" in np.load(path).files


class LinearController:
    """Per-wingbeat runtime of a controller dict: ``policy(y) -> correction to theta`` (25), with reset().
    ``y`` = the sensor channel vector (18) or the true strobed state (12, only for true-state channels)."""

    def __init__(self, ctl: dict):
        self.ctl = ctl
        self.kind = str(ctl["kind"])
        self.obs = np.asarray(ctl["obs"], int)
        self.u_max = float(ctl["u_max"])
        self.mlp = ctl.get("mlp")
        self.nu = (ctl["K"] if "K" in ctl else ctl["Ky"]).shape[0]
        self.reset()

    def reset(self):
        self.z = None
        self.p = self.xh_prev = None                     # dead-reckoned position axes (ctl["dr"])
        self.u_prev = np.zeros(self.nu)
        self.log = []

    def _act(self, v):
        if self.mlp is None:
            return None
        x = np.asarray(v) * self.mlp["obs_scale"]
        *hidden, (W, b) = self.mlp["layers"]
        for Wh, bh in hidden:
            x = np.tanh(Wh @ x + bh)
        return self.u_max * np.tanh(W @ x + b)

    def __call__(self, y):
        y = np.asarray(y)
        if len(y) != S.N_CH and self.obs.max() >= len(y):
            raise ValueError(f"policy reads channels {S.obs_names(self.obs)}: fly it with a sensor model")
        ys = y[self.obs]
        c = self.ctl
        if self.kind == "state":
            u = np.clip(-c["K"] @ ys, -self.u_max, self.u_max)
        elif self.kind == "sof":
            yt = ys - c["J"] @ self.u_prev
            u = self._act(yt)
            u = np.clip(-c["Ky"] @ yt, -self.u_max, self.u_max) if u is None else u
        else:
            yt = ys - c["J"] @ self.u_prev
            zp = np.zeros(c["Az"].shape[0]) if self.z is None else c["Az"] @ self.z + c["Bz"] @ self.u_prev + c["cz"]
            self.z = zp + c["L"] @ (yt - c["Cz"] @ zp)
            xh = self.z[:len(c["xs"])]
            v, K = xh, c["K"]
            if "dr" in c:                                # dead reckoning: predict the unmeasured position axes
                self.p = np.zeros(len(c["dr"])) if self.xh_prev is None else \
                    self.p + c["A_dr"] @ self.xh_prev + c["B_dr"] @ self.u_prev + c["d_dr"]
                v, K = np.concatenate([xh, self.p]), np.hstack([c["K"], c["K_dr"]])
            u = self._act(v)
            u = np.clip(-K @ v, -self.u_max, self.u_max) if u is None else u
            self.xh_prev = xh
            self.log.append(xh.copy())
        self.u_prev = u
        return np.concatenate([[0.0], u])


def noise_from_args(a) -> S.SensorNoise:
    return S.SensorNoise(scale=a.noise_scale if a.noise_scale is not None else 1.0, pos_scale=a.pos_noise_scale,
                         pos_latency_ms=a.pos_latency_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lqr", type=Path, required=True, help="npz from python -m hover.lqr (with the sensor model)")
    ap.add_argument("--obs", default="state", help="channel subset (hover.sensors.parse_obs)")
    ap.add_argument("--mode", default="lqg", choices=("state", "sof", "lqg"))
    ap.add_argument("--noise-scale", type=float, default=1.0, help="LQG design noise (SensorNoise.scale)")
    ap.add_argument("--pos-noise-scale", type=float, default=1.0)
    ap.add_argument("--pos-latency-ms", type=float, default=0.0)
    ap.add_argument("--process-scale", type=float, default=1.0, help="LQG process noise x PROCESS_STD")
    ap.add_argument("--dead-reckon", action="store_true", help="lqg: integral action on unmeasured position axes")
    ap.add_argument("--bias", default=None, help="lqg: channels whose slow bias is estimated as a state (e.g. att_xy)")
    ap.add_argument("--beat-s", type=float, default=2 * math.pi / 53.6681584313, help="wingbeat [s] (the design's)")
    ap.add_argument("--report", default=None, help="';'-separated subsets: print observability only")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    model = load_model(a.lqr)
    lat = a.pos_latency_ms * 1e-3
    if a.report:
        for spec in a.report.split(";"):
            r = observability(model, spec.strip(), lat)
            print(f"{spec.strip():24s} non-IMU {r['n_non_imu']}: unobservable {r['unobservable_states'] or '-'} | "
                  f"unobservable |modes| {r['unobservable_mode_abs'] or '-'} | detectable {r['detectable']} | "
                  f"design states detectable {r['design_detectable']} {r['design_unobservable_mode_abs'] or ''}")
            if r["design_detectable"] and r["obs"]:
                _, ki = kalman(model, spec.strip(), noise_from_args(a), beat_s=a.beat_s, process_scale=a.process_scale)
                ps = dict(zip(r["design_states"], ki["P_post_std"]))
                g = lambda pre, f: [round(float(f * ps[k]), 2) for k in ps if k.startswith(pre)]
                print(f"{'':24s} LQG posterior std: att {g('att', 180 / math.pi)} deg, vel {g('vel', 1e3)} mm/s, "
                      f"w {g('w_', 1.0)} rad/s, pos {g('pos', 1e3)} mm")
        return
    ctl = design(model, a.obs, a.mode, noise=noise_from_args(a), beat_s=a.beat_s, process_scale=a.process_scale,
                 dead_reckon=a.dead_reckon, bias=a.bias, verbose=True)
    info = ctl["info"]
    print(json.dumps({k: v for k, v in info.items() if k not in ("noise", "P_post_std", "V_std")}, indent=1, default=float))
    if a.out:
        save_controller(a.out, ctl)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
