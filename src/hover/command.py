"""Command-following flight of the free bird (host side, numpy/scipy): HOVER / FWD / REV / UP / DOWN.

The bird is flown only through its four cranks: once per wingbeat a correction u (the 24 harmonic depths) is
added to the trimmed open loop, as hover.launch.fly / hover.mjx_flight.fly do. The controller is a one-trim
linear servo on the per-wingbeat hover model (A, B, drift) of hover.lqr / hover.mjx_flight.linearize:

* feedforward tables (:func:`steady_state_tables`): per command axis and speed, the steady attitude and input
  that hold x_{k+1} = x_k on the att / vel / w rows at vel = speed * dir (a tilt-bounded min-inf LP, HiGHS),
  refined from one batched pass of constant-command flights by :func:`ilc_update` (ILC);
* references (:func:`build_refs`): the commanded velocity is rate-limited into v_ref, p_ref integrates it, and
  x_ref = [p_ref, att_ref(v_ref), vel_ref(v_ref), w_ref(v_ref)], u_ff = u_ff(v_ref) (tables looked up per axis and
  superposed, so mixed velocities during transitions are covered; in flight att / w / u_ff are looked up one
  wingbeat ahead, ``ff_lead``, since the attitude lags a velocity change); optional columns 13 / 14 carry the
  integral handoff (:func:`integral_keep`: a deceleration bleeds the integral with the reference speed) and gate
  (:func:`integral_gate`: no integration for 8 wingbeats after a stop);
* feedback (:func:`servo_gains`): LQR on the model augmented with the integral of the position error
  (zeta_{k+1} = keep_k zeta_k + gate_k (p_k - p_ref_k), conditional integration while no input saturates):
        u = clip(u_ff + dither - Ka[:, :12] (s - x_ref) - Ka[:, 12:] zeta, +-u_max)
  :func:`servo_gains_vel` is the no-mocap variant (integral of the velocity error, no position feedback). The ILC
  passes fly the learning gain (q_int 100: its integral settles inside the ILC window); the policy flies a softer
  one (opt/hover_cmd.py --fly-q-int / --fly-q-vel).
* IMU-only flight (no velocity measurement): :func:`velocity_ilc` moves each cell's trim by the linear
  :func:`trim_sensitivity` times the TRUE mean-motion velocity error measured in the learning flights.
* :func:`allocate` (opt-in, --alloc-iters): the box projection in the servo's Q-function metric; it flew worse than
  clip (opt/runs_cmd/NOTES.md 14: the hover model's null space is not null at speed).

:class:`CommandPolicy` flies the law on CPU (hover.launch.fly, unchanged); hover.mjx_command flies it batched
on the device; :func:`segment_metrics` scores random command sequences (:func:`random_schedules`) from the TRUE
strobed states. Frames: world; forward e_f = the release attitude's body +x projected on the horizontal (the
repo's convention: +x is "forward", hover.evaluate.VERIFY_PUSHES); yaw is held at 0. The attitude error is plain
rotation-vector subtraction (fine at <= 15 deg).

    .venv/bin/python opt/hover_cmd.py learn --out opt/runs_cmd/<run>      # on the GPU box (gpu-box skill)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

from hover.lqr import cost

CMDS = ("HOVER", "FWD", "REV", "UP", "DOWN")
AXES = ("FWD", "REV", "UP", "DOWN")            # table axes (index 0..3); CMDS index = axis index + 1
U_MAX = 0.15
ZETA_MAX = 1.0                                  # integral clip [m * wingbeat]
V_OP_H = 0.3
V_OP_Z = 0.3
SPEEDS_H = np.round(np.arange(1, 16) * 0.05, 6)     # 0.05 .. 0.75 m/s
SPEEDS_Z = np.round(np.arange(1, 16) * 0.06, 6)     # 0.06 .. 0.90 m/s (0.30 on the grid: the G7 operating speed)
TILT_MAPS = (5.0, 15.0)


# ---------------------------------------------------------------------------------------------- frames, commands

def quat_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def forward_axis(q_ref) -> np.ndarray:
    """Body +x of the release attitude projected on the horizontal plane, normalised."""
    f = quat_mat(q_ref)[:, 0].copy()
    f[2] = 0.0
    return f / np.linalg.norm(f)


def command_dirs(e_f) -> np.ndarray:
    """(5, 3) unit directions of CMDS (HOVER = 0)."""
    ez = np.array([0.0, 0.0, 1.0])
    return np.stack([np.zeros(3), e_f, -e_f, ez, -ez])


def axis_speeds(speeds_h=SPEEDS_H, speeds_z=SPEEDS_Z) -> np.ndarray:
    """(4, 16) speed grid per table axis, entry 0 = hover (s = 0)."""
    h, z = np.r_[0.0, speeds_h], np.r_[0.0, speeds_z]
    return np.stack([h, h, z, z])


# ---------------------------------------------------------------------------------------------- steady states (LP)

def steady_state(A, B, drift, vel, tilt_max, *, use_drift=True):
    """min t + 1e-3 |u|_1 s.t. |u_i| <= t, |roll|, |pitch| <= tilt_max, yaw = 0, and the att / vel / w rows of
    x_{k+1} = x_k at x = [*, att, vel, 0]. Returns (att (3), u (nu), t, residual) or None if infeasible."""
    nu = B.shape[1]
    M = (A - np.eye(12))[3:12]
    rhs = -(M[:, 6:9] @ np.asarray(vel, float) + (drift[3:12] if use_drift else 0.0))
    n = 3 + 2 * nu + 1                                    # att, u, |u| bounds a, t
    Aeq = np.zeros((9, n)); Aeq[:, 0:3] = M[:, 3:6]; Aeq[:, 3:3 + nu] = B[3:12]
    I = np.eye(nu)
    Aub = np.zeros((3 * nu, n))
    Aub[:nu, 3:3 + nu] = I; Aub[:nu, 3 + nu:3 + 2 * nu] = -I            # u - a <= 0
    Aub[nu:2 * nu, 3:3 + nu] = -I; Aub[nu:2 * nu, 3 + nu:3 + 2 * nu] = -I   # -u - a <= 0
    Aub[2 * nu:, 3 + nu:3 + 2 * nu] = I; Aub[2 * nu:, -1] = -1.0          # a - t <= 0
    c = np.zeros(n); c[3 + nu:3 + 2 * nu] = 1e-3; c[-1] = 1.0
    tm = math.radians(tilt_max)
    bounds = [(-tm, tm), (-tm, tm), (0.0, 0.0)] + [(None, None)] * nu + [(0, None)] * nu + [(0, None)]
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(3 * nu), A_eq=Aeq, b_eq=rhs, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    z = r.x
    return z[0:3], z[3:3 + nu], float(z[-1]), float(np.abs(Aeq @ z - rhs).max())


def lp_speed_at(A, B, drift, d, tilt_max, level, *, use_drift=True, s_hi=5.0, tol=1e-3):
    """Largest speed s along d with LP optimum t(s) <= level (t is convex in s: bisection)."""
    def ok(s):
        r = steady_state(A, B, drift, s * d, tilt_max, use_drift=use_drift)
        return r is not None and r[2] <= level
    if not ok(0.0):
        return 0.0
    if ok(s_hi):
        return s_hi
    lo, hi = 0.0, s_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def steady_state_tables(A, B, drift, e_f, tilt_max_deg, u_max=U_MAX, speeds_h=SPEEDS_H, speeds_z=SPEEDS_Z,
                        use_drift=None, log=print) -> dict:
    """Feedforward tables for one tilt map: speeds (4, 16), att (4, 16, 3), u (4, 16, nu), dv (4, 16, 3) (strobed
    minus mean-motion velocity, 0 until ILC), w (4, 16, 3) (0 until ILC), t (4, 16) LP optimum, feasible (4, 16),
    and the LP envelope (speed where t reaches u_max / 2 and u_max) per axis. Entry 0 of every axis = hover.
    ``use_drift`` None: with the drift, unless |u_ff(0)|_inf > 0.02 (then relative to hover, noted in 'note')."""
    dirs = command_dirs(e_f)[1:]
    sp = axis_speeds(speeds_h, speeds_z)
    nu = B.shape[1]
    note = ""
    if use_drift is None:
        r0 = steady_state(A, B, drift, np.zeros(3), tilt_max_deg)
        use_drift = r0 is not None and np.abs(r0[1]).max() <= 0.02
        if not use_drift:
            note = (f"|u_ff(0)|_inf = {np.abs(r0[1]).max() if r0 else float('nan'):.4f} > 0.02 with the drift: "
                    "tables solved without it (relative to hover)")
    att = np.zeros((4, sp.shape[1], 3)); u = np.zeros((4, sp.shape[1], nu)); t = np.full((4, sp.shape[1]), np.nan)
    feas = np.zeros((4, sp.shape[1]), bool); res = np.zeros((4, sp.shape[1]))
    for a in range(4):
        last = None
        for j, s in enumerate(sp[a]):
            r = steady_state(A, B, drift, s * dirs[a], tilt_max_deg, use_drift=use_drift)
            if r is not None:
                last = r
                feas[a, j] = True
                t[a, j], res[a, j] = r[2], r[3]
            if last is not None:                          # infeasible: clamp to the last feasible entry
                att[a, j], u[a, j] = last[0], last[1]
    env = {}
    for a, name in enumerate(AXES):
        env[name] = {"half_budget": lp_speed_at(A, B, drift, dirs[a], tilt_max_deg, u_max / 2, use_drift=use_drift),
                     "full_budget": lp_speed_at(A, B, drift, dirs[a], tilt_max_deg, u_max, use_drift=use_drift)}
    tab = {"speeds": sp, "att": att, "u": u, "dv": np.zeros((4, sp.shape[1], 3)), "w": np.zeros((4, sp.shape[1], 3)),
           "t": t, "feasible": feas, "residual": res, "tilt_max_deg": float(tilt_max_deg), "use_drift": bool(use_drift),
           "lp_envelope": env, "note": note}
    if log:
        log(f"tables tilt {tilt_max_deg:g} deg: drift {'in' if use_drift else 'OUT'}, |u_ff(0)| {np.abs(u[0, 0]).max():.4f}, "
            f"LP residual max {np.nanmax(res):.2e}, infeasible cells {int((~feas).sum())} | LP envelope (t = u_max/2, u_max): "
            + ", ".join(f"{k} {v['half_budget']:.2f}/{v['full_budget']:.2f}" for k, v in env.items()) + " m/s"
            + (f" | {note}" if note else ""))
    return tab


def _interp(grid, vals, x):
    """Piecewise-linear interpolation of vals (G, ...) over grid (G,) at x (M,), clamped at both ends."""
    x = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, x, "right") - 1, 0, len(grid) - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    w = w.reshape(w.shape + (1,) * (vals.ndim - 1))
    return vals[i] * (1 - w) + vals[i + 1] * w


def lookup(tab, e_f, v):
    """(att_ref, vel_ref, w_ref, u_ff) at reference velocities v (M, 3): the FWD/REV entry at |v . e_f| plus the
    UP/DOWN entry at |v_z| minus the hover entry (superposition)."""
    v = np.asarray(v, float)
    vf, vz = v @ e_f, v[:, 2]
    af = np.where(vf >= 0, 0, 1); az = np.where(vz >= 0, 2, 3)
    out = []
    for key in ("att", "dv", "w", "u"):
        T = tab[key]
        hf = np.stack([_interp(tab["speeds"][a], T[a], np.abs(vf)) for a in (0, 1)])       # (2, M, d)
        hz = np.stack([_interp(tab["speeds"][a], T[a], np.abs(vz)) for a in (2, 3)])
        idx = np.arange(len(v))
        out.append(hf[af, idx] + hz[az - 2, idx] - T[0, 0])
    att, dv, w, u = out
    return att, v + dv, w, u


# ---------------------------------------------------------------------------------------------- gains

def _dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def servo_gains(A, B, q_int=1 / 0.1 ** 2, weights=None):
    """Ka (nu, 15) of the LQR on [x; zeta], zeta_{k+1} = zeta_k + p_k (- p_ref_k); hover.lqr.cost weights plus
    q_int on zeta. Returns (Ka, rho of the augmented closed loop)."""
    nu = B.shape[1]
    Aa = np.zeros((15, 15)); Aa[:12, :12] = A; Aa[12:, :3] = np.eye(3); Aa[12:, 12:] = np.eye(3)
    Ba = np.vstack([B, np.zeros((3, nu))])
    Q, R = cost(nu, **(weights or {}))
    Qa = np.zeros((15, 15)); Qa[:12, :12] = Q; Qa[12:, 12:] = q_int * np.eye(3)
    Ka = _dlqr(Aa, Ba, Qa, R)
    rho = float(np.abs(np.linalg.eigvals(Aa - Ba @ Ka)).max())
    assert rho < 1, rho
    return Ka, rho


def servo_hessian(A, B, q_int=1 / 0.1 ** 2, weights=None):
    """The Hessian H = R + Ba' P Ba of the servo LQR's Q-function in u (P = the augmented DARE solution of
    :func:`servo_gains`), normalised to max eigenvalue 1. Projecting the unconstrained servo command onto the input
    box in this metric (:func:`allocate`) is the one-step optimal constrained LQR (one-step MPC with the LQR's
    cost-to-go as terminal cost): the 24 depths have a 12-dim null space in B, so the saturated channels' share
    moves onto the free ones instead of being dropped (clip)."""
    nu = B.shape[1]
    Aa = np.zeros((15, 15)); Aa[:12, :12] = A; Aa[12:, :3] = np.eye(3); Aa[12:, 12:] = np.eye(3)
    Ba = np.vstack([B, np.zeros((3, nu))])
    Q, R = cost(nu, **(weights or {}))
    Qa = np.zeros((15, 15)); Qa[:12, :12] = Q; Qa[12:, 12:] = q_int * np.eye(3)
    P = solve_discrete_are(Aa, Ba, Qa, R)
    H = R + Ba.T @ P @ Ba
    H = 0.5 * (H + H.T)
    return H / np.linalg.eigvalsh(H).max()


def fista_betas(iters):
    """Nesterov momentum coefficients of :func:`allocate` (fixed sequence, so host and device run the same ops)."""
    t, out = 1.0, []
    for _ in range(iters):
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        out.append((t - 1) / tn)
        t = tn
    return np.array(out)


def allocate(ud, H, u_max, iters):
    """argmin_{|u|_inf <= u_max} (u - ud)' H (u - ud) by FISTA (H normalised: step 1), from clip(ud); ud (..., nu)
    batched on the leading axes. Returns u."""
    u = np.clip(ud, -u_max, u_max)
    y = u
    for b in fista_betas(iters):
        un = np.clip(y - (y - ud) @ H, -u_max, u_max)
        y = un + b * (un - u)
        u = un
    return u


def servo_gains_vel(A, B, T, weights=None):
    """No-mocap servo: z = [att, vel, w] (9) + zeta_v (3), zeta_v += (vel - vel_ref) T (dead-reckoned position
    error); the position weight moves onto zeta_v. Returned in the (nu, 15) layout of :func:`servo_gains` (zero
    position columns; columns 12:15 act on zeta_v). Returns (Ka, rho)."""
    nu = B.shape[1]
    A9, B9 = A[3:12, 3:12], B[3:12]
    Aa = np.zeros((12, 12)); Aa[:9, :9] = A9; Aa[9:, 3:6] = T * np.eye(3); Aa[9:, 9:] = np.eye(3)
    Ba = np.vstack([B9, np.zeros((3, nu))])
    Q, R = cost(nu, **(weights or {}))
    Qa = np.zeros((12, 12)); Qa[:9, :9] = Q[3:12, 3:12]; Qa[9:, 9:] = Q[0, 0] * np.eye(3)
    Kv = _dlqr(Aa, Ba, Qa, R)
    rho = float(np.abs(np.linalg.eigvals(Aa - Ba @ Kv)).max())
    assert rho < 1, rho
    Ka = np.zeros((nu, 15)); Ka[:, 3:12] = Kv[:, :9]; Ka[:, 12:] = Kv[:, 9:]
    return Ka, rho


def hover_ka(K):
    """The plain hover LQR K (nu, 12) in the servo layout (baseline arm)."""
    return np.hstack([K, np.zeros((K.shape[0], 3))])


# ---------------------------------------------------------------------------------------------- schedules, references

def random_schedules(rng, n, n_beats, e_f, seg=(30, 50), speeds=(V_OP_H, V_OP_Z), mag_mode="fixed", vmax=None):
    """Random command sequences: segments of uniform length in [seg[0], seg[1]], commands uniform over CMDS with
    no immediate repeat (the first one random too). ``mag_mode`` 'fixed' = speeds (horizontal, vertical[, DOWN]);
    'uniform' = per segment U[0.1, vmax[axis]] (vmax: dict AXES -> m/s). Returns dict with v_cmd (n, n_beats, 3),
    cmd (n, n_beats) and seg (n, S, 4) = [start, end, cmd, speed] rows (-1 padded)."""
    dirs = command_dirs(e_f)
    v_cmd = np.zeros((n, n_beats, 3)); cmd = np.zeros((n, n_beats), int)
    max_s = n_beats // seg[0] + 2
    segs = -np.ones((n, max_s, 4))
    for i in range(n):
        k, j, prev = 0, 0, -1
        while k < n_beats:
            L = int(rng.integers(seg[0], seg[1] + 1))
            c = int(rng.integers(0, 5))
            while c == prev:
                c = int(rng.integers(0, 5))
            if c == 0:
                s = 0.0
            elif mag_mode == "fixed":
                s = speeds[0] if c in (1, 2) else (speeds[2] if c == 4 and len(speeds) > 2 else speeds[1])
            else:
                s = float(rng.uniform(0.1, max(0.1, vmax[CMDS[c]])))
            e = min(k + L, n_beats)
            v_cmd[i, k:e] = s * dirs[c]; cmd[i, k:e] = c
            segs[i, j] = (k, k + L, c, s)
            k, j, prev = k + L, j + 1, c
    return {"v_cmd": v_cmd, "cmd": cmd, "seg": segs}


def integral_keep(v):
    """Integral handoff factors for reference velocities v (n, N+1, 3): keep_k = clip(v_{k+1} . v_k / |v_k|^2, 0, 1)
    (1 at rest and for the last entry). The servo scales its position integral by keep_k at each update, so a
    deceleration to hover bleeds the integral in step with the reference speed: the integral held the moving
    trim's table error, which the stop removes (without it FWD -> HOVER overshoots ~8 cm; NOTES 16-17)."""
    v = np.asarray(v, float)
    keep = np.ones(v.shape[:-1])
    n2 = (v[:, :-1] ** 2).sum(-1)
    dot = (v[:, 1:] * v[:, :-1]).sum(-1)
    keep[:, :-1] = np.where(n2 > 1e-18, np.clip(dot / np.maximum(n2, 1e-18), 0.0, 1.0), 1.0)
    return keep


def integral_gate(v, v_cmd, hold):
    """Integral gate (n, N+1) for reference velocities v (n, N+1, 3) and per-beat commands v_cmd (n, N, 3): 0 while
    the command is HOVER and the reference is still decelerating or stopped less than ``hold`` wingbeats ago
    (the stop transient is left to the proportional loop: integrating it wound up an undershoot), else 1."""
    v = np.asarray(v, float)
    n, N1, _ = v.shape
    g = np.ones((n, N1))
    ramp = np.zeros((n, N1), bool)
    ramp[:, :-1] = np.abs(v[:, 1:] - v[:, :-1]).max(-1) > 1e-12
    since = np.full(n, 10 ** 6)
    for k in range(N1 - 1):
        since = np.where(ramp[:, k], 0, since + 1)
        hov = np.abs(v_cmd[:, k]).max(-1) == 0
        g[:, k] = np.where(hov & (since <= hold), 0.0, 1.0)
    return g


def build_refs(v_cmd, tab, e_f, T, rate, descend_after_stop=0.0, descend_delay=0, zeta_handoff=False, stop_rate=1.0,
               zeta_gate=None, ff_lead=0, climb_after_stop=0.0):
    """References for per-beat commands v_cmd (n, N, 3): v_ref rate-limited per world axis by ``rate`` ((3,) or
    (n, 3) m/s per beat), p_ref_{k+1} = p_ref_k + v_ref_k T. ``descend_after_stop`` > 0: a descent does not start
    (v_ref,z is not lowered) while the horizontal v_ref exceeds this speed, so FWD -> DOWN decelerates first, and for
    ``descend_delay`` more wingbeats after that (the attitude settles from the forward trim first).
    Returns x_ref (n, N+1, 12), u_ff (n, N+1, nu), v_ref (n, N+1, 3) (entry N is the state after the last beat,
    for the metrics). ``zeta_handoff``: x_ref gets a 13th column, the :func:`integral_keep` factors (ones when
    False and ``zeta_gate`` is set). ``zeta_gate`` (hold wingbeats): a 14th column, :func:`integral_gate`.
    ``stop_rate``: the rate x this while the command is HOVER (< 1: slower stops). ``ff_lead``: the att / w / u_ff
    references are looked up at v_ref ``ff_lead`` wingbeats ahead (the attitude lags a velocity change).
    ``climb_after_stop`` > 0: likewise a climb does not start while the horizontal v_ref exceeds this speed (no
    delay): FWD -> UP is the most stressed transition, and the estimate-fed servo loses it (NOTES 20)."""
    n, N, _ = v_cmd.shape
    rate = np.broadcast_to(np.asarray(rate, float), (n, 3))
    v = np.zeros((n, N + 1, 3)); p = np.zeros((n, N + 1, 3))
    since = np.full(n, 10 ** 6)                              # wingbeats since the horizontal v_ref fell below the limit
    stop = np.abs(v_cmd).max(-1, keepdims=True) == 0                    # (n, N, 1): the command is HOVER
    for k in range(N):
        rk = np.where(stop[:, k], stop_rate * rate, rate)
        step = np.clip(v_cmd[:, k] - v[:, k], -rk, rk)
        if descend_after_stop > 0:
            moving = np.linalg.norm(v[:, k, :2], axis=1) > descend_after_stop
            since = np.where(moving, 0, since + 1)
            hold = moving | (since <= descend_delay)
            step[:, 2] = np.where(hold & (v[:, k, 2] + step[:, 2] < 0), np.maximum(step[:, 2], 0.0), step[:, 2])
        if climb_after_stop > 0:
            climb_hold = np.linalg.norm(v[:, k, :2], axis=1) > climb_after_stop
            step[:, 2] = np.where(climb_hold & (v[:, k, 2] + step[:, 2] > 0), np.minimum(step[:, 2], 0.0), step[:, 2])
        v[:, k + 1] = v[:, k] + step
        p[:, k + 1] = p[:, k] + v[:, k] * T
    # the reference velocity at beat k is v[:, k]; the last command also sets v[:, N] (used only by the metrics)
    att, vel, w, u = lookup(tab, e_f, v.reshape(-1, 3))
    if ff_lead:                  # attitude / rate / feedforward references lead the velocity reference by ff_lead beats
        att, _, w, u = lookup(tab, e_f, v[:, np.minimum(np.arange(N + 1) + ff_lead, N)].reshape(-1, 3))
    x_ref = np.concatenate([p, att.reshape(n, N + 1, 3), vel.reshape(n, N + 1, 3), w.reshape(n, N + 1, 3)]
                           + ([integral_keep(v)[..., None] if zeta_handoff else np.ones((n, N + 1, 1))]
                              if (zeta_handoff or zeta_gate is not None) else [])
                           + ([integral_gate(v, v_cmd, zeta_gate)[..., None]] if zeta_gate is not None else []), -1)
    return x_ref, u.reshape(n, N + 1, -1), v


def baseline_refs(v_ref, T):
    """Baseline arm references: x_ref = [p_ref, 0, 0, 0], u_ff = 0 (p_ref integrates the same v_ref)."""
    n, N1, _ = v_ref.shape
    p = np.concatenate([np.zeros((n, 1, 3)), np.cumsum(v_ref[:, :-1] * T, 1)], 1)
    return np.concatenate([p, np.zeros((n, N1, 9))], -1)


# ---------------------------------------------------------------------------------------------- control law

ALLOC_FREE = 0.01      # integrate while the delivered feedback misses the demanded one by <= 10% (H-norm)


def servo_law(s, zeta, x_ref, u_ff, dither, Ka, u_max=U_MAX, zeta_max=ZETA_MAX, cp=1.0, cv=0.0, T=0.0, H=None, iters=0):
    """One wingbeat of the servo (numpy; hover.mjx_command.law is the same on device). Returns (u, zeta').
    ``iters`` 0: u = clip(raw), the integral pauses while any input saturates. ``iters`` > 0: u = allocate(raw)
    in the metric H (:func:`servo_hessian`), the integral pauses while the delivered feedback misses the demanded
    one by more than 10% in that metric. A 13 / 14-wide ``x_ref`` carries the integral handoff factor
    (:func:`integral_keep`) / gate (:func:`integral_gate`): zeta' = keep zeta + gate free dz."""
    e = s - x_ref[:12]
    keep = x_ref[12] if len(x_ref) > 12 else 1.0
    gate = x_ref[13] if len(x_ref) > 13 else 1.0
    raw = u_ff + dither - Ka[:, :12] @ e - Ka[:, 12:] @ zeta
    if iters:
        u = allocate(raw, H, u_max, iters)
        d, fb = u - raw, raw - u_ff - dither
        free = 1.0 if d @ H @ d <= ALLOC_FREE * (fb @ H @ fb) else 0.0
    else:
        u = np.clip(raw, -u_max, u_max)
        free = 0.0 if np.any(np.abs(raw) >= u_max) else 1.0
    dz = cp * e[:3] + cv * T * e[6:9]
    return u, np.clip(keep * zeta + gate * free * dz, -zeta_max, zeta_max)


class CommandPolicy:
    """The servo on CPU for hover.launch.fly (true state): policy(s) -> np.r_[0, u], one call per wingbeat."""

    def __init__(self, Ka, x_ref, u_ff, dither=None, *, u_max=U_MAX, zeta_max=ZETA_MAX, cp=1.0, cv=0.0, T=0.0, H=None,
                 iters=0):
        self.Ka, self.x_ref, self.u_ff = np.asarray(Ka, float), np.asarray(x_ref, float), np.asarray(u_ff, float)
        self.dither = np.zeros_like(self.u_ff) if dither is None else np.asarray(dither, float)
        self.u_max, self.zeta_max, self.cp, self.cv, self.T = u_max, zeta_max, cp, cv, T
        self.H, self.iters = (None if H is None else np.asarray(H, float)), int(iters)
        self.reset()

    def reset(self):
        self.k = 0
        self.zeta = np.zeros(3)
        self.U = []

    def __call__(self, s):
        k = self.k
        u, self.zeta = servo_law(np.asarray(s, float), self.zeta, self.x_ref[k], self.u_ff[k], self.dither[k], self.Ka,
                                 self.u_max, self.zeta_max, self.cp, self.cv, self.T, self.H, self.iters)
        self.k += 1
        self.U.append(u)
        return np.concatenate([[0.0], u])


def trim_consistent_cz(ctl, x_ref, u_ff):
    """Per-beat observer offset (N, nz) that makes the reference trajectory the observer model's own trajectory:
    cz_k = x_ref_k - Az x_ref_{k-1} - Bz u_ff_{k-1} on the design-state rows (zero on the x_{k-1} copy and bias
    rows; row 0 = the constant hover cz, unused: the first beat has no prediction). The plain LQG predicts with the
    hover model + constant hover drift, so at an ILC-learned moving trim its Kalman filter explains the steady
    model mismatch through its slowest states (bias / velocity) as a fake tilt / velocity error (NOTES 19)."""
    xs = np.asarray(ctl["xs"], int)
    n = len(xs)
    Az, Bz, cz = np.asarray(ctl["Az"]), np.asarray(ctl["Bz"]), np.asarray(ctl["cz"])
    xr = np.asarray(x_ref, float)[:, :12][:, xs]
    uf = np.asarray(u_ff, float)
    out = np.tile(cz, (len(xr), 1))
    out[1:, :n] = xr[1:] - xr[:-1] @ Az[:n, :n].T - uf[:-1] @ Bz[:n].T
    out[1:, n:] = 0.0
    return out


class CmdLinearController:
    """hover.reduced.LinearController's LQG observer (copied, not modified) + the servo on the estimate: for
    hover.launch.fly(..., sensor=hover.sensors.Sensors). ``ctl`` = a hover.reduced lqg dict. ``cz_seq`` (N, nz):
    per-beat observer offset (:func:`trim_consistent_cz`) in place of the constant ``ctl["cz"]``."""

    def __init__(self, ctl, Ka, x_ref, u_ff, *, u_max=U_MAX, zeta_max=ZETA_MAX, cp=1.0, cv=0.0, T=0.0, H=None, iters=0,
                 cz_seq=None):
        if str(ctl["kind"]) != "lqg" or "dr" in ctl:
            raise ValueError("CmdLinearController takes a plain lqg controller (no dead reckoning)")
        self.cz_seq = None if cz_seq is None else np.asarray(cz_seq, float)
        self.ctl, self.obs, self.xs = ctl, np.asarray(ctl["obs"], int), np.asarray(ctl["xs"], int)
        self.Ka, self.x_ref, self.u_ff = np.asarray(Ka, float), np.asarray(x_ref, float), np.asarray(u_ff, float)
        self.u_max, self.zeta_max, self.cp, self.cv, self.T = u_max, zeta_max, cp, cv, T
        self.H, self.iters = (None if H is None else np.asarray(H, float)), int(iters)
        self.reset()

    def reset(self):
        self.z = None
        self.u_prev = np.zeros(self.Ka.shape[0])
        self.k = 0
        self.zeta = np.zeros(3)
        self.log = []

    def __call__(self, y):
        c = self.ctl
        ys = np.asarray(y)[self.obs]
        yt = ys - c["J"] @ self.u_prev
        k = self.k
        cz = c["cz"] if self.cz_seq is None else self.cz_seq[k]
        zp = np.zeros(c["Az"].shape[0]) if self.z is None else c["Az"] @ self.z + c["Bz"] @ self.u_prev + cz
        self.z = zp + c["L"] @ (yt - c["Cz"] @ zp)
        xh = np.zeros(12); xh[self.xs] = self.z[:len(self.xs)]
        u, self.zeta = servo_law(xh, self.zeta, self.x_ref[k], self.u_ff[k], 0.0, self.Ka, self.u_max, self.zeta_max,
                                 self.cp, self.cv, self.T, self.H, self.iters)
        self.k += 1
        self.u_prev = u
        self.log.append(xh)
        return np.concatenate([[0.0], u])


# ---------------------------------------------------------------------------------------------- live commands

class RefStream:
    """:func:`build_refs` one wingbeat at a time, for a command that changes live (the viewer / GUI):
    ``next(v_cmd)`` returns (x_ref_k, u_ff_k) for this wingbeat's commanded velocity v_cmd (3). Row for row equal
    to build_refs on the same command sequence when ff_lead <= 1 (a longer lead looks ahead with the current
    command held, since the future commands are unknown)."""

    def __init__(self, tab, e_f, T, rate, *, descend_after_stop=0.0, descend_delay=0, zeta_handoff=False,
                 stop_rate=1.0, zeta_gate=None, ff_lead=0, climb_after_stop=0.0):
        self.tab, self.e_f, self.T = tab, np.asarray(e_f, float), float(T)
        self.rate = np.broadcast_to(np.asarray(rate, float), (3,)).copy()
        self.das, self.dd, self.cas = float(descend_after_stop), int(descend_delay), float(climb_after_stop)
        self.handoff, self.gate, self.stop_rate, self.lead = bool(zeta_handoff), zeta_gate, float(stop_rate), int(ff_lead)
        self.reset()

    def reset(self):
        self.v, self.p = np.zeros(3), np.zeros(3)
        self.since = self.since_ramp = 10 ** 6

    def _step(self, v, since, v_cmd):
        rk = self.stop_rate * self.rate if not np.any(v_cmd) else self.rate
        step = np.clip(v_cmd - v, -rk, rk)
        if self.das > 0:
            moving = np.linalg.norm(v[:2]) > self.das
            since = 0 if moving else since + 1
            if (moving or since <= self.dd) and v[2] + step[2] < 0:
                step[2] = max(step[2], 0.0)
        if self.cas > 0 and np.linalg.norm(v[:2]) > self.cas and v[2] + step[2] > 0:
            step[2] = min(step[2], 0.0)
        return v + step, since

    def next(self, v_cmd):
        v_cmd = np.asarray(v_cmd, float)
        v_k = self.v
        v_next, since = self._step(v_k, self.since, v_cmd)
        att, vel, w, u = lookup(self.tab, self.e_f, v_k[None])
        if self.lead:
            v_lead, s_lead = v_next, since
            for _ in range(self.lead - 1):
                v_lead, s_lead = self._step(v_lead, s_lead, v_cmd)
            att, _, w, u = lookup(self.tab, self.e_f, v_lead[None])
        cols = [self.p, att[0], vel[0], w[0]]
        if self.handoff or self.gate is not None:
            n2 = v_k @ v_k
            keep = float(np.clip(v_next @ v_k / n2, 0.0, 1.0)) if self.handoff and n2 > 1e-18 else 1.0
            cols.append([keep])
        if self.gate is not None:
            self.since_ramp = 0 if np.abs(v_next - v_k).max() > 1e-12 else self.since_ramp + 1
            cols.append([0.0 if not np.any(v_cmd) and self.since_ramp <= self.gate else 1.0])
        x_ref = np.concatenate(cols)
        self.p = self.p + v_k * self.T
        self.v, self.since = v_next, since
        return x_ref, u[0]


class LiveCommand:
    """The command-following policy flown with a live command (src/view_free_hover.py, the GUI): ``set_command``
    at any time, then one call per wingbeat. ``pol`` = :func:`load_policy` of a ``command_loop`` file. Without an
    observer the policy reads the true strobed state (12); with ``observer`` (a hover.reduced plain lqg dict: the
    ``command_observer`` file) it reads the 18-channel sensor reading and flies the servo on the estimate, with
    the sensor-flight servo gain / climb shaping stored in the policy (``sensor_*`` keys) and the observer's
    per-beat offset :func:`trim_consistent_cz` computed online. Same law as :class:`CommandPolicy` /
    :class:`CmdLinearController` on the same command sequence."""

    def __init__(self, pol, observer=None):
        if observer is not None and (str(observer["kind"]) != "lqg" or "dr" in observer):
            raise ValueError("the command observer must be a plain lqg controller (hover.reduced)")
        self.pol, self.ctl = pol, observer
        sensor = observer is not None
        self.Ka = np.asarray(pol["sensor_Ka"] if sensor else pol["Ka"], float)
        self.u_max, self.zeta_max = float(pol["u_max"]), float(pol["zeta_max"])
        self.v_op = np.asarray(pol["v_op"], float)
        g = int(pol.get("zeta_gate", -1))
        self.refs = RefStream(pol["tab"], pol["e_f"], float(pol["T"]),
                              np.array([self.v_op[0], self.v_op[0], self.v_op[1]]) / float(pol["rate_beats"]),
                              descend_after_stop=float(pol.get("descend_after_stop", 0.0)),
                              descend_delay=int(pol.get("descend_delay", 0)),
                              zeta_handoff=bool(pol.get("zeta_handoff", False)), stop_rate=float(pol.get("stop_rate", 1.0)),
                              zeta_gate=None if g < 0 else g, ff_lead=int(pol.get("ff_lead", 0)),
                              climb_after_stop=float(pol.get("sensor_climb_after_stop", 0.0)) if sensor else 0.0)
        self.dirs = command_dirs(pol["e_f"])
        self.envelope = {a: float(v) for a, v in zip(AXES, pol["envelope"])} if "envelope" in pol else {}
        if sensor:
            self.obs, self.xs = np.asarray(observer["obs"], int), np.asarray(observer["xs"], int)
        self.set_command("HOVER")
        self.reset()

    def default_speed(self, name) -> float:
        """The operating speed the policy was evaluated at (v_op: FWD/REV, UP, DOWN)."""
        return {"HOVER": 0.0, "FWD": self.v_op[0], "REV": self.v_op[0], "UP": self.v_op[1],
                "DOWN": self.v_op[2] if len(self.v_op) > 2 else self.v_op[1]}[name]

    def set_command(self, name, speed=None) -> float:
        """Command HOVER / FWD / REV / UP / DOWN at ``speed`` m/s (default: the operating speed), clipped to the
        measured envelope. Returns the speed commanded."""
        name = name.upper()
        c = CMDS.index(name)
        s = self.default_speed(name) if speed is None else abs(float(speed))
        if name != "HOVER" and name in self.envelope:
            s = min(s, self.envelope[name])
        self.command, self.speed = name, (0.0 if name == "HOVER" else s)
        self.v_cmd = self.speed * self.dirs[c]
        return self.speed

    def set_velocity(self, v_fwd, v_up) -> tuple[float, float]:
        """Command a forward (+) / reverse (-) and a climb (+) / descent (-) speed at once, each clipped to its
        axis's envelope (the viewer's arrow keys). A combined command superposes the per-axis tables, as the
        references of a transition do; it was not in the random-command evaluation (single-axis segments)."""
        env = lambda name, v: min(abs(float(v)), self.envelope.get(name, abs(float(v))))
        vf = math.copysign(env("FWD" if v_fwd >= 0 else "REV", v_fwd), v_fwd) if v_fwd else 0.0
        vz = math.copysign(env("UP" if v_up >= 0 else "DOWN", v_up), v_up) if v_up else 0.0
        names = ([("FWD" if vf > 0 else "REV")] if vf else []) + ([("UP" if vz > 0 else "DOWN")] if vz else [])
        self.command = "+".join(names) or "HOVER"
        self.speed = math.hypot(vf, vz)
        self.v_cmd = vf * self.dirs[1] + vz * self.dirs[3]
        return vf, vz

    def reset(self):
        self.refs.reset()
        self.zeta = np.zeros(3)
        self.z = None
        self.u_prev = np.zeros(self.Ka.shape[0])
        self.prev_ref = None
        self.x_ref = None

    def __call__(self, obs):
        x_ref, u_ff = self.refs.next(self.v_cmd)
        if self.ctl is None:
            s = np.asarray(obs, float)
        else:
            c, n = self.ctl, len(self.xs)
            yt = np.asarray(obs)[self.obs] - c["J"] @ self.u_prev
            if self.z is None:
                zp = np.zeros(c["Az"].shape[0])
            else:
                xr0, uf0 = self.prev_ref
                cz = np.zeros_like(np.asarray(c["cz"], float))
                cz[:n] = x_ref[:12][self.xs] - c["Az"][:n, :n] @ xr0[:12][self.xs] - c["Bz"][:n] @ uf0
                zp = c["Az"] @ self.z + c["Bz"] @ self.u_prev + cz
            self.z = zp + c["L"] @ (yt - c["Cz"] @ zp)
            s = np.zeros(12); s[self.xs] = self.z[:n]
        u, self.zeta = servo_law(s, self.zeta, x_ref, u_ff, 0.0, self.Ka, self.u_max, self.zeta_max)
        self.prev_ref, self.u_prev, self.x_ref = (x_ref, u_ff), u, x_ref
        return np.concatenate([[0.0], u])


def linear_sim(A, B, drift, Ka, x_ref, u_ff, *, dither=None, u_max=U_MAX, zeta_max=ZETA_MAX, cp=1.0, cv=0.0, T=0.0,
               H=None, iters=0):
    """The servo on the linear model itself, batched over flights: (S (n, N+1, 12), U (n, N, nu))."""
    n, N1, _ = x_ref.shape
    N = N1 - 1
    x = np.zeros((n, 12)); zeta = np.zeros((n, 3))
    S = np.zeros((n, N + 1, 12)); U = np.zeros((n, N, B.shape[1]))
    for k in range(N):
        S[:, k] = x
        e = x - x_ref[:, k, :12]
        keep = x_ref[:, k, 12:13] if x_ref.shape[-1] > 12 else 1.0
        gate = x_ref[:, k, 13:14] if x_ref.shape[-1] > 13 else 1.0
        raw = u_ff[:, k] - e @ Ka[:, :12].T - zeta @ Ka[:, 12:].T
        if dither is not None:
            raw = raw + dither[:, k]
        if iters:
            u = allocate(raw, H, u_max, iters)
            d, fb = u - raw, raw - u_ff[:, k] - (dither[:, k] if dither is not None else 0.0)
            free = (np.einsum("ni,ij,nj->n", d, H, d) <= ALLOC_FREE * np.einsum("ni,ij,nj->n", fb, H, fb))[:, None]
        else:
            u = np.clip(raw, -u_max, u_max)
            free = (np.abs(raw) < u_max).all(1, keepdims=True)
        zeta = np.clip(keep * zeta + gate * free * (cp * e[:, :3] + cv * T * e[:, 6:9]), -zeta_max, zeta_max)
        U[:, k] = u
        x = x @ A.T + u @ B.T + drift
    S[:, N] = x
    return S, U


# ---------------------------------------------------------------------------------------------- metrics

def lost_at(alive) -> np.ndarray:
    """hover.mjx_flight.lost_at (kept here so the host module does not import JAX)."""
    alive = np.asarray(alive)
    return np.where(alive.all(-1), alive.shape[-1] - 1, np.argmin(alive, -1))


def segment_metrics(S, U, alive, sched, x_ref, T, *, e_f, u_max=U_MAX, min_len=30):
    """Per-segment and per-flight scores from TRUE strobed states S (n, N+1, 12), inputs U (n, N, nu), alive
    (n, N+1), sched (random_schedules), x_ref (n, >= N+1, 12). Segments that end past the horizon are kept when at
    least ``min_len`` beats of them were flown. Returns (segment rows, flight rows)."""
    n, N1, _ = S.shape
    N = N1 - 1
    dirs = command_dirs(e_f)
    la = lost_at(alive)
    sat = (np.abs(U) >= u_max * (1 - 1e-6)).any(-1)                     # (n, N)
    v_cmd = sched["v_cmd"]
    segs, flights = [], []
    for i in range(n):
        p, pr = S[i, :, :3], x_ref[i, :N1, :3]
        dev = np.linalg.norm(p - pr, axis=1)
        tiltdev = np.degrees(np.linalg.norm(S[i, :, 3:5] - x_ref[i, :N1, 3:5], axis=1))
        tilt = np.degrees(np.linalg.norm(S[i, :, 3:5], axis=1))
        last = int(la[i])
        ok_fl = last == N
        sl = slice(10, last + 1)
        flights.append({"flight": i, "survived": bool(ok_fl), "lost_at": last,
                        "rms_dev": float(np.sqrt(np.mean(dev[sl] ** 2))) if last >= 10 else float("nan"),
                        "max_dev": float(dev[sl].max()) if last >= 10 else float("nan"),
                        "max_tilt_dev_deg": float(tiltdev[:last + 1].max()), "max_tilt_deg": float(tilt[:last + 1].max())})
        for a, b, c, sp in sched["seg"][i]:
            if a < 0:
                break
            a, b, c = int(a), int(b), int(c)
            b = min(b, N)
            if b - a < min_len:
                continue
            mid = a + (b - a) // 2
            row = {"flight": i, "start": a, "end": b, "cmd": CMDS[c], "speed": float(sp), "alive": bool(last >= b)}
            if not row["alive"]:
                row.update(passed=False, follow=False)
                segs.append(row)
                continue
            d = dirs[c]
            v_seg = (p[b] - p[mid]) / ((b - mid) * T)
            prev = v_cmd[i, a - 1] if a > 0 else np.zeros(3)
            step = np.linalg.norm(v_cmd[i, a] - prev)
            if step > 0:
                hit = np.flatnonzero(np.linalg.norm(S[i, a:b + 1, 6:9] - v_cmd[i, a], axis=1) <= 0.37 * step)
                rise = int(hit[0]) if hit.size else b - a
            else:
                rise = 0
            row.update(rise=rise, tilt_dev_max_deg=float(tiltdev[a:b + 1].max()), sat_frac=float(sat[i, mid:b].mean()))
            if c == 0:
                vmag = float(np.linalg.norm(v_seg))
                row.update(v_seg=vmag, hold_max_dev=float(dev[mid:b + 1].max()), drift=float(np.linalg.norm(p[b] - p[mid])),
                           passed=bool(vmag <= 0.02 and dev[mid:b + 1].max() <= 0.05), follow=True)
            else:
                al = float(v_seg @ d)
                cross = float(np.linalg.norm(v_seg - al * d))
                row.update(along=al - sp, cross=cross, passed=bool(abs(al - sp) <= max(0.05, 0.15 * sp) and cross <= 0.05),
                           follow=bool(al > 0 and al >= 0.5 * sp))
            segs.append(row)
    return segs, flights


def _q(x, q):
    x = np.asarray([v for v in x if v == v], float)
    return float(np.quantile(x, q)) if x.size else float("nan")


def summarize(segs, flights) -> dict:
    """Median / p95 / worst overall and per command."""
    out = {"n_flights": len(flights), "survival": float(np.mean([f["survived"] for f in flights])) if flights else float("nan"),
           "n_segments": len(segs), "follow_rate": float(np.mean([s["passed"] for s in segs])) if segs else float("nan"),
           "rms_dev_median": _q([f["rms_dev"] for f in flights], 0.5), "max_dev_p95": _q([f["max_dev"] for f in flights], 0.95),
           "max_tilt_dev_p95_deg": _q([f["max_tilt_dev_deg"] for f in flights], 0.95),
           "max_tilt_survivors_deg": max([f["max_tilt_deg"] for f in flights if f["survived"]], default=float("nan"))}
    alive = [s for s in segs if s["alive"]]
    mov = [s for s in alive if s["cmd"] != "HOVER"]
    hov = [s for s in alive if s["cmd"] == "HOVER"]
    ab = [abs(s["along"]) for s in mov]
    out.update(along_abs_median=_q(ab, 0.5), along_abs_p95=_q(ab, 0.95), along_abs_max=max(ab, default=float("nan")),
               cross_p95=_q([s["cross"] for s in mov], 0.95), rise_median=_q([s["rise"] for s in alive], 0.5),
               sat_frac_mean=float(np.mean([s["sat_frac"] for s in alive])) if alive else float("nan"),
               direction_follow=float(np.mean([s["follow"] for s in segs if s["cmd"] != "HOVER"])) if mov else float("nan"),
               hover_v_median=_q([s["v_seg"] for s in hov], 0.5), hover_v_p95=_q([s["v_seg"] for s in hov], 0.95),
               hover_hold_p95=_q([s["hold_max_dev"] for s in hov], 0.95), hover_drift_p95=_q([s["drift"] for s in hov], 0.95))
    per = {}
    for c in CMDS:
        sc = [s for s in segs if s["cmd"] == c]
        if not sc:
            continue
        al = [s for s in sc if s["alive"]]
        d = {"n": len(sc), "pass_rate": float(np.mean([s["passed"] for s in sc])),
             "rise_median": _q([s["rise"] for s in al], 0.5), "sat_frac_mean": float(np.mean([s["sat_frac"] for s in al])) if al else float("nan"),
             "tilt_dev_p95_deg": _q([s["tilt_dev_max_deg"] for s in al], 0.95)}
        if c == "HOVER":
            d.update(v_median=_q([s["v_seg"] for s in al], 0.5), v_p95=_q([s["v_seg"] for s in al], 0.95),
                     hold_p95=_q([s["hold_max_dev"] for s in al], 0.95), drift_p95=_q([s["drift"] for s in al], 0.95))
        else:
            a_ = [abs(s["along"]) for s in al]
            d.update(along_signed_median=_q([s["along"] for s in al], 0.5), along_abs_median=_q(a_, 0.5),
                     along_abs_p95=_q(a_, 0.95), cross_p95=_q([s["cross"] for s in al], 0.95),
                     direction_follow=float(np.mean([s["follow"] for s in sc])))
        per[c] = d
    out["per_cmd"] = per
    return out


# ---------------------------------------------------------------------------------------------- ILC and envelope

def cell_list(n_maps=2, n_speeds=16):
    """[(map, axis, speed index)]: per map the hover cell (axis 0, index 0) then 4 axes x (n_speeds - 1) speeds."""
    cells = []
    for m in range(n_maps):
        cells.append((m, 0, 0))
        for a in range(4):
            cells += [(m, a, j) for j in range(1, n_speeds)]
    return cells


def cell_velocity(tab, e_f, axis, j):
    return tab["speeds"][axis, j] * command_dirs(e_f)[axis + 1]


def cell_window_stats(S, U, alive, u_ff, dither, flights, window, T):
    """Mean over alive flights x window beats of (U - u_ff - dither), strobed att / vel / w, and the mean-motion
    velocity over the window."""
    lo, hi = window
    fl = [i for i in flights if bool(np.asarray(alive[i][:hi + 1]).all())]
    if not fl:
        return None
    fl = np.array(fl)
    fb = (U[fl, lo:hi] - u_ff[fl, lo:hi] - dither[fl, lo:hi]).mean((0, 1))
    xs = S[fl, lo:hi].mean((0, 1))
    vm = ((S[fl, hi, :3] - S[fl, lo, :3]) / ((hi - lo) * T)).mean(0)
    return {"n": len(fl), "du": fb, "att": xs[3:6], "vel": xs[6:9], "w": xs[9:12], "v_mean": vm}


def ilc_update(tabs, S, U, alive, u_ff, dither, cell_of, cells, e_f, T, *, window=(15, 30), u_max=U_MAX, gain=1.0,
               good=None, dv_clip=0.2):
    """One ILC iteration on every cell (or only those with ``good[ci]``): u_ff += gain * mean feedback effort
    (clipped to 0.9 u_max), att and w references := their measured means, the strobed-minus-mean-motion velocity
    offset dv := measured (its component along the command clipped to +-dv_clip of the speed; 0.2: r_v in
    [0.8, 1.2]; ``dv_clip`` may be a per-map sequence). Returns refined copies of ``tabs``."""
    new = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in tabs]
    info = []
    for ci, (m, a, j) in enumerate(cells):
        st = cell_window_stats(S, U, alive, u_ff, dither, np.flatnonzero(cell_of == ci), window, T)
        if st is None or (good is not None and not good[ci]):
            info.append({"cell": ci, "n": 0})
            continue
        t = new[m]
        uo = t["u"][a, j]
        un = np.clip(uo + gain * st["du"], -0.9 * u_max, 0.9 * u_max)
        v_c = cell_velocity(t, e_f, a, j)
        dv = st["vel"] - st["v_mean"]
        s = np.linalg.norm(v_c)
        if s > 0:
            d = v_c / s
            dc = dv_clip[m] if np.ndim(dv_clip) else dv_clip
            al = float(np.clip(dv @ d, -dc * s, dc * s))
            dv = dv - (dv @ d) * d + al * d
        axes = range(4) if (a, j) == (0, 0) else [a]          # the hover cell is entry 0 of every axis
        for aa in axes:
            t["u"][aa, j], t["att"][aa, j], t["w"][aa, j], t["dv"][aa, j] = un, st["att"], st["w"], dv
        info.append({"cell": ci, "n": st["n"], "du_inf": float(np.abs(st["du"]).max()), "u_inf": float(np.abs(un).max()),
                     "r_v": float((st["vel"] @ v_c) / (st["v_mean"] @ v_c)) if s > 0 and abs(st["v_mean"] @ v_c) > 1e-9 else 1.0})
    return new, info


def trim_sensitivity(A, B):
    """Linear steady-state sensitivity of the trim to its velocity: the minimum-norm (d att, d u) (yaw fixed) that
    keeps the att / vel / w rows of x_{k+1} = x_k when vel changes by d vel. Returns (G_att (3, 3), G_u (nu, 3))."""
    M = (A - np.eye(12))[3:12]
    X = np.hstack([M[:, 3:5], B[3:12]])                  # roll, pitch, u
    G = np.linalg.pinv(X) @ (-M[:, 6:9])
    G_att = np.zeros((3, 3)); G_att[:2] = G[:2]
    return G_att, G[2:]


def velocity_ilc(tabs, S, alive, cell_of, cells, e_f, T, G_att, G_u, *, maps=(1,), window=(15, 30), gain=1.0, u_max=U_MAX):
    """For controllers without a velocity measurement (the IMU-only N variant): the cells of ``maps`` move their
    trim (att, u) by gain x the linear trim sensitivity times the TRUE mean-motion velocity error of their flights
    (measured in the learning flights; the policy itself never reads it). Returns copies of ``tabs`` and info."""
    new = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in tabs]
    lo, hi = window
    info = []
    for ci, (m, a, j) in enumerate(cells):
        if m not in maps or (a, j) == (0, 0):
            continue
        fl = [i for i in np.flatnonzero(cell_of == ci) if bool(np.asarray(alive[i][:hi + 1]).all())]
        if not fl:
            continue
        vm = ((S[fl, hi, :3] - S[fl, lo, :3]) / ((hi - lo) * T)).mean(0)
        dv = cell_velocity(new[m], e_f, a, j) - vm
        t = new[m]
        t["att"][a, j] += gain * G_att @ dv
        t["u"][a, j] = np.clip(t["u"][a, j] + gain * G_u @ dv, -0.9 * u_max, 0.9 * u_max)
        info.append({"cell": ci, "dv": dv.tolist()})
    return new, info


def extrapolate_tables(tabs_new, tabs_lp, good, cells, *, max_factor=2.0):
    """Continuation: cells whose flights did not give a usable ILC measurement (``good[ci]`` False) get the
    measured corrections (table - LP table, for u / att / w / dv) of their axis extrapolated linearly in speed
    from the two largest good speeds (from hover and the largest one when only one is good), with s capped at
    ``max_factor`` x the largest good speed. Returns (tables, per-axis info). Axes with no good cell above hover
    keep their entries."""
    out = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for t in tabs_new]
    info = {}
    for m, t in enumerate(out):
        lp = tabs_lp[m]
        sp = t["speeds"]
        for a in range(4):
            gj = [j for j in range(1, sp.shape[1]) if good[cells.index((m, a, j))]]
            bj = [j for j in range(1, sp.shape[1]) if not good[cells.index((m, a, j))]]
            info[f"{m}/{AXES[a]}"] = {"good": [float(sp[a, j]) for j in gj], "extrapolated": [float(sp[a, j]) for j in bj]}
            if not gj or not bj:
                continue
            pts = [0] + gj
            j1, j0 = pts[-1], pts[-2]
            s1, s0 = float(sp[a, j1]), float(sp[a, j0])
            corr = lambda key, j: t[key][a, j] - (lp[key][a, j] if key in ("u", "att") else 0.0)
            for key in ("u", "att", "w", "dv"):
                g = (corr(key, j1) - corr(key, j0)) / (s1 - s0)
                c1 = corr(key, j1)
                for j in bj:
                    s = min(float(sp[a, j]), max_factor * s1)
                    base = lp[key][a, j] if key in ("u", "att") else 0.0
                    t[key][a, j] = base + c1 + (s - s1) * g
            t["u"][a] = np.clip(t["u"][a], -0.9 * U_MAX, 0.9 * U_MAX)
    return out, info


def cell_pass(S, U, alive, flights, cell_v, window, T, u_max=U_MAX):
    """Envelope-pass verdict of one clean constant-command flight (index flights[0])."""
    i = flights[0]
    lo, hi = window
    if not bool(np.asarray(alive[i][:hi + 1]).all()):
        return {"passed": False, "alive": False}
    vm = (S[i, hi, :3] - S[i, lo, :3]) / ((hi - lo) * T)
    s = float(np.linalg.norm(cell_v))
    sat = float((np.abs(U[i, lo:hi]) >= u_max * (1 - 1e-6)).any(-1).mean())
    if s == 0:
        return {"passed": bool(np.linalg.norm(vm) <= 0.05 and sat <= 0.2), "alive": True, "v": float(np.linalg.norm(vm)), "sat": sat}
    d = cell_v / s
    al = float(vm @ d) - s
    cr = float(np.linalg.norm(vm - (vm @ d) * d))
    return {"passed": bool(abs(al) <= max(0.05, 0.15 * s) and cr <= 0.05 and sat <= 0.2), "alive": True, "along": al,
            "cross": cr, "sat": sat}


def envelope(cell_rows, cells, speeds, n_maps=2) -> dict:
    """Per map and axis: V_max = the largest grid speed s such that every grid speed <= s passes."""
    out = {}
    for m in range(n_maps):
        out[m] = {}
        for a, name in enumerate(AXES):
            vmax = 0.0
            for j in range(1, speeds.shape[1]):
                ci = cells.index((m, a, j))
                if not cell_rows[ci]["passed"]:
                    break
                vmax = float(speeds[a, j])
            out[m][name] = vmax
    return out


def local_models(S, U, alive, flights, window, A, B, Ka):
    """Regress x_{k+1} - [p_k; 0] on [x_k (att vel w), u_k, 1] over the (dithered) flights' window: (A_v, B_v)
    vs (A, B) and the servo's augmented closed-loop radius on the regressed model."""
    lo, hi = window
    fl = [i for i in flights if bool(np.asarray(alive[i][:hi + 1]).all())]
    if len(fl) < 2:
        return None
    X, Y = [], []
    for i in fl:
        for k in range(lo, hi):
            X.append(np.r_[S[i, k, 3:12], U[i, k], 1.0])
            y = S[i, k + 1].copy(); y[:3] -= S[i, k, :3]
            Y.append(y)
    X, Y = np.array(X), np.array(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A_v = np.zeros((12, 12)); A_v[:3, :3] = np.eye(3); A_v[:, 3:] = W[:9].T
    B_v = W[9:9 + B.shape[1]].T
    nu = B.shape[1]
    Aa = np.zeros((15, 15)); Aa[:12, :12] = A_v; Aa[12:, :3] = np.eye(3); Aa[12:, 12:] = np.eye(3)
    Ba = np.vstack([B_v, np.zeros((3, nu))])
    return {"n_samples": len(X), "cond": float(np.linalg.cond(X)),
            "A_rel_err": float(np.linalg.norm(A_v[:, 3:] - A[:, 3:]) / np.linalg.norm(A[:, 3:])),
            "B_rel_err": float(np.linalg.norm(B_v - B) / np.linalg.norm(B)),
            "rho_servo": float(np.abs(np.linalg.eigvals(Aa - Ba @ Ka)).max())}


# ---------------------------------------------------------------------------------------------- policy file

def save_policy(path, *, tabs, chosen, Ka, K_hover, e_f, T, v_op, u_max=U_MAX, zeta_max=ZETA_MAX, rate_beats=4,
                A=None, B=None, drift=None, extra=None):
    arrs = {"Ka": Ka, "K_hover": K_hover, "e_f": e_f, "T": T, "v_op": np.asarray(v_op, float), "u_max": u_max,
            "zeta_max": zeta_max, "rate_beats": rate_beats, "chosen_map": chosen,
            "tilt_maps": np.array([t["tilt_max_deg"] for t in tabs])}
    for m, t in enumerate(tabs):
        for k in ("speeds", "att", "u", "dv", "w", "t", "feasible"):
            arrs[f"tab{m}_{k}"] = t[k]
    for k, v in (("A", A), ("B", B), ("drift", drift)):
        if v is not None:
            arrs[k] = v
    arrs.update(extra or {})
    np.savez(path, **arrs)


def load_policy(path) -> dict:
    z = np.load(path)
    n = len(z["tilt_maps"])
    tabs = [{k: z[f"tab{m}_{k}"] for k in ("speeds", "att", "u", "dv", "w", "t", "feasible")} | {"tilt_max_deg": float(z["tilt_maps"][m])}
            for m in range(n)]
    out = {k: z[k] for k in z.files if not k.startswith("tab")}
    out["tabs"] = tabs
    out["tab"] = tabs[int(z["chosen_map"])]
    return out
