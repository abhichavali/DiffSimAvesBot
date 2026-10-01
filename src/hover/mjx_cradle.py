"""The launch cradle (hover.launch.release) and the open-loop trim (hover.trim) on the device, batched.

``release`` holds the bird's wingbeat-averaged pose with a PID while the flapping settles, then lets go; its
mean wrench over the last wingbeats is the trim error hover.trim drives to zero by Gauss-Newton. Here every
trim candidate (theta, extra tilt) is one element of a vmapped batch, so one GN iteration (1 + 2 x 27 cradle
runs of 16 wingbeats) is a single device call.

The crank speed e^g may differ per candidate, so the wingbeat length n (steps) is traced: the scan runs
``n_cradle * n_max`` steps and masks the steps past ``n_cradle * n``; the one-wingbeat boxcar of the strobed
pose is a ring buffer of ``n_max`` samples with a running sum. Everything else is hover.launch.release step
for step (the strobe reads the same one-step-stale kinematics the CPU loop reads after ``mj_step``).

    cr = Cradle(bird, K)                    # the rest state (gravity-free settle, CPU) on the device
    wrench = cr.wrenches(thetas, attitudes, n_cradle=16, n_avg=8)      # (B, 6): [force (3), moment (3)]

Moving cradle (``vels``: the bird held on a target moving at v): :func:`trim_at_speed` solves steady-flight trims
(24 depths + tilt + yaw) at command speeds, :func:`orbit_entries` turns them into hover.command table entries
(`opt/hover_cmd.py trims`, which writes the `learn --ff-trims` input; docs/history/command_flight_faster_notes_2026-10.md X8).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import flying
from hover import launch as L
from hover import mjx_flight as F
from runtime.mjx import aero_runtime, rollout
from runtime.sim import PlateWingSim

TWO_PI = 2 * math.pi


class Cradle:
    """The released-from-rest start of hover.launch.release on the device (rest state + gains)."""

    def __init__(self, bird: L.Bird, K: int, *, k_phi=L.K_PHI, body_inertia=4e-5 + flying.BODY_INERTIA, **flight_kw):
        m = bird.model
        g0 = m.opt.gravity.copy()
        m.opt.gravity[:] = 0
        sim = PlateWingSim(m, bird.morph)             # the gravity-free rest settle, as release()
        m.opt.gravity[:] = g0
        sim.data.qvel[:] = 0
        crank_len = sim.data.actuator_length[sim.crank_ids].copy()
        rel = SimpleNamespace(sim=sim, drive=SimpleNamespace(phi=0.0, L_des=crank_len), com_ref=np.zeros(3),
                              q_ref=np.array([1.0, 0, 0, 0]), x_orbit=np.zeros(12))
        theta0 = np.zeros(1 + 8 * K)
        self.fl = F.Flight(bird, rel, theta0, K, k_phi=k_phi, **flight_kw)
        self.bird = bird
        M, I, ws = m.body_mass.sum(), body_inertia, 2 * math.pi * 1.5
        self.gains = (M * ws**2, 2 * 0.8 * M * ws, M * ws**3 / 4, I * ws**2, 2 * 0.8 * I * ws, I * ws**3 / 4)

    def n_steps(self, g) -> int:
        return round(TWO_PI / (self.bird.omega * math.exp(g)) / self.fl.static.dt)

    def run(self, ar, theta, att, n_cradle: int, n_avg: int, n_max: int, v=None):
        """One cradle run (traceable): theta (1 + 8K), att (wxyz), ``v`` (3, traced; None = hover): a MOVING cradle,
        hover.launch.release(v=...): the free joints start at linear velocity v and the cradle error is
        strobe - [v t, 0, v, 0]. Returns (mean cradle wrench (6), final carry, com_ref, q_ref, phi)."""
        S = self.fl.static
        kp, kd, ki, kr, krd, kri = self.gains
        af, dt = S.af, S.dt
        c = F.carry0(ar)
        dx = F.forward(ar, c.dx)
        dx = rotate_quat(S, ar, dx, att)
        if v is not None:
            qvel = dx.qvel
            for qa, va, b in S.free:
                qvel = qvel.at[va:va + 3].set(v.astype(qvel.dtype))
            dx = F.forward(ar, dx.replace(qvel=qvel))
        com0, q0 = dx.subtree_com[0], dx.xquat[af]
        K = S.K
        a = theta[1:1 + 4 * K].reshape(4, K)
        b = theta[1 + 4 * K:].reshape(4, K)
        w_f = self.bird.omega * jnp.exp(theta[0])
        speed = self.bird.a0 * jnp.exp(theta[0])
        n = jnp.round(TWO_PI / (w_f * dt)).astype(jnp.int32)
        total, acc_from = n_cradle * n, (n_cradle - n_avg) * n
        ft = dx.qpos.dtype
        ks = jnp.arange(1, K + 1, dtype=ft)
        crank = jnp.asarray(S.crank_ids, jnp.int32)
        cq = jnp.asarray(S.crank_qadr, jnp.int32)
        gear = jnp.asarray(S.crank_gear, ft)
        q0c = q0 * jnp.array([1.0, -1.0, -1.0, -1.0], ft)

        def strobe(dx):
            q, v = S.qadr, S.vadr
            com = dx.subtree_com[0]
            w_world = dx.xmat[af] @ dx.qvel[v + 3:v + 6]
            qe = F.qmul(dx.xquat[af], q0c)
            att_e = 2.0 * jnp.where(qe[0] >= 0, 1.0, -1.0) * qe[1:]
            vcom = dx.qvel[v:v + 3] + jnp.cross(w_world, com - dx.qpos[q:q + 3])
            return jnp.concatenate([com - com0, att_e, vcom, w_world])

        def step(carry, k):
            c, buf, tot, iE, acc = carry
            # carry[1] is still the unmodified buffer below (functional update), so an inactive step restores it
            dx = c.dx
            active = k < total
            s = strobe(dx)
            if v is not None:
                vv = v.astype(ft)
                s = s - jnp.concatenate([vv * (k.astype(ft) * dt), jnp.zeros(3, ft), vv, jnp.zeros(3, ft)])
            gone = jnp.where(k >= n, buf[(k - n) % n_max], 0.0)
            buf = buf.at[k % n_max].set(s)
            tot = tot + s - gone
            avg = tot / jnp.minimum(k + 1, n).astype(ft)
            iE_n = iE + avg[:6] * dt
            Fc = -kp * avg[:3] - kd * avg[6:9] - ki * iE_n[:3]
            Tc = -kr * avg[3:6] - krd * avg[9:12] - kri * iE_n[3:]
            xfrc = jnp.zeros_like(dx.xfrc_applied).at[af, :3].set(Fc).at[af, 3:].set(
                Tc + jnp.cross(dx.subtree_com[0] - dx.xipos[af], Fc))
            dx = dx.replace(xfrc_applied=xfrc)
            phi = k.astype(ft) * (w_f * dt)
            w = speed * (1.0 + jnp.sum(a * jnp.cos(ks * phi) + b * jnp.sin(ks * phi), axis=1))
            q_c = dx.qpos[cq]
            ctrl = jnp.zeros_like(dx.ctrl).at[crank].set(w + S.k_phi * (c.e + gear * q_c - dx.actuator_length[crank]))

            def aero(d):
                def recompute(_):
                    qf, _, ns, _, _ = aero_runtime.aero_wrench(ar.mx, d, S.aero, c.st)
                    return qf, ns
                qf, ns = jax.lax.cond(c.i % S.substep == 0, recompute, lambda _: (c.qfrc, c.st), None)
                return qf, (qf, ns)

            dx2, (qf, st) = rollout.physics_step(ar.mx, dx, ctrl, ar.pitch, aero)
            nc = F.Carry(dx2, st, qf, (c.i + 1) % S.substep, c.e + w * dt - gear * (dx2.qpos[cq] - q_c))
            nc = nc._replace(dx=F.unwind(S, nc.dx))
            acc_n = acc + jnp.where(k >= acc_from, jnp.concatenate([Fc, Tc]), 0.0)
            new = (nc, buf, tot, iE_n, acc_n)
            out = jax.tree_util.tree_map(lambda x, y: jnp.where(active, x, y), new, carry)
            return rollout.match_dtypes(out, carry), None

        z = lambda *sh: jnp.zeros(sh, ft)
        carry0 = (c._replace(dx=dx), z(n_max, 12), z(12), z(6), z(6))
        (c, _, _, _, acc), _ = jax.lax.scan(step, carry0, jnp.arange(n_cradle * n_max))
        dx = F.forward(ar, c.dx.replace(xfrc_applied=jnp.zeros_like(c.dx.xfrc_applied)))
        wrench = acc / (n_avg * n).astype(ft)
        phi = jnp.mod(total.astype(ft) * (w_f * dt), TWO_PI)
        return wrench, c._replace(dx=dx), dx.subtree_com[0], dx.xquat[af], phi

    def wrenches(self, thetas, atts, *, n_cradle=16, n_avg=8, chunk=None, vels=None):
        """Mean cradle wrench per candidate (B, 6): hover.trim.cradle_wrench for every (theta, attitude);
        ``vels`` (B, 3): per-candidate moving-cradle velocity (:meth:`run` ``v``; None = hover)."""
        thetas, atts = np.asarray(thetas, float), np.asarray(atts, float)
        n_max = max(self.n_steps(g) for g in thetas[:, 0])
        if vels is None:
            f = jax.jit(jax.vmap(lambda ar, th, q: (self.run(ar, th, q, n_cradle, n_avg, n_max)[0],), in_axes=(None, 0, 0)))
            (W,) = F.run_batched(f, self.fl.arrays, thetas, atts, chunk=chunk)
            return W
        f = jax.jit(jax.vmap(lambda ar, th, q, v: (self.run(ar, th, q, n_cradle, n_avg, n_max, v)[0],), in_axes=(None, 0, 0, 0)))
        (W,) = F.run_batched(f, self.fl.arrays, thetas, atts, np.asarray(vels, float), chunk=chunk)
        return W

    def orbits(self, thetas, atts, vels, *, n_cradle=24, n_avg=8, chunk=None):
        """:meth:`wrenches` of moving cradles plus the bird's state at the end of each run (a wingbeat boundary,
        drive phase 0, as the flight strobes): (W (B, 6), X (B, 13) = [com (3), airframe quat wxyz (4), COM velocity
        (3), world angular velocity (3)], absolute world quantities, the cradle force already removed)."""
        thetas, atts = np.asarray(thetas, float), np.asarray(atts, float)
        n_max = max(self.n_steps(g) for g in thetas[:, 0])
        S = self.fl.static

        def one(ar, th, q, v):
            w, c, com, qe, _ = self.run(ar, th, q, n_cradle, n_avg, n_max, v)
            dx = c.dx
            w_world = dx.xmat[S.af] @ dx.qvel[S.vadr + 3:S.vadr + 6]
            vcom = dx.qvel[S.vadr:S.vadr + 3] + jnp.cross(w_world, com - dx.qpos[S.qadr:S.qadr + 3])
            return w, jnp.concatenate([com, qe, vcom, w_world])
        f = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))
        W, X = F.run_batched(f, self.fl.arrays, thetas, atts, np.asarray(vels, float), chunk=chunk)
        return W, X


def rotate_quat(S, ar, dx, quat):
    """hover.launch.rotate_bird by the quaternion ``quat`` about the COM, then forward."""
    R = F.quat2mat(quat)
    com = F._com(ar, dx)
    qpos = dx.qpos
    for qa, va, b in S.free:
        qpos = qpos.at[qa:qa + 3].set(com + R @ (qpos[qa:qa + 3] - com))
        qpos = qpos.at[qa + 3:qa + 7].set(F.qmul(quat, qpos[qa + 3:qa + 7]))
    return F.forward(ar, dx.replace(qpos=qpos))


def trim(cr: Cradle, fp, *, iters=6, chunk=None, log=print):
    """hover.trim's Gauss-Newton on the device. ``fp`` = the starting hover.policy.FlightPolicy. Returns
    (x = [theta, tilt_x, tilt_y], last wrench, attitude (wxyz), Jacobian or None)."""
    from hover.trim import SCALE, attitude
    K, att0 = fp.K, fp.attitude
    nt = 1 + 8 * K
    x = np.concatenate([fp.theta, [0.0, 0.0]])
    steps = np.concatenate([[0.01], np.full(nt - 1, 0.02), np.full(2, 0.02)])
    J = None
    for it in range(iters):
        X = [x] + [x + s * np.eye(len(x))[j] * sg for j, s in enumerate(steps) for sg in (1, -1)]
        W = cr.wrenches([xx[:nt] for xx in X], [attitude(att0, xx[nt:]) for xx in X], chunk=chunk)
        w = W[0]
        if not np.isfinite(w).all():
            raise FloatingPointError("base rollout unstable")
        log(f"it{it}: cradle F {np.round(w[:3] * 1e3, 2)} mN  M {np.round(w[3:] * 1e3, 3)} mNm  "
            f"a0 {fp.control.a0 * math.exp(x[0]):.3f}  tilt+ {np.round(np.degrees(x[nt:]), 2)} deg")
        if np.abs(w[:3]).max() < 0.5e-3 and np.abs(w[3:]).max() < 0.02e-3:
            break
        J = np.stack([(W[1 + 2 * j] - W[2 + 2 * j]) * SCALE / (2 * steps[j]) for j in range(len(x))], 1)
        J[:, ~np.isfinite(J).all(0)] = 0.0
        x = x - np.linalg.pinv(J) @ (w * SCALE)
    return x, w, attitude(att0, x[nt:]), J


def attitude3(att0, rv):
    """``att0`` (wxyz) rotated by the world-frame rotation vector ``rv`` = (tilt_x, tilt_y, yaw) (hover.trim.attitude
    with a yaw component)."""
    rv = np.asarray(rv, float)
    a = float(np.linalg.norm(rv))
    qe = np.array([1.0, 0, 0, 0])
    if a > 1e-12:
        mujoco.mju_axisAngle2Quat(qe, rv / a, a)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qe, np.asarray(att0, float))
    return q


def _qmul_np(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def orbit_entries(X, X_hover, v):
    """Command-table entries (hover.command tables: att, strobed-minus-commanded velocity dv, strobed rate w) of a
    steady flight whose wingbeat-boundary state is ``X`` (:meth:`Cradle.orbits`) at velocity ``v``, relative to the
    hover cradle's ``X_hover`` (the flight's release reference: its q_ref and x_orbit)."""
    qh = X_hover[3:7] * np.array([1.0, -1.0, -1.0, -1.0])
    qe = _qmul_np(X[3:7], qh)
    att = 2.0 * np.sign(qe[0] if qe[0] != 0 else 1.0) * qe[1:]
    vel = X[7:10] - X_hover[7:10]
    return {"att": att, "dv": vel - np.asarray(v, float), "w": X[10:13] - X_hover[10:13]}


def _lp_trim_step(J, w, u, trust, ang_trust, ang_cost=1e-3, ang=None, ang_box=None):
    """min t + 1e3 |s|_1 + ang_cost |d ang|_1 s.t. J dx + s = -w, |u_i + du_i| <= t, |du_i| <= trust,
    |d ang_j| <= ang_trust (dx = [du (nu), d ang (n_ang)]); ``ang_box`` (n_ang,): |ang_j + d ang_j| <= ang_box_j.
    u = 0 makes it a plain (min |du|_inf) Newton step."""
    from scipy.optimize import linprog
    m, n = J.shape
    nu = len(u)
    na = n - nu
    # variables: dx (n), t, s+ (m), s- (m), a+ (na), a- (na)   with d ang = a+ - a- tied below
    N = n + 1 + 2 * m + 2 * na
    c = np.zeros(N); c[n] = 1.0; c[n + 1:n + 1 + 2 * m] = 1e3; c[n + 1 + 2 * m:] = ang_cost
    Aeq = np.zeros((m + na, N)); beq = np.zeros(m + na)
    Aeq[:m, :n] = J; Aeq[:m, n + 1:n + 1 + m] = np.eye(m); Aeq[:m, n + 1 + m:n + 1 + 2 * m] = -np.eye(m); beq[:m] = -w
    for j in range(na):                                   # d ang_j - a+_j + a-_j = 0
        Aeq[m + j, nu + j] = 1.0; Aeq[m + j, n + 1 + 2 * m + j] = -1.0; Aeq[m + j, n + 1 + 2 * m + na + j] = 1.0
    Aub = np.zeros((2 * nu, N)); bub = np.zeros(2 * nu)
    for i in range(nu):                                   # +-(u_i + du_i) <= t
        Aub[2 * i, i] = 1.0; Aub[2 * i, n] = -1.0; bub[2 * i] = -u[i]
        Aub[2 * i + 1, i] = -1.0; Aub[2 * i + 1, n] = -1.0; bub[2 * i + 1] = u[i]
    ab = [(-ang_trust, ang_trust)] * na
    if ang_box is not None:
        ab = [(max(-ang_trust, -bx - a0), min(ang_trust, bx - a0)) for a0, bx in zip(ang, ang_box)]
        ab = [(lo, hi) if lo <= hi else (hi, hi) for lo, hi in ab]
    bounds = [(-trust, trust)] * nu + ab + [(0, None)] * (1 + 2 * m + 2 * na)
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(f"trim LP failed: {r.message}")
    return r.x[:n], float(np.abs(r.x[n + 1:n + 1 + 2 * m]).sum())


def trim_at_speed(cr: Cradle, fp, vels, x0s, *, iters=4, trust=0.03, ang_trust=math.radians(5.0), n_cradle=24,
                  n_avg=8, tol=0.02, du_stop=0.003, step_u=0.01, step_ang=0.005, chunk=None, log=print, min_iters=None,
                  ang_box=None):
    """Steady-flight trims at the moving-cradle velocities ``vels`` (C, 3), solved together (one batched call per
    iteration): per cell x = [24 harmonic depths (correction to fp.theta), tilt_x, tilt_y, yaw] (world rotation
    vector applied to the release attitude; crank speed g and a0 fixed). Each iteration measures the mean cradle
    wrench w of x and its central-difference Jacobian J (1 + 2 n runs per cell, :meth:`Cradle.orbits`), then takes
    the LP step min |u + du|_inf s.t. w + J dx = 0 (slack penalised), |du| <= ``trust``, |d ang| <= ``ang_trust``.
    A cell stops once its scaled residual |w SCALE|_inf <= ``tol`` (forces in units of the weight, moments in mNm)
    and its last step moved u by <= ``du_stop``. ``min_iters`` (default all): only the first min_iters steps minimise
    |u|_inf; later ones are plain Newton steps (min |dx|, same trust region) that converge onto the trim manifold
    without sliding along it. ``ang_box`` (3,) [rad]: absolute bounds on |tilt_x|, |tilt_y|, |yaw|. One hover cradle run (fp.theta, release attitude, v = 0) rides
    along for :func:`orbit_entries`. Returns (list of per-cell dicts: x, w, residual, history, X (end state)),
    X_hover)."""
    from hover.trim import SCALE
    K, att0 = fp.K, fp.attitude
    nu = 8 * K
    n = nu + 3
    steps = np.r_[np.full(nu, step_u), np.full(3, step_ang)]
    vels = np.asarray(vels, float)
    xs = [np.asarray(x, float).copy() for x in x0s]
    done = [False] * len(xs)
    hist = [[] for _ in xs]
    last = [None] * len(xs)
    X_h = None
    for it in range(iters + 1):
        act = [c for c in range(len(xs)) if not done[c]]
        if not act:
            break
        th, at, vv = [fp.theta.copy()], [np.asarray(att0, float)], [np.zeros(3)]
        for c in act:
            pts = [xs[c]] + ([xs[c] + s * sg * np.eye(n)[j] for j, s in enumerate(steps) for sg in (1, -1)] if it < iters else [])
            for x in pts:
                th.append(np.r_[fp.theta[0], fp.theta[1:] + x[:nu]]); at.append(attitude3(att0, x[nu:])); vv.append(vels[c])
        npad = 1 + len(xs) * (1 + 2 * n) - len(th)          # one batch shape for every call (no recompiles)
        th, at, vv = th + [th[0]] * npad, at + [at[0]] * npad, vv + [vv[0]] * npad
        W, X = cr.orbits(np.array(th), np.array(at), np.array(vv), n_cradle=n_cradle, n_avg=n_avg, chunk=chunk)
        X_h = X[0]
        o = 1
        for c in act:
            npt = 1 + (2 * n if it < iters else 0)
            Wc, Xc = W[o:o + npt], X[o:o + npt]
            o += npt
            w = Wc[0]
            res = float(np.abs(w * SCALE).max())
            row = {"it": it, "x": xs[c].tolist(), "w": w.tolist(), "residual": res, "u_inf": float(np.abs(xs[c][:nu]).max()),
                   "tilt_deg": np.degrees(xs[c][nu:nu + 2]).tolist(), "yaw_deg": float(np.degrees(xs[c][nu + 2]))}
            last[c] = {"x": xs[c].copy(), "w": w, "residual": res, "X": Xc[0]}
            if not np.isfinite(w).all():
                row["unstable"] = True
                hist[c].append(row)
                done[c] = True
                log(f"  cell {c} v {vels[c].round(3).tolist()} it{it}: UNSTABLE cradle run")
                continue
            moved = hist[c][-1].get("du_inf", 1.0) if hist[c] else 1.0
            newton = min_iters is not None and it >= min_iters
            if it == iters or (res <= tol and (moved <= du_stop or newton)):
                hist[c].append(row)
                done[c] = True
            else:
                J = np.stack([(Wc[1 + 2 * j] - Wc[2 + 2 * j]) * SCALE / (2 * steps[j]) for j in range(n)], 1)
                J[:, ~np.isfinite(J).all(0)] = 0.0
                minimise = min_iters is None or it < min_iters
                dx, slack = _lp_trim_step(J, w * SCALE, xs[c][:nu] if minimise else np.zeros(nu), trust, ang_trust,
                                          ang_cost=1e-3 if minimise else 0.3, ang=xs[c][nu:], ang_box=ang_box)
                row.update(du_inf=float(np.abs(dx[:nu]).max()), dang_deg=np.degrees(dx[nu:]).tolist(), lp_slack=slack,
                           pred_u_inf=float(np.abs(xs[c][:nu] + dx[:nu]).max()))
                last[c]["J"] = J
                hist[c].append(row)
                xs[c] = xs[c] + dx
            log(f"  cell {c} v {vels[c].round(3).tolist()} it{it}: F {np.round(w[:3] * 1e3, 2)} mN M {np.round(w[3:] * 1e3, 4)} mNm "
                f"res {res:.4f} | |u|inf {row['u_inf']:.4f} tilt {np.round(row['tilt_deg'], 2)} yaw {row['yaw_deg']:.2f} deg"
                + (f" | step |du| {row['du_inf']:.4f} slack {row['lp_slack']:.2g}" if "du_inf" in row else " | stop"))
    return [{**last[c], "history": hist[c]} for c in range(len(xs))], X_h

