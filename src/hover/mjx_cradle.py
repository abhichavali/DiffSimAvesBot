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

    def run(self, ar, theta, att, n_cradle: int, n_avg: int, n_max: int):
        """One cradle run (traceable): theta (1 + 8K), att (wxyz). Returns (mean cradle wrench (6), final carry,
        com_ref, q_ref, phi)."""
        S = self.fl.static
        kp, kd, ki, kr, krd, kri = self.gains
        af, dt = S.af, S.dt
        c = F.carry0(ar)
        dx = F.forward(ar, c.dx)
        dx = rotate_quat(S, ar, dx, att)
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

    def wrenches(self, thetas, atts, *, n_cradle=16, n_avg=8, chunk=None):
        """Mean cradle wrench per candidate (B, 6): hover.trim.cradle_wrench for every (theta, attitude)."""
        thetas, atts = np.asarray(thetas, float), np.asarray(atts, float)
        n_max = max(self.n_steps(g) for g in thetas[:, 0])
        f = jax.jit(jax.vmap(lambda ar, th, q: (self.run(ar, th, q, n_cradle, n_avg, n_max)[0],), in_axes=(None, 0, 0)))
        (W,) = F.run_batched(f, self.fl.arrays, thetas, atts, chunk=chunk)
        return W


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
