"""Train a state-feedback policy that makes the free-flying default hummer design HOVER, via
reverse-mode BPTT through the differentiable MJX rollout.

WHY A BODY WRENCH: the tensegrity flapping mechanism is *fundamentally unstable*
in free hover -- with gravity matched to the mean flap lift it still tumbles to
90-180 deg within ~15 wingbeats at every timestep (0.13-0.40 ms), because our
quasi-static wings can't do the within-wingbeat modulation a real hummingbird
stabilizes with. So the flapping supplies the hover LIFT (gravity-matched) and
the trained policy supplies the flight-control authority the mechanism lacks: a
stabilizing wrench on the airframe -- attitude torque (3) + vertical thrust trim
(1) -- as a function of the (low-passed) body state:

    u = K @ [pos_err(3), linvel(3), att_err(3), angvel(3)] + b   (4,)
    tau = tau_max*tanh(u[:3])   Fz = fz_max*tanh(u[3])

`i don't care if this transfers past the mujoco visualizer`: gravity is scaled
to the flap lift, the solver runs at iterations=1, and the stabilizer is a
virtual body wrench -- all fine for a viewer demo, none physical.

    .venv/bin/python opt/hover_policy.py --tg designs/hummer/hover/shoulder.tg \
        --iters 300 --out opt/runs_hover
    python opt/hover_policy.py --selfcheck     # CPU mapping/parity smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import NamedTuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

import flying
from compile_tg import compile_tg
from morphology.tg_parser import load_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.disk_orientation import DiskOrientation
from runtime.morphology_runtime import DEFAULT_HUMMER_TG
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.mjx import aero_runtime, rollout as mroll, wing_pitch as mwp

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


# --------------------------------------------------------------------------- #
# model build (free-flying it009p, gravity matched to lift)
# --------------------------------------------------------------------------- #
def build_free_model(tg_path: Path, scene_out: Path, *, dt_ms, gravity, aero_substep):
    with open(ROOT / "opt" / ".compile.lock", "w") as _lk:
        import fcntl, os
        fcntl.flock(_lk, fcntl.LOCK_EX)
        os.environ["MORPHOLOGY_PATH"] = str(tg_path)
        os.environ["MUJOCO_XML_PATH"] = str(scene_out)
        tmp = scene_out.with_suffix(".bench.xml")
        compile_tg(tg_path, tmp)
        xml = flying.to_airframe(tmp.read_text())
    scene_out.write_text(xml)
    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_out))
    model.opt.integrator = 0  # euler
    if dt_ms is not None:
        model.opt.timestep = dt_ms * 1e-3
    model.opt.gravity[2] = -gravity
    model.opt.iterations = 1  # reverse-mode differentiable (rollout.set_reverse_mode_solver)
    level_stroke_plane(model)  # pure vertical stroke -> hover in place, no fwd thrust

    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, morph.motors.crank_phase_lag_rad)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)
    cfg = HoverCfg(a0=float(mroll.build_drive_config(model).a0), f_max=0.0, tau_max=0.0,
                   qadr=int(model.jnt_qposadr[jid]), dofadr=int(model.jnt_dofadr[jid]))
    return model, data, morph, cfg


def measure_mean_lift(tg_path: Path, scene_bench: Path, *, ncyc=6) -> float:
    """Mean grounded-bench aero lift over ncyc flap cycles (pins the body)."""
    from runtime.aero_runtime import AeroBladeRuntime
    from runtime.wing_pitch_limit import WingPitchLimiter
    from runtime.viewer_runtime import get_actuator_id
    import os
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_bench)
    compile_tg(tg_path, scene_bench)
    morph = load_morphology(tg_path)
    m = mujoco.MjModel.from_xml_path(str(scene_bench))
    m.opt.integrator = 0
    level_stroke_plane(m)  # match the leveled config the free model flies with
    d = mujoco.MjData(m)
    calibrate_wing_pitch_at_rest(m, d)
    sync_crank_disks_at_rest(m, d, morph.motors.crank_phase_lag_rad)
    drive = mroll.build_drive_config(m)
    a0, omega = float(drive.a0), float(drive.omega)
    aero = AeroBladeRuntime(m); pitch = WingPitchLimiter(m, d)
    aids = [get_actuator_id(m, nm) for nm in ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")]
    spc = int((2 * math.pi / omega) / m.opt.timestep)
    lifts = []
    for _ in range(ncyc * spc):
        d.qfrc_applied[:] = 0.0; d.ctrl[:] = 0.0
        aero.apply(m, d); pitch.apply(m, d)
        for aid in aids:
            if aid >= 0: d.ctrl[aid] = a0
        lifts.append(sum(float(np.sum([f[2] for f in (w.cached_forces or [])])) for w in aero.wings))
        mujoco.mj_step(m, d)
    return float(np.mean(lifts[spc:]))  # drop first-cycle transient


# --------------------------------------------------------------------------- #
# differentiable rollout + loss
# --------------------------------------------------------------------------- #
def make_loss(mx, dx0, drive, pitch_cfg, aero_cfg, cfg: HoverCfg, *,
              horizon, aero_substep, com0, weights, perturbs, filter_alpha):
    flap_ids = drive.flap_actuator_ids
    a0 = float(drive.a0)
    w_pos, w_vel, w_att, w_ang, w_u = weights
    alpha = float(filter_alpha)
    dof = cfg.dofadr

    def rollout_cost(theta, perturb):
        K = theta[: N_CTRL * N_FEAT].reshape(N_CTRL, N_FEAT)
        b = theta[N_CTRL * N_FEAT :]
        qvel0 = dx0.qvel.at[dof : dof + 6].add(perturb)
        dx_init = dx0.replace(qvel=qvel0)
        aero0 = aero_runtime.init_aero_state(aero_cfg)
        ema0 = hover_features(dx_init.qpos, dx_init.qvel, cfg, com0, jnp)
        carry0 = (dx_init, aero0, ema0)

        def step(carry, _):
            dx, aero_state, ema = carry
            feats = hover_features(dx.qpos, dx.qvel, cfg, com0, jnp)
            ema = ema + alpha * (feats - ema)   # light low-pass, less noise into the gains
            force, tau, u = policy_commands(K, b, ema, cfg, jnp)
            ctrl = jnp.zeros_like(dx.ctrl).at[flap_ids].set(a0)   # constant flap -> lift
            dx = dx.replace(ctrl=ctrl, qfrc_applied=jnp.zeros_like(dx.qfrc_applied))
            dx = mwp.apply(dx, pitch_cfg)
            q, ns = aero_runtime.aero_qfrc(mx, dx, aero_cfg, aero_state)
            wrench = jnp.zeros_like(dx.qvel).at[dof : dof + 3].add(force).at[dof + 3 : dof + 6].add(tau)
            dx = dx.replace(qfrc_applied=dx.qfrc_applied + q + wrench)
            dx = mjx.step(mx, dx)
            carry = mroll._match_dtypes((dx, ns, ema), carry0)
            # Score the NET drift (ema of every channel): the flap bob averages out,
            # so the floor -> 0 for a true hover and the gradient targets station-keeping
            # (raw per-step velocity would swamp the loss with the irreducible bob).
            pe, li, at, an = ema[:3], ema[3:6], ema[6:9], ema[9:12]
            cost = (w_pos * jnp.sum(pe ** 2) + w_vel * jnp.sum(li ** 2)
                    + w_att * jnp.sum(at ** 2) + w_ang * jnp.sum(an ** 2)
                    + w_u * jnp.sum(u ** 2))
            return carry, cost

        # Gradient checkpointing: recompute mjx.step internals in backward pass so
        # reverse-mode BPTT over ~2k steps fits in 16 GB.
        _, costs = jax.lax.scan(jax.checkpoint(step), carry0, xs=None, length=horizon)
        return jnp.mean(costs)

    def loss(theta):
        return jnp.mean(jax.vmap(lambda p: rollout_cost(theta, p))(perturbs))

    return loss


def make_perturbs(batch, v_lin, v_ang, seed):
    """Random airframe velocity kicks (linvel, angvel); element 0 is the nominal (zeros)."""
    rng = np.random.default_rng(seed)
    p = np.zeros((batch, 6))
    p[1:, :3] = rng.uniform(-v_lin, v_lin, size=(batch - 1, 3))
    p[1:, 3:] = rng.uniform(-v_ang, v_ang, size=(batch - 1, 3))
    return jnp.asarray(p)


# --------------------------------------------------------------------------- #
# hand-rolled adam with global-norm gradient clipping (no optax dependency)
# --------------------------------------------------------------------------- #
def adam(loss_fn, theta0, *, iters, lr, log_every, out_dir, save_fn, grad_clip=50.0):
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    theta = theta0
    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = math.inf
    t0 = time.time()
    for it in range(1, iters + 1):
        val, g = grad_fn(theta)
        gval = float(val)
        if not math.isfinite(gval):
            print(f"[hover] iter {it}: non-finite loss ({gval}); stopping", flush=True)
            break
        gn = jnp.linalg.norm(g)
        g = jnp.where(gn > grad_clip, g * (grad_clip / (gn + 1e-12)), g)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** it)
        vhat = v / (1 - b2 ** it)
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
        if gval < best:
            best = gval
            save_fn(theta, out_dir, tag="best")
        if it % log_every == 0 or it == 1:
            print(f"[hover] iter {it:4d}/{iters}  loss {gval:10.5f}  best {best:10.5f}  "
                  f"|g|raw {float(gn):11.2f}  {(time.time()-t0)/it:5.2f}s/it", flush=True)
    save_fn(theta, out_dir, tag="final")
    return theta, best


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=DEFAULT_HUMMER_TG)
    ap.add_argument("--out", type=Path, default=ROOT / "opt" / "runs_hover")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--grad-clip", type=float, default=50.0)
    ap.add_argument("--filter-cycles", type=float, default=0.3,
                    help="EMA low-pass time constant in flap cycles")
    ap.add_argument("--cycles", type=float, default=8.0, help="BPTT horizon in flap cycles")
    ap.add_argument("--batch", type=int, default=4, help="perturbed initial conditions per grad")
    ap.add_argument("--dt-ms", type=float, default=None, help="override timestep (default: tg native)")
    ap.add_argument("--gravity", type=float, default=None, help="m/s^2 (default: match mean lift)")
    ap.add_argument("--aero-substep", type=int, default=1, help="1 = recompute aero every step (CPU-matched)")
    ap.add_argument("--tau-max", type=float, default=0.02, help="attitude torque bound [Nm]")
    ap.add_argument("--f-max", type=float, default=0.15, help="per-axis stabilizing force bound [N]")
    ap.add_argument("--pd-init", action="store_true", help="init policy from the stabilizing PD (else zeros)")
    ap.add_argument("--v-lin", type=float, default=0.12)
    ap.add_argument("--v-ang", type=float, default=0.6)
    ap.add_argument("--w-pos", type=float, default=40.0)
    ap.add_argument("--w-vel", type=float, default=2.0)
    ap.add_argument("--w-att", type=float, default=25.0)
    ap.add_argument("--w-ang", type=float, default=2.0)
    ap.add_argument("--w-u", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    args.out.mkdir(parents=True, exist_ok=True)
    scene = args.out / "hover_scene.xml"
    bench = args.out / "bench_scene.xml"

    grav = args.gravity
    if grav is None:
        model0, data0, morph0, cfg0 = build_free_model(
            args.tg, scene, dt_ms=args.dt_ms, gravity=9.81, aero_substep=args.aero_substep)
        mass = float(np.sum(model0.body_mass))
        L0 = measure_mean_lift(args.tg, bench)
        grav = L0 / mass
        print(f"[hover] mean lift {L0*1e3:.1f} mN / mass {mass*1e3:.1f} g -> gravity {grav:.3f} m/s^2", flush=True)

    model, data, morph, cfg0 = build_free_model(
        args.tg, scene, dt_ms=args.dt_ms, gravity=grav, aero_substep=args.aero_substep)
    cfg = cfg0._replace(f_max=args.f_max, tau_max=args.tau_max)

    drive = mroll.build_drive_config(model)
    pitch_cfg = mwp.config_from_runtime(model, centers=mroll._pitch_centers(model, data))
    aero_cfg = aero_runtime.build_aero_config(model, substep=args.aero_substep)
    mx = mjx.put_model(model)
    dx0 = mjx.put_data(model, data)
    com0 = jnp.asarray(np.array(data.qpos[cfg.qadr : cfg.qadr + 3]))

    omega = float(drive.omega)
    spc = (2 * math.pi / omega) / float(model.opt.timestep)
    horizon = int(args.cycles * spc)
    perturbs = make_perturbs(args.batch, args.v_lin, args.v_ang, args.seed)
    filter_alpha = 1.0 - math.exp(-1.0 / max(1.0, args.filter_cycles * spc))
    print(f"[hover] dt {model.opt.timestep*1e3:.3f}ms  spc {spc:.0f}  horizon {horizon} steps "
          f"({args.cycles} cyc)  batch {args.batch}  f_max {args.f_max} tau_max {args.tau_max}  "
          f"params {N_CTRL*N_FEAT+N_CTRL}", flush=True)

    loss = make_loss(mx, dx0, drive, pitch_cfg, aero_cfg, cfg,
                     horizon=horizon, aero_substep=args.aero_substep, com0=com0,
                     weights=(args.w_pos, args.w_vel, args.w_att, args.w_ang, args.w_u),
                     perturbs=perturbs, filter_alpha=filter_alpha)

    # Start from the hand-sized stabilizing PD, not zero: BPTT from K=0 explodes
    # (the plant tumbles -> exponential gradients); from a controller that already
    # keeps trajectories bounded, the diff-sim refines cleanly.
    theta0 = jnp.asarray(pd_init_theta(cfg)) if args.pd_init else jnp.zeros(N_CTRL * N_FEAT + N_CTRL)

    meta = dict(tg=str(args.tg), dt_ms=model.opt.timestep * 1e3, gravity=grav,
                aero_substep=args.aero_substep, a0=cfg.a0, f_max=cfg.f_max, tau_max=cfg.tau_max,
                qadr=cfg.qadr, dofadr=cfg.dofadr, com0=[float(x) for x in np.array(com0)],
                n_feat=N_FEAT, n_ctrl=N_CTRL, filter_alpha=filter_alpha)
    (args.out / "hover_meta.json").write_text(json.dumps(meta, indent=2))

    def save_fn(theta, out_dir, tag):
        K = np.array(theta[: N_CTRL * N_FEAT]).reshape(N_CTRL, N_FEAT)
        b = np.array(theta[N_CTRL * N_FEAT :])
        np.savez(out_dir / f"policy_{tag}.npz", K=K, b=b)

    l0 = float(loss(theta0))
    print(f"[hover] open-loop (K=0) loss = {l0:.5f}", flush=True)
    adam(loss, theta0, iters=args.iters, lr=args.lr, log_every=args.log_every,
         out_dir=args.out, save_fn=save_fn, grad_clip=args.grad_clip)
    print(f"[hover] done -> {args.out}", flush=True)


def selfcheck():
    """np==jnp parity of the wrench mapping + bound sanity."""
    cfg = HoverCfg(a0=65.0, f_max=0.15, tau_max=0.02, qadr=0, dofadr=0)
    K = np.arange(N_CTRL * N_FEAT).reshape(N_CTRL, N_FEAT) * 0.01
    b = np.arange(N_CTRL) * 0.1
    feats = np.linspace(-0.5, 0.5, N_FEAT)
    fn, tn, un = policy_commands(K, b, feats, cfg, np)
    fj, tj, uj = policy_commands(jnp.asarray(K), jnp.asarray(b), jnp.asarray(feats), cfg, jnp)
    assert np.allclose(fn, np.array(fj), atol=1e-6), "force np/jnp mismatch"
    assert np.allclose(tn, np.array(tj), atol=1e-6), "tau np/jnp mismatch"
    assert np.all(np.abs(fn) <= cfg.f_max + 1e-9) and np.all(np.abs(tn) <= cfg.tau_max + 1e-9), "bounds violated"
    print("hover_policy selfcheck OK: 6-DOF wrench np/jnp parity + bounds")


if __name__ == "__main__":
    main()
