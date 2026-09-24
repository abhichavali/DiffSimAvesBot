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

The controller itself (features, wrench, PD init) is runtime.hover_control; this
file is the model build and the BPTT trainer. The trainer does not work (gradients
explode through the unstable rollout, docs/flight_demos.org): the shipped policy in
designs/hummer/hover/hover_policy/ is a sweep-tuned PD.

    .venv/bin/python opt/hover_policy.py --tg designs/hummer/hover/shoulder.tg \
        --iters 300 --out opt/runs_hover
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import mujoco.mjx as mjx  # noqa: E402
import numpy as np  # noqa: E402

import flying  # noqa: E402
from compile_tg import compile_tg  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.control_signal import load_control_for_morphology  # noqa: E402
from runtime.hover_control import (N_CTRL, N_FEAT, HoverCfg, hover_features,  # noqa: E402
                                   level_stroke_plane, pd_init_theta, policy_commands)
from runtime.morphology_runtime import DEFAULT_HUMMER_TG  # noqa: E402
from runtime.sim import PlateWingSim, rest_state  # noqa: E402
from runtime.mjx import aero_runtime, rollout as mroll, wing_pitch as mwp  # noqa: E402

# --------------------------------------------------------------------------- #
# model build (free-flying it009p, gravity matched to lift)
# --------------------------------------------------------------------------- #
def build_free_model(tg_path: Path, scene_out: Path, *, dt_ms, gravity, aero_substep):
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
    model.opt.iterations = 1  # one solver iteration: no while_loop, so reverse mode can trace it
    level_stroke_plane(model)  # pure vertical stroke -> hover in place, no fwd thrust

    data = rest_state(model, morph)
    drive = mroll.build_drive_config(model, morph, load_control_for_morphology(tg_path))

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, flying.AIRFRAME_FREEJOINT)
    cfg = HoverCfg(a0=float(drive.a0), f_max=0.0, tau_max=0.0,
                   qadr=int(model.jnt_qposadr[jid]), dofadr=int(model.jnt_dofadr[jid]))
    return model, data, morph, drive, cfg


def measure_mean_lift(tg_path: Path, scene_bench: Path, *, ncyc=6) -> float:
    """Mean grounded-bench aero lift over ncyc flap cycles (pins the body)."""
    compile_tg(tg_path, scene_bench)
    morph = load_morphology(tg_path)
    m = mujoco.MjModel.from_xml_path(str(scene_bench))
    m.opt.integrator = 0
    level_stroke_plane(m)  # match the leveled config the free model flies with
    sim = PlateWingSim(m, morph)
    drive = mroll.build_drive_config(m, morph, load_control_for_morphology(tg_path))
    a0, omega = float(drive.a0), float(drive.omega)
    spc = int((2 * math.pi / omega) / m.opt.timestep)
    lifts = [sim.step(a0) for _ in range(ncyc * spc)]
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
            carry = mroll.match_dtypes((dx, ns, ema), carry0)
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
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scene = args.out / "hover_scene.xml"
    bench = args.out / "bench_scene.xml"

    grav = args.gravity
    if grav is None:
        model0, *_ = build_free_model(
            args.tg, scene, dt_ms=args.dt_ms, gravity=9.81, aero_substep=args.aero_substep)
        mass = float(np.sum(model0.body_mass))
        L0 = measure_mean_lift(args.tg, bench)
        grav = L0 / mass
        print(f"[hover] mean lift {L0*1e3:.1f} mN / mass {mass*1e3:.1f} g -> gravity {grav:.3f} m/s^2", flush=True)

    model, data, morph, drive, cfg0 = build_free_model(
        args.tg, scene, dt_ms=args.dt_ms, gravity=grav, aero_substep=args.aero_substep)
    cfg = cfg0._replace(f_max=args.f_max, tau_max=args.tau_max)

    pitch_cfg = mwp.build_wing_pitch_config(model, morph, mroll.pitch_centers(model, data))
    aero_cfg = aero_runtime.build_aero_config(model, morph, substep=args.aero_substep)
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


if __name__ == "__main__":
    main()
