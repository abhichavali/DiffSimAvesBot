"""Evolutionary search over the structural-tendon log-factors for LIGHTLY-DAMPED
BUT STABLE seed configs, to warm-start the codesign optimizer.

Fitness rewards the resonant/elastic benefit (more flap amplitude + less motor
power/torque for the SAME frozen baseline control) subject to hard stability
constraints (finite, bounded torque, lift floor, amplitude band, clean limit
cycle). Over-damped -> low fitness; too-lightly-damped -> violates a constraint;
the optimum sits in the middle. Then picks 7 high-fitness but MUTUALLY DISTANT
seeds (greedy max-min L2 in log-factor space).

Runs on the GPU box (uses the MJX rollout metrics). CPU-safe too (slower).
    uv run python opt/evolve_tendons.py --out opt/tendon_seeds
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

import optimize_control as oc

ROOT = Path(__file__).resolve().parent.parent


def build_metrics(tg, scene, aero_substep, timestep_ms):
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = oc.setup(
        tg, scene, aero_substep=aero_substep, integrator="euler", timestep_ms=timestep_ms)
    K = 5
    base_a0 = float(drive.a0); omega = float(control.omega)
    dt = model.opt.timestep; period = control.period
    steps_per_cycle = max(1, round(period / dt))
    warmup_steps = math.ceil(8.0 * period / dt)   # match the run's settle
    sample_steps = math.ceil(3.0 * period / dt)
    # Pure fundamental drive: a0 only at omega, NO higher harmonics (a_n=b_n=0).
    # control_fixed = [a0_log=0 -> a0=base_a0, zeros(K), zeros(K)].
    control_seed = jnp.zeros(2 * K + 1, dtype=jnp.float32)
    import mujoco
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    flap_t1_mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names],
                               dtype=jnp.float32)
    metrics = oc.build_rollout(
        mx, dx0, drive, pitch, aero, tip_ids, omega, K, warmup_steps, sample_steps,
        steps_per_cycle=steps_per_cycle, aero_substep=aero_substep, base_a0=base_a0,
        tune_control=False, control_fixed=control_seed, tune_frequency=False,
        tune_tendons=True, tendons=tendons, tune_pitch=False, tune_motor=False,
        motor_ids=motor_ids, flap_t1_mask=flap_t1_mask)
    n_tend = int(np.asarray(tendons[0]).shape[0])
    return jax.jit(metrics), jax.jit(jax.vmap(metrics)), 2 * n_tend


def evaluate(vmetrics, genomes, batch):
    """Return (P, 9) metrics for a population via PARALLEL vmap rollouts, in
    batches of ``batch`` to bound GPU memory (forward-only rollout is light --
    no grad tape -- so a few-wide batch fits a 10 GB card with prealloc off)."""
    outs = []
    for i in range(0, len(genomes), batch):
        g = jnp.asarray(genomes[i:i + batch], dtype=jnp.float32)
        outs.append(np.asarray(vmetrics(g)).T)  # (b, 9)
    return np.concatenate(outs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", type=Path, default=ROOT / "flight-orientations/hovering/shoulder.tg")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--aero-substep", type=int, default=2)
    p.add_argument("--timestep", type=float, default=0.06)
    p.add_argument("--pop", type=int, default=96)
    p.add_argument("--batch", type=int, default=8, help="vmap width for parallel rollouts (tune to VRAM)")
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=0.18)   # mutation std (log-space), annealed
    p.add_argument("--n-seeds", type=int, default=7)
    p.add_argument("--min-dist", type=float, default=0.5)  # diversity L2 threshold (report only)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-seed", type=Path, default=None,
                   help="warm-start the population around a found design: a JSON with "
                        "'tendon_theta' (48 log-factors) or 'pairs' (a tendon_optimized.json)")
    p.add_argument("--init-frac", type=float, default=0.7,
                   help="fraction of the initial pop centered on --init-seed (rest random)")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "tendon_seeds")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    args.pop = ((args.pop + args.batch - 1) // args.batch) * args.batch  # multiple of batch
    metrics_j, vmetrics, D = build_metrics(args.tg, args.scene, args.aero_substep, args.timestep)
    print(f"[evolve] tendon genome dim = {D}  pop={args.pop} batch={args.batch}", flush=True)

    # Baseline (1x) reference.
    b = np.asarray(metrics_j(jnp.zeros(D, dtype=jnp.float32)))
    Lb, taub, Ab, Sb, Clcb, Pinb, Ovlb, swb, Psupb, Cvb, alphab, kb, convb = [float(x) for x in b]
    print(f"[baseline] L={Lb*1e3:.1f}mN tau={taub*1e3:.1f}mNm A={Ab*1e3:.1f}mm "
          f"P={Psupb*1e3:.0f}mW Clc={Clcb:.2e} Cv={Cvb:.2e} "
          f"alpha={math.degrees(alphab):.0f}deg k={kb:.2g}", flush=True)

    LIFT_FLOOR = 0.90 * Lb
    A_LO, A_HI = 0.90 * Ab, 1.60 * Ab
    TAU_CAP = 1.30 * taub
    CLC_CAP = max(5e-3, 30 * Clcb)   # clean limit cycle

    def fitness(m):
        L, tau, A, S, Clc, Pin, Ovl, sw, Psup, Cv, alpha, k, conv = m
        if not np.all(np.isfinite(m)):
            return -1e9, False
        ok = (L >= LIFT_FLOOR and A_LO <= A <= A_HI and tau <= TAU_CAP and Clc <= CLC_CAP)
        # resonant/elastic benefit: more amplitude, less power, less peak torque.
        f = (A / Ab - 1.0) + 0.5 * (1.0 - Psup / Psupb) + 0.5 * (1.0 - tau / taub) - Cv
        return (f if ok else -1e9 + f), ok

    # Init population: broad log-factors, biased slightly stiffer (explore undamped).
    stiff = rng.uniform(-0.3, 0.9, size=(args.pop, D // 2))
    pre = rng.uniform(-0.15, 0.30, size=(args.pop, D // 2))
    pop = np.concatenate([stiff, pre], axis=1).astype(np.float32)
    if args.init_seed is not None:  # warm-start: center init_frac of the pop on a found design
        d = json.loads(args.init_seed.read_text())
        if "tendon_theta" in d:
            sg = np.asarray(d["tendon_theta"], dtype=np.float32)
        else:  # a tendon_optimized.json: pairs in l_id order -> log(factor)
            vals = list(d["pairs"].values())
            sg = np.array([math.log(v["stiffness_factor"]) for v in vals]
                          + [math.log(v["pretension_factor"]) for v in vals], dtype=np.float32)
        assert sg.shape[0] == D, f"init-seed dim {sg.shape[0]} != {D}"
        n_c = int(args.init_frac * args.pop)
        pop[:n_c] = (sg[None, :] + rng.normal(0, 0.12, (n_c, D))).astype(np.float32)
        pop[0] = sg  # keep the exact seed
        print(f"[evolve] warm-started {n_c}/{args.pop} around {args.init_seed.name} "
              f"(stiff~{np.exp(sg[:D//2]).mean():.2f}x pre~{np.exp(sg[D//2:]).mean():.2f}x)", flush=True)

    archive_g, archive_f, archive_m = [], [], []
    n_elite = max(2, int(args.elite_frac * args.pop))
    for gen in range(args.gens):
        M = evaluate(vmetrics, pop, args.batch)
        fits = np.array([fitness(m)[0] for m in M])
        oks = np.array([fitness(m)[1] for m in M])
        for g, f, m, ok in zip(pop, fits, M, oks):
            if ok:
                archive_g.append(g.copy()); archive_f.append(f); archive_m.append(m)
        order = np.argsort(fits)[::-1]
        elites = pop[order[:n_elite]]
        best = M[order[0]]
        print(f"[gen {gen:02d}] best_fit={fits[order[0]]:.3f} feas={int(oks.sum())}/{args.pop} "
              f"| A={best[2]*1e3:.1f}mm tau={best[1]*1e3:.1f}mNm P={best[8]*1e3:.0f}mW "
              f"L={best[0]*1e3:.1f}mN | archive={len(archive_g)}", flush=True)
        # Checkpoint the running best genome so a restart can --init-seed from it.
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "best_genome.json").write_text(json.dumps(
            {"gen": gen, "fitness": float(fits[order[0]]),
             "tendon_theta": [float(x) for x in pop[order[0]]]}))
        # Reproduce: elites + mutated offspring from tournament winners.
        sigma = args.sigma * (1.0 - 0.6 * gen / args.gens)   # anneal
        children = [e.copy() for e in elites]
        while len(children) < args.pop:
            i, j = rng.integers(0, args.pop, 2)
            parent = pop[i] if fits[i] >= fits[j] else pop[j]
            child = parent + rng.normal(0, sigma, size=D).astype(np.float32)
            child[:D//2] = np.clip(child[:D//2], -0.6, 1.2)   # stiffness log bounds
            child[D//2:] = np.clip(child[D//2:], -0.4, 0.5)   # pretension log bounds
            children.append(child)
        pop = np.array(children[:args.pop], dtype=np.float32)

    if not archive_g:
        print("[evolve] NO feasible designs found -- loosen constraints.", flush=True); return
    archive_g = np.array(archive_g); archive_f = np.array(archive_f)
    archive_m = np.array(archive_m)
    # Dedup near-identical genomes.
    keep = []
    for idx in np.argsort(archive_f)[::-1]:
        if all(np.linalg.norm(archive_g[idx] - archive_g[k]) > 0.15 for k in keep):
            keep.append(idx)
    keep = np.array(keep)
    print(f"[archive] {len(keep)} distinct feasible designs", flush=True)

    # Greedy max-min diversity among the top-fitness half.
    top = keep[archive_f[keep] >= np.median(archive_f[keep])]
    chosen = [top[np.argmax(archive_f[top])]]
    while len(chosen) < min(args.n_seeds, len(top)):
        d = np.array([min(np.linalg.norm(archive_g[c] - archive_g[t]) for c in chosen) for t in top])
        d[[np.where(top == c)[0][0] for c in chosen]] = -1
        chosen.append(top[int(np.argmax(d))])

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for si, idx in enumerate(chosen, 1):
        g = archive_g[idx]; m = archive_m[idx]
        rec = {"seed": si, "tendon_theta": [float(x) for x in g],
               "fitness": float(archive_f[idx]),
               "Lbar": float(m[0]), "tau_hat": float(m[1]), "A_stroke": float(m[2]),
               "C_lc": float(m[4]), "P_supplied": float(m[8]),
               "mean_stiffness_x": float(np.exp(g[:D//2]).mean()),
               "mean_pretension_x": float(np.exp(g[D//2:]).mean())}
        (args.out / f"seed_{si}.json").write_text(json.dumps(rec, indent=2))
        summary.append(rec)
        dmin = min([np.linalg.norm(g - archive_g[c]) for c in chosen if c != idx] or [0])
        print(f"[seed {si}] fit={rec['fitness']:.3f} A={m[2]*1e3:.1f}mm tau={m[1]*1e3:.1f}mNm "
              f"L={m[0]*1e3:.1f}mN stiff~{rec['mean_stiffness_x']:.2f}x "
              f"pre~{rec['mean_pretension_x']:.2f}x  min_dist={dmin:.2f}", flush=True)
    (args.out / "seeds_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(chosen)} seeds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
