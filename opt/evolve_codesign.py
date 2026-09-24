"""Evolutionary CODESIGN of Fourier control + structural tendons + motor tendon.

Why evolutionary: gradient descent reliably finds NUMERICAL artifacts (a design
whose peak torque dips at exactly the training timestep, then rebounds at any
other dt -- verified: a "160 mNm" design that is ~194/193/209 at dt=0.13/0.03/
0.015). The exploit is literally the smoothest descent direction, so the gradient
walks straight into it. A gradient-free search can't slide into a razor-thin hole;
mutation-survival favors BROAD basins, which correlates with physical robustness.

To attack the timestep axis directly, fitness is evaluated at TWO timesteps
(default 0.06 + 0.03 ms): a genome must be feasible at BOTH, is scored on the
WORSE peak torque, and pays a penalty for the cross-timestep torque spread. So a
design that only looks good at one dt is culled. (The baseline is dt-invariant,
~225 mNm at every dt, so this is a fair, well-posed target.)

Genome = [control(2K+1)] [tendon stiff(n) + pre(n)] [motor(2)] [geometry(G)]

The first three blocks are log-factors on MJX model arrays: one compiled model
serves the whole population, so they are nearly free. The optional geometry
block (--geom-genes) holds .tg scalars that are baked into the MJCF -- strut
dimensions, strut arrangement, placement, motor disk geometry -- so each genome
gets its own compiled model (~0.2 s). Changing those values never changes an
array SHAPE (verified: 124 mx leaves, identical shapes), so the batched rollout
still traces once and vmaps over the stacked models.

    uv run python opt/evolve_codesign.py --out opt/codesign_seeds --gens 40 --pop 96
    uv run python opt/evolve_codesign.py --geom-genes placement.mount_height,arrange.gap
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tg_scalars  # noqa: E402
from codesign.artifacts import fourier_control_tg, motor_json, repoint_control, tendon_pairs_json  # noqa: E402
from codesign.genome import Genome  # noqa: E402
from codesign.metrics import Metrics, build_rollout  # noqa: E402
from codesign.rig import flap_t1_mask, setup, step_counts  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.morphology_runtime import HOVER_BASE_TG  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def geom_value(base: float, g: float, unit: str) -> float:
    """Decode one geometry gene into a .tg value.

    Log-factor around the baseline, so a single mutation sigma means the same
    *relative* move for a 258 mm mount height and a 4 mm rod radius, and the
    sign can never flip. Params whose baseline is exactly 0 (`tilt: 0.0 rad`,
    `angle = 0 deg`, `disk_dx: 0 mm`) have no scale to be relative to, so they
    move additively against a per-unit reference instead.
    """
    if abs(base) > 1e-12:
        return base * math.exp(g)
    return g * {"rad": 0.1, "deg": 5.0, "mm": 5.0}.get(unit, 0.1)


def geom_tg(tg_src: Path, dst: Path, names, gvals) -> Path:
    """Write the .tg variant a geometry genome describes."""
    params = tg_scalars.parse_tg(tg_src)
    by_name = {p.name: p for p in params}
    updates = {n: geom_value(by_name[n].number, float(v), by_name[n].unit)
               for n, v in zip(names, gvals)}
    dst.write_text(tg_scalars.substitute(tg_src.read_text(), params, updates))
    return dst


def build_metrics(tg, scene, aero_substep, timestep_ms, K=5, tune_motor=True, n_geom=0):
    """Return (vmetrics, metrics, info) for one timestep. info carries the base
    arrays needed to materialize a genome into a design.

    With geometry genes the caller also supplies a per-genome (mx, dx0), so the
    vmap runs over three axes instead of one.

    ponytail: `drive`, `pitch` and `aero` stay at the BASE design's values even
    when geometry moves. They are wing-side and drive-side configs (actuator
    ids, wing chords, pitch rest centers) and none of the exposed geometry genes
    touch the wing -- exposing wing.* as genes would mean stacking these too.
    """
    rig = setup(tg, scene, aero_substep=aero_substep, integrator="euler", timestep_ms=timestep_ms)
    omega = float(rig.control.omega)
    steps_per_cycle, warmup_steps, sample_steps = step_counts(
        rig.control.period, rig.model.opt.timestep, 8.0, 3.0)
    genome = Genome(K, int(np.asarray(rig.tendons[0]).shape[0]), tendons=True, motor=tune_motor)
    metrics = build_rollout(
        rig, genome, omega, warmup_steps=warmup_steps, sample_steps=sample_steps,
        steps_per_cycle=steps_per_cycle, aero_substep=aero_substep,
        control_fixed=jnp.zeros(2 * K + 1, dtype=jnp.float32),
        flap_t1_mask=flap_t1_mask(rig.model, rig.drive))
    D = genome.size + n_geom
    info = dict(rig=rig, genome=genome, omega=omega, n_geom=n_geom)
    vmetrics = jax.jit(jax.vmap(metrics) if not n_geom
                       else jax.vmap(metrics, in_axes=(0, 0, 0)))
    return vmetrics, jax.jit(metrics), D, info


def stack_models(mds):
    """Stack per-genome models into one batched pytree for vmap.

    A plain tree_map refuses: MJX keeps topology arrays as STATIC pytree
    metadata compared by identity, and separate compiles produce distinct
    objects. Of the 163 static fields, exactly one differs by value across a
    geometry change -- `_impl.geom_rbound_hfield`, the collision broad-phase
    radius -- so the rest can safely come from the first model. That one is
    taken as the elementwise MAX: a larger bound only widens the candidate
    pair set, so the batch can never miss a contact one of its genomes has.
    """
    ref = mds[0]
    impl = getattr(ref, "_impl", None)
    if impl is not None and hasattr(impl, "geom_rbound_hfield"):
        rb = np.max([np.asarray(m._impl.geom_rbound_hfield) for m in mds], axis=0)
        ref = ref.replace(_impl=impl.replace(geom_rbound_hfield=rb))
    leaves = [jax.tree_util.tree_leaves(m) for m in mds]
    widths = {len(x) for x in leaves}
    assert len(widths) == 1, f"models have different leaf counts: {widths}"
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(ref),
                                        [jnp.stack(x) for x in zip(*leaves)])


def evaluate(vmetrics, genomes, batch, label="eval", models_for=None):
    """models_for(rows) -> (mx_batch, dx_batch) when geometry genes are active."""
    outs = []
    total = (len(genomes) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(genomes), batch), 1):
        t0 = time.time()
        rows = genomes[i:i + batch]
        log(f"[{label}] batch {bi}/{total} rows {i}:{min(i + batch, len(genomes))} start")
        g = jnp.asarray(rows, dtype=jnp.float32)
        args = (g,) if models_for is None else (g, *models_for(rows))
        outs.append(np.asarray(vmetrics(*args)).T)  # (b, len(Metrics._fields))
        log(f"[{label}] batch {bi}/{total} done in {time.time() - t0:.1f}s")
    return np.concatenate(outs, axis=0)


def materialize(g, info, out_dir, tg_src, geom_names=()):
    """Write a genome as a usable design dir (shoulder.tg + control.tg + tendon/
    motor JSON + theta_best.json), in optimize_control's artifact layout.

    With geometry genes the emitted shoulder.tg carries the EVOLVED dimensions,
    not the source ones -- otherwise the winner's shape would be lost and the
    design dir would silently describe a different rig than the one scored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rig, genome = info["rig"], info["genome"]
    K = genome.K
    genes = genome.split(np.asarray(g), None)
    a0 = float(rig.drive.a0) * math.exp(float(genes.control[0]))
    a_out = a0 * genes.control[1:1 + K]; b_out = a0 * genes.control[1 + K:]
    (out_dir / "tendon_optimized.json").write_text(json.dumps(
        {"pairs": tendon_pairs_json(rig.model, rig.tendons, genes.stiff, genes.pre)}, indent=2))
    if genome.motor:
        (out_dir / "motor_tendon_optimized.json").write_text(json.dumps(
            motor_json(rig.tendons, rig.motor_ids, genes.motor), indent=2))
    control = fourier_control_tg(load_morphology(Path(tg_src)).bird, K, info["omega"] / math.tau,
                                 a0, a_out, b_out)
    (out_dir / "control.tg").write_text(control)
    (out_dir / "control_optimized.tg").write_text(control)
    src_text = Path(tg_src).read_text()
    geom_out = {}
    if geom_names:
        params = tg_scalars.parse_tg(Path(tg_src))
        by_name = {p_.name: p_ for p_ in params}
        geom_out = {nm: geom_value(by_name[nm].number, float(v), by_name[nm].unit)
                    for nm, v in zip(geom_names, np.asarray(g)[-len(geom_names):])}
        src_text = tg_scalars.substitute(src_text, params, geom_out)
        (out_dir / "geometry_optimized.json").write_text(json.dumps(
            {nm: {"value": v, "unit": by_name[nm].unit, "base": by_name[nm].number}
             for nm, v in geom_out.items()}, indent=2))
    (out_dir / "shoulder.tg").write_text(repoint_control(src_text))
    (out_dir / "theta_best.json").write_text(json.dumps(
        {"a0": float(a0), "a": [float(x) for x in a_out], "b": [float(x) for x in b_out],
         "omega": float(info["omega"]), "frequency_hz": float(info["omega"] / math.tau)}, indent=2))
    (out_dir / "codesign_seed.json").write_text(json.dumps({"genome": [float(x) for x in g]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", type=Path, default=HOVER_BASE_TG)
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--aero-substep", type=int, default=2)
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.06, 0.03],
                   help="fitness is evaluated at ALL these dts; scored on the worst + spread penalty")
    p.add_argument("--pop", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=0.06)
    p.add_argument("--w-dt", type=float, default=1.0, help="cross-timestep torque-spread penalty weight")
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--init-genome", type=Path, default=None,
                   help="center the initial population on this genome instead of the baseline "
                        "(continuation runs: pass the previous EA's best_genome.json)")
    p.add_argument("--init-spread", type=float, default=1.0,
                   help="multiplier on ALL init-population noise scales (large sweeping "
                        "searches from a far-from-baseline center want 3-6)")
    p.add_argument("--dump-every", type=int, default=0,
                   help="every N gens, materialize the top --dump-k diverse archive genomes "
                        "into <out>/progressive/ so Adam can start on them mid-run (0=off)")
    p.add_argument("--dump-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--genes", default="control,tendon_stiffness,tendon_pretension,motor",
                   help="comma list of genome BLOCKS the EA may mutate; frozen blocks stay at "
                        "the baseline (or --init-genome) value for the whole run. Blocks: "
                        "control (a0 + 2K harmonics), tendon_stiffness (n), "
                        "tendon_pretension (n), motor (2)")
    p.add_argument("--geom-genes", default="",
                   help="comma list of .tg SCALARS to evolve as geometry genes, dotted "
                        "(e.g. placement.mount_height,arrange.gap,motors.disk.crank_throw). "
                        "These are compiled into the MJCF, so each genome gets its own "
                        "model (~0.2 s to compile); array shapes are unchanged so the "
                        "batched rollout still vmaps. Empty = the fast single-model path")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "codesign_seeds")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    args.pop = ((args.pop + args.batch - 1) // args.batch) * args.batch
    # Validate --genes names before the (multi-minute) model build; the slice
    # bounds still need K/n, so the mask itself is assembled after setup.
    want = [s.strip() for s in args.genes.split(",") if s.strip()]
    GENE_NAMES = ("control", "tendon_stiffness", "tendon_pretension", "motor")
    if unknown := sorted(set(want) - set(GENE_NAMES)):
        p.error(f"--genes: unknown block(s) {unknown}; pick from {sorted(GENE_NAMES)}")
    geom_names = [s.strip() for s in args.geom_genes.split(",") if s.strip()]
    if not (want or geom_names):
        p.error("--genes/--geom-genes: nothing left to evolve")
    if geom_names:  # fail before the model build if a name isn't a scalar in this .tg
        scalars = {p_.name: p_ for p_ in tg_scalars.parse_tg(args.tg)}
        if miss := sorted(set(geom_names) - set(scalars)):
            p.error(f"--geom-genes: not scalars in {args.tg.name}: {miss}")
        if bad := sorted(n for n in geom_names if not scalars[n].tunable):
            p.error(f"--geom-genes: not bare numbers, cannot evolve: {bad}")

    n_geom = len(geom_names)
    # One rollout per timestep (same genome layout); scene path per-dt to avoid clobber.
    evs, D, info = [], None, None
    for dt in args.timesteps:
        t0 = time.time()
        log(f"[setup] building metrics dt={dt:g}ms")
        scene_dt = args.scene.with_name(f"scene_dt{int(dt*1000)}.xml")
        vm, mj, D, info = build_metrics(args.tg, scene_dt, args.aero_substep, dt,
                                        n_geom=n_geom)
        evs.append(vm)
        log(f"[setup] built metrics dt={dt:g}ms in {time.time() - t0:.1f}s")
    genome = info["genome"]
    K, n = genome.K, genome.n_tend
    n_motor = 2 if genome.motor else 0
    print(f"[codesign] genome dim={D} (control {2*K+1} + tendon {2*n} + motor {n_motor}"
          f" + geometry {n_geom})  pop={args.pop} dts={args.timesteps}", flush=True)

    # Per-genome models for the geometry block. The .tg variant must sit beside
    # its source (the `control =` include resolves relative to it), and batches
    # are built one at a time, so one scratch name is enough.
    geom_lo = genome.size

    def model_of(row, dt):
        """Compile ONE genome's geometry into its own (mx, dx0)."""
        tgv = geom_tg(args.tg, args.tg.with_name("_ea_genome.tg"),
                      geom_names, np.asarray(row)[geom_lo:])
        rig = setup(tgv, args.scene.with_name("scene_ea_genome.xml"),
                    aero_substep=args.aero_substep, integrator="euler", timestep_ms=dt)
        return rig.mx, rig.dx0

    def models_for(rows, dt):
        mds = [model_of(r, dt) for r in rows]
        return stack_models([m[0] for m in mds]), stack_models([m[1] for m in mds])

    def evs_for(t):
        """(vmetrics, models_for) for timestep index t."""
        if not n_geom:
            return evs[t], None
        return evs[t], lambda rows: models_for(rows, args.timesteps[t])

    # Gene selection: a 0/1 mask over the genome, applied to BOTH the initial
    # population noise and every mutation, so a frozen block never leaves its
    # seed value (the EA still evaluates it -- it just isn't searched).
    gene_slices = genome.slices()
    GENE_BLOCKS = {"control": gene_slices["control"], "tendon_stiffness": gene_slices["stiff"],
                   "tendon_pretension": gene_slices["pre"]}
    if genome.motor:
        GENE_BLOCKS["motor"] = gene_slices["motor"]
    gene_mask = np.zeros(D, np.float32)
    for name in want:
        if name not in GENE_BLOCKS:  # only reachable for "motor" with tune_motor off
            p.error(f"--genes: block '{name}' is not part of this genome")
        gene_mask[GENE_BLOCKS[name]] = 1.0
    gene_mask[geom_lo:] = 1.0  # every geometry gene named on the CLI is searched
    print(f"[codesign] genes={want} geometry={geom_names} -> "
          f"{int(gene_mask.sum())}/{D} dims searched", flush=True)

    # Baseline (genome 0); baseline peak torque is dt-invariant, so evs[0] is fine for norm.
    _ev0, _mf0 = evs_for(0)
    base = evaluate(_ev0, np.zeros((1, D), np.float32), 1,
                    label=f"baseline dt={args.timesteps[0]:g}ms", models_for=_mf0)[0]
    b0 = Metrics(*(float(x) for x in base))
    Lb, taub, Ab, Clcb, Psupb, Cvb, alphab, kb = (b0.Lbar, b0.tau_hat, b0.A_stroke, b0.C_lc,
                                                  b0.P_supplied, b0.C_valid, b0.alpha_pk, b0.k_eff)
    print(f"[baseline] L={Lb*1e3:.1f}mN tau={taub*1e3:.1f}mNm A={Ab*1e3:.1f}mm "
          f"P={Psupb*1e3:.0f}mW Clc={Clcb:.2e} Cv={Cvb:.2e} "
          f"alpha={math.degrees(alphab):.0f}deg k={kb:.2g}", flush=True)
    LIFT_FLOOR, A_LO, A_HI = 0.90 * Lb, 0.90 * Ab, 1.60 * Ab
    TAU_CAP, CLC_CAP = 1.30 * taub, max(5e-3, 30 * Clcb)
    # C_valid thresholds calibrated to the BASELINE operating point, not textbook
    # quasi-steady limits: a hovering flapper lives at alpha~77deg, k~0.33, so the
    # build_rollout defaults (45deg/0.10) score C_valid~5 for EVERY genome incl. the
    # baseline and flatten the fitness landscape (observed: elite frozen 30 gens).
    # The term's job is only to block going FURTHER into invalidity (the lift hack).
    ALPHA_STALL, K_MAX = 1.05 * alphab, 1.15 * kb
    print(f"[calib] C_valid thresholds: alpha_stall={math.degrees(ALPHA_STALL):.1f}deg "
          f"k_max={K_MAX:.3f} (baseline alpha={math.degrees(alphab):.1f}deg k={kb:.3f})", flush=True)

    def fitness_from(mlist):
        """mlist: metric vectors, one per timestep. Soft-penalize violations.

        Hard rejection left the EA selecting the baseline foothold forever. The
        EA's job is basin discovery; robust Adam/CPU checks enforce transfer.
        """
        if not all(np.all(np.isfinite(m)) for m in mlist):
            return -1e9, False
        mlist = [Metrics(*m) for m in mlist]
        feas = all((m.Lbar >= LIFT_FLOOR and A_LO <= m.A_stroke <= A_HI and m.tau_hat <= TAU_CAP
                    and m.C_lc <= CLC_CAP) for m in mlist)
        taus = np.array([m.tau_hat for m in mlist])
        tau_worst = taus.max(); tau_spread = taus.max() - taus.min()
        A = np.mean([m.A_stroke for m in mlist]); Psup = np.mean([m.P_supplied for m in mlist])
        swing = np.mean([m.swing for m in mlist])
        # Recompute C_valid from raw alpha_pk/k_eff against the baseline-calibrated
        # thresholds; Metrics.C_valid uses the miscalibrated defaults.
        a_pk = np.mean([m.alpha_pk for m in mlist]); k_e = np.mean([m.k_eff for m in mlist])
        Cvalid = (max(0.0, a_pk - ALPHA_STALL) ** 2 / ALPHA_STALL ** 2
                  + max(0.0, k_e - K_MAX) ** 2 / K_MAX ** 2)
        lift_short = np.mean([max(0.0, LIFT_FLOOR - m.Lbar) / max(LIFT_FLOOR, 1e-9) for m in mlist])
        amp_bad = np.mean([max(0.0, A_LO - m.A_stroke, m.A_stroke - A_HI) / max(Ab, 1e-9) for m in mlist])
        tau_bad = max(0.0, tau_worst - TAU_CAP) / max(taub, 1e-9)
        clc_bad = np.mean([max(0.0, m.C_lc - CLC_CAP) / max(CLC_CAP, 1e-9) for m in mlist])
        f = ((A / Ab - 1.0) + 0.5 * (1.0 - Psup / Psupb) + 0.5 * (1.0 - tau_worst / taub)
             - 0.5 * swing / taub - Cvalid - args.w_dt * tau_spread / taub
             - 4.0 * lift_short - 2.0 * amp_bad - 3.0 * tau_bad - 0.5 * clc_bad)
        return f, feas

    def init_block():
        # Perturb around zero = the feasible baseline (L=385/tau=225). Wide random
        # tendon/control init blows past the torque cap with negative lift (all
        # infeasible -> nothing to select on). Row 0 is the exact baseline seed.
        # With --init-genome, the center (and foothold) is that genome instead.
        center = np.zeros(D, np.float32)
        if args.init_genome is not None:
            center = np.asarray(json.loads(args.init_genome.read_text())["genome"], np.float32)
        sp = args.init_spread
        ctrl = np.concatenate([rng.normal(0, 0.03 * sp, (args.pop, 1)),          # a0 log
                               rng.normal(0, 0.05 * sp, (args.pop, 2 * K))], 1)   # harmonics
        stiff = rng.normal(0, 0.06 * sp, (args.pop, n)); pre = rng.normal(0, 0.04 * sp, (args.pop, n))
        blocks = [ctrl, stiff, pre]
        if genome.motor:
            blocks.append(rng.normal(0, 0.04 * sp, (args.pop, 2)))
        if n_geom:
            # Geometry starts tight: these are log-factors on real dimensions, and
            # a wide init walks struts/mounts into configurations that fail to
            # compile or start the rig in self-collision.
            blocks.append(rng.normal(0, 0.03 * sp, (args.pop, n_geom)))
        g = (center + gene_mask * np.concatenate(blocks, 1)).astype(np.float32)
        g[0] = center  # foothold
        return g

    def clip(pop):
        # Wide box: per-tendon stiffness may land anywhere in ~[35, 800] N/m (base 200)
        # so a soft-start (50 N/m) population can keep some tendons loose while others
        # stiffen several-fold toward resonance.
        pop[:, 0] = np.clip(pop[:, 0], -0.5, 0.5)                       # a0 log
        pop[:, 1:2 * K + 1] = np.clip(pop[:, 1:2 * K + 1], -0.8, 0.8)   # harmonics depth
        for name, lo, hi in (("stiff", -1.8, 1.4), ("pre", -0.6, 0.6), ("motor", -0.6, 0.6)):
            if name in gene_slices:
                pop[:, gene_slices[name]] = np.clip(pop[:, gene_slices[name]], lo, hi)
        if n_geom:
            # +-0.4 log = a 0.67x..1.49x span on each dimension. Wider than that
            # and the compiled rig stops resembling the design it started from.
            pop[:, geom_lo:] = np.clip(pop[:, geom_lo:], -0.4, 0.4)
        return pop

    pop = clip(init_block())
    arch_g, arch_f = [], []
    n_elite = max(2, int(args.elite_frac * args.pop))
    for gen in range(args.gens):
        log(f"[gen {gen:02d}] start pop={args.pop} batch={args.batch}")
        Ms = []                                      # per-dt metrics (P, n_metrics)
        for t in range(len(evs)):
            vm_t, mf_t = evs_for(t)
            Ms.append(evaluate(vm_t, pop, args.batch, models_for=mf_t,
                               label=f"gen {gen:02d} dt={args.timesteps[t]:g}ms"))
        res = [fitness_from([Ms[t][k] for t in range(len(evs))]) for k in range(args.pop)]
        fits = np.array([r[0] for r in res]); oks = np.array([r[1] for r in res])
        for k in range(args.pop):
            if np.isfinite(fits[k]):
                arch_g.append(pop[k].copy()); arch_f.append(fits[k])
        order = np.argsort(fits)[::-1]
        M0 = Metrics(*Ms[0][order[0]])
        taus0 = [Metrics(*Ms[t][order[0]]).tau_hat for t in range(len(evs))]
        print(f"[gen {gen:02d}] best_fit={fits[order[0]]:.3f} feas={int(oks.sum())}/{args.pop} | "
              f"tau/dt={[round(x*1e3) for x in taus0]}mNm A={M0.A_stroke*1e3:.0f}mm L={M0.Lbar*1e3:.0f}mN "
              f"alpha={math.degrees(M0.alpha_pk):.0f}deg k={M0.k_eff:.2f} | archive={len(arch_g)}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "best_genome.json").write_text(json.dumps(
            {"gen": gen, "fitness": float(fits[order[0]]), "genome": [float(x) for x in pop[order[0]]]}))
        if args.dump_every and gen % args.dump_every == args.dump_every - 1 and arch_g:
            ag = np.array(arch_g); af = np.array(arch_f)
            keep = []
            for idx in np.argsort(af)[::-1]:
                if len(keep) >= args.dump_k:
                    break
                if all(np.linalg.norm(ag[idx] - ag[k]) > 0.20 for k in keep):
                    keep.append(idx)
            for r, idx in enumerate(keep, 1):
                d = args.out / "progressive" / f"gen{gen:02d}_s{r}"
                if not (d / "codesign_seed.json").exists():
                    materialize(ag[idx], info, d, args.tg, geom_names)
            print(f"[gen {gen:02d}] progressive dump: {len(keep)} seeds -> {args.out}/progressive", flush=True)
        sigma = args.sigma * (1.0 - 0.6 * gen / args.gens)
        children = [pop[i].copy() for i in order[:n_elite]]
        while len(children) < args.pop:
            i, j = rng.integers(0, args.pop, 2)
            parent = pop[i] if fits[i] >= fits[j] else pop[j]
            # Mutate a ~25% gene subset, not all 61 dims at once: full-vector Gaussian
            # gives ||delta||~sqrt(D)*sigma~0.5 log-space, which ALWAYS breaks the
            # lift/amp/Clc constraints near the sharp baseline optimum (observed:
            # 0 new feasible children in 10 gens x 512).
            mask = (rng.random(D) < 0.25).astype(np.float32) * gene_mask
            children.append(parent + mask * rng.normal(0, sigma, D).astype(np.float32))
        pop = clip(np.array(children[:args.pop], np.float32))

    if not arch_g:
        print("[codesign] NO feasible designs -- loosen constraints or add gens.", flush=True); return
    arch_g = np.array(arch_g); arch_f = np.array(arch_f)
    keep = []
    for idx in np.argsort(arch_f)[::-1]:
        if all(np.linalg.norm(arch_g[idx] - arch_g[k]) > 0.20 for k in keep):
            keep.append(idx)
    keep = np.array(keep)
    top = keep[arch_f[keep] >= np.percentile(arch_f[keep], 60)]
    if len(top) < min(args.n_seeds, len(keep)):
        top = keep
    chosen = [top[np.argmax(arch_f[top])]]
    while len(chosen) < min(args.n_seeds, len(top)):
        d = np.array([min(np.linalg.norm(arch_g[c] - arch_g[t]) for c in chosen) for t in top])
        d[[np.where(top == c)[0][0] for c in chosen]] = -1
        chosen.append(top[int(np.argmax(d))])

    # dt-transfer acid test: peak torque of each winner across a WIDE dt sweep.
    sweep_dts = [0.13, 0.06, 0.03, 0.015]
    sweep = [(d, build_metrics(args.tg, args.scene.with_name(f"scene_sw{int(d*1000)}.xml"),
                               args.aero_substep, d, n_geom=n_geom)[1]) for d in sweep_dts]
    summary = []
    for si, idx in enumerate(chosen, 1):
        g = arch_g[idx]
        gj = jnp.asarray(g, jnp.float32)
        if n_geom:  # each sweep dt needs this genome's own compiled model
            taus = [float(mj(gj, *model_of(g, d)).tau_hat) for d, mj in sweep]
        else:
            taus = [float(mj(gj).tau_hat) for _, mj in sweep]
        spread = (max(taus) - min(taus)) / max(taus)
        materialize(g, info, args.out / f"seed_{si}", args.tg, geom_names)
        rec = {"seed": si, "fitness": float(arch_f[idx]),
               "tau_by_dt_mNm": {str(d): round(t * 1e3, 1) for d, t in zip(sweep_dts, taus)},
               "tau_spread_frac": round(spread, 3),
               "genome": [float(x) for x in g]}
        (args.out / f"seed_{si}" / "dt_transfer.json").write_text(json.dumps(rec, indent=2))
        summary.append(rec)
        print(f"[seed {si}] fit={arch_f[idx]:.3f}  tau across dt {sweep_dts} = "
              f"{[round(t*1e3) for t in taus]} mNm  spread={spread*100:.0f}%", flush=True)
    (args.out / "seeds_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(chosen)} codesign seeds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
