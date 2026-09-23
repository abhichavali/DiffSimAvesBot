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
import argparse, json, math, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import mujoco

import optimize_control as oc
import tg_scalars

ROOT = Path(__file__).resolve().parent.parent


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
    model, mx, dx0, control, drive, pitch, aero, tip_ids, tendons, motor_ids = oc.setup(
        tg, scene, aero_substep=aero_substep, integrator="euler", timestep_ms=timestep_ms)
    base_a0 = float(drive.a0); omega = float(control.omega)
    dt = model.opt.timestep; period = control.period
    steps_per_cycle = max(1, round(period / dt))
    warmup_steps = math.ceil(8.0 * period / dt)
    sample_steps = math.ceil(3.0 * period / dt)
    flap_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
                  for i in np.asarray(drive.flap_actuator_ids)]
    flap_t1_mask = jnp.asarray([1.0 if (n and "t1" in n) else 0.0 for n in flap_names],
                               dtype=jnp.float32)
    metrics = oc.build_rollout(
        mx, dx0, drive, pitch, aero, tip_ids, omega, K, warmup_steps, sample_steps,
        steps_per_cycle=steps_per_cycle, aero_substep=aero_substep, base_a0=base_a0,
        tune_control=True, control_fixed=jnp.zeros(2 * K + 1, dtype=jnp.float32),
        tune_frequency=False, tune_tendons=True, tendons=tendons, tune_pitch=False,
        tune_motor=tune_motor, motor_ids=motor_ids, flap_t1_mask=flap_t1_mask)
    l_ids = np.asarray(tendons[0]); n_tend = int(l_ids.shape[0])
    tnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(i)) for i in l_ids]
    D = (2 * K + 1) + 2 * n_tend + (2 if tune_motor else 0) + n_geom
    info = dict(base_a0=base_a0, omega=omega, K=K, n_tend=n_tend, tnames=tnames,
                base_stiff=np.asarray(tendons[2]), base_lenspring=np.asarray(tendons[3]),
                l_ids=l_ids, motor_ids=np.asarray(motor_ids), tune_motor=tune_motor,
                n_geom=n_geom)
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
        outs.append(np.asarray(vmetrics(*args)).T)  # (b, n_metrics)
        log(f"[{label}] batch {bi}/{total} done in {time.time() - t0:.1f}s")
    return np.concatenate(outs, axis=0)


def materialize(g, info, out_dir, tg_src, geom_names=()):
    """Write a genome as a usable design dir (shoulder.tg + control.tg + tendon/
    motor JSON + theta_best.json), mirroring optimize_control.dump().

    With geometry genes the emitted shoulder.tg carries the EVOLVED dimensions,
    not the source ones -- otherwise the winner's shape would be lost and the
    design dir would silently describe a different rig than the one scored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    K, n = info["K"], info["n_tend"]
    a0 = info["base_a0"] * math.exp(float(g[0]))
    a_out = a0 * np.asarray(g[1:1 + K]); b_out = a0 * np.asarray(g[1 + K:1 + 2 * K])
    i = 2 * K + 1
    stiff_log = np.asarray(g[i:i + n]); pre_log = np.asarray(g[i + n:i + 2 * n]); i += 2 * n
    bs, bl, l_ids, tn = info["base_stiff"], info["base_lenspring"], info["l_ids"], info["tnames"]
    pairs = {tn[k]: {"stiffness": float(bs[l_ids[k]] * math.exp(stiff_log[k])),
                     "stiffness_factor": float(math.exp(stiff_log[k])),
                     "pretension_factor": float(math.exp(pre_log[k])),
                     "rest_length": float(bl[l_ids[k]][0] * math.exp(pre_log[k]))}
             for k in range(n)}
    (out_dir / "tendon_optimized.json").write_text(json.dumps({"pairs": pairs}, indent=2))
    if info["tune_motor"]:
        ml = np.asarray(g[i:i + 2]); mids = info["motor_ids"]
        (out_dir / "motor_tendon_optimized.json").write_text(json.dumps(
            {"stiffness": float(bs[mids[0]] * math.exp(ml[0])),
             "stiffness_factor": float(math.exp(ml[0])),
             "pretension_factor": float(math.exp(ml[1]))}, indent=2))
    lines = [f"bird = {oc.load_morphology(Path(tg_src)).bird}",
             f"harmonics = {K}", f"fundamental_frequency = {info['omega']/math.tau:g} Hz",
             f"a0 = {a0:.6g}"]
    for j in range(K):
        lines += [f"a_{j+1} = {a_out[j]:.6g}", f"b_{j+1} = {b_out[j]:.6g}"]
    (out_dir / "control.tg").write_text("\n".join(lines) + "\n")
    (out_dir / "control_optimized.tg").write_text("\n".join(lines) + "\n")
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
    shoulder = oc.re.sub(r"(?m)^(\s*control\s*=\s*).*$", r"\1control.tg", src_text)
    (out_dir / "shoulder.tg").write_text(shoulder)
    (out_dir / "theta_best.json").write_text(json.dumps(
        {"a0": float(a0), "a": [float(x) for x in a_out], "b": [float(x) for x in b_out],
         "omega": float(info["omega"]), "frequency_hz": float(info["omega"] / math.tau)}, indent=2))
    (out_dir / "codesign_seed.json").write_text(json.dumps({"genome": [float(x) for x in g]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", type=Path, default=ROOT / "flight-orientations/hovering/shoulder.tg")
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
    K, n = info["K"], info["n_tend"]
    n_motor = 2 if info["tune_motor"] else 0
    print(f"[codesign] genome dim={D} (control {2*K+1} + tendon {2*n} + motor {n_motor}"
          f" + geometry {n_geom})  pop={args.pop} dts={args.timesteps}", flush=True)

    # Per-genome models for the geometry block. The .tg variant must sit beside
    # its source (meshdir and the `control =` include resolve relative to it),
    # and oc.setup serializes compiles on a lock, so one scratch name per dt is
    # enough -- batches are built one at a time.
    geom_lo = (2 * K + 1) + 2 * n + n_motor

    def model_of(row, dt):
        """Compile ONE genome's geometry into its own (mx, dx0)."""
        tgv = geom_tg(args.tg, args.tg.with_name("_ea_genome.tg"),
                      geom_names, np.asarray(row)[geom_lo:])
        _, mxg, dxg, *_ = oc.setup(tgv, args.scene.with_name("scene_ea_genome.xml"),
                                   aero_substep=args.aero_substep,
                                   integrator="euler", timestep_ms=dt)
        return mxg, dxg

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
    GENE_BLOCKS = {"control": (0, 2 * K + 1),
                   "tendon_stiffness": (2 * K + 1, 2 * K + 1 + n),
                   "tendon_pretension": (2 * K + 1 + n, 2 * K + 1 + 2 * n)}
    if info["tune_motor"]:
        GENE_BLOCKS["motor"] = (2 * K + 1 + 2 * n, geom_lo)
    gene_mask = np.zeros(D, np.float32)
    for name in want:
        if name not in GENE_BLOCKS:  # only reachable for "motor" with tune_motor off
            p.error(f"--genes: block '{name}' is not part of this genome")
        lo, hi = GENE_BLOCKS[name]
        gene_mask[lo:hi] = 1.0
    gene_mask[geom_lo:] = 1.0  # every geometry gene named on the CLI is searched
    print(f"[codesign] genes={want} geometry={geom_names} -> "
          f"{int(gene_mask.sum())}/{D} dims searched", flush=True)

    # Baseline (genome 0); baseline peak torque is dt-invariant, so evs[0] is fine for norm.
    _ev0, _mf0 = evs_for(0)
    base = evaluate(_ev0, np.zeros((1, D), np.float32), 1,
                    label=f"baseline dt={args.timesteps[0]:g}ms", models_for=_mf0)[0]
    # Take the first 13 by slice, not by exact-width unpack: build_rollout appends
    # to its metrics tuple (tau_peak/tau_drift came in that way) and a rigid unpack
    # turns every such append into a crash here. fitness_from only reads 0..11.
    assert len(base) >= 13, f"metrics tuple shrank to {len(base)}; fitness_from reads 0..11"
    Lb, taub, Ab, Sb, Clcb, Pinb, Ovlb, swb, Psupb, Cvb, alphab, kb, convb = \
        [float(x) for x in base[:13]]
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
        feas = all((m[0] >= LIFT_FLOOR and A_LO <= m[2] <= A_HI and m[1] <= TAU_CAP
                    and m[4] <= CLC_CAP) for m in mlist)
        taus = np.array([m[1] for m in mlist])
        tau_worst = taus.max(); tau_spread = taus.max() - taus.min()
        A = np.mean([m[2] for m in mlist]); Psup = np.mean([m[8] for m in mlist])
        swing = np.mean([m[7] for m in mlist])
        # Recompute C_valid from raw alpha_pk/k_eff (m[10]/m[11]) against the
        # baseline-calibrated thresholds; m[9] uses the miscalibrated defaults.
        a_pk = np.mean([m[10] for m in mlist]); k_e = np.mean([m[11] for m in mlist])
        Cvalid = (max(0.0, a_pk - ALPHA_STALL) ** 2 / ALPHA_STALL ** 2
                  + max(0.0, k_e - K_MAX) ** 2 / K_MAX ** 2)
        lift_short = np.mean([max(0.0, LIFT_FLOOR - m[0]) / max(LIFT_FLOOR, 1e-9) for m in mlist])
        amp_bad = np.mean([max(0.0, A_LO - m[2], m[2] - A_HI) / max(Ab, 1e-9) for m in mlist])
        tau_bad = max(0.0, tau_worst - TAU_CAP) / max(taub, 1e-9)
        clc_bad = np.mean([max(0.0, m[4] - CLC_CAP) / max(CLC_CAP, 1e-9) for m in mlist])
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
        if info["tune_motor"]:
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
        s = 2 * K + 1
        pop[:, s:s + n] = np.clip(pop[:, s:s + n], -1.8, 1.4)           # stiffness log
        pop[:, s + n:s + 2 * n] = np.clip(pop[:, s + n:s + 2 * n], -0.6, 0.6)  # pretension log
        if info["tune_motor"]:
            pop[:, s + 2 * n:geom_lo] = np.clip(pop[:, s + 2 * n:geom_lo], -0.6, 0.6)
        if n_geom:
            # +-0.4 log = a 0.67x..1.49x span on each dimension. Wider than that
            # and the compiled rig stops resembling the design it started from.
            pop[:, geom_lo:] = np.clip(pop[:, geom_lo:], -0.4, 0.4)
        return pop

    pop = clip(init_block())
    arch_g, arch_f, arch_m = [], [], []
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
                arch_m.append([Ms[t][k] for t in range(len(evs))])
        order = np.argsort(fits)[::-1]
        M0 = Ms[0][order[0]]; taus0 = [Ms[t][order[0]][1] for t in range(len(evs))]
        print(f"[gen {gen:02d}] best_fit={fits[order[0]]:.3f} feas={int(oks.sum())}/{args.pop} | "
              f"tau/dt={[round(x*1e3) for x in taus0]}mNm A={M0[2]*1e3:.0f}mm L={M0[0]*1e3:.0f}mN "
              f"alpha={math.degrees(M0[10]):.0f}deg k={M0[11]:.2f} | archive={len(arch_g)}", flush=True)
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
            taus = [float(np.asarray(mj(gj, *model_of(g, d)))[1]) for d, mj in sweep]
        else:
            taus = [float(np.asarray(mj(gj))[1]) for _, mj in sweep]
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
