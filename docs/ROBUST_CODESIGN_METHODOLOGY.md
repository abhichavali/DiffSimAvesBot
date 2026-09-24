# Robust hover co-design: methodology & replication log

> **Historical log.** Commands are as run during the campaign. Most run scripts,
> warm-start JSONs and run dirs it names are not in the tree (the committed ones are in
> git history before the refactor, e.g. `4889063`). Design paths point at `designs/`
> (see `designs/README.md`).

**Date**: 2026-07-07/14. **Branches**: `june-26-demo` (old campaign, §1–10) →
`fixed-codesign` worktree (fixed-protocol campaign, §11–14). **Goal**: starting from the
hand-tuned hover design, use Adam (via MJX differentiable simulation) to co-optimize a
control policy (≤3 Fourier harmonics) together with structural tendon stiffness/pretension
and motor-tendon preload, such that the result **transfers to unseen integrator × timestep
combinations** — i.e. it learned physics, not a discretization trick. This document is
self-contained: an LLM with access to this repo should be able to replicate the experiment
end-to-end from it.

> **STATUS (2026-07-11): CAMPAIGN CONCLUDED — GOAL MET.** Flagship = run-5 it009
> ("it009p"), certified **156.5 mNm doctrine worst-case = −10.4% vs seed** at 345–348 mN
> lift, baked in `designs/hummer/hover-soft-it009p/`. The §7 "simulator gap"
> diagnosis was falsified (§11 — it was a metric mismatch plus a beat-sampling artifact);
> §1–10 are kept as history of the old campaign. The fixed-protocol campaign, final
> standings, the harmonics verdict, and phase-lag findings are in §11–14. All cloud
> instances are destroyed; artifacts live in the `DiffSimAvesBot-fixed` worktree (§14).

---

## 1. Threat model: what "hacking" looks like here

Gradient descent through a differentiable simulator will exploit any regularity of the
*discretization* rather than the physics if that is the cheapest way to lower the loss:

- **Integrator hacking**: peak torque drops under Euler but rebounds under
  implicitfast/RK4 (observed in earlier single-integrator runs: 20–30% "gains" that
  evaporated on the deployment integrator).
- **Timestep hacking**: the design is only good at the exact trained dt (observed:
  designs trained at dt=0.06 ms degraded 10–30% off that dt).
- **Simulator-implementation hacking** (discovered in run 1 of THIS experiment, §7):
  the design passes a *full integrator × dt matrix inside MJX* yet fails the same matrix
  on CPU MuJoCo, because the optimizer parked its gains in a dt-dependent MJX↔CPU
  physics gap. This is the subtlest failure mode and the reason the certification
  simulator must be a **different implementation**, not just different settings.

## 2. Defense stack (five layers)

1. **Gradient level — rotating robust mini-batches** (`--rotate-batch`): each Adam step
   draws one of N precompiled batches of K (integrator, dt) rollout variants; the loss is
   a **soft worst-case** envelope over the batch plus a cross-variant spread penalty.
   The dt centers are now jittered (`--dt-jitter`) and each batch can include a
   precision-perturbed duplicate rollout (`--precision-perturb`) so the optimizer cannot
   only memorize the exact grid. Hacking one scheme *raises* the loss. Rotation covers
   the full matrix over ~10 steps at only K rollouts/step.
2. **Schedule level — safer knobs first** (`--control-after N`): tendon/motor parameters
   (bulk physical stiffnesses; no fine temporal structure to exploit) train alone first;
   harmonics (where dt-tricks live) unlock later, and phase-boundary snapshots make their
   contribution separately attributable/discardable.
3. **Selection level — full-matrix gate** (`--full-check-every`): candidates are only
   promoted after evaluating ALL bank variants; promotion requires bounded spread and is
   *scored on the worst case*. No selection from per-iteration (subset) metrics.
4. **Certification level — independent simulator gate** (`--cpu-every` +
   `--cpu-timestep-factors`): re-simulate on CPU MuJoCo (different codebase: numpy aero,
   `mj_step`) across integrators AND timesteps. Added dt factors after run 1 proved the
   MJX-only matrix insufficient.
5. **Final exam — holdout dts** (`--holdout-dts`): timesteps never used in any training
   batch or gate, evaluated once at the end (MJX ×3 integrators + CPU ×2). The only
   unbiased generalization estimate.

Known limitation: gradients flow only through MJX (CPU is not differentiable), so the
optimizer can still *drift toward* MJX/CPU gaps; layers 4–5 refuse to certify but cannot
steer. On slow/cheap boxes, `--cpu-every 0` is acceptable for a continuation run, but then
CPU validation is a manual checkpoint replay, not an in-loop gate.

## 3. Seed design

- Morphology: `designs/hummer/hover/shoulder.tg` — hand-tuned hover
  design, **24 per-tendon stiffness/pretension overrides baked into the .tg** `overrides:`
  block. Native `timestep = 0.13 ms`, `integrator = Euler` (0.13 is the documented
  stability edge for this scale).
- Control: sibling `control.tg` — pure sinusoid, `a0 = 65`, `fundamental = 10.345 Hz`,
  all harmonics zero.
- Baseline numbers (sanity checks for replication):
  - MJX euler@0.06: peak torque **144.0 mNm**, lift **364.4 mN**, stroke amp 342.5 mm.
  - MJX full 12-case matrix: 141–144 mNm (2% spread), lift_min 360.8 mN.
  - CPU MuJoCo: **168–176 mNm and 358–364 mN across ALL of euler/implicitfast/rk4 ×
    dt ∈ {0.03…0.16} ms** — the seed is remarkably flat; any claimed gain must beat its
    worst case (~174–176 mNm), not its best.
  - Persistent MJX↔CPU absolute offset ≈ 1.2× on peak torque, uniform across dts for
    the seed. **Watch this ratio**: if it becomes dt-dependent for an optimized design,
    that design is exploiting a simulator gap (§7).

## 4. The optimizer and reward

The optimizer was one file, `opt/optimize_control.py`, during this campaign; it is now
that CLI over the `src/codesign/` library (`opt/README.md`). Parameterization
(57 params total, all log-factors except harmonic depths):

- Control (7): `[a0_log, a_1..a_3, b_1..b_3]`; a0 as log-factor of the seed a0; harmonics
  as **modulation depths** (fractions of a0): `u(t) = a0·(1 + Σ aₙcos(nωt) + bₙsin(nωt))`.
- Structural tendons (48): per-tendon `stiff_log` (12+12 L/R individual) + `pre_log`
  (rest-length factor), applied to `model.tendon_stiffness` / `tendon_lengthspring`.
- Motor tendons (2): one shared stiffness log-factor + one shared pretension log-factor.

Optimizer: hand-rolled Adam (b1=0.9, b2=0.999), forward-mode gradients
(`jax.jacfwd` — memory-light for 57 params through ~18k-step `lax.scan` rollouts).
Rollout: 8 warmup periods + 3 sample periods (warmup ≥8 is load-bearing; the rig builds
lift for ~8 cycles), aero recompute-and-hold every `--aero-substep` steps.

Reward (maximized; `--single-stage --lift-torque-tradeoff 5` selects this form), with
per-batch soft worst-case aggregation over the variant set V:

```
softmaxβ(x) = (logsumexp(βx) - log |V|) / β
softminβ(x) = -softmaxβ(-x); β anneals from --softmax-beta-start to --softmax-beta-end
τ_rob   = softmaxβ_v τ_peak(v)         (Lp-smoothed peak |motor torque|, p=10)
τ_hard  = max_v τ_peak(v)              (logged/certification metric)
L_min   = softminβ_v lift(v);   A ∈ [softmin, softmax over V of stroke amplitude]
swing   = softmaxβ_v std_t(τ(t))       (flat-profile term)
R = − wτ·(τ_rob/τ_max)                        # peak torque, normalized by baseline
    − wτ·T·softplus((lift0 − L_min)/s)/lift0  # lift drop costs T:1 vs torque (T=5)
    + wτ·max(0, L_min − lift0)/(T·lift0)      # reverse: 5% lift gain pays for 1% torque
    − w_swing·(swing/τ_max)²
    − wK·(hinge(A_lo−A_min)² + hinge(A_max−A_hi)²)/A*²   # two-sided amp band ±5%
    − ws·Σ n²(aₙ²+bₙ²)                        # harmonic energy (dt-trick regularizer)
    − w_power·P_inactive − w_overlap·C_overlap
    − w_gap·((τ_rob − min_v τ)/τ_max)²        # cross-variant spread penalty
```

Weights used in the July 9/10 runs: `wL=0, wτ=10, wK=8, ws=0.03`
(`--weights 0 10 8 0.03`), `w_swing=5`, `w_power=3`, `w_overlap=0`, `w_gap=15`,
tradeoff `T=5`, amp band ±5%. Initial exploration used lr `8e-4`; post-harmonics
continuation from an edge design required lr `5e-5`.
All hinges are softplus-smoothed (gradients flow everywhere).

Important metric caveat: the log's `tau=` / `tau_hat` is the MJX differentiable Lp
peak proxy, while CPU validation reports an exact sampled max. A persistent ~30 mNm
MJX<->CPU offset is therefore partly a metric mismatch and partly implementation
precision/solver/aero mismatch. Judge deliverables by exact MJX full-matrix and CPU replay,
not by per-iteration `tau_hat` alone.

## 5. New machinery added for this experiment

Flags added to `opt/optimize_control.py` (commit on `june-26-demo`, 2026-07-07):

| Flag | Value used | Meaning |
|---|---|---|
| `--rotate-batch K` | 3 | rotating robust mini-batches of K (integrator,dt) variants; replaces `--tri-integrator/--tri-timestep` |
| `--dt-bank MS...` | 0.04 0.06 0.09 0.13 | training dt grid; bank = {euler,implicitfast,rk4} × bank = 12 variants |
| `--dt-jitter` | 0.15 | per-batch sampled dt = center × (1+U[-δ,δ]); fixes exact-grid overfit |
| `--precision-perturb` | 1e-7 | adds a duplicate member with tiny qvel noise; cheap proxy for float32/CPU sensitivity |
| `--softmax-beta-start/end` | 60 → 300 | smooth worst-case early, sharper robust max late |
| `--n-batches` | 6 | precompiled random batches; each forced to ≥2 distinct integrators AND ≥2 distinct dts; seeded by `--batch-seed` (0) |
| `--full-check-every` | 25 | full 12-case MJX forward eval; promotes `robust_best` |
| `--full-gap` | 0.15 | max torque spread across the full matrix for promotion |
| `--holdout-dts MS...` | 0.03 0.11 0.16 | end-only unseen-dt report (MJX ×3 integ + CPU ×2) |
| `--control-after N` | 16 in 5090 continuation; 0 when restarting from it027 | zero control-slice gradient before harmonics unlock; restart from an already-harmonic seed must use 0 |
| `--cpu-every` | 50 on 5090; 0 on 3080 continuation | CPU MuJoCo certification cadence; 3080 CPU checks were too slow for in-loop use |
| `--cpu-timestep-factors` | 0.5 1.5 *(run 2; run 1 omitted — see §7)* | extra CPU checks at factor×dt: euler@0.03 and euler@0.09 |
| `--cpu-gap` | 0.15 | max spread across the 5 CPU cases for `cpu_best` promotion |
| `--aero-substep` | 1 | removes the aero hold/substep artifact from the differentiable rollout |

Implementation notes for the replicator:
- Each (integrator, dt) variant is its own `build_rollout` closure over its own
  `mjx.put_model` (dt changes steps/cycle, warmup and sample step counts rescale;
  period is fixed). Batches reuse the existing `make_loss` robust-envelope math by
  passing the 3 variant metrics as (primary, metrics_impl, metrics_rk4).
- Per-iteration selection is **disabled** in rotate mode (batch metrics are an
  optimistic subset); `best`/`stage_best`/`robust_best` update only at full checks.
- Promotion gates: full-matrix spread ≤ 15%, worst-case amp inside the ±5% band,
  worst-case lift ≥ 0.9 × baseline worst-case lift; score = worst-case τ (+swing,
  +5:1 lift-drop equivalent).
- In staged runs with `control_after > 0`, current `robust_best`/`cpu_best` are
  snapshotted at the unlock point to `<out>/robust_best_preharmonics/` and
  `<out>/cpu_best_preharmonics/` (phase attribution: diff final bests against these).
- Startup is compile-heavy: 12 forward jits (baseline matrix) + 6 jacfwd jits (one per
  batch, lazily on first draw). Budget 20–40 min on an RTX 3080 before steady pace
  (~1.5–2 min/iter; the 264-it continuation is roughly 7–9 h, about $0.9–$1.2 at
  $0.129/hr).

## 6. Exact run commands

The old run-2 command below is now superseded for continuation work. Current settings:
`aero-substep=1`, jitter/precision enabled, `lift_floor=330 mN`, soft robust beta
annealing, and harmonics unlocked early/always depending on the seed.

### 6.1 2×5090 continuation from the it009 seed

```bash
uv run python opt/optimize_control.py \
  --tg designs/hummer/hover/shoulder.tg \
  --init-codesign opt/init_from_g1_it009.json \
  --tune-tendons --tune-motor-tendons --single-stage \
  --lift-floor 0.330 --lift-torque-tradeoff 5 \
  --rotate-batch 3 --n-batches 6 --batch-seed 17 \
  --dt-bank 0.04 0.06 0.09 0.13 \
  --dt-jitter 0.15 --precision-perturb 1e-7 \
  --softmax-beta-start 60 --softmax-beta-end 300 \
  --full-check-every 25 --full-gap 0.15 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after 16 \
  --cpu-every 50 --cpu-gap 0.15 --cpu-timestep-factors 0.5 1.5 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage 291 --lr 8e-4 \
  --out opt/runs_robust_hover_5090_h25_l330_g1
```

### 6.2 3080 continuation from transferred g1 it027 seed

The 2×5090 run was destroyed after copying `g1/checkpoints/it027` into
`opt/restart_from_g1_it027.json`. On the cheap 3080, CPU validation took ~9 minutes
before the first iteration, so the live continuation is MJX-fast and CPU replay is manual:

```bash
uv run python opt/optimize_control.py \
  --tg designs/hummer/hover/shoulder.tg \
  --init-codesign opt/restart_from_g1_it027.json \
  --tune-tendons --tune-motor-tendons --single-stage \
  --lift-floor 0.330 --lift-torque-tradeoff 5 \
  --rotate-batch 3 --n-batches 6 --batch-seed 17 \
  --dt-bank 0.04 0.06 0.09 0.13 \
  --dt-jitter 0.15 --precision-perturb 1e-7 \
  --softmax-beta-start 60 --softmax-beta-end 300 \
  --full-check-every 25 --full-gap 0.15 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after 0 \
  --cpu-every 0 --cpu-gap 0.15 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage 264 --lr 5e-5 \
  --out opt/runs_robust_hover_3080_from_g1_it027_l330_stable_mjxfast
```

Local smoke test first (Mac CPU, ~10 min; catches shape/name bugs before paying for GPU):
motor-tendons only, `--harmonics 1 --iters-per-stage 3 --dt-bank 0.12 0.13
--full-check-every 2 --holdout-dts 0.125 --warmup-periods 0.3 --sample-periods 0.3`.
Note: short warmup makes MJX peaks huge (~35 N·m) because the sample window includes the
t=0 transient — expected artifact, not a bug.

## 7. Run history: finding → continuation

**Run 1** used everything above EXCEPT `--cpu-timestep-factors` (CPU gate at dt 0.06 only).
Trajectory: tendon-only phase drifted torque +2% while banking +5% lift, recovered by
it050 (CPU robust 173, −1%); harmonics unlocked at it080; by it100–150 CPU@0.06 robust
hit **158–159 mNm (−9%)** with MJX full-matrix spread 9–13%, all gates green.

**Certification sweep exposed the hack.** Method (do this for ANY winner):

```bash
# fetch checkpoint, then CPU-sweep it AND the untouched seed (it000) as control:
for dt in 0.03 0.04 0.06 0.09 0.11 0.13 0.16; do
  uv run python opt/design_io.py <design_dir> \
    --integrator euler implicitfast rk4 --timestep $dt
done
```

Worst-integrator CPU peak torque (mNm) per dt, run-1 `robust_best` (it125) vs seed:

| dt (ms) | 0.03 | 0.04 | **0.06** | 0.09 | 0.11 | 0.13 | 0.16 |
|---|---|---|---|---|---|---|---|
| seed | 174 | 174 | 174 | 175 | 175 | 175 | 176 |
| run-1 best | 182 | **196** | **148** | **203** | 173 | 175 | 218 |

−15% at the deploy dt, but **+12–16% WORSE than the seed at 0.04/0.09** — while the MJX
matrix showed every case ≤144 with ≤13% spread. Diagnosis: the MJX↔CPU torque ratio,
flat 1.2× for the seed, became dt-dependent (1.1× @0.06, 1.5× @0.09) for the optimized
design. The optimizer had found a dt-dependent MJX/CPU physics divergence and put the
gains inside it. It entered through the **harmonics** (the tendon-only phase was
CPU-flat), and the harmonic amplitudes were *small* (≤3% modulation depth) — subtle
waveform shaping, invisible to in-MJX certification.

**Lesson (headline)**: a full integrator × dt matrix *inside the training simulator* is a
necessary but insufficient transfer certificate. Certification must cross to the
deployment implementation, on the same axes.

> **SUPERSEDED (2026-07-09, see §11).** The diagnosis above is wrong in its mechanism:
> there is **no MJX↔CPU physics gap** (exact peaks agree to <1% at every dt once the
> metrics match). The dt table above is real data but its cause is (a) MJX gates scoring
> an Lp10 peak proxy vs CPU scoring exact max, and (b) a 3-cycle sample window parked on
> a ~40-cycle torque beat whose phase shifts with dt. The headline lesson survives in
> amended form: certification must use **exact per-cycle peaks over ≥1 beat period with a
> stationarity gate** — in whichever simulator.

**Run 2 changes**: (a) `--cpu-timestep-factors 0.5 1.5` — the CPU gate now spans
euler/implicitfast/rk4 @0.06 + euler @0.03/@0.09; run-1's it125 design shows ~27% spread
under this gate and is correctly rejected; (b) pre/post-harmonics best snapshots (§5).
Run-1's checkpoint trail (every 3rd iter, through it177) was archived for post-hoc
CPU certification before restarting.

**Run 3 / July 9–10 changes**: the 2×5090 run introduced the smoother/jittered bank
(`--dt-jitter 0.15`, precision perturb, softmax β annealing), `aero_substep=1`, lift floor
330 mN, and a symmetric lift/torque trade: 1% lift loss must buy 5% torque reduction, while
5% lift gain can pay for 1% torque increase. Harmonics were unlocked early (`it016`) from
the it009 seed because the tendon/motor-only phase had already reached the local peak.
The best-looking seed was `g1/checkpoints/it027` (MJX baseline after transfer:
Lbar 391.2 mN, tau_hat 142.4 mNm; full-matrix MJX robust 148.9 mNm; CPU robust 180.7 mNm).
It was transferred to a single RTX 3080 and the 2×5090 instance was destroyed.

Continuation lesson: restarting that edge design with lr `8e-4` and β start `82` caused
repeated `NaN/diverged -> revert` events and rapid lr collapse. The cause was not a random
GPU failure: the seed has `conv≈2.8`, `alpha≈80°`, `k≈0.337`, CPU robust far above the
MJX proxy, and a stage target below full-matrix robust. It is a numerically sharp/high-AoA
design. The stable continuation uses lr `5e-5`, β start `60`, and disables in-loop CPU
checks on the 3080.

## 8. Infrastructure (vast.ai)

```bash
vastai search offers 'gpu_name=RTX_3080 num_gpus=1 reliability>0.97 cuda_vers>=12.4 \
  inet_down>200 rentable=true' -o dph+ --raw          # pick cheapest sane offer
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.8.1-auto --disk 64 --ssh --direct
# wait for actual_status=running; endpoint = public_ipaddr:direct_port_start
# (test BOTH direct and sshN.vast.ai proxy; usually exactly one works)
ssh root@HOST -p PORT "apt-get install -y rsync; mkdir -p /workspace/DiffSimAvesBot"
rsync -az --delete -e "ssh -p PORT -i KEY" --exclude .git --exclude .venv \
  --exclude __pycache__ --exclude data --exclude 'opt/runs*' ./ root@HOST:/workspace/DiffSimAvesBot/
ssh root@HOST -p PORT "cd /workspace/DiffSimAvesBot && \
  (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
  export PATH=\$HOME/.local/bin:\$PATH && uv sync -q && uv pip install -q 'jax[cuda12]==0.10.2'"
# Then create/run a wrapper matching §6.2, including LD_LIBRARY_PATH for the nvidia wheels.
```

Poll with `tail`/`grep` over ssh; the `.done` sentinel carries the exit code. Current
3080 cost is ~$0.129/hr; the 2×5090 was ~$0.931/hr and has been destroyed.

Operational gotchas hit during this experiment:
- `pkill -f optimize_control` **kills your own ssh session** (the pattern matches the
  remote shell's own command line). Use `pkill -f '[o]ptimize_control'` and keep kill and
  relaunch in SEPARATE ssh invocations (the relaunch line contains the literal string).
- The pytorch image lacks `rsync` — install it before syncing.
- Python stdout is block-buffered under redirection: pre-loop prints appear only when the
  first `flush=True` iteration line flushes the buffer. Don't mistake it for a hang; check
  `ps` (the optimizer runs at >100% CPU during setup/CPU-baseline).
- Fresh boxes have no `/workspace` — `mkdir -p` before the first rsync.
- The base `uv sync` installs CPU-only JAX. Install CUDA extras and set the NVIDIA wheel
  library path before launching:
  `uv pip install 'jax[cuda12]==0.10.2'` and
  `export LD_LIBRARY_PATH="$(find "$PWD/.venv/lib/python3.11/site-packages/nvidia" -path '*/lib' -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"`.

## 9. Monitoring runbook (HISTORICAL — no live boxes since 2026-07-11)

> The connection constants below refer to instances that no longer exist (3080
> `44351697`, 2×5090 `44330669`, and the fixed-campaign 5090 `44363824` are all
> destroyed). The runbook pattern remains valid for any future run.

Everything a monitor needs. The monitor's job is **read-only**: report progress, relay
gate events, detect stalls/completion. Never kill, restart, or edit anything on the box.

### 9.1 Connection constants (current live 3080 continuation, 2026-07-10)

```bash
HOST=89.221.67.143  PORT=27122                 # vast.ai instance 44351697 (RTX 3080)
SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$HOST"
DIR=/workspace/DiffSimAvesBot                  # repo copy on the box
LOG=$DIR/opt/run_robust_hover_3080_from_g1_it027_l330_stable_mjxfast.log
OUT=$DIR/opt/runs_robust_hover_3080_from_g1_it027_l330_stable_mjxfast
DONE=$DIR/opt/run_robust_hover_3080_from_g1_it027_l330_stable_mjxfast.done
```

The direct endpoint above works; proxy is `ssh5.vast.ai:31696` if direct fails.
If a different run/box is live, get HOST/PORT from `vastai show instances --raw`
(fields `public_ipaddr` / `direct_port_start`).

### 9.2 The one health-check command (run every 20–30 min)

```bash
$SSH "cd $DIR && pgrep -fc '[o]ptimize_control' ; \
  cat $DONE 2>/dev/null ; \
  grep -cE '^  it[0-9]' $LOG ; \
  tail -n 3 $LOG"
```

Output, in order: (1) process count — **2 = healthy** (uv wrapper + python), 0 = dead;
(2) exit code IF finished, blank otherwise; (3) iterations completed (of 264 for the
current continuation); (4) last log lines. CPU checks are disabled in the live 3080 run,
so silence is mostly JIT/full-matrix work, not CPU certification.

### 9.3 Event extraction

```bash
$SSH "grep -E '\[(full|cpu|phase)' $LOG"          # all gate/phase events so far
$SSH "grep -c 'NaN/diverged' $LOG"                # divergence reverts (occasional = ok)
$SSH "grep -E '^\[(cpu|full)-baseline\]' $LOG"    # the baselines everything is % vs
```

How to read each line type:

| Line | Meaning | Healthy looks like |
|---|---|---|
| `it047 loss=… tau=…mNm … [e@0.04/i@0.09/r@0.13 rob=143.4 gap=1%]` | one Adam step; bracket = this batch's variants, robust (worst-case) τ, cross-variant spread | τ_rob drifting ≤144 (baseline), gap ≤ a few %, `L=` ≥ ~360 mN, `A=` ≈ 345±17 mm |
| `[full itNNN] robust=… (−X% vs base) gap=Y% liftmin=… [TAG]` | full 12-case MJX matrix, every 25 iters | tag `TRANSFERS`; `HACK(gap N%)` / `AMP-OUT` = candidate rejected (informational, run continues) |
| `[cpu itNNN] mjx=… euler@0.06=… impl@0.06=… rk4@0.06=… euler@0.03=… euler@0.09=… robust=… [TAG]` | 5-case CPU MuJoCo gate when `--cpu-every > 0` — **the one that catches dt-hacks** | `TRANSFERS` with robust ≤ ~176 (seed worst case); on the 3080 this took ~9 min before `it000`, so the live continuation disables it and requires manual CPU replay |
| `[phase] itNNN: harmonics UNLOCKED …` | tendon-only phase over; `*_preharmonics/` snapshots written in staged runs | appears at `--control-after`; absent/non-informative for fully-unlocked restarts (`--control-after 0`) |
| `itNNN NaN/diverged -> revert, lr=…` | NaN guard fired; reverted + halved lr | fine occasionally; **many in a row = report to user** |
| `[holdout …]` lines then `=== … DONE exit 0 ===` | end-of-run holdout report + exit | exit 0 and `$DONE` contains `0` |

### 9.4 Stall vs. buffering (do NOT restart on silence alone)

Log unchanged for >20 min AND no `[cpu itNNN]` check plausibly in progress:

```bash
$SSH "ps -eo pid,pcpu,etime,args | grep '[o]ptimize_control'"
```

>100% CPU ⇒ it's working (compiling, or inside a CPU gate — those are silent for
10–15 min); just wait. Process gone and no `$DONE` ⇒ crashed: capture
`tail -n 60 $LOG`, report to the user, **do not relaunch yourself**.
SSH itself unreachable ⇒ `vastai show instances --raw` — check `actual_status`
(instance may have been outbid/died); report, don't recreate.

### 9.5 On completion (`$DONE` exists)

```bash
$SSH "cat $DONE ; sed -n '/holdout/,\$p' $LOG"               # exit code + final report
rsync -az -e "ssh -i $KEY -p $PORT" \
  root@$HOST:$OUT/ opt/runs_robust_hover_3080_from_g1_it027_l330_stable_mjxfast_remote/
```

Report to the user: exit code, final `robust_best` numbers, the holdout section verbatim,
and any `cpu_best` / `*_preharmonics` deltas that exist. For the current MJX-fast
continuation, the user (or a capable session) must run the §7 CPU sweep locally on the
fetched winner + seed — the monitor does not certify.

The current 3080 box costs ~$0.129/hr. The previous 2×5090 instance `44330669` was
destroyed after the it027 seed was transferred. Never destroy the live 3080 before
artifacts are fetched.

## 10. Artifacts & how to judge the outcome

Current output dir (`opt/runs_robust_hover_3080_from_g1_it027_l330_stable_mjxfast/`):
- `robust_best/` — best design passing the full MJX-matrix gate (MJX-certified only).
- `cpu_best/` — best design passing the 5-case CPU gate (**the honest deliverable**);
  absent when launched with `--cpu-every 0`.
- `robust_best_preharmonics/`, `cpu_best_preharmonics/` — phase-1 (tendons/motor only)
  winners, only meaningful for staged runs with `control_after > 0`.
- `checkpoints/itNNN/` every 3 iters; `stage_*/`; `cpu_validation.json` (final CPU
  validation + `"holdout"` section when CPU checks are enabled); `log.json` (full metric
  trail incl. per-check matrices).
- Each design dir: `shoulder.tg`, `control.tg`/`theta_best.json` (a/b are ABSOLUTE,
  divide by a0 for depths), `tendon_optimized.json` (per-tendon stiffness/pretension
  factors), `motor_tendon_optimized.json`. Reconstruct/validate via `opt/design_io.py`
  (warmup=8!).

Success criteria, in order of authority:
1. `cpu_best` worst-case over the CPU sweep grid (§7 method) **beats the seed's worst
   case (~174–176 mNm)** at equal-or-better worst-case lift (~358 mN).
2. Holdout section: unseen dts within ~5% of trained-matrix performance (0.16 may
   legitimately degrade/diverge — it is past the tg's stability edge; report, don't fail).
3. If present, `cpu_best` vs `cpu_best_preharmonics` quantifies what harmonics bought —
   if the harmonics phase added nothing certified, the tendon-phase design IS the result.
4. Always re-run the §7 sweep locally on the final winner AND the seed. Trust no green
   tag you didn't reproduce in the deployment simulator.

For the current `stable_mjxfast` run, there will be no in-loop `cpu_best/` because
`--cpu-every 0` is deliberate. Treat `robust_best/` as an MJX candidate only; replay its
checkpoints through CPU before calling it a design.

---

## 11. Root-cause revision (2026-07-09): there was no simulator gap

Local CPU + MJX-on-Mac replays of the run-1 winner and the seed falsified §7's
"simulator-implementation hacking" diagnosis. With euler, aero substep 1 and the jacSite
velocity fix, **MJX and CPU MuJoCo agree to <1% on exact in-window peak torque at every
dt tested** (f32 or f64, seed and hacked design alike). The two believed "gaps" were
pipeline artifacts:

1. **Metric mismatch.** Every MJX gate (`robust_best` promotion, `[full]` logs,
   HACK/TRANSFERS tags) scored `tau_hat` = an Lp(p=10)-smoothed peak, while CPU gates
   scored the exact max. For the seed: Lp10 = 144 vs exact = 174 — that ratio IS the
   "persistent 1.2× MJX↔CPU offset" of §3; there is no physics offset. Lp10 is
   duty-cycle hackable: narrow the torque spikes and Lp10 drops while the true peak
   doesn't (run-1 winner @0.04: Lp10 143 vs exact 188).
2. **Beat-window parking.** Small (≤3% depth) harmonics excite a lightly damped
   ~40-cycle beat in per-cycle peak torque. The 8-warmup/3-sample protocol used
   *everywhere* (training loss, MJX matrix, CPU gate, holdouts, the §7 sweep) samples one
   beat phase; Adam parks the beat minimum inside the window at the trained dts. The
   run-1 winner @0.06 reports 139.7–195.8 mNm depending on window start (a 40% lottery);
   its per-cycle peaks climb 143→196 over 35 cycles. Changing dt shifts the beat phase —
   that is the wild dt table in §7, not a simulator divergence.

Honest converged numbers (last-20-of-100-cycle mean, euler @0.06): seed 168.5, run-1
winner 186.9 (197.9 @0.04) — the "−15% winner" was a **+11–17% true steady-state
regression**.

**Protocol fixes** (commit `ce66dca`, branch `fixed-codesign`): exact per-cycle peak
scoring everywhere (identical metric in loss, MJX gates, CPU gates); evaluation windows
≥1 beat period (24-cycle gates: `--full/cpu-sample-periods 24`); a per-cycle-peak
stationarity gate (`--drift-gate 0.03`), later also priced in the loss (`--w-drift`);
randomized window starts (`--warmup-bank 8 16 24`). Once metrics match, MJX is a
faithful proxy — in-loop CPU checks are certification, not steering.

## 12. Fixed-protocol campaign (branch `fixed-codesign` worktree, 2026-07-10/11)

Single RTX 5090 (vast.ai instance 44363824, destroyed 2026-07-11 after full artifact
fetch). Run numbering restarts at 1; §7's runs are "the old campaign".

**Honest baselines** (post-fix, exact peaks, 8 warmup + 24 sample cycles): MJX exact
174.2 / CPU robust 174.4 / full-matrix 174.5; certification math uses seed worst =
**174.7 mNm** at 358–364 mN lift. The −10% dream target = 157.

**Certification doctrine** (deploy target dt ≤ 0.06 ms): worst case over
{euler, implicitfast, rk4} @0.06 plus euler/implicitfast @0.03 ("doctrine"); the "wide"
sweep adds euler @0.04/0.09. Certify offline via `opt/design_io.py`
(`codesign.artifacts.reconstruct_design`, warmup=8 is load-bearing, then
`codesign.cpu_check.cpu_rollout`, whose `cycle_peaks` are the per-cycle peaks);
require stationary per-cycle peaks.

**Pre-run physics sweeps** (the campaign's compass):

- Flap frequency == crank speed == a0/2π exactly (a0 = 65 rad/s → 10.345 Hz). An ω-only
  `--tune-frequency` knob is physically inert for a sinusoid and desyncs harmonics once
  they unlock — dropped from run 4 onward.
- **No amplitude resonance exists**: the crank kinematically pins stroke at 342 mm.
  What reads as "resonance" on this rig is spring compensation of transmission torque.
- The compensation corridor lies BELOW seed stiffness: uniform 0.8× tendons =
  CPU-certified 166.3 mNm worst-case (**−4.7% with zero optimization**) at 357 mN lift,
  stationary across all integrators × dt 0.04–0.13. The seed sits under a torque cliff
  in the other direction (1.5× = +22%).
- Reward bug: the 5:1 lift trade was anchored at *baseline* lift, making the corridor
  score-negative — the reason every earlier run hoarded lift. Fixed: anchor at
  `--lift-floor` (330 mN).

**Run log**:

| Run | Config delta | Outcome |
|---|---|---|
| 1 | protocol fixes only | gates correctly vetoed every beat-chasing candidate (DRIFT 12–18%) but the loss didn't price drift; ~12 min/iter; killed |
| 2 | `--w-drift 20`, + tune-frequency, + tune-pitch | killed unfinished once the frequency knob was proven inert |
| 3 | rotate-batch 2, no precision-perturb (~2× faster) | killed unfinished — corridor findings obsoleted the config |
| 4 "corridor" | 0.8× tendon warm start, floor-anchored trade, tune-pitch (`run_fixed_codesign.sh`) | it015 −9.3%; **it018 −10.1% = first dream crossing**; it021 deploy-dt champion but dt-specialized; it039 = beat poison at 1.44% harmonic depth |
| queue | 0.9×/0.95×/1.25× stiffness warm-start variants ×100 it (`run_queue_fixed.sh`) | killed with run 5, never produced results |
| 5 "polish" | it018 warm start (full 57-value genome), w_drift 60, rotate-batch 3, sample-periods 4, lr 2e-4, swing out of selection score, harmonics unlock at it008 (`run_polish_it018.sh`) | **it009 = FLAGSHIP, −10.4%**; it021 harmonics dt-specialized again; it025 DRIFT-rejected; killed after it025 |
| 6 "lag arbitrage" | crank_phase_lag 12° baked in seed (`hovering-tuned-lag12/`), it009 genome warm start, harmonics locked all run, lr 8e-4 (`run_lag12_it009.sh`) | parity + ~5 mN lift; lift→torque conversion collapsed after ~10 iters; killed at it057 |

**Final standings** (CPU exact peaks, 8+24 cycles, vs seed worst 174.7):

| Design | Doctrine worst | Wide worst | Lift (mN) | Status |
|---|---|---|---|---|
| seed (hand-tuned) | 174.7 | ~176 | 358–364 | baseline |
| uniform 0.8× tendons | 166.3 (−4.7%) | 166.3 | 357 | corridor proof, unoptimized |
| run-4 it015 | 158.4 (−9.3%) | 158.4 | 347–351 | baked `designs/hummer/checkpoints/it015/` |
| run-4 it018 | 157.1 (−10.1%) | 158.2 | 347–349 | baked `designs/hummer/checkpoints/it018/` |
| run-4 it021 | 157.6 | 162 @0.04/0.09 | — | dt-specialized (deploy-dt champion only); baked `designs/hummer/checkpoints/it021-harmonic/` |
| **run-5 it009 ("it009p")** | **156.5 (−10.4%)** | 158.5 (−9.3%) | 345–348 | **FLAGSHIP**; baked `designs/hummer/hover-soft-it009p/` |
| run-5 it021 | 158.0 | 162.0 | — | harmonics replication #3, checkpoint only |
| run-6 it050 | ~156 | 159.2 | ~351 | parity + lift; checkpoints only |

Campaign gotchas (beyond §8): partial init flags (`--init-tendons`, since removed) did
**not** carry motor pretension (~5 mNm of the result!) — warm-start with the full
57-value `--init-codesign` genome (§4 layout, now `codesign.genome.Genome`).
Two optimizer processes on one GPU → cuSolver crash; verify kills by PID and
`nvidia-smi --query-compute-apps` before relaunch. `uv run` re-syncs and removes
jax[cuda12] — always `uv run --no-sync`.

## 13. Harmonics verdict, phase-lag findings, marginal-channel measurement

**Harmonics: abandoned, with a precise story.** Two distinct failure modes:

1. *Transient-riding* (beat excitation) — **fixed** by the §11 protocol; with w_drift 60
   every later candidate is stationary.
2. *Discretization-fitting* — **not fixed** by anything tried, including dt-bank
   rotation with jitter. Three independent replications: old-campaign run 1 (§7),
   run-4 it021/it039, run-5 it021 (147.7–153 @0.06 but 162 @0.04/0.09 → doctrine 158.0,
   worse than the sinusoid flagship) plus it025 DRIFT-rejected in-run. Harmonic gains
   (~4 mNm at the trained dt) reverse at 0.04/0.09, reproducibly. Depth doesn't save it:
   ≤0.3% depths were the only survivors; 1.44% (it039) was beat poison (171–172
   @0.04/0.09).

Paper claim: **harmonic gains certify only above the integrator-discrepancy floor, and
on this rig they never did.** it009 (sinusoid) vs it021 (harmonics) from the same run 5
is the clean within-run ablation pair. Scope caveat: this holds for a crank-driven,
kinematically pinned transmission (stroke and frequency both pinned by the crank);
direct-drive or series-elastic morphologies might certify harmonics.

**Marginal-channel measurement (2026-07-14)** — separates "harmonics are physically
useless here" from "the reward tunes them into exploits": a first harmonic
(a₁ = d·cosφ, b₁ = d·sinφ) injected on top of it009p with **no optimizer in the loop**
(`opt/harmonic_sweep.py`; exact per-cycle peaks, 8+24 cycles, euler). Grid
d ∈ {0.25, 0.5, 1, 2}% × φ ∈ {0…315°} @0.06: baseline 154.4 mNm; best point −0.4 mNm
(d=2%, φ=135°), with the φ=135° direction flat (−0.2…−0.4) at every amplitude and all
other phases worse (up to +10 mNm at 2%). Cross-dt check of that best direction
(all points stationary, drift ≤1.6):

| φ=135° depth | @0.03 (base 156.5) | @0.06 (base 154.4) | @0.09 (base 158.2) |
|---|---|---|---|
| 0.5% | +1.7 | −0.3 | +2.1 |
| 1% | +3.5 | −0.3 | +4.2 |
| 2% | +6.9 | −0.4 | +8.5 |

**The honest first-harmonic channel on this morphology is ≤ 0 — strictly negative in
cross-dt worst-case.** Off the trained dt the harmonic has a real, amplitude-proportional
physical cost (~+2 mNm per 0.5% depth); at the trained dt that cost is exactly masked by
an integration-error term of opposite sign, producing the flat "free" direction. Every
d > 0 is worse than the pure sinusoid under the doctrine. This closes the last charitable
reading ("harmonics work but the reward mistunes them"): with no optimizer in the loop at
all, a hand-injected harmonic reproduces the dt-specialization signature — the ~4 mNm the
runs "found" at trained dt was never a mistuned physical knob; the knob is negative.
Remaining formal gap: the marginal test holds tendons/pitch fixed at their
sinusoid-co-tuned values; a harmonic that pays only under joint re-tuning is not excluded
by this sweep alone — but three co-design runs plus this measurement all point the same
way. (Harness + raw data: `opt/harmonic_sweep.py`, `opt/harmonic_sweep_006.log`,
`opt/harmonic_sweep_xdt.log` in the worktree.)

**Crank phase lag** (`crank_phase_lag`: t4-vs-t1 crank qpos offset, sign-flipped on R;
`src/runtime/crank.py`): a structural parameter of the limit cycle — persistent
under the shared velocity command, no beat risk. Sweep on it009p (euler@0.06, 24
cycles): torque valley flat at 0–6° (154.4), then a **lift** knob at ~+1.6 mN/deg
(+12°: 157.4 pk / 355 lift; +30°: 172.7/378; collapse past 60°). Run 6 tested the
arbitrage thesis (buy lift via lag at ~0.32 mNm/mN, sell back via tendon softening at
0.5–1 mNm/mN): conversion ran ~10 iterations then collapsed → **lag is a lift knob, not
a torque knob; it009 had already extracted the corridor.** Lag is deliberately NOT in
theta: Adam moves ~lr per iteration in theta units and cannot traverse 6°→12° within a
run — sweep it globally, bake the optimum into the .tg, let the gradient adapt the rest
(coordinate descent, not joint descent).

## 14. Where everything lives (post-teardown)

All cloud compute torn down 2026-07-11; nothing is live or billing. Worktree
`~/Code/DiffSimAvesBot-fixed` (branch `fixed-codesign`) holds:

- **Baked, committed designs**: `designs/hummer/hover-soft-it009p/` (flagship),
  `checkpoints/it018/`, `checkpoints/it021-harmonic/`, `checkpoints/it015/` (all also in
  the main repo); lag-12 seed variant `hovering-tuned-lag12/`.
- **Raw artifacts** (gitignored, on disk): `opt/runs_lag12_5090/` (run 6: checkpoints
  it000–it057 + `robust_best` + log), `opt/runs_fixed4_5090_archive/` (run 4, incl.
  it039 and `robust_best_preharmonics`), `opt/runs_polish_5090_archive/` (run 5),
  checkpoint pulls in `opt/runs_fixed4_local/` (it015/it018/it021/it024) and
  `opt/runs_polish_local/` (it009/it021), and all run logs.
- **Warm-start genomes**: `opt/init_codesign_it018.json`, `opt/init_codesign_it009.json`
  (57-value layout of §4; `[ln(a0/65), a/a0×3, b/a0×3][ln sf×24][ln pf×24][ln msf, ln
  mpf]`, tendon order = `tendon_optimized.json` "pairs" key order).
- **Exact run configs**: `opt/run_fixed_codesign.sh` (run 4), `opt/run_polish_it018.sh`
  (run 5), `opt/run_lag12_it009.sh` (run 6).

To certify any design: reconstruct via `opt/design_io.py` (warmup=8), run the §12
doctrine battery with 24 sample cycles, require stationary per-cycle peaks. Trust
nothing scored on 3-cycle windows or Lp-smoothed peaks.
