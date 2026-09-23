# Control-signal optimization

Gradient-based warm-start: optimize only the Fourier **velocity-servo** command
(angular velocity driving the crank motors) at fixed 10 Hz, holding the
morphology/tendons fixed. Objective and curriculum follow `aves_control_diff.org`.

```
R = wL*PhiL(Lbar) - wtau*Ptau(tau_hat) - wK*Ckin - ws*S - w_lc*C_lc   (maximized)
```

- **PhiL** — smooth lift **floor** (default `185 mN`, `--lift-floor`): softplus hinge
  penalty below the floor, none above (keeps lift margin). Optional `--lift-cap` adds
  an upper penalty. `--lift-smooth` sets the hinge softness.
- **Ptau** — peak motor torque over budget; `tau_hat` = exact recorded max of
  `|actuator_force|` across the 4 crank actuators, matching CPU validation.
- **Ckin** — stroke-amplitude band; `A*` is bootstrapped from the design signal's
  wing-tip z-excursion so the optimizer can't collapse the flap to dodge torque.
- **S** — smoothness, `sum n^2 (a_n^2 + b_n^2)`.
- **C_lc** — limit-cycle stability: relative cross-cycle variance of the lift
  trace (`var_across_cycles / mean_square`, dimensionless, 0 ⇔ stable orbit).
  `--w-lc` (default 1.5); needs `--sample-periods >= 2`. With the aero substep fix
  (below) the design gait is already mostly periodic (`C_lc≈0.25`); a modest weight
  locks it in.

### `--objective` — what actually gets minimized

- **`torque`** (default) — robust peak motor torque, the original behaviour above.
- **`power`** — robust mean **supplied** motor power `m[8]` (an exact window mean, so
  unlike an Lp peak it can't be narrowed by duty-cycle tricks), normalized by the
  baseline. Peak torque drops to a soft ceiling guard (`--tau-guard`, default 4).
- **`energy`** — robust **specific power** `P/L` [W/N]: the same watts, credited by
  the lift they buy. `P_obj = (P_worst / L_den) / (P0/L_den0)`, normalized to ~1 at
  the seed, where `L_den = lift_floor + softplus_hinge(L_worst - lift_floor)`.
  `P_worst` is the soft-max over the batch/matrix variants and `L_worst` the
  soft-min, so both extremes push the ratio up — a true worst case.

Why the ratio and not `P - λ·L`: a **linear** credit fixes the exchange rate, so any
knob that buys lift below `λ` W/N is a score fountain, which is exactly what run 7
farmed (a0 up, +67 mW of watts booked as "lift"). The ratio prices lift at the
design's *own current average* `P/L`, so a lift knob only pays if it beats the
efficiency already achieved, and the reward is self-limiting (`d(P/L)/dL = -P/L²`).
The denominator is floored at `--lift-floor` so a collapsed stroke can't divide by a
vanishing lift; below the floor the objective degenerates to plain power with the
floor wall (`wL·PhiL`) pushing lift back up. Do **not** add `--lift-torque-tradeoff`
on top — the ratio *is* the trade, and the linear credit would re-open the fountain
(`make_loss` zeroes `w_gain` off the torque objective for exactly this reason).
The hinge smooths on `0.002·lift_floor`, not `0.02`: `softplus(0)=0.693`, so a wider
scale offsets the objective at the floor (the run-9 hinge-normalization bug).

Selection scores and the full-matrix / CPU transfer gates all follow the same metric,
and under `power`/`energy` a power spread wider than the gap threshold rejects a
candidate the way a torque spread does. Self-check: `uv run python opt/test_reward.py`.

### `--lock-wings` — one design for both wings

The control signal was always shared (one scalar command drives all four crank
actuators), but the morphology was not: `structural_tendon_pairs(individual=True)`
gives each of the 24 structural tendons its own gene, and the seed `.tg` bakes very
different L/R values on top (`hovering-soft-it009p`: `L_t1_to_t1_2` 184 N/m vs `R`
341 N/m). Every previous run therefore optimized **two different wings** sharing one
command — the two wingtips traced paths 6.5% of the stroke apart.

`--lock-wings` geometric-means each L/R tendon pair at load (stiffness is a
multiplicative knob; rest length is arithmetic) and tunes the 12 **mirrored** pairs —
24 params instead of 48. With it the two wingtips are mirror-identical to 0.0000 mm
and the two shoulders carry identical torque. Costs almost nothing: CPU baseline
62.6 W/N locked vs 63.3 W/N unlocked. Check: `uv run python opt/test_wing_lock.py`.

This also needed a real bug fixed in `src/runtime/crank_sync.py`: the sync paired
`L_t{a}` with `R_t{b}` across the sigma station swap (1↔4) and copied the angle
**unnegated**, which took a mirror-correct settled state (+0.330° / −0.330°) and
un-mirrored it. `R_t{s}_disk` is the Y-mirror of `L_t{s}_disk` at the *same* station,
so the mirrored angle is the negated one. This changes startup for every design.

### Aero substep (the CPU↔MJX lift fix)

`--aero-substep` (default **4**) recomputes aero every N steps and zero-order-holds
between, with the FD window = `N·dt` — matching the CPU runtime. The MJX default was
substep=1 (every step, 1·dt FD), which injects huge added-mass noise (RMS ~2 N vs
~0.2 N): it *halved* mean lift (106 vs 325 mN) and manufactured apparent chaos
(`C_lc` 0.99 vs 0.25). Verified that same-state aero force, pitch torque, integrator,
solver, and float64 all match CPU — the gap was purely the substep cadence. A residual
CPU↔MJX gap remains from the `implicitfast` integrator (Euler matches per-step,
implicitfast doesn't); surfaced at CPU-validation, not a blocker.
- **Curriculum** — `tau_max`: 400 → 200 → 100 → 80 mNm.

Gradients: MJX rollout, **forward mode** (`jax.jacfwd`) — memory-light for the
2K+1 params and keeps the full-accuracy solver. Optimizer: hand-rolled Adam.

## Run

Local (CPU, slow — use for a smoke test):
```sh
uv run python opt/optimize_control.py --warmup-periods 0.2 --sample-periods 0.2 \
    --iters-per-stage 1 --harmonics 2
```

On the vast 5090 (syncs the private repo via rsync, installs CUDA jax, runs,
pulls `opt/runs/` back):
```sh
opt/sync_and_run.sh                          # defaults: K=5, 40 iters/stage
HOST=ssh7.vast.ai PORT=26920 opt/sync_and_run.sh --iters-per-stage 60
```

## Phase 2 — structural-tendon co-design

Optimize each structural tendon's **stiffness + pretension**, tied left=right
(12 pairs → 24 params), perturbing the MJX `tendon_stiffness` / `tendon_lengthspring`
arrays directly as traced inputs (gradient-optimizable; no recompile). Control is
warm-started from phase 1 and frozen:

```sh
opt/sync_and_run.sh --tune-tendons --freeze-control \
    --init-control opt/runs/theta_best.json --out opt/runs_tendons --iters-per-stage 60
```

Drop `--freeze-control` to refine control jointly. Output: `tendon_optimized.json`
(per-pair stiffness, stiffness/pretension factors, rest length).

## Outputs (`opt/runs/`)

- `control_optimized.tg` — best coefficients, design frame. Use it:
  `MORPHOLOGY_PATH=... uv run python src/run_mjx_fixed_frequency.py` after pointing
  `design/shoulder.tg`'s `control_file` at it (or copy over `design/control.tg`).
- `theta_best.json` — raw params (actuator frame) + omega.
- `log.json` — per-iteration loss / Lbar / tau_hat / A_stroke / S.

## Knobs

`--harmonics K` (default 5), `--lift-target N`, `--weights wL wtau wK ws`,
`--lr`, `--iters-per-stage`, `--warmup-periods`, `--sample-periods`.
