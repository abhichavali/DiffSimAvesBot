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
