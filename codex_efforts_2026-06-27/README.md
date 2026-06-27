# Codex Reward Tuning Notes - 2026-06-27

This folder preserves my optimization work before the 5090 instance is shut down.

## What Is Here

- `remote_artifacts/robust_seed/`: best pre-power reward design from the original staged optimizer.
- `remote_artifacts/power_seed_added_mass/`: first power-reward run, before the added-mass discrepancy fix.
- `remote_artifacts/power_seed_quasisteady/`: current trusted power-reward run with added mass disabled.
- `remote_artifacts/power_reward.log`: old power run log with added-mass active.
- `remote_artifacts/power_reward_quasisteady.log`: final quasi-steady power run log.

Each design folder contains:

- `control_optimized.tg`
- `theta_best.json`
- `motor_tendon_optimized.json`
- `pitch_optimized.json`
- `morphology_optimized.tg`
- `log.json`

## Best Designs

### 1. `power_seed_quasisteady` - best trusted power design

Use this as the safest restart point. This run used Euler, CPU-matched aero substep cadence, power reward, and `added_mass=False`.

- Mean lift: `116.9 mN`
- Peak torque estimate: `133.1 mNm`
- Mean motor power: `7.29 W`
- Inactive power: `0.000312 W`
- Limit-cycle metric: `0.177`
- Frequency: `10.0737 Hz`
- `a0`: `34.6506 rad/s`
- Motor tendon stiffness factor: `1.2162`
- Motor tendon pretension factor: `0.9931`
- Upstroke pitch: `0.1079 rad`
- Downstroke pitch: `0.1095 rad`

Caveat: it did not clear the `150 mN` lift floor. The final logged iteration reached about `121.9 mN`, but the saved checkpoint selected lower inactive power instead of maximum lift.

### 2. `robust_seed` - strongest pre-fix lift/torque design

This was the best design before the added-mass discrepancy was understood.

- Optimizer-side mean lift: `247.3 mN`
- Peak torque estimate: `201.0 mNm`
- Frequency: `10.0737 Hz`
- `a0`: `33.3795 rad/s`
- Motor tendon stiffness factor: `1.0240`
- Motor tendon pretension factor: `0.9846`
- Upstroke pitch: `0.08585 rad`
- Downstroke pitch: `0.08369 rad`

Caveat: do not trust the lift number directly. It was found while the unsteady/added-mass term could inflate MJX lift relative to CPU.

### 3. `power_seed_added_mass` - invalidated first power run

This run is preserved for comparison only.

- Optimizer-side mean lift: `236.0 mN`
- Peak torque estimate: `208.5 mNm`
- Mean motor power: `7.27 W`
- Inactive power: `0.530 W`
- Frequency: `10.0737 Hz`
- `a0`: `34.0817 rad/s`

Caveat: same added-mass issue as `robust_seed`, so treat it as an invalidated experiment.

## Code Changes Made

Committed on `codex/reward-tuning-5090`:

- `595a73c Add motor power reward mode`
- `9bffc84 Disable added-mass aero by default`

Power reward mode replaced the peak-torque hinge with:

- mean absolute motor power penalty
- inactive/regenerative power penalty
- lift floor penalty
- kinematic target penalty
- Fourier smoothness penalty

Added-mass aero is now opt-in with `--added-mass`; default is quasi-steady aero.

## Learned Principles

- Added-mass/unsteady aero was the main MJX/CPU mismatch. With it disabled, CPU and MJX agree closely on the same trajectory.
- Euler is the trustworthy integrator for MJX gradients; `implicitfast` diverges from CPU at the per-step dynamics level.
- Aero substep cadence matters. MJX must recompute-and-hold aero like CPU instead of recomputing every tiny step with a different finite-difference window.
- High lift numbers from old added-mass runs are not transferable.
- Power reward successfully killed inactive power, but it also made the optimizer accept too little lift.
- Harmonic control coefficients are important; low learning rates on absolute `a0`/harmonics barely move them, so log/scaled parameterization was necessary.
- A hard lift feasibility gate or stronger lift weight is needed if the next run should prioritize clearing `150 mN` over minimizing power.
- Best current direction: restart from `power_seed_quasisteady`, keep added mass off, raise lift pressure, and only then optimize power.

## Minimal Restart Commands

Validate a saved design:

```bash
uv run python opt/validate_control.py \
  --control codex_efforts_2026-06-27/remote_artifacts/power_seed_quasisteady/control_optimized.tg \
  --periods 4 \
  --warmup-periods 4 \
  --integrator euler
```

Restart power optimization from the trusted quasi-steady design:

```bash
uv run python opt/optimize_control.py \
  --integrator euler \
  --power-reward \
  --init-control codex_efforts_2026-06-27/remote_artifacts/power_seed_quasisteady/theta_best.json \
  --iters-per-stage 120 \
  --warmup-periods 3 \
  --sample-periods 3 \
  --out opt/codex_power_runs/restart_from_quasisteady \
  --harmonics 4 \
  --lift-floor 0.150 \
  --tune-motor-tendons \
  --tune-pitch \
  --lc-max 0.40 \
  --lr 0.003 \
  --tau-curriculum-mnm 200 \
  --power-weights 20 1 3 1 0.08
```

Do not add `--added-mass` unless the goal is specifically to debug the unsteady term.
