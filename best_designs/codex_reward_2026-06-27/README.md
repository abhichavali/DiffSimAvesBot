# Codex Reward Designs - 2026-06-27

Branch: `june-26-demo`

These are copied from the isolated Codex 5090 worktree:
`/workspace/DiffSimAvesBot-codex-reward/opt/codex_runs`.

## robust_seed_optimizer_best

Status: optimizer-side best checkpoint. CPU/MJX validation was not available at copy time.

Source run:

```text
opt/reward_sweep.py --iters-per-stage 80 --joint-refine-iters 40 --warmup-periods 3 --sample-periods 3
```

Optimizer metrics:

```text
mean lift: 247.3 mN
tau_hat: 201.0 mNm
C_lc: 0.310
frequency: 10.0737 Hz
```

Files:

```text
robust_seed_optimizer_best/control_optimized.tg
robust_seed_optimizer_best/theta_best.json
robust_seed_optimizer_best/motor_tendon_optimized.json
robust_seed_optimizer_best/pitch_optimized.json
robust_seed_optimizer_best/morphology_optimized.tg
```
