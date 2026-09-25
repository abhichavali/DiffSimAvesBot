---
title: Backprop through MJX diverges past one wingbeat; one-wingbeat forward-mode gradients are accurate
date: 2026-09-24
status: current
tags: [mjx, free-flight]
---

## Finding
Pathwise gradients through more than one wingbeat of this model are useless: stiff tendons,
tendon limits and the dead-zone wing pitch make the rollout piecewise and chaotic. Forward-mode
AD over a single wingbeat with the full solver is accurate and can build linear models.

## Evidence
- July 2026 (old wrench-based hover): BPTT gradients reached ~1e11 at every horizon and start
  unless initialized from a stabilizing PD.
- 2026-09-24, `opt/train_hover.py` (MJX, `iterations=1` so reverse mode can trace the solver):
  from scratch over 4 wingbeats loss 1.5 -> 3.1, |g| 1e2-5e5; fine-tuning FROM the working
  LQR-MLP over 8 wingbeats loss 2.2 -> 3.3, |g| 1e5-7e8 in 8 iterations.
- One-wingbeat `jacfwd` with the full solver (iterations 100): B matrix vs CPU finite
  differences median column cosine 0.978, relative error 0.31, ~25 s on an RTX 5090. With
  `iterations=1` it drops to cosine 0.83, relative error 0.82.

## Consequences
- Do not train hover policies by BPTT over multiple wingbeats. Use per-wingbeat linearization
  (CPU finite differences or one-wingbeat `jacfwd`) with LQR / gain scheduling, or
  gradient-free methods (ES).
- `opt/train_hover.py` and `hover.env` are kept as the negative result.
- Co-design is unaffected: it uses forward-mode `jacfwd` over a small parameter count.

## Status notes
Current.
