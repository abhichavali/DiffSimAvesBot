---
title: Rotating integrator x dt mini-batches train robustly, but selection must happen only on the full-matrix gate
date: 2026-07-07
status: current
tags: [codesign, certification, mjx]
---

## Finding
`--rotate-batch K` trains on a random K-variant subset of an (integrator x dt) bank each Adam
step (each batch has >= 2 integrators and >= 2 dts), giving worst-case training at ~K rollouts
per step instead of the whole matrix. Because a batch is an optimistic subset, per-iteration
"best" selection is disabled in rotate mode: a design is promoted only by the periodic
full-matrix evaluation (`--full-check-every`), which requires bounded spread, amplitude in
band, lift held, and stationarity. Holdout dts are never trained on and are evaluated once at
the end: the only unbiased generalization estimate.

## Evidence
Added 2026-07-07 (branch `june-26-demo`), canonical config: K = 3 harmonics, dt bank
{0.04, 0.06, 0.09, 0.13} ms, holdout {0.03, 0.11, 0.16} ms. Run 1 still produced a design that
passed the MJX matrix and failed on CPU (`mjx-matrix-pass-failed-on-cpu`), which led to the
exact-peak metric and longer windows (`smoothed-peak-metric-mismatch`,
`torque-beat-window-parking`).

## Consequences
- Code: `src/codesign/train.py` (`build_rotate_batches`; `select_here` is off when rotate +
  full-check are on), flags in `opt/optimize_control.py` (`--dt-bank`, `--n-batches`,
  `--dt-jitter`, `--warmup-bank`, `--full-sample-periods`, `--drift-gate`, `--holdout-dts`).
- Winner priority: `robust_best` (MJX-gated) > `cpu_best` > MJX best; only CPU certification
  makes any of them a result.
- Startup is compile-heavy (one forward jit per matrix cell, one jacfwd per batch): expect
  30-50 min on a GPU before the first iteration line.

## Status notes
Current. The original motivation for `--control-after` (train tendons first because
"harmonics are where dt tricks live") was largely the `harmonic-clock-offset` bug.
