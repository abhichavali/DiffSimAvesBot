---
title: The co-design EA ranked on a smoothed torque, pinned a0, and several replay paths rebuilt the wrong design
date: 2026-09-24
status: current
tags: [codesign, certification]
---

## Finding
Found during the September refactor triage and fixed without moving any committed design:
- The EA ranked, capped and reported `tau_hat` (Lp10 soft peak), which runs 0.71-0.81 of the
  exact peak and reordered 8 of 16 genomes; it now uses the exact per-cycle `tau_peak`.
- `C_lc` was cut at the design period while a0 moves the crank period: a0 x0.97 / x1.03 read
  1000x worse, so the EA could not move a0 more than ~1%. Now cut at the crank's own period.
- Replay / warm start mismatches: `--lock-wings` dirs replayed on an unsymmetrized rig (361 vs
  346 mN); `--init-codesign` decoded seeds by position; the GUI GA -> Adam chain passed the
  wrong harmonic count and dropped geometry and the EA's control.
- `--cpu-timestep-factors` without `--timestep` re-checked the base dt (false pass); curriculum
  stages overwrote each other's checkpoints.

## Evidence
e8afbd8 (EA exact peak), bb6a492 (C_lc period), 06cc229 (lock-wings replay), a771dae (seeds
record their layout), 749b308 (GA -> Adam hand-off), 1379ef0 (cpu factors), b0b8c05
(checkpoint numbering). Tests: `tests/test_genome.py`, `tests/test_wing_lock.py`.

## Consequences
- Judge designs by `tau_peak` on CPU replay (the `certify-design` skill), never `tau_hat`.
- Warm-start files carry `layout`; older layout-less files are still read by position.
- Remaining codesign known issues (MJX window sized by design period, `--lift-floor`,
  `--wavelet` unloadable): `opt/README.md` Known issues.

## Status notes
Current. EA runs before 2026-09-24 may have picked the wrong winners; re-certify on CPU.
