---
title: The t1/t4 crank phase lag is not held by velocity servos; it drifts every wingbeat unless locked
date: 2026-08-19
status: current
tags: [design, cpu, free-flight]
---

## Finding
`apply_crank_phase_lag` (`src/runtime/crank.py`) sets the t1/t4 crank offset once; the two
cranks are velocity servos loaded at different points of the tendon cycle, so their mean
rates differ and the declared `crank_phase_lag` drifts. Drift per cycle = (mean torque t1 -
mean torque t4) / kv x T. The hummer path (runners, GUI plots, codesign) still sends one shared
velocity command and does not hold the lag.

## Evidence
- 2026-08-19, 120 cycles: 12 deg designs -0.065 deg/cycle (4.5 deg left), hovering-tuned 6 deg
  -0.054, eagle 160 deg -0.31 deg/cycle.
- 2026-09-03: it009p flagship +0.126 deg/cycle (L t4), mean servo torques 97 vs 74 mNm, kv 1:
  the formula predicts +0.126 exactly; the 6 deg lag is gone by ~cycle 50.
- 2026-09-24 free flight: L -0.073, R +0.048 deg per wingbeat, enough to slide the free bird out
  of trim; "stable" open-loop candidates still fell.

## Consequences
- Rollouts of <= ~40 cycles lose < ~3 deg; long runs (viewer, hundreds of `--periods`) are not
  measuring the lag the `.tg` declares.
- Two fixes exist: the eagle's `CrankDrive` (`src/runtime/crank.py`, trims t4 to hold q_t1+q_t4;
  used by `view_eagle` and the GUI impulse pane's `hold_phase`), and the free-flight encoder
  phase lock `ctrl = w + K_PHI (L_des - L)`, K_PHI = 20 /s (`src/hover/launch.py`, the flight
  control's `phase_lock` key).
- Porting either to the hummer bench path changes every certified number: a deliberate
  re-baseline, not a cleanup.

## Status notes
Current. Drift numbers predate the 2026-09 aero fixes (the mechanism does not).
