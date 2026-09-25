---
title: The C_valid stall / reduced-frequency thresholds must be calibrated to the baseline, or they freeze the EA
date: 2026-07-06
status: current
tags: [codesign, aero]
---

## Finding
The hovering flapper's baseline already operates far past textbook quasi-steady validity
limits: peak AoA ~77 deg and reduced frequency k ~0.33, versus the `C_valid` defaults
`alpha_stall = 45 deg`, `k_max = 0.10`. With the defaults every genome, the baseline included,
scores C_valid ~5, which flattens the EA fitness landscape. The term's job is only to stop a
design going FURTHER into invalidity (the lift hack), so its thresholds are set relative to
the baseline: 1.05 x baseline alpha_pk and 1.15 x baseline k.

## Evidence
2026-07-06 EA run: best fitness frozen at -5.14 for 30 generations, 3-4 of 512 genomes
feasible. After calibration, gen-0 best fitness -0.151 (pre-2026-09 aero).

## Consequences
- `opt/evolve_codesign.py` calibrates automatically and prints a `[calib]` line; it recomputes
  C_valid from raw `alpha_pk` / `k_eff` against those thresholds.
- `Metrics.C_valid` (`src/codesign/metrics.py`, `ALPHA_STALL` 45 deg, `K_MAX` 0.10) still uses
  the textbook defaults: the C_valid logged by `optimize_control` reads ~5 for every design
  and means nothing. It is logged only; the validity-penalized Adam stage was removed
  (opt/README Known issues).
- If C_valid ever re-enters a loss, calibrate it to the seed first.

## Status notes
Current (the EA code embodies it). The 77 deg / 0.33 operating point was measured on the
pre-2026-09 aero (`aero-overpredicted-lift` moved the strips and hinge offset); the
calibration is relative, so it self-adjusts.
