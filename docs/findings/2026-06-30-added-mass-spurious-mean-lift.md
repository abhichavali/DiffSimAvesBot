---
title: The added-mass (unsteady) aero term adds a spurious mean lift - keep it off
date: 2026-06-30
status: re-check
tags: [aero, cpu, mjx]
---

## Finding
Turning on added mass nearly doubled MEAN lift on the June flap baseline (244 -> 412 mN).
Reactive, acceleration-proportional forces should integrate to ~0 over a periodic cycle, so the
extra DC lift is an artifact, suspected rectification by the capped finite-difference
accelerations (`OMEGA_DOT_CAP` 5e4 rad/s^2, `LIN_ACC_CAP` 1e3 m/s^2 when enabled,
`src/runtime/aero.py`). Earlier "lift = quasi-steady + ~110 mN unsteady" framings were inflated
by it.

## Evidence
2026-06-30 flap-codesign runs (pre-2026-09 aero). The later MJX<->CPU agreement fix for added
mass (`mj_jacSite` velocities, ratio ~1.01) made the two sims agree; it did not address the DC
component.

## Consequences
- Every committed result is quasi-steady. Keep `--added-mass` (optimizer) and
  `set_unsteady_aero` (runtime.aero) off.
- Before enabling it: check that the cycle mean of the added-mass force alone is ~0.

## Status notes
Re-check: never re-examined after the 2026-09 aero fixes, which moved the strips, hinge offset
and leading edge (`aero-overpredicted-lift`).
