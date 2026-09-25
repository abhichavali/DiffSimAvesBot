---
title: MuJoCo tendon range limits carry real load - any port or reduced model must include them
date: 2026-07-14
status: re-check
tags: [tendons, tooling]
---

## Finding
Structural tendons ride their upper range bound 22-100% of steps on the hover designs, and
designs with per-edge pretension > 1 rest pinned at the bound. A MATLAB/Simscape port that left
the limits out made the tuned design's R wing pitch anti-correlated with MuJoCo (-0.65). Adding a
one-sided limit spring/damper per tendon, with stiffness and damping taken from MuJoCo's soft
constraint linearized at the rest pose (via the tendon's effective mass), brought every
correlation to >= 0.80 and motor speeds to >= 0.99. Per-tendon forces then agree only to ~2x
(constraint coupling): a calibration knob, not a match.

## Evidence
2026-07-14 Simscape transfer fix. The `simscape/` port and `src/export_simscape.py` were never
committed and are not in the tree.

## Consequences
- Any port, reduced model or analytic estimate of the tendon net must model the range limits.
- The motor-tendon ranges have their own known bug (L/R asymmetric; `R_motor_t4`'s limit is
  active at qpos0; AGENTS.md Known bugs).
- Porting notes: aero-law parity with the port was exact (1e-12 fixtures) with added mass off
  (`added-mass-spurious-mean-lift`). `matlab -batch` on macOS 26 segfaults at exit: redirect to a
  log and check the output artifacts, not the exit code.

## Status notes
Re-check: the bound-riding fractions were measured on pre-2026-09 designs and aero.
