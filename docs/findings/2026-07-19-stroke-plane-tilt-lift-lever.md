---
title: Stroke-plane DC tilt is a large lift lever, but peak torque is speed-inelastic, so tilt-for-frequency cannot lower torque
date: 2026-07-19
status: re-check
tags: [design, codesign]
---

## Finding
Static sweeps of the crank-disk mount (tilt, pitch): DC tilt moves lift ~12% per 0.2 rad (+12% at -0.2 rad). Peak
torque is speed-inelastic: -8% frequency gives only -1.5% peak torque. So "tilt for more lift,
then lower the frequency" does not convert into a torque win. The learned stroke plane
(`--tune-strokeplane`) went for this lift lever (the v5 dive under the floor, see
`reward-lift-credit-fountain`) rather than a within-cycle torque shaping; the within-cycle
hypothesis was never confirmed before the runs stopped.

## Evidence
2026-07-18/19 stroke-plane campaign (scratch sweeps `sp_sweep.json`, `sp_coupled_sweep.json`,
not in the tree; box destroyed). Pre-2026-09 aero.

## Consequences
- How it works (still in the code): per DOF a mean + 2 harmonics of the L_t1 crank ANGLE (10
  genes, last in theta, tanh-bounded to `--sp-max`, default 0.3 rad), applied by rewriting the
  four `*_disk` body quats each step (MJX `body_quat`, CPU `runtime/disk_orientation.py`).
  Crank angle, not time, so it stays locked to the wingbeat if frequency moves.
- `strokeplane_optimized.json` is applied by `view_fixed_frequency` and the GUI only, never by
  certification (`design_io`): a stroke-plane "gain" is uncertified by construction.
- Its Adam steps need `--sp-lr-mult` (`adam-moves-knobs-at-lr-per-iter`).

## Status notes
Re-check: pre-aero numbers, and no certified stroke-plane result exists.
