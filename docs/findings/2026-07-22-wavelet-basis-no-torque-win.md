---
title: A localized (dyadic Ricker wavelet) crank-speed basis behaved like Fourier harmonics - it banked lift, no peak-torque win
date: 2026-07-22
status: re-check
tags: [codesign, harmonics]
---

## Finding
`--wavelet` reinterprets the 2K control coefficients as amplitudes of a dyadic Ricker
(Mexican-hat) frame over crank phase (sigma = pi, pi/2, pi/4), same genome width as Fourier.
Run from the it009p flagship (baseline reproduced exactly: 347.6 mN, 156.5 mNm) with a one-sided
amplitude floor (`--amp-floor 0.314`, stroke free to grow). Through it030 the optimizer spent the
freedom on lift (347.6 -> ~352 mN) and stroke (330.8 -> ~333.5 mm) while exact peak torque crept
UP (156.5 -> 157.6-158.4; MJX full check it025: robust 158.4 mNm, +1.2%). Clipped before the CPU
cross-dt gate.

## Evidence
2026-07-22 run on a destroyed box; log not in the tree. Pre-2026-09 aero.

## Consequences
- `--wavelet` is still in the optimizer (`src/codesign/genome.py` `ricker_atoms`), but its
  `control.tg` cannot be loaded by any runtime tool and `design_io` fails on its
  `theta_best.json` (opt/README Known issues): a wavelet result cannot be certified as is.
- No reason to retry without a new hypothesis, but see the status note.

## Status notes
Re-check, inconclusive: pre-aero numbers, and the atoms are evaluated at phase
`omega_eff * data.time` (`src/codesign/metrics.py`), the same clock that carried the settle
offset (`harmonic-clock-offset`). "Same verdict as harmonics" inherits that verdict's flaw
(`harmonics-verdict-july`).
