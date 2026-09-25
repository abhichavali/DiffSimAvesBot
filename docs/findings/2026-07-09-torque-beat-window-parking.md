---
title: Small harmonics excite a ~40-cycle torque beat; a 3-cycle window let the optimizer park the beat minimum in it
date: 2026-07-09
status: current
tags: [certification, codesign, harmonics]
---

## Finding
Harmonics of <= 3% depth excite a lightly damped ~40-cycle beat in the per-cycle peak torque.
The warmup-8 / sample-3 protocol then used everywhere (training loss, MJX matrix, CPU gate,
holdouts, sweeps) sampled one beat phase, and Adam learned to park the beat minimum inside the
window at the trained dts. The reported peak became a lottery on the window start. The
"-15% winner" of run 1 was a +11-17% steady-state regression.

## Evidence
2026-07-09 (history log section 11; pre-2026-09 aero): the run-1 winner @0.06 ms reported
139.7-195.8 mNm depending on window start (a 40% spread); its per-cycle peaks climbed
143 -> 196 over 35 cycles. The seed settles fast (1% spread). Converged last-20-of-100-cycle
means: seed 168.5, winner 186.9 @0.06 / 197.9 @0.04. Protocol fix in `ce66dca` (archived
`fixed-codesign` bundle).

## Consequences
- Certification samples 24 cycles after 8 warmup and requires stationary per-cycle peaks (the
  `certify-design` skill, `cycle_lift.py` prints the per-cycle peaks and drift).
- Optimizer: `--full-sample-periods 24`, `--drift-gate 0.03` (per-cycle-peak stationarity),
  `--w-drift` (drift priced in the loss), `--warmup-bank 8 16 24` (random window starts).
- Never trust a peak from a short window on a design with harmonics.

## Status notes
Current as protocol. The within-run beat is independent of the clock; but the July reading
that "dt shifts the beat phase" explained the wild cross-dt tables is partly the
`harmonic-clock-offset` bug, which rotated every harmonic by a dt-dependent angle.
