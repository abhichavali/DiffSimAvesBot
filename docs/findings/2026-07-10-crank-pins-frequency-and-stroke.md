---
title: The crank pins flap frequency and stroke amplitude - there is no resonance to tune, and an omega-only frequency knob is inert
date: 2026-07-10
status: current
tags: [design, codesign]
---

## Finding
Flap frequency equals crank speed exactly (a0 / 2 pi: a0 65 rad/s -> 10.345 Hz), and the crank
kinematically pins the stroke amplitude (342 mm tip stroke on the seed, pre-aero). So there is
no amplitude resonance: what looked like "resonance" is the tendon springs compensating the
transmission torque (`soft-tendon-torque-corridor`). The rig is drag-dominated (no braking
power, see `power-frequency-only-lever`), so there is no oscillator to tune either.
`--tune-frequency` (f_log) moves only the harmonic clock, not the crank: for a sinusoid peak
torque is identical at every omega, and once harmonics unlock it desyncs them from the crank.

## Evidence
- Frequency + stiffness sweeps, 2026-07-10 (`docs/history/codesign_campaign_2026-07.md` §12);
  `--tune-frequency` dropped from run 4 on.
- June 30 / July 1 hover exploration: freed `--tune-frequency` with lift floors 370 and 300 mN;
  f never left ~10 Hz (drifted +0.5%).
- Code: the harmonic clock is omega*exp(f_log + a0_log) (`Genes.clock_factor()`,
  `src/codesign/genome.py`); crank speed follows a0 only.

## Consequences
- Change flap frequency through a0 (`a0_log`, or bake `a0` into control.tg), never f_log.
- `--tune-frequency` still exists and its help text ("the resonance lever for torque") is
  misleading for this crank-driven rig: don't use it.
- Flight controls enforce a0 == 2 pi fundamental_frequency (`hover.policy`).

## Status notes
Current: kinematic, independent of the aero and clock fixes. Scope: a crank-driven, kinematically
pinned transmission; a direct-drive or series-elastic morphology could behave differently.
