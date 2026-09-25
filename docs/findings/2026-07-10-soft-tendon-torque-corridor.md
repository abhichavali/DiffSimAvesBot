---
title: Peak torque drops by softening the tendon net below the seed's stiffness (the "soft corridor")
date: 2026-07-10
status: re-check
tags: [tendons, codesign, design]
---

## Finding
Uniform 0.8x tendon stiffness, with no optimization, certified 166.3 mNm worst case on CPU
(-4.7% vs the seed's 174.7) at 357 mN lift, stationary across all integrators x dt 0.04-0.13 ms.
The seed sits under a torque cliff in the other direction (1.5x = +22%). Codesign started in the
corridor (run 4 warm-started at 0.8x) produced it015 (-9.3%), it018 (-10.1%) and the flagship
it009p (-10.4%, stiffnesses ~0.81x the seed's).

## Evidence
- `docs/history/codesign_campaign_2026-07.md` §12 (pre-2026-09 aero).
- After the aero fixes the flagship still certifies -11.7% (`hover-soft-it009p` 170.5 vs seed
  193.1 mNm; `designs/README.md`), so the direction survives.
- The June 30 / July 1 exhaustive hover exploration (control Fourier + tendons at fixed
  amplitude and frequency) stalled at a ~157 mNm euler floor, the level the July flagship
  reached (156.5): at a fixed wing, amplitude and frequency this looks like the codesign limit.

## Consequences
- Start tendon codesign from a softened seed; `designs/hummer/hover-soft-it009p` is that result.
- Earlier runs never entered the corridor because the lift trade was anchored at the seed's lift
  (`lift-trade-anchored-at-floor`).
- Remaining untried levers named then: wing morphology (area / planform, where the aero work
  lives) and feathering (`--tune-pitch`); tendon damping turned out to be a third
  (`tendon-damping-lift-cost`).

## Status notes
Re-check: the 0.8x and ~157 mNm numbers predate the 2026-09 aero fixes (`aero-overpredicted-lift`);
the corridor's direction is confirmed by the re-certified flagship.
