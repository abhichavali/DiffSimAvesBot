---
title: A design that passed the full 12-case MJX integrator x dt matrix was 12-16% worse than the seed on CPU off the trained dt
date: 2026-07-08
status: superseded
superseded_by: smoothed-peak-metric-mismatch, torque-beat-window-parking, harmonic-clock-offset (the cause); the CPU-certification consequence stands
tags: [certification, mjx, cpu, harmonics]
---

## Finding
Robust hover run 1's `robust_best` (it125) passed the MJX 3-integrator x 4-dt matrix with
<= 13% spread, yet on CPU MuJoCo its worst-integrator peak was 148 mNm @0.06 ms (-15% vs the
seed's 174) but 196 @0.04 and 203 @0.09 (+12-16% worse than the seed). The regression entered
in the harmonics phase; the tendon-only phase was CPU-flat. It was read at the time as the
optimizer "parking its gains in an MJX/CPU physics gap".

## Evidence
2026-07-08, run 1 of the robust-hover campaign (log: `docs/history/codesign_campaign_2026-07.md`
section 7). The seed was CPU-flat, 168-176 mNm across euler/implicitfast/rk4 x dt 0.03-0.16 ms
(pre-2026-09 aero).

## Consequences
- The durable part: certification needs a CPU sweep over integrators AND dts; an MJX matrix
  is not a certificate. In-run: `--cpu-every` with `--cpu-timestep-factors`; offline: the
  `certify-design` skill.

## Status notes
Superseded as a diagnosis. The next day's root cause found no MJX/CPU physics gap under euler
(<1%): the "gap" was the smoothed-peak metric (`smoothed-peak-metric-mismatch`) plus a
3-cycle window parked on a torque beat (`torque-beat-window-parking`); the harmonics' dt
dependence was further inflated by `harmonic-clock-offset`.
