---
title: "Harmonics are dt-specialized and useless on this rig" (July verdict) was the harmonic clock bug
date: 2026-07-11
status: superseded
superseded_by: harmonic-clock-offset (the dt-dependence was the unreset clock); no harmonic campaign re-run since
tags: [harmonics, codesign, certification]
---

## Finding
As claimed in July: harmonics have two failure modes. (1) Transient-riding: small harmonics
excite a ~40-cycle beat in per-cycle peak torque. (2) Discretization-fitting: harmonic gains
(~4 mNm at the trained dt) reverse at other dts, replicated three times (old-campaign run 1,
run-4 it021/it039, run-5 it021: 147.7-153 mNm @0.06 but 162 @0.04/0.09), not fixed by dt-bank
rotation with jitter. A no-optimizer first-harmonic sweep on it009p (depth 0.25-2%, all phases)
showed the same signature, and the winning phase rotated with dt (180 deg @0.03, 45 @0.06,
270 @0.09, 315 @0.15), read then as Euler phase-error coupling. Verdict: harmonics dead for
torque, power and lift.

## Evidence
`docs/history/codesign_campaign_2026-07.md` sections 12-13 (pre-2026-09 aero). Run artifacts:
the `fixed-codesign` worktree tarball + branch bundle in `~/Code/DiffSimAvesBot-archive/2026-09-24/`.

## Consequences
What survives:
- Failure mode (1) and its fix are real and still the protocol: exact per-cycle peaks,
  24-cycle windows, stationarity gate, `--w-drift` (`torque-beat-window-parking`,
  certify-design skill). Large depths still look dangerous: 1.44% depth (it039) was beat
  poison at 0.04/0.09 (re-check on the fixed clock).
- The crank-lag result survives on its own: `crank-lag-is-a-lift-knob`.
What does not:
- The phase rotating with dt is exactly the clock offset k*omega*2000*dt. With the clock reset
  it021 certifies flat (euler 164.5 @0.06, 164.3 @0.03, today's aero) and is the best of the
  three re-certified designs (168.4 mNm, -12.8% vs seed; `designs/README.md`).
- The it009 (sinusoid) vs it021 (harmonics) "ablation pair" is not evidence against harmonics.

## Status notes
Superseded 2026-09-23/24 by `harmonic-clock-offset`. Whether harmonics help peak torque, power
or lift on the fixed clock and aero is open. The Sep 10 reward screen and the July wavelet run
carry the same flaw (`reward-screen-harmonics-dt`, `wavelet-basis-no-torque-win`).
