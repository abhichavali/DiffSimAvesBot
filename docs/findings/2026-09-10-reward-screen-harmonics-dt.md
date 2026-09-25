---
title: Torque-shape rewards (swing, swing_rms) gained only via harmonics and lost it at other dts (Sep 10 screen) - the clock bug again
date: 2026-09-10
status: superseded
superseded_by: harmonic-clock-offset (training shared one settled state; CPU re-settled per dt, rotating the harmonics)
tags: [codesign, harmonics, certification]
---

## Finding
As claimed on Sep 10: with `--lock-wings` (one shared 7-gene Fourier theta for both wings) on
it009p (354 mN, 154 mNm, 22.15 W; pre-2026-09 aero), a screen of harmonic authority x
cross-cell gradient consensus ranked tau_swing highest (consensus 0.996; tau_peak had authority
but consensus 0.39). After 22 iterations the MJX bank certified swing -7.9% and swing_rms -7.1%,
all of it from harmonics (a0 moved < 0.4%). On CPU swing_rms held at dt 0.06 (swing -13.5%,
peak -5.4%) but vanished at 0.03 (-1.0% / +1.2%). A variable-dt retrain (bank 0.03-0.13) still
gave -8.6% @0.06, -1.0% @0.03 and swing +7.2% @0.02. Read then as "harmonics fit the Euler phase
error of whatever dt they see". The energy objective (P/L) bought lift with watts: dead.

## Evidence
2026-09-09/10 campaign; screen and certification scripts were never committed and the box is
destroyed. Artifacts were fetched to a local `opt/runs_campaign_sep10/` (not tracked).

## Consequences
What survives: the energy objective is dead (`power-frequency-only-lever`); `--lock-wings` and
the torque-shape objectives exist in the optimizer (opt/README `--objective`); the user's rule
that ~22 iterations is enough to tell whether a reward moves.

## Status notes
Superseded 2026-09-23/24: a dt-dependent loss of a harmonic gain is exactly the
`harmonic-clock-offset` signature. Whether swing / swing_rms gains transfer across dt on the
fixed clock and aero is open; re-run before citing either way.
