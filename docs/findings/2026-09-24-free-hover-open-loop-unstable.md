---
title: The trimmed free bird is open-loop unstable (|eig| ~1.08 per wingbeat); no open-loop waveform hovers
date: 2026-09-24
status: current
tags: [free-flight, harmonics]
---

## Finding
The open loop is the phase-locked trim: shared crank speed a0 plus 3 harmonics per crank and a
hover tilt, chosen so the launch cradle carries nothing (< 0.5 mN / 0.02 mNm). Trimmed, the
bird still diverges: the per-wingbeat linear map has max |eig| 1.076, so errors grow ~7.6%
per wingbeat and the bird is lost after ~40-50 wingbeats. This is a property of the plant,
not of the waveform.

## Evidence
- `python -m hover.trim` (Gauss-Newton on theta + 2 tilt angles); `hover.lqr` prints the
  open-loop eigenvalues. Committed trim: a0 53.668 rad/s (8.54 Hz), depths <= ~7%.
- `opt/verify_hover.py --open-loop`: lost at wingbeat 38-47 across dt 0.10-0.16 ms.
- CMA-ES in the 21-dim null space of the trim Jacobian, re-trimming every candidate, cost per
  wingbeat = min(3, tilt/10deg + |v|/0.3 + |w|/1): best ~49 wingbeats
  (`designs/hummer/free-hover/control_open_loop.tg`).
- An earlier "70 wingbeats" was a dt artifact: the search snapped dt to the trim's rate, not the
  candidate's. At the candidate's own snapped dt it is ~49.
- Open-loop search lessons: raw flight cost is noise-bound (position is a pure integrator);
  a spectral-radius fitness exploited trim offsets unless every candidate is re-trimmed.

## Consequences
- Judge an open loop by how long it lasts, never expect it to hover; stability needs feedback
  (`free-hover-lqr-mlp`).
- Evaluate open-loop claims at the candidate's own snapped dt and across dt 0.10-0.16.

## Status notes
Current. July's version (free hover tumbles within ~15 wingbeats at every dt) was on the old,
heavier bird without phase lock or inertia (`hover-body-wrench-july`).
