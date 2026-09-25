---
title: The "1.2x MJX vs CPU torque offset" was a metric mismatch (Lp10 soft peak vs exact peak), not a simulator gap
date: 2026-07-09
status: current
tags: [certification, codesign, mjx, cpu]
---

## Finding
Every MJX gate scored `tau_hat`, a smoothed Lp(p=10) peak, while CPU gates scored the exact
max. For the seed Lp10 read 144 mNm vs exact 174: that ratio was the whole "persistent 1.2x
MJX<->CPU offset". There was no physics offset: with euler, aero substep 1 and the velocity
fix, MJX and CPU exact in-window peaks agree to < 1% at every dt, float32 or float64. Worse,
Lp10 is duty-cycle hackable: narrowing torque spikes lowers it without lowering the true
peak (a run-1 design @0.04 ms: Lp10 143 vs exact 188).

## Evidence
2026-07-09 local CPU + MJX-on-Mac replays of the seed and the run-1 winner (history log
`docs/history/codesign_campaign_2026-07.md` section 11; numbers pre-2026-09 aero). Protocol
fix: commit `ce66dca` (branch `fixed-codesign`, archived as a bundle in
~/Code/DiffSimAvesBot-archive/2026-09-24/branches/).

## Consequences
- `src/codesign/metrics.py`: `tau_hat` (soft-max over cycles of per-cycle Lp10) is only the
  gradient surrogate; `tau_peak` (exact per-cycle max) is what gates, selection and
  certification use. The EA was switched to `tau_peak` on 2026-09-24
  (`codesign-selection-bugs`).
- Judge any design by exact peaks on CPU (the `certify-design` skill), never by `tau_hat`.
- MJX is a faithful proxy once the metrics match: in-loop CPU checks are certification, not
  steering.

## Status notes
Current. Supersedes the June 30 note that `tau_hat` was an exact `jnp.max` (the code changed
in between) and the "simulator-implementation hacking" framing (`mjx-matrix-pass-failed-on-cpu`).
