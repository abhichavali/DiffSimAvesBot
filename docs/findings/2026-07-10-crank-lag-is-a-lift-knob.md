---
title: The t4-vs-t1 crank phase lag is a lift knob (~+1.6 mN/deg), not a peak-torque knob
date: 2026-07-10
status: re-check
tags: [design, codesign]
---

## Finding
`crank_phase_lag` (shoulder.tg, e.g. `crank_phase_lag = 6 deg` on the seed) shapes the limit
cycle without harmonics. Sweep on it009p (euler @0.06, 24 cycles): peak torque is flat from
-12 to +6 deg (154.4 mNm at 0 and 6 deg), then lag buys lift at ~+1.6 mN/deg while peak torque
rises: +12 deg 157.4 mNm / 355 mN, +20 deg 163.4 / 367, +30 deg 172.7 / 378; it collapses past
~60 deg.

## Evidence
- Run 6 "lag arbitrage" (July 2026): lag 12 deg baked into the seed, it009 genome warm start,
  harmonics locked. Thesis: tendons convert the extra lift back to torque at 0.5-1 mNm/mN.
  The conversion ran ~10 iterations then stalled (~-0.4 mNm per 25 iterations); it050 reached
  parity with it009p (~156 mNm CPU doctrine, 159.2 wide) with ~5 mN more lift. Killed at it057.
- `docs/history/codesign_campaign_2026-07.md` section 13; run artifacts in the archived
  `fixed-codesign` worktree tarball (`~/Code/DiffSimAvesBot-archive/2026-09-24/`).

## Consequences
- Use lag to buy lift when a design is short of it; don't expect a torque win.
- Lag is deliberately not a genome gene: Adam moves each gene ~lr per iteration and cannot go
  6 -> 12 deg within a run (`adam-moves-knobs-at-lr-per-iter`). Sweep it globally and bake it
  into the .tg, then codesign the rest (coordinate descent).
- On the hummer bench path the lag is not held during a run anyway (`crank-phase-lag-drifts`);
  the free-hover phase lock holds it.

## Status notes
Re-check: numbers predate the 2026-09 aero fixes (`aero-overpredicted-lift`). The sweep was
sinusoid-only, so the harmonic clock bug does not affect it.
