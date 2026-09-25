---
title: A linear lift credit, or a too-soft floor hinge, lets the optimizer buy score with the constraint
date: 2026-07-15
status: current
tags: [codesign]
---

## Finding
One reward-shaping bug class, seen twice:
- Linear credit: under `--objective power` (run 7) the trade's gain term paid lift above the
  floor at a fixed rate (~13 W/N equivalent). Any lift knob cheaper than that rate mints score:
  the first arm raised a0 +0.12% to buy +3.3 mN for +67 mW while the loss fell.
- Soft hinge: the stroke-plane run v5 (floor 345 mN, no trade) normalized the lift shortfall by
  floor^2, so a 16% shortfall cost 0.26 while the torque term paid 5.7. The run dove 60 mN under
  the floor through DC stroke-plane tilt (MJX torque -22%, every candidate rejected at the gates).

## Evidence
Run 7 caught at it003 (2026-07-15), fixed and restarted; the v5 dive on 2026-07-19 (checkpoint
it069, not in the tree). Pre-2026-09 aero numbers.

## Consequences
In `src/codesign/loss.py` today:
- No linear lift credit off the torque objective (`w_gain = 0` unless `objective == "torque"`).
- `--objective energy` puts lift in the denominator (P / L, floored at the lift floor), never
  as a credit.
- Walls are normalized by their tolerance (amplitude floor `A_smooth`; trade hinge
  `0.02 * lift0`). softplus(0) = 0.693 sits a hinge's penalty AT its threshold, so smooth on a
  small fraction (the energy floor uses 0.002 * floor). The removed `--objective lift` had
  exactly that bug at its power budget.
When adding a reward term, ask whether any knob can buy it more cheaply than its exchange rate.

## Status notes
Current: the fixes are in the code.
