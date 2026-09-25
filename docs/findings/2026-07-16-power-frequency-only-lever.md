---
title: At fixed frequency, control and tendon codesign cannot cut supplied power; frequency (a0) is the power lever
date: 2026-07-16
status: re-check
tags: [codesign, design]
---

## Finding
The hover rig does no braking work (inactive power Pin = 0 everywhere), so all ~22-24 W supplied
is aero work pinned by the lift requirement. With `--objective power` the optimizer found
nothing at fixed frequency (run 8: 57 iterations, +3 mN lift, +0.5% power; control gradients
NaN-spiralled after unlock), and a June power term `(P/P0)^2` moved power <1%. Frequency does
move it: baking a0 -2% (63.7 rad/s) gave -3.6% power (21.20 W at 341 mN), and a pure-sinusoid
envelope at 0.96x frequency gave 332.7 mN at 20.39 W (-7.3%) above the then 330 mN floor.
Measured elasticities vs frequency at the default damping: lift ~0.8, power ~1.8 (not 2 and 3),
so a frequency cut gives up lift at ~139 W/N. `--objective energy` (P/L) is dead the same way:
its first iterations buy lift with watts (run 7, and the Sep 10 screen).

## Evidence
Runs 7-8 (2026-07-15/16) and the June 30 / July 1 hover exploration; boxes destroyed, logs not
in the tree. Pre-2026-09 aero.

## Consequences
- To cut power, change frequency by coordinate descent: bake a0 into control.tg, then codesign
  the rest. A gradient on a0 moves only ~lr per iteration (`adam-moves-knobs-at-lr-per-iter`).
- The low lift elasticity (0.8) was the default tendon damping suppressing aero lift
  (`tendon-damping-lift-cost`): with fiber-like damping lift ~ f^2 returns, which makes a
  frequency cut much cheaper. Damping is the second power lever and is not in the genome.

## Status notes
Re-check: pre-aero numbers (the 330 mN floor is unreachable on today's aero), measured at
damping 0.5, and the "harmonics don't help power" part ran on the offset clock
(`harmonic-clock-offset`).
