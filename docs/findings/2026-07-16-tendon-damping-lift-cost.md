---
title: Structural tendon damping 0.5 N s/m (an unexamined scaffolding default) costs lift, not watts
date: 2026-07-16
status: current
tags: [tendons, design, aero, codesign]
---

## Finding
Structural tendon damping 0.5 N s/m (motor tendons 1.0) came from the first compiler commit
(15764ad "Add barebones morphology compiler"); nothing ever swept it and it is not a genome gene.
The dampers are not where the power goes (they dissipate 1.41 of 21.97 W, 6.4%), but they
suppress LIFT: less structural damping gives much more lift at almost the same power. The July
"design-invariant 62-64 W/N specific-power wall" was this default, not physics.

## Evidence
Sweep on it009p, doctrine windows 8+24 cycles (pre-2026-09 aero):

| c_struct | c_motor | P (W) | lift (mN) | W/N | vs default |
|---|---|---|---|---|---|
| 1.00 | 2.00 | 22.88 | 316.5 | 72.3 | +13.6% |
| 0.50 | 1.00 | 21.97 | 345.3 | 63.6 | default |
| 0.25 | 0.50 | 21.61 | 379.0 | 57.0 | -10.4% |
| 0.05 | 0.10 | 21.39 | 414.9 | 51.6 | -19.0% |
| 0.00 | 1.00 | 21.46 | 423.7 | 50.6 | -20.4% |

- Cross-dt robust (c_struct 0: 426.4 / 423.7 / 422.1 mN at dt 0.03 / 0.06 / 0.12 ms). Lift
  elasticity vs frequency rises 0.91 -> 2.01 as damping drops (lift ~ f^2 returns); power
  elasticity stays 1.83-1.91. Both dampers at exactly 0 is numerically unstable.
- Today's aero (free-hover work): 0.5 costs ~12% lift (`designs/hummer/free-hover/shoulder.tg`).

## Consequences
- Recipe: structural damping down, motor damping unchanged. Fiber tendons (Dyneema/aramid,
  loss factor ~1-2%, c ~ eta*k/omega ~ 0.05) capture ~95% of the gain; 0.5 means eta ~16%,
  i.e. elastomer/silicone tendons.
- `designs/hummer/free-hover` uses 0.05; the seed and every codesign result keep 0.5.
- Untested next step: damping as genome genes (24 structural + 1 motor), re-bake, re-certify.

## Status notes
Current as a mechanism (re-measured on today's aero); the table predates the 2026-09 aero fixes
(`aero-overpredicted-lift`). Open: what the real tendons are made of decides 0.5 vs 0.05.
