---
title: The deploy target is dt <= 0.06 ms on CPU; a 2x-coarser-dt CPU check is over-strict
date: 2026-07-07
status: current
tags: [certification, cpu, mjx]
---

## Finding
MJX and CPU MuJoCo diverge structurally at coarse dt (~0.12 ms) for aggressive designs, and
MJX-side timestep robustness (`--tri-timestep`) does not buy CPU robustness at 2x dt. The user
ruled that a CPU check at 2x the training dt is over-strict: the deployment target is
dt <= 0.06 ms on CPU, so CPU timestep variants are finer, not coarser (factor 0.5 -> 0.03 ms).

## Evidence
2026-07-07 (EA -> Adam validity campaign): MJX @0.13 ms ~161 mNm vs CPU @0.12 ms 273 mNm for an
aggressive design (others: 243, 289). Under the ruling, EA seed_1 was that campaign's flagship:
CPU euler 188 mNm @0.06 (-17% vs its 225.7 baseline), implicitfast 207, rk4 194, euler 182
@0.03, lift 345 mN (pre-2026-09 aero; long since superseded as a design by
`designs/hummer/hover-soft-it009p`).

## Consequences
- Certification doctrine (the `certify-design` skill): {euler, implicitfast, rk4} @0.06 plus
  {euler, implicitfast} @0.03 ms; the optional wide check adds 0.04 / 0.09 ms, not 0.12+.
- In-run CPU gate: `--cpu-timestep-factors 0.5` (fine dt only).
- A design that only fails at >= 0.12 ms is not rejected for that alone; 0.13 ms is the
  hummer `.tg` stability edge anyway.

## Status notes
Current as a decision. The divergence numbers predate the 2026-09 aero fixes and the
`harmonic-clock-offset` fix (which moved every harmonic design's behavior across dt); the
coarse-dt MJX/CPU divergence itself was not re-measured since.
