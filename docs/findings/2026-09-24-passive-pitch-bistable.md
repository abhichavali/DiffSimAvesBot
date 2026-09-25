---
title: The seed's passive wing pitch is bistable (~190 vs ~210 mN lift modes); compare lift per mode
date: 2026-09-24
status: current
tags: [aero, certification, design]
---

## Finding
On `designs/hummer/hover` (pitch armature 1e-6) the passive wing pitch settles into one of two
periodic modes, ~190 mN or ~210 mN mean lift, depending on dt, integrator and aero hold rule. A
mean over a run can be either mode, or a mix. At the viewer/GUI cadence (0.13 ms, aero substep
4) the seed's pitch is not periodic at all; at 0.06 and 0.03 ms it is.

## Evidence
- Found while fixing the CPU aero hold rule (1d8c2be): at dt 0.03, substep 1 and the held rule
  stay at ~209-210 mN for 32 cycles, the old re-projection rule fell to ~190 at cycle 5
  (hover 0.03 lift 189.30 -> 210.23 in that commit).
- Seed certification (current aero): lift 187.3-210.2 mN across the doctrine matrix.
- Per-cycle probe: `.claude/skills/certify-design/cycle_lift.py` (seed euler 0.03: 24 cycles
  at 209.4-211.0 mN).
- The free-hover design sets pitch armature 0: flapping is periodic at dt 0.06 and 0.13
  (`designs/hummer/free-hover/shoulder.tg`; the 1e-6 was numerical padding).

## Consequences
- Never compare one run's mean lift against another's: compare per-cycle lift, per mode.
- A "lift gain" from a code or design change may just be a mode switch.
- The held trim of the free bird jumped between two wrench clusters for the same reason.

## Status notes
Current for the seed family. Not observed on free-hover (armature 0).
