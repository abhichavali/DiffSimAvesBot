---
title: Replaying an optimizer design is convention-bound - a/b are stored absolute, warmup must be 8 cycles
date: 2026-07-01
status: current
tags: [codesign, cpu, tooling]
---

## Finding
Rebuilding a runnable design from optimizer artifacts silently breaks unless every
convention holds:
- `theta_best.json` stores a/b ABSOLUTE (x a0); the CPU runner wants modulation DEPTHS
  (`cmd = a0 (1 + sum a_n cos + b_n sin)`). Passing them undivided re-multiplies by a0 ~ 63
  and the rollout explodes to ~4000 mNm (it000 hides it: its harmonics are ~0).
- Warmup must be 8 periods: the rig keeps building lift for ~8 cycles, and a 2-cycle warmup
  certifies a transient (175 vs the true 163 mNm).
- Tendon stiffness in the JSON is absolute; rest length = base spring length x
  pretension_factor on both range columns.
- A `.tg` CAN carry all 24 per-tendon values (`overrides:` block); run dirs use sidecar JSON
  only because the writer never emits overrides.

## Evidence
~2026-07-01 (dated from the session that built it): the reconstruction reproduced
`cpu_validation.json` to the decimal (it000 185.3; cpu_best 163.4 euler / 169.7
implicitfast; pre-2026-09 aero).

## Consequences
- Use the one canonical path: `src/codesign/artifacts.py` (`reconstruct_design`, `validate`,
  warmup=8 default, divides a/b by a0) via `opt/design_io.py`; never re-derive it.
- Baking a run dir into a `.tg`: the `run-codesign` skill "Promoting a result".
- Pre-2026-09-24 harmonic artifacts also need `--pre-clock-fix-dt` (`harmonic-clock-offset`).

## Status notes
Current (artifacts.py embodies all of it; the old `opt/replay_checkpoint.py` and
`MORPHOLOGY_PATH` env var are gone since the 2026-09 refactor).
