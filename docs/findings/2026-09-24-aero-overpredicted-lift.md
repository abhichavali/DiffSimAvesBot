---
title: The pre-September aero overpredicted lift ~75% (edge-leading assumption, wrong wing geometry, rotational terms)
date: 2026-09-24
status: current
tags: [aero, cpu, mjx, certification]
---

## Finding
The blade-element aero (CPU `src/runtime/aero.py` and MJX `src/runtime/mjx/aero.py`) had
errors that all raised lift:
- Lift direction, center of pressure and the rotational-lift 3/4-chord point all assumed the
  hinge edge leads. The far edge leads ~40-50% of the stroke; there lift was mirrored.
- The aero put the hinge on the leading edge and started the strips at the hinge. The mesh's
  leading edge is 17.3 mm ahead of the hinge and its root 6.66 mm out; aero now takes strips,
  per-strip hinge offset and root offset from the mounted mesh (`wing_geometry.py`).
- Rotational lift was scaled by strip radius instead of width; rotational drag was a
  right-endpoint sum that never sampled the leading edge.

## Evidence
Seed `designs/hummer/hover` at dt 0.06, lift: 362 -> 235 (lift sign, d0d8940) -> 270 (CoP /
3/4-chord, 360d774) -> 185 (mesh geometry, ad23b4a) -> 184 (strip width, bba78f7) -> 189
(rotational drag, 0c667f8) mN. Certified peak torque 174.7 -> 193.1 mNm, lift 187-210 mN
(bistable). Tests: `tests/test_aero.py`. Per-commit tables: the refactor fix log.

## Consequences
- Every lift / torque number before 2026-09-24 is stale (`docs/history/`, pre-fix rows of
  `designs/README.md`). Re-run `opt/design_io.py` before comparing.
- Co-design still pays: re-certified it009p 170.5 (-11.7%) and it021 168.4 mNm (-12.8%) vs seed.
- `--lift-floor` default 300 mN is now unreachable (AGENTS.md Known bugs).
- `legacy-baseline` lift is positive (~110 mN); old "negative lift" claims are wrong.
- Change CPU and MJX aero together; check with `opt/diag_mjx_cpu_parity.py`.

## Status notes
Current. Side effects: `passive-pitch-bistable`; the rotational-drag fix made the seed's pitch
non-periodic at 0.13 ms / aero substep 4.
