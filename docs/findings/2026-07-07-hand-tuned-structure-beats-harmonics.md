---
title: For peak torque, soften the most-loaded tendon family; large control harmonics made torque worse
date: 2026-07-07
status: re-check
tags: [tendons, design, harmonics]
---

## Finding
Hand-tuning the hover design on CPU euler took peak torque 227 -> 180 mNm and swing 204 -> 131
mNm at 340 mN lift (tip amplitude -9%). Three lessons:
- Every 1st/2nd-harmonic phase tried at |A| ~8-16 (absolute, on a0 ~63 rad/s) raised peak torque
  and/or swing: the velocity servo pays inertial torque to modulate the crank.
- Load is concentrated: at the mid-downstroke peak the M[1B]--F[1B] sigma family
  (`L_t1_2_to_t4_2` and its sigma image `L_t4_to_t4`) carries ~2x the tension of the other edges.
  Softening that pair to 60 N/m with defaults 200 -> 150 N/m gets most of the uniform-soft win at
  less lift cost.
- a0 buys lift almost torque-free in this regime: 63 -> 67 rad/s ~ +15 mN for +1 mNm peak
  (keep `fundamental_frequency` = a0 / 2 pi).

## Evidence
The July 7 `hand_tuned/` design, the ancestor of the seed `designs/hummer/hover` (lineage in
`designs/README.md`: hover-base -> hand-tuned -> hover, 24 per-tendon overrides). Per-tendon load
= k * (length - rest), rest from `model.tendon_lengthspring`.

## Consequences
- `.tg` overrides are per edge, not per sigma family: an override on M[1A]--F[4B] does not touch
  its image M[4B]--F[1A]; list both edges (sigma: stations 1<->4, 2<->3, faces A<->B).
  See the add-or-edit-design skill.
- Structure (stiffness) is the torque knob that worked; see `soft-tendon-torque-corridor`.

## Status notes
Re-check: numbers predate the 2026-09 aero fixes. The harmonic lesson was at large depths
(~12-25% of a0) and on the offset clock; each phase scan was at one dt, so the offset only
relabeled the phases, but small-depth harmonics are open again after `harmonic-clock-offset`.
