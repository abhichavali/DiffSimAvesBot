---
title: July free hover used a 6-DOF stabilizing body wrench and scaled gravity - not physical, removed
date: 2026-07-23
status: superseded
superseded_by: free-hover-lqr-mlp (real gravity, crank motors only; wrench stack deleted 2026-09-24)
tags: [free-flight]
---

## Finding
In July the free-flying it009p (73 g airframe) was "hovered" by a policy that wrote a bounded
6-DOF force + torque straight into the airframe freejoint (`qfrc_applied`), with gravity scaled
down to the flap lift (~5 m/s^2) and the stroke plane leveled. A PD initialization ("att8x_pos2":
kp_pos 6, kd_pos 2.5, kp_att 0.4, kd_att 0.1) held tilt < 4.3 deg and drift < 40 mm; attitude
authority was the key lever. Open loop the bird tumbled 90-180 deg within ~15 wingbeats at every
dt, and a cycle-averaged linear policy over shared crank speed / stroke plane could not catch it.

## Evidence
`opt/hover_policy.py`, `src/view_hover.py`, `src/runtime/hover_control.py`,
`designs/hummer/hover/hover_policy/` (last present at `9d9640e^`); `docs/flight_demos.org`.

## Consequences
- None of it was physical; the wrench stack, gravity scaling and render assists were deleted
  (9d9640e..3b08f3b). AGENTS.md "No fake forces": the bird is flown only through its four
  crank motors.
- Surviving lessons: a level stroke plane removes built-in forward thrust; per-crank harmonic
  authority (not shared crank speed) is what can steer (`free-hover-lqr-mlp`).

## Status notes
Superseded by `free-hover-lqr-mlp`. Numbers predate the 2026-09 aero fixes and the 226 mN bird.
