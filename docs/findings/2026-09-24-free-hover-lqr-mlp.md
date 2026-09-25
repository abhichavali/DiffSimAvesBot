---
title: A per-wingbeat LQR on the harmonic depths, distilled into an MLP, hovers the free bird with no fake forces
date: 2026-09-24
status: current
tags: [free-flight, cpu]
---

## Finding
The working closed loop: strobe a 12-state (COM position, attitude rotation vector, COM
velocity, world angular velocity, relative to the release orbit) once per wingbeat; linearize
one wingbeat by central finite differences on CPU (A 12x12, B 12x24 over the 24 harmonic
depths, a0 fixed); LQR with Bryson weights (2 cm, 0.05 rad, 0.1 m/s, 0.5 rad/s; input 0.05);
closed-loop |eig| <= 0.66. The LQR is distilled into an MLP 12 -> 64 -> 64 -> 24 (tanh, output
0.15 tanh) by regression on 20k push-sized states (relative MSE ~0.2%).

## Evidence
- Code: `src/hover/lqr.py`, `opt/distill_hover.py`, `src/hover/policy.py`; files
  `designs/hummer/free-hover/{control.tg, mlp_closed_loop.npz, lqr.npz}` (497f3c4, e844d95,
  82684ed, 1166b12).
- `opt/verify_hover.py`: 120 wingbeats (14 s), tilt <= 0.4 deg, ~11 mm drift; pushes at
  wingbeats 30/60/90 recovered (3.5 deg peak). `opt/push_sweep.py`: recovers side/forward
  0.8 m/s, up 0.6 m/s, roll/pitch 4 rad/s, yaw 8 rad/s.
- Re-derived end to end on a laptop (2026-09-24): trim 2.3 min, LQR 6 s, distill 20 s,
  60/60 wingbeats.

## Consequences
- The policy is valid only for the bird and trim it was derived on: re-run trim -> LQR ->
  distill -> verify after any change (`free-flight-policy` skill).
- State is simulator ground truth: position and velocity are not IMU-observable; a real build
  needs mocap or optical flow + range sensing.
- The MLP adds nothing over the LQR yet; it is the form a learned fine-tune would start from.

## Status notes
Current.
