---
title: Noise-aware hover tuning that did not help - Kalman bias states, heavy noise augmentation, a fixed-noise design, reweighted SOF, the att,w LQG
date: 2026-09-26
status: current
tags: [free-flight]
---

## Finding
Against the plain recipe (LQG with V = the flight's noise, or the SOF MLP distilled with `--noise-aug 1`),
at datasheet noise with the Mahony filter:
- **Bias states in the Kalman filter** (`hover.reduced --bias att_xy`, a random-walk bias on the attitude
  channel, meant to learn the accel-bias tilt): worse. P at 2x 69% good vs 100%; N at 2x 62% vs 88%; the P
  closed loop becomes slow (rho 0.95). Screen at dt 0.10, 16 seeds.
- **Heavier noise augmentation** when distilling (SOF `att,w` MLP aug 3 / 10 vs 1): 62% vs 88% good at 1x
  (dt 0.10). Augmentation does nothing for the P LQG MLP (aug 0, 1, 3 equal).
- **Designing for the wrong noise level**: P LQG with V fixed at 1x drops to 58% no-push good at 5x; V
  fixed at 5x is robust to 5x but too sluggish for the 8 rad/s yaw push at 1x (144/152 at dt 0.16).
- **Reweighted SOF `att,w`** (q_w = 1): 25% good at 2x vs 72%.
- **Noise-aware LQG on `att,w`** (track N): worse than the plain SOF. It sinks faster (median -92 mm
  after 120 wingbeats) and 2/16 seeds reach the floor within 14 s: 88% no-push good at 1x, 47% at 5x.

## Evidence
`opt/runs_imu/noise/` (`C1.jsonl`, `C2.jsonl`, `E.jsonl`, `local_mlp_screen.jsonl`, `table.md`,
`ctl/P_lqgV1_bias.npz`, `ctl/P_lqgV5.npz`); rates here use the older "good" (tilt gate on no-push
flights, survival on pushed ones) unless marked no-push.

## Consequences
Match V to the expected flight noise; distill with `--noise-aug 1` at most. The accel bias is better
removed at the sensor (calibration) than estimated in flight without a position reference.
