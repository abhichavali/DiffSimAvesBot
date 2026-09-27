---
title: At datasheet IMU noise the pos+IMU hover holds once its LQG is designed with the noise model; IMU-only hover holds attitude for ~14 s but not level or position, and the accel bias is what limits it
date: 2026-09-26
status: current
tags: [free-flight]
---

## Finding
`hover.sensors` noise (BMI088 gyro/accel, 0.3 mm mocap) x noise_scale, Mahony filter on, 120 wingbeats,
16 seeds x dt 0.10/0.13/0.16 ms x (no push, verify pushes); sweeps 2-8 seeds per dt. "good_np" = share of
NO-PUSH flights that fly every wingbeat with tilt max and last-20 tilt < 2 deg.
- **The accel bias is the term that breaks policies reading the filter attitude.** 20 mg per axis (1x)
  leaves the filter ~1.4 deg off level and the bird hovers at that tilt. Gyro/accel white noise are harmless
  to 10-30x; uncalibrated gyro bias hurts from 3x; mocap noise (to 100x) is harmless.
- **Noise-free designs fly but tilt at 1x**: good_np 79% (full-state LQR), 88% (P LQG), 85% (N SOF); the
  failures are a steady tilt (every push recovered relative to it). Crashes from 2x (LQR) and 5x (P LQG).
- **Track P**: `hover.reduced --obs pos,att,w --mode lqg --noise-scale 1` (V from the noise model, rho 0.668):
  good_np 100% at 1x and 2x, 95% at 5x; tilt max median 0.55 deg, drift median 15 mm; all 19 pushes at
  every dt; same with 5 ms mocap latency (latency-aware or not) and over 240 wingbeats; its MLP (Kalman
  filter kept in the loop) too. Crashes from 10x. Full state, same recipe: 100% at 1x, 98% at 5x.
- **Track N**: `att,w` SOF hovers every flight to 5x, but cannot see the tilt: good_np 85% at 1x, drift
  0.66-0.74 m median in 14 s. Fixes: the `--noise-aug 1` SOF MLP (94%), an accel calibrated to 5 mg on a
  level cradle (100%, unchanged SOF), or a raw `gyro,acc` LQG with no attitude filter (100% to 2x, 94% at
  5x; its `--noise-aug 1` MLP also recovers the 8 rad/s yaw push at dt 0.16, which the LQG loses).
  **But the gyro,acc policies are good only for ~14 s**: at 240 wingbeats 8/24 are good, at 480 0/18,
  noise 0 or 1 alike: a slow deterministic spiral (yaw 21-74 deg, 0.25-0.87 m/s, reaches the floor).

## Evidence
`opt/runs_imu/noise/` (gitignored): `*.jsonl` (one row per flight, ~22k), `table.md` / `report.json`
(`report.py`, columns good / good_np / seeds), `noise_mc.py`, `build.py`, `ctl/`, `mlp/`, `phase*.sh`
(exact commands). Re-check on fresh seeds: `opt/runs_imu/verify_noise/` (`V1`, `V2_240`, `V3_sweep`, `W`,
`X_480` jsonl). Committed-CLI equivalent: `opt/verify_hover.py --ctrl ctl.npz --noise-scale 1
--pos-latency-ms 5 --dt-ms 0.10 0.13 0.16` and `opt/hover_mc.py`.

## Consequences
- Design every LQG with the flight's noise model (`--noise-scale 1`), not at noise 0.
- A position-free bird needs its accel bias calibrated (level cradle) or an external level reference, and
  still walks; IMU-only hover is a seconds-scale result, not station keeping.

## Status notes
Verified 2026-09-27. Corrected then: the "good%" in the first tables gated tilt on no-push flights only
(verify flights counted if they survived), which halved the tilt failures (N SOF was quoted 93%, is 85%);
noise_mc reuses a seed across dt/scenario/config, so each rate rests on 16 bias draws, not 96 flights; the
3-40 ms latency sweep ran only on the noise-free P LQG. Model gaps: V gives `w` only gyro noise, but the
filter's tilt error rotates the orbit's body rate (w_z error ~0.09 rad/s, 10x V); the filter's orbit-offset
calibration uses the true "up"; gyro bias 1 sigma is 0.5 deg/s vs the datasheet's +-1 deg/s typ (affects only
the uncalibrated gyro-bias rows). The bias uses +-20 mg as 1 sigma (pessimistic). Linear model from dt 0.13.
