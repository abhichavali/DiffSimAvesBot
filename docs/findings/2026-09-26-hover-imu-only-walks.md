---
title: IMU-only hover cannot hold position or altitude, and neither dead reckoning nor the true velocity fixes it
date: 2026-09-26
status: current
tags: [free-flight]
---

## Finding
Every track-N controller (no position channel) holds attitude but walks and sinks steadily:
- Noise-free SOF `att,w`: 26/19/14 mm/s horizontally at dt 0.10/0.13/0.16 (true attitude; 25/15/12 with
  the filter), sinking 2-5 mm/s; ~1 m and the floor (0.22 m below release) within ~40-70 s. LQG `att,w`
  sinks ~7 mm/s (biased vz estimate) and hits the floor after ~30 s, which shows up as 3-5 deg tilt bursts on
  contact (no saturation: |u| ~0.002).
- Under 1x noise the walk grows: 0.3-0.75 m in 14 s; the SOF's walk grows with the accel-bias tilt
  (3.2 m median at 5x).
- The raw `gyro,acc` LQG and its noise-augmented MLP (the most noise-robust IMU-only policy at 120
  wingbeats) spiral slowly: at 480 wingbeats yaw 21/45/74 deg and speed 0.25/0.53/0.87 m/s at dt
  0.10/0.13/0.16, tilt 4-5 deg, floor reached; 0/18 good, at noise 0 as well as 1.
- Not fixes: adding the true velocity (`att,w,vel`) still walks 0.35 m in 120 wingbeats (the orbit's
  constant drift term, no integrator). Dead reckoning (`hover.reduced --dead-reckon`, integral action on
  the predicted position) cuts 120-wingbeat drift 23% on `att,w` (60% where more velocity is measured) but
  still walks 0.4-0.8 m in 480 wingbeats and tilts 4 deg with the filter.

## Evidence
`opt/runs_imu/minimize/long.json` (480 wingbeats), `opt/runs_imu/verify_minimize/flights_long.json`
(240), `opt/runs_imu/verify_noise/X_480.jsonl`, `opt/runs_imu/noise/table.md`.

## Consequences
The 120-wingbeat gate hides all of this: judge a position-free policy over 240-480 wingbeats. Holding
position needs a position reference; altitude alone could come from a barometer standing in for `pos_z`
(every `pos_z` row holds altitude to ~5 mm), which is not modeled yet.
