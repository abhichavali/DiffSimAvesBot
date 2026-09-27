---
title: A static output-feedback hover gain on position + IMU without velocity cannot recover the 8 rad/s yaw push; an observer (LQG) or vel_z is needed
date: 2026-09-26
status: current
tags: [free-flight]
---

## Finding
Noise-free, track P (position tracked). SOF (`hover.reduced --mode sof`, a static gain on the measured
channels) on `pos,att,w` or `pos,att,gyro` hovers 120 wingbeats at every dt (no-push tilt max 0.93 deg,
drift 36/25/10 mm) but recovers only 17/19 sweep pushes at every dt: roll 4 rad/s and yaw 8 rad/s are lost.
Four Q/R reweightings (r = 0.1, q_w = 0.25, q_pos = 0.05, q_att = 0.03 with q_w = 0.25) reach at best
18/19 and never recover yaw 8; r = 0.1 falls to 9-10/19. `pos_z,att,w` SOF loses yaw 8 at one or two dt.

What works instead: the LQG on the same channels (19/19; the Kalman filter recovers velocity from position
plus the model), or adding one velocity axis: SOF `pos,att,w,vel_z` with q_pos 0.05 (19/19, bounded over
240 wingbeats: 19-52 mm horizontal, altitude within 7 mm).

## Evidence
`opt/runs_imu/minimize/table_truth.md` (SOF rows), `ctl/`, `ctl_fix/` (`sof__pos+att+w+vel_z_qp5.npz`);
standard-CLI re-check `opt/runs_imu/verify_minimize/cli_*.json`
(`opt/push_sweep.py --ctrl <npz> --noise-scale 0 --att-mode truth`).

## Consequences
With position tracked, fly an LQG (or its MLP with the filter in the loop). A static policy needs `vel_z`.
