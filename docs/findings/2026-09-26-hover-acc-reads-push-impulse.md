---
title: LQGs that read the accelerometer fail after pushes because the simulated IMU sees each instantaneous kick as an impulse the model never contains, not because the linear acc model is wrong
date: 2026-09-26
status: current
tags: [free-flight, tooling]
---

## Finding
Designed at noise 0 (V at the floor), every LQG that reads `acc` (`att,w,acc`, `imu`, `pos,att,w,acc`,
`pos_z,att,w,acc`, `gyro,acc`) loses 9-19 of the 19 sweep pushes and tumbles on the verify pushes
(`att,w,acc` 97/95/98 wingbeats, `pos,att,w,acc` 93/67/95 at dt 0.10/0.13/0.16). The first write-up blamed
"the linear model of the beat-mean accelerometer is wrong for push-sized states". Verification refuted that:
- `hover.launch.fly` applies `kick()` between wingbeats without telling the sensor, so `Sensors.v_prev`
  (`src/hover/sensors.py`, finite-difference accelerometer) keeps the pre-kick velocity. The next IMU sample
  reads dv/dt_imu and the beat-mean `acc` carries dv/T, ~7 m/s^2 for a 0.8 m/s push.
- `hover.lqr.one_beat_obs` starts the sensor after its perturbation, so the identified `C_prev` / `J` never
  contain that impulse; a noise-0 Kalman filter trusts `acc` fully and is thrown off.
- Resetting `v_prev` right after the kick (nothing else changed) makes the verify-push flights pass at all
  three dt for `att,w,acc`, `imu` and `pos,att,w,acc`; side 0.8 m/s goes from 0/3 to 3/3 dt. Yaw 8 rad/s
  still fails, and `att,w,acc` also fails with no push at dt 0.13 (88 wingbeats): that part is not the kick.

## Evidence
`opt/runs_imu/verify_minimize/vflights.py` (kind `kick` monkeypatches `L.kick`), `flights_a.json`;
original failures in `opt/runs_imu/minimize/fly_all.json`, `fly_fix.json`.

## Consequences
- Designing with the real noise (`--noise-scale 1`) hides it (V down-weights `acc`): every `_V1` acc LQG
  passes. Process noise x10 does not (0-8/19).
- Left unfixed (AGENTS.md Known bugs). Before judging acc-reading controllers under pushes, decide whether a
  push is a real impulse the IMU should feel (then model it as a disturbance in W) or a state reset (then
  reset `v_prev` in `kick`).
