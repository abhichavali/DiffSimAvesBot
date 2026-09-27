---
title: The hover policy can now be flown on modeled IMU + mocap readings; position tracking is an allowed input
date: 2026-09-26
status: current
tags: [free-flight, tooling]
---

## Finding
`hover.sensors` models what the bird could really read (IMU noise + per-flight bias sampled at 1 kHz,
a Mahony attitude filter, beat-mean accelerometer, mocap position/velocity), and `hover.reduced`
designs SOF / LQG controllers on any channel subset. Decision (user, 2026-09-26): tracking the COM
position externally is acceptable, so experiments run two tracks: position allowed (minimize the other
non-IMU inputs) and no position (down to IMU only).

## Evidence
- Defaults are unchanged: `hover.lqr` A, B, K are bit-identical with the sensor hook in the loop;
  `--noise-scale 0 --att-mode truth` flies the true-state trajectory bit for bit
  (`tests/test_hover_sensors.py`).
- Baseline reproduced on a 16-core box (fresh `hover.lqr`, committed MLP control): 120 wingbeats at dt
  0.10/0.13/0.16 ms with and without pushes; tilt max 0.4 deg at 0.13 (0.7 at 0.10); drift 11 mm at
  0.13; every `push_sweep` push recovered at all three dt, MLP and LQR.
- On the orbit the IMU's beat-mean specific force is ~5 deg off the beat-mean "up" (IMU lever arm under
  the ~4 rad/s flapping wobble): an uncalibrated accelerometer correction biased the filter by 5 deg of
  pitch. The filter now learns that constant offset (as firmware would, in the cradle); with a 20 mg
  accel bias the remaining tilt error is ~1 deg.
- Probe: LQG on `att,w` only, noise x1: 16/16 seeds fly 120 wingbeats, tilt max ~0.45 deg, but drift
  ~0.3 m (no position feedback).
- The committed `designs/hummer/free-hover/lqr.npz` is not what `hover.lqr` gives today on the committed
  control (K differs by up to 0.26): derive a fresh one before comparing controllers.

## Consequences
Use `opt/hover_mc.py` (CPU, all cores) for noise Monte Carlo; the MJX env has no sensor model.
Workflow in the `free-flight-policy` skill, "Sensors".
