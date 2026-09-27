---
title: The hover policy needs no velocity input when position is tracked, and no non-IMU input at all if drift is accepted (noise-free)
date: 2026-09-26
status: current
tags: [free-flight]
---

## Finding
Noise-free sensors (noise_scale 0), fresh `hover.lqr` model. "Good" = 120 wingbeats with and without the
verify pushes at dt 0.10/0.13/0.16 ms, tilt < 2 deg, all 19 `push_sweep` pushes recovered at every dt.
- **Position tracked (track P)**: velocity can be dropped. LQG on `pos,att,w` (or `pos,att,gyro`) is as
  good as the full state: no-push tilt max 0.71/0.37/0.34 deg, drift 20/12/10 mm, 19/19 pushes at every dt,
  flat over 240 and 480 wingbeats; its distilled MLP (input = the Kalman estimate) is too. A static gain
  without velocity is not (finding `2026-09-26-hover-sof-needs-velocity-or-observer`). `pos_z` alone holds
  altitude to ~3-5 mm but not horizontal position (0.14-0.42 m in 120 wingbeats).
- **No position (track N)**: IMU-only `att,w` (filter attitude + rate) is good with SOF, LQG and the
  distilled SOF MLP (six inputs, no observer). Attitude stays 0.3-0.7 deg, but the bird walks at a steady
  rate that depends on dt (SOF: 26/19/14 mm/s horizontally at dt 0.10/0.13/0.16 with true attitude, 25/15/12
  with the filter) and sinks 2-5 mm/s (SOF) or ~7 mm/s (LQG). It reaches the floor 0.22 m below the release
  after ~30 s (LQG) and ~40-70 s (SOF). Details and what does not fix it: `2026-09-26-hover-imu-only-walks`.
- Gyro alone (no attitude channel) loses the 8 rad/s yaw push with the default LQG and SOF (18/19); other
  weightings lose more (SOF r = 0.1: 15/19). Raw `gyro,acc` (no attitude filter) passes as an LQG designed
  with `--noise-scale 1`.

## Evidence
`opt/runs_imu/minimize/` (gitignored): `ctl/`, `ctl_fix/` (controllers + `build.json`), `fly_*.json`
(5733 flights), `table_truth.md`, `filter/` (the same suite with the Mahony filter), `long.json`
(480 wingbeats), `mlp/`; scripts `build.py`, `build_fix.py`, `batch_fly.py`, `longrun.py`, `summarize.py`.
Independent re-check with the standard CLIs (`verify_hover.py`, `push_sweep.py`), 240-wingbeat flights and a
channel-corruption test (unselected channels replaced by N(0, 100) garbage: bit-identical trajectories, so
no ground truth leaks in): `opt/runs_imu/verify_minimize/`. Rebuild: `PYTHONPATH=src python -m hover.reduced
--lqr opt/runs_imu/baseline/lqr.npz --obs pos,att,w --mode lqg --noise-scale 0 --out ctl.npz` (rho 0.651).

## Consequences
- The recommended inputs: track P = mocap `pos` + IMU `att,w` through an LQG; track N = `att,w` SOF.
- Under sensor noise the ranking changes and both need re-deriving: `2026-09-26-hover-imu-noise`.
- An LQG that reads `acc` and is designed at noise 0 fails after pushes; the cause is a sensor-model
  inconsistency, not the linear model: `2026-09-26-hover-acc-reads-push-impulse`.

## Status notes
Verified 2026-09-27 on fresh flights. Corrected then: the walk rate is dt-dependent (was "18 mm/s"), the SOF
reaches the floor after ~41 s at dt 0.16 with the filter (was "~70 s"), and "gyro alone fails only yaw 8
under every variant" holds only for the default weightings. The linear model is identified at dt 0.13 only.
