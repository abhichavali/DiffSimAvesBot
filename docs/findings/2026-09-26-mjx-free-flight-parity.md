---
title: MJX free flight is the CPU hover loop step for step (x64 ~1e-11); float32 on the GPU keeps every verdict
date: 2026-09-26
status: current
tags: [mjx, free-flight, cpu]
---

## Finding
`hover.mjx_flight` (released flight), `mjx_sensors` (sensor model + observers) and `mjx_cradle` (cradle +
trim) reproduce `hover.launch` / `hover.sensors` / `hover.reduced` / `hover.trim` exactly: in float64 the
strobed states match CPU to 1e-11 (4e-8 after 120 wingbeats), cradle wrenches to 1e-11 mN, the linear model
A, B to 1e-11 and the sensor model C_now/C_prev/J to 1e-8. Float32 on an RTX 5090 is the only remaining
difference: over 120 wingbeats the per-beat state error (max over components of |GPU-CPU| / {1 cm, 0.01 rad,
0.02 m/s, 0.1 rad/s}) is 0.16-0.44 under the LQR with the verify pushes at dt 0.10/0.13/0.16 ms, and every
verify and push-sweep verdict is the same (19/19 pushes, tilt within 0.1 deg, drift within 1 mm).
What the old `hover.env` loop got wrong (and the port fixes):
- `iterations=1` (and set on the shared CPU model): walks the trim, error 1.3-7.7, drift 30-90 mm vs 9-44 mm;
- the drive phase and the encoder target L_des summed in float32 over the whole flight (~1e-4 rad per wingbeat):
  the port restarts the phase every wingbeat, unwinds the crank angles and carries the encoder error L_des - L;
- a fresh aero hold at release (the CPU continues its counter, held force and history);
- no `mj_forward` at the strobe: the CPU's refreshes the one-step-stale actuator lengths the phase lock reads;
- pushes only at release. Its `gravity = 0` is not a bug: it only brackets `rest_state`.

## Evidence
- `python opt/hover_gpu.py parity --dt-ms 0.10 0.13 0.16 --lqr lqr.npz --cpu-ref ref.npz` (+ `--x64`, solver
  knobs); `tests/test_hover_mjx.py` (x64 on CPU-JAX, 2 wingbeats, LQR + a push, and the LQG sensor pipeline).
- Float32 error, LQR / LQR+push, dt .10/.13/.16: full solver, line search 50: 0.23/0.22, 0.20/0.29, 0.31/0.41;
  line search 8: 0.16/0.23, 0.20/0.32, 0.30/0.44; iterations 2: 0.26/0.30, 0.30/0.33, 0.32/0.82;
  iterations 1: 1.25/4.9, 2.9/5.3, 6.1/7.7. Open loop: lost at wingbeat 36-48 (CPU 38-47), growth per wingbeat
  1.119-1.133 (CPU 1.130-1.135): chaotic after ~10 wingbeats, so compare it only statistically.
- Linearization, float32 finite differences on device: A 4.8%, B 5.1% relative, median B column cosine 0.9998
  (min 0.99), closed-loop |eig| 0.652 vs 0.651 (device gain on the CPU model 0.656). The acc rows of the
  sensor model (C_prev, J) are float32 noise (errors 0.4-1.2): build the sensor model with `--x64` (44 s).
  One-wingbeat `jacfwd` (float32): median B cosine 0.98 but min -0.22, gain on the CPU model |eig| 0.79: use FD.
- Throughput (RTX 5090, float32, full solver, line search 8): 295 / 1782 / 2749 wingbeats/s at batch 512 / 2048
  / 8192 = 23 flights of 120 wingbeats per second; model defaults (line search 50) 365 at 2048; iterations 1
  5406 at 8192 (not accurate enough). RTX 5070 (12 GB): 267 / 472 / 826 / 1045 wingbeats/s at batch 256 / 512 /
  1024 / 2048; 1000 noisy LQG flights of 120 wingbeats in 356 s. ~50 s XLA compile per shape. CPU MuJoCo:
  35 s per flight per core (1000 flights ~ 36 min on 16 free cores).

- Sensor-noise Monte Carlo (LQG on pos,att,w, 12 seeds, verify pushes, dt 0.13), GPU vs CPU `opt/hover_mc.py`:
  pass 100% in both; median max tilt / final drift at noise x1 0.51 deg / 13.8 mm vs 0.52 / 14.3 (no push),
  4.37 / 12.5 vs 4.31 / 13.5 (pushes); x3 0.84 / 13.0 vs 0.84 / 15.3, 4.74 / 15.1 vs 4.64 / 17.7
  (different noise draws: compare statistics, not traces).
- Whole loop on the GPU (`opt/hover_gpu.py derive --trim`, RTX 5070): trim 3 Gauss-Newton iterations (cradle
  0.04 mN / 0.015 mNm) 133 s, linearize 51 s, distill 61 s, verify at 3 dts + push sweep 505 s (mostly XLA
  compiles): PASS, 6/6 flights, 19/19 pushes. The CPU `opt/verify_hover.py` on that policy agrees (dt 0.13:
  tilt max 4.4 vs 4.4 deg, pos max 33 vs 34 mm).
- XLA GPU bug, worked around: quaternion products written as `jnp.stack` of scalar expressions became a Triton
  block-level fusion (a 4-way concatenate fed by a small matvec) that silently produced NaN attitudes in ~all
  vmapped sensor flights after 3-5 wingbeats on an RTX 5090 and failed to compile on an RTX 5070 (Triton
  `scf.if` type mismatch); no XLA flag helped. `qmul` / `quat2mat` as einsums with constant tables fixed both.

## Consequences
- GPU is the primary path for batched hover work (`opt/hover_gpu.py`); the CPU tools are the reference: run
  `opt/verify_hover.py` once on a policy you promote.
- Keep the model's solver iterations; `--ls-iterations 8` and contacts off (the free bird has none) are free.
- Any change to `hover.launch`, `hover.sensors` or `hover.reduced` must be mirrored in the device port and
  `tests/test_hover_mjx.py` must stay at ~1e-9.
- If a new vmapped device program gives NaNs that the same code on CPU-JAX does not, suspect XLA GPU fusion
  (dump HLO with `XLA_FLAGS=--xla_dump_to=...`; look for `"kind":"__triton"` fusions around the NaN).
- float64 on the 5090 is ~40x slower and crashed at batch 512 (CUDA illegal address): use it only for small
  exact checks.

## Status notes
Current. Supersedes the "no free-flight parity check" caveat of `bptt-diverges-past-one-wingbeat` for flight
(BPTT itself is still a negative result).
