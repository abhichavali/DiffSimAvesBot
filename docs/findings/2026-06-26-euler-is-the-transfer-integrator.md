---
title: Euler is the integrator where MJX and CPU MuJoCo agree per step; implicitfast diverges
date: 2026-06-26
status: current
tags: [mjx, cpu, codesign]
---

## Finding
Under `implicitfast`, MJX and CPU MuJoCo diverge per step (qvel diff ~0.17 after one step on
the hummer rig), so an MJX peak torque optimized under implicitfast is an unfaithful proxy
and gradient descent overfits it. Under `euler` the per-step qvel diff is ~3e-5, and euler is
stable for this stiff tensegrity at the design timesteps. Switching both simulators to euler
was the core fix that made MJX results transfer to CPU.

## Evidence
June 2026 autonomous reward-tuning task (branch `june-26-demo`): an aggressive design scoring
tau 118 mNm on MJX (implicitfast) gave 81 mN / 186 mNm on CPU (no longer hovers); a mild tau 188
MJX design CPU-validated at 193 mN / 249 mNm. Per-step qvel comparison: implicitfast 0.17,
euler 3e-5 (pre-2026-09 aero). Later confirmed by the July root-cause work
(`smoothed-peak-metric-mismatch`): with euler, exact in-window peaks agree to <1% at every dt.

## Consequences
- The optimizer's default `--integrator` is euler; optimize and gate on euler, and use
  implicitfast/rk4 only as robustness variants (`--tri-integrator`, `--rotate-batch`) or in
  CPU certification (the `certify-design` skill), never as the single training integrator.
- AGENTS.md "Aero" / opt/README "Robustness" state the same rule; the hover designs'
  `.tg` files use `integrator = Euler`.
- Per-step parity check after touching either aero: `uv run python opt/diag_mjx_cpu_parity.py --steps 40`.

## Status notes
Current. Numbers predate the 2026-09 aero fixes (`aero-overpredicted-lift`); parity was
re-verified on today's code (`cpu-mjx-parity-restored`: ~0.02% on lift and peak under euler).
Full `implicit` is not supported by MJX at all.
