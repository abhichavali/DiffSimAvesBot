---
title: CPU aero read a one-step-stale wing velocity (mj_objectVelocity after mj_step), a 17x added-mass lift gap vs MJX
date: 2026-06-29
status: current
tags: [aero, cpu, mjx]
---

## Finding
The CPU runtime read the wing root velocity with `mj_objectVelocity`, which uses `d.cvel`.
After `mj_step`, `cvel` still reflects the pre-integration qvel, so the CPU aero ran on a
one-step-lagged velocity (~20% off during fast flapping). The quasi-steady terms tolerate
that, but the added-mass term finite-differences velocity (x lamda/dt), which blew the error
up into a ~17x MJX<->CPU lift gap (MJX 468 vs CPU 27 mN) and instability. MJX was right
(`jac @ qvel`). Fix: read the velocity from the site Jacobian (`mj_jacSite`, then
`jacr @ qvel`, `jacp @ qvel`).

## Evidence
Probe chain (2026-06-29, `flap-codesign` worktree): MJX `jac @ qvel` == CPU `mj_jacSite @ qvel`
to 1.2e-4; CPU `mj_jacSite @ qvel` != `mj_objectVelocity` by ~20%; after `mj_forward` refreshes
cvel they match exactly. After the fix the MJX/CPU lift ratio with added-mass ON (euler) went
17.3 -> 1.01 (per-step |diff| 647 -> 10 mN). Pre-2026-09 aero numbers. The worktree is
archived in ~/Code/DiffSimAvesBot-archive/2026-09-24/.

## Consequences
- `src/runtime/aero.py` `_compute_wing` reads velocity via `mj_jacSite` (its comment explains
  why); never switch back to `mj_objectVelocity` / `cvel` after a step.
- Added-mass transfers between CPU and MJX, but stays off by default
  (`--added-mass` in `opt/optimize_control.py`; `set_unsteady_aero`).

## Status notes
Current (fix in code). It supersedes the June 27 conclusion that the added-mass term was
irreconcilable between JAX and numpy ("~4x, opposite sign") and that ~200 mN lift was MJX
inflation. The same class of bug (aero on stale kinematics) recurred for positions and was
fixed on 2026-09-24 (`cpu-mjx-parity-restored`).
