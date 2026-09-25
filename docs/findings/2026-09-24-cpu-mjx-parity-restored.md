---
title: CPU and MJX now agree to ~0.02% on lift and peak torque; three step-order differences were the gap
date: 2026-09-24
status: current
tags: [cpu, mjx, aero]
---

## Finding
Three differences made the CPU reference and the MJX port disagree by up to ~3%:
- stale pose: after `mj_step` the site frames are still at step k while qpos is at k+1, and the
  CPU aero mixed them;
- hold rule: aero is recomputed every `aero_substep` (4) steps; CPU re-applied the held world
  wrench at the moved points, MJX held the generalized force;
- the two runners defaulted to different designs, the MJX runner started as a step (no ramp)
  and recomputed aero every step.

## Evidence
- e244f67 (kinematics refreshed before aero, CPU and MJX), 1d8c2be (CPU holds qfrc like MJX:
  substep-4 parity +3.69/+1.52% -> +0.02/-0.00% lift/peak, CPU 78 -> 70 us/step), 3475ef0 and
  7f2a57f (runners: same default design, ramp, hold every 4 steps; mean lift CPU 0.204828 vs
  MJX 0.204820 N).
- c0d8434: MJX aero tangents were NaN at zero strip velocity (forward values unchanged).
- Probe: `uv run python opt/diag_mjx_cpu_parity.py --steps 40` (per-step aero, pitch, force and
  state diffs).

## Consequences
- MJX is a faithful proxy for the CPU reference at euler; use it for gradients, certify on CPU.
- Keep `runtime/aero.py` and `runtime/mjx/aero.py` consistent; run the parity probe after
  touching either.
- `rollout.simulate` still recomputes aero every step (its callers' choice): AGENTS.md "Aero".

## Status notes
Current. Earlier "no MJX/CPU gap" claims (July, see the July codesign metric findings) were
about metric mismatch, not these step-order differences.
