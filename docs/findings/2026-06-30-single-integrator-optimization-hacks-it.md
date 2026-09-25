---
title: Optimizing peak torque on one integrator hacks that integrator; train and gate across integrators
date: 2026-06-30
status: current
tags: [codesign, certification, mjx, cpu]
---

## Finding
Gradient descent on MJX euler peak torque finds euler-only optima: euler torque drops while
implicitfast/rk4 rebound. An in-loop CPU check on the same integrator cannot see it. Two
defenses work: (1) a training loss over several integrators (robust max plus a spread
penalty), so lowering one integrator while another rebounds raises the loss; (2) a CPU
selection gate that rejects designs whose integrators disagree. A harmonic "dose" penalty
(ws = 10) did not stop the hacking; the gate did.

## Evidence
2026-06-30: after a hover optimization, MJX reported peak torque -30% while the real CPU peak
went up. The learned design read euler 167 mNm but implicitfast 238 mNm; the seed read
~188 on both (pre-2026-09 aero). EA seeds showed euler-vs-rk4 gaps of 16-25% (mjx_best).
MJX differentiates euler, implicitfast and rk4 (only full `implicit` fails). The
`--tri-integrator` smoke baseline read e186 / i184 / r186, gap 1%.

## Consequences
- `opt/optimize_control.py`: `--tri-integrator` (loss = robust max over euler/implicitfast/rk4
  + `--w-gap` x spread^2, ~3x rollout cost), `--cpu-every N` + `--cpu-gap 0.15` -> `cpu_best/`
  (CPU euler + implicitfast must agree), and the rotating integrator x dt batches that
  generalize this (`rotating-batch-select-on-full-matrix`).
- Certification runs all three integrators on CPU (the `certify-design` skill); the GUI runs
  the `.tg`'s own `integrator` (hover designs use Euler).

## Status notes
Current as a method. Numbers predate the 2026-09 aero fixes. Part of what looked like
integrator/dt hacking was later traced to evaluation artifacts (`smoothed-peak-metric-mismatch`,
`torque-beat-window-parking`) and, for harmonics, to `harmonic-clock-offset`; the
cross-integrator gate stays in the doctrine regardless.
