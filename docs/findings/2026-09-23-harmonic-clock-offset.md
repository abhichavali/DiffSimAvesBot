---
title: 'The "harmonics overfit dt" result was a clock bug: the rest settle offset every harmonic by omega*2000*dt'
date: 2026-09-23
status: current
tags: [harmonics, codesign, certification, cpu, mjx]
---

## Finding
Every harmonic is evaluated at phase `omega * data.time`, but the 2000-step wing-pitch rest
settle never reset `data.time`. Each harmonic therefore started rotated by `k*omega*2000*dt`:
112 deg at 0.015 ms, 223 at 0.03, 447 at 0.06, 969 at 0.13. The optimizer shared one settled
state across its dt bank (one consistent offset), while CPU certification re-settled at each
dt, so a learned harmonic was replayed rotated at every other dt. That is what looked like
"harmonics are dt-specialized". With the clock reset, harmonic designs are dt-flat.

## Evidence
- Fix 649cab1 (`data.time = 0.0` after the settle, `src/runtime/wing_pitch.py`); 417b744 makes
  CPU replay/export use the MJX crank-locked clock `omega*exp(f_log + a0_log)`.
- it021-harmonic (old aero): 157.6 mNm @0.03 / 162 @0.04 and 0.09 before, 151.5-151.7 mNm flat
  over dt 0.03-0.09 after.
- A first-harmonic lift sweep relabeled to the reset clock collapsed exactly across all five
  euler cells dt 0.015-0.13 (pre-aero-fix numbers; the sweep script was deleted, c0b0a39).

## Consequences
- The four committed harmonic designs were rotated by `k*omega*0.12 s` so dt 0.06 replays as
  before. Other pre-2026-09-24 harmonic artifacts: replay with
  `opt/design_io.py <dir> --pre-clock-fix-dt <run --timestep>`; rotate old `theta_full.json` /
  `codesign_seed.json` before `--init-codesign` (AGENTS.md "Harmonic clock"). New JSON carries
  `"time_origin": "settle_end_reset"`; loading one without it warns (31f7943).
- A design that now differs between dt 0.03 and 0.06 is a real finding, not this bug.

## Status notes
Current. It supersedes the harmonics verdicts in `harmonics-verdict-july` and in the Sept 10
reward screen; those conclusions need re-running on the fixed clock (and the fixed aero,
`aero-overpredicted-lift`).
