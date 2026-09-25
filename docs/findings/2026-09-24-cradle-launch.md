---
title: Launch the free bird from a mean-only cradle; releasing from rest tumbles it
date: 2026-09-24
status: current
tags: [free-flight]
---

## Finding
Releasing the free bird from rest while the wings spin up tumbles it: the start-up transient
alone is enough. A launch cradle that holds only the bird's wingbeat-averaged pose lets the
flapping settle onto its natural periodic orbit (within-wingbeat wobble stays free), then is
removed. Its mean wrench over the last wingbeats is exactly the open loop's trim error, which
is what `hover.trim` drives to zero.

## Evidence
- `src/hover/launch.py` `release`: PID (bandwidth 1.5 Hz, damping 0.8, integral action) on the
  one-wingbeat boxcar mean of the strobed pose error, applied through `xfrc_applied`, zeroed at
  release; `Released.cradle_wrench` = mean [force, moment] over its last `n_avg` wingbeats.
- A soft (spring) cradle let the bird sag ~17 cm; a hard hold (weld) gave a trim that did not
  carry over to free flight (held trim != free trim: body wobble changes the aero).

## Consequences
- Every free-flight tool launches through `release` (trim, lqr, verify, push_sweep, the
  viewer). The cradle is the only non-physical force and acts only before release (AGENTS.md
  "No fake forces").
- Trim quality is measured as the cradle's residual wrench (< 0.5 mN / 0.02 mNm).

## Status notes
Current.
