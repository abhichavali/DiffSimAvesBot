---
title: The free-standing bird weighs 226 mN and only makes lift in free flight with added body inertia; it hovers pitched ~21 deg
date: 2026-09-24
status: current
tags: [free-flight, design]
---

## Finding
- Weight: the bench rig's pillars (4 x 10 g placeholder "reactivity knob") and crank disks
  (4 x 5 g hard-coded) made the airframe 73 g = 716 mN. The user ruled pillars are not part of
  the bird. In `flying.to_airframe` only, pillars are massless and each crank disk (motor + disk)
  is 2.5 g: 23.1 g = 226 mN (struts 12.8 g, wings 0.27 g unchanged). Bench rig and certified
  numbers are untouched.
- Inertia: with the body's own rotational inertia (~4e-5 kg m^2, about that of the flapping
  struts) each wingbeat rocked the body 13-48 deg peak to peak and the stroke plane with it;
  the free bird made ~0 lift (-14.6 mN free vs 238.8 mN held in the cradle, a0 65).
  Adding massless airframe inertia at the same 226 mN restored it: +4e-5 -> 323 mN, +1.6e-4 ->
  271 mN (4-10 deg wobble), +6.4e-4 -> 252 mN. Moving strut mass to the body made it worse.
- Tilt: the mean aero force is ~38 deg off body vertical, so hover needs a pitched body; the
  trims settle at ~21-26 deg (`hover_attitude` in the flight control).

## Evidence
`src/flying.py`: `DISK_MASS = 0.0025`, `BODY_INERTIA = 1.6e-4` (user-approved, 3a7af3e),
`FLIGHT_WEIGHT_RANGE_N = (0.200, 0.250)` asserted by its `__main__` self-check (cd535f0).

## Consequences
- Keep the free bird in 200-250 mN; change mass only with the user.
- Any change to weight or inertia invalidates the hover policy: re-derive it
  (`free-flight-policy` skill).
- A real build needs the same inertia (battery / electronics spread about the COM).

## Status notes
Current.
