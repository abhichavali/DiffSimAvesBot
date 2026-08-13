# Flight orientations

Three AVES-Wing-Motor configurations that differ **only in motor disk angles**.
Each folder holds a `shoulder.tg` and an identical `control.tg` (copied from
`design/baseline_control.tg`) — same drive signal, different stroke geometry.

Two disk knobs do the work (see `src/morphology/model.py` `DiskSpec`):

- **`pitch` (deg)** — orbit pitch about world Y. Tilts the stroke plane fore/aft.
  `0` = flat disk = horizontal stroke; larger = stronger forward inclination.
- **`crank_phase_lag` + `crank_throw_t4`** — stagger the two counter-rotating
  cranks (timing + throw mismatch) to break down/up stroke symmetry. Zero lag and
  equal throws = symmetric.

| Config | pitch | crank_phase_lag | throw t1 / t4 | result |
|---|---|---|---|---|
| `horizontal-flapping` | 0 deg | 0 deg | 23 / 23 mm | horizontal stroke, no up/down asymmetry |
| `hovering` | 20 deg | 12 deg | 23 / 26 mm | mild forward inclination, mild asymmetry |
| `forward-flight` | 50 deg | 28 deg | 23 / 30 mm | strong forward inclination, strong asymmetry |

`tilt` stays `0` in all three (it rolls the stroke plane laterally, which none of
these need).
