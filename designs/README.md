# Designs

A design is a directory holding `shoulder.tg`, the whole morphology (the name is
legacy), and `control.tg`, the Fourier crank-velocity command (`a0` in rad/s,
`a_n`/`b_n` absolute). `shoulder.tg`'s `control = control.tg` resolves next to it,
a relative `meshdir = assets` resolves against the repo root, and both files carry
the same `bird =` tag. The directory name is the design name. Tool defaults are
the constants in `src/design_paths.py` (`DEFAULT_HUMMER_TG`, ...).

Peak torque is the CPU certification number: worst case over integrators/dt, exact
per-cycle peaks (see `docs/ROBUST_CODESIGN_METHODOLOGY.md` section 12). Reproduce it
with `opt/design_io.py <dir> --periods 24` (its docstring has the two doctrine
commands); the seed `hummer/hover` certifies at 174.3 mNm that way.
The harmonic designs (`hover-soft-it009p`, `it018`, `it021-harmonic`, `tuned-dt-hacked`)
were certified before the rest settle reset the clock; their `control.tg` harmonics
are rotated so dt 0.06 ms replays unchanged, and other dts now match it.

| design | bird | what it is | lineage | key numbers | default for |
|---|---|---|---|---|---|
| `hummer/hover/` | hummer | The codesign seed: hover geometry with 24 hand-tuned per-tendon overrides, zero harmonics. `hover_policy/` holds its trained 6-DOF stabilizing hover controller (`policy_best.npz` = the PD in `hover_meta.json`'s `pd_gains`; `policy_final.npz` is the failed BPTT result) for `src/view_hover.py`. | hover-base -> hand-tuned (git history) -> this. Was also committed as `hovering-tuned` and, misleadingly, `hovering-it009p`. | a0 65 (10.345 Hz), lag 6 deg, mean k 246.7 N/m; peak 174.7 mNm | `view_fixed_frequency`, `run_mjx_fixed_frequency`, renders, GUI first tab, `hover_policy`, `view_hover`, `harmonic1_lift_sweep` |
| `hummer/hover-soft-it009p/` | hummer | FLAGSHIP: codesign run-5 it009. Softened per-tendon overrides (L and R differ) and three small harmonics. `strokeplane_optimized.json` is applied automatically by `view_fixed_frequency` and the GUI, never by the optimizer or certification. | seed -> run-4 it018 -> this | a0 65.07, mean k 197.2 N/m; peak 156.5 mNm (-10.4% vs seed), lift 345-348 mN | `tests/test_wing_lock.py` |
| `hummer/hover-base/` | hummer | Untuned hover geometry: stroke plane pitched 20 deg, uniform tendons. | ancestor of the seed | a0 63 (10.017 Hz), lag 12 deg, 200 N/m | `evolve_codesign`, `run_gui` |
| `hummer/legacy-baseline/` | hummer | The June demo baseline (pitch 50 deg, lag 180 deg, implicitfast). NOT a hover design: negative lift. | pre-hover | a0 63 (10.017 Hz), motor tendon rest 53 mm | `optimize_control`, `compile_tg`, `run_fixed_frequency` (pass `--tg`!); `tests/test_tendon_overrides.py` fixture |
| `hummer/checkpoints/it015/` | hummer | Codesign run-4 it015, sinusoid (control = the seed's). | seed -> this | peak 158.4 mNm (-9.3%) | - |
| `hummer/checkpoints/it018/` | hummer | Codesign run-4 it018, warm-start parent of the flagship. | seed -> this | peak 157.1 mNm (-10.1%) | - |
| `hummer/checkpoints/it021-harmonic/` | hummer | Codesign run-4 it021: harmonics. It looked dt-specialized (162 mNm at dt 0.04/0.09) until the rest settle stopped offsetting the harmonic clock by 2000*dt. | seed -> this | euler peak 151.5-151.7 mNm at dt 0.03-0.09 (not re-certified) | - |
| `hummer/checkpoints/tuned-dt-hacked/` | hummer | Seed-derived, lightly retuned tendons (mean k 246.1 N/m; motor rest 54.2 mm, motor k 1035 N/m) plus large harmonics (`a_2` = 1.81): a known dt-hacked design. | seed -> this | a0 65.40 | `opt/diag_mjx_cpu_parity.py` |
| `eagle/` | eagle | AvesEagle: the same shoulder carrying a three-linkage bone wing (humerus / ulna / manus). | - | 2.5 Hz (a0 15.708), lag 12 deg | `view_eagle` |
