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
commands). `hummer/hover`, `hummer/hover-soft-it009p` and
`hummer/checkpoints/it021-harmonic` were re-certified on 2026-09-24 on the current aero
(the wing's aero follows the drawn mesh and takes lift, CoP and rotational lift from the
edge that leads; the harmonic clock starts at rollout start). Every other peak and lift in
the table is pre-fix history, certified on an aero whose hinge edge always led, sat on the
leading edge and started the strips at the hinge (lift 1.7-1.9x today's, peak 8-10% lower):
re-run `design_io` before comparing against them. The seed's passive pitch is bistable
(a ~190 or a ~210 mN mode depending on dt and integrator), so compare lift per mode. The
harmonic designs' `control.tg` harmonics were rotated when the clock moved, so dt 0.06 ms
replays as it did before the clock fix. Harmonic run artifacts from before 2026-09-24
elsewhere (run dirs, other worktrees) were not: replay them with
`opt/design_io.py <dir> --pre-clock-fix-dt <the run's --timestep>` (AGENTS.md "Harmonic clock").

| design | bird | what it is | lineage | key numbers | default for |
|---|---|---|---|---|---|
| `hummer/hover/` | hummer | The codesign seed: hover geometry with 24 hand-tuned per-tendon overrides, zero harmonics. `hover_policy/` holds its trained 6-DOF stabilizing hover controller (`policy_best.npz` = the PD in `hover_meta.json`'s `pd_gains`; `policy_final.npz` is the failed BPTT result) for `src/view_hover.py`. | hover-base -> hand-tuned (git history) -> this. Was also committed as `hovering-tuned` and, misleadingly, `hovering-it009p`. | a0 65 (10.345 Hz), lag 6 deg, mean k 246.7 N/m; peak 193.1 mNm, lift 187-210 mN (2026-09-24; pre-fix 174.7 mNm / 358-364 mN) | `optimize_control`, `view_fixed_frequency`, `run_fixed_frequency`, `run_mjx_fixed_frequency`, renders, GUI first tab, `hover_policy`, `view_hover`, `harmonic1_lift_sweep` |
| `hummer/hover-soft-it009p/` | hummer | FLAGSHIP: codesign run-5 it009. Softened per-tendon overrides (L and R differ) and three small harmonics. `strokeplane_optimized.json` is applied automatically by `view_fixed_frequency` and the GUI, never by the optimizer or certification. | seed -> run-4 it018 -> this | a0 65.07, mean k 197.2 N/m; peak 170.5 mNm (-11.7% vs seed; rk4 @0.06), lift 183-186 mN (2026-09-24; pre-fix 156.5 mNm / 345-348 mN) | `tests/test_wing_lock.py` |
| `hummer/hover-base/` | hummer | Untuned hover geometry: stroke plane pitched 20 deg, uniform tendons. | ancestor of the seed | a0 63 (10.017 Hz), lag 12 deg, 200 N/m | `evolve_codesign`, `run_gui` |
| `hummer/legacy-baseline/` | hummer | The June demo baseline (pitch 50 deg, lag 180 deg, implicitfast). NOT a hover design (its lift is positive, 68-113 mN over dt 0.03-0.06; its old "negative lift" predates the 2026-09 aero fixes). | pre-hover | a0 63 (10.017 Hz), motor tendon rest 53 mm | `compile_tg` (pass `--tg`!); `tests/test_tendon_overrides.py` fixture |
| `hummer/checkpoints/it015/` | hummer | Codesign run-4 it015, sinusoid (control = the seed's). | seed -> this | pre-fix: peak 158.4 mNm (-9.3%) | - |
| `hummer/checkpoints/it018/` | hummer | Codesign run-4 it018, warm-start parent of the flagship. | seed -> this | pre-fix: peak 157.1 mNm (-10.1%) | - |
| `hummer/checkpoints/it021-harmonic/` | hummer | Codesign run-4 it021: harmonics. It looked dt-specialized (162 mNm at dt 0.04/0.09) until the rest settle stopped offsetting the harmonic clock by 2000*dt; now dt-flat (euler 164.5 @0.06, 164.3 @0.03). | seed -> this | peak 168.4 mNm (-12.8% vs seed; rk4 @0.06, implicitfast 161.5), lift 184-186 mN (2026-09-24) | - |
| `hummer/checkpoints/tuned-dt-hacked/` | hummer | Seed-derived, lightly retuned tendons (mean k 246.1 N/m; motor rest 54.2 mm, motor k 1035 N/m) plus large harmonics (`a_2` = 1.81): once taken for a dt-hacked design; with the clock reset dt 0.03 and 0.06 agree (178.4 vs 178.8 mNm, pre-aero-fix). | seed -> this | a0 65.40 (pre-fix history) | `opt/diag_mjx_cpu_parity.py` |
| `eagle/` | eagle | AvesEagle: the same shoulder carrying a three-linkage bone wing (humerus / ulna / manus). | - | 2.5 Hz (a0 15.708), lag 12 deg | `view_eagle` |
