---
name: certify-design
description: Use when you need a design's trustworthy CPU numbers - certifying a new or promoted design dir, checking an optimizer result (cpu_best/, robust_best/, checkpoints/itNNN/) before believing it, re-measuring a design after a physics change, or comparing two designs' peak torque / lift / power.
---

# Certify a design (CPU doctrine)

The certified number is the **worst exact peak motor torque** over the doctrine matrix on
MuJoCo CPU (the reference; MJX results are candidates, not certificates):
{euler, implicitfast, rk4} at dt 0.06 ms plus {euler, implicitfast} at dt 0.03 ms,
8 warmup + 24 sampled cycles (methodology section 12). Run from the repo root.

## Commands

```bash
D=designs/hummer/hover          # or an optimizer dir: opt/runs/cpu_best, .../checkpoints/it009
uv run python opt/design_io.py $D --integrator euler implicitfast rk4 --periods 24
uv run python opt/design_io.py $D --timestep 0.03 --integrator euler implicitfast --periods 24
```

Each line prints `lift`, `peak_torque`, `power`; the worst `peak_torque` over all five
lines is the certified number. ~5-10 s per integrator on a laptop. `--warmup` defaults to 8:
keep it (the rig builds lift for ~8 cycles; a shorter warmup certifies a transient).

Reference (2026-09-24, current aero): `hummer/hover` 193.1 mNm, lift 187.3-210.2 mN;
`hummer/hover-soft-it009p` 170.5 mNm, 182.7-185.5 mN; `checkpoints/it021-harmonic`
168.4 mNm, 184.3-185.9 mN. Anything older in `designs/README.md` or the methodology is
pre-fix history: re-run before comparing.

## Per-cycle lift and stationarity

The seed's passive pitch is **bistable**: it settles in a ~190 mN or a ~210 mN lift
mode depending on dt and integrator. A mean lift is one mode's number; compare modes,
not means, and look at the cycles:

```bash
uv run python .claude/skills/certify-design/cycle_lift.py $D --integrator euler implicitfast --timestep 0.03
```

It reruns the same rollout (`codesign.artifacts.validate`) and prints per-cycle lift and
per-cycle peak torque plus `drift` (last minus first cycle peak). Accept a design only if
the per-cycle peaks are stationary (drift a small fraction of the peak, no beat); a
design that looks good on 3 cycles can be parked on a ~40-cycle torque beat. Verified
on the seed at dt 0.03 euler: 24 cycles at 209.4-211.0 mN, drift -1.72 mNm.

## Rules

- **Never run two replays of the same design dir at once** (design_io, cycle_lift, the
  two doctrine commands): each writes `<dir>/_recon_scene.xml` with no lock. Run them
  one after the other, or `cp -r` the dir and certify the copy.
- An optimizer dir replays with its JSON (tendons, motor, pitch) applied and
  `theta_best.json` a/b divided by a0; a committed dir runs its own `control.tg`.
  `strokeplane_optimized.json` and `--wavelet` dirs are NOT replayed by design_io.
- Harmonics now run on the crank-locked clock that starts at rollout start; a design
  that differs between dt 0.03 and 0.06 is a real finding, not the old clock bug.
- Optional wide check: `--timestep 0.04` and `0.09` with `--integrator euler`.
- Report the per-case table (integrator x dt: lift, peak, power) and the worst peak.
