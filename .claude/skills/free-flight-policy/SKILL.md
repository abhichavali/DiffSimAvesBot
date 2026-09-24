---
name: free-flight-policy
description: Use when working on free flight of the standalone bird (designs/hummer/free-hover, src/hover) - re-deriving the hover policy after changing the bird (mass, inertia, tendons, aero, any physics), checking whether a flight control still hovers, reading or writing a flight control .tg, tuning the LQR / MLP closed loop or the open-loop harmonics, or debugging a bird that tumbles.
---

# Free-flight hover policy

The free-standing bird (`designs/hummer/free-hover/`, 226 mN, `flying.to_airframe`) is
flown only by its four crank motors: **no force of any kind acts on it after release**
(AGENTS.md "No fake forces"). Its policy lives entirely in a flight control `.tg`.

## How it works (read once)

- **Drive** (`hover.launch.Drive`): theta = 25 numbers = `[g, a (4 cranks x 3), b (4 x 3)]`;
  crank i runs `w_i(phi) = a0 e^g (1 + sum_k a_ik cos k phi + b_ik sin k phi)`. Per-crank
  waveforms are the only steering: L/R and t1/t4 differences make roll, pitch, yaw.
- **Phase lock**: `ctrl = w + phase_lock * (L_des - L)`, `L_des` the integral of `w`
  (an encoder lock). Without it the crank lag drifts ~0.05-0.07 deg per wingbeat and slides
  the bird out of trim.
- **dt snapping** (`load_bird`): dt is adjusted so one wingbeat is an integer number of
  steps (901 at ~0.13 ms), so the per-wingbeat strobe always samples the same phase.
- **Strobed state** (12, once per wingbeat, relative to the release orbit): COM position,
  attitude rotation vector, COM velocity, world angular velocity. Simulator ground truth:
  no sensor model.
- **Launch cradle** (`release`): a PID on the one-wingbeat *average* pose holds the bird
  while the flapping settles, then is removed. Its mean wrench at release is the trim
  error (what the open loop fails to supply).
- **Open loop** = the trim: harmonics + hover tilt that zero the cradle's wrench. It is
  open-loop *unstable* (per-wingbeat |eig| ~1.08): it tips over after ~40-50 wingbeats
  whatever the waveform. Judge an open loop by how long it lasts, never expect it to hover.
- **Closed loop** = per-wingbeat correction of the 24 harmonic depths (not a0),
  `u = 0.15 tanh(MLP(s))`, the MLP distilled from an LQR on the one-wingbeat linear model.

## Flight control `.tg` keys (`hover.policy`)

On top of a plain control file: `a_<n>.<crank>` / `b_<n>.<crank>` (absolute rad/s, cranks
`L_t1 L_t4 R_t1 R_t4`), `hover_attitude = w x y z` (release attitude, the ~21 deg hover
tilt), `phase_lock = 20` (1/s), `closed_loop = <file>.npz` (relative to the `.tg`; omit =
open loop). `a0` must equal `2 pi fundamental_frequency` (checked on load). Read/write
with `load_flight_policy` / `write_flight_control`, never by hand-editing numbers.

## Re-derive the policy (after ANY change to the bird or its physics)

A policy is only valid for the exact bird and trim it was derived on: the MLP acts on the
deviation from *that* orbit. Run all four steps, in a scratch dir `D`, from the repo root:

```bash
T=designs/hummer/free-hover/shoulder.tg; C=designs/hummer/free-hover/control.tg
PYTHONPATH=src uv run python -m hover.trim --tg $T --control $C --out $D/control.tg     # ~45 s/iteration, 8 cores
PYTHONPATH=src uv run python -m hover.lqr  --tg $T --control $D/control.tg --out $D/lqr.npz   # ~6 s
uv run python opt/distill_hover.py --lqr $D/lqr.npz --out $D/mlp_closed_loop.npz          # ~20 s
echo "closed_loop = mlp_closed_loop.npz" >> $D/control.tg
uv run python opt/verify_hover.py --tg $T --control $D/control.tg                         # ~25 s for 120 wingbeats
```

- **Trim** starts from an existing flight control (it needs `hover_attitude`); Gauss-Newton
  on theta + 2 tilt angles until the cradle carries < 0.5 mN / 0.02 mNm (`--iters`,
  default 6). It writes the open loop only: the old `closed_loop` is dropped on purpose.
  "base rollout unstable" = the starting control can't even settle in the cradle.
- **LQR** prints open-loop |eig| (max > 1 is normal) and closed-loop |eig| (all must be
  < 1; ~0.65 max today). Weights: `hover.lqr.lqr` (tolerances 2 cm, 0.05 rad, 0.1 m/s,
  0.5 rad/s; input 0.05).
- **Distill** prints the MLP's relative MSE against the clipped LQR.
- **Verify** flies closed loop with and without pushes (wingbeats 30/60/90). Pass = "flew
  all N wingbeats" in both, tilt within a degree or two. Also run `--dt-ms 0.10` and
  `--dt-ms 0.16` (dt robustness), `--open-loop` (how long the trim alone lasts),
  `--lqr $D/lqr.npz` (the LQR itself, to separate LQR from distillation problems), and
  `opt/push_sweep.py --control ...` (push margins). Reference today (committed
  `control.tg`): 120 wingbeats, tilt <= 0.4 deg, recovers 0.8 m/s and 4 rad/s pushes.
- Promote: copy `control.tg`, `mlp_closed_loop.npz`, `lqr.npz` into the design dir,
  update the `designs/README.md` row and the header comment of `control.tg`.

## Watch it

`mjpython src/view_free_hover.py [--control ...] [--speed 0.25]` (macOS needs the
`DYLD_LIBRARY_PATH` from AGENTS.md Gotchas), or open the design in the GUI and use the
Viewer tab. Keys: P = random push, O = toggle the closed loop.

## What does not work (don't repeat)

- **BPTT through MJX** (`opt/train_hover.py`, `hover.env`): with `iterations=1` over 4-8
  wingbeats the gradients reached 1e5-7e8 and diverged, from scratch and from the LQR.
  One-wingbeat `jacfwd` with the full solver does match CPU finite differences (B cosine
  0.978), so MJX can build the linear model; long-horizon learning cannot.
- **Releasing from rest** tumbles the bird: always launch through the cradle.
- **Plain velocity servos** (no phase lock): "stable" candidates still fall.
- **Weightless body inertia**: without `flying.BODY_INERTIA` the airframe recoils against
  every wingbeat and the free bird loses its lift (-15 mN free vs 239 mN held).
- Open-loop search (CMA-ES in the trim's null space, `control_open_loop.tg`) bought
  ~49 vs 38-47 wingbeats: a small real gain, not stability. Evaluate any open-loop claim
  at the candidate's own snapped dt and across dt 0.10-0.16 (an early "70 wingbeats" was a
  dt artifact).
