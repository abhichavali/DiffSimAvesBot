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
  attitude rotation vector, COM velocity, world angular velocity. Simulator ground truth,
  unless a sensor model is attached (see "Sensors").
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

## On the GPU (primary): `opt/hover_gpu.py`

Everything below also runs as batched MJX flights on a GPU box (`gpu-box` skill: `.venv/bin/python`, the
CUDA `LD_LIBRARY_PATH`, jobs detached). The device code (`hover.mjx_flight`, `mjx_sensors`, `mjx_cradle`) is
the CPU loop step for step (x64: equal to ~1e-11); float32 on the GPU keeps the verdicts and is within a few %
of a push (`mjx-free-flight-parity` finding). The CPU tools stay the reference: confirm a policy you promote
with `opt/verify_hover.py` once.

```bash
P=".venv/bin/python opt/hover_gpu.py"
$P derive --out $D --dt-ms 0.10 0.13 0.16 --trim --cpu-check  # trim -> linearize+LQR -> distill -> verify + sweep
$P mc --lqr $D/lqr.npz --obs pos,att,w --noise-scales 1 3 --seeds 500   # sensor-noise Monte Carlo, 1000s of flights
$P parity --lqr $D/lqr.npz --cpu-ref ref.npz                  # after touching hover.launch / runtime / mjx code
```

- Device knobs: keep the model's solver iterations (`--iterations 1` is 3x faster but walks the trim: pos
  drift 30-80 mm vs 9-21 mm on CPU); `--ls-iterations 8` (default) matches the model's 50; contacts are off
  (the free bird never touches itself; `--contact` restores them).
- Any change to `hover.launch` (drive, strobe, cradle, kicks), `hover.sensors` or `hover.reduced` must be
  mirrored in the device port: `uv run pytest -q tests/test_hover_mjx.py` (x64, CPU) must stay at ~1e-9.
- The release (cradle settle) runs on CPU, ~10 s per (design, dt); the trim runs the cradle on the device.

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

## Sensors and reduced observations (`hover.sensors`, `hover.reduced`)

What a real bird could feed the policy instead of the simulator's true state. Channels (18, each a
deviation from the release orbit, read at the strobe): `pos` (mocap COM), `att` (on-board attitude
filter), `vel` (mocap/flow), `w` (world rate = R_est gyro), `gyro` (body rate), `acc` (body specific
force, mean over the last wingbeat). Subsets: groups, `state` / `imu` / `all`, per-axis `pos_z`,
`att_xy`. Non-IMU = `pos`, `vel`. Noise defaults are datasheet numbers (BMI088 gyro/accel, lab mocap;
sources in the `SensorNoise` docstring), all scaled by `--noise-scale`; `0` = noise-free.

```bash
PYTHONPATH=src uv run python -m hover.lqr --out $D/lqr.npz          # also saves the sensor model (C_now, C_prev, J)
PYTHONPATH=src uv run python -m hover.reduced --lqr $D/lqr.npz --report "state; pos,att,w; att,w; imu"
PYTHONPATH=src uv run python -m hover.reduced --lqr $D/lqr.npz --obs pos,att,w --mode lqg --noise-scale 1 --out $D/ctl.npz
uv run python opt/verify_hover.py --ctrl $D/ctl.npz --noise-scale 1 --pos-latency-ms 5 --seed 0 --dt-ms 0.10 0.13 0.16 --json v.json
uv run python opt/verify_hover.py --lqr $D/lqr.npz --obs pos,att,w --mode sof --noise-scale 1   # designed on the fly
uv run python opt/push_sweep.py --ctrl $D/ctl.npz --noise-scale 1 --json p.json
uv run python opt/distill_hover.py --ctrl $D/ctl.npz --noise-aug 1 --out $D/ctl_mlp.npz        # fly with --ctrl
uv run python opt/hover_mc.py --lqr $D/lqr.npz --obs att,w --noise-scales 0 1 2 --seeds 8 --json mc.json
uv run python opt/hover_mc.py --ctrl $D/ctl.npz --noise-scales 1 --pos-latency-ms 5 --seeds 16 --dt-ms 0.10 0.13 0.16 --scenarios none verify --json mc.json
```

**Deriving a reduced-input, noise-aware policy** (what Prompt 1 settled on; findings
`2026-09-26-hover-minimal-inputs`, `2026-09-26-hover-imu-noise`):
1. Fresh `hover.lqr` (the committed `lqr.npz` has no sensor model).
2. Pick the subset: position tracked -> `pos,att,w` LQG; no position -> `att,w` SOF (6 inputs) or the raw
   `gyro,acc` LQG (no attitude filter). Design with `--noise-scale 1` (V = the flight's noise), not 0.
3. Distill with `--ctrl ... --noise-aug 1` if an MLP is wanted (an LQG MLP keeps its Kalman filter).
4. Verify at dt 0.10/0.13/0.16, with `--noise-scale 1` (+ `--pos-latency-ms 5` for mocap), `push_sweep`,
   and `hover_mc` over 16 seeds. Gate tilt on the NO-PUSH flights (< 2 deg), and fly a position-free
   policy for 240-480 wingbeats (`--beats`): it walks and sinks, and the `gyro,acc` ones spiral after ~14 s.
   Seeds are shared across dt/scenario in `hover_mc`: the rate rests on the number of seeds, not flights.

- `sof` = static output feedback (gradient on the LQR cost, may report `ok: false`: no static gain
  stabilizes that subset); `lqg` = Kalman filter on the per-wingbeat model + LQR; the observer state is
  carried across wingbeats (`LinearController.reset()` per flight). Without a position channel the design
  drops the position states (pure integrators, unobservable): the bird then holds attitude and velocity
  but drifts (`att,w` at noise 1: 120 wingbeats, ~0.3 m drift). Which subsets hover (noise-free): finding
  `2026-09-26-hover-minimal-inputs` (with position tracked: LQG `pos,att,w`, no velocity needed; without:
  `att,w`, walking ~18 mm/s and sinking). `--dead-reckon` (lqg) adds integral action on the predicted
  position; it helps little (see the finding).
- Design any LQG that reads `acc` with `--noise-scale 1` (or more): at noise 0 (V at the floor, also what
  `verify_hover --lqr --obs ... --mode lqg` uses without `--noise-scale`) it tumbles after pushes, because
  the simulated IMU feels each instantaneous `kick` as an impulse the identified model lacks (Known bugs;
  finding `2026-09-26-hover-acc-reads-push-impulse`).
- Under datasheet noise (finding `2026-09-26-hover-imu-noise`), the LQG `pos,att,w` designed with
  `--noise-scale 1` passes everything at 1x (also with 5 ms mocap latency, and over 240 wingbeats) and
  crashes from 10x. Controllers designed at noise 0 still fly, but hover at the attitude filter's
  accel-bias tilt (~1-2 deg). IMU-only: `att,w` SOF needs a calibrated accel (5 mg) or `--noise-aug 1`
  distillation to stay level; the `gyro,acc` LQG / its MLP is level for ~14 s only. The accel bias is
  the term that breaks things. What did not help: `2026-09-26-hover-noise-tuning-dead-ends`.
- A reading is only as good as its model: check `hover.reduced --report` (observable states, LQG
  posterior std) before blaming the controller; pass `--att-mode truth` to separate filter error from
  control error, `--noise-scale 0 --att-mode truth` reproduces the true-state flight bit for bit.
- The attitude filter is calibrated on the orbit (the IMU's beat-mean specific force is ~5 deg off the
  mean up: lever arm under the flapping wobble); the accel bias still tilts it ~1 deg (20 mg). Yaw drifts
  with the residual gyro bias (no magnetometer).
- Monte Carlo runs on CPU (`hover_mc.py`, one flight per core, ~25 core-s per 120 wingbeats), not MJX:
  the sensor model and observers are CPU-only and the MJX closed loop has no free-flight parity check.

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
