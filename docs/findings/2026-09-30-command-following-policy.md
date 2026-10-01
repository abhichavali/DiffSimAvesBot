---
title: A one-trim LQR servo with ILC-learned feedforward flies the free bird on HOVER / FWD / REV / UP / DOWN commands at 0.3 m/s. It is learned and evaluated in about 206 s on one GPU, and mocap position + IMU is the smallest sensor set that follows every command.
date: 2026-09-30
status: current
tags: [free-flight, mjx, cpu, tooling]
---

# Prompt 2: a command-following closed-loop policy (how it was done)

> Status 2026-10-01: still current as the r18 method and its dead ends, but `command.tg` no longer ships r18: it ships `learn --preset fast` (V_op FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.45 m/s), finding `2026-10-01-command-flight-faster`.

This report covers Prompt 2 of `prompts/morphology_search.org` ("Find Optimal Closed Loop Policy"): hover,
forward, reverse, up and down on the free bird (`designs/hummer/free-hover`); learned and evaluated on the GPU
within 3-4 minutes; judged on random command sequences; with as few sensor inputs as possible under realistic
noise. It is long-form on purpose. The template's 15-40 line limit does not apply, because the request was
"a detailed report of how it was done".

Sources: the lab notes `opt/runs_cmd/NOTES.md` (sections 0-23 are cited as "NOTES n"; frozen copy in the repo: `docs/history/command_flight_notes_2026-09.md`), the run artifacts in
`opt/runs_cmd/<run>/` (`metrics.json`, `timings.json`, `dt.json`, `cpu.json`, `sensors.json`,
`xcheck/xcheck.json`), and the code. Both `opt/runs_cmd/` and every other `opt/runs*/` directory are
**gitignored** (`.gitignore:20`), so these artifacts exist only in the working copy that produced them. This
file, the code and the exported policy in `designs/` are what the repo keeps. Every number below was measured on
an RTX 5070 Ti (vast.ai) or on a CPU. Nothing is estimated unless it says so.

## 1. Summary

**The policy.** Once per wingbeat it adds a correction u (24 harmonic depths: a/b of 3 harmonics x 4 cranks) to
the free bird's trimmed open loop, as the hover LQR already does. No other actuation and no force of any kind is
used. The controller is a one-trim linear servo on the per-wingbeat hover model:

- The model (A, B, drift) is linearized **in x64 on the device**.
- Per command axis and speed, a tilt-bounded min-inf LP gives a steady-state feedforward (attitude + input).
  Two batched ILC passes of constant-command flights refine these tables.
- A rate-limited, shaped reference generator turns the command into a reference.
- An LQR servo with a position-error integral tracks it. The integral is handed off as the reference slows,
  and gated for 8 wingbeats after a stop.

Code: `src/hover/command.py` (host), `src/hover/mjx_command.py` (device), `opt/hover_cmd.py` (CLI). The shipped
policy is the run `opt/runs_cmd/r18`.

**Headline numbers, true state.** r18: 384 flights at V_op 0.3 m/s, each 240 wingbeats (28 s) of random
30-50-beat command segments (2238 segments), 96 of the flights also pushed. Float32 device, dt 0.13 ms.

| measure | value |
|---|---|
| survival (all / no-push / pushed) | 100% (384/384) / 288/288 / 96/96 |
| follow rate: segments passing | 98.6% overall; HOVER 100, FWD 100, REV 100, UP 100, DOWN 93.2% |
| steady velocity error, moving segments | \|along\| median 0.0035, p95 0.019 m/s; cross-track p95 0.028 m/s |
| hover hold (HOVER segments) | max \|p - p_ref\| p95 0.70 cm; \|v\| median 0.0010, p95 0.0038 m/s; drift p95 0.75 cm |
| path tracking | RMS \|p - p_ref\| median 2.5 cm, max p95 9.9 cm; 63% rise median 7 wingbeats |
| tilt / saturation | tilt deviation p95 5.3 deg, worst survivor 9.4 deg; 1.0% of settle beats saturated |
| envelope: measured post-ILC V_max | FWD 0.30, REV 0.40, UP 0.36, DOWN 0.30 m/s |
| vs plain hover LQR on the same schedules | the baseline survives 4.7% (3/64); servo / baseline \|along\| median 0.42 |
| other timesteps (0.10 / 0.16 ms, 64 flights x 240 beats) | 100% / 100% survival, follow 98.9% / 98.9% |
| CPU reference (float64 `hover.launch.fly`) | identical survival, 100% segment-verdict agreement at 0.10 / 0.13 / 0.16 ms |

**GPU wall time.** One process on the RTX 5070 Ti runs release, x64 linearize, design, 2 ILC passes and the
120-wingbeat evaluation:

- warm persistent XLA cache: **205.9 s** (r18), 208.3 s (r19); a verifier measured 204.4 / 209.1 s;
- cold cache: **242.9 s**, just over the 240 s (4 min) limit;
- the gates are graded on the untimed continuation to 240 wingbeats (+118.6 s) and the envelope verify
  pass (+37.8 s). See section 6.

**Minimal-input sensor variant (recommended): P = `pos,att,w`.** Three non-IMU channels: mocap COM position with
5 ms latency. On top of that, the on-board IMU attitude filter and the gyro. The observer is a trim-consistent
LQG. At datasheet noise x1 (128 flights x 120 beats, 3 seeds):

- survival 100 / 100 / 100%;
- follow rate 96.2 / 95.7 / 96.8%;
- \|along\| p95 0.033 / 0.034 / 0.031 m/s;
- hover hold p95 3.3 / 3.7 / 3.6 cm.

At x2 noise: 97.7% survival and 90.1% follow. At x3: 96.1% and 85.5%. Nothing without position follows the
command set. IMU-only att,w survives but slides sideways under FWD at 1.4 m/s and walks 6.7 m. Every
accelerometer dead-reckoning variant is lost.

**Where it lives and how to fly it.** `designs/hummer/free-hover/command.tg` (+ `command_policy.npz`,
`command_observer.npz`) is a flight control with two new keys: `command_loop` (the servo) and `command_observer`
(the P observer).

- Viewer: `src/view_free_hover.py --control designs/hummer/free-hover/command.tg [--sensors] [--noise-scale 1]`.
  Keys (seen side-on): hold Right / Left = forward / back, hold Up / Down = climb / descend (combine for a diagonal), release = hover, Home = hover, + / - = speed; hold-to-move needs pynput (macOS: allow Input Monitoring), else each tap toggles (speeds clipped to the envelope; a diagonal flies each axis at 1/sqrt(2)),
  P = push, O = toggle the loop. `--stdin` also takes one command per line
  (`FWD`, `UP 0.2`, `HOVER`, `PUSH`, `LOOP`).
- GUI (`plot_fixed_frequency_gui.py`), Viewer tab, box "Command flight (controls with command_loop)": buttons
  Hover / Forward / Reverse / Up / Down, a speed slider (0.05-0.4 m/s), Push, Toggle loop, and a "Fly on sensors"
  checkbox (applies at launch). The buttons write to the viewer's stdin.
- Command flight in the viewer is free air, as evaluated: contacts are off and the floor plane is moved 100 m
  down.

## 2. The task and its constraints

- **Commands.** HOVER, FWD, REV, UP, DOWN. Commands are velocities (m/s). The heading (yaw) is held at 0.
  "Forward" is the release heading.
- **Actuation.** Only the four crank motors. The policy changes the 24 harmonic depths of the phase-locked
  crank drive once per wingbeat. The crank speed g and a0 stay fixed (AGENTS.md "No fake forces": no assist
  wrench, no weight cancel after release). The verifier audited this on CPU and on the device (section 5.3).
- **Budget.** Learning plus evaluation in 3-4 GPU minutes. The box was an RTX 5070 Ti (16 GB), float32, full
  MJX solver (never `iterations=1`), line search 8, contacts off.
- **Evaluation.** Random command sequences: segments of 30-50 wingbeats, commands uniform over the five with no
  immediate repeat. The bird's true response must follow.
- **Sensors.** Use as few inputs beyond the IMU as possible, with realistic noise. The sensor model is Prompt 1's
  `hover.sensors`: IMU noise + bias, on-board attitude filter, mocap noise and latency.
- **Hygiene (from the build spec).** New files only for the method. `tests/test_hover_mjx.py` passes. The
  golden `check --quick` output is byte-identical to the pre-change capture: it fails 23/32 on this Linux box by
  design (AGENTS.md), so it is compared, not expected to pass. No commits.

## 3. How it was done

The work ran as a multi-agent workflow:

1. three independent method proposals;
2. a judge who picked a method, wrote a build spec with numeric gates G1-G13 and ordered fallbacks F1-F7;
3. a build;
4. three tune rounds, each preceded by a fresh-eyes diagnosis;
5. a sensor-subset round;
6. two verifiers: an adversarial CPU cross-check and a code review.

### 3.1 Design: three proposals and the judge's pick

| proposal | idea | the judge's use |
|---|---|---|
| **A, reference tracking.** A one-trim linear servo | One hover linearization. A steady-state feedforward from a tilt-bounded min-inf LP, affine in the commanded velocity. An LQR with a position-error integral (measured on `designs/.../lqr.npz`: closed loop \|eig\| 0.804 vs 0.656 for the plain LQR). The integral absorbs the linear model's velocity-dependent bias. The LQG observers of Prompt 1 are reused unchanged (separation). Its budget estimate was about 200 s warm. | **Core.** |
| **B, gain scheduling.** Envelope trims + gain schedule | "Moving-cradle" trims at commanded velocities (the launch cradle holding a moving target), batched Gauss-Newton over 24 depths + tilt with g fixed, per-trim FD linearization, and (u_ff, att*, K) interpolated by the command. Its numpy check: 0.5 m/s forward already needs \|u\| 0.147 against the 0.15 clip, because the per-beat velocity damping is strong (A vel->vel diagonal 0.43 / 0.80 / 0.72). | Fallback only (F2 local models, F4 moving-cradle trims). Never needed. |
| **C, learned policy.** Servo-seeded batch refinement | Its argument: a learned policy cannot beat an analytic servo in 3-4 GPU minutes (roughly 150-190k flight-wingbeats of budget), and DAgger only copies its teacher. So spend the GPU on what the linear model gets wrong. It proposed one batched pass of constant-command flights whose settled integral output is the feedforward error (ILC), a CEM over about 8 servo knobs, and an optional DAgger MLP, all run through **one compiled flight program** fed by per-flight traced arrays. | The one-program execution and the ILC pass were adopted. CEM was a fallback (never run). DAgger was dropped. |

The judge's reasoning:

- A is the least new code, all of it on verified modules (`mjx_flight.beat/kick/strobe/park`,
  `mjx_sensors.SensorPipeline`, `hover.reduced`).
- A needs no Gauss-Newton trims and no new shapes.
- The wall-clock floor is set by **serial physics steps**, not by synthesis: a 120-beat flight at 0.13 ms is
  120 x 901 steps. So only a design that spends almost no GPU time learning fits 240 s. That means closed-form
  design plus one 30-beat ILC pass.

The build spec fixed the rest:

- Command frames.
- LP tables per axis over a 15-point speed grid, for two tilt maps (5 and 15 deg).
- Superposed table lookup.
- The servo law with conditional-integration anti-windup.
- Metrics from TRUE strobed states.
- The ILC update rule.
- Two compiled shapes on the budgeted path (linearize 67x1, flight program B x 30).
- A one-process `learn` CLI with a stage clock.
- The gates (section 4.9) and fallbacks F1-F7.

### 3.2 Build (NOTES 0-13): runs r1-r7

The first build followed the spec and failed badly. Each fix below is a discovery that shaped the final method.

1. **Tumbling flights stall the vmapped batch (solver iterations 8).**
   - What happened: at B = 1024 one 30-beat ILC call took **296 s**, against about 30 s for hover flights. The
     pre-ILC envelope was only 0.10-0.16 m/s and 60 of 122 cells were lost.
   - Cause: tumbling flights' solver while-loops stall the whole batch.
   - Test (`scratch/probe_solver.py`): on the same inputs, iterations 8 against the model's 100 changes flights
     that fly by no more than the float32 run-to-run noise floor (the same state difference as a
     tolerance-only change). It also cut the call from 69.2 s to 27.3 s (2.5x; about 5x on the ILC pass).
   - Result: the command pipeline flies `--iterations 8`. The never-1 rule stands.
   - B dropped to 512. The bench (0.838 ms/step at B 512, 1.037 at 1024) had first suggested 1024.
2. **The linear model was the problem (x64 linearization).**
   - Device float32 FD linearization vs the CPU x64 `hover.lqr` reference: A off by 17-18%, B by 12%, the
     velocity columns by 33-34%. The error is *systematic*: 4 repetitions agree within 1%, and larger FD steps
     make it worse.
   - The committed `designs/hummer/free-hover/lqr.npz` is **stale** against the current code (A 15.6%,
     B 8.4%).
   - On the LP tables (21 cells x 12 configs), the x64 model passes 8-10 cells and the float32 model 4-5.
   - Fix: `learn` switches JAX to float64 for the single linearize call. It is exact against the CPU (r18:
     A and B relative error 1.4e-11) and costs about 1 s per 67-flight call.
   - With it the measured envelope rose from about 0.10 to 0.25-0.40 m/s (r4).
3. **Continuation ILC diverged at first.**
   - r2/r3 ran 4 passes that set the att/w references to window means of drifting flights. Good cells went
     24 -> 19 -> 19 -> 22 of 122, and V_op survival was 0%.
   - The final method uses 2 passes on top of the x64 model.
   - The ILC also stores the measured strobed angular rate w and a velocity offset per cell. The strobed w at
     speed is 4-7x the true rotation rate (an intra-wingbeat phase artifact), and with w_ref = 0 the w feedback
     fights it.
4. **Slower references (`--rate-beats 8`, r5).**
   - Reaching V_op in 8 beats instead of the spec's 4 raised 240-beat survival from 37.0% (r4) to 78.4%.
   - Most remaining losses were FWD -> DOWN: yaw runs away (-2 -> -34 deg in 10 beats), then lateral velocity,
     saturation and tumbling.
5. **Descent needs the horizontal stop first (`--descend-after-stop 0.05`, r7).**
   - A descent starts only once the horizontal reference speed is below 0.05 m/s.
   - With DOWN at 0.24 m/s (r6, fallback G7) survival reached 96.9%. With the shaping (r7) it reached 99.5%.
   - On CPU, descents at 0.16-0.24 m/s showed a sudden event: the passive wing-pitch extremes change within
     2 beats, then vz jumps and pitch diverges. This is consistent with (not proven to be) the bistable
     passive-pitch gotcha (`2026-09-24-passive-pitch-bistable`).
6. **The CPU/GPU split was the floor (F7).**
   - At first the CPU reference lost flights the device flew (9/16 survival agreement, 75% of verdicts).
   - Cause: the scene has a floor 0.25 m below the release COM. The CPU crashes into it (struts touch from
     beat 13), while the device flies with contacts off and passes through. In `final`, 212 of 384 eval
     flights go below the first-touch depth.
   - An x64 device run with the CPU's exact knobs (contacts on) lost the same flights at the same beat (33),
     so the port is exact.
   - Consequence: **the evaluation is free air**, and the `cpu` stage now runs contacts off (`--cpu-contacts`
     keeps them). r7 with contacts off: 16/16 identical survival, 98.9% verdict agreement.
7. **Sensor variants failed at speed.**
   - The Prompt 1 observers flew P (pos,att,w) at 5.5-44% survival.
   - The attitude estimate was 5-20 deg off in sustained FWD/UP. This was first blamed on the Mahony filter.
     Diagnosis 3 overturned that attribution (section 3.6).

r7 result: survival 99.5%, follow 96.7%, and G7 failed. The failing gates were:

- FWD follow 87.8%;
- hover drift p95 3.2 cm (gate 2 cm);
- max \|p - p_ref\| p95 18.2 cm (gate 15);
- DOWN envelope 0.24 < 0.30;
- the sensor gates.

Warm budget 209.6 s; cold 247.5 s (`final_cold`).

### 3.3 Tune round 1 (NOTES 14-15): r8a-r11b

Blockers from r7:

- FWD failed on *cross-track* (0.051-0.084 m/s, always +y) with 8-80% of settle beats saturated.
- The worst hover drifts all came right after FWD.
- The DOWN 0.32 cell followed (along 0.017 m/s) but was 73% saturated, so only the saturation rule failed it.

Changes and their effects:

- **Vertical speed grid 0.06-0.90 m/s** (0.30 is now a grid point) and DOWN back at V_op 0.30. In r8a,
  98% of DOWN segments pass. The old DOWN failure had been the FWD -> DOWN transition, which the r7 shaping
  fixed.
- **Separate learning and flight gains.**
  - The ILC must learn with the stiff integral (q_int 100). Learning with q_int 30 (r9) gave 79.9% survival,
    because its integral had not settled inside the 15-30-beat ILC window.
  - The policy flies a softer gain: `--fly-q-int 50 --fly-q-vel 0.15`. r10b reached 100% survival. The
    replicate r10c reached 98.7%, so the configuration is about 99 +- 1%.
- **`--descend-delay 8`**: a descent waits 8 more wingbeats after the horizontal stop. It was swept on fixed
  tables: delay 8 gave 100%, while 16 lets DOWN follow only 60%.
- **P observer with attitude-bias states** (`--p-bias att_xy`) and a soft integral (`--p-q-int 3`): P x1
  survival went from 21% to 62-99%. A faster bias walk was a dead end.
- **N (IMU-only) velocity ILC** (`velocity_ilc` + `trim_sensitivity`): during learning only, each cell's trim
  moves by the linear trim sensitivity times the *true* mean-motion velocity error. The flying policy still
  reads only the IMU. N direction-follow went from 76.9% to 100%.

Round-1 policy r11b:

- survival 99.7%, follow 97.7%, post-ILC DOWN 0.30 (G7 passes for the first time), warm budget 209.0 s;
- G4 drift still fails: 3.57 cm;
- P follow is 77.7%.

### 3.4 Diagnosis 2 (NOTES 16)

**Hover drift.** Drift p95 by the command before the hover segment:

| preceding command | drift p95 (cm) |
|---|---|
| release | 0.25 |
| REV | 1.47 |
| UP | 1.93 |
| DOWN | 1.99 |
| FWD | 4.76 |

- The **FWD 0.30 cell was never ILC-measured**. Under the learning gain it was 100% / 93% saturated, so the
  `good_sat 0.6` rule dropped it in both passes. Its table entry was extrapolated.
- The soft flight gain covers that table error with the integral: 0.106 of the 0.125 mean correction during
  FWD comes from the integral. At the stop, the stored integral (0.39 sideways) pushes the bird off by
  8.3 cm, peaking 10 beats into HOVER. That is exactly what G4 measures.

**The remaining losses are real dynamics.**

- The CPU replays the two lost flights at the same beats: flight 238 at beat 164-169 on CPU (device 165);
  flight 438 at 153 on CPU (device 152).
- Flight 238 runs away sideways in a DOWN -> REV transition. The ILC's learned yaw references swing by up to
  7 deg between cells.
- The CPU check could also hide a blow-up: MuJoCo printed "Nan, Inf or huge value in QACC" and silently reset
  the state. The `cpu` stage now counts such a warning as the loss beat.

### 3.5 Tune round 2 (NOTES 17-18): r12-r18

- **Learning from saturated cells** (r12, the diagnosis's first suggestion): WORSE, drift 5.1 cm. The 93%
  saturated FWD 0.30 flight writes a 9.2 deg yaw reference.
- **Flying the last ILC pass with the flight gain** (r13): WORSE, survival 95.3%.
- **Integral handoff** (`--zeta-handoff`, `integral_keep`): as the reference decelerates, the integral is scaled
  by keep_k = clip(v_{k+1}.v_k / \|v_k\|^2, 0, 1), so a stop bleeds off the integral that held the moving trim's
  table error. r14: survival 100%, drift 2.40 cm.
- **Integral gate** (`--zeta-gate 8`, `integral_gate`): no integration while HOVER is commanded and the
  reference stopped less than 8 beats ago. The stop transient is left to the proportional loop, because
  integrating it wound up an undershoot. Paired sweep on r14's tables, drift p95:

  | variant | drift p95 (cm) |
  |---|---|
  | no handoff | 2.24 |
  | handoff | 2.08 |
  | + gate 4 | 1.31 |
  | + gate 8 | 0.82 (100% survival) |

- **Feedforward lead** (`--ff-lead 1`): att / w / u_ff are looked up at v_ref one beat ahead, because the
  attitude lags a velocity change. On r15's tables, gate 8 + lead 1 gave drift 0.58 cm, max p95 8.5 cm and
  cross p95 0.024. Lead 2 dropped FWD to 85%; lead 3 dropped survival to 64%.
- **The ILC "good cell" criterion** was the fragile knob:
  - `good_sat 0.6` flipped DOWN 0.30 in and out between runs, at 9-10 of 15 window beats saturated (r15:
    post-ILC DOWN 0.24).
  - 0.8 let FWD 0.30 in at 12/15 saturated with cross 0.095 m/s and spoiled FWD (r16: saturation 14.1%).
  - Final: `--good-sat 0.7 --good-cross 0.08`. A cell is measured when it is alive, saturated for at most 70%
    of the window, has \|along\| <= max(0.05, speed/2) and cross <= 0.08 m/s. Other cells are extrapolated.
- **Lead in learning** (r17): FWD 85.3%. Using the lead in the ILC passes moved the learned FWD 0.20/0.25
  entries, so the extrapolated 0.30 entry saturated. Hence `--learn-ff-lead 0`: learn without the lead, fly with
  it.
- **r18 = the round-2 policy, all defaults.** Every learn gate PASSES (table in section 1). The replicate
  r18b also passes every learn gate (budget 205.4 s).
- The sensor variant P was still failing: 93.0% survival, 80.9% follow. The best variant (design noise x3)
  reached 97.7% / 88.2%.

### 3.6 Diagnosis 3 (NOTES 19): the P observer's process model

- **Root cause.** The P observer predicts with the *hover* model plus a constant hover drift, but it is fed the
  full input including u_ff. The ILC-learned moving trims are not equilibria of that model. At V_op the per-beat
  mismatch reaches 0.115 m/s in velocity, 0.81 rad/s in rate and 2.3 deg in attitude (FWD). The Kalman filter
  absorbs it through its slowest states (the att_xy bias and velocity, error modes 0.953 / 0.941) and reports a
  fake 3-6 deg tilt bias plus a 0.06-0.14 m/s velocity error.
- **Evidence** (`scratch/obs_replay.py`). The same observer, replayed offline on r18's *true-state* flights
  with a noise-free sensor and no Mahony filter, reproduces the device errors:

  | command | replay (median) | device (median) |
  |---|---|---|
  | FWD | roll +2.9 deg, vz -0.097 m/s | roll +3.2 deg, vz -0.139 m/s |
  | UP | vx +0.057 m/s | vx +0.074 m/s |

  So the build's attribution to the attitude filter is **not supported**.
- **Fix** (`trim_consistent_cz`). A per-beat observer offset makes the reference the observer model's own
  trajectory:

  cz_k = x_ref_k - Az x_ref_{k-1} - Bz u_ff_{k-1} (design rows; 0 on copy and bias rows)

  In the offline replay it halves the errors: attitude RMS 1.44 -> 0.69 deg, velocity RMS 0.039 -> 0.025 m/s,
  REV roll 3.5 -> 0.26 deg.

### 3.7 Tune round 3 (NOTES 20-21): the P sensor variant

| run (P change; all with cz_seq and the true-state tables) | P x1 survival / follow / hold p95 | P x3 survival |
|---|---|---|
| r18, round 2 (hover-drift observer) | 93.0% / 80.9% / 13.6 cm | 85.2% |
| r18_cz: cz_seq | 95.3% / 90.9% / 3.6 cm | 91.4% |
| + design noise x3 | 97.7% / 87.4% / 4.9 cm | 82.0% |
| + sensor-ILC tables | 85.2% / 78.0% / 4.2 cm | 82.8% |
| + q_int 10 / 30 / flight gain | 95-96% / 86-89% | 85-91% |
| + climb after stop 0.05 | 100% / 91.7% / 3.6 cm | 94.5% |
| + process noise x3, bias walk 9e-4 | 100% / 95.2% / 3.2 cm | 96.1% |
| + process x3 + climb 0.05 (the defaults) | 100% / 96.2% / 3.3 cm | 96.1% |

- Every remaining r18_cz loss was a **FWD -> UP** transition (6 of the 15 in 128 flights). A noise-free replay
  reproduces the device's 0.3 m/s velocity-estimate error beat for beat: the observer's process model is wrong
  in that transient.
- Two independent fixes each helped:
  - process noise x3, so the observer trusts mocap position more and the velocity estimate reacts faster;
  - `climb_after_stop` 0.05: a climb waits until the horizontal v_ref is at or below 0.05 m/s.
- Both are now defaults of `sensors`.
- The offline process-noise sweep did *not* predict the device gain; only the device measured it.
- P flies the **true-state tables** (`--p-tables true`) with cz_seq. Its own sensor-ILC tables flew worse.
- r19 (`learn`, all defaults, after the round-3 code) passed every learn gate with r18-like numbers. Section 6
  explains why "reproduces r18" is too strong.

### 3.8 Sensor-subset round (NOTES 22)

Five input sets were flown on r18's policy at noise x1 and x2 (plus x3 for P). The table is in section 5.6.
P = pos,att,w is the smallest set that follows all five commands. Adding mocap velocity does not help. The
IMU-only and accelerometer variants fail.

### 3.9 Verification (NOTES 23 and a code review)

- **CPU cross-check** (`opt/runs_cmd/xcheck/`, 108 CPU flights, 1541 s on 18 processes):
  - The true-state policy on float64 `hover.launch.fly` at dt 0.10 / 0.13 / 0.16 has identical survival and
    100% segment-verdict agreement with the device.
  - P was rebuilt on CPU and flown with `hover.sensors` noise x1: 16/16, 24/24 and 16/16 survive, with follow
    95.6-97.8%.
  - Force audit: none.
- **Reproducibility.** The learn stage is **not** bit-reproducible (section 6).
- **Review.** It found:
  - the DOWN metric ceiling;
  - the too-weak IMU-only gate;
  - optimistic reporting of G10 cold and G11;
  - generous survival counting at the last strobe;
  - no latency on the velocity channel;
  - `all` not forwarding `--tg`.

  The IMU-only gate was fixed after the review: `sensors` now gates N on the same follow rate as P
  (`opt/hover_cmd.py` `cmd_sensors`). The rest is listed in section 6.

### 3.10 After verification: export and live commands

- `opt/hover_cmd.py export` packages a run as a flight control .tg:
  - the free-hover drive + trim;
  - `command_loop` = the policy npz plus the sensor servo gain (q_int 3), climb shaping, 5 ms latency and the
    measured envelope;
  - `command_observer` = the P observer, designed exactly as `sensors` designs it.
- `hover.command.RefStream` is `build_refs` one wingbeat at a time, for a command that changes live. It matches
  `build_refs` row for row when ff_lead <= 1.
- `hover.command.LiveCommand` flies either the true state (12) or the 18-channel sensor reading. In the sensor
  case the observer computes cz_seq online.
- Tests: `test_live_reference_stream_equals_build_refs` and `test_live_command_equals_the_evaluated_controllers`.
  LiveCommand equals CommandPolicy and CmdLinearController + trim_consistent_cz to 1e-12.

## 4. The method in detail (the final controller)

### 4.1 Frames and commands

- **State.** The strobed state `s` (12) is `hover.launch.strobe - x_orbit`, taken once per wingbeat: world
  position, attitude rotation vector against the release attitude, COM velocity, and world angular rate.
- **Forward axis.** e_f = the release attitude's body +x projected on the horizontal plane
  (`forward_axis`): [0.99993, -0.0119, 0].
  - The head mesh is not in this checkout, so the sign could not be checked on the geometry. The repo
    convention was used: +x is forward (`hover.evaluate.VERIFY_PUSHES`).
- **Command directions** (`command_dirs`): HOVER = 0, FWD = +e_f, REV = -e_f, UP = +z, DOWN = -z.
- **Operating speed.** V_op = 0.3 m/s for every axis (`v_op` [0.3, 0.3, 0.3]).
- **Attitude error.** Plain rotation-vector subtraction, which is fine at <= 15 deg.
- **Wingbeat.** dt 0.13 ms: n_step 901, T = 0.117075 s (8.5416 Hz).

### 4.2 The linear model

- `device_linearize` (`opt/hover_cmd.py`) runs `hover.mjx_flight.lqr_tasks`: 67 one-wingbeat central-difference
  rollouts from the released state, as one vmapped call in **float64**. JAX is switched to x64 for this call
  only; the model's own solver settings are used. `mjx_flight.ab_from_tasks` then gives A (12x12), B (12x24)
  and the drift.
- r18: identical to the CPU `hover.lqr` reference (`opt/runs_gpu/ref/lqr.npz`) to 1.4e-11.
- Spectral radii (\|eig\| max): open loop 1.113, hover LQR 0.651, servo 0.804.

### 4.3 Steady-state feedforward tables (LP) and their refinement

**The LP** (`steady_state`, `steady_state_tables`). For each tilt map (5 and 15 deg), axis and grid speed s:

- unknowns: att (3; yaw pinned to 0), u (24) and t;
- constraints: the att / vel / w rows of x_{k+1} = x_k at x = [*, att, s*dir, w = 0], including the drift;
  \|roll\|, \|pitch\| <= tilt_max; \|u_i\| <= t;
- objective: minimise t + 1e-3 \|u\|_1 (`linprog`, HiGHS);
- grids: horizontal 0.05-0.75 m/s (15 points), vertical 0.06-0.90 m/s (15 points), plus hover as entry 0;
- 2 maps x (1 + 4 x 15) = **122 cells**;
- the LP residual is <= 1e-9 and \|u_ff(0)\| is 0.0043 (the tests check both).

**The LP's own envelope** (speed where t reaches u_max/2 | u_max), in m/s:

| map | FWD | REV | UP | DOWN |
|---|---|---|---|---|
| 5 deg | 0.37 \| 0.74 | 0.39 \| 0.75 | 1.04 \| 1.92 | 1.07 \| 1.95 |
| 15 deg | 0.37 \| 0.75 | 0.39 \| 0.77 | 1.30 \| 2.26 | 1.31 \| 2.29 |

**Lookup** (`lookup`). At a reference velocity v, the FWD/REV entry at \|v.e_f\| plus the UP/DOWN entry at
\|v_z\| minus the hover entry. Interpolation is piecewise-linear in speed, clamped at the grid ends.
Superposition covers the mixed velocities of transitions. It returns att_ref, vel_ref = v + dv (a learned
strobed-minus-mean-motion offset), w_ref and u_ff.

**ILC** (`ilc_update`), per measured cell, over the alive flights and window beats 15-30:

- u_ff += the mean feedback effort (U - u_ff - dither), clipped to 0.9 u_max;
- att_ref := the mean strobed att; w_ref := the mean strobed w;
- dv := strobed minus mean-motion velocity, with its along-command component clipped to +-20% of the speed.

The fixed point has zero feedback on att, vel and w.

**Which cells are measured** (`run_cell_pass`): alive, saturated for <= 70% of the window, \|along\| <=
max(0.05, speed/2) and cross <= 0.08 m/s.

**Continuation** (`extrapolate_tables`). A cell that is not measured gets its axis's measured correction
(table - LP) extrapolated linearly in speed from the two largest good speeds, capped at 1.5x the largest good
speed (`--extrap 1.5`).

**Map choice.** The larger minimum-over-axes V_max after the last pass wins; a tie goes to the 5 deg map. r18
chose 5 deg. A sum tie-break once picked the 15 deg map, which flew much worse (74.2% survival); it was reverted.

### 4.4 Gains

`servo_gains`: an LQR on [x; zeta], where zeta_{k+1} = zeta_k + (p_k - p_ref_k):

- Q, R from `hover.lqr.cost` tolerances: pos 0.02 m, att 0.05 rad, vel 0.1 m/s, w 0.5 rad/s, input r 0.05;
- q_int = 100 on the integral (1/0.1^2);
- solved with `solve_discrete_are`, which gives Ka (24x15).

| gain | used for | settings |
|---|---|---|
| learning gain `Ka_learn` | ILC passes | q_int 100; rho 0.804 |
| flight gain `Ka` | evaluation, dt, cpu, viewer | q_int 50, q_vel 0.15 m/s |
| sensor gain `sensor_Ka` | P variant on the estimate | q_int 3 (a biased estimate winds up a stiff integral) |
| baseline | G8 comparison | the plain hover LQR K with zero integral columns, u_ff = 0, x_ref = [p_ref, 0, 0, 0] |

### 4.5 Reference generation (`build_refs`; live: `RefStream`)

- **Rate limit.** v_ref_{k+1} = v_ref_k + clip(v_cmd_k - v_ref_k, +-V_op/8) per world axis (`--rate-beats 8`):
  V_op is reached in 8 wingbeats (about 0.94 s).
- **Position.** p_ref_{k+1} = p_ref_k + v_ref_k T. With mocap, the bird tracks the commanded path, not only its
  velocity.
- **Descend after stop.** A descent (v_ref,z lowered below 0) is held while the horizontal v_ref exceeds
  0.05 m/s, and for 8 more wingbeats after that (`--descend-after-stop 0.05 --descend-delay 8`).
- **Climb after stop.** For the P variant only (`--p-climb-after-stop 0.05`, no delay), a climb waits likewise.
- **Feedforward lead.** att / w / u_ff are looked up at v_ref one beat ahead in flight (`ff_lead 1`). The ILC
  passes learn without the lead.
- **x_ref layout.** x_ref = [p_ref, att_ref, vel_ref, w_ref, keep, gate], 14 columns:
  - keep = `integral_keep` (handoff factor);
  - gate = `integral_gate` (0 while HOVER is commanded and the reference stopped less than 8 beats ago).

  The host precomputes x_ref and u_ff per beat, since they do not depend on the state, and feeds them to the
  device as traced arrays.

### 4.6 The control law (host `servo_law` == device `mjx_command.law`)

```
e     = s - x_ref[:12]
raw   = u_ff + dither - Ka[:, :12] e - Ka[:, 12:] zeta
u     = clip(raw, -0.15, 0.15)
free  = 0 if any |raw_i| >= 0.15 else 1                 # conditional integration (anti-windup)
zeta' = clip(keep * zeta + gate * free * (p - p_ref), -1, 1)     # [m * wingbeat]
theta_k = theta_trim + [0, u]                           # g (crank speed) never changes
```

For the sensor variant, `s` is the observer estimate:

- `CmdSensorPipeline.control` on the device; `CmdLinearController` / `LiveCommand` on the CPU.
- The observer predicts zp = Az z + Bz u_prev + cz_seq[k] and corrects z = zp + L (y - J u_prev - Cz zp).

The opt-in `allocate` (the box projection in the servo's Q-function metric, FISTA) stays in the code with
`--alloc-iters 0`. It flew worse (section 7).

### 4.7 The GPU pipeline (`opt/hover_cmd.py learn`, one process, B = 512)

| stage | what | r18 warm (s) | r18_cold (s) |
|---|---|---|---|
| imports | numpy, jax | 0.5 | 0.5 |
| release | `hover_gpu.flight`: cradle release on CPU, build the float32 `mjx_flight.Flight` | 2.8 | 2.8 |
| linearize | 67 x 1-beat FD rollouts in x64 (shape 1) | 6.8 (5.7 cache load) | 22.9 (21.7 compile) |
| design | LP tables (2 maps x 64 cells), DAREs | 0.5 | 0.6 |
| linear sim | 200 random schedules on (A, B), sanity check | 0.1 | 0.1 |
| ILC pass x 2 | 512 x 30 beats (shape 2): 122 clean constant-command cells + 390 dithered copies (N(0, 0.01) on u), loss tilt 35 deg; ILC update + extrapolation on the host | 76.1 (35.8 + 35.0; 5.0 cache load) | 96.2 (24.0 compile) |
| eval 120 | the same 512 x 30 program chained 4x through a batched carry | 119.0 (28.2 / 29.2 / 30.9 / 30.3) | 119.8 |
| metrics, write | policy.npz, metrics.json, timings.json, traces.npz | 0.1 | 0.1 |
| **budget** (process start to here) | | **205.9** | **242.9** |
| untimed: continue to 240 | 4 more chained calls on the same flights | 118.6 | - |
| untimed: envelope verify | one post-ILC cell pass flown with the flight gain -> post-ILC V_max | 37.8 | - |

Why it fits the budget:

- **Two compiled shapes only.** Every run-specific quantity is a per-flight traced array: per-beat x_ref / u_ff /
  dither / kicks, the gain Ka, the loss tilt. So the ILC passes, the evaluation, the baseline arm and the verify
  pass all share one program.
- **Persistent cache.** `JAX_COMPILATION_CACHE_DIR` with `jax_persistent_cache_min_compile_time_secs = 0` turns
  the compiles into about 5 s cache loads.
- **Serial physics sets the wall time.** One 30-beat call is 27,030 serial physics steps, about 30 s at B 512.
  The run is 6 such calls plus about 15 s of overhead.
- **No gradients** are taken through a flight, so the BPTT divergence (`2026-09-24-bptt-diverges-past-one-wingbeat`)
  does not apply.
- **iterations 8** keeps tumbling flights from stalling a batch.
- **Shapes change with code.** Changing the device law or the x_ref width (12 -> 13 -> 14 columns) is a new
  program and a cache miss (r14: first call 18.6 s). A warm first call needs the code frozen between runs.

Evaluation batch (`eval_inputs`), 512 flights sharing one seed:

- 384 servo flights at V_op (75%). The last quarter of them (96) get one verify-sized push
  (`hover_gpu.VERIFY_PUSHES`) 5 beats into their second segment.
- 64 servo flights with magnitudes U[0.1, pre-ILC V_max] ("uniform").
- 64 baseline hover-LQR flights, on the schedules of the first 64 V_op flights.

### 4.8 Evaluation protocol (`random_schedules`, `segment_metrics`, `summarize`)

- **Schedules.** Segments of uniform length 30-50 wingbeats; commands uniform over the five with no immediate
  repeat; the first segment is random too. Every flight starts from the hover release.
- **Scoring.** Everything is scored from the **true strobed states**, over the **second half** of each
  segment.
- **Velocity.** v_seg = mean-motion velocity (p_end - p_mid) / ((end - mid) T). along = v_seg.dir - speed;
  cross = the off-axis part.
- **Moving pass:** \|along\| <= max(0.05, 0.15 x speed) and cross <= 0.05 m/s.
- **Hover pass:** \|v_seg\| <= 0.02 m/s and max \|p - p_ref\| over the half <= 5 cm. Hover drift = \|p_end - p_mid\|.
- **Rise.** Beats until \|v - v_cmd\| is within 37% of the step (63% rise).
- **Saturation.** The fraction of second-half beats with any \|u\| = u_max.
- **Survival.** `mjx_flight.lost_at`: non-finite, or tilt > 60 deg. A lost flight's remaining segments fail.
  Segments that the horizon cuts below 30 flown beats are dropped.

### 4.9 Gates: thresholds and final values

Learn gates on eval_240 (384 V_op flights). G7 comes from the post-ILC verify pass.

| gate | threshold | r18 | r19 |
|---|---|---|---|
| G1 survival all / no-push / push | >= 99% / 100% / >= 99% | 100 / 100 / 100% | 100 / 100 / 100% |
| G2 follow all; each command | >= 95%; >= 90% | 98.6%; DOWN 93.2% lowest | 98.6%; DOWN 93.2% |
| G3 \|along\| median / p95; cross p95 | <= 0.02 / 0.05; <= 0.05 m/s | 0.0035 / 0.019; 0.028 | 0.0035 / 0.020; 0.026 |
| G4 hold p95; \|v\| median / p95; drift p95 | <= 5 cm; <= 0.01 / 0.02 m/s; <= 2 cm | 0.70 cm; 0.0010 / 0.0038; 0.75 cm | 0.68 cm; 0.0011 / 0.0036; 0.70 cm |
| G5 RMS median; max p95; rise median | <= 4 cm; <= 15 cm; <= 8 beats | 2.5 cm; 9.9 cm; 7 | 2.5 cm; 9.9 cm; 7 |
| G6 tilt-dev p95; worst survivor; settle saturation | <= 15 deg; <= 45 deg; <= 5% | 5.3; 9.4 deg; 1.0% | 5.2; 9.1 deg; 1.2% |
| G7 post-ILC V_max FWD / REV / UP / DOWN | >= 0.30 each | 0.30 / 0.40 / 0.36 / 0.30 | 0.30 / 0.40 / 0.48 / 0.30 |
| G8 servo / baseline \|along\| median | <= 0.5 | 0.42 | 0.41 |
| G9 dt 0.10 / 0.16: survival, follow, \|along\| median | 100%, >= 90%, <= 0.03 | 100%, 98.9%, 0.0036 / 100%, 98.9%, 0.0041 | (r18) |
| G10 warm budget; first call per shape; shapes | <= 240 s; <= 10 s; <= 2 | 205.9 s; 5.7 / 5.0 s; 2 | 208.3 s; 5.7 / 5.1 s; 2 |
| G10 cold (reported) | (<= 240 s) | **242.9 s: over** (first calls 21.7 / 24.0 s) | - |
| G11 CPU: survival, verdicts, \|dv_seg\| median / max | identical, >= 95%, <= 0.005 / 0.02 m/s | 16/16, 100%, 0.00023 / 0.0028 | (r18) |
| G11 state error / ERR_SCALE, beats 0-120 | <= 1.0 | **FAIL**: 0.56-2.95 on V_op flights (85.5 on a tumbling baseline) | - |
| G12 P x1: survival, follow, \|along\| p95, hold p95; P x3 survival | >= 99%, >= 90%, <= 0.08, <= 8 cm; >= 95% | 100%, 96.2%, 0.033, 3.3 cm; 96.1% (r18_s3) | - |
| G12 N x1: survival; follow rate (was direction-follow) | >= 99%; >= 90% | 100%; **47.6% FAIL** (old direction-follow 99.7%) | - |
| G13 tests, golden, no module changed | pass / identical / - | pass / identical (see section 9) | - |

The G12 thresholds are looser than the learn gates (hold 8 vs 5 cm, \|along\| p95 0.08 vs 0.05, follow 90 vs
95%), and G12 has no per-command gate. The looser limits were set in the build spec for estimate-fed flight. The
per-command P rates are given in sections 5.5 and 5.6.

## 5. Results

### 5.1 True state, per command (r18, eval_240: 384 V_op flights, 2238 segments)

| command | segments | pass rate | \|along\| median / p95 (m/s) | cross p95 (m/s) | rise median | settle saturation | tilt-dev p95 |
|---|---|---|---|---|---|---|---|
| HOVER | 458 | 100% | \|v\| 0.0010 / 0.0038; hold p95 0.70 cm, drift p95 0.75 cm | - | 7 | 0% | 3.6 deg |
| FWD | 435 | 100% | 0.0032 / 0.0101 | 0.020 | 9 | 0% | 4.4 deg |
| REV | 454 | 100% | 0.0014 / 0.0102 | 0.015 | 6 | 0% | 4.4 deg |
| UP | 433 | 100% | 0.0034 / 0.0055 | 0.012 | 6 | 0% | 5.1 deg |
| DOWN | 458 | 93.2% | 0.0099 / 0.0715 | 0.040 | 10 | 4.95% | 4.9 deg |

| arm | beats | flights | survival | follow |
|---|---|---|---|---|
| V_op, no push | 240 | 288 | 100% | 98.4% |
| V_op, pushed | 240 | 96 | 100% | 99.3% |
| uniform (U[0.1, pre-ILC V_max]) | 240 | 64 | 100% | 100% |
| baseline hover LQR | 240 | 64 | 4.7% | 20.8% |
| baseline hover LQR | 120 | 64 | 17.2% | 36.6% |

At 120 beats: V_op survival 100%, follow 98.7% (DOWN 94.0%). The linear-model simulation (200 schedules on
A, B) follows 98.6% with no saturation.

### 5.2 Timestep robustness (`dt --run opt/runs_cmd/r18`)

The fixed policy flew the first 64 no-push V_op schedules for 240 beats at each timestep:

| dt | n_step | survival | follow | \|along\| median / p95 | hold p95 | DOWN pass | wall (compile) |
|---|---|---|---|---|---|---|---|
| 0.10 ms | 1171 | 100% | 98.9% | 0.0036 / 0.019 | 0.91 cm | 94.9% | 354 s (90 s) |
| 0.16 ms | 732 | 100% | 98.9% | 0.0041 / 0.023 | 0.82 cm | 94.9% | 161 s (16 s) |

Earlier policies were less robust. r8a at 0.16 ms: 82.8%; r10b at 0.16 ms: 92.2%; r11b at 0.10 ms: 98.4%.

### 5.3 CPU cross-check

**True-state policy, float64 CPU vs float32 device** (`xcheck/xcheck.json`; same flights and dt, 240 beats;
model solver, contacts off):

| dt | CPU: survival, follow, \|along\| median / p95, hold p95 | GPU: survival, follow, \|along\| median / p95, hold p95 | verdicts agree | \|dv_seg\| median / max | position diff max |
|---|---|---|---|---|---|
| 0.10 | 16/16, 97.8%, 0.0048 / 0.030, 0.90 cm | 16/16, 97.8%, 0.0044 / 0.030, 0.91 cm | 100% (92) | 0.00017 / 0.0026 | 1.4 cm |
| 0.13 (4 pushed) | 20/20, 98.3%, 0.0038 / 0.031, 0.67 cm | 20/20, 98.3%, 0.0039 / 0.031, 0.94 cm | 100% (116) | 0.00028 / 0.0036 | 2.2 cm |
| 0.16 | 16/16, 97.8%, 0.0054 / 0.028, 1.31 cm | 16/16, 97.8%, 0.0062 / 0.028, 1.59 cm | 100% (92) | 0.00040 / 0.0085 | 4.3 cm |

- DOWN is the only command below 100% on both (92-93%). There were no MuJoCo warnings.
- **The r18 `cpu` stage** (16 flights: 8 plain, 4 pushed, 4 baseline) agreed as well: identical survival,
  100% of 93 segment verdicts, \|dv_seg\| median 0.00023 and max 0.0028 m/s. The new MuJoCo-warning rule caught
  the CPU blow-up of baseline flight 449 (CPU lost at 29, GPU at 31).
- **F7, port exactness.** The same flights re-flown on the GPU in **x64** with the CPU's solver settings
  (`scratch/f7b.py r18 120`) match the CPU to 2.6e-8 .. 1.6e-7 of ERR_SCALE on the V_op flights. The float32
  state error (0.56-2.95) is therefore float32 drift of chaotic command flights, not a port bug. The verdicts
  agree.
- **P on CPU** (`hover.launch.fly` + `CmdLinearController`-equivalent + `hover.sensors` noise x1, 120 beats):

  | dt | survival | follow | \|along\| p95 | hold p95 |
  |---|---|---|---|---|
  | 0.10 | 16/16 | 97.8% | 0.043 | 3.2 cm |
  | 0.13 | 24/24 | 95.7% | 0.041 | 2.8 cm |
  | 0.16 | 16/16 | 95.6% | 0.043 | 3.6 cm |

  At 0.13 the GPU on the same 24 schedules gives 24/24, 95.7%, 0.040 m/s and 2.9 cm, with 100% verdict
  agreement. Position drift over beats 60-120 is about 11-12 cm p95 on both.
- **Force audit** (every step after release in all 108 CPU flights, 17.9 M steps):
  - xfrc_applied max 0;
  - commands to non-crank actuators 0 (nu = 4: L/R_motor_t1/t4);
  - qfrc_applied equals aero + the pitch limiter exactly;
  - neq 0, nmocap 0, gravity -9.81.

  The device shows the same (xfrc 0, nu 4, neq 0, nmocap 0 at all 3 dt).

### 5.4 Envelope (r18)

V_max = the largest grid speed such that every speed up to it passes a 30-beat constant-command flight
(\|along\| <= max(0.05, 15%), cross <= 0.05, saturation <= 20%, alive). All values in m/s.

| | FWD | REV | UP | DOWN |
|---|---|---|---|---|
| LP prediction, half input budget (5 / 15 deg) | 0.37 / 0.37 | 0.39 / 0.39 | 1.04 / 1.30 | 1.07 / 1.31 |
| pre-ILC (both maps) | 0.25 | 0.40 | 0.36 | 0.24 |
| after ILC pass 2 (5 / 15 deg) | 0.25 / 0.25 | 0.40 / 0.40 | 0.42 / 0.36 | 0.24 / 0.24 |
| **post-ILC verify, 5 deg map (chosen)** | **0.30** | **0.40** | **0.36** | **0.30** |
| post-ILC verify, 15 deg map | 0.25 | 0.40 | 0.42 | 0.24 |
| r19 post-ILC, 5 deg map | 0.30 | 0.40 | 0.48 | 0.30 |

- Good (ILC-measured) cells: 52 and 55 of 122 in the two passes.
- Horizontal speed is limited by the bird's strong velocity damping, which Proposal B also noted.
- Vertical speed is limited far below the LP's prediction. On CPU, descents show the sudden passive-pitch event
  (section 3.2). F3 (crank speed g as an input) was never tried.
- DOWN at 0.30 sits on the ILC envelope edge: some runs learn it (0.30), others do not (0.24).
- `LiveCommand.set_command` clips commanded speeds to the post-ILC envelope stored in the policy.

### 5.5 Sensor noise: the recommended P variant (r18_s3, seeds 0 / 1 / 2)

| | P x1 | P x3 |
|---|---|---|
| survival | 100 / 100 / 100% | 96.1 / 97.7 / 98.4% |
| follow | 96.2 / 95.7 / 96.8% | 85.5% (seed 0) |
| \|along\| p95 (m/s) | 0.033 / 0.034 / 0.031 | 0.040 (seed 0) |
| hover hold p95 (cm) | 3.3 / 3.7 / 3.6 | 7.5 (seed 0) |

P x1, seed 0, per command (pass rate / \|along\| p95 / cross p95):

| command | pass | \|along\| p95 | cross p95 |
|---|---|---|---|
| FWD | 100% | 0.030 | 0.040 |
| REV | 93.0% | 0.032 | 0.058 |
| UP | 98.7% | 0.021 | 0.030 |
| DOWN | 91.9% | 0.075 | 0.048 |
| HOVER | 98.6% | - | - |

DOWN's per-command pass rate is 90.1% on seed 1.

Observer: LQG on pos,att,w from the x64 sensor model `opt/runs_gpu/ref/lqr_sensor.npz` (`hover.reduced.design`):

- att_xy bias states, bias walk 9e-4 per beat (3x `hover.reduced.BIAS_WALK`);
- process noise x3; designed at noise x1; 5 ms mocap latency;
- closed-loop rho 0.949;
- cz_seq on.

### 5.6 Input-subset ladder (NOTES 22)

All rows fly r18's policy: 128 random V_op schedules x 120 beats, float32, dt 0.13 ms, scored from true states.
"Non-IMU" counts the mocap channels read.

| inputs (non-IMU channels) | run | noise | survival | follow | \|along\| median / p95 | cross p95 | hold p95 / hover drift p95 | position drift 60-120 median / p95 | tilt-dev p95 / worst survivor | saturation |
|---|---|---|---|---|---|---|---|---|---|---|
| true state (12) | r19 eval_120 | 0 | 100% | 98.7% | 0.0033 / 0.020 | 0.027 | 0.63 / 0.59 cm | - | 5.0 / 9.1 deg | 1.2% |
| pos,att,vel,w (6) | s4_full | x1 | 100% | 95.2% | 0.0088 / 0.038 | 0.043 | 3.9 / 3.9 cm | 4.9 / 13.5 cm | 6.8 / 10.1 deg | 1.0% |
| pos,att,vel,w (6) | s4_full | x2 | 96.9% | 88.2% | 0.0086 / 0.050 | 0.051 | 5.4 / 5.8 cm | 5.3 / 14.8 cm | 8.1 / 13.6 deg | 2.7% |
| **P: pos,att,w (3)** | s4_x2 | x1 | **100%** | **96.2%** | 0.0070 / 0.034 | 0.042 | 3.3 / 3.6 cm | 5.0 / 12.1 cm | 6.4 / 9.7 deg | 0.5% |
| P: pos,att,w (3) | s4_x2 | x2 | 97.7% | 90.1% | 0.0079 / 0.039 | 0.047 | 5.1 / 5.3 cm | 5.1 / 14.3 cm | 7.8 / 11.3 deg | 1.3% |
| P: pos,att,w (3) | r18_s3 | x3 | 96.1% | 85.5% | 0.0083 / 0.040 | 0.052 | 7.5 / 7.1 cm | 5.4 / 14.7 cm | 8.3 / 12.5 deg | 1.7% |
| N: att,w (0) | s4_x2 | x1 | 100% | 47.0% | 0.041 / 0.195 | 1.40 | 674 cm / 9.1 cm | 26.5 / 571 cm | 33.4 / 25.1 deg | 18.8% |
| N: att,w (0) | s4_x2 | x2 | 100% | 43.8% | 0.039 / 0.204 | 1.40 | 659 cm / 14.5 cm | 30.0 / 565 cm | 33.3 / 25.4 deg | 18.8% |
| att,w,acc + acc bias, velocity servo (0) | s4_acc | x1 | 0% | 0% | 0.31 / 0.58 | 1.18 | - | - | 98 deg p95 | 40% |
| att,w,acc + acc bias, velocity servo (0) | s4_acc | x2 | 0% | 0% | 0.25 / 0.43 | 0.97 | - | - | 100 deg p95 | 52% |
| att,w,acc + acc and att_xy bias, design x3 (0) | s4_acc2 | x1 | 28.9% | 4.0% | 0.14 / 0.41 | 0.57 | 347 cm | 95 / 218 cm | 102 / 34.1 deg | 13% |
| att,w,acc + acc and att_xy bias, design x3 (0) | s4_acc2 | x2 | 27.3% | 3.2% | 0.12 / 0.42 | 0.60 | 352 cm | - | 104 / 37.0 deg | 19% |

Pass rate per command at x1 (HOVER / FWD / REV / UP / DOWN):

| inputs | pass rates |
|---|---|
| true state | 100 / 100 / 100 / 100 / 94.0 |
| pos,att,vel,w | 96 / 97 / 93 / 97 / 93 |
| P | 97 / 100 / 93 / 99 / 93 (x2: 84 / 94 / 92 / 96 / 85) |
| N (att,w) | 17 / 0 / 96 / 49 / 67 |
| acc variants | 0 on every command, except s4_acc2 DOWN 17 |

**Recommendation: P.** It is the smallest input set that follows all five commands.

- **Mocap velocity** (6 channels) gives a better velocity estimate (RMS 0.037 vs 0.055 m/s) but no better
  flight: follow 95.2 vs 96.2%, x2 survival 96.9 vs 97.7%.
- **IMU-only att,w** (Prompt 1's `N_lqgV1` with its own gain; tables from the sensor ILC + velocity ILC)
  survives 100%. But:
  - FWD saturates and slides sideways at 1.4 m/s;
  - REV follows (96%);
  - UP and DOWN go the right way but slowly;
  - position is unobservable (hover hold p95 6.7 m).
- **Accelerometer dead reckoning** is lost. The observer's closed-loop rho is 0.994: the accelerometer bias is
  barely detectable. This matches `2026-09-26-hover-imu-only-walks`.

GPU time of one `sensors` stage (warm, untimed, on top of the 208 s learn) is 222-247 s:

- 2 cache loads, about 37 s in total;
- 2 sensor-ILC passes of 512 x 30, 66-75 s (only N uses them; P discards its sensor tables);
- the 512 x 120 evaluation, 113-122 s (2 variants x 2 noise levels x 128).

## 6. Caveats and known issues

- **DOWN's pass rate is the reference's own ceiling, not the controller's.**
  - The reference holds a descent until the horizontal stop plus 8 beats, then ramps over 8 beats. The metric
    grades the second half of a short DOWN segment (30-32 beats, after FWD or REV) against the *commanded*
    speed while the reference is still shaped.
  - Grading the reference itself through `segment_metrics` (128 schedules, learn defaults) gives DOWN 94.2% at
    120 beats and 92.8% at 240. Every other command scores 100%.
  - r19's true-state DOWN is 94.0% / 93.2%: exactly that ceiling.
  - So "DOWN is the weakest command" measures the reference design.
  - A fix: grade after v_ref reaches the command, or against v_ref / p_ref. Not done.
- **The learn stage is not bit-reproducible.** Float32 device ILC and non-deterministic GPU reductions make
  each run a different draw.
  - A fresh run of the exact r19 command (`xc_learn_full`) failed G6 (saturated settle beats 5.01% > 5%; DOWN
    segments 24.5% saturated vs 5.8%) and G7 DOWN (0.24 < 0.30). Everything else passed.
  - A `--continue-to 0 --no-verify` rerun failed G8 at eval_120 (0.538 vs 0.43 / 0.44 for r18 / r19), with
    DOWN at 91.1%.
  - r18 and r19 tables differ by up to 0.008 in u at the V_op cells and up to 0.23 at the extrapolated
    high-speed cells.
  - Earlier replicates spread about +-1% in survival.
  - **r18 is one passing draw.** The learn gates are statistics over runs, not a property of the method.
- **The budget covers learn + eval-120 only.** 205.9 s warm (r18). The graded gates are on eval_240 (+118.6 s)
  and the envelope verify (+37.8 s): about 364 s to the gate table, and about 590 s including the sensor stage.
  eval_120 alone also passes G1-G8 for r18/r19 (G8 0.43 / 0.44).
- **The cold cache is 242.9 s, just over the 4 minutes.** The run's own log prints FAIL. The cold first calls
  are 21.7 s and 24.0 s against the 10 s first-call gate. NOTES 18 and the round-2 report listed cold as PASS;
  that was wrong.
- **The G11 float32 state-error sub-gate fails:** 0.56-2.95 of ERR_SCALE on V_op flights, 85.5 on a tumbling
  baseline flight. The x64 re-run matches the CPU to 1.6e-7, so under F7 this is float32 drift and the CPU
  verdicts (which agree 100%) are trusted. "G11 PASS" in earlier summaries meant the other four sub-gates.
- **The IMU-only gate was too weak.** The recorded `sensors.json` runs (r18_s3, s4_*) gated N on
  `direction_follow`, which only checks the sign along the command. It passed N at 99.7% while FWD slid
  sideways at 1.4 m/s and HOVER passed 5-17%. `cmd_sensors` now gates N on the same follow rate as P (>= 90%).
  **Under that gate N fails (44-52%).** The stored JSON files predate the change.
- **P at higher noise misses the 99% level** that G12 applies at x1: x2 97.7%, x3 96.1-98.4%. P x3 follow
  (85-87%) is not gated. P's position drift over beats 60-120 is about 12 cm p95. The "3.7 cm" quoted for P
  in round 3 is the hover-segment drift.
- **The velocity channel has no latency.** In `hover.sensors` / `hover.mjx_sensors` only `pos` carries the
  5 ms mocap latency; `vel` is the lag-free true velocity plus noise. The pos,att,vel,w row therefore uses an
  idealised velocity sensor, and "mocap velocity does not help" holds even so.
- **No floor.** Evaluation is free air (contacts off). The scene's floor is 0.25 m below the release COM, and
  descending sequences pass through it. With contacts on, the CPU crashes those flights into it.
  - The viewer disables contacts and moves the floor plane 100 m down for command flight.
  - The free bird has no self-contacts to lose.
- **Yaw is held at 0, and "forward" is the release heading** (body +x projected). The sign follows the repo
  convention and was not checked on the head mesh, which is missing in this checkout. There are no turn
  commands.
- **Survival is counted slightly generously**, as in `mjx_flight`. `lost_at` returns N both for a full flight
  and for a loss at the final strobe. The final strobe's loss rule checks finiteness only, not tilt > 60 deg.
- **`hover_cmd.py all` launches `cpu` without `--tg` / `--control`.** A non-default bird would be compared
  against the default free-hover bird. Not fixed.
- **The committed `designs/hummer/free-hover/lqr.npz` is stale** against the current code (A 15.6%, B 8.4%;
  NOTES 5). The command pipeline linearizes fresh on every `learn`. The host tests still use the stale file
  (only as a plausible model).
- **The policy is valid for this bird and this physics only.** Like the hover policy, any change to the free
  bird or its physics requires a new `learn` (and `export`).
- **The x64 device sensor test is limited.** It compares `CmdSensorPipeline` with `mjx_sensors` only at zero
  command, with a constant cz_seq. It is skipped when the gitignored `opt/runs_imu/noise/ctl/P_lqgV1.npz` is
  missing. The non-zero-reference, cz_seq path is covered on the host (LiveCommand == CmdLinearController to
  1e-12) and by the CPU cross-check of P, not by an x64 device test.

## 7. Dead ends (measured)

| tried | result | why / note |
|---|---|---|
| spec's float32 device linearization | envelope 0.10-0.16 m/s; V_op survival 0% | A 17%, B 12% off, systematically; replaced by x64 |
| model solver (100 iterations) in flights | one 30-beat ILC call 296 s (B 1024) / 130 s (B 512) | tumbling flights stall the vmapped batch; a 35 deg loss tilt did not help; iterations 8 used |
| 4 continuation ILC passes (r2 / r3) | good cells 24 -> 19 -> 19 -> 22; V_op survival 0% | references from drifting flights made cells worse |
| GPU gain sweep, 64 weight sets on the LP tables | nothing passes FWD 0.3 or DOWN 0.32 | the feedforward was the problem, not the gains |
| pos,att-only LQG so the servo never sees strobed w / vel (CPU) | FWD 0.3 fails (yaw 25 deg), DOWN 0.2 lost | - |
| map tie-break by the sum of V_max | picked 15 deg; V_op survival 74.2% at 120 beats | reverted to "larger min V_max, tie -> 5 deg" |
| Prompt 1 observers unchanged (P_lqgV1) at speed | P 5.5% / 43.8% / 21.1% survival (r5 / final / r7) | observer model mismatch at moving trims (diagnosis 3) |
| input allocation: box projection in the Q-function metric (r8b) | survival 97.7%, FWD 69.9% | the hover model's B null space is not null at speed; `--alloc-iters 0` |
| table reprojection to the smallest equivalent input (r8c) | survival 3.9% | same reason; code removed |
| more aggressive flight gains on fixed tables | q_pos 0.01: 36%; q_int 300: 52%; q_vel 0.05: 58%; r 0.035: 72% | tables learned under the base gain |
| learning with q_int 30 (r9) | survival 79.9% | the integral has not settled inside the 15-30-beat window |
| faster P bias walk 3e-3 / 1e-2 | P survival 35.9% / 1.6% | - |
| local-model regression from dither (spec's diagnostic, F2 source) | condition number 300-700, A error 0.7-2.5 | too few samples per cell; unused |
| learning from saturated cells (r12) | drift 5.1 cm; yaw reference 9.2 deg | saturated flights write garbage references |
| last ILC pass with the flight gain (r13) | survival 95.3%, DOWN 77% | - |
| `--good-sat 0.8` (r16) | saturation 14.1%, max p95 25.1 cm | lets a non-tracking FWD 0.30 in |
| ff_lead in learning (r17) | FWD 85.3% | moved the learned entries the extrapolation uses |
| ff_lead 2 / 3 in flight | FWD 85% / survival 64% | - |
| stop_rate 0.67 / 0.5 (slower stops) | drift 1.65 / 2.02 cm, worse than gate 8 (0.82) | - |
| descend_delay 16 | DOWN follow 60% | too late |
| sensor ILC measured over beats 40-60 of 60 | P survival 47.7% | - |
| P observer on pos,att_z,w (no roll / pitch filter channel) | P survival 1.6% | - |
| P velocity ILC | P x1 93.8%, follow 78.2% | - |
| P's own LQR gain / position latency 0 | 96.1% / 91.4% at x1 | - |
| P design noise x3 (with cz_seq) | x3 survival 82.0% | best P before cz_seq (97.7% x1) |
| P on its sensor-ILC tables with cz_seq | x1 survival 85.2% | the true-state tables are consistent with cz_seq |
| P with q_int 10 / 30 / the flight gain at process x1 | 95-96% survival, 86-89% follow | - |
| offline process-noise sweep | did not predict the device gain (vel RMS ~0.03 m/s for every scale x1-x30) | only the device showed it |
| N att,w with zeroed velocity columns | follow 44.6 vs 55.4% | - |
| IMU + accelerometer with bias states (+ velocity servo) | lost at hover even noise-free on CPU; device 0% / 28.9% | acc bias barely detectable (rho 0.994) |
| mocap velocity added to P | no better (95.2 vs 96.2% follow) | - |
| **not tried** | F3 crank speed g as an input; F4 moving-cradle trims; CEM over servo knobs; DAgger MLP | the envelope and gates were met without them (DOWN at 0.30 is marginal: F3 is the lead if a faster descent is needed) |

## 8. Reproduce

On a GPU box (the `gpu-box` skill: sync the private repo, never `git clone`; CUDA jax; `.venv/bin/python`, never
plain `uv run`). Jobs run detached under the GPU lock:

```bash
R=/workspace/DiffSimAvesBot; cd $R
export LD_LIBRARY_PATH=$(find $R/.venv/lib/python3.*/site-packages/nvidia -path '*/lib' -type d | paste -sd:)
export JAX_COMPILATION_CACHE_DIR=/workspace/jaxcache_final           # persistent XLA cache (warm = 2nd run)
P=".venv/bin/python opt/hover_cmd.py"
# the budgeted run (all defaults): release, x64 linearize, design, 2 ILC passes, eval 120 -> policy.npz;
# then untimed: continue to 240 beats, post-ILC envelope verify, gates
setsid nohup sh -c "flock /workspace/gpu.lock $P learn --out opt/runs_cmd/r18 > opt/runs_cmd/run_r18.log 2>&1; \
  echo \$? > opt/runs_cmd/run_r18.done" >/dev/null 2>&1 </dev/null &
# follow-ups (each detached the same way)
$P dt --run opt/runs_cmd/r18                        # dt 0.10 / 0.16, 64 flights x 240 beats  (-> dt.json)
mkdir -p opt/runs_cmd/r18_s3 && cp opt/runs_cmd/r18/policy.npz opt/runs_cmd/r18_s3/
$P sensors --run opt/runs_cmd/r18_s3 [--seed 1|2] [--noise-scales 1 2]   # P and N (-> sensors.json)
JAX_PLATFORMS=cpu $P cpu --run opt/runs_cmd/r18    # CPU reference, 16 flights x 240 beats, no GPU lock (-> cpu.json)
.venv/bin/python opt/runs_cmd/scratch/f7b.py r18 120   # x64 device re-run of the CPU flights (F7)
```

Notes:

- `sensors` and `export` read gitignored inputs: `opt/runs_gpu/ref/lqr_sensor.npz` (the x64 CPU sensor model
  from `PYTHONPATH=src uv run python -m hover.lqr --out ...`) and `opt/runs_imu/noise/ctl/N_lqgV1.npz` (Prompt
  1's att,w LQG, from `hover.reduced`).
- `learn --ref-lqr` (default `opt/runs_gpu/ref/lqr.npz`) is only a comparison.
- The earlier stages' chains are `opt/runs_cmd/scratch/chain*.sh`.

Locally:

```bash
uv run python opt/hover_cmd.py export --run opt/runs_cmd/r18 --out designs/hummer/free-hover/command.tg
uv run pytest -q tests/test_hover_command.py        # host tests + one x64 CPU-JAX device test (~6 min)
uv run python -m tests.golden check --quick         # compare with the pre-change capture (wall times stripped)
# viewer (macOS: DYLD_LIBRARY_PATH=... uv run mjpython, AGENTS.md Gotchas; Linux: uv run python)
uv run mjpython src/view_free_hover.py --control designs/hummer/free-hover/command.tg [--sensors] [--noise-scale 1]
uv run python plot_fixed_frequency_gui.py           # Viewer tab -> launch on command.tg -> "Command flight" box
```

## 9. Files added and changed

New (method):

| file | what |
|---|---|
| `src/hover/command.py` | host side. Frames (`forward_axis`, `command_dirs`); LP tables (`steady_state`, `steady_state_tables`, `lookup`); gains (`servo_gains`, `servo_gains_vel`, opt-in `servo_hessian` / `allocate`); schedules and references (`random_schedules`, `build_refs`, `integral_keep`, `integral_gate`); `servo_law`, `CommandPolicy` (CPU true state), `trim_consistent_cz`, `CmdLinearController` (CPU LQG); live commands (`RefStream`, `LiveCommand`); `linear_sim`; metrics (`segment_metrics`, `summarize`); ILC (`ilc_update`, `extrapolate_tables`, `cell_pass`, `envelope`, `trim_sensitivity`, `velocity_ilc`, `local_models`); `save_policy` / `load_policy` |
| `src/hover/mjx_command.py` | device side. `law`, `fly_cmd` (`mjx_flight.fly`'s scan starting from a given carry), `batch_cmd_flyer`, `Chain` (chained B x 30 calls, compile / cache-load timing), `release_carry`; sensors: `pad_observer`, `CmdSensorPipeline` (per-flight `cz_seq`), `fly_cmd_sensor`, `batch_cmd_sensor_flyer` |
| `opt/hover_cmd.py` | CLI: `learn` / `dt` / `sensors` / `cpu` / `all` / `export`; gates; timings |
| `tests/test_hover_command.py` | 10 tests: LP tables, servo stability, linear-model follow, allocation KKT, CPU policy == plain LQR at zero command, `trim_consistent_cz`, climb shaping, x64 device == `mjx_flight` / `mjx_sensors` (incl. keep / gate columns and the 1+1 chain), `RefStream` == `build_refs`, `LiveCommand` == the evaluated controllers |
| `designs/hummer/free-hover/command.tg`, `command_policy.npz`, `command_observer.npz` | the exported r18 policy + P observer |

Changed (existing modules, for export and live flight):

| file | change |
|---|---|
| `src/hover/policy.py` | `command_loop` / `command_observer` keys, `FlightPolicy.command` / `.observer`, `has_command_loop`, `write_flight_control(command_loop=, command_observer=)` |
| `src/runtime/control_signal.py` | the two keys added to `FLIGHT_KEYS` |
| `src/view_free_hover.py` | live commands (keys, `--stdin`), `--sensors` / `--noise-scale`, contacts off + floor moved for command flight, on-screen command / velocity / position-error readout |
| `plot_fixed_frequency_gui.py` | the Viewer tab's "Command flight" box; launches the viewer with `--stdin` (and `--sensors`) for a `command_loop` control |

Local only (gitignored): `opt/runs_cmd/` (NOTES.md, every run dir, logs, `scratch/` analysis scripts,
`xcheck/`).

Hygiene, checked while writing this report:

- The 9 host tests of `tests/test_hover_command.py` pass (`-k "not device"`, 9 s). The x64 device test last
  passed in the verifier's run (11 passed with `tests/test_hover_mjx.py`, 409.9 s). That run was before the two
  live-command tests were added.
- `tests.golden check --quick` after today's changes to existing modules gives 9/32 passed. Its output is
  identical to `opt/runs_cmd/golden_quick_before.txt` once wall times and trailing whitespace are stripped.
- Build spec G13 said "no existing module changed". That held through verification. The export / live-command
  work then changed the four modules above, without moving the golden output.

Related findings: `2026-09-26-mjx-free-flight-parity` (device vs CPU),
`2026-09-26-hover-minimal-inputs` and `2026-09-26-hover-imu-only-walks` (the Prompt 1 sensor results these extend),
`2026-09-24-passive-pitch-bistable` (the descent event), `2026-09-26-hover-acc-reads-push-impulse` (why acc
observers are designed at noise >= 1), `2026-09-24-bptt-diverges-past-one-wingbeat` (why no gradients).
