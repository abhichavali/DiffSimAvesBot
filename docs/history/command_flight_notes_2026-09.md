<!-- Frozen copy (2026-09-30) of opt/runs_cmd/NOTES.md, the Prompt 2 lab notes. Do not update; the current state is docs/findings/2026-09-30-command-following-policy.md. The run directories it cites (opt/runs_cmd/*) are gitignored and not in the repo. -->

# Prompt 2: command-following closed-loop policy (running notes)

Spec: servo (hover A,B from device FD linearize + LP steady-state feedforward tables + LQR with position
integral) + ILC pass; code in src/hover/command.py, src/hover/mjx_command.py, opt/hover_cmd.py,
tests/test_hover_command.py. Box: RTX 5070 Ti (vast.ai, ssh -p 41308 root@115.73.216.179).

## 0. Hygiene baseline
- `uv run python -m tests.golden check --quick` before any change: opt/runs_cmd/golden_quick_before.txt
  (9/32 passed, exit 1, as documented). The file contains wall times ("17.9s", "in 36s"), so the
  before/after comparison strips the timing columns (sed) before diffing.

## 1. Bench (box, `opt/hover_gpu.py bench --batch 512 1024`, 4 beats, dt 0.13, n_step 901)
- B=512: 0.838 ms/step (first call 32.1 s); B=1024: 1.037 ms/step (first call 33.0 s).
  Faster than bench_5070.json (1.21 / 1.38): different box / driver.
- Choice of B: 30*901*1.037e-3 + 120*901*1.037e-3 + 25 = 28.0 + 112.1 + 25 = 165 s <= 220 s -> B = 1024.

## 2. Code (new files only)
- src/hover/command.py (host: LP tables, servo gains, schedules, refs, CommandPolicy / CmdLinearController,
  linear sim, metrics, ILC, envelope, local-model regression), src/hover/mjx_command.py (device: fly_cmd,
  Chain of B x 30 calls, CmdSensorPipeline + padded observer), opt/hover_cmd.py (learn / dt / sensors / cpu / all),
  tests/test_hover_command.py.
- Forward axis: the head mesh (rufous_head.obj) is not in this checkout (BIRD_ASSETS_DIR sibling missing), so
  the sign could not be checked on the mesh. Used the repo convention: world +x = forward (hover.evaluate's
  VERIFY_PUSHES comment "0.2 m/s forward" = +x; the visual torso sits at +x 0.05 of the airframe). e_f from the
  released q_ref = [0.9999, -0.0119, 0]. Not flipped.
- Spec arithmetic: the cell layout is 2 maps x (hover + 4 x 15) = 122 cells, not 128.
- Deviation (documented): ILC also stores the measured strobed w and a velocity offset vector dv (strobed minus
  mean-motion, along-command component clipped to +-20% = r_v in [0.8, 1.2]) per cell, so the ILC fixed point
  has zero feedback on att, vel AND w (with w_ref = 0 the w feedback double counts).
- x64 CPU-JAX parity (tests/test_hover_command.py, 2 beats): fly_cmd vs mjx_flight.fly 1.2e-11, 1+1 chain vs
  2-beat call 0.0, padded CmdSensorPipeline vs mjx_sensors P controller 1.2e-13. Host tests pass (LP residual
  <= 1e-9, |u_ff(0)| 0.0043, servo rho 0.804 on designs/.../lqr.npz). The chain needed rollout.match_dtypes on
  the scan carry (x64 promotes MJX int32 contact fields).

## 3. First GPU runs (dead ends first)
- `learn --out opt/runs_cmd/cold` (cold cache, spec defaults): linearize 67x1 compile 19.8 s (incl. run);
  device A,B vs lqr.npz: A err 0.22, B err 0.13 (float32 FD), hover LQR closed |eig| 0.673 (ref 0.656), servo
  0.804. LP envelope (t = u_max/2 / u_max): 5 deg FWD 0.36/0.72 REV 0.38/0.75 UP 0.94/1.73 DOWN 0.97/1.77;
  15 deg: same horizontal, UP 1.22/2.04, DOWN 1.25/2.07 m/s. Flight program 1024x30 compile 19.3 s.
  ILC pass: ONE 30-beat call took 296 s (10x the 30 s of hover flights): the pre-ILC envelope was
  FWD/REV 0.10, UP/DOWN 0.16 m/s (both maps), 60 of 122 cells lost. Killed before eval.
- Speed check (opt/runs_cmd/scratch/speed.py, B 1024 x 30): hover 30.6 s (1.13 ms/step), constant FWD 0.3
  31.8 s, FWD 0.15 31.1 s: the 10x slowdown comes from TUMBLING flights (the 1.2 m/s / 0.75 m/s cells), whose
  solver while-loop stalls the whole vmapped batch. Fix: per-flight traced loss tilt (fly_cmd lost_tilt), 60 deg
  in the evaluation (the gate rule), lower in the ILC passes.
- CPU (hover.launch.fly + CommandPolicy, lqr.npz model, LP tables, spec gains): FWD 0.15 drifts in yaw / roll
  and saturates by beat 24; FWD 0.3 saturates from beat 9 (att 22 deg at beat 30); UP 0.16 fine (mean motion
  0.16, strobed vel 0.19: r_v ~1.2); hover fine. Strobed w offsets at speed (0.3-1 rad/s at FWD 0.15-0.3,
  while the attitude changes only ~0.03 rad/s) are intra-wingbeat phase artifacts the w feedback fights.
- CPU continuation-ILC probe: after ILC at FWD/REV 0.05-0.1, UP 0.08-0.16: REV 0.3 and UP 0.32 pass,
  FWD 0.2 / 0.3 / 0.4 and DOWN 0.16 / 0.32 fail (saturation), UP 0.48 lost.
- GPU gain sweep (opt/runs_cmd/scratch/sweep_gains.py, 64 weight sets x 16 cells, LP tables, no ILC; 242 s
  because of tumbling flights): nothing passes FWD 0.3 or DOWN 0.32. Best: q_w 2, q_att 0.1, q_vel 0.1,
  r 0.05, q_int 25 (rho 0.903): 9/16 cells (FWD <= 0.2, REV <= 0.3, UP <= 0.48, DOWN 0.16); its FWD 0.3 is
  alive with along -0.027 m/s but 53% saturated; DOWN 0.32 along -0.29 (does not descend), 100% saturated.

## 4. Solver knob (measured, opt/runs_cmd/scratch/probe_solver.py; B 512 x 30, cells hover / FWD 0.3 / UP 0.64 /
   DOWN 0.32, 128 flights each, same inputs)
- model solver iterations (100, tol 1e-8): 69.2 s; tol 1e-6: 68.2 s; iterations 8: 27.3 s; iterations 4: 25.9 s.
- State difference to the default run (max |ds| / ERR_SCALE): hover 0.26 @10 beats / 0.63-0.69 @30 for EVERY
  variant including the tolerance-only change, i.e. the float32 GPU run-to-run noise floor; FWD 0.3 0.39 / 1.2
  likewise. So iterations 8 changes nothing measurable in flights that fly, and stops a tumbling flight's solver
  while-loop from stalling the whole vmapped batch (2.5x on this batch, ~5x on the ILC pass). Used
  --iterations 8 for the command pipeline (the never-1 rule stands; 8 is 8x above it). The GPU is also not
  bitwise deterministic: two identical learns gave slightly different device A,B (LP envelope UP 1.73 vs 1.77).
- r2 (`learn`, default solver, B 512, q_w 2 q_att 0.1 q_int 25, 35 deg pass tilt, 4 continuation passes):
  each 30-beat pass call 130 s (the 35 deg loss tilt did not fix the stall); pre-ILC envelope FWD 0.10, REV 0.15,
  UP 0.16, DOWN 0.08 (both maps); after continuation pass 2 the good cells FELL from 26 to 19 (FWD good <= 0.1,
  REV <= 0.2, UP <= 0.16, DOWN <= 0.08): the ILC refs (att / w := window means of drifting flights) made cells
  worse. Killed.
- r3 (`learn --iterations 8 --ilc-iters 4 --continue-to 0 --no-verify`, B 512): pass calls 35 s each (vs 130 s),
  eval calls 30-32 s; warm-ish budget wall 303 s (4 passes + 4 eval calls: too many passes for 240 s).
  Continuation ILC did not converge: good cells 24 -> 19 -> 19 -> 22 of 122; last-pass envelope FWD 0.10,
  REV 0.15, UP 0.16, DOWN 0.16 (15 deg map chosen). Eval at V_op 0.3: V_op servo survival 0.0% (384 flights),
  follow 7%; the uniform arm (U[0.1, <= 0.16]) 85.9% survival, follow 89.8%, |along| median 0.0016;
  baseline LQR 23.4% survival. Hover segments hold well (p95 0.3 cm) while alive.
- r3 pass-1 traces (LP tables, no ILC): FWD 0.2 pitches the WRONG way (ref +0.9 deg, actual -3 -> -26 deg over
  30 beats, 21/24 inputs saturated by beat 30); DOWN 0.16 pitches +10 deg (ref +1.8); UP 0.32 pitches -22
  (ref -4.7) and is lost at beat 23; REV 0.3 rolls +-5-8 deg with 2-8 saturated inputs. The strobed w is 4-7x
  the mean rotation rate (att difference / T) in these flights.

## 5. The linear model was the problem (measured)
- designs/hummer/free-hover/lqr.npz is STALE: vs a fresh CPU x64 hover.lqr (opt/runs_gpu/ref/lqr.npz, same code as
  now) A differs 15.6%, B 8.4%, drift 0.0026. (Not fixed here: committed design; noted for a doc edit.)
- Device float32 FD linearization vs the CPU x64 reference (opt/runs_cmd/scratch/probe_lin.py): A 17-18%,
  B 12%, velocity columns 33-34%, drift 0.006, REPEATABLE (4 reps within 1%; iterations 8 vs default identical),
  so a systematic float32 effect, not noise. Larger steps: x5 A 16% / B 22%; x20 A 20% / B 56% (worse).
  Device x64: exact (0.000) and 1.0 s per 67-flight call on the 5070 Ti (21.8 s first call cold).
- Which model flies the float32 device flights (opt/runs_cmd/scratch/sweep_models.py, LP tables, no ILC,
  21 cells x 12 configs, 30 beats; P = passes, s = alive but fails): CPU x64 model 8-10 passes (FWD <= 0.2,
  REV <= 0.3-0.4, UP <= 0.32, DOWN 0.16), device f32 model 4-5 (FWD 0.1, REV 0.1, UP 0.16, DOWN 0.16), stale
  designs model 8-10. -> learn now linearizes in x64 (switches JAX to float64 for that one call).
- CPU reference (hover.launch.fly + CommandPolicy, x64 model, LP tables, q_w 2 / q_att 0.1 / q_int 25):
  FWD 0.2 / 0.3, REV 0.3, UP 0.3 pass (|along| <= 0.016 m/s); DOWN 0.2 is lost at beat 27. Descent shows a
  sudden event: DOWN 0.08 flies steady for 24 beats then vel_z jumps -0.10 -> +0.10 m/s in 4 beats; DOWN 0.16
  and 0.24 pitch up abruptly (4 -> 14 -> 29 deg) at beats 16-20. Hypothesis (not verified): the passive wing
  pitch flips between its two lift modes (AGENTS.md "Passive pitch is bistable") in descent.
- A pos,att-only LQG observer (so the servo never sees the strobed w / vel offsets) did not help on CPU: FWD 0.3
  fails (yaw 25 deg), DOWN 0.2 lost.

## 6. r4: x64 model (defaults: hover.lqr weights, q_int 100, iterations 8, 2 continuation passes, B 512, rate 4)
- Warm-ish budget wall 232.7 s (PASS <= 240): release 3.4, linearize x64 6.8 (compile/cache 5.7), 2 passes
  36-37 s each, eval 4 calls 28.6-39.8 s (143 s), metrics 0.5 s. First calls: linearize 5.7 s, flight program
  5.1 s (cache hits), 2 shapes on the budget path.
- Envelope (30-beat constant commands; V_max per axis, tilt 5 / 15 deg): pre-ILC FWD 0.25/0.25, REV 0.35/0.4,
  UP 0.32/0.32, DOWN 0.24/0.24; post-ILC (verify pass) FWD 0.25/0.25, REV 0.35/0.40, UP 0.40/0.48,
  DOWN 0.24/0.24. LP prediction (t = u_max/2 | u_max): FWD 0.37|0.74-0.75, REV 0.39|0.75-0.77,
  UP 1.04-1.30|1.92-2.26, DOWN 1.07-1.31|1.95-2.29 m/s.
- Eval at V_op 0.3 (15 deg map): 120 beats: V_op servo survival 64.8%, follow 67.5%; uniform arm 90.6% / 94.5%;
  baseline LQR 17.2% / 36.6%. 240 beats: V_op 37.0% survival (no-push 36.8%, push 37.5%), follow 53.3%
  (HOVER 65%, FWD 49%, REV 63%, UP 70%, DOWN 21%), |along| median 0.0053 p95 0.038, cross p95 0.099, hover hold
  p95 3.6 cm, rise median 4 beats, settle saturation 18.5% (FWD 70%!). Baseline 4.7% survival.
- Losses (242 of 384 V_op flights by beat 240): command active at the loss DOWN 124, REV 61 (50 of them right
  after FWD -> REV), HOVER 30, UP 22, FWD 5; median 16 beats into the segment.
- Descent event (CPU, hover.launch.fly, logging the passive wing-pitch joint extremes per wingbeat): DOWN 0.16
  descends steadily (strobed vz -0.22) for 14 beats, then in 2 beats vz jumps to +0.08 while the wing-pitch
  extremes change (L max 12.4 -> 11.8 deg, R min -8.55 -> -7.3 deg); the pitch attitude then diverges
  (3 -> 13 -> 56 deg). DOWN 0.24: same at beat 12. Hover: extremes constant (11.6-11.7 / -7.9 deg) for 40 beats.
  -> consistent with the passive-pitch mode flip (bistable pitch, AGENTS.md gotcha) triggered by descent.

## 7. r5: + slower references (--rate-beats 8: V_op reached in 8 beats instead of 4)
- Budget wall 221.2 s (PASS). 5 deg map chosen. Post-ILC envelope (5 / 15 deg): FWD 0.30/0.25, REV 0.40/0.45,
  UP 0.40/0.40, DOWN 0.24/0.24 -> G7 passes FWD / REV / UP, FAILS DOWN (0.24).
- Eval V_op 0.3: 120 beats survival 88.3%, follow 90.9%; 240 beats survival 78.4% (no-push 77.8%, push 80.2%),
  follow 84.5% (HOVER 87%, FWD 87%, REV 91%, UP 91%, DOWN 67%), |along| median 0.0053 p95 0.024, cross p95 0.050,
  hover hold p95 3.5 cm, hover |v| median 0.005 p95 0.015, hover drift p95 3.3 cm, RMS |p - p_ref| median 3.7 cm,
  rise median 7 beats, settle saturation 5.0%. Uniform arm (U[0.1, V_max_last_pass]) 100% survival and 100%
  follow at 240 beats. Baseline LQR 4.7% survival.
- Losses: 83 of 384; 62 of them in DOWN right after FWD (FWD -> DOWN), median 20 beats into the segment. Traces:
  during/after FWD -> DOWN the yaw runs away (att_z -2 -> -34 deg within 10 beats), lateral velocity builds
  (0.2 m/s), inputs saturate (up to 17 of 24), then the bird tumbles. UP -> DOWN and HOVER -> DOWN are not lost.

## 8. r6: + DOWN operating speed 0.24 (--v-op-down 0.24: the G7 fallback, V_max DOWN = 0.24 in r4-r6)
- Eval 240 beats: V_op servo survival 96.9% (no-push 96.5%, push 97.9%), follow 95.6% (HOVER 96.5, FWD 86.0,
  REV 99.1, UP 99.8, DOWN 96.3), |along| median 0.0036 p95 0.024, cross p95 0.042, hover hold p95 2.8 cm, hover
  |v| median 0.0045 p95 0.0125, hover drift p95 2.7 cm, RMS |p - p_ref| median 2.8 cm, max p95 22 cm, rise
  median 6, tilt-dev p95 10.0 deg, max tilt of survivors 15.6 deg, settle saturation 2.7%. Post-ILC envelope
  (5 deg map): FWD 0.30, REV 0.40, UP 0.56, DOWN 0.24. Losses 12/384: 9 in DOWN after FWD.
- FWD failures (52 alive FWD segments that fail): cross-track 0.053-0.068 m/s (just over 0.05), 36% saturated.
- Budget wall 254 s here only because a 16-process CPU job ran on the box during it (eval call 1: 64 s instead of
  30 s); r5 (same path, no concurrent job) 221 s.
- `cpu` on r5 (V_op 0.3 incl. DOWN 0.3): CPU and device agree beat by beat until the flight saturates in DOWN
  0.3 (e.g. flight 2: equal to 3 decimals through beat 8, then both 100% saturated and diverging); 3 flights CPU
  lost at beat 32 while the device flew all. Not a port bug; the saturated DOWN 0.3 regime is chaotic.
- `sensors` on r5 (first run of the stage): P (pos,att,w LQG P_lqgV1 unchanged, 5 ms latency) survival 5.5% at
  x1, N (att,w,acc + acc-bias LQG, velocity servo) 3.1%. CPU, noise 0: the P observer hovers (0.7 deg max tilt)
  but at FWD 0.3 / UP 0.3 the attitude ESTIMATE departs from the truth by 5-20 deg within 12-18 beats: the
  on-board Mahony filter's beat-mean accelerometer "up" is not gravity in sustained forward / climbing flight
  (drag and thrust tilt it), so the servo holds the wrong attitude. The Prompt 1 observers were only hover-tested.

## 9. Final protocol (defaults now: B 512, iterations 8, x64 linearize, hover.lqr weights, q_int 100, rate 8 beats,
   2 ILC/continuation passes, both tilt maps; run with --v-op-down 0.24)
- final_cold (fresh cache /workspace/jaxcache_final, `learn --out opt/runs_cmd/final_cold --v-op-down 0.24
  --continue-to 0 --no-verify`): budget wall 247.5 s (shell 248.0 s): imports 0.6, release 2.8, linearize 22.2
  (x64 compile 21.1), design 0.6, 2 ILC passes 98.2 (flight-program compile 24.4, calls 36.0 / 36.5), eval
  123.1 (calls 28.6 / 30.2 / 32.3 / 31.8). Eval 120 beats: V_op survival 96.1%, follow 95.3%.
- First warm attempt (opt/runs_cmd/final_15deg_choice, log run_final_15deg_choice.log): 224.9 s, but the map
  choice (min V_max, then SUM as a tie-break, my deviation) picked 15 deg on a tie and that map flew worse (V_op
  survival 74.2% at 120 beats, saturation 14%). Reverted to the spec rule (larger min V_max, tie -> 5 deg) and
  reran; the GPU passes are not bitwise reproducible, so ties do happen.
- final (warm cache, `learn --out opt/runs_cmd/final --v-op-down 0.24`, nothing else on the box):
  BUDGET 209.0 s: imports 0.5, release 2.8, linearize 6.9 (x64 cache load 5.8), design 0.5, linear sim 0.04,
  2 ILC passes 75.8 (calls 35.8 / 34.5; flight program cache load 5.1), eval 122.4 (28.2 / 30.2 / 32.0 / 31.7),
  metrics 0.1. Untimed: continue to 240 121.2 s, envelope verify 36.4 s.
  Linear model: open |eig| 1.113, hover LQR 0.651, servo 0.804 (x64 = CPU reference exactly).
  LP envelope (t = u_max/2 | u_max), 5 deg: FWD 0.37|0.74, REV 0.39|0.75, UP 1.04|1.92, DOWN 1.07|1.95; 15 deg:
  FWD 0.37|0.75, REV 0.39|0.77, UP 1.30|2.26, DOWN 1.31|2.29. Measured (5 / 15 deg): pre-ILC FWD 0.25/0.25,
  REV 0.40/0.40, UP 0.32/0.32, DOWN 0.24/0.24; post-ILC FWD 0.30/0.25, REV 0.40/0.45, UP 0.32/0.32, DOWN
  0.24/0.24. 5 deg map chosen.
  Eval 120 beats: V_op survival 98.7% (push 96.9%), follow 98.3%. 240 beats (gates): survival 96.4% (no-push
  96.2%, push 96.9%), follow 97.2% (HOVER 96.9, FWD 95.4, REV 98.0, UP 99.3, DOWN 96.3), |along| median 0.0035
  p95 0.023, cross p95 0.030, hover hold p95 3.3 cm, hover |v| median 0.0049 p95 0.0135, hover drift p95 3.2 cm,
  RMS |p - p_ref| median 2.9 cm, max p95 13.3 cm, rise median 6, tilt-dev p95 7.9 deg, max tilt of survivors
  12.6 deg, settle saturation 1.2%. Uniform arm 95.3% / 98.1%. Baseline (hover LQR on p_ref only) 4.7% survival,
  21.3% follow; its |along| median over its FEW surviving segments 0.0030 vs the servo twin 0.0038 (ratio 1.27,
  G8 as specified fails; the baseline loses 95% of flights).
  Local-model regression diagnostic (dither 0.01, 3-4 flights per cell, 45-60 samples for 34 regressors):
  cond 300-700, A rel err 0.7-2.5, B rel err 0.36-0.9: uninformative at this sample size (not used).
- dt (`dt --run opt/runs_cmd/final`, fixed policy, the first 64 no-push V_op schedules, 240 beats):
  dt 0.10 (1171 steps/beat): survival 96.9% (62/64), follow 96.2%, |along| median 0.0036; dt 0.16 (732):
  survival 96.9%, follow 97.3%, |along| median 0.0040. Compile 31.9 s / 16.0 s, calls 18 s (B 64).
- cpu (`cpu --run opt/runs_cmd/final`, 16 flights, 240 beats, box CPU, 16 processes, 87 s): survival agrees on
  9/16 (both flew 5 + both lost 4 incl. baselines); CPU lost and device flew: flights 2, 5, 291 (all at beat 33,
  all start with DOWN) and 3 (CPU 210), 289 (CPU 231). Segment verdict agreement 75.3%, |v_seg| diff median
  0.0005 m/s. Max state error / ERR_SCALE 0-120: 0.55-5.4 on flights that fly the same, 144-178 on the split ones.
  Same flights split identically in the r5 check -> deterministic, not noise.
- sensors (`sensors --run opt/runs_cmd/final`: P = P_lqgV1 unchanged + full-state servo, N = att,w,acc LQG with
  acc-bias states (designed from opt/runs_gpu/ref/lqr_sensor.npz at noise x1, closed-loop rho 0.994) + velocity
  servo (rho 0.655); 5 ms mocap latency; 2 sensor-ILC passes of 512 x 30 constant-command sensor flights at noise
  x1 relearn the tables in estimate coordinates (38 s per pass, compile 18.8 s), then 512 x 120 (141 s, compile
  18.3 s)). Sensor-ILC envelope: P FWD 0.15, REV 0.30, UP 0.16-0.24, DOWN 0.24; N 0 everywhere.
  P (3 non-IMU channels) x1: survival 43.8%, follow 61.3%, |along| p95 0.034, hover hold p95 6.6 cm, velocity
  estimate bias [0.011, -0.009, -0.048] m/s, drift 60-120 median 8.4 cm; x3: survival 33.6%, follow 54.3%.
  N (0 non-IMU) x1 / x3: survival 0.0% / 0.0%, direction-follow 3.3% / 3.6%.
  Without the sensor ILC (r5 tables in truth coordinates) P was 5.5% and N 3.1%.

## 10. r7 = the final policy: + FWD -> DOWN reference shaping (--descend-after-stop 0.05, now the default)
- `learn --out opt/runs_cmd/r7 --v-op-down 0.24 --descend-after-stop 0.05` (warm cache, box otherwise idle):
  BUDGET 209.6 s. Post-ILC envelope (5 / 15 deg): FWD 0.30/0.25, REV 0.40/0.45, UP 0.40/0.48, DOWN 0.24/0.24.
  240 beats: V_op survival 99.5% (no-push 99.65% = 287/288, push 99.0% = 95/96), follow 96.7% (HOVER 96.5,
  FWD 87.8, REV 99.3, UP 100, DOWN 99.8), |along| median 0.0047 p95 0.025, cross p95 0.034, hover hold p95 3.0 cm,
  |v| median 0.0048 p95 0.0146, hover drift p95 3.2 cm, RMS |p - p_ref| median 2.9 cm, max p95 18.2 cm, rise
  median 7, tilt-dev p95 9.8 deg, max tilt of survivors 12.8 deg, settle saturation 2.6%.
  Run-to-run spread (GPU float32 not bitwise reproducible + different ILC outcomes): FWD follow was 95.4% in
  `final` (same code path without the shaping) and 87.8% here.
- IMU-only variants on CPU (hover.launch.fly + hover.sensors, 60 beats): att,w,acc LQG with acc-bias states +
  velocity servo is lost at hover even noise-free (beats 19-41); att,w,acc without bias states hovers but FWD 0.3
  tumbles / walks 0.9 m/s sideways; the Prompt 1 att,w LQG (N_lqgV1) with its own gain and no velocity servo
  hovers (0.5 deg) and survives FWD 0.3 but moves [0.12, 0.43, 0.04] m/s (mostly sideways). -> sensors now
  defaults to that (F6 "att, w only, no velocity servo"; --n-variant acc_bias keeps the other).

## 11. CPU vs device: it was the FLOOR (F7)
- F7 on `final` (opt/runs_cmd/scratch/f7.py, the 16 CPU-reference flights, 60 beats on the device): float32
  iterations 8, float32 iterations 100, and x64 (model iterations, ls 8, contacts off) ALL fly flights 2, 5, 291
  past beat 33, where the CPU loses them. A local CPU-JAX x64 run with the CPU's exact knobs (contacts on, model
  line search) matches hover.launch.fly to 3e-11 over the first 12 beats of flight 2 (the port is exact).
- Cause: contacts. The scene has a floor 0.25 m below the release COM (release COM z = 0.246 m); flights 2 / 5 /
  291 start with DOWN, the struts touch the floor from beat 13 (CPU mj contacts: floor vs L/R_part_*_rod_*),
  and the bird crashes at beat 32-33. The device flies with contacts off (hover_gpu's default: "the free bird
  never touches itself") and descends through the floor plane. In `final`, 212 of the 384 V_op eval flights go
  below the depth where the first strut touched (COM 0.254 m below release) at some point.
- Confirmed on the device: x64 with the CPU's exact knobs (model solver + line search, contacts ON; f7.py
  x64_exact, 60 beats, 195 s) loses flights 2, 5, 291 at beat 33 and baseline 450 at 47, exactly as the CPU
  (contacts-on reference: 32-33 and 47). The port is exact; the split was contacts.
- So the evaluation is a free-air evaluation (no floor). The CPU reference now also runs contacts off by
  default (`cpu --cpu-contacts` keeps them); r7's contacts-on CPU check is kept as cpu_contacts_on.json:
  identical survival 12/16 there (the 3 floor crashes plus flight 3, CPU lost at 224), verdict agreement 76.3%, |v_seg| diff
  median 0.0004 m/s.
- r7 `cpu` with contacts off (as the device; 16 flights x 240 beats, box CPU 16 processes, 131 s): survival
  identical 16/16 (12 V_op incl. 4 pushed fly all on both; the 4 baseline flights are lost on both at beats
  75/75, 31/29, 110/108, 69/69), segment verdict agreement 98.9% (74 segments), |v_seg cpu - gpu| median
  0.00034 m/s, p95 0.0017, max 0.0077 m/s. State error / ERR_SCALE over beats 0-120: 0.70-5.2 per flight, 13.9 on
  flight 3 (gate <= 1.0 FAILS; hover parity was 0.2-0.44: command flights amplify the float32 difference). (The
  gate line printed "nan" for the max |v_seg| and error aggregation: a bug in my aggregation, fixed after the run;
  the values above are recomputed from r7/cpu.json.)
- r7 `dt` (fixed r7 policy, 64 no-push V_op schedules x 240 beats): dt 0.10: survival 100%, follow 96.2%,
  |along| median 0.0044; dt 0.16: survival 98.4% (63/64), follow 98.9%, |along| median 0.0048.
- r7 `sensors` (N = att_w now): sensor-ILC envelope P FWD 0.15-0.20, REV 0.30, UP 0.16, DOWN 0.24; N FWD 0,
  REV 0.20-0.25, UP 0.08-0.24, DOWN 0.08. P (3 non-IMU channels) x1: survival 21.1%, follow 39.5%, |along| p95
  0.19, hover hold p95 6.3 cm, vel-estimate bias [-0.011, -0.005, -0.022] m/s, drift 60-120 median 13 cm;
  x3: survival 22.7%. N (0 non-IMU channels) x1: survival 100%, follow 3.0%, direction-follow 76.9%, hover hold
  p95 2.7 m, vel-estimate bias [0.000, 0.025, -0.059] m/s, drift 60-120 median 0.68 m; x3: survival 100%,
  direction-follow 75.2%. (P was 43.8% survival in `final`: the sensor-ILC outcome varies run to run.)
- Hygiene: `uv run pytest -q tests/test_hover_command.py tests/test_hover_mjx.py` 8 passed (271 s);
  golden --quick output identical to the pre-change capture after stripping the wall-time columns (the raw
  files differ only in "17.9s"-style timings). No existing module changed (git status: 4 new files).

## 12. GATES (final policy = r7; budget numbers from the r7 warm run and final_cold)
| gate | measured | pass |
|---|---|---|
| G1 V_op servo survival, 240 beats | 99.48% (382/384) | PASS |
| G1 no-push (target 100%) | 99.65% (287/288) | FAIL (target) |
| G1 push flights | 98.96% (95/96) | FAIL (by one flight) |
| G2 follow rate, all segments | 96.7% | PASS |
| G2 per command HOVER/FWD/REV/UP/DOWN | 96.5 / 87.8 / 99.3 / 100 / 99.8 % | FWD FAIL, rest PASS |
| G3 |along| median / p95, cross p95 | 0.0047 / 0.025 / 0.034 m/s | PASS |
| G4 hover hold p95, |v| median / p95 | 3.0 cm, 0.0048 / 0.0146 m/s | PASS |
| G4 hover settle-window drift p95 | 3.2 cm | FAIL (<= 2 cm) |
| G5 RMS median / max p95 / rise median | 2.9 cm / 18.2 cm / 7 beats | max FAIL, rest PASS |
| G6 tilt-dev p95 / max tilt survivors / settle saturation | 9.8 deg / 12.8 deg / 2.6% | PASS |
| G7 post-ILC V_max FWD/REV/UP/DOWN (5 deg map) | 0.30 / 0.40 / 0.40 / 0.24 | DOWN FAIL (V_op DOWN = 0.24) |
| G8 servo / baseline |along| median | 0.0050 / 0.0030 = 1.65 (baseline survives 4.7%, follows 21.3%) | FAIL as specified |
| G9 dt 0.10 / 0.16: survival, follow, |along| median | 100% / 96.2% / 0.0044; 98.4% / 98.9% / 0.0048 | 0.16 survival FAIL |
| G10 warm wall to eval-120 | 209.6 s | PASS |
| G10 cold wall | 247.5 s (final_cold) | reported |
| G10 warm first call per shape | linearize x64 5.7 s, flight 512x30 5.1 s | PASS |
| G10 shapes on the budget path | 2 | PASS |
| G11 CPU (contacts off): survival / verdicts / |dv| median / max | 16/16, 98.9%, 0.00034, 0.0077 m/s | PASS |
| G11 state error / ERR_SCALE 0-120 | 0.70-5.2, 13.9 max | FAIL |
| G12 P x1 survival / follow / |along| p95 / hold p95 | 21.1% / 39.5% / 0.19 / 6.3 cm | FAIL (hold PASS) |
| G12 P x3 survival | 22.7% | FAIL |
| G12 N x1 survival / direction-follow | 100% / 76.9% | survival PASS, direction FAIL |
| G13 tests, golden, no module changed | as above | PASS |

## 13. Proposed doc edits (not made: no commit in this task)
- AGENTS.md "Free flight" / free-flight-policy skill: command following lives in src/hover/command.py +
  mjx_command.py, CLI opt/hover_cmd.py; linearize in x64 on the device (float32 FD is 17% / 12% off);
  designs/hummer/free-hover/lqr.npz is stale vs the current code (A 15.6%, B 8.4%).
- AGENTS.md Gotchas: the free-flight scene has a floor 0.25 m below the release COM; the device default
  (contacts off) flies through it; descending command sequences crash into it on the CPU.
- Findings to write: command-following servo + ILC (numbers above); descent limit ~0.24 m/s and the FWD -> DOWN
  failure (yaw runaway; passive-pitch mode change seen on CPU); the Mahony attitude filter is biased by 5-20 deg
  in sustained forward / climbing flight (sensor variants fail for that reason); solver iterations 8 on the
  device (float32 noise-level change, no stall from tumbling flights).

## 14. Tuning round 1 (2026-09-29, second session)
Diagnosis from r7 (opt/runs_cmd/scratch/an_r7.py):
- FWD segments that fail do so on CROSS-track (0.051-0.084 m/s, always +y; along is fine), with 8-80% of the
  settle beats saturated. The largest |p - p_ref| (G5 max) is the same thing: 38 of the 40 worst flights peak in
  FWD, lateral component ~0.23 m. The worst hover drifts all come right after FWD (the lateral error being
  pulled back). -> saturation in FWD at 0.3 m/s, and the conditional integral freezing while saturated.
- The ILC tables are 2-6x the LP inputs (FWD 0.30: |u_ff| 0.131 vs LP 0.061; UP 0.32: 0.091 vs 0.018): the
  measured feedback efforts added by the ILC lie in range(B') (LQR gains), so the tables lose the LP's
  null-space flattening. The smallest input with the SAME model effect B u (LP, all 12 rows) is much smaller:
  FWD 0.30 0.131 -> 0.071, UP 0.48 0.135 -> 0.102, DOWN 0.32 0.108 -> 0.076, REV 0.30 0.119 -> 0.086.
- DOWN 0.32 in r7's verify pass: along 0.017, cross 0.040 (it follows) but 73% saturated: it failed the cell
  rule only on saturation (sat <= 0.2).
New mechanisms (new code only in the 4 new files):
- Input allocation (hover.command.servo_hessian / allocate, mjx_command.allocate): the servo command is projected
  onto the input box in the metric H = R + Ba' P Ba (the servo LQR's Q-function Hessian), by 30 FISTA iterations,
  instead of clipped: the one-step optimal constrained LQR; saturated channels' share moves to free ones through
  B's 12-dim null space. The integral pauses only while the delivered feedback misses the demand by > 10%
  (H-norm). The baseline arm gets H = I (= clip exactly). --alloc-iters (default 30; 0 = clip). Test: KKT
  optimality, H = I == clip, device vs CPU (hover.launch.fly + CommandPolicy) with saturating inputs, x64, 2 beats.
- Table reprojection (hover.command.min_inf_equivalent / reproject_tables): after every ILC pass each table entry
  becomes the smallest-|u|_inf input with the same B u (--no-reproject turns it off). ~1 s host time.
- SPEEDS_Z 0.06 .. 0.90 (0.30 is now a grid point; 122 cells as before). V_op DOWN back to 0.30 (default).
- Sensors: --p-obs / --p-bias / --p-design-noise design the P observer in the stage; --n-dv-clip / --n-ilc-any for
  the N sensor ILC.
Runs (warm cache /workspace/jaxcache_final, B 512, default knobs unless named; 240-beat numbers on the 384 V_op flights):
- r8a (clip, grid 0.06, V_op DOWN 0.30, descend-after-stop 0.05): budget 211.5 s. Survival 98.7% (no-push 99.3%,
  push 96.9%), follow 97.9% (HOVER 97.4, FWD 95.9, REV 98.7, UP 99.5, DOWN 98.0), |along| med 0.0054 p95 0.025,
  cross p95 0.034, hold p95 3.1 cm, |v| med 0.0051, drift p95 3.07 cm (FAIL), RMS med 3.4 cm, max p95 14.1 cm, rise 7,
  sat 4.7%. Post-ILC envelope 5 deg: FWD 0.30, REV 0.40, UP 0.36, DOWN 0.24; DOWN 0.30 cell: along 0.014, cross
  0.034, 27% saturated -> fails only the sat <= 20% cell rule. So DOWN at V_op 0.30 now FLIES in random sequences
  (98% of DOWN segments pass); the old DOWN failure was the FWD -> DOWN transition, fixed by the r7 shaping.
  Losses (5): 2 no-push (UP-FWD-DOWN at 210, FWD-DOWN-REV at 232), 3 pushed (all after FWD).
- r8b (= r8a + allocation, 30 FISTA iterations): WORSE. Survival 97.7%, follow 92.1%, FWD 69.9%, cross p95 0.060,
  sat 7.8%, max p95 21 cm. DEAD END (the model's null space is not null at speed); --alloc-iters now defaults to 0.
- r8c (= r8b + table reprojection to the smallest equivalent input after each ILC pass): CATASTROPHIC, survival
  3.9%, follow 22.8%. The hover model's B null space is far from null for the real bird. Reprojection removed from
  the code (negative result recorded here).
- Knob sweep on r8a's tables (opt/runs_cmd/scratch/sweep_knobs.py: 8 candidates x the SAME 64 V_op no-push
  schedules x 240 beats, per-flight Ka, 4.5 min; r8a/sweep_knobs.json): base: surv 98.4%, follow 98.4%, drift p95
  2.8 cm. q_int 30: surv 100%, follow 98.7%, drift 3.0 cm, sat 4.0%. Everything more aggressive loses flights:
  q_pos 0.01 36%, q_int 300 52%, q_vel 0.05 58%, r 0.035 72%, q_att 0.1/q_w 1 58% (the tables were learned with the
  base gain). No candidate reaches drift <= 2 cm while surviving (q_pos 0.01: 1.4 cm but 36% survival).
- G11 / F7 on r8a (opt/runs_cmd/scratch/f7b.py, r8a/f7b_x64.json): the 16 CPU-reference flights re-flown on the GPU
  in x64 with the CPU's solver settings (model iterations and line search, contacts off), 120 beats: state error /
  ERR_SCALE vs hover.launch.fly <= 1.3e-5 on all 12 V_op flights (max abs 1.3e-6), 3.9e-4 on a tumbling baseline
  flight; the float32 eval flights are 0.6-28 off (same flights). The port is exact; the G11 state-error miss is
  float32 + iterations-8 drift of chaotic command flights, so per F7 the CPU verdicts are trusted (they agree:
  identical survival 16/16, 98.9% verdicts). r8a cpu: |v_seg| diff median 0.00035, max 0.056 m/s (one segment).
- Sensors on r8a (P designed on pos,att,w + att_xy bias states, noise x1, 5 ms; N = N_lqgV1 as r7, --n-dv-clip 1
  --n-ilc-any): P x1 survival 61.7% (r7: 21.1%), follow 60.5%, |along| p95 0.058, hold p95 6.9 cm; x3 60.2%.
  N x1 survival 100%, direction-follow 76.9% (unchanged: FWD is never followed; CPU probe: N at FWD 0.3 flies
  (0.10, 0.45, 0.04) m/s, i.e. sideways, while its estimate says (0.12, 0.31, -0.45)).
  CPU probe of P observers (opt/runs_cmd/scratch/p_obs_cpu.py, r7 truth tables, no sensor ILC, 60 beats, constant
  0.3 m/s): pos,att,w (Prompt 1) loses FWD/UP (attitude estimate 20-36 deg off); pos,gyro loses DOWN and its yaw
  estimate is 60 deg off; pos,att,gyro loses FWD/UP/DOWN; pos,att,w + att_xy bias flies FWD/UP/REV/DOWN at x1
  (0.299-0.306 m/s, estimate error <= 5.6 deg). -> the bias states are what P needed at speed.
- r8a dt (fixed policy, 64 no-push V_op schedules x 240): dt 0.10 survival 98.4%, follow 96.2%, |along| med 0.0052;
  dt 0.16 survival 82.8%, follow 81.1% (FAIL; r7 with DOWN 0.24 had 98.4%): DOWN 0.30 costs dt robustness at 0.16.
- Sensors on r8a without the sensor ILC (r8a_s0, P = pos,att,w + att_xy bias): P x1 survival 66.4% (vs 61.7% with
  the ILC: the sensor ILC is not what limits P). The P losses sit in DOWN / REV and their transitions (DOWN->REV 13,
  REV->DOWN 8 of 43); the attitude estimate is 8-18 deg off in the 10 beats before the loss. The bias states'
  random walk (hover.reduced.BIAS_WALK 3e-4 rad/beat) is far slower than a transition changes the filter's tilt
  error; --p-bias-walk 3e-3 / 1e-2 (closed-loop rho 0.953 -> 0.737 / 0.685) are being tried.
- Second knob sweep on r8a's tables (same 64 schedules; r8a/sweep_knobs2.json): base again: surv 96.9% (98.4% in
  sweep 1: the GPU run-to-run spread on identical inputs is ~+-2%), q_int 50: 100% / follow 99.7% / drift 2.45 cm /
  max p95 12.7 cm, q_int 10: 100% / 98.7% / 2.85 cm, q_vel 0.15: 100% / 99.7% / 3.0 cm, r 0.07: 89%, q_pos 0.03: 97%.
- r9 (learn from scratch with q_int 30): survival 79.9%, follow 82.0% (DOWN 61%): BAD. Flying r8a's tables with
  q_int 30 was fine, learning with it is not: the slower integral (rho 0.894) has not settled inside the 15-30 beat
  ILC window, so the measured mean effort (the ILC update) is incomplete. -> learn with q_int 100, fly with a softer
  gain: --fly-q-int / --fly-q-vel (the flight gain is saved as Ka, the learning gain as Ka_learn; the verify pass
  flies the flight gain).
- Sensors, N with the velocity ILC (r8a_s1 / r8a_s2: --n-vel-ilc 1 --n-ilc-any --n-dv-clip 1; hover.command.
  trim_sensitivity / velocity_ilc move each N cell's trim by the linear trim sensitivity x its TRUE mean-motion
  velocity error measured in the learning flights; the policy reads only IMU): N x1 survival 100%, direction-follow
  100% (both runs; was 76.9%: FWD was never followed), x3 100% / 100%. Velocity-estimate bias [-0.005, -0.057,
  -0.122] m/s, drift 60-120 median 0.26 m (reported, not gated). s2 also zeroed N's velocity columns (--n-no-vel):
  same direction-follow, follow 44.6% vs 55.4%. -> G12 N passes with the velocity ILC (N keeps its velocity columns).
- P with a faster bias walk is WORSE: walk 3e-3 survival 35.9%, 1e-2 1.6% (3e-4: 61.7-66.4%). Dead end.
- r10 (learn q_int 100, fly q_int 50): survival 94.3% (no-push 92.7%), follow 94.2%: 19 of the 22 losses follow a
  FWD -> DOWN transition. r10b (learn q_int 100, fly q_int 50 AND q_vel 0.15): survival 100% (384/384: no-push
  288/288, push 96/96), follow 97.6% (HOVER 94.3, FWD 95.9, REV 100, UP 100, DOWN 98.0), |along| med 0.0065 p95
  0.028, cross p95 0.032, hold p95 4.4 cm, hover |v| med 0.0062 p95 0.0212 (FAIL by 1.2 mm/s), drift p95 4.2 cm
  (FAIL), RMS med 3.9 cm, max p95 15.6 cm (FAIL by 6 mm), rise 7, tilt-dev p95 8.2 deg, max tilt 10.5 deg, settle
  sat 4.0%, post-ILC envelope 5 deg FWD 0.30 REV 0.40 UP 0.36 DOWN 0.24; budget 207.3 s, first calls 5.8 / 5.1 s.
  Run-to-run spread is large (r8a 98.7%, r9 79.9%, r10 94.3%, r10b 100%): a replicate r10c is queued.
- --descend-delay N (hover.command.build_refs): N more wingbeats of hold after the horizontal stop before a descent;
  swept on r10's tables (r10/sweep_descend.json).
- Descent sweep on r10's tables (same 64 schedules): base 93.8% / follow 95.7%; q_int 50 96.9% / drift 1.96 cm;
  descend_delay 4: 96.9%; 8: 100% / 96.8%; 12: 98.4%; 16: 95.3% (DOWN follow 60%: too late); q_int 50 + delay 8:
  98.4% / drift 1.98 cm.
- r10c = replicate of r10b's config: survival 98.7% (no-push 98.6%, push 99.0%), follow 98.1%, drift 3.4 cm,
  post-ILC DOWN 0.24. r10b's 100% was partly the draw; the config is ~99% +- 1%.
- r10b dt: 0.10 survival 100%, follow 98.4%, |along| med 0.0057 (G9 PASS); 0.16 survival 92.2% (FAIL), follow 93.0%.
- r10b cpu (contacts off, 16 flights x 240): identical survival 16/16, verdict agreement 98.9%, |v_seg| diff median
  0.00049 max 0.0096 m/s (PASS); float32 state error 0.54-5.5 on V_op flights (x64 port exact: see r8a f7b).
- Sensors on r10b (flight gain q_int 50 / q_vel 0.15): P with the flight servo gain x1 85.2% survival, follow 75.8%.
  P own LQR gain, no integral (r8a_s3): x1 99.2% / follow 83.1% / hold 11.3 cm; x3 90.6%. P servo gain re-solved
  with q_int 10 (r10b_pq10): x1 97.7% / 80.6% / hold 6.3 cm; x3 89.1%. q_int 3 (r10b_pq3): x1 survival 99.2% (PASS),
  follow 83.3% (FAIL: cross p95 0.061), |along| p95 0.029 (PASS), hold p95 6.2 cm (PASS); x3 survival 91.4% (FAIL).
  N (velocity ILC) on r10b: x1 100% / direction-follow 100% (PASS), x3 100% / 100%.
- Hygiene: pytest tests/test_hover_command.py tests/test_hover_mjx.py: 9 passed (307 s). Golden --quick
  (opt/runs_cmd/golden_quick_round1.txt) identical to golden_quick_before.txt after stripping wall times (the before
  file also carries a trailing exit-code line "1" from its capture).
- r11 / r11b (NEW DEFAULTS: learn q_int 100, fly q_int 50 + q_vel 0.15, --descend-delay 8): two replicates, both
  survival 99.7% (383/384; push 96/96, no-push 287/288), follow 96.8% / 97.7%, post-ILC V_max DOWN 0.30 on BOTH
  (G7 passes for the first time), drift p95 4.0 / 3.6 cm, max p95 16.4 / 11.6 cm. r11b is the round-1 policy.

## 15. Round-1 result: r11b (`learn --out opt/runs_cmd/r11b`, all defaults; follow-ups with defaults)
| gate | measured (r11b) | pass |
|---|---|---|
| G1 survival all / no-push / push | 99.74% (383/384) / 99.65% (287/288) / 100% (96/96) | all PASS, no-push target 100% FAIL |
| G2 follow all; HOVER/FWD/REV/UP/DOWN | 97.7%; 96.3 / 99.8 / 99.8 / 100 / 93.0 % | PASS |
| G3 |along| med / p95, cross p95 | 0.0057 / 0.029, 0.033 m/s | PASS |
| G4 hold p95, |v| med / p95, drift p95 | 2.9 cm, 0.0050 / 0.0170 m/s, 3.57 cm | drift FAIL, rest PASS |
| G5 RMS med / max p95 / rise med | 3.4 cm / 11.6 cm / 7 | PASS |
| G6 tilt-dev p95 / max tilt / settle sat | 6.1 deg / 9.2 deg / 4.96% | PASS |
| G7 post-ILC V_max FWD/REV/UP/DOWN (5 deg) | 0.30 / 0.40 / 0.36 / 0.30 (15 deg: 0.25 / 0.40 / 0.48 / 0.24) | PASS |
| G8 servo/baseline |along| med | 0.66 (baseline survives 4.7%, follows 20.8%) | FAIL as specified (informational) |
| G9 dt 0.10 / 0.16 | 98.4% (63/64), 95.7%, 0.0059 / 100%, 98.1%, 0.0062 | 0.10 survival FAIL, rest PASS |
| G10 warm wall, first calls, shapes | 209.0 s, 5.75 / 5.14 s, 2 | PASS (cold: 247.5 s, final_cold, pre-round program) |
| G11 cpu survival / verdicts / |dv| med / max | 16/16, 100%, 0.00034, 0.0058 m/s | PASS |
| G11 state error (float32 eval) | 0.54-8.0 on V_op flights | FAIL float32; x64 port <= 1.3e-5 (r8a f7b): F7 -> trust CPU |
| G12 P x1 surv / follow / |along| p95 / hold | 97.7% / 77.7% / 0.029 / 9.1 cm | |along| PASS, rest FAIL |
| G12 P x3 survival | 87.5% | FAIL |
| G12 N x1 survival / direction-follow | 100% / 99.7% | PASS |
| G13 tests / golden / no module changed | 9 passed; identical; 4 new files | PASS |
Spread between replicates of one config: +-1% survival, drift 3.4-4.2 cm. P variant: best x1 survival 99.2%
(r10b_pq3 / r8a_s3), follow 83% at best; the x1 result moves 97.7-99.2% between policies.

Next (round 2): hover drift (catch-up after FWD: the position error left by FWD is still being removed in the
second half of the next hover segment); P follow (cross-track 0.06-0.07 m/s: the soft integral lets the biased
attitude estimate walk the bird sideways; try P velocity ILC like N, or a q_int between 3 and 10 with the sensor ILC
in estimate coordinates); P x3; G9 dt 0.10 (one flight); the no-push 100% target (one flight lost per run).

## 16. Diagnosis of r11b (fresh-eye pass, 2026-09-29; no GPU jobs; scripts in the session scratchpad)
Data: r11b traces.npz / ilc_passes.npz / sensor_traces.npz fetched from the box.
- G4 drift is almost entirely FWD -> HOVER. Hover-segment drift p95 by preceding command: release 0.25 cm, REV 1.47,
  UP 1.93, DOWN 1.99, FWD 4.76 cm (63% of FWD->HOVER segments > 2 cm). 90% of all > 2 cm segments follow FWD; without
  them the p95 is 1.83 cm. The drift is the position error being pulled back (cos(drift, -e_mid) median 0.97).
- Why: the V_op FWD cell (FWD 0.30, cell 6) was NEVER ILC-measured. Its clean flight saturated 100% / 93% of the window
  under the learning gain, so it failed good_sat 0.6 in both passes (ilc_info n=0 for cell 6 in both). Its entry is a
  linear extrapolation from 0.20 / 0.25 (w_ref 1.06 rad/s yaw rate, yaw 2.2 deg). The flight gain holds the table
  error with the integral: settle-half mean feedback 0.125 (zeta term 0.106), |zeta| at FWD end p50 0.46 and still
  growing (HOVER 0.03, REV 0.08, UP 0.13). At the stop the stored zeta (y 0.39) drives a mean excursion peaking at
  8.3 cm 10 beats into HOVER (y -5.8, x +4.9, z +3.3 cm). The verify-pass residual du_inf is 0.036 at FWD 0.30, against
  0.0045 at REV 0.30 and 0.004 at UP 0.30. DOWN 0.30 is second worst: residual 0.020, zeta_y -0.33, settle saturation
  19.5%. Saturation reaches only 5% in the eval because the flight gain is softer.
- G12 P follow: the position estimate is good (0.1-0.5 cm), but the attitude-estimate error is a slow transient (the
  observer's closed-loop rho is 0.953). The roll estimate error grows over a segment: REV 0.5 -> 2.0 deg and FWD 0 ->
  3.5 deg by beat 25; DOWN pitch goes 0.5 -> -3.0 deg. The sensor ILC measures in beats 15-30 of 30-beat cell flights,
  before this settles, so the late-segment bias is not learned. With q_int 3 the lateral error ramps: in failing REV
  segments it goes from -1.5 cm at mid to +12.8 cm at the end.
- The remaining losses are real dynamics, not float32. On the CPU (float64, model solver, contacts off), flight 238
  departs at beat 164-169 and flight 438 is lost at beat 153 (tilt 104 deg); the device lost them at 165 / 152.
  - Flight 238: lateral runaway in a DOWN -> REV transition. The ILC-learned yaw references swing from -5.0 (DOWN)
    through +1.2 deg; y error 17 -> 25 cm, vy 0.45 m/s, 10 inputs saturated.
  - Flight 438 (uniform arm, UP 0.48): a growing roll / lateral oscillation.
  - On flight 238 MuJoCo printed "Nan, Inf or huge value in QACC at DOF 13" (R_wing_pitch) and auto-reset the state,
    so the CPU loss rule (non-finite or tilt > 60) scores it as survived: `cpu` can hide a blow-up.

## 17. Tuning round 2 (2026-09-30; third session)
Round-2 code (new files only): `--good-cross` / `--good-sat` (ILC cell rule), `--ilc-fly-passes`, `--zeta-handoff`
(hover.command.integral_keep: x_ref column 13), `--zeta-gate` / `--stop-rate` (hover.command.integral_gate: column
14; slower stops), `sensors --p-vel-ilc` (velocity ILC for P as for N), `sensors --sensor-good-sat` (the sensor ILC
keeps its own 0.6), `--p-bias ""` = no bias states. `cpu`: a MuJoCo warning during a CPU flight (bad QACC -> MuJoCo
silently resets the state) now counts as the loss at that beat (mujoco.set_mju_user_warning in the worker; the
row carries `mujoco_warning`). Test: 13-wide x_ref with keep 1 == 12-wide (2 beats, x64), device law == servo_law
with keep / gate columns, integral_keep / integral_gate on hand cases.
- r12 = diagnosis 16's "measure saturated cells" (good_sat 1 + cross <= 0.05): WORSE. FWD 0.30 measured in pass 2
  while 93% saturated writes garbage references: yaw 9.2 deg (was 2.2), pitch -1.0 (was 3.9), w_y -0.62; the
  extrapolated 0.35 / 0.40 entries inherit it (yaw 17 / 25 deg). Eval 240: survival 99.7%, follow 95.6%, HOVER
  88.9%, hold p95 6.0 cm, drift p95 5.1 cm, RMS med 5.6 cm, max p95 18.6 cm; FWD->HOVER |e| 10 cm at the stop and
  -10.6 cm y overshoot 10 beats later; post-ILC DOWN 0.24. DEAD END (good_sat back to 0.6, cross test off).
- r13 = `--ilc-fly-passes 1` (pass 2 flies the flight gain, diagnosis 16's fallback): WORSE. FWD 0.30 is still 93%
  saturated in pass 2 (its table entry, extrapolated after pass 1, differs from r11b's final one, which the verify
  pass flew unsaturated), and DOWN learned under the flight gain loses: survival 95.3%, follow 92.6% (DOWN 77%),
  sat 9.3%, post-ILC DOWN 0.24. DEAD END. (r13b = r13 + cross <= 0.05 killed.)
- Sensor ILC 60 beats measured over 40-60 (`sensors --ilc-beats 60 --window 40 60` on r11b's policy, diagnosis 16's
  P fix): P x1 survival 47.7%, follow 44.6% (was 97.7% / 77.7%), x3 47.7%. DEAD END (N unchanged: 100% / 99.0%).
- P observer on pos,att_z,w (no roll / pitch filter channel, no bias states; rho 0.679 vs 0.953) on r11b: P x1
  survival 1.6%. DEAD END: roll / pitch from mocap position + rate is not enough for this servo.
- r14 = `--zeta-handoff` (the integral is scaled by keep_k = clip(v_{k+1}.v_k / |v_k|^2, 0, 1) as the reference
  changes, so a stop bleeds the integral that held the moving trim's table error; ILC cells only ramp up: keep 1,
  learning unchanged). Cold program compile (13-wide x_ref): first call 18.6 s, budget 222.6 s. Eval 240 (384 V_op):
  survival 100% (no-push 288/288, push 96/96), follow 98.4% (HOVER 100, FWD 99.5, REV 99.6, UP 100, DOWN 93.2),
  |along| med 0.0056 p95 0.029, cross p95 0.034, hold p95 2.9 cm, |v| med 0.0033 p95 0.0105, drift p95 2.40 cm
  (FAIL by 4 mm), RMS med 3.2 cm, max p95 10.5 cm, rise 7, tilt-dev p95 6.3 deg, max tilt 9.6 deg, sat 4.2%;
  post-ILC 5 deg FWD 0.30 REV 0.40 UP 0.48 DOWN 0.30. Drift by preceding command (p95): release 0.23, FWD 2.95,
  REV 0.89, UP 1.21, DOWN 2.08 cm (without FWD-preceded 1.64 cm). What is left after FWD: the stop itself
  overshoots (mean |e| 6.3 cm 10 beats after the stop: x +5.1, y -2.9, z +2.2 cm), and the pull-back runs into
  the drift window. -> sweep of the integral gate / slower stops (scratch/sweep_refs.py).
- r14b (replicate of r14): survival 99.7% (one no-push flight lost), follow 97.8%, drift p95 2.37 cm, max p95 16.0 cm,
  post-ILC DOWN 0.30. First call 19.6 s: the gate edit to mjx_command.law between r14 and r14b changed the program
  (cache miss); a warm first call needs code frozen between runs.
- Sweep on r14's tables (scratch/sweep_refs.py: 8 candidates x the same 64 no-push V_op schedules x 240 beats, one
  B x 30 program with a 14-wide x_ref; r14/sweep_refs.json), drift p95 / hold p95 / survival: no handoff 2.24 cm /
  2.8 / 98.4%; handoff 2.08 / 2.9 / 100%; + gate 0 1.89; gate 4 1.31 / 1.8; gate 8 0.82 / 1.0 / 100% (follow 98.4%);
  stop_rate 0.67 1.65; 0.5 2.02; 0.67 + gate 4 1.94. -> gate 8 (integral paused while HOVER is commanded and the
  reference stopped < 8 beats ago) = default.
- P velocity ILC (`sensors --p-vel-ilc 1` on r14's policy): P x1 survival 93.8%, follow 78.2%, hold p95 10.3 cm; x3
  89.1%. Not better. N unchanged (100% / 99.7%).
- r15 / r15b (defaults handoff + gate 8; the 14-wide program was already in the cache from the sweep: first call
  5.1 s, budget 206.4 / 206.5 s): survival 100% / 100%, follow 97.3 / 96.9% (DOWN 87.6 / 88.7% FAIL), hold p95 1.9 /
  1.8 cm, |v| med 0.0014, drift p95 2.45 / 2.31 cm (all of it FWD -> HOVER: other preceding commands <= 0.74 cm), sat
  7.5 / 7.2% (FAIL), post-ILC DOWN 0.24 (FAIL). Why DOWN: its V_op cell saturated 10 of 15 window beats (0.667) in
  pass 1 against 9 of 15 (0.600) in r11b / r14 / r14b, so good_sat 0.6 dropped it (extrapolated entry 100% saturated
  in pass 2). The DOWN 0.30 cell sits on the good_sat edge; FWD 0.30 is at 0.93-1.0. -> good_sat 0.8.
- Sweep on r15's tables (r15/sweep_refs.json, same protocol): gate 8 drift 0.86 cm / max p95 14.1 cm / follow 97.9%;
  gate 12 0.80; gate 8 + ff_lead 1 (att / w / u_ff looked up at v_ref one beat ahead) drift 0.58 cm, hold 0.5 cm,
  max p95 8.5 cm, RMS 2.3 cm, cross p95 0.024, rise 6, sat 5.6%, follow 98.7% (DOWN 93%); lead 2 FWD 85%; lead 3
  survival 64%; stop_rate 0.67 worse. -> ff_lead 1 = default.
- r16 (defaults + good_sat 0.8 + ff_lead 1): budget 208.2 s (first call 5.1 s), survival 100%, follow 96.4%, |along|
  med 0.0039, hold p95 1.5 cm, |v| med 0.0011, drift PASS, but sat 14.1%, max p95 25.1 cm, post-ILC FWD 0.25 / DOWN
  0.24 (FAIL): good_sat 0.8 let FWD 0.30 into pass 1 at 12/15 saturated beats with cross 0.095 (DOWN 0.30 at 10/15,
  cross 0.047), which spoiled FWD (pass 2: 13/15 saturated, cross 0.149). -> good_sat 0.7 + good_cross 0.08 (DOWN
  in, FWD 0.30 out unless it tracks). r16b killed.
- r17 (good_sat 0.7, good_cross 0.08, ff_lead 1 in learning AND flight): budget 207.9 s, survival 100% (384/384),
  follow 95.1% but FWD 85.3% (FAIL), sat 15.7%, max p95 27 cm, post-ILC FWD 0.25; drift p95 1.58 cm, hold 1.5 cm
  PASS; DOWN learned (0.667 in), post-ILC DOWN 0.30. FWD 0.30 was out in both passes (as in r15) but the lead in the
  ILC passes moved the learned FWD 0.20 / 0.25 entries (pitch 3.5 deg at 0.25 vs 4.1), so the extrapolated FWD
  0.30 entry came out at pitch 1.6 deg (r15: 4.1), w_y -0.45, and FWD saturated 67% of its settle beats. ->
  `--learn-ff-lead 0` (ILC and sensor-ILC passes learn without the lead; the flights and the verify pass use it:
  the r15 sweep's lead-1 result was on lead-0 tables). r17 chain stopped.
- r18 = THE ROUND-2 POLICY (`learn --out opt/runs_cmd/r18`, all defaults: learn q_int 100 / fly q_int 50 + q_vel 0.15,
  descend-delay 8, handoff, gate 8, ff_lead 1 in flight / 0 in learning, good_sat 0.7, good_cross 0.08). Every learn
  gate PASSES. Replicate r18b: every learn gate PASSES too (survival 100%, follow 98.6%, DOWN 93.2%, drift p95 0.70 cm,
  hold 0.68 cm, max p95 9.9 cm, sat 1.2%, post-ILC 5 deg FWD 0.30 REV 0.40 UP 0.48 DOWN 0.30, budget 205.4 s,
  first calls 5.0 s). Cold cache (r18_cold, fresh /workspace/jaxcache_cold_r18, --continue-to 0 --no-verify):
  242.9 s (linearize first call 21.7 s, flight program 24.0 s).
- r18 follow-ups: dt 0.10 / 0.16 PASS (100% / 100%, follow 98.9 / 98.9%, |along| med 0.0036 / 0.0041). cpu: identical
  survival 16/16 (flight 449, baseline: CPU "LOST 29 (MuJoCo warning)", GPU lost 31: the new warning rule caught
  it), verdicts 100%, |dv_seg| med 0.00023 max 0.0028; float32 state error 0.56-2.95 on V_op flights (85 on a
  tumbling baseline). F7 x64 (scratch/f7b.py r18 120): V_op flights 2.6e-8 .. 1.6e-7 of ERR_SCALE vs the CPU (max
  abs 1.6e-8), baseline 448/450/451 <= 3.9e-4, 449 nan (CPU stops at the warning): the port with the 14-wide
  references is exact; the float32 miss is drift of chaotic flights (F7: trust the CPU verdicts).
- r18 sensors (defaults: P = pos,att,w + att_xy bias, q_int 3; N = N_lqgV1 + velocity ILC): P x1 93.0% survival,
  follow 80.9%, |along| p95 0.039, hold p95 13.6 cm; x3 85.2%. N x1 100% / direction-follow 99.7% PASS; x3 100%.
  F6 variants of P on r18: design noise x3 (r18_pd3): x1 97.7% / follow 88.2% / |along| p95 0.036 / hold 5.2 cm
  (best P so far; survival and follow still FAIL), x3 91.4%; pos latency 0 (r18_pl0): x1 91.4% / 82.8% / hold 13.0
  cm, x3 85.2%; P's own LQR gain (r18_pown): x1 96.1% / 83.6% / hold 14.5 cm, x3 89.8%.

## 18. Round-2 result: r18 (all defaults)
| gate | measured (r18) | pass |
|---|---|---|
| G1 survival all / no-push / push | 100% (384/384) / 100% (288/288) / 100% (96/96) | PASS (r18b: same) |
| G2 follow all; HOVER/FWD/REV/UP/DOWN | 98.6%; 100 / 100 / 100 / 100 / 93.2% | PASS |
| G3 |along| med / p95, cross p95 | 0.0035 / 0.019, 0.028 m/s (DOWN alone: med 0.0099, p95 0.072, cross 0.040) | PASS overall |
| G4 hold p95, |v| med / p95, drift p95 | 0.70 cm, 0.0010 / 0.0038 m/s, 0.75 cm | PASS |
| G5 RMS med / max p95 / rise med | 2.5 cm / 9.9 cm / 7 | PASS |
| G6 tilt-dev p95 / max tilt / settle sat | 5.3 deg / 9.4 deg / 1.0% | PASS |
| G7 post-ILC V_max (5 deg) FWD/REV/UP/DOWN | 0.30 / 0.40 / 0.36 / 0.30; 15 deg 0.25 / 0.40 / 0.42 / 0.24; pre-ILC 0.25 / 0.40 / 0.36 / 0.24 (both maps); LP half budget 0.37 / 0.39 / 1.04 / 1.07 (15 deg 0.37 / 0.39 / 1.30 / 1.31) | PASS |
| G8 servo/baseline |along| med | 0.42 (baseline survives 4.7%) | PASS |
| G9 dt 0.10 / 0.16 | 100%, 98.9%, 0.0036 / 100%, 98.9%, 0.0041 | PASS |
| G10 warm wall, first calls, shapes; cold | 205.9 s, 5.7 / 5.0 s, 2; cold 242.9 s | PASS |
| G11 survival / verdicts / |dv| med / max | 16/16, 100%, 0.00023, 0.0028 m/s | PASS |
| G11 state error (float32) | 0.56-2.95 V_op (85 tumbling baseline); x64 <= 1.6e-7 | FAIL float32; F7 -> trust CPU |
| G12 P x1 surv / follow / |along| p95 / hold; x3 | 93.0% / 80.9% / 0.039 / 13.6 cm; 85.2% (best variant pd3: 97.7% / 88.2% / 0.036 / 5.2 cm; 91.4%) | FAIL (|along| PASS) |
| G12 N x1 survival / direction-follow | 100% / 99.7% | PASS |
| G13 tests / golden / no module changed | 9 passed (372 s); golden --quick identical (wall times stripped); 4 new files + scratch | PASS |
Next (round 3): P only. Best so far is the x3-noise design (97.7% / 88.2% at x1). Leads: design noise 3 with the
integral handoff tuned for P, a P observer model that includes the attitude filter's accel-driven tilt error, or a
P-specific learning gain; x3 survival is 85-91% for every P tried.

## 19. Diagnosis of G12 P (fresh-eye pass, 2026-09-30; no GPU jobs)
Scripts: scratch/sens_est_err.py <run> (device estimate-minus-true per command, last 10 beats of each segment),
scratch/obs_trim_mismatch.py (hover-model mismatch at the V_op trims -> the P observer's steady error),
scratch/obs_replay.py (the P observer replayed offline on r18's TRUE-state eval flights with a noise-free sensor
y = C [x_k; x_{k-1}]: no Mahony filter, no noise). Data fetched: r18/traces.npz, r18/sensor_traces.npz, r18_pd3.
- ROOT CAUSE: the P observer predicts with the HOVER model (Az z + Bz u_prev + constant hover drift) and is fed the
  full u including u_ff. The ILC-learned moving trims are not equilibria of that model: per-beat mismatch at V_op
  is up to vel 0.115 m/s, w 0.81 rad/s, att 2.3 deg (FWD). The Kalman filter absorbs it in the att_xy bias states
  and the velocity (slowest error modes 0.953 / 0.941) as a fake 3-6 deg tilt bias + 0.06-0.14 m/s velocity error.
- Evidence: the filter-free replay reproduces the device errors (design x1, median over segment ends):
  FWD roll +2.9 deg / vz -0.097 (device +3.2 / -0.139), REV roll +3.5 (device lost flights +4..7), UP vx +0.057
  (device +0.074), DOWN vel (-0.035,-0.035,-0.038) (device (-0.065,-0.021,-0.019)); design x3 moves the FWD error
  from roll to yaw -5.5 deg / vz -0.143 in the replay and yaw -2.4 / vz -0.127 on the device (pd3). So the error
  is the observer's process model, not the Mahony filter (the r5 / section 16 attribution is not supported).
- Replay with a trim-consistent predict (cz_k = x_ref_k - A x_ref_{k-1} - B u_ff_{k-1} on the design-state rows):
  att rms 1.44 -> 0.69 deg, vel rms 0.039 -> 0.025 m/s; FWD vz -0.097 -> +0.026, REV roll 3.5 -> 0.26 deg,
  UP vx 0.057 -> -0.001 (x3 design: 1.82 -> 0.90 deg, 0.054 -> 0.027 m/s). DOWN keeps ~1.9 deg roll.
- Losses (r18 P x1, 9/128): 8 of 9 end a >= 30-beat REV segment (flights 14 47 56 78 83 88 92 113; 93 is DOWN):
  est roll error grows +3 -> +7..13 deg while est vy stays ~0 and true vy runs to 1 m/s.
- P's sensor ILC never measures FWD V_op (P envelope FWD 0.20 / 0.15 in both passes): that entry is extrapolated.

## 20. Tuning round 3 (2026-09-30; fourth session): G12 P
Code (new files only): `hover.command.trim_consistent_cz` (per-beat observer offset cz_k = x_ref_k - Az x_ref_{k-1}
- Bz u_ff_{k-1} on the design rows, 0 on copy / bias rows) used by `CmdLinearController(cz_seq=)` and, as a
per-flight `cz_seq` (N, nz) ctl entry, by `mjx_command.CmdSensorPipeline` (N flies its constant cz tiled: same
numbers); `build_refs(climb_after_stop=)` (a climb waits until the horizontal v_ref <= this; default 0 = identical);
`sensors --p-cz-seq` (default on), `--p-tables true|sensor` (default true: P flies the true-state tables),
`--p-process-scale` (P observer process noise x PROCESS_STD), `--p-climb-after-stop` (P references only). The sensor
program gained one input (cz_seq): one new compile (18 s cold). Tests: trim_consistent_cz makes the reference the
observer model's trajectory; constant cz_seq == plain observer (host); the x64 device sensor test now runs through
the cz_seq path (== mjx_sensors); climb_after_stop 0 == old refs. Every run below is `sensors` on r18's policy.npz
(copied into the named dir), 128 seeds x {P, N} x noise {1, 3} x 120 beats; "ilc0" = `--sensor-ilc 0` (P does not
use it; N is then not meaningful). Scripts: scratch/p_losses.py <run> [noise] (which segment each lost P flight
dies in), scratch/p_replay_dev.py (noise-free replay of the P observer on one device flight),
scratch/p_obs_sweep.py (offline process-noise sweep), scratch/obs_replay_fn.py (section 19's replay through
trim_consistent_cz: same numbers, att rms 0.69 deg / vel 0.025 m/s).
| run | P change (all: cz_seq, true tables unless said) | P x1 surv / follow / |along| p95 / cross p95 / hold p95 | P x3 surv / follow |
|---|---|---|---|
| r18 (round 2) | none (hover-drift observer, sensor tables) | 93.0% / 80.9% / 0.039 / - / 13.6 cm | 85.2% |
| r18_cz | cz_seq (section 19's fix) | 95.3% / 90.9% / 0.034 / 0.067 / 3.6 cm | 91.4% / 78.2% |
| r18_czd3 | + design noise x3 | 97.7% / 87.4% / 0.031 / 0.063 / 4.9 cm | 82.0% / 75.5% |
| r18_czs | + sensor-ILC tables | 85.2% / 78.0% / 0.048 / 0.148 / 4.2 cm | 82.8% / 66.9% |
| r18_cz_qf (ilc0) | + flight gain (q_int 50) | 95.3% / 86.0% / 0.037 / 0.073 / 3.5 cm | 85.2% / 74.7% |
| r18_cz_q10 (ilc0) | + q_int 10 | 96.1% / 89.2% / 0.034 / 0.070 / 3.6 cm | 90.6% / 77.4% |
| r18_cz_q30 (ilc0) | + q_int 30 | 96.1% / 87.1% / 0.037 / 0.073 / 3.3 cm | 89.8% / 78.8% |
| r18_cz_c05 (ilc0) | + climb after stop 0.05 | 100% / 91.7% / 0.033 / 0.062 / 3.6 cm | 94.5% / 79.3% |
| r18_cz_c15 (ilc0) | + climb after stop 0.15 | 100% / 92.2% / 0.035 / 0.064 / 3.6 cm | 94.5% / 79.6% |
| r18_cz_ps3 (ilc0) | + process x3, bias walk 9e-4 | 100% / 95.2% / 0.037 / 0.049 / 3.2 cm | 96.1% / 83.3% |
| r18_cz_ps3s1 (ilc0) | same, --seed 1 | 100% / 94.4% / 0.038 / 0.049 / 3.8 cm | 98.4% / 86.8% |
| r18_cz_ps3s2 (ilc0) | same, --seed 2 | 99.2% / 94.3% / 0.035 / 0.053 / 3.8 cm | 97.7% / 85.7% |
| r18_cz_ps10 (ilc0) | + process x10, bias walk 3e-3 | 100% / 96.0% / 0.036 / 0.042 / 3.1 cm | 96.1% / 84.7% |
| r18_cz_ps3c05 (ilc0) | + process x3, walk 9e-4, climb 0.05 | 100% / 96.2% / 0.034 / 0.042 / 3.3 cm | 96.1% / 84.4% |
- Where r18_cz lost P flights (scratch/p_losses.py): all 6 x1 losses in a FWD -> UP transition (6 of the 15 FWD -> UP
  transitions in the 128 flights; 0 of every other transition pair lost). The true-state r18 flights survive it but
  it is their most stressed transition (tilt-dev max p95 5.3 deg, |v - v_ref| max p95 0.23 m/s; every other pair
  <= 4.9 deg / 0.22). In the lost flights the velocity estimate error grows 0.1 -> 0.3 m/s (z low, y high) while the
  position estimate stays within 1 cm, then the servo saturates (87% of the last 15 beats) and the bird yaws off
  (35 deg). scratch/p_replay_dev.py r18_cz 13: the noise-free, filter-free replay of the same observer on the same
  true states reproduces the device estimate error beat for beat (beat 90: vel err (-0.036, 0.331, -0.300) replay
  vs (-0.030, 0.335, -0.308) device): the process model, not the sensor pipeline, is wrong in that transient.
- Offline process-noise sweep (scratch/p_obs_sweep.py r18_cz, noise-free replay on the r18_cz P x1 flights) did
  NOT predict the device gain: model-error vel rms ~0.03 m/s for process x1 .. x30. The device measured it (table).
- The raised integral (q_int 10 / 30 / flight gain) alone did not help: 95-96% survival, follow 86-89%.
- CHOSEN (now the sensors defaults): `--p-process-scale 3 --p-bias-walk 9e-4 --p-climb-after-stop 0.05`, with
  cz_seq on and true tables. Three seeds of process x3 alone: x1 survival 100 / 100 / 99.2%, follow 95.2 / 94.4 /
  94.3%, x3 survival 96.1 / 98.4 / 97.7%.
- r18_s3 / r18_s3_seed1 / r18_s3_seed2 = the round-3 sensors follow-up with ALL DEFAULTS (`sensors --run <dir>`,
  --seed 0 / 1 / 2; N with its sensor ILC). Every G12 gate PASSES on all three seeds:
  | seed | P x1 surv / follow / |along| p95 / hold p95 | P x3 surv | N x1 surv / dir-follow |
  |---|---|---|---|
  | 0 (r18_s3) | 100% / 96.2% / 0.033 / 3.3 cm (cross p95 0.043, drift p95 3.7 cm, tilt-dev p95 6.3 deg) | 96.1% | 100% / 99.7% |
  | 1 | 100% / 95.7% / 0.034 / 3.7 cm | 97.7% | 100% / 100% |
  | 2 | 100% / 96.8% / 0.031 / 3.6 cm | 98.4% | 100% / 100% |
  P x1 per command (seed 0) pass / |along| p95 / cross p95: FWD 100% / 0.030 / 0.040, REV 93.0% / 0.032 / 0.058,
  UP 98.7% / 0.021 / 0.030, DOWN 91.9% / 0.075 / 0.048, HOVER 98.6%. N is unchanged (N's follow rate, 44-52%, is
  not gated: IMU-only walks, as round 2). Sensors stage wall ~225 s (untimed), sensor program 512 x 30 / 512 x 120
  cache loads ~19 s each (the cz_seq input changed that program once).
- Remaining weak spots (P x1): DOWN along p95 0.066-0.075 (G12's P gate is 0.08 overall; its per-command follow
  91-92%), and P x3 follow 85-87% (not gated).
- r19 = `learn --out opt/runs_cmd/r19` (all defaults) after the round-3 code, to check the learn path is untouched
  (the edits are in CmdSensorPipeline, CmdLinearController and a default-off build_refs keyword): every learn gate
  PASSES with r18's numbers (survival 100%, follow 98.6%, DOWN 93.2%, hold p95 0.68 cm, drift p95 0.70 cm, max
  p95 9.9 cm, sat 1.2%, post-ILC FWD 0.30 / REV 0.40 / UP 0.48 / DOWN 0.30, servo/baseline 0.41); warm budget
  208.3 s, first calls 5.7 s (linearize 67x1) / 5.1 s (flight program 512x30: cache hit), 2 shapes.
- Local: `uv run pytest -q tests/test_hover_command.py tests/test_hover_mjx.py` 11 passed (374 s;
  pytest_round3.log); golden --quick (golden_quick_round3.txt) identical to golden_quick_before.txt with wall
  times stripped; git status: still only the 4 untracked new files.

## 21. Round-3 result: policy r18 (= r19) + P defaults of section 20
| gate | measured | pass |
|---|---|---|
| G1-G8, G10 | r19 (= r18): see section 18 / r19 bullet above | PASS |
| G9 dt, G11 CPU | r18 (learn path unchanged; r19 reproduces r18's learn numbers) | PASS (G11 float32 state-error sub-gate: F7, as round 2) |
| G12 P x1 surv / follow / |along| p95 / hold p95 | 100% / 96.2% / 0.033 / 3.3 cm (seeds 1, 2: 100 / 95.7 / 0.034 / 3.7; 100 / 96.8 / 0.031 / 3.6) | PASS |
| G12 P x3 survival | 96.1% (seeds 1, 2: 97.7%, 98.4%) | PASS |
| G12 N x1 survival / direction-follow | 100% / 99.7% (seeds 1, 2: 100 / 100; 100 / 100) | PASS |
| G13 | 11 tests pass, golden identical, no existing module changed | PASS |

## 22. Round 4 (2026-09-30, fifth session): input-subset table at noise x1 / x2
Every run is `sensors` on r18's policy.npz (copied into the dir), all defaults except as listed, 128 seeds x 2 variants
x noise {1, 2} x 120 beats (512 flights: the cached 512 x 30 / 512 x 120 shapes, cache loads 18.4-18.7 s each).
Chains: scratch/chain_r4.sh (s4_x2, s4_full, s4_acc), scratch/chain_r4b.sh (s4_acc2). Table script:
`uv run python opt/runs_cmd/scratch/subset_table.py <run>...`. True-state row = r19 eval_120 (384 flights incl. 96
pushed, no noise). Measured from TRUE strobed states, float32, dt 0.13 ms, 5 ms mocap latency where pos/vel is read.
| subset (non-IMU ch.) | run | noise | surv | follow | |along| med / p95 | cross p95 | hold p95 / hover drift p95 | drift 60-120 med / p95 | tilt-dev p95 / max surv | sat |
|---|---|---|---|---|---|---|---|---|---|---|
| true state (12) | r19 | 0 | 100% | 98.7% | 0.0033 / 0.020 | 0.027 | 0.63 / 0.59 cm | - | 5.0 / 9.1 deg | 1.2% |
| pos,att,vel,w (6) | s4_full | x1 | 100% | 95.2% | 0.0088 / 0.038 | 0.043 | 3.9 / 3.9 cm | 4.9 / 13.5 cm | 6.8 / 10.1 deg | 1.0% |
| pos,att,vel,w (6) | s4_full | x2 | 96.9% | 88.2% | 0.0086 / 0.050 | 0.051 | 5.4 / 5.8 cm | 5.3 / 14.8 cm | 8.1 / 13.6 deg | 2.7% |
| P pos,att,w (3) | s4_x2 | x1 | 100% | 96.2% | 0.0070 / 0.034 | 0.042 | 3.3 / 3.6 cm | 5.0 / 12.1 cm | 6.4 / 9.7 deg | 0.5% |
| P pos,att,w (3) | s4_x2 | x2 | 97.7% | 90.1% | 0.0079 / 0.039 | 0.047 | 5.1 / 5.3 cm | 5.1 / 14.3 cm | 7.8 / 11.3 deg | 1.3% |
| P pos,att,w (3) | r18_s3 | x3 | 96.1% | 85.5% | 0.0083 / 0.040 | 0.052 | 7.5 / 7.1 cm | 5.4 / 14.7 cm | 8.3 / 12.5 deg | 1.7% |
| N att,w (0) | s4_x2 | x1 | 100% | 47.0% | 0.041 / 0.195 | 1.40 | 674 cm / 9.1 cm | 26.5 / 571 cm | 33.4 / 25.1 deg | 18.8% |
| N att,w (0) | s4_x2 | x2 | 100% | 43.8% | 0.039 / 0.204 | 1.40 | 659 cm / 14.5 cm | 30.0 / 565 cm | 33.3 / 25.4 deg | 18.8% |
| att,w,acc + acc bias, vel servo (0) | s4_acc | x1 | 0% | 0% | 0.31 / 0.58 | 1.18 | - | - | 97.8 deg p95 | 40.3% |
| att,w,acc + acc bias, vel servo (0) | s4_acc | x2 | 0% | 0% | 0.25 / 0.43 | 0.97 | - | - | 100 deg p95 | 51.9% |
| att,w,acc + acc,att_xy bias, design x3 (0) | s4_acc2 | x1 | 28.9% | 4.0% | 0.14 / 0.41 | 0.57 | 347 cm | 94.8 / 218 cm | 102 / 34.1 deg | 13.3% |
| att,w,acc + acc,att_xy bias, design x3 (0) | s4_acc2 | x2 | 27.3% | 3.2% | 0.12 / 0.42 | 0.60 | 352 cm | - | 104 / 37.0 deg | 19.2% |
Per command (pass rate, x1): true state HOVER 100 FWD 100 REV 100 UP 100 DOWN 94.0; pos,att,vel,w 96 / 97 / 93 / 97 / 93;
P 97 / 100 / 93 / 99 / 93 (x2: 84 / 94 / 92 / 96 / 85); N att,w 17 / 0 / 96 / 49 / 67; acc variants 0 everywhere
(acc2 DOWN 17). N att,w per command (x1): FWD saturates 100% of its settle beats and slides SIDEWAYS (cross p95 1.43 m/s,
along +0.167 m/s over the command) -- its 99.7% "direction-follow" counts only the along sign and hides that; REV
follows (|along| p95 0.019, cross 0.048); UP is slow (along -0.047) and DOWN -0.036: vertical and REV are trackable
open-loop in velocity, FWD is not, and position is unobservable (hover hold p95 6.7 m after 120 beats).
- Adding mocap velocity to P (pos,att,vel,w, 6 non-IMU) does NOT help: vel-est rms 0.037 vs 0.055 m/s, but follow
  95.2 vs 96.2% and survival at x2 96.9 vs 97.7%.
- IMU + accelerometer with bias states and a velocity servo (the "dead reckoning" subset) is lost: 0% (acc bias),
  28.9% (acc + att_xy bias, design x3); the observer's closed-loop rho is 0.994 in both (the acc bias is barely
  detectable), and the sensor ILC finds no good N cell after pass 2. Same verdict as the CPU check in section 10.
- The x1 rows of s4_x2 reproduce r18_s3 (survival 100 / follow 96.2 / hold 3.3 cm; N 100 / 47.0 vs 47.6): run-to-run
  float32 spread only.
- GPU wall (5070 Ti, warm cache): sensors stage 222 s (s4_x2), 223 s (s4_full), 240 s (s4_acc), 247 s (s4_acc2); of it:
  2 cache loads ~37 s, 2 sensor-ILC passes 512 x 30 ~66-75 s (only N uses them; P discards its sensor tables), 512 x
  120 eval 113-122 s (both variants x 2 noise scales). The true-state learn it builds on (r19) is 208 s warm.
- RECOMMENDED minimal-input policy: P = pos,att,w (3 non-IMU channels: mocap COM position, 5 ms latency) + IMU attitude
  filter + gyro, the trim-consistent LQG observer (process x3, att_xy bias, walk 9e-4) feeding the r18 servo with q_int
  3 and r18's true-state tables. Meets G12 at x1 on 3 seeds; at x2 survival 97.7% (below the 99% x1 gate; not gated).
  Without position no subset tried follows the command set (FWD fails in all of them).

## 23. Adversarial cross-check (2026-09-30, verifier): CPU reference + reruns
Scripts: opt/runs_cmd/xcheck/xc_cpu.py (CPU, box, JAX_PLATFORMS=cpu, 18 processes, 108 flights, 1541 s),
xc_gpu.py (device float32, the same 20 flights at 3 dt), xc_an.py (-> xcheck/xcheck.json). The flights are r18
policy.npz; true-state flights 8-23 (V_op, no push; 0-7 were the earlier CPU check) + pushed 292-295 at 0.13;
P = the sensors defaults rebuilt on CPU (hover.reduced.design pos,att,w / att_xy / walk 9e-4 / process x3 / 5 ms,
servo q_int 3, cz_seq, climb_after_stop 0.05, true tables), CPU Sensors noise x1 (its own RNG), schedules =
sensors seed 0 (flights 0-23 at 0.13, 0-15 at 0.10 / 0.16). All CPU runs: float64, model solver, contacts off.
- Force audit (CPU, every step of all 108 flights after release, 17.9 M steps): xfrc_applied max 0, ctrl on
  non-crank actuators 0 (nu = 4: L/R_motor_t1/t4), qfrc_applied == aero + pitch limiter exactly (sampled), neq 0,
  nmocap 0, gravity -9.81. Device: dx0.xfrc_applied 0, nu 4, neq 0, nmocap 0 at all 3 dt.
- True state, CPU vs device float32 (same dt, same 16 flights; 0.13 has 20 incl. 4 pushed), 240 beats:
  | dt | CPU surv / follow / along med / p95 / hold p95 | GPU same | surv identical / verdicts / dv med / max | pos diff max |
  | 0.10 | 16/16, 97.8%, 0.0048 / 0.030, 0.90 cm | 16/16, 97.8%, 0.0044 / 0.030, 0.91 cm | yes / 100% (92) / 0.00017 / 0.0026 | 1.4 cm |
  | 0.13 | 20/20, 98.3%, 0.0038 / 0.031, 0.67 cm | 20/20, 98.3%, 0.0039 / 0.031, 0.94 cm | yes / 100% (116) / 0.00028 / 0.0036 | 2.2 cm |
  | 0.16 | 16/16, 97.8%, 0.0054 / 0.028, 1.31 cm | 16/16, 97.8%, 0.0062 / 0.028, 1.59 cm | yes / 100% (92) / 0.00040 / 0.0085 | 4.3 cm |
  (0.13 vs the r18 eval traces themselves: identical survival, 100% verdicts, dv max 0.0035.) float32 state error /
  ERR_SCALE 0-120 max 3.0-6.0 (F7 as before). DOWN is the only command below 100% (92-93%) on both.
- P (pos,att,w) on CPU, noise x1, 120 beats: 0.10: 16/16, follow 97.8%, |along| p95 0.043, hold p95 3.2 cm;
  0.13: 24/24, 95.7%, 0.041, 2.8 cm (GPU r18_s3 same 24 schedules: 24/24, 95.7%, 0.040, 2.9 cm; verdicts 100%);
  0.16: 16/16, 95.6%, 0.043, 3.6 cm. Drift 60-120 p95 11-12 cm on CPU and GPU. No MuJoCo warnings.
- Timing (warm, /workspace/jaxcache_final, idle CPU): `learn --continue-to 0 --no-verify` budget 204.4 s (shell wall
  incl. interpreter 204.8 s); `learn` all defaults (xc_learn_full) 209.1 s. First calls 5.8 / 5.1 s. Reproduces G10.
- NOT reproducible: the learned policy varies run to run (float32 device ILC). r19 vs r18 tables differ up to 0.008
  (u) at the V_op cells and 0.23 at extrapolated high-speed cells. xc_learn_full (exact r19 command): G6 saturated
  settle 5.01% FAIL (r19 1.2%; DOWN segments 24.5% saturated vs 5.8%), G7 post-ILC DOWN 0.24 FAIL (r19 0.30); all
  else PASS (survival 100%, follow 98.6%, DOWN 93.0%). xc_learn (continue 0, so NB = 120 changes the eval schedules):
  G8 0.538 FAIL at eval_120 (r18 / r19 at eval_120: 0.43 / 0.44), DOWN 91.1%. DOWN at V_op 0.30 sits on the ILC
  envelope edge: some runs learn it (0.30), some do not (0.24).
