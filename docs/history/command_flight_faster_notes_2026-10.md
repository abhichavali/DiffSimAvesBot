<!-- Frozen copy (2026-10-01) of opt/runs_fast/NOTES.md, the lab notes of the "fly faster" rounds X1-X10 (learn --preset fast, the shipped command.tg). Do not update; the current state is docs/findings/2026-10-01-command-flight-faster.md. The run directories and scratch/ scripts it cites (opt/runs_fast/*) are gitignored and not in the repo; the scripts it names that were promoted are opt/hover_cmd.py trims (scratch/edge_trims.py) and opt/hover_cmd.py torque (scratch/cpu_torque.py). -->


## LIMITS PROBE (2026-09-30): what caps the speed under r18 today (measured, GPU + CPU)

Scripts (no src/ change): `opt/runs_fast/scratch/limits_probe.py` (device ladder + g FD, one program per n_step;
on_step only READS actuator_force / actuator_velocity / ctrl), `limits_post.py` (departure analysis ->
`limits/departure.json`), `cpu_torque.py` (CPU hover.launch.fly per-step crank force, float64, contacts off),
`diag_torque_steps.py` (device vs CPU per-step force, open loop: identical to print precision).
Runs: `limits/` (512 x 90 beats ladder + g FD 3 x 64 x 60; 459 s wall), `limits_t15/` (64 x 90; 131 s),
`cpu_torque/*.json` (10 CPU flights x 60 beats on the box CPU). GPU total ~10 min. r18 policy, flight gain,
ff_lead 1, ramp rate = max(V_op, s/2)/8 per beat, window beats 45-90, 1 clean + dithered (N(0, 0.01)) copies.

Ladder, base arm (r18 as shipped: 5 deg map, u_max 0.15), survival over 90 beats:
- FWD 0.30 100% (sat 8%) | 0.35 10% | >= 0.40 0%.  REV <= 0.40 100% | >= 0.45 0%.
- UP 0.30 100% | 0.40 89% | >= 0.50 0%.  DOWN 0.30 100% | >= 0.40 0%.  (CPU agrees on FWD .35 / REV .45 / DOWN .4
  lost, UP .4 flies.)
How flights fail (limits_post: first beat with |yaw| > 10 deg or tilt > 20 deg):
- FWD: whatever the command (0.35-0.8), departs when the bird reaches 0.28-0.32 m/s, YAW first (10 / 10), 50-75%
  of the preceding beats saturated; top channels b1.R_t1, a2.L_t1, b2.L_t1, b3.L_t4. A hard ~0.3 m/s ceiling of
  the 5 deg tables, a yaw / saturation loss.
- REV 0.45-0.5: departs at 0.30-0.36 m/s, 80-100% saturated, yaw first; >= 0.6: pitch-over (tilt 22 deg) at 0.2 m/s.
- UP >= 0.5: reaches the commanded 0.5-0.6 m/s, then yaw runs away with only 15-20% saturation (umax 0.25 arm:
  6-15%): NOT the input budget -> the extrapolated tables / hover linearization at speed (yaw axis).
- DOWN >= 0.4: departs at 0.30-0.39 m/s with 100% saturation, pitch 15-21 deg nose-down + yaw -11: input budget.
- Strobed vs mean-motion velocity: FWD 0.3 0.267 vs 0.294; UP 0.3 0.367 vs 0.300; DOWN 0.3 0.358 vs 0.301;
  t15 FWD 0.7 0.609 vs 0.703 (strobe is a phase sample: never grade speed on it).
u_max 0.25 (umax arm): no gain anywhere; FWD 0.35 0/3 (base 1/10), UP 0.30 2/3, UP 0.40 1/3 (worse than base).
15 deg map (tilt15 arm, same policy file): REV 0.45 100% (8/8), 0.50 75%; UP 0.40 100%; FWD 0.30 33%, 0.35-0.60
  0%, but FWD 0.65 75%, **0.70 100% (8/8 + 3/3; CPU 60 beats flies too: pitch 16.9 deg, along +0.003, cross
  0.005)**, 0.75 12%, 0.80 0%; 0.70 is 82% saturated (marginal island). => horizontal speed is not capped by physics
  at 0.3: at ~17 deg pitch the bird holds 0.70 m/s forward; the cap is the trim/table (attitude map, yaw).
Authority of g (crank speed; FD at n_step 910 / 892 / 874 = crank x0.990 / x1.010 / x1.031, no re-trim):
- Open loop per unit g (beat 1): dvz +1.37 m/s per beat (= +0.117 m/s^2 per 1% g), rising to ~4.5-6 by beat 5;
  but attitude moves 1.3-2.3 deg per 1% g in 5 beats (roll/pitch/yaw all), because L/R cranks are not symmetric.
- One-beat model (host LP, att/w rows held): the 24 depths within u_max give 0.343 m/s/beat pure vertical
  (2.93 m/s^2) and 0.303 m/s/beat level forward (2.59 m/s^2): 1% g ~ 4% of the vertical input budget.
- Closed loop (r18 servo, same tables): -1% g: every g0 cell still flies; +1% g: DOWN 0.30 lost 4/4, REV 0.40
  50%; +3% g: DOWN 0.30 and REV 0.40 lost, HOVER |u|inf 0.035 -> 0.085. No cell beyond the g0 cutoffs flies at any
  g. => g is a weak, attitude-coupled lift knob; as a fixed offset it only costs trim; useful only with a re-trim
  (F3/F4) at that g.
Actuator cost (motors: MuJoCo velocity servos kv 1 N m/(rad/s), ctrlrange +-120 rad/s, NO forcerange ->
  no torque limit is modelled; crank speed 53.7 rad/s):
- ctrl (speed command) peak 60 (CPU hover) .. 82-88 rad/s at speed / +3% g: never clamped (0 steps).
- Mean mechanical power ~16.0 W at hover, 16.0-16.4 W (<= 1.02x) at every flown speed, 16.3 W at t15 FWD 0.7;
  +1% g +1.7%, +3% g +5.7%.
- CPU per-step crank torque (hover.launch.fly, 60 beats, window 30-60), smooth peak (beat steps >= 5) / p99 / RMS:
  hover 214 / 211 / 88-105 mNm; FWD 0.3 263 / 242 (1.23x); UP 0.3 203; UP 0.4 331 (1.54x); DOWN 0.3 356 / 325
  (1.66x); REV 0.4 387 / 359 (1.80x); t15 FWD 0.7 388 / 366 / 130-157 (1.81x peak, 1.4x RMS). REV 0.4 and DOWN
  0.3 (already in the shipped envelope) exceed 1.5x the hover peak.
- Beat-boundary spikes: the servo changes the harmonic depths once per wingbeat, a step in the commanded crank
  velocity (53.7 rad/s x du) that the kv servo turns into a one-step torque impulse ~ 0.24 + 45-50 x |du|_inf N m
  (the 0.24 N m baseline is there in the open loop too, at step 1 of each beat; CPU median 0.24 N m at hover,
  0.44 FWD 0.3, 2.1 N m UP 0.4, 5 N m FWD 0.35 before loss; corr with |du| 0.3-0.97). Device per-beat peaks
  (2-9 N m) are these spikes (device == CPU per step in the open loop). Smoothing u across the beat would remove them.
Dead ends here: fixed g offsets (+1/+3%) and u_max 0.25 on today's tables: no envelope gain, both worse.
Leads (measured, not tried): tables/trim at larger tilt with yaw handled (t15 FWD 0.70 flies); UP needs yaw work,
  not budget; DOWN / REV are budget-bound (100% sat) -> g re-trim or different harmonics.

## IDEA ANALYST: feedforward learning at speed (2026-09-30, local numpy on r18 files, no GPU)
Scripts: scratchpad refceil.py / linacc.py (not in repo). All numbers measured on r18 policy.npz / ilc_passes.npz /
metrics.json and the limits ladders.
- ILC batch waste (r18): pass 1 / 2 lose 219 / 216 of 512 flights; 171 / 188 flights sit on cells where every copy
  dies; good cells 52 / 55 of 122. ILC calls take 35.8 / 35.0 s vs eval calls 28.2-30.9 s (same program).
- Map 1 (15 deg) FWD 0.40-0.75: 4/4 lanes alive in BOTH passes but 100% saturated, window pitch ~0 deg; the verify
  pass (post-ILC tables) tracks FWD 0.65-0.75 (along -0.01..-0.04, sat 0.73-0.93) and the 90-beat t15 ladder holds
  0.70 at pitch 17-18 deg from beat 10 on. The 0.35-0.60 gap was never measured; FWD entries >= 0.35 in map 0 are the
  0.9 u_max clip (|u|inf 0.135), i.e. extrapolation garbage.
- Learned FWD 0.30 (map 0) |u|inf 0.131 vs LP 0.061; its min-inf equivalent on hover B is 0.048 (2.7x inefficient);
  the ILC correction is 93% in range(B^T) (it comes through Ka). Real ladder: FWD 0.3 map0 pitch 5.5 |U|inf 0.120 rms
  0.055; map1 pitch 9.4: 0.142 / 0.070 (1/3 alive); FWD 0.7 pitch 16.9: 0.149 / 0.080.
- Hover-model LP, min |u|inf at fixed pitch, FWD 0.7: 0.151 (0 deg) 0.140 (6) 0.139 (9) 0.140 (12-15) 0.142 (18):
  flat -> the model cannot choose the pitch at speed; only flights can.
- Reference-grading ceiling (the reference itself through segment_metrics, 384 x 240, r18 map-0 tables): rate V/8
  keeps follow 97.9-98.6% up to V_h 0.7 but DOWN falls to 89.5% at V_z 0.3 when V_h >= 0.5 (descend shaping);
  rate V/6: 99.2-100% (DOWN 96-100%); fixed acceleration 0.3/8 per beat: FWD 78%, follow 82.7% at V_h 0.7;
  V/12: ~91%. => a new V_op needs rate scaled with V_op (acceleration x V_op/0.3) and its own ceiling reported.
- Linear-model sim (LP 25-deg tables, flight gain, 256 x 240): an acceleration feedforward (quasi-steady min-norm
  att/u for a = dv_ref) changes nothing at V_h <= 0.5 (sat 0-0.4%), at 0.7 tilt-dev p95 10.3 -> 6.6 deg, max dev p95
  13.2 -> 9.3 cm, sat 33 -> 29.8%: a small lever, CEM knob at most.

## X1 (round 1): rigid-rotation linearization `--lin-rigid-rotation` (2026-09-30, GPU box 40346, RTX 5070 Ti)
Code: `hover.launch.rotate_bird(..., rotate_vel=False)` / `launch.com_velocity`, `mjx_flight.rotate(..., rotate_vel)` /
`com_velocity` / `one_beat` / `linearize`, `hover.lqr --rigid-rotation`, `opt/hover_cmd.py learn --lin-rigid-rotation`
(dev group; default off). rotate_vel: v' = v_com + R (v - v_com) per free joint, v_com = whole-bird mass-weighted COM
velocity from cvel (== mj_subtreeVel). Default path unchanged: golden --quick output identical (9/32 as before), pytest
test_hover_command + test_hover_mjx 13 passed (375 s local), r18-config learn on this code (r18_calib) reproduces r18's
A / B to 2e-11.
STEP 0 (CPU, `scratch/model_at_speed/rotchk2.py`, 12 s): the whole-bird COM velocity at the strobe phase is 0.67 m/s
  (strobed 0.70: the intra-beat orbit), so "rotate v" has two readings. att_y column (att_y, vx, vz, w_y, w_z):
  today 1.113 / 0.695 / -0.965 / 0.915 / 1.399 (the perturbation ALSO moves the strobed vel by (-0.575, 0, -0.926)/rad);
  abs (v' = R v, = rotchk.py) 0.990 / 0.922 / -0.755 / -0.245 / 0.171 (the task's expected numbers: reproduced) but it
  moves strobed vz by -0.702/rad (R v_orbit): still a mislabelled att+vel perturbation;
  com (the flag, v_com kept) 1.020 / 0.968 / -0.276 / -0.178 / 0.107, strobed vel moves only (-0.011, 0, -0.032): the
  clean attitude column. att_x: com vy -1.039, w_x -1.087 (abs -1.041 / -1.084; today -0.537 / -0.114).
  CPU `hover.lqr --rigid-rotation` (box CPU, 39 s) -> x1_rot/lqr_rigid_cpu.npz; device x64 relinearize vs it: A 1.1e-11,
  B 1.4e-11 relative. Hover LQR closed |eig| 0.658 (r18 model 0.651); servo learn rho 0.808 (0.804), FLIGHT Ka rho 0.866
  (0.866).
STEP 1 learn --lin-rigid-rotation --tilt-maps 5 15 (V_op 0.3) -> x1_rot; repeat x1_rot_b (independent learn):
  LP envelope half/full budget: 5 deg FWD 0.51/0.86 REV 0.52/0.87 UP 0.83/1.67 DOWN 0.85/1.69; 15 deg FWD 0.83/1.19 REV
  0.84/1.20 UP 0.83/1.67 DOWN 0.85/1.69 (r18 model: FWD 0.37 on BOTH maps -> the old model could not buy speed with
  tilt; UP 1.04 / 1.30 half-budget came partly from the spurious vz<-pitch term).
  pre-ILC V_max: 5 deg FWD 0.35 REV 0.50 UP 0.42 DOWN 0.48 | 15 deg FWD 0.65 REV 0.60 (b: 0.55) UP 0.42 DOWN 0.48
  (r18 config on this box: 5 deg 0.25 / 0.40 / 0.30 / 0.24). Good ILC cells 75-78/122 (r18 52-55).
  post-ILC V_max (both runs identical): 5 deg FWD 0.30 REV 0.55 (b 0.50) UP 0.42 DOWN 0.54 | 15 deg FWD 0.65 REV 0.60
  UP 0.48 DOWN 0.54 (r18: 0.30 / 0.40 / 0.36 / 0.30). The chosen map is now 15 deg.
  eval-240 at V_op 0.3 (384 flights): x1_rot survival 98.4% (no-push 100%, pushed 93.8%), follow 97.5% (DOWN 92.1%),
  |along| p95 0.0057, hold p95 0.35 cm, tilt p95 5.9 deg; x1_rot_b 99.2% (pushed 96.9%), follow 97.9%. All lost flights
  are PUSHED flights in or just after UP (loss_ctx.py). CPU (16 flights x 240): identical survival, 100% verdicts.
STEP 2 limits_probe --only (8 flights/cell x 90 beats, window 45-90; x1_rot/limits/ladder_table.txt; 157 s):
  15 deg map: FWD 0.30-0.65 100% (sat 0%, |u|inf 0.045-0.080, pitch 7-12 deg mean, tilt max <= 17.7), FWD >= 0.70 lost
  (yaw first at 0.52-0.65 m/s); REV 0.30-0.55 100%, 0.60 25%; DOWN 0.30-0.50 100%, 0.55 100% alive but tilt 51 deg;
  **UP 0.30 38%, 0.35 12%, >= 0.40 0%: the climb departs in ROLL (+-20 deg) after 25-45 beats**, 0-30% saturated (r18
  flew UP 0.30 100% / 0.40 89% for 90 beats). 5 deg map: FWD >= 0.35 0% (yaw), REV <= 0.40 100%, UP 0.30 12%.
  => the FWD 0.35-0.60 hole is closed (0.40 / 0.50: 8/8); the ILC / envelope window (15-30 beats) cannot see the slow
  climb divergence; eval UP segments (30-50 beats) mostly end before it.
Random-command eval at raised V_op (learn with --v-op-*):
  x1_v05 (V_op h 0.5, z 0.4): 53.6% survival (240 beats), follow 69.3%; all 178 losses in / right after UP (~32 beats in)
  -> DEAD END at z 0.4 with this servo. CPU agrees (same flights lost at the same beats; 1 disagreement of 16).
  x1_h05 (V_op h 0.5, z 0.3): survival 98.7% (no-push 100%, pushed 94.8%; all 5 losses pushed, around UP), follow 96.8%
  (FWD 99.8 REV 99.1 UP 98.9 HOVER 98.0 DOWN 88.6% = at the ~89.5% reference ceiling of descend shaping at V_h 0.5),
  |along| p95 0.0136 m/s (2.7% of V_op), hold p95 0.41 cm, tilt dev p95 6.7 deg, sat 0.1%.
Actuator cost (CPU hover.launch.fly, x1 policy, 60 beats, window 30-60; cpu_torque/x1_*.json): smooth peak / RMS per
  crank / mean power: HOVER 217 / 88-106 mNm / 16.03 W; FWD 0.3 236 (1.09x); FWD 0.5 257 (1.18x) / 95-117 / 16.16 W;
  FWD 0.65 277 (1.28x) / 99-124 / 16.19 W (beat-boundary spike 1.5 N m, du 0.012); REV 0.5 228 (1.05x); UP 0.3 194;
  DOWN 0.5 350 (1.61x: FLAG > 1.5x) / 127-145 / 16.10 W. No torque limit is modelled (velocity servos, ctrl +-120 rad/s:
  peak ctrl 60-73, never clamped). Cheaper than r18 at 0.3-0.4 (r18 FWD 0.3 263, DOWN 0.3 356, REV 0.4 387 mNm):
  the tilt now carries the speed, |u| stays small.
Timing (this box is ~25% slower than the one r18 was timed on: eval calls 34-39 s vs 28-31 s): warm learn+eval-120
  x1_rot_b 267 s, x1_h05 273 s; r18 config on the same box (r18_calib, 13 s of cold linearize compile) 290 s. Cold
  (x1_rot): 308 s. All > 240 s on this box. GPU wall this round ~50 min (7 jobs).
Verdict: KEEP as the linearization for X2+ (FWD post-ILC 0.30 -> 0.65, REV 0.40 -> 0.60, DOWN 0.30 -> 0.54, flight rho
  0.866). NOT a new shipped best: at V_op 0.3 pushed survival 94-97% (r18 100%), sustained UP diverges in roll after
  ~30 beats (a regression vs r18), budget > 240 s here. Next: fix the climb (relinearize / schedule Ka at UP speed,
  or a longer ILC window for UP), then certify V_op h 0.5.
  x1_h05 extra checks: dt 0.10 / 0.16 (64 flights x 240 beats): survival 100% / 100%, follow 97.3% / 97.3% (545 s);
  CPU (16 flights x 240, box CPU, 95 s): segment verdict agreement 95.7% (bar 95%), survival NOT identical: pushed
  flight 290 (push at 43, in UP) is lost on CPU at beat 69 and flies on the device; the baseline flight 451 is lost on
  both (beats 217 / 113). => at V_op h 0.5 / z 0.3 everything but pushed-in-climb survival (94.8%) and the budget meets
  the bar; UP under a push is the marginal mode on both simulators.
Scripts added this round: scratch/model_at_speed/rotchk2.py (3-mode column check), scratch/loss_ctx.py (where eval
  flights die: prev -> cmd, beats after switch), limits_probe.py (now uses the policy's own HOVER cells as the torque
  reference when present), cpu_torque.py --prefix.

## X2' (round 2): the sustained-climb roll departure, fixed with a flight-regressed climb gain (2026-09-30, box 40346)
Code (defaults unchanged: golden --quick output identical to round 1, 9/32; pytest test_hover_command + test_hover_mjx 13
passed, 379 s local): `hover.launch.release(v=)` and `hover.mjx_cradle.Cradle.run(v=)` / `wrenches(vels=)` (moving
cradle); `hover.command.regress_ab` (A, B regression from flights, factored out of local_models; `B_fixed`, `A_prior`,
`ridge`) and `servo_closed`; `opt/hover_cmd.py learn --climb-gain reg` (+ `--climb-reg fixB|full`, `--climb-ridge 0.3`,
`--climb-rho-max 0.95`, `--climb-speeds 0.24 0.48`, `--climb-window 8 30`): after the ILC passes, regress A (B = the
hover linearization's) from the passes' dithered UP-cell flights (both maps, beats 8-30), shrink toward the hover A
(ridge, x3 until the gain holds the hover model at rho <= 0.95, up to 3 raises), DARE with the flight weights -> the
FLIGHT gain for every axis (a.Ka_fly; saved as policy Ka, so export / LiveCommand need no change; climb_A / climb_B
saved as extras). No schedule, no law change. Scripts: scratch/climb_diag.py, climb_lin.py, climb_ka.py,
climb_ka_lib.py, climb_probe.py, climb_ladder.py, climb_reg_variants.py, climb_robust.py, climb_push.py, ref_ceiling.py.
STEP 0 (x2c_diag, 104 x 120 beats, 165 s): x1_rot UP 0.30 / 0.36 / 0.42 / 0.50, 8 flights (1 clean + 7 dither 0.01):
  map1 (15 deg) 25 / 12 / 0 / 0%, map0 (5 deg) 38 / 12 / 0 / 0%, swap (map1 + map0's UP axis) 12 / 12 / 0 / 0%. Every
  departure is ROLL first (sign random), beat 11-53 (one at 104), 0-10% saturated in the 10 beats before (whole flight
  8-30%). => not cause A (5 deg UP fails too; the swap does not help), not C (unsaturated): cause B.
  UP table entries (rel. hover): L/R antisymmetric part ~0.7x the symmetric one on both maps; through B it is a
  w_x moment of +0.29 rad/s/beat at UP 0.30, the LP's cancellation of the hover model's w_x <- v_z term (-0.737 per
  m/s/beat): a consistent steady trim, not an asymmetric-table fault.
  Hover Ka lateral block on the hover model: rho 0.865 (open 1.125). A model regressed from the dithered climb flights
  (beats 8-30): hover Ka rho 1.009-1.05 (lateral block 0.99-1.09), a complex pair |0.691 +- 0.735j| = 1.009, period 7.7
  beats = the observed roll oscillation (clean flight decays; dither / pushes grow it); same regression at HOVER 0.887
  (FD 0.866: the regression is sound).
STEP 1 (x2c_lin, 308 s; device x64 moving cradle): device vs CPU (mcradle.py == launch.release(v=)) mean wrench at UP 0.3,
  hover theta, 24/8 beats: identical to 1.5e-10 (bar 2%). Climb trims (24 depths + tilt, LP min |du|inf, trust 0.03,
  Newton FD in one 107-run batch, 5 iterations x 40-53 s): UP 0.30 |u|inf 0.048 tilt (1.36, -0.86) deg, F 0.06 mN / M
  0.002 mNm; UP 0.45 |u|inf 0.066 tilt (2.0, -2.26) deg. CPU release at speed (12 s) + x64 linearize (rigid rotation,
  25 s): A rel diff vs hover 0.42 / 0.30, B 0.95 / 0.82; open rho 1.25 / 1.28; lateral |eig| 1.25.
  FALSIFICATION: the hover flight Ka on the FD climb models: rho 0.878 (UP 0.30) / 0.870 (0.45), lateral 0.859 / 0.849:
  STABLE -> scheduling Ka on these models cannot help (said so; --schedule-z NOT built). The FD moving-cradle model does
  not show the flight instability that the flight-regressed model shows (7.7-beat pair): an unmodelled / larger-amplitude
  effect (not resolved: candidates are the internal strut / passive-pitch states the 12-state one-beat model omits).
  Instead of the prescribed fallback (longer-window UP ILC: feedforward cannot fix a perturbation-driven oscillation;
  x1 UP 0.40 is lost even clean), gains were tested on flights:
STEP 1b probes (x2c_probe, 190 x 120, 180 s; x2c_ladder / 2 / 3, 138 flights per gain x 120 beats, 3-5 min each):
  UP 0.30 / 0.40 with dither 0.01: x1 gain 1/4 / 0/4 (0.40: even clean and every push lost), q_att 0.025 2/4 / 0/4,
  DARE on FD climb 4/4 / 1/4, r18 gain 0/4 / 0/4, DARE on the flight-regressed climb model (dare_reg) 4/4 / 4/4 and every
  push. All-axis ladder (1 clean, 3 dither, roll 1 rad/s push, side 0.4 push): dare_reg UP 0.30-0.60 6/6 each, FWD <=
  0.65, REV <= 0.55, but DOWN 0.50 77% saturated and DOWN 0.54 lost (x1: DOWN 0.54 flies). Full (A, B) regression is
  fragile across data sources (the ILC-pass version is unstable on the hover model, 1.055): closed-loop data split A vs B
  poorly. fixB (A only) + ridge toward hover A is consistent across 3 sources (|dK| spread 0.08-0.13, all rho <= 0.93);
  its ladder (both ILC sources, + blend / lateral-rows variants): FWD <= 0.65, REV <= 0.55, UP 0.30-0.60, DOWN 0.30-0.60
  all 6/6 (one UP 0.60 dither loss for one source); FWD 0.70 and REV 0.60 lost with every gain.
DEAD ENDS: x2c_climb_noridge (first learn, full-A fixB without ridge): the regressed gain was unstable on the hover model
  (1.024) -> guard kept the hover gain -> = x1_h05 (98.96% survival, pushed 95.8%). x2c_z04 (first V_op z 0.4 learn,
  ridge 0.3 only): gain rho 1.008 on hover -> hover gain kept -> 53.4% survival (as x1_v05). => the ridge ladder.
  B 384 (timing trim, x2c_b384): budget 285.9 s (cold 384x30 compile 29 s; warm ~267 s): eval calls cost the same at
  B 384 as 512 (latency-bound): NO timing gain; eval-240 only 288 V_op flights. Not kept.
STEP 2 learn --lin-rigid-rotation --tilt-maps 5 15 --climb-gain reg --v-op-h 0.5 --v-op-z 0.3:
  x2c_climb / x2c_climb_b (independent learns; gain accepted at ridge 0.3, hover-gain rho on the regressed climb model
  0.936 / 0.944 -> climb gain 0.866, on hover 0.872 / 0.866, |dKa|/|Ka| 0.14 / 0.16):
  eval-240 (384 flights): survival 100% / 100% (no-push 100, pushed 100 / 100), follow 97.86% / 97.86%, per command
  HOVER FWD REV UP 100%, DOWN 89.5% = exactly the reference ceiling (ref_ceiling.py: the reference itself passes 89.5% of
  DOWN segments; the flights pass 100% of those and none it fails), |along| p95 0.0133 / 0.0134 m/s (2.7% of V_op_h),
  hover hold p95 0.34 / 0.35 cm, tilt dev p95 4.5 / 4.6 deg, saturation 0.0%.
  post-ILC V_max (15 deg map): FWD 0.65 / 0.70, REV 0.55 / 0.70, UP 0.66 / 0.66, DOWN 0.54 / 0.54 (x1: 0.65 / 0.60 /
  0.48 / 0.54).
  Ladder (x2c_climb/limits, 8 flights x 120 beats, window 60-120): FWD 0.30-0.65 100% (0.70 0%), REV 0.30-0.55 100%
  (>= 0.60 0%), UP 0.30-0.60 100% (sat 0-2%), DOWN 0.30-0.60 100% (0.60: 66% saturated). x1: UP 0.30 38%, >= 0.40 0%.
  Push probe (x2c_climb/push: 16 pushes x UP 0.3 / 0.4, at beat 60-90 of 150): 32/32 survive (side +0.4 departs >10 deg
  at beat 76 and recovers).
  dt 0.10 / 0.16 (64 x 240): 100% / 100%, follow 97.3% / 97.3% (508 s). CPU (28 flights x 240, box CPU 12 procs, 135 s):
  identical survival, segment verdicts 100%, |v_seg| median 7e-5; policy flights' state err/scale <= 1.08 (the
  baseline-arm flights 9.8-24.6).
  V_op h 0.5 / z 0.4 (UP 0.4 flies 100% in the ladder and push probe): x2c_z04b (ridge raised to 0.9: 0.3 gave 0.978 on
  hover) / x2c_z04c (ridge 0.3): survival 100% / 100% (pushed 100 / 100), follow 97.86% / 97.86%, DOWN 89.5% (at its
  ceiling: 100% of reference-passed segments), |along| p95 0.017 m/s, hold p95 0.4 cm, tilt dev p95 5.4 deg, sat 0%;
  post-ILC (15 deg) FWD 0.70 / 0.70, REV 0.60 / 0.55, UP 0.60 / 0.78, DOWN 0.60 / 0.54. dt 0.10 / 0.16 (z04b): 100% /
  100%, follow 97.3%. CPU z04b: segment verdicts 98.8%, policy flights all fly on both (the one survival mismatch is a
  baseline-arm flight, GPU lost / CPU flew), but policy state err/scale 2-20 (vs <= 1.08 for x2c_climb): z04b's gain
  CHATTERS in a climb: a period-2 input limit cycle, |du| per beat median 0.022 in UP segments (z04c 0.004, x2c_climb
  0.003, x1_h05 0.008), CPU UP 0.4 |du| 0.025 every beat -> 1.4 N m one-step crank impulses. Not predicted by any linear
  model (no eigenvalue near -1 on the FD or regressed models). => V_op z 0.4 passes the grading bar in 2/2 learns, but
  one of the two gains chatters: flagged.
Actuator cost (CPU hover.launch.fly, 60 beats, window 30-60; cpu_torque/x2c_*.json = x2c_climb, x2cz04_* = x2c_z04b):
  smooth peak (beat steps >= 5) / per-crank RMS mNm / mean power W / median beat-boundary spike mNm:
  x2c_climb: HOVER 217 / 88-106 / 16.03 / 237; UP 0.3 197 / 87-105 / 16.22 / 285; UP 0.4 197 / 90-105 / 16.30 / 632 (all
  979); UP 0.5 210 / 92-107 / 16.32 / 307; UP 0.6 235 (1.08x) / 104-112 / 16.33 / 634 (all 2781); FWD 0.5 258 (1.19x);
  FWD 0.65 262 (1.21x) (all 1038); REV 0.5 216; DOWN 0.3 269 (1.24x); DOWN 0.5 349 (1.61x, > 321: FLAGGED, as in X1).
  x2c_z04b: UP 0.4 201 / 90-115 / 16.30 / 1390 (all 1480: chatter); UP 0.6 274 (1.26x) / 107-172 / 16.43 / 2705 (all
  8337: FLAGGED); DOWN 0.4 295 (1.36x) / 113-130 / 16.01; DOWN 0.5 355 (1.64x: FLAGGED). No torque limit is modelled
  (velocity servos kv 1, ctrl +-120 rad/s): ctrl peak 60-73 rad/s, never clamped. Power <= 1.03x hover everywhere.
Timing (this box ~1.34x slower than the 5070 Ti reference: r18_calib 277 s warm here vs r18's 206 s): warm learn +
  eval-120 x2c_climb 272.2 s, x2c_climb_b 275.4 s, x2c_z04b 274.6, x2c_z04c 274.7 (= x1's 267-273: the climb gain is
  free) -> ~203-205 s at the reference ratio (not measured there). Cold: + ~18 s flight-program compile (x1_rot cold 308
  s). G10 first-call sub-gates 10.6-11.1 s (bar 10). Stages: release 6.7, linearize 12.2, ILC 2 passes 104-107, eval-120
  146-149 s. Eval-240 continuation +147-151 s, verify +41-47 s, dt 508-512 s, CPU 93-135 s. GPU wall this round ~110 min.
Verdict: KEEP --climb-gain reg. At V_op h 0.5 / z 0.3 every bar except the budget on this box (and DOWN at its reference
  ceiling) is met by 2/2 independent learns (x2c_climb, x2c_climb_b); V_op h 0.5 / z 0.4 also meets them in 2/2
  (x2c_z04b, x2c_z04c) but z04b's gain chatters in a climb. Not exported (command.tg still r18).
  x2c_z04c CPU (28 x 240, 95 s): segment verdicts 98.8%, |v_seg| median 8e-5 / max 0.016, every policy flight flies on
  both (state err/scale <= 2.9); the survival mismatch is baseline-arm flight 449 again (GPU lost 107, CPU flew).
  => of the four raised-speed learns, x2c_climb / x2c_climb_b (z 0.3) and x2c_z04c (z 0.4) are clean; x2c_z04b chatters.

## X3' (round 3): carried-lane continuation ILC, per-axis V_op, and the learn inside 240 s (2026-09-30, box 40346)
Code (all new flags default off: golden --quick output identical to the start of the round modulo timings, 9/32,
opt/runs_fast/golden_quick_x3.log; pytest test_hover_command + test_hover_mjx 13 passed, 376 s local):
- `hover.command`: `build_refs(v0=, p0=)` (a carried lane continues its reference), `merge_axis_tables`,
  `extrapolate_tables(u_max=)` (it clipped at the module U_MAX: wrong for --u-max != 0.15), `random_schedules` takes a 4th
  speed (REV), `ref_rate(pol)` (reference rate incl. optional `rate_beats_z`; LiveCommand / dt / sensors use it),
  `LiveCommand.default_speed` REV = v_op[3] when present.
- `hover.launch.release`: the cradle's running mean over a (n, 12) array instead of np.mean over a deque of 901 rows:
  bit-identical release (checked: every state array equal), 10.4 -> 6.1 s locally, 6.6 -> 4.6 s on the box.
- `opt/hover_cmd.py learn`: `--ilc-carry N` (run_carry_call: after the --ilc-iters passes, N more calls that CONTINUE
  the lanes: carry / alive / zeta / reference; good lanes step up --carry-step 1 grid speed, lost or unusable lanes
  re-seed from the release round-robin onto frontier+1..frontier+--carry-ahead 3; hover lanes stay; measured set and
  envelope accumulate), `--carry-gain fly|learn` (fly), `--carry-ok-update` (a cell with ANY usable lane is updated,
  from its usable lanes only), `--carry-dither`, `--climb-refit` (re-solve the climb gain on pass + carry flights),
  `--chatter-max` (du_chatter: UP clean lanes at V_op z..2 V_op z vs FWD/REV/DOWN <= V_op; logged always, rejects
  only when > 0), `--merge-axes`, `--pass-ramp-beats`, `--eval-base-tilt`, `--v-op-rev`, `--rate-beats-z`.
- Scripts: scratch/edge_diag.py (departure mode / sat / |u| / tilt / table entry per cell), du_chatter.py (eval
  per-command |du|), du_ladder.py (ladder per-cell |du|), ref_ceiling_vop.py (reference ceiling at a hypothetical
  V_op, host only), call_cost.py (device seconds per wingbeat by batch content).
Ladder spec used for every "ladder" below: limits_probe --only, 8 flights per cell (1 clean + 7 dither 0.01) x 120
beats, window 60-120, HOVER + FWD/REV 0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90 + UP/DOWN 0.30 0.40 0.50
0.55 0.60 0.65 0.70 (280 flights, ~4 min). Grid tables end at 0.75 (FWD/REV) / 0.90 (UP/DOWN): 0.80 / 0.90 are clamped.
STEP 0 (x3_edge: x2c_climb policy, 50 x 120, 157 s; edge_diag.json, dev 8 deg from the flight's own beats-15-30 attitude):
  FWD 0.65 8/8 (no departure, |u|inf 0.089, tilt max 15.9, pitch 11.5: measured) | FWD 0.70 0/8, YAW first 8/8 at beat
  ~20, sat 0% in the 10 beats before, |u|inf 0.149, tilt 14.8 (pitch 9.7 < the 15 deg map), table entry "measured"
  but only by pass 2's single update, never flown (u_ff 0.066 -> 0.113, yaw ref 5.1 deg): unconverged | FWD 0.75
  0/8 yaw first, sat 25%, extrapolated (u_ff 0.135 = the 0.9 u_max clip, yaw ref 9.4) | REV 0.55 8/8 | REV 0.60 0/8
  ROLL first 7/8 at ~32, sat 40%, tilt 19, extrapolated | REV 0.65 0/8 mixed, sat 30%, extrapolated.
  => branch (a): extrapolated / unconverged edge tables, departing at < 40% saturation -> STEP 1 as written.
STEP 1 runs (flags = x2c_climb's + the listed ones; eval V_op h 0.5 / z 0.3; 512 flights):
  x3_carry1 (--ilc-iters 1 --ilc-carry 1 --merge-axes): pass 46.9 s, carry 42.7 s (<= the pass it replaces), budget
    268.7 s warm (x2c_climb 272-275). 311 lanes carried, 201 re-seeded. Post-ILC FWD 0.70 REV 0.70 UP 0.66 DOWN 0.54.
    eval-240: survival 100% (pushed 100%), follow 97.86%, DOWN 89.5% = its ceiling, |along| p95 0.0139, hold p95
    0.25 cm, tilt p95 4.8. Ladder: FWD <= 0.65 (0.70 0/8: slow YAW divergence 5 -> 30 deg over beats 15-70 while |u|inf
    climbs to the 0.15 clip at ~beat 35; table yaw ref 6 deg), REV <= 0.70 (0.75 1/8, 100% saturated), UP <= 0.50
    (0.55-0.70 6/8, roll first), DOWN <= 0.60 (0.65 6/8). Climb gain from 23 flights (one pass): CHATTERS above UP
    0.36 (carry lanes |du| 0.02-0.24; CPU UP 0.5 |du| 0.023, 1.18 N m median beat-boundary spikes).
  x3_carry2 (--ilc-carry 2): budget 311.6 s. Post-ILC FWD 0.70 REV 0.55 UP 0.72 (merged: UP from the 5 deg map)
    DOWN 0.54; eval-240 as carry1 (100%, 97.86%). Ladder: FWD <= 0.70 (FIRST 8/8 at FWD 0.70), REV <= 0.55 (pass 1
    missed REV 0.60 and its extrapolation had the wrong roll / yaw trend: carried REV 0.60 lanes saturate 53% and
    never give a measurement), UP <= 0.60, DOWN <= 0.60. => the REV / FWD edge moves with the learn's luck.
  x3_umax20 (branch-b arm run anyway, since after the carry the edges saturate: carry1 flags + --u-max 0.20; cold 283 s):
    post-ILC FWD 0.65 REV 0.55 UP 0.90 DOWN 0.60. Ladder FWD <= 0.65, REV <= 0.55 (0.60 lost at only 5-13% saturation),
    UP 0.30-0.70 and DOWN 0.30-0.70 ALL 100%. => a larger input budget buys vertical speed, not horizontal: FWD / REV
    are capped by the yaw (FWD) / roll-pitch (REV) dynamics at speed, not by u_max.
  DEAD END --merge-axes: the per-axis choice rides on the noisy last-call envelope; it never helped (carry1 / f20:
    all axes 15 deg; carry2: UP 5 deg, no gain) and x3_f20b took REV from the 5 deg map -> ladder REV <= 0.50.
  --chatter-max as specified ("UP |du| > 3x hover"): the hover |du| (~0.0007) is so small that 3x rejects the clean
    certified gains too (eval UP/HOVER: x2c_climb 2.8, x2c_climb_b 3.5, x2c_z04c 3.5; chattering z04b 34), so the
    check compares UP (V_op z..2 V_op z, carry lanes) with FWD/REV/DOWN (<= V_op): carry1 35, t24 12, t24p25 23,
    umax20 6.8, carry2 2.8. With one ILC pass most climb gains chatter ABOVE V_op z (in every learn, even x2c_climb's
    ladder shows UP 0.42-0.6 |du| 0.01-0.037); at UP 0.3 (the eval) all are clean (eval UP |du| 0.0020-0.0023).
    Not used to reject in the final config (it would fall back to the hover gain, which cannot climb); --climb-refit
    (pass + carry flights, 34-55 climb flights) is used instead and graded post hoc (ladder / CPU).
STEP 2 timing (warm = XLA cache hit; the first call of a new shape costs ~27 s):
  x3_t24 (24-beat calls, window 12 24, climb-window 8 24): cold 267.4 -> warm x3_t24p25 249.5 s (pass 37.0, carry
  35.4, eval 5 x 24 beats 144.4). --pass-tilt 25 does NOT shorten the ILC calls (37.0 / 35.4 vs 36.4 / 35.9).
  call_cost.py (512 x 24, s per wingbeat): hover no dither 0.98, hover dither 0.01 1.21, the ILC pass inputs 1.47
  (undithered 1.32), eval ~1.2: the vmapped solver pays for every lane's per-beat input step (dither, transitions);
  lost / tumbling lanes cost nothing extra (the baseline arm surviving 100% in x3_umax20 left the eval calls unchanged).
  --eval-base-tilt therefore not used. B 384 vs 512 eval calls: 22-25 s vs 23-26 s per 20 beats.
  x3_f20 / x3_f20b (20-beat calls, window 10 20, climb-window 6 20, --pass-ramp-beats 8, --carry-dither 0,
  --carry-ok-update --climb-refit --merge-axes, V_op FWD 0.6 / REV 0.5 / z 0.3, B 512): 244.4 / 245.1 s warm
  (pass 34.7 / 33.0, carry 28.3 / 29.3, eval 6 x 20 beats 148.8 / 143.7). x3_f20n (no --pass-ramp-beats) 242.6 s.
  Final (x3_g*, release fix + B 384 = "fewer dithered copies"): x3_gb 231.9 s warm, x3_gc 233.6 s warm (PASS <= 240
  on this box), x3_g 253.4 s cold (27.2 s compile of the new 384 x 20 shape). Stages (gb): imports 1.0, release 4.7,
  linearize 12.4, design 0.9, ILC 69.0 (program load 10.7, pass 30.9, carry 26.6), eval 143.7. At the 1.34x box
  factor that is ~173-174 s on the 5070 Ti -- a scaling, NOT a measurement. G10 first-call sub-gates still 10.7-11.1
  s (bar 10), as in every earlier round.
STEP 3 per-axis V_op. Ladders with carry1 flags gave FWD 0.65-0.70 / REV 0.55-0.70 by run, so the rule's
  FWD >= 0.70 & REV >= 0.65 fails; REV lags in 3 of 4 learns -> --v-op-rev. Every learn's ladder holds FWD 0.65 and
  REV 0.55 at 8/8, so V_op FWD 0.60 / REV 0.50 (ladder max - 0.05), z 0.3 (z 0.4 not retried: the one-pass climb
  gains chatter above 0.36). DOWN reference ceiling at V_op 0.6 / 0.5 / 0.3 (ref_ceiling_vop.py, host): 93.7% (at
  V_h 0.5: 89.5%; V_h 0.6 both 0.6: 89.5%, and --rate-beats-z 6 would give 93.2 / 95.9%): >= 90%, so rate-beats-z 6
  was NOT needed (--rate-beats-z added, default off). Final config G = x2c_climb flags + --v-op-h 0.6 --v-op-rev 0.5
  --v-op-z 0.3 --ilc-iters 1 --ilc-carry 1 --carry-ok-update --carry-dither 0 --climb-refit --call-beats 20 --window
  10 20 --climb-window 6 20 --pass-ramp-beats 8 --batch 384 (no merge). eval-240 (288 V_op flights, 72 pushed):
    x3_g : survival 100% (pushed 100%), follow 98.51% (HOVER FWD REV UP 100%, DOWN 92.8% = its reference ceiling on
           this schedule set; 100% of reference-passed segments), |along| p95 0.0179 m/s (3.0% of 0.6), hold p95
           0.24 cm, tilt dev p95 4.7 deg, sat 0%; post-ILC FWD 0.65 REV 0.60 UP 0.48 DOWN 0.54.
    x3_gb: 100% (100%), 98.51% (DOWN 92.8%), |along| p95 0.0128 (2.1%), hold p95 0.12 cm, tilt p95 4.8; post-ILC
           FWD 0.65 REV 0.60 UP 0.48 DOWN 0.48. Every gate passes except the G10 first-call sub-gates.
    x3_gc: 100% (100%), 98.51% (DOWN 92.8%), |along| p95 0.0145, hold p95 0.22 cm, tilt p95 4.8.
    (x3_f20 at B 512, same V_op: 100% / 98.70%, DOWN 93.7%, |along| p95 0.0111, hold 0.13 cm.)
  Ladders: x3_g FWD <= 0.65, REV <= 0.55, UP <= 0.55 (0.60-0.65 7/8, 0.70 5/8), DOWN <= 0.60 (0.65 3/8); x3_gb FWD
    <= 0.65, REV <= 0.55, UP 0.30-0.70 all 8/8, DOWN <= 0.60 (0.65 2/8). FWD >= 0.70 and REV >= 0.60 lost in both.
  Chatter: eval UP |du| 0.0023 (all three); ladder x3_g UP 0.40-0.70 |du| 0.016-0.027 (chatters above V_op z),
    x3_gb 0.007-0.011 (mild); CPU x3_g UP 0.5: |du| 0.0255, 1.41 N m median beat-boundary spikes: FLAGGED.
  dt 0.10 / 0.16 (x3_gb, 64 x 240, 542 s): survival 100% / 100%, follow 98.9% / 98.9%, |along| median 0.0025 / 0.0021.
  CPU (28 x 240, box CPU 12 procs, 127 s each): every policy flight (24 per run) flies on both simulators; segment
    verdicts x3_g 100%, x3_gb 99.4%; |v_seg| median 8e-5 / 6e-5; policy state err/scale <= 2.2 / 1.8. The survival
    mismatch (x3_gb) is baseline-arm flight 338 (CPU lost at 189, GPU flew), as in round 2.
Actuator cost (CPU hover.launch.fly, 60 beats, window 30-60; cpu_torque/x3gb_*.json, x3ga_*, x3c1_*, x3c2_*): smooth
  peak (beat steps >= 5) mNm (x hover) / per-crank RMS mNm / mean power W / median beat-boundary spike mNm (max):
  x3_gb: HOVER 217 / 88-106 / 16.03 / 238; FWD 0.60 (V_op) 261 (1.21x) / 102-117 / 16.15 / 223 (323); FWD 0.65
  (ladder max) 257 (1.19x) / 97-121 / 16.19 / 417 (1215); REV 0.50 (V_op) 217 (1.00x) / 89-106 / 16.03 / 328; REV
  0.55 (max) 225 (1.04x) / 90-108 / 16.06 / 347; UP 0.3 194 (0.90x) / 86-105 / 16.22 / 284; UP 0.70 (max) 249 (1.15x)
  / 100-112 / 16.51 / 546 (1191); DOWN 0.3 269 (1.24x) / 105-123 / 16.00 / 254; DOWN 0.60 (max) 436 (2.01x) / 148-176
  / 16.38 / 328: FLAGGED (> 321 mNm and > 1.5x hover; outside V_op). x3_g UP 0.5 221 / 92-117 / 16.36 / 1406 (1592):
  chatter FLAGGED. x3_carry1: FWD 0.65 264, REV 0.60 235, REV 0.65 263, REV 0.70 309 (1.43x), DOWN 0.5 354 (1.63x),
  UP 0.5 220 with 1.18 N m spikes (chatter); x3_carry2 FWD 0.70 274 (1.26x) / 110-123 / 16.18. No torque limit is
  modelled (velocity servos kv 1, ctrl +-120 rad/s, no forcerange): ctrl peak 60-81 rad/s, never clamped. Power
  <= 1.03x hover everywhere. Every V_op cell costs <= 1.24x the hover peak (DOWN 0.3, unchanged; FWD 0.6 1.21x).
GPU wall this round ~2.6 h (15 learns / probes, 7 ladders, 1 dt check).
Verdict: KEEP config G (x3_gb; reproduced by x3_g and x3_gc). V_op FWD 0.5 -> 0.6 (+20%) with REV 0.5 / z 0.3, every
  quality bar met in 3/3 learns (DOWN at its reference ceiling 92.8%, above 90%), ladder margin 0.05 on both
  horizontal axes in 2/2 ladders, and the learn + eval-120 now fits 240 s warm ON THIS BOX (231.9 / 233.6 s; cold
  253.4 s). The post-ILC / ladder envelope did not move past FWD 0.65 / REV 0.55 reliably: the carry reaches FWD 0.70
  or REV 0.70 in single learns (x3_carry2 / x3_carry1) but not both, and not reproducibly. Not exported
  (designs/hummer/free-hover/command.tg is still r18).
Next: FWD 0.70 is a slow yaw divergence into saturation (not tilt, not u_max: u_max 0.20 does not help) -> a yaw
  integral or a yaw-weighted gain at speed; REV 0.60 needs a measured (not extrapolated) trim: its roll / yaw trend
  reverses above 0.55; the one-pass climb gain chatters above V_op z (a period-2 input cycle): more climb data or a
  du penalty in the DARE; DOWN >= 0.6 costs 2x the hover torque.

## X4' + X5 (round 4): vertical V_op per axis, a scheduled FWD gain, certification (2026-09-30, box 40346)
Code (all new flags default off; golden --quick output identical to golden_quick_x3.log apart from timings, 9/32,
opt/runs_fast/golden_quick_x4.log; pytest test_hover_command + test_hover_mjx 14 passed (13 + the new
test_scheduled_forward_gain_live_equals_evaluated), 377 s local):
- `hover.command`: `gain_weight` (reference FORWARD speed -> [0, 1] between lo / hi), `policy_ka` (Ka, or the 30-wide
  [Ka, Ka_fwd - Ka] of a policy with `Ka_fwd` / `fwd_gain_v`), `blend_ka`, `pad_ka`, `pol_gain_v`; `build_refs(gain_v=)`
  and `RefStream(gain_v=)` add a 15th x_ref column (the weight; needs the zeta gate); `servo_law`, `linear_sim` and
  `mjx_command.law` fly Ka[:, :15] + w Ka[:, 15:] for a 30-wide gain; `LiveCommand` flies it (not the sensor variant).
- `opt/hover_cmd.py learn`: `--v-op-up` (the UP V_op; replaces --v-op-z as UP speed / vertical rate, --v-op-down
  then defaults to --v-op-z), `--fwd-gain reg` (+ --fwd-speeds 0.5 0.7, --fwd-ridge 0.3, --fwd-window, --fwd-blend
  0.4 0.6: fwd_gain(), as climb_gain on the chosen map's FWD pass + carry lanes; the eval / verify fly their own
  chain of the new program shape), `--climb-chatter-max R` (acceptance gate on the gain's own eval-120 V_op flights:
  UP segments' median per-beat |du|inf vs the median of FWD / REV / DOWN; > R rejects the climb gain and re-flies the
  eval with the hover flight gain; eval_du()), verify-pass chatter logged (metrics verify_chatter). dt / cpu / ladder
  (limits_probe) / cpu_torque use policy_ka + pol_ref_args(gain_v). The sensors path ignores the FWD gain.
- Scripts: scratch/gain_probe.py (candidate gains re-derived on the host from a run's own pass / carry flights, flown
  side by side with limits_probe's program), probe_du.py (clean-flight |du| per cell: the dithered copies' |du| is
  the dither, ~0.04), trans_probe.py (scripted two-segment transitions, eval grading), along_vs_ref.py (per-axis
  |along| p95 vs the reference's own), ref_ceiling_vop.py --down / --batch, cpu_torque.py boundary_spike_mNm_max.
ARM A (V_op FWD 0.6 / REV 0.5 / UP 0.5 / DOWN 0.4; --climb-speeds 0.3 0.6). DOWN reference ceiling (ref_ceiling_vop,
  B 384): 92.8% at UP 0.5 / DOWN 0.4 (UP 0.4 / DOWN 0.4: 91.4%; rate-beats-z 6: 95.7%) -> >= 90%, rate-beats-z 6 not
  needed.
  x4_z_a (flags = x3_gb + --v-op-z 0.5 --v-op-down 0.4 --climb-speeds 0.3 0.6): warm 231.0 s; eval-240 survival 100%
  (pushed 100%), follow 98.69% (DOWN 93.66%, others 100%), |along| p95 0.0172, hold p95 0.34 cm, tilt p95 6.5, sat 0%;
  post-ILC FWD 0.65 REV 0.60 UP 0.60 DOWN 0.60. BUT the climb gain CHATTERS at V_op: eval UP |du| 0.0147 vs FWD/REV/DOWN
  0.0027 (ratio 5.5; x3_gb at UP 0.3: 0.93). Its first climb gain was ridge-escalated (0.3 -> 0.9, like x2c_z04b) and
  chattered in the carry (UP 0.11 / MOVE 0.0018 = 61); the refit (ridge 0.3) learned from those lanes. CPU UP 0.5:
  smooth 217 mNm but beat-boundary spikes median 819 / max 1186 mNm (FLAGGED > 1 N m).
  Gain probe 1+2 (x4_probe1/2: 5-8 gains re-derived from x4_z_a's own flights x 9-12 cells x 4 flights x 120 beats,
  ~190 s each): the plant has a period-2 oscillation in the strobed rates at climb >= 0.4 (UP 0.5 clean flight:
  alternating wz 0.20, wy 0.12 rad/s, vx 0.014 m/s; the hover gain shows the same before it departs UP 0.4 at beat 43,
  UP 0.5 at 27). Every climb gain derived from x4_z_a's data (refit, first, r x2, 0.5 blend, ridge 0.9, q_w x2 / x4,
  q_att x2) chatters at UP 0.4-0.6 (clean |du| 0.008-0.03); softer ones lose FWD 0.6 or UP 0.6. Excluding the
  chattering carry lanes from the refit leaves ~pass-only data and re-escalates the ridge (host check): not flown.
  => chatter is learn-to-learn luck of the climb data, not fixed by weights. DEAD END: re-weighting the climb DARE.
  x4_fwd_a (same + --fwd-gain reg; first climb gain at ridge 0.3): eval UP |du| 0.0044 (ratio 1.6: clean).
  --climb-chatter-max 2 then went into the certified config: it would have rejected x4_z_a (5.5 -> hover gain, which
  cannot hold UP >= 0.4: the learn would fail its gates) and passes x4_fwd_a / x4_cert_a / x4_cert_b (1.6 / 1.32 / 1.24).
ARM B (--fwd-gain reg, x4_fwd_a, 1 learn): FWD model regressed from 30 lanes at 0.5-0.7 (ridge 0.3, rho on hover 0.884,
  |dKa|/|Ka| 0.15, yaw rows 0.34). Linear models do not show the divergence: the base gain has rho 0.89-0.95 on every
  regressed FWD model (0.5-0.7 / 0.65-0.75 / 0.7-0.75 / carry-only). eval-240 (V_op as ARM A): 100% / 98.51%, |along|
  p95 0.0107 (budget 259.8 s incl. a 25.3 s compile of the new 384 x 20 x 15 / Ka 30 program: ~234.5 s warm). Ladder
  (x4_fwd_a/limits, 8 x 120): FWD 0.30-0.65 8/8 (no loss), FWD 0.70 4/8 (x3 / cert ladders: 0/8; lost ones yaw 30 deg,
  77% sat), 0.75 / 0.80 0/8. Transitions (x4_fwd_a/trans, 32 x 120, FWD 0.7 for 60 beats then V_op): -> HOVER 22/32,
  -> DOWN 0.4 18/32, -> UP 0.5 23/32 (losses at beats 53-86). Gain probe (2 FWD-gain variants on x4_z_a data): FWD 0.7
  clean flight flies, dithered copies 0/3. => FWD 0.70 not 8/8 in the first learn, so the keep rule (8/8 in 2 of 2)
  cannot be met: DEAD END, --v-op-h stays 0.6. Second learn not run. Yaw integral not built.
CERTIFY (X5): config C4 = x3_gb flags with --v-op-up 0.5 --v-op-down 0.4 --climb-speeds 0.3 0.6 --climb-chatter-max 2
  (V_op FWD 0.6 / REV 0.5 / UP 0.5 / DOWN 0.4). Two learns, eval-240 (288 V_op flights, 72 pushed):
    x4_cert_a (warm): survival 100% (pushed 100%), follow 98.69% (HOVER FWD REV UP 100%, DOWN 93.66% vs its reference
      ceiling 92.8%), |along| p95 FWD 0.0151 / REV 0.0045 / UP 0.0074 / DOWN 0.0695, hold p95 0.25 cm, tilt p95 5.8,
      sat 0%; chatter gate 1.32 PASS; post-ILC FWD 0.65 REV 0.60 UP 0.54 DOWN 0.60.
    x4_cert_b (cold): 100% (100%), 98.63% (REV 99.7%, DOWN 93.66%), |along| p95 FWD 0.0135 / REV 0.0043 / UP 0.0064 /
      DOWN 0.0695, hold p95 0.25 cm, tilt p95 5.9; chatter gate 1.24 PASS; post-ILC FWD 0.65 REV 0.60 UP 0.60 DOWN 0.60.
    DOWN |along| p95 0.0695 is ABOVE 15% of 0.4 (0.060) on all segments; the reference itself scores 0.0725 there
      (along_vs_ref.py; x3_gb at DOWN 0.3: 0.068 vs its 0.045 bar, same cause), and on the reference-passed segments
      the flights give 0.0204 / 0.0205: passes only graded against the reference.
    dt 0.10 / 0.16 (64 x 240): a 100% / 100% (follow 99.5 / 99.5%); b 100% / 100% (99.5 / 99.5%). ~510 s each.
    CPU (28 x 240, box CPU 12 procs, ~81 s each): identical survival, segment verdicts 100% / 100%, |v_seg| median
      1e-4 / 1.4e-4, max 0.011 / 0.008; policy-flight state err/scale <= 13.1 / 9.0 (x3: <= 2.2; worst are no-push
      flights 1 and 6), baseline arm 40-175.
    Ladders (8 x 120, both learns identical): FWD <= 0.65 (0.70+ 0/8, yaw), REV <= 0.55 (0.60+ 0/8), UP 0.30-0.70
      all 8/8, DOWN <= 0.60 (0.65 6/8). Clean-flight |du|: UP 0.4 0.007-0.008 and UP 0.6-0.7 0.005-0.013 chatter, UP
      0.5-0.55 clean (0.0007-0.0016); FWD 0.65 0.008; everything else <= 0.004.
    Timing: x4_cert_a warm 232.2 s (release 4.7, linearize 12.3, ILC 69.9, eval 143.2) PASS <= 240 on this box; x4_cert_b
      cold (empty JAX cache) 270.2 s (linearize compile 26.9, flight program 29.6). At the 1.34x box factor that is
      ~173 s warm / ~202 s cold on the 5070 Ti: a scaling, NOT a measurement. G10 first-call sub-gates 10.7-11.1 s warm.
Actuator cost (CPU hover.launch.fly, 60 beats, window 30-60, x4_cert_a; opt/runs_fast/x4_torque, cpu_torque/x4ca_*):
  smooth peak mNm (x hover) / per-crank RMS mNm / power W / beat-boundary spike median (max) mNm / clean |du|:
  HOVER 216 / 89-106 / 16.03 / 239 (244) / 0.0011
  FWD 0.60 (V_op) 260 (1.20x) / 103-118 / 16.16 / 233 (274); FWD 0.65 (max) 259 (1.20x) / 97-121 / 16.19 / 333 (1120:
    FLAGGED > 1 N m)
  REV 0.50 (V_op) 216 (1.00x) / 89-105 / 16.02 / 328 (358); REV 0.55 (max) 223 (1.03x) / 90-107 / 16.05 / 352 (566)
  UP 0.50 (V_op) 206 (0.95x) / 92-104 / 16.39 / 314 (350) / 0.0020; UP 0.40 197 (0.91x) / 88-107 / 16.32 / 632 (1025:
    FLAGGED, chatter below V_op); UP 0.60 225 (1.04x) / 16.49 / 318 (500); UP 0.70 (max) 256 (1.18x) / 100-115 / 16.51 /
    845 (1969: FLAGGED)
  DOWN 0.40 (V_op) 295 (1.36x, < 321 mNm) / 112-130 / 16.01 / 235 (253); DOWN 0.60 (max) 440 (2.03x) / 150-180 / 16.38 /
    456 (526): FLAGGED (> 321 mNm, > 1.5x hover; outside V_op)
  x4_z_a (chattering learn): UP 0.5 217 / 16.37 W / 819 (1186: FLAGGED); UP 0.4 198 / 645 (862); DOWN 0.4 294; FWD 0.6
    261; REV 0.5 217. No torque limit is modelled (velocity servos kv 1, ctrl +-120 rad/s, no forcerange): ctrl peak
    60-82 rad/s, never clamped. Power <= 1.03x hover everywhere. Every V_op cell <= 1.36x (DOWN 0.4).
CEM: not run (no gate missed by <= 3 points; DOWN |along| is a reference-shaping effect, not a controller miss).
GPU wall this round ~75 min (4 learns, 2 gain probes, 3 ladders, 1 transition probe, 2 dt checks).
Verdict: KEEP C4 (x4_cert_a, reproduced by x4_cert_b; also x4_z_a / x4_fwd_a at the same V_op). V_op UP 0.3 -> 0.5
  (+67%) and DOWN 0.3 -> 0.4 (+33%) with FWD 0.6 / REV 0.5 unchanged; every gate met in 2/2 learns (DOWN follow
  outright, DOWN |along| only against its reference), warm budget 232.2 s on this box. Flags: chatter at UP 0.4 (1.03
  N m) and above UP 0.6 (1.97 N m at 0.7), FWD 0.65 spike 1.12 N m, DOWN 0.6 2.03x. The climb gain's chatter is learn
  luck (1 of 4 learns at this V_op chattered at V_op: x4_z_a); --climb-chatter-max 2 rejects such a learn (it then
  fails its UP gates) rather than fixing it. FWD 0.70 via a scheduled FWD gain: dead end (4/8). Not exported
  (designs/hummer/free-hover/command.tg is still r18).
Next: the period-2 rate oscillation at climb 0.4 / >= 0.6 is plant-side (the hover gain shows it too): a Nyquist-aware
  law (a du penalty / u_{k-1} state in the DARE, or a two-beat average on the rate feedback) is the structural fix;
  FWD 0.70 needs something other than a linear gain regressed at 0.5-0.7 (the regressed models all look stable):
  a yaw integral, or a measured 0.7 trim (lanes seeded there); --rate-beats-z 6 would lift the DOWN reference ceiling
  (95.7%) and its |along| tail.

## X7 (round 5): augmented servo (input-rate state, yaw integral), --rate-beats-z 6, DOWN 0.45, g pilot (2026-09-30, box 40346)
Code (all new flags default off; r18 / command.tg / x4_cert paths unchanged: a plain gain keeps its 15-wide Ka and 3-wide
carry and the old code path verbatim; golden --quick output identical to golden_quick_x4.log apart from timings, 9/32,
opt/runs_fast/golden_quick_x7.log; pytest test_hover_command + test_hover_mjx 15 passed, 383 s local, incl. the new
test_augmented_servo_live_equals_evaluated):
- `hover.command`: `aug_layout` / `servo_nz` (carry width <-> layout: + yaw integral, + d_{k-1} = the previous wingbeat's
  delivered feedback u - u_ff - dither), `servo_aug_model`, `servo_gains_aug` (cross-term DARE: stage cost u'Ru +
  du'(r_d R)du -> Q_dd = R_d, R + R_d, S = -R_d; yaw integral weight q_yi), `servo_gains(q_yi=, r_d=)`, `servo_closed`
  for augmented gains; `servo_law` / `linear_sim` / `CommandPolicy` / `LiveCommand` carry the wider state;
  `blend_ka(nz=)` (the scheduled-gain split now at 12 + nz). `hover.mjx_command.law` the same on device;
  `release_carry(fl, B, nz=3)`.
- `opt/hover_cmd.py learn`: `--du-penalty R_d` (x R), `--yaw-int q_yi`: the flight / climb / FWD gains are augmented,
  the ILC learning gain stays plain (zero-padded to the program shape); logs the period-2 |z| of the flight gain on the
  hover model; policy extras du_penalty / yaw_int. dt / cpu / eval / ladders size the carry by servo_nz.
- Scripts: scratch/aug_screen.py (STEP 1 linear screen), gain_probe.py `<variant>@d<r_d>y<q_yi>` (augmented
  re-derived gains, all variants zero-padded to one program), limits_probe.py (carry width), probe_table.py (clean-flight
  |du| / |dw| per cell), p2_predict.py (is the climb gain's period-2 mode a chatter predictor?), g_pilot.py (CPU g pilot),
  x7_queue.sh (the GPU queue; VDOWN / EXTRA env).
STEP 1 (linear screen, aug_screen.py, host; flight weights, q_int 50). Period-2 mode = largest |z| with Re z < -0.2:
  hover flight gain on the hover model: rho 0.866 at every r_d (p2 0 -> 0.22-0.28: the d state's own pole);
  the CLIMB gain on the HOVER model has a lightly damped period-2 mode: x4_cert_a 0.825, x4_z_a 0.895 (= its rho);
  r_d 0.5 / 1 / 3 damp it to 0.678 / 0.591 / 0.443 (cert_a) and 0.745 / 0.657 / 0.496 (z_a), rho on hover <= 0.883;
  but the du penalty makes the HOVER gain less robust on the regressed climb model: rho 0.960 -> 1.008 / 1.032 / 1.078
  (cert_a), 1.001 -> 1.048 / 1.074 / 1.126 (z_a). Yaw integral (q_yi 25-1600): rho unchanged (0.866) everywhere.
  Chatter predictor? (p2_predict.py, refit climb gain of each learn, p2 on hover vs eval UP/MOVE): cert_a 0.825 / 1.32,
  cert_b 0.802 / 1.24, rz6a 0.916 / 1.34, rz6b 0.883 / 0.89, rz6c 0.700 / 4.14 (chattered), z_a 0.895 / 5.5: NOT a
  predictor (the chattering rz6c has the most damped mode). As round 4 found, no linear model sees the chatter.
STEP 2 (gain probes, gain_probe.py, 8 flights (1 clean + 7 dither 0.01) x 120 beats per cell; |du| = clean flight,
  median per-beat |du|inf beats 30-120; |dw| = its strobed body-rate alternation):
  x7_probeA (x4_z_a, the chattering learn; 384 flights, 193 s):
    pol (its own gain)  UP 0.4 7/8 0.0097 | UP 0.6 6/8 0.0246 | UP 0.7 6/8 0.0502 (dw 1.63 rad/s) | FWD 0.6-0.65 8/8 | FWD 0.70 / 0.75 0/8 | REV 0.6 0/8
    refit@d0.5          UP 0.4-0.7 8/8, |du| 0.0063 / 0.0015 / 0.0087 / 0.0091
    refit@d1            UP 0.7 5/8 (0.0104)
    refit@d3            UP 0.4-0.7 8/8, 0.0029 / 0.0037 / 0.0067 / 0.0078; FWD 0.70 1/8; REV 0.6 0/8
    -> the du penalty cuts UP >= 0.6 input chatter 3-6x and saves the lost UP 0.6 / 0.7 flights, but never down to the
       FWD / REV level (0.0012-0.0017), and the strobed-rate alternation stays (dw 0.1-0.25 rad/s; at UP 0.5 d1 / d3
       RAISE it 0.02 -> 0.25): the period-2 oscillation is the plant's, the penalty only stops the input amplifying it.
  x7_probeB (x4_cert_a; 480 flights): pol UP 0.7 0.0115; d1 0.0149; d3 UP 0.6 0.0009 (pol 0.0053) but UP 0.7 0.0113;
    REV 0.6 8/8 with d1 / d3 (pol 0/8); FWD 0.70 0/8 with every variant incl. the yaw integral (d1y100, d1y400: lost at
    beats 42-63, as pol 35-45). x7_probeC (d3 / d6 / d10 on both runs): no gain over d3 at UP 0.6 / 0.7 (0.0042-0.0129),
    UP 0.7 4-5/8 on z_a, REV 0.6 0/8 again with d3 (8/8 in probe B: the edge cells flip with the dither draw).
  Chosen R_d = 3 (best clean |du| with no survival loss in probes A / B). Rule outcomes: FWD 0.70 is 0/8 -> V_op FWD
  stays 0.6; UP 0.7 is not clean (|du| 0.008-0.011) -> V_op UP stays 0.5. Yaw integral: DEAD END for FWD 0.70.
STEP 3-4 learns (x4_cert_a flags + the listed changes; eval-240, 288 V_op flights, 72 pushed):
  x7_a (--du-penalty 3 --rate-beats-z 6): survival 100%, follow 96.0% (REV 89.5% FAIL, UP 94.8%), hold p95 1.28 cm,
    cross p95 0.040, tilt dev p95 13.5 deg; post-ILC DOWN 0.24; carry UP chatter 0.0465 (21x MOVE); budget 246.5 s
    (incl. the new 384 x 20 / Ka 39 compile). The penalised gains are slower and less robust (hover gain rho 1.13-1.15
    on the regressed climb model): NOT kept; its queued b / c learns were cancelled.
  x7_d05a / x7_d05b (--du-penalty 0.5 --rate-beats-z 6): survival 99.3% (pushed 97.2%: flights 227 / 272, the same side
    + pitch-rate push during REV after DOWN) / 100%; follow 98.51 / 99.11%; hold p95 0.43 / 0.50 cm; tilt p95 7.7 / 9.2;
    eval UP |du| 0.0049 / 0.0019 (ratio 1.86 / 0.81; x4_cert 1.24-1.32); carry chatter 0.031 / 0.044 (not lower than
    plain). 1 of 2 misses the survival bar: NOT kept. => --du-penalty: DEAD END as the flight gain (r_d 3 costs tracking,
    r_d 0.5 costs push robustness and does not lower V_op chatter).
  x7_probeD (x7_rz6c's data, the chattering plain learn): refit / @d0.5 / @d1 / @d3 all fly UP 0.4-0.6, FWD 0.6,
    REV 0.5, DOWN 0.4 at 8/8; UP 0.5 clean |du| 0.0077 / 0.0074 / 0.0071 / 0.0068: the penalty does NOT cure its
    chatter at V_op. And the chatter gate had REJECTED that refit, which flies 8/8.
  --rate-beats-z 6 alone (plain gain; V_op FWD 0.6 / REV 0.5 / UP 0.5 / DOWN 0.4): DOWN reference ceiling 92.8% ->
    95.7% (ref_ceiling_vop, B 384); reference's own DOWN |along| p95 0.0725 -> 0.0511.
    x7_rz6a: 100% (pushed 100%), follow 99.11% (DOWN 95.7% = ceiling), |along| p95 FWD 0.0132 REV 0.0047 UP 0.0079
      DOWN 0.0479 on ALL segments (bar 0.060: PASSES outright, x4_cert 0.0695), hold p95 0.35 cm, tilt p95 7.0, budget
      235.8 s warm; CPU (28 x 240): policy flights identical survival, verdicts 99.4% (the one mismatch is baseline
      flight 338, as rounds 3-4), policy err/scale max 28 (median 2.1); dt 0.10 / 0.16 100% / 100%.
    x7_rz6b: 100%, 99.0% (UP 99.4%), DOWN |along| 0.0478, hold 0.36 cm, budget 240.1 s (first climb gain ridge-
      escalated to 0.9, refit at 0.3, eval UP ratio 0.89); dt 0.10 100%, dt 0.16 98.4% (1 of 64 lost: below 99%).
    x7_rz6c (COLD, empty JAX cache): the chatter gate REJECTED the climb gain (ratio 4.14), the hover-gain fallback
      cannot hold UP: survival 28.8%, follow 50.0%; budget 433.1 s (linearize compile 26.8 s, flight program 29.4 s,
      +150 s eval re-fly).
  DOWN 0.45 (--rate-beats-z 6 --v-op-down 0.45; torque cap check first: CPU DOWN 0.45 on rz6a 318.8 mNm = 1.47x hover,
    just under 321 / 1.5x; DOWN reference ceiling 95.7%, reference's own DOWN |along| p95 0.0667 vs the 0.0675 bar):
    x7_d45a / x7_d45b (gate --climb-chatter-max 2): gate REJECT (ratio 2.33 / 9.62; d45b's climb gains both ridge 0.9)
      -> hover fallback -> survival 27.4% / 23.3%. The x4 gate turns a chattering-but-flying climb gain into a lost
      learn: 3 of the 5 plain learns with the gate this round (rz6c, d45a, d45b) failed this way (round 4: 1 of 4 would
      have). The fallback is the failure, not the chatter (probe D).
    Gate log-only (--climb-chatter-max 1000: the ratio is still measured and logged; never rejects):
    x7_d45c: 100% (pushed 100%), follow 99.11% (DOWN 95.7% = ceiling), |along| p95 FWD 0.0147 REV 0.0054 UP 0.0060
      DOWN 0.0670 all segments (bar 0.0675; ref's own 0.0667; ref-passed 0.030), hold 0.32 cm, cross 0.016, tilt 7.1,
      sat 0.6%; eval UP ratio 1.50; budget 230.9 s warm; CPU: identical policy survival, verdicts 99.4% (baseline 338),
      policy err/scale max 13.4.
    x7_d45d: 100%, 98.87% (DOWN 94.5% vs ceiling 95.7%), DOWN |along| 0.0682 all segments (0.0007 over the bar; ref
      0.0667; ref-passed 0.0296 -> passes graded against the reference), hold 0.44 cm, tilt 8.2; ratio 0.69; budget
      232.8 s warm; CPU: identical survival, verdicts 100%.
    x7_d45e (COLD): 100%, 98.87% (DOWN 94.5%), DOWN |along| 0.0688 (ref-graded), hold 0.38 cm, cross 0.025, tilt 8.5;
      ratio 1.24; budget 272.3 s cold (linearize compile 26.9, flight program 29.4 s).
    dt 0.10 / 0.16 (64 x 240): d45c 100% / 100% (follow 100 / 100%); d45d 100% / 100% (99.7 / 99.7%).
    Ladders (8 x 120; rz6a / rz6b / d45c / d45d): FWD <= 0.65 in all (0.70 0/8, yaw ~10 deg, lost ~beat 45); REV <= 0.55
      in all (0.60+ 0/8); UP 0.70 7/8 / 5/8 / 8/8 / 5/8 (UP 0.65 8/8, 6/8, 8/8, 7/8); DOWN <= 0.60 in all (0.65 1-6/8).
    Chatter at V_op across the 7 plain learns of this round (eval UP / MOVE): rz6a 1.34, rz6b 0.89, rz6c 4.14, d45a 2.33,
      d45b 9.62, d45c 1.50, d45d 0.69, d45e 1.24 -> 3 of 8 above 2 (with round 4: 4 of 12).
Actuator cost (CPU hover.launch.fly, 60 beats, window 30-60, contacts off; opt/runs_fast/x7_torque, cpu_torque/x7*):
  smooth peak mNm (x hover) / per-crank RMS mNm / power W / beat-boundary spike median (max) mNm / clean |du|:
  x7_d45c: HOVER 216.3 / 88-106 / 16.03 / 235 (249) / 0.0010
    FWD 0.60 (V_op) 259.2 (1.20x) / 103-118 / 16.17 / 230 (253); FWD 0.65 (max) 262.4 (1.21x) / 16.19 / 383 (1055 FLAGGED)
    REV 0.50 (V_op) 216.1 (1.00x) / 16.02 / 328 (448); REV 0.55 (max) 223.6 (1.03x) / 16.05 / 364 (530)
    UP 0.40 195.9 (0.91x) / 16.31 / 621 (936) / 0.0076 (chatter below V_op, as x4); UP 0.50 (V_op) 215.2 (0.99x) / 16.38 /
      308 (380); UP 0.70 (max) 256.5 (1.19x) / 16.47 / 636 (1096 FLAGGED) / 0.0103
    DOWN 0.45 (NEW V_op) 320.6 (1.48x) / 118-135 / 16.04 / 258 (347): FLAGGED at the limit (<= 321 mNm and < 1.5x, by
      0.4 mNm); DOWN 0.50 352.5 (1.63x); DOWN 0.60 (max) 440.4 (2.04x) FLAGGED (outside V_op)
  x7_rz6a: HOVER 216.7; DOWN 0.40 293.7 (1.36x); DOWN 0.45 318.8 (1.47x); DOWN 0.5 349.8 (1.61x); DOWN 0.6 434.2
    (2.00x); FWD 0.6 263.2 (1.21x); FWD 0.65 259.7 / spike 455 (953); REV 0.5 215.5; REV 0.55 221.3; UP 0.4 196.6 / spike
    897 (1146 FLAGGED); UP 0.5 212.2 / 315 (355); UP 0.7 240.0 / 2321 (3382 FLAGGED: its UP 0.7 chatters, |du| 0.049).
  No torque limit is modelled (velocity servos kv 1, ctrl +-120 rad/s, no forcerange): ctrl peak 60-82 rad/s, never
  clamped. Power <= 1.03x hover everywhere.
g PILOT (box CPU, g_pilot.py, x4_cert_a tables / gain, 8 seeds (seed 0 clean, 7 dither 0.005) x 60 beats, 144 flights,
  144 s on 12 procs). B_g (one beat, n0 901 -/+ 3 steps, quantum 0.111% per step): vel (-0.38, -0.02, +1.39) m/s, w
  (4.2, -7.9, 1.6) rad/s per unit g (L/R-asymmetric). Redistribution LP (same one-beat effect on att/vel/w rows,
  |g| <= 5%): DOWN 0.4 / 0.5 / 0.6 table |u_ff|inf 0.078 / 0.109 / 0.135 -> 0.043 / 0.066 / 0.101 at g -5.0% (n 947); UP
  0.6 / 0.7 0.084 / 0.112 -> 0.033 / 0.042 at g +5.0% (n 857). Arms (median of 8, all 8/8 survive):
    h24 (as is)   HOVER 230.7 mNm (dithered median; clean 216.5) | DOWN 0.4 309 (1.34x) 16.01 W | DOWN 0.5 365 (1.58x)
                  16.11 W | DOWN 0.6 453 (1.96x) | UP 0.6 237 16.46 W | UP 0.7 263 16.50 W
    gff (g ff)    DOWN 0.4 288 (1.25x) 14.62 W (0.91x) | DOWN 0.5 282 (1.22x) 14.67 W (0.92x) | DOWN 0.6 288 (1.25x) 14.77 W |
                  UP 0.6 246 17.72 W (1.106x) | UP 0.7 251 17.78 W (1.109x)
    g25 (+g fb)   DOWN 0.5 320 (1.39x) 15.47 W | DOWN 0.6 360 (1.56x) | UP 0.6 236 17.60 W (1.098x) | UP 0.7 248 1.109x
    clean flights: DOWN 0.5 h24 353 -> gff 270 mNm; gff UP 0.6 / 0.7 beat-boundary spikes 1.16 / 1.52 N m (h24 0.34 /
    0.75). (spike medians of the dithered copies are ~1.2 N m in every arm: the dither.)
  => g PASSES for DOWN (0.5: 1.58x -> 1.22x hover smooth peak, power 0.92x hover; DOWN 0.6 at 1.25x), FAILS for UP (power
  1.106-1.109x > 1.10, torque unchanged, spikes up). The device port (g as a per-beat input on MJX) was NOT built this
  round: hover.mjx_flight's wingbeat is a static n_step scan, so a quantized g needs a variable-length beat (a masked
  max-length scan or one program per n); that is the next round's F3 work, DOWN only.
Timing (this box ~1.34x slower than the 5070 Ti reference; not measured on the reference):
  warm learn + eval-120: d45c 230.9, d45d 232.8, rz6a 235.8, rz6b 240.1 (first climb gain ridge-escalated), d05a/b
  230.5 / 233.9 s -> ~172-179 s on the 5070 Ti by the factor; cold d45e 272.3 s (~203 s); learns whose chatter gate
  re-flies the eval: 384-433 s. G10 first-call sub-gates 10.4-11.4 s warm (as every round). Untimed: eval-240 145-157 s,
  verify 27-31 s, dt ~510 s each, ladders 158-180 s, CPU checks 126 s, gain probes 150-230 s.
  GPU wall this round ~2.7 h (11 learns, 5 probes, 4 ladders, 5 dt checks).
Checks: pytest test_hover_command + test_hover_mjx 15 passed; golden --quick identical to golden_quick_x4.log apart
  from timings (9/32; opt/runs_fast/golden_quick_x7.log).
Verdict: KEEP config C5 = x4_cert_a flags + --rate-beats-z 6 --v-op-down 0.45 --climb-chatter-max 1000 (gate log-only),
  i.e. V_op FWD 0.6 / REV 0.5 / UP 0.5 / DOWN 0.4 -> 0.45 (+12.5%): x7_d45c, reproduced by x7_d45d and the cold x7_d45e
  (3/3: survival 100% incl. pushed, follow 98.9-99.1%, every command >= 94.5%, hold p95 <= 0.44 cm, dt 100%, CPU
  verdicts 99.4-100%, warm 230.9-232.8 s). DOWN is graded against its reference: follow 94.5-95.7% vs ceiling 95.7%,
  |along| p95 0.0670-0.0688 vs the 0.0675 bar and the reference's own 0.0667 (ref-passed 0.030). FLAGS: DOWN 0.45 costs
  1.48x hover smooth peak (320.6 mNm, 0.4 mNm under 321); chatter spikes ~1 N m at UP 0.4 and UP 0.7 remain; the gate
  is off, so a chattering learn (3 of 8 this round) now keeps flying with ~0.8-1.2 N m spikes instead of being lost.
  Dead ends: --du-penalty (all r_d), --yaw-int (FWD 0.70 still 0/8), the linear period-2 predictor, the chatter gate's
  hover fallback. Also usable at DOWN 0.4: x7_rz6a / rz6b (rz6b dt 0.16 98.4%). Not exported (command.tg is still r18).
Next: FWD 0.70 / REV 0.60 need something other than a linear gain (measured trims at the edge); the climb chatter is the
  plant's period-2 oscillation (neither the penalty nor weights cure it): a per-beat-alternation-aware feedback (average
  the rate over two beats) or accept and bound the spikes; g feedforward (F3) for DOWN 0.5-0.6 once the device can fly
  variable-length beats.

## X8 (round 6): measured edge trims (moving cradle) for FWD / REV; crank-speed g feedforward for DOWN (2026-10-01, box 40346)
Base: the certified x7_d45c config (V_op FWD 0.6 / REV 0.5 / UP 0.5 / DOWN 0.45, --rate-beats-z 6, --climb-chatter-max 1000).
All new flags default off; r18 / command.tg / x7 paths unchanged (golden --quick identical to golden_quick_x4.log apart from
timings, 9/32, opt/runs_fast/golden_quick_x8.log; pytest test_hover_command + test_hover_mjx 16 passed, 459 s local, incl. the
new test_crank_speed_g_device_and_live). Calibration: the x7_d45c flags rerun today (x8_calib) = 233.3 s warm (x7: 230.9-232.8):
the box speed is unchanged.
Code:
- `hover.mjx_cradle`: `Cradle.orbits` (moving-cradle wrench + the wingbeat-boundary state at the end of the run),
  `trim_at_speed` (per cell x = [24 depths, tilt_x, tilt_y, yaw]; batched central-difference J (1 + 2x27 runs per cell, one
  device call per iteration, fixed batch shape); LP step min |u + du|inf s.t. w + J dx = 0 (slack 1e3), |du| <= trust,
  |d ang| <= 5 deg, |yaw| <= 10 deg; the first --min-iters steps minimise |u|inf, later ones are plain Newton steps (min
  |du|inf + 0.3 |d ang|_1) so the trim converges onto the manifold instead of sliding along it), `orbit_entries` (table
  entries att / dv / w of a trim relative to the hover cradle = the flight's q_ref / x_orbit), `attitude3`.
- `hover.command`: crank-speed feedforward (opt-in, a table key "g"): `lookup_g`, `beat_steps` / `g_of_steps` (g quantized
  to integer wingbeat lengths, g = ln(n0 / n)), `g_beats`, `retime_refs` (p_ref re-integrated over the longer wingbeats),
  `beat_times`, `g_table` (DOWN g(s) = g_down min(1, s / s_full), DOWN LP entries re-solved with B_g g); `segment_metrics` /
  `cell_pass` / `cell_window_stats` accept per-beat durations; `CommandPolicy(g=)` returns [g_k, u]; `RefStream(g_n0=)` /
  `LiveCommand` (policy key g_n0) fly the same g and p_ref; save/load_policy carry tab{m}_g.
- `hover.mjx_command.beat_var`: one wingbeat of n (traced) steps, crank speed a0 e^g0 n0 / n, phase phi0 + 2 pi j / n, a
  scan to n_max with the steps j >= n masked (mjx_flight.beat untouched); `fly_cmd` / `batch_cmd_flyer` / `Chain(n_max=)`
  take per-beat lengths `nbeat` (B, N) when n_max is set (one more int input; else the old program verbatim).
- `opt/hover_cmd.py learn`: `--ff-trims <trims.npz>` (+ `--ff-trims-from 0.70 0.60`, `--ff-trims-keep`), `--g-down G`
  (+ `--g-down-full 0.4`, `--g-fd-steps 3`; B_g from a CPU one-beat central difference at the release, n0 -/+ 3 steps, 0.5 s);
  every pass / carry / eval / continuation / verify flies its per-beat lengths and is scored on its per-beat durations; the
  eval's baseline arm flies n0; traces carry nbeat; `dt` re-quantizes g for its own n0; `cpu` flies CommandPolicy(g=).
- Scripts (opt/runs_fast/scratch): edge_trims.py (the trim continuation), mcradle_trim.py (CPU reference of the trims:
  wrench check + cradle crank torque), make_trim_policy.py (splice trims into a policy for the falsification ladder),
  x8_learn.sh / x8_checks.sh (GPU queue: learn; dt / cpu / ladder), x8_summary.py, x8_losses.py (lost flights: push and
  segment); limits_probe.py (variable-length beats: per-beat lengths, per-flight elapsed time), cpu_torque.py (g policies:
  variable-length beat windows), along_vs_ref.py (per-beat durations).

ARM A (a) TRIMS (x8_trims: edge_trims.py, x64 device, 24-beat moving cradle, mean of the last 8; trust 0.02, 2 min-|u| steps
  then Newton, <= 6 iterations; FWD and REV advance together one grid speed per stage, stage 0 seeded from x7_d45c's measured
  15-deg entries (FWD 0.60, REV 0.50), stage 1 from stage 0 + the table's slope, later stages by linear extrapolation of the
  two previous trims; 6 stages, 1429 s = ~40 s per 111-run call). Every cell converged: |F| <= 0.47 mN (0.21% W),
  |M| <= 0.098 mNm (a ~0.1 mNm measurement floor on the FWD 0.60 / 0.70 cells, 7 iterations); "residual <= 2% W" holds with
  100x margin. |u|inf / tilt (x, y) / yaw [deg]:
    FWD 0.60 0.033 (1.9, 12.2) -6.9 | 0.65 0.039 (2.1, 12.5) -5.3 | 0.70 0.064 (2.4, 12.9) -3.7 | 0.75 0.081 (2.3, 13.7) -2.1 |
        0.80 0.099 (2.2, 14.4) -0.6 | 0.85 0.090 (2.1, 15.3) +1.0
    REV 0.50 0.028 (-0.5, -13.6) -3.8 | 0.55 0.020 (0.3, -15.5) -3.1 | 0.60 0.020 (0.7, -17.0) -2.1 | 0.65 0.023 (0.7, -18.4)
        -1.2 | 0.70 0.021 (0.7, -19.8) -1.5 | 0.75 0.022 (0.7, -21.3) +0.8
  The ILC-learned x7 entries at the same speeds use 2x the input (FWD 0.60 0.065, REV 0.50 0.058) and the extrapolated ones
  above the frontier were garbage (FWD 0.70 0.118 with +4 deg yaw, 0.75 0.135; REV 0.60-0.75 0.094-0.135).
  CPU check (mcradle_trim.py, box CPU, 16 s): device vs CPU wrench at the SEED points of FWD 0.70 / 0.80 / REV 0.60 (large
  wrenches, 1-5 mN / 0.1-0.7 mNm): scaled-norm rel err 6.0e-11 / 4.8e-10 / 4.6e-10 (PASS 2%); at all 12 converged trims
  |dev - CPU| <= 1.5e-10 (scaled). CPU cradle crank torque (smooth peak / RMS per crank / power): hover 217.3 / 89-106 / 16.03
  W; FWD 0.70 239.7 (1.10x) / 101-135 / 16.25 W; FWD 0.75 243.7 (1.12x); FWD 0.85 257.3 (1.18x) / 110-160 / 16.32 W; REV
  0.50-0.75 207-220 (0.95-1.01x) / 86-109 / 15.82-16.01 W.
ARM A (b) FALSIFY (x8_trims/ladder: limits_probe --only, x7_d45c's servo with only the entries at FWD >= 0.70 / REV >= 0.60
  of the 15-deg table replaced; 8 flights (1 clean + 7 dither 0.01) x 120 beats, window 60-120; 88 flights, 159 s):
  abs (entries = the trims): FWD 0.70 8/8, 0.75 8/8, REV 0.60 8/8, 0.65 8/8, all 0% saturated (FWD 0.75 2%), |u|inf 0.087 /
    0.113 / 0.046 / 0.048, yaw max 4.3-5.0 deg; control cells FWD 0.65 / REV 0.55 (x7 entries) 8/8.
  delta (x7 entry at the last measured speed + trim(s) - trim(anchor)): FWD 0.70 / 0.75 / REV 0.60 / 0.65 all 8/8, |u|inf
    0.116 / 0.124 / 0.060 / 0.062, yaw max 4.8-7.9.
  (x7_d45c with its own tables: FWD 0.70 / 0.75 and REV 0.60 / 0.65 0/8.) => PASSES: FWD 0.70 no longer departs yaw-first
  (yaw max 5 deg over 120 beats). The FWD 0.70 / REV 0.60 edge of rounds 2-5 was the TABLE (extrapolated entries), not the
  dynamics; physical trims at the edge need half the input of the ILC-learned entries below it.
ARM A (c) LEARNS (x7_d45c flags + --ff-trims opt/runs_fast/x8_trims/trims.npz --v-op-h 0.7 --v-op-rev 0.6; the trims are
  precomputed outside learn: 1429 s device + 16 s CPU, NOT in the timed budget):
  DEAD END first: x8_trim_a_frontier (trims spliced only above the "measured frontier" after each ILC update): the loose
    lane_good rule had marked FWD 0.70 / 0.75 measured in pass 1, so no FWD trim was spliced; eval at FWD 0.7: survival 29.5%,
    FWD follow 16%. => --ff-trims now splices into the LP BASE tables before pass 1 at FWD >= 0.70 / REV >= 0.60 (the ILC
    refines them and extrapolates relative to them); --ff-trims-keep re-splices them after every update.
  refine (default): x8_trim_c / x8_trim_d; keep (--ff-trims-keep): x8_trim_k / x8_trim_k2 (COLD, empty JAX cache).
  Eval-240 (288 V_op flights, 72 pushed), c / d / k / k2:
    survival 100 / 100 / 100 / 100% (pushed 100% in all); follow 98.69% in all; HOVER FWD REV UP 100% in all; DOWN 93.66% in all
    = its reference ceiling at this V_op (ref_ceiling_vop: 93.7%; the descend shaping waits longer after FWD 0.7);
    |along| p95 FWD 0.0084 / 0.0071 / 0.0065 / 0.0068 (bar 0.105), REV 0.0094 / 0.0076 / 0.0032 / 0.0043 (bar 0.090), UP
    0.0059 / 0.0064 / 0.0055 / 0.0065 (bar 0.075); DOWN 0.088 all segments vs the reference's own 0.089, ref-passed
    0.031-0.032 (bar 0.0675: passes only against the reference, as x7); hold p95 0.45 / 0.41 / 0.29 / 0.28 cm; tilt dev p95
    7.2 / 7.1 / 7.5 / 6.2 deg; sat 0.6%; eval chatter ratio 2.20 / 1.00 / 0.54 / 1.94.
  Budget (learn + eval-120): c 234.6, d 228.4, k 227.0 s warm; k2 COLD 267.7 s (linearize compile 26.8, flight program
    29.3 s). G10 first-call sub-gates 10.7-11.2 s warm (as every round).
  dt 0.10 / 0.16 (64 x 240): c 100 / 100% (follow 99.5 / 99.5%); d 100 / 100% (99.5 / 99.5%); k2: see below.
  CPU check (28 x 240, box CPU): identical survival in c / d / k; segment verdicts 100% in all three; |v_seg| median
    1.4-2.6e-4, max 0.27 (c: a baseline flight) / 0.043 (d) / 0.0097 (k); policy-flight err/scale max 14.2 / 12.8 / 17.2
    (x7: 13.4); the large errors are the baseline arm's tumbling flights, as rounds 3-5.
  Post-ILC envelope (15 deg): c FWD 0.75 REV 0.50 UP 0.60 DOWN 0.48; d 0.75 / 0.65 / 0.54 / 0.48; k 0.75 / 0.75 / 0.48 / 0.54;
    k2 0.75 / 0.75 / 0.72 / 0.54 (x7: FWD 0.65, REV 0.60).
  Ladders (8 x 120): c FWD 0.65-0.80 8/8 (0.80 is past the grid: the 0.75 entry, 20% saturated), REV 0.55-0.75 8/8 (yaw max
    14 deg at 0.75), UP 0.5, DOWN 0.45 / 0.50 8/8; d the same except REV 0.70 6/8; k and k2 everything 8/8 incl. FWD 0.80
    (2% / 14% saturated) and REV 0.75.
  Actuator cost (CPU hover.launch.fly, 60 beats, window 30-60, contacts off; smooth peak mNm (x hover) / per-crank RMS mNm /
    power W / beat-boundary spike median (max) mNm), x8_trim_c: HOVER 216.6 / 88-106 / 16.03 / 239 (243); FWD 0.70 (V_op)
    237.2 (1.10x) / 101-134 / 16.24 / 280 (407); FWD 0.75 246.6 (1.14x) / 96-141 / 16.26 / 284 (353); REV 0.60 (V_op) 205.7
    (0.95x) / 89-108 / 15.93 / 292 (305); REV 0.65 202.7 (0.94x); FWD 0.65 259.7 / spike 497 (1297 FLAGGED: |du| 0.0067 at
    0.65, the ILC entry next to the spliced 0.70 trim); UP 0.5 212.6 / spike 775 (972, this learn's climb chatter, ratio 2.2);
    DOWN 0.45 322.2 (1.49x, FLAGGED: 1.2 mNm over 321 as x7_d45c's 320.6; unchanged cell). The new FWD / REV V_op cells cost
    LESS than x7's FWD 0.60 (259 mNm, 1.20x): the trims carry the speed on tilt with half the harmonic input.

V_op RAISED FURTHER (keep mode; every learn's ladder flew FWD 0.80 and REV 0.75 at 8/8 except trim_d REV 0.70 6/8):
  FWD 0.75 / REV 0.70: x8_v75a (refine) 95.5% survival, REV follow 76.9%, 13 losses all REV 0.70 -> UP; x8_v75k_a (keep)
    99.3% (pushed 97.2%: 2 flights pushed (vy 0.2, vz 0.1, w_y 0.5) in REV 0.70 lost 22 beats later), follow 98.09%;
    x8_v75k_b (keep) 100% / 98.69%. => 1 of 2 keep learns: NOT certified (REV 0.70 push margin).
  FWD 0.75 / REV 0.65 (keep): x8_k7565_a 100% (pushed 100%) / 98.69% (HOVER FWD REV UP 100%, DOWN 93.66% = ceiling 93.7%),
    |along| p95 FWD 0.0042 REV 0.0037 UP 0.0055 (bars 0.1125 / 0.0975 / 0.075), DOWN 0.088 vs the reference's 0.089 (ref-passed
    0.031), hold 0.29 cm, tilt 6.7, sat 0.6%, chatter 2.58, budget 231.6 s warm; x8_k7565_b the same 100% / 98.69%, |along|
    FWD 0.0036 REV 0.0044, hold 0.27 cm, chatter 0.84, budget 241.7 s (ILC 75.9 s, first flight call 15.0 s: 18 CPU
    processes of my own torque / CPU checks were running on the box during it); x8_k7565_c (quiet box, third learn)
    100% / 98.69%, |along| FWD 0.0067 REV 0.0047, hold 0.32 cm, chatter 0.58, budget 238.3 s warm (ILC 69.5, eval 149.0 s).
  dt 0.10 / 0.16 (64 x 240): k7565_a 100 / 100%, k7565_b 100 / 100% (follow 99.5% in all four). CPU check (28 x 240):
    k7565_a / b identical survival, verdicts 100% / 100%, |v_seg| median 1.9e-4 / 1.4e-4, policy err/scale max 22.5 / 10.2
    (the |v_seg| max 0.49 and err 120-140 are baseline-arm flights, as every round).
  Ladders (8 x 120): k7565_a and k7565_b every cell 8/8: FWD 0.65-0.80 (0.80 past the grid, 10-16% sat), REV 0.55-0.75, UP
    0.5, DOWN 0.45 / 0.50.
  Actuator cost (x8_k7565_a): HOVER 216.4; FWD 0.70 244.2 (1.13x) / 100-136 / 16.25 W / spike 311 (576); FWD 0.75 (V_op) 248.5
    (1.15x) / 95-142 / 16.27 W / 337 (512); REV 0.60 213.3 (0.99x); REV 0.65 (V_op) 211.8 (0.98x) / 88-110 / 15.88 W / 283
    (303); REV 0.70 214.0 (0.99x); UP 0.5 213.1 / spike 303 (333); DOWN 0.45 320.3 (1.48x, at the 321 limit as before).

ARM B (crank speed g, DOWN; variable-length wingbeats): CPU parity: the device beat_var at n0 == mjx_flight.beat and at
  n0 + 30 / n0 - 20 == hover.launch.Drive flying theta[0] = g0 + ln(n0 / n), < 1e-9 in x64 (test_crank_speed_g_device_and_live);
  in the learns' CPU checks (float32 device vs CPU, g flown by CommandPolicy) identical survival and 100% verdicts.
  B_g (CPU FD, n0 -/+ 3): vel (-0.384, -0.015, +1.387) m/s, w (4.17, -7.85, 1.58) rad/s per unit g (as the X7 pilot).
  g -5% (n 901..947), V_op DOWN 0.5 (x7 flags otherwise): x8_g05a / x8_g05b: survival 98.96% / 98.96% (pushed 95.8%: the same 3
    flights, pushed (vy 0.2, vz 0.1, w_y 0.5) at a DOWN 0.5 segment start, lost 23-31 beats later, in both learns), follow
    97.97 / 97.92%; DOWN 93.4 / 92.8%. Torque (g05a, CPU): DOWN 0.45 270.7, 0.50 277.8 (1.28x), 0.55 269.3, 0.60 302.0 mNm
    (1.39x), power 14.6 W (0.91x). FAILS survival.
  Control without g, V_op DOWN 0.5 (x8_d05ng): 100% (pushed 100%) / 98.69% -> the push loss is g's (the gain is solved on the
    n0 model; 5% longer wingbeats), but DOWN 0.5 without g costs 354.7 mNm (1.64x hover, FLAGGED > 321) -> not claimable.
  g -3% (n 901..929), V_op DOWN 0.5: x8_g03a / x8_g03b: 100% / 100% (pushed 100 / 100%), follow 98.81 / 98.81% (DOWN 94.2% vs
    ceiling 93.7%), DOWN |along| p95 0.080 / 0.080 vs the reference's own 0.083 (ref-passed 0.034; bar 0.075: passes
    against the reference), hold 0.40 / 0.27 cm, chatter 5.05 / 1.93. dt (g03a, g re-quantized per dt: n0 1171 / 732) 100 /
    100%. CPU check: g03a / g03b identical survival, verdicts 100%, |v_seg| max 0.0053 / 0.014.
    Torque (CPU): g03a DOWN 0.50 283.5 mNm (1.31x hover; no g: 354.7) / 106-122 / 15.14 W (0.94x), DOWN 0.55 310.1 (1.43x),
    DOWN 0.45 268.1; g03b DOWN 0.50 280.4 (1.29x), 0.55 308.6. UP 0.5 spikes 967 (1435) in g03a (its climb gain chatters,
    ratio 5.05: FLAGGED).
    Timing: g03b 238.0 s warm (calib 233.3: the g program costs ~+3 s eval, ~+1 s ILC at n_max 929; g05 ~+10 s at 947);
    g03a 253.3 s incl. the new 384 x 20 (n_max 929) program compile 28.6 s. => well under the 40 s allowance.
  => g -3% CERTIFIES V_op DOWN 0.5 (2 / 2 learns) at 1.29-1.31x hover torque; g -5% buys more torque but loses push margin.

COMBINED (keep trims + g -3%, V_op FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.5):
  x8_cg_a 100% (pushed 100%) / 98.69%, |along| FWD 0.0051 REV 0.0035 UP 0.0069, DOWN 0.103 vs the reference's 0.106
    (ref-passed 0.040; ceiling 93.1%), hold 0.31 cm, chatter 3.96, 234.0 s warm; dt 100 / 100%; CPU identical survival,
    verdicts 100%, policy err/scale max 8.6; ladder (variable-length beats) every cell 8/8 (FWD 0.65-0.80, REV
    0.55-0.75, UP 0.5, DOWN 0.45 / 0.50). Torque: DOWN 0.50 285.9 (1.32x) / 15.15 W, DOWN 0.55 313.0 (1.45x), FWD 0.75 246.7
    (1.14x), REV 0.65 211.8 (0.98x), UP 0.5 spike 685 (906, chatter).
  x8_cg_b FAILED 1.7%: the learn flew the 5-DEG map (its 15-deg carry envelope read UP 0.30; the chosen-map rule takes the
    smallest per-axis envelope, ties -> map 0) at V_op FWD 0.75. Fixed: --ff-trims now forces the trims' map (AGENTS.md
    known bug; the default rule is unchanged).
  x8_cg_b2 (after the fix) 98.3% (pushed 98.6%), follow 97.56%, UP 97.0%: 5 flights lost 13-15 beats into UP after DOWN 0.5
    (no push) - not seen in cg_a / cg_c, g03a / b or any no-g learn.
  x8_cg_c (COLD, empty cache) 100% / 98.69%, chatter 1.25, budget 277.2 s cold (linearize compile 26.6, flight program 28.9 s).
  => 2 of 3 post-fix learns pass: NOT certified as the round's best (the DOWN -> UP loss mode of cg_b2 is unexplained).

TIMING (this box; the x7 flags rerun today, x8_calib, 233.3 s warm = round 5's 230.9-232.8, so the box speed is unchanged;
  not measured on the 5070 Ti reference, x 1/1.34 is a scaling only):
  learn + eval-120 warm: trim_c 234.6, trim_d 228.4, trim_k 227.0, k7565_a 231.6, k7565_c 238.3 s, k7565_b 241.7 s (box CPU
  loaded by my own 18 CPU processes); cold trim_k2 267.7 s, cold cg_c 277.2 s. g: g03b 238.0 warm (+~5 s vs calib), g03a 253.3
  (first compile of the n_max 929 program 28.6 s), g05b 245.5 (+12 s at n_max 947). Eval calls at the raised V_op take
  146-149 s vs 143.8 s (calib). G10 first-call sub-gates 10.7-11.7 s warm (as every round).
  The trims are precomputed OUTSIDE learn (x64 device 1429 s + CPU check 16 s), untimed; a learn only loads trims.npz.
  Untimed per learn: eval-240 continuation 143-152 s, verify 26-30 s, dt 508-570 s, ladder 140-159 s, CPU check 80-90 s.
  GPU wall this round ~4.3 h (24 learns incl. 2 cancelled / failed set-ups, 7 dt checks, 10 ladders, 1 trim continuation).
CHECKS: pytest tests/test_hover_command.py tests/test_hover_mjx.py 16 passed (459 s local; new test_crank_speed_g_device_and_live);
  golden --quick identical to golden_quick_x4.log apart from timings (9/32, opt/runs_fast/golden_quick_x8.log), run after
  the last src/ change. New flags default off (no --ff-trims / --g-down: the learn path, the program and the policy file are
  the old ones; r18 / command.tg / LiveCommand unchanged: tests pass).
DEAD ENDS / FAILURES this round: the frontier-splice --ff-trims (x8_trim_a_frontier, 29.5%); refine-mode trims at V_op
  REV 0.70 (x8_v75a 95.5%: REV 0.70 -> UP); keep-mode V_op REV 0.70 (1 of 2: pushed REV 0.70 flights); g -5% (push margin at
  DOWN 0.5 in 2 of 2); the combined trims + g -3% config (2 of 3 post-fix learns; cg_b2 loses DOWN 0.5 -> UP flights).
  A first edge_trims run that minimised |u|inf every step (trust 0.03) slid along the trim manifold (|u|inf 0.065 -> 0.023
  at FWD 0.60 while the moment residual grew to 0.64 mNm): the min-|u| phase is now 2 steps, then Newton.
VERDICT: KEEP. Certified V_op FWD 0.6 -> 0.75 (+25%), REV 0.5 -> 0.65 (+30%), UP 0.5, DOWN 0.45 = config K =
  x7_d45c flags + --ff-trims opt/runs_fast/x8_trims/trims.npz --ff-trims-keep --v-op-h 0.75 --v-op-rev 0.65:
  x8_k7565_a, reproduced by x8_k7565_b and x8_k7565_c (3 / 3: survival 100% incl. pushed, follow 98.69%, every moving command
  100%, DOWN 93.66% = its ceiling, |along| p95 FWD 0.0036-0.0067, REV 0.0037-0.0047, UP 0.0055-0.0083 (<= 1.7% of V_op), hold p95 <=
  0.32 cm, dt 100 / 100%, CPU verdicts 100%, warm 231.6 / 238.3 s quiet (241.7 loaded); DOWN graded against its reference
  as in rounds 3-5. Also certified (separately): V_op FWD 0.7 / REV 0.6 in 4 / 4 learns (trim_c / d refine, k / k2 keep), and
  Arm B's g -3% at V_op DOWN 0.5 (with FWD 0.6 / REV 0.5) in 2 / 2 learns (g03a / b) at 1.29-1.31x hover torque.
  FLAGS: DOWN 0.45 still at the torque limit (320.3-322.2 mNm, 1.48-1.49x; g -3% brings DOWN 0.45 to 268 and 0.50 to 280-286);
  FWD 0.65 beat-boundary spike 1.3 N m in trim_c (the ILC 0.65 entry next to the spliced 0.70 trim); UP 0.5 climb-gain
  chatter spikes (0.7-1.4 N m) in the learns whose eval chatter ratio is >= 2 (trim_c 2.2, k7565_a 2.58, cg_a 3.96, g03a 5.05).
  Not exported: designs/hummer/free-hover/command.tg is still r18. Nothing committed.
NEXT: (1) the combined config (trims + g -3%, DOWN 0.5): find the DOWN 0.5 -> UP loss of cg_b2 (transition shaping of g,
  or the climb gain), then certify; (2) extend the speed grid past 0.75 (--n-speeds / SPEEDS_H) with trims at 0.80 / 0.85:
  every keep ladder already flies FWD 0.80 on the clamped 0.75 entry and the trims exist to 0.85; (3) REV 0.70 needs push
  margin (2 pushed losses in v75k_a), e.g. the REV gain at speed or the climb_after_stop shaping for REV -> UP;
  (4) a g-aware gain (B_g in the DARE, or the gain solved on the n_k model) for the g -5% push loss.

## X9 (round 7): certify config K as `learn --preset fast`, export it over command.tg (2026-10-01, box 40346)
Code:
- `opt/hover_cmd.py learn|all --preset fast` = config K as the defaults (PRESETS["fast"]; explicit flags still override;
  no --preset = r18, every old default: checked by parsing both and diffing against x8_k7565_a's and r18's metrics.json
  config: identical apart from tg / control paths). The trims are now a committed file,
  designs/hummer/free-hover/command_trims.npz (byte copy of opt/runs_fast/x8_trims/trims.npz, md5 5a750951971281e8093b2a66e015da24).
- BUG FIXED (pre-existing since X8): `sensors` crashed (cell_pass_inputs returns 8 values since the g port; sensors
  unpacked 7). Every `sensors` run since X8 would have failed.
- `sensors` / `export` / `all`: `--p-gain {auto,hover,climb}` (p_servo_gain). P's servo gain (q_int 3) solved on the hover
  model (old) or on the policy's regressed climb model; auto (default) = climb when the policy has climb_A (r18 has none:
  r18's P gain is bit-identical, max |dKa| 0).
- `export`: the stored envelope (LiveCommand clips to it) is max(post-ILC V_max, V_op) per axis (x9_fast_a's post-ILC UP
  was 0.48 < the certified V_op 0.5; the r18 export is unchanged: its envelope is above V_op on every axis).
- tests/test_hover_command.py `_live_setup`: rate = C.ref_rate(pol) (what LiveCommand flies; the old formula ignored
  rate_beats_z and failed 4 live-equivalence tests on the new command.tg, identical for r18).
Scripts: scratch/x9_queue.sh (learn --preset fast, then dt, sensors), x9_cpu.sh (cpu checks + cpu_torque cells),
  x9_summary.py.
LEARNS (`learn --preset fast`, nothing else), eval-240, 288 V_op flights (72 pushed), V_op FWD 0.75 / REV 0.65 / UP 0.5 /
  DOWN 0.45; a / b warm, c COLD (empty JAX cache):
  survival 100 / 100 / 100% (pushed 100% in all); follow 98.57 / 98.69 / 98.57%; per command HOVER FWD UP 100%, REV 99.4 /
  100 / 99.4%, DOWN 93.66% in all = its reference ceiling (the reference itself passes 93.7% of DOWN segments);
  |along| p95 all 0.0136 / 0.0133 / 0.0140 (median 0.0019-0.0023); per axis FWD 0.0053 / 0.0059 / 0.0052 (bar 0.1125), REV
  0.0039 / 0.0037 / 0.0039 (bar 0.0975), UP 0.0061 / 0.0046 / 0.0089 (bar 0.075), DOWN all segments 0.0888 / 0.0889 / 0.0891
  vs the reference's own 0.0890 (ref-passed 0.031: passes only graded against the reference, as rounds 3-6);
  hold p95 0.28 / 0.39 / 0.28 cm; tilt dev p95 6.9 / 8.2 / 6.7 deg; sat 0.3 / 0.6 / 0.3%; post-ILC (15 deg) FWD 0.75 in all,
  REV 0.75 / 0.50 / 0.75, UP 0.48 / 0.54 / 0.54, DOWN 0.48 / 0.60 / 0.48; eval chatter ratio 4.25 / 4.63 / 5.15 (all three
  chatter at UP: K's learns were 2.58 / 0.84 / 0.58).
  Budget (learn + eval-120): a 235.3 s, b 233.0 s warm (ILC 68.5-68.8, eval 145.3-147.2); c COLD 276.8 s (linearize compile
  26.9, flight program 29.8 s). G10 first-call sub-gates 10.8-11.3 s warm (as every round).
  dt 0.10 / 0.16 (64 x 240): a, b, c 100 / 100% (follow 99.46% in all).
  CPU check (16 x 240, box CPU, contacts off): a, b, c identical survival, segment verdicts 100%, |v_seg| median 2-3e-4;
    the 12 policy flights fly on both (err/scale max 23.9 / 29.5 / 27.3); |v_seg| max 0.45-0.51 and err 160-270 are the
    baseline arm's tumbling flights, as every round.
SENSORS (P = pos,att,w LQG, 128 random V_op schedules x 120 beats, noise x1 / x2; survival / follow):
  hover-model P gain (the old default): a 62.5 / 68.8%, 61.7 / 66.1%; b 65.6 / 69.6, 61.7 / 67.5; c 65.6 / 71.2, 60.9 / 66.7.
    38 of a's 48 x1 losses are in UP, ~30 beats into the segment: the sustained-climb roll mode the hover gain cannot hold
    (X2'). `--p-q-int -1` (P flies the policy's own climb flight gain, q_int 50) on a: 89.8 / 88.7%, 86.7 / 86.6%.
  climb-model P gain (--p-gain auto -> climb, now default): a 99.2 / 94.1% (x2 94.5 / 88.2), b 89.1 / 87.9% (x2 81.3 / 82.0),
    c 98.4 / 93.8% (x2 94.5 / 90.9). Per command x1 (a): HOVER 94.2 FWD 100 REV 83.1 UP 98.7 DOWN 94.2. P hold p95 3.9-4.2 cm.
    r18's P at V_op 0.3 (finding 5.5/5.6): x1 100% / 96.2%, x2 97.7% / 90.1%. => P at the new speeds is WORSE than r18's P
    at 0.3 (x1 survival 89-99% vs 100%, REV follow 76-83%); not a gate of this round, flagged.
  N (IMU only) survives 100% and follows 29-36% (as r18: 47%).
ACTUATOR COST (CPU hover.launch.fly, 60 beats, window 30-60, contacts off; smooth peak mNm (x hover) / RMS per crank / power
  W / beat-boundary spike median (max) mNm; a then b): no actuator limit in the model (velocity servos kv 1, ctrlrange +-120
  rad/s, forcelimited False, checked on the compiled model); ctrl peak 60-70 rad/s, never clamped.
  HOVER 216.5 / 88-106 / 16.03 / 237 (241); 216.8.
  FWD 0.75 247.7 (1.14x) / 95-143 / 16.27 / 317 (388); 248.0 (1.15x), spike max 512.   FWD 0.70 243.9 / 243.4, spike max 602 / 555.
  REV 0.65 212.0 (0.98x) / 88-109 / 15.88 / 280 (317); 212.5.   REV 0.70 213.3 / 215.2.
  UP 0.5 212.0 (0.98x) / 92-111 / 16.38 / 963 (1057); 207.9 / spike 1154 (1391). FLAGGED: climb-gain chatter (|du| 0.014-0.019)
    gives 1.0-1.4 N m one-step crank impulses at every wingbeat boundary (4.5-6.4x the hover peak).
  DOWN 0.45 322.6 (1.49x) / 118-135 / 16.04 / 280 (371); 321.1 (1.48x). FLAGGED: at the 1.5x limit (as K: 320.3).
  r18 at its V_op 0.3, same script: HOVER 214.4; FWD 263 (1.22x), REV 315 (1.47x), UP 203 (spike max 354), DOWN 356 (1.66x,
    over 1.5x). The new policy at 1.5-2.5x the speed costs less smooth torque than r18 on every axis; power 15.9-16.4 W
    everywhere (r18 16.0-16.2).
EXPORT: every criterion passes in a and b (and c, cold) => `uv run python opt/hover_cmd.py export --run
  opt/runs_fast/x9_fast_a --out designs/hummer/free-hover/command.tg` (envelope FWD 0.75 REV 0.75 UP 0.5 DOWN 0.48; P gain on
  the climb model). pytest tests/test_hover_command.py + test_hover_mjx.py 16 passed after the test fix (474 s local); golden --quick identical
  to golden_quick_x8.log apart from timings (9/32, opt/runs_fast/golden_quick_x9.log).
GPU wall this round ~75 min (3 learns, 3 dt, 3 + 3 + 2 sensors runs). Untimed per learn: eval-240 145-147 s, verify 26-30 s,
  dt 510-560 s, sensors 200-330 s, CPU check 80 s.
VERDICT: SHIPPED. command.tg = x9_fast_a (V_op 0.3 -> FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.45). FLAGS: UP 0.5 crank impulses
  (chatter, in all 3 learns), DOWN 0.45 at 1.49x hover torque, the P sensor variant at the new speeds (x1 89-99% survival),
  the cold budget (276.8 s).

## X9 adversarial cross-check (2026-10-01, box 40346; tried to refute the shipped `--preset fast` claims)
Script opt/runs_fast/scratch/x9_xcheck.py (+ x9_xcheck.sh); outputs opt/runs_fast/x9_xcheck/. CPU = hover.launch.fly, float64,
contacts off, the model's own solver (100 / 50 iterations), a per-step recorder auditing forces and crank actuator_force.
- Random V_op flights, dt 0.13: 24 eval flights NOT in the cert's CPU check (16 plain + 8 pushed, seed 7) x 240 beats: CPU = GPU
  survival 100 / 100%, segment verdicts 142 / 142 identical, |along| diff median 1.6e-4 max 2.4e-3; follow 99.3% both (DOWN 96.7%,
  the same single DOWN segment fails on both). dt 0.10 / 0.16: 10 + 10 of the dt-check flights vs a GPU dump of that check
  (gpu_dt.npz): 100% survival both, verdicts 56 / 56 and 59 / 59 identical (dt 0.16 DOWN 92.9% on both = ceiling).
- Constant commands (10 hover, 110 command, 60 hover) at dt 0.10 / 0.13 / 0.16, CPU and GPU: HOVER, FWD 0.75, REV 0.65, REV 0.75,
  UP 0.5, DOWN 0.45, DOWN 0.48 all fly on both, every command and stop segment passes, steady |v_along - v_cmd| <= 1e-3.
- Viewer path (LiveCommand on the EXPORTED command.tg, view_free_hover's release + beat loop, headless): 3 scripts (viewer default
  speeds = REV 0.75 / DOWN 0.48 by the envelope clip; V_op; direct axis-to-axis transitions) fly all 540 / 540 / 395 beats,
  31 / 31 segments pass, max tilt 24 deg.
- Force audit (every step of every flight): xfrc_applied 0; qfrc_applied == aero + wing-pitch limiter exactly (0.0 mismatch);
  gravity -9.81, mass 23.07 g constant; 4 actuators = the cranks; no equality constraints; with contacts ON (floor moved away)
  0 contacts over 30 beats (contacts off hides nothing).
- Actuator cost: steady numbers reproduce (HOVER 216, FWD 0.75 248, REV 0.65 212, UP 0.5 212 / spikes 1.1-1.25 N m, DOWN 0.45
  321-326 mNm). NOT in the cert's table: DOWN 0.48 (the shipped envelope, the viewer's default DOWN) steady 342-348 mNm = 1.58-1.61x
  hover; random V_op flights: DOWN second-half windows up to 365 mNm (1.68x), whole-flight smooth peak max 387 (1.79x; 88% of
  flights > 1.5x; r18 at 0.3: 399, 1.86x); one-step beat-boundary crank impulses at command changes median 9.1 N m, max 14.2 N m
  (65x hover; FWD/REV/UP starts, not only UP) vs r18 1.9 / 2.8 N m (input steps |du| max 0.19-0.20 vs r18 0.059).
  BUT with a crank torque limit (forcelimited, +-0.325 N m = 1.5x hover peak; also 0.5, 1.0) every constant command and the same
  24 random flights fly identically (survival 100%, follow 99.3%, verdicts = the unlimited GPU): the impulses are an artefact of
  the unlimited kv-1 velocity servo, not needed for flight.
- Other: learn --preset fast config == x8_k7565_a apart from paths (checked); command_trims.npz == x8_trims/trims.npz;
  pytest test_hover_command + test_hover_mjx 16 passed (470 s local). x9_fast_b's post-ILC REV V_max (0.50) is below the
  certified REV V_op (0.65) that b flies 100%: the short-cell post-ILC envelope is not a reliable V_max.
VERDICT: the speed claims reproduce on CPU. Report fixes: add the transition impulses (all axes, 5x r18) and DOWN 0.48 at 1.6x
  hover to the actuator-cost flags, and note the viewer defaults fly REV 0.75 / DOWN 0.48 (above V_op; they fly here).

## X10: fix the review findings (actuator-cost metric, rebuildability) (2026-10-01, box 40346)
Two verified findings against X9: (1) the actuator-cost report compared only the "smooth" peak (first 5 steps of each
wingbeat left out) and missed full-step peaks > 1.5x hover at FWD 0.70 / 0.75, REV 0.70, DOWN 0.45; (2) `--preset fast`
reads designs/hummer/free-hover/command_trims.npz but its generator (scratch/edge_trims.py) and these notes were gitignored.
CODE (no behaviour change for learn / export / any existing subcommand; golden --quick identical to x9, golden_quick_x10.log;
pytest test_hover_command + test_hover_mjx 16 passed, 465 s local):
- `opt/hover_cmd.py trims` = edge_trims.py moved into the tracked CLI (same solver, defaults, npz keys; output trims.npz +
  trims.json); --policy defaults to the tracked command_policy.npz; --seed-trims replays an old trims.npz's x0.
- `opt/hover_cmd.py torque` = cpu_torque.py as a tracked CLI: CPU hover.launch.fly (float64, the model's own solver, contacts
  off), constant cells (HOVER + V_op per axis + the exported envelope + --cells) of 120 beats measured over beats 30-120, plus
  --eval-flights random V_op eval flights (traces.npz, 240 beats, from beat 10, graded with segment_metrics); peak_all (every
  step) AND smooth peak, each / the unlimited HOVER's, FLAG > 1.5x on either; --limit L re-flies all with cranks forcelimited
  +-L. Reference rate = hover.command.ref_rate (cpu_torque.py used V_op_h / rate_beats for vertical too). Speed check =
  mean-motion (strobed position) velocity error (the strobed velocity carries the table's dv offset: 0.2-0.3 m/s at speed).
- these notes frozen to docs/history/command_flight_faster_notes_2026-10.md; docs point at the tracked paths.
TORQUE (opt/runs_fast/x10_torque/*.log; scratch/x10_cpu.sh; 2 min per run on 12 box cores). peak_all mNm (x hover) a / b / c,
  smooth x hover:
  HOVER 241 / 249 / 241 (smooth 217).
  FWD 0.75 490 / 512 / 451 = 2.03 / 2.06 / 1.87x FLAG (smooth 1.14-1.15x).   FWD 0.70 602 / 662 / 462 = 2.50 / 2.66 / 1.92x FLAG.
  REV 0.65 317 / 335 / 346 = 1.32 / 1.35 / 1.44x (smooth 0.98x).   REV 0.70 1.29 / 1.70 / 1.59x (FLAG b, c).
  REV 0.75 (envelope) 455 / 722 / 315 = 1.89 / 2.90 / 1.31x (FLAG a, b).
  UP 0.5 1103 / 1353 / 1243 = 4.6 / 5.4 / 5.2x FLAG (smooth 0.96-0.99x; climb-gain chatter).   UP 0.54 5.0 / 8.1 / 2.1x.
  DOWN 0.45 404 / 436 / 401 = 1.68 / 1.75 / 1.66x FLAG (smooth 1.48-1.49x).   DOWN 0.48 (envelope) 1.72-1.89x, smooth 1.60-1.61x
  FLAG both.   DOWN 0.50 smooth 1.63-1.64x.
  random V_op eval flights (8 per run, command changes): peak_all 7.8-13.2 N m = 32-55x (smooth 1.18-1.71x), every flight.
  r18 (V_op 0.3): FWD 2.45x, DOWN 2.12x (smooth 1.66x), REV 1.41x, UP 1.31x; random flights 1.75-2.43 N m = 6.5-9.0x (smooth
  1.56-1.80x). Power 15.8-16.4 W everywhere.
  --limit 0.325 N m (1.5x the hover smooth peak): every cell of a, b, c, shipped, r18 survives, mean-motion |v - v_cmd| <= 0.003
  m/s; all 8 random flights of each run keep the same segment verdicts (a/b/c 46/46, r18 45/46 both ways).
  The x9 table's FWD 0.75 smooth 247.7 / peak_all 387.7 (60 beats, window 30-60) -> 248.7 / 489.5 here (window 30-120): the
  peak_all is a max over spikes, it grows with the window.
TRIMS (GPU, x64, scratch/x10_gpu.sh): `trims --policy opt/runs_fast/x7_d45c/policy.npz` (x10_rt_x7, 1451 s) reproduces
  command_trims.npz to 1.8e-9 (every key; GPU x64 round-off). `trims` with the default seed (tracked command_policy.npz;
  x10_rt_cp, ~24 min) converges to other trims: FWD |u|inf 0.038-0.053 (shipped 0.033-0.099), FWD 0.75 yaw +10.0 deg (at the
  --yaw-max-deg box, res 0.051; shipped -2.1 deg, 0.017); REV within 0.03 in u and 4 deg in yaw. The trim set is not unique.
REBUILD (scratch/x10_gpu2.sh): `learn --preset fast --ff-trims opt/runs_fast/x10_rt_cp/trims.npz` = x10_rb_a: 229.2 s warm;
  eval-240 survival 100% (push 100%), follow 98.69% (HOVER FWD REV UP 100%, DOWN 93.66% = ceiling), |along| p95 FWD 0.0058 REV
  0.0051 UP 0.0059 DOWN 0.0884 (ref-graded), hold p95 0.29 cm, post-ILC FWD 0.75 REV 0.75 UP 0.54 DOWN 0.60; chatter UP/MOVE
  0.76 (x9: 4.25-5.15) -> torque UP 0.5 peak_all 340 mNm = 1.33x (x9 4.6-5.4x); FWD 0.75 2.65x, DOWN 0.45 1.74x, random flights
  27-34x. One learn: no dt / CPU / second learn -> not certified, not exported.
VERDICT: the shipped command.tg (x9_fast_a) is unchanged and still meets every gate; its crank-cost flags are now FWD 0.70 /
  0.75, REV 0.70 / 0.75, UP 0.5, DOWN 0.45 / 0.48 and every command change (peak_all), DOWN 0.48 (smooth). With the cranks
  limited to 1.5x hover it flies the same on CPU. command_trims.npz is reproducible from tracked code.
GPU wall ~62 min (2 trims + 1 learn); box CPU ~15 min.
