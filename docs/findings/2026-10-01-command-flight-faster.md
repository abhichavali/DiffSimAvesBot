---
title: The command-following free bird flies 2.5x faster forward (V_op FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.45 m/s vs 0.3) after five fixes to the r18 servo; command.tg now ships it
date: 2026-10-01
status: current
tags: [free-flight, mjx, cpu, tooling]
---

## Finding
`opt/hover_cmd.py learn --preset fast` (exported to `designs/hummer/free-hover/command.tg`, run `x9_fast_a`) flies
random HOVER / FWD / REV / UP / DOWN sequences at V_op FWD 0.75 / REV 0.65 / UP 0.5 / DOWN 0.45 m/s with the r18
quality. Its smooth crank torque is below r18's at 0.3 m/s on FWD, REV and DOWN (UP: 212 vs 203 mNm). Its one-step
crank impulses are much larger than r18's, and over every step most cells exceed 1.5x hover (actuator cost below).
The speed came from five changes (each opt-in; no preset = r18):
1. `--lin-rigid-rotation`: the linearization's attitude columns no longer carry a spurious velocity change, so the
   LP can buy horizontal force with tilt (15-deg map). This roughly doubled the envelope.
2. `--climb-gain reg --climb-refit`: the flight gain is re-solved on a model regressed from the learn's own climb
   flights. The hover gain loses sustained climbs to an ~8-beat roll oscillation that the linearizations do not show.
3. The carried-lane continuation (`--ilc-iters 1 --ilc-carry 1 ...`, 20-beat calls, B 384) fits the 240 s budget.
4. `--rate-beats-z 6`: the DOWN reference shaping. It raised DOWN's reference ceiling from 92.8% to 95.7%.
5. `--ff-trims command_trims.npz --ff-trims-keep`: measured moving-cradle trims at FWD >= 0.70 / REV >= 0.60. The
   old FWD 0.70 / REV 0.60 edge came from extrapolated table entries, not from the dynamics.

## Evidence (lab notes: `docs/history/command_flight_faster_notes_2026-10.md` X1-X10; runs x9_fast_a / b warm, c cold, all `learn --preset fast`)
- Eval-240 (288 V_op flights, 72 pushed): survival 100 / 100 / 100%; follow 98.57 / 98.69 / 98.57%. Every command is
  >= 99.4% except DOWN, at 93.7%, which equals its reference ceiling. |along| p95 0.013-0.014 m/s. DOWN on all
  segments scores 0.089, the same as the reference's own score (0.031 on the segments the reference passes). Hold
  p95 0.28-0.39 cm.
- dt 0.10 / 0.16 ms: 100 / 100%. CPU (`hover.launch.fly`, contacts off): identical survival, 100% segment verdicts.
- Learn + eval-120: 235.3 / 233.0 s warm, 276.8 s cold, on the vast.ai RTX 5070 Ti that measured r18's config at
  about 277 s (r18's 206 s box was not re-measured).
- Crank actuator cost: `opt/hover_cmd.py torque` (CPU `hover.launch.fly`, float64, contacts off; constant commands
  of 120 beats measured over beats 30-120, plus 8 random V_op eval flights of 240 beats). Two peaks of |actuator_force|
  over the 4 cranks, each as a ratio to the same run's HOVER: **peak_all** over every step, and the **smooth** peak
  without the first 5 steps of each wingbeat. They differ because the policy changes its 24 harmonic depths once per
  wingbeat, so the kv-1 velocity-servo command jumps at the wingbeat boundary and the crank force spikes for one step.
  Runs a / b / c; FLAG = either ratio > 1.5x:

  | cell | peak_all mNm (x hover) | smooth mNm (x hover) | flag |
  |---|---|---|---|
  | HOVER | 241 / 249 / 241 | 217 / 217 / 217 | |
  | FWD 0.75 (V_op) | 490 / 512 / 451 (2.03 / 2.06 / 1.87x) | 249 / 248 / 247 (1.14-1.15x) | **peak_all** |
  | FWD 0.70 | 602 / 662 / 462 (2.50 / 2.66 / 1.92x) | 244 (1.12-1.13x) | **peak_all** |
  | REV 0.65 (V_op) | 317 / 335 / 346 (1.32 / 1.35 / 1.44x) | 212 (0.98x) | |
  | REV 0.70 | 310 / 423 / 383 (1.29 / 1.70 / 1.59x) | 214-215 (0.99x) | **peak_all** in b, c |
  | REV 0.75 (envelope) | 455 / 722 / 315 (1.89 / 2.90 / 1.31x) | 218 / 222 / 210 | **peak_all** in a, b |
  | UP 0.5 (V_op, envelope) | 1103 / 1353 / 1243 (4.6 / 5.4 / 5.2x) | 212 / 208 / 215 (0.96-0.99x) | **peak_all** |
  | DOWN 0.45 (V_op) | 404 / 436 / 401 (1.68 / 1.75 / 1.66x) | 323 / 321 / 321 (1.48-1.49x) | **peak_all** (smooth just under) |
  | DOWN 0.48 (envelope) | 456 / 428 / 436 (1.72-1.89x) | 348 / 346 / 348 (1.60-1.61x) | **both** |
  | random V_op flights (8 per run) | 7.8-13.2 N m (32-55x) | 256-370 (1.18-1.71x) | **both** |

  r18 at its 0.3 m/s, same tool: FWD 663 mNm (2.45x), DOWN 573 (2.12x; smooth 1.66x), REV 1.41x, UP 1.31x, its random
  flights 1.7-2.4 N m (6.5-9.0x; smooth 1.56-1.80x). Power 15.8-16.4 W in every cell (r18 16.0-16.2). The compiled model
  has no torque limit (kv-1 velocity servos, forcelimited false, ctrl +-120 rad/s, never reached). Re-flown with the
  cranks forcelimited to +-0.325 N m (1.5x the hover smooth peak), every cell and every random flight of a, b, c and
  r18 still flies: mean-motion speed error <= 0.003 m/s in every cell, the same segment verdicts in all 8 random flights
  of each run. So the impulses are an artefact of the unlimited servo, not something the flight needs. That check is
  CPU-only, on 11 cells and 8 flights per run; the certified gates were not re-run with a limit.

## Consequences / open issues
- Export stores the CERTIFIED V_op as the envelope LiveCommand / the viewer clip to (FWD 0.75, REV 0.65, UP 0.5,
  DOWN 0.45); the post-ILC V_max (above V_op on REV and DOWN in run a: 0.75 / 0.48, below it in run b) is kept as
  `envelope_post_ilc` only. Before this change the export stored max(post-ILC V_max, V_op), so the viewer flew REV 0.75
  and DOWN 0.48, which the random-command evaluation never certified.
- Actuator-cost flags (table above). Over every step (peak_all), FWD 0.70 / 0.75, REV 0.70 / 0.75, UP 0.5 and DOWN
  0.45 / 0.48 all exceed 1.5x the hover peak, and so does every random flight with command changes (32-55x, against
  r18's 6.5-9x). On the smooth peak, only DOWN 0.48 (1.6x; the exported envelope was later clipped to the certified V_op, so the viewer and LiveCommand stop at DOWN 0.45 / REV 0.65)
  and the random flights (up to 1.71x; r18 1.80x) exceed it, with DOWN 0.45 just under at 1.48-1.49x. An earlier
  version of this finding compared only the smooth peak (over beats 30-60) and flagged only UP and DOWN. UP 0.5 is
  the worst constant cell: the climb gain chatters (period-2 in u) in all three learns. Going beyond DOWN 0.45 needs
  the crank-speed feedforward (`--g-down -0.03` certified DOWN 0.5 at 1.3x smooth in 2 of 2 learns).
- No torque limit is modelled, so none is exceeded. A real crank motor must either supply these one-step impulses
  or saturate at about 0.33 N m (the limited CPU flights fly identically). The open fix is a slew-limited or
  intra-wingbeat-blended input change.
- Sensors (P = pos,att,w LQG) at the new speeds: x1 noise 89-99% survival, 88-94% follow; x2 81-95% survival. P
  needs its servo gain solved on the climb model (`--p-gain auto`, the default now). On the hover model it loses
  ~35%, in sustained climbs. That is still worse than r18's P at 0.3 m/s (x1 100% / 96.2%).
- `sensors` had crashed since the X8 crank-speed port; this is now fixed.
- Rebuilding: `--preset fast` reads `designs/hummer/free-hover/command_trims.npz`. `opt/hover_cmd.py trims` (it was
  the gitignored `opt/runs_fast/scratch/edge_trims.py` until 2026-10-01) re-measures it. After a change to the bird,
  run `trims` and then `learn --preset fast`. Checked 2026-10-01 on the 5070 Ti box, about 24 min per `trims`:
  - Seeded from x7_d45c's table (`--policy opt/runs_fast/x7_d45c/policy.npz`, the shipped file's seed), it
    reproduces `command_trims.npz` to 2e-9.
  - Seeded from the tracked `command_policy.npz` (the default), it lands on different trims of the same
    under-determined trim problem. FWD 0.75 needs |u|inf 0.044 instead of 0.081 at 12 deg more yaw (10 deg, the yaw
    box). One `learn --preset fast --ff-trims` on those trims (`x10_rb_a`) passes every learn gate: 100% survival
    (pushed too), follow 98.69%, DOWN 93.66% at its ceiling, hold p95 0.29 cm, 229.2 s warm. That was one learn; dt
    and the CPU check were not run.

## Status notes
Supersedes the "shipped policy" and envelope parts of `2026-09-30-command-following-policy`. Its method and dead
ends stand.
