#!/bin/bash
# RUN 8 "energy" (2026-09-09): peak torque is the wrong thing to minimize -- any
# velocity change fights a torque reward. The cost that actually matters is
# mechanical ENERGY, credited by the lift it buys: --objective energy minimizes
# the ROBUST specific power P/L [W/N] (worst-case supplied power over worst-case
# lift across the batch), normalized to ~1 at the seed, with the denominator
# floored at --lift-floor so it cannot be won by not flapping and peak torque
# demoted to a soft ceiling guard (--tau-guard 4).
#
# Why a RATIO and not "P - lambda*L": a linear credit fixes the exchange rate, so
# every knob that buys lift under lambda W/N is a score fountain -- run 7 farmed
# exactly that (a0 up, +67 mW of watts booked as lift). The ratio prices lift at
# the design's own current average P/L, so a lift knob pays only if it beats the
# efficiency already achieved. No --lift-torque-tradeoff here: the ratio IS the
# trade, and paying twice re-opens the fountain. wL=10 carries the floor wall.
#
# --lock-wings: one design for BOTH wings. The seed ships L_t1_to_t1_2 at 184 N/m
# against R's 341, and the genome gave all 24 structural tendons independent
# genes, so every previous run optimized two different wings sharing one command.
# Locked = geometric-mean the pairs at load + tune the 12 mirrored pairs (24
# params, not 48). Costs ~nothing: CPU baseline 62.6 W/N locked vs 63.3 unlocked.
#
# --a0-free with --control-after 10: a0 is the crank speed = flap frequency, the
# dominant power lever (P~f^3, L~f^2), live from it000; harmonics wait until
# it010. So the full-checks before/after it010 answer the actual question --
# did efficiency come from FREQUENCY or from WAVEFORM SHAPE? Expect the run to
# trade frequency down against the lift floor first; whatever it finds after
# it010 is shape. Robustness machinery unchanged from runs 5/7 (rotate-batch 3,
# full-matrix + CPU transfer gates, drift pricing, holdouts) so the winner is
# still CPU-transferable. Torque shaping out of loss and selection
# (--w-swing 0 --score-swing 0); w_power/w_overlap off -- supplied power already
# subsumes braking, and proxies are what get gamed.
#
# n-batches 2, not run 7's 6: each rotating batch is a separate jacfwd-over-37-
# params jit, and one took 36 min to compile on this box -- six of them is 3.5 h
# of XLA before iteration 1. Two batches still cover 6 of the 12 bank cells in the
# GRADIENT (each batch keeps >=2 integrators and >=2 dts), and the full-matrix
# check still referees all 12, so the transfer gate is unchanged.
#
# --bank-integrators euler implicitfast: rk4 inside a jacfwd over 37 params did
# not finish compiling in 50 min on this box. The CPU gate still rolls rk4 in
# numpy (no compile), so the integrator-hacking check survives -- the gradient
# just stops paying 4x physics for it. Warmup bank 8/16, not 8/16/24, for the
# same reason: w24 at dt 0.04 ms is a 54k-step scan.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NVLIBS="$(find "$PWD/.venv/lib" -path '*nvidia*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
# The box is shared: another tenant's sweep holds ~11 of the 16 GB. Take memory on
# demand instead of grabbing 75% up front, and do not fight them for it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=opt/runs_energy
LOG=opt/run_energy.log
DONE=opt/run_energy.done
rm -f "$DONE"
uv run --no-sync python opt/optimize_control.py \
  --tg flight-orientations/hovering-soft-it009p/shoulder.tg \
  --lock-wings \
  --objective energy --tau-guard 4 \
  --tune-tendons --tune-motor-tendons --tune-pitch --single-stage \
  --lift-floor 0.335 --lift-smooth 0.015 \
  --weights 10 10 8 0.03 --w-power 0 --w-overlap 0 --w-swing 0 --score-swing 0 \
  --rotate-batch 3 --n-batches 2 --batch-seed 17 \
  --bank-integrators euler implicitfast \
  --dt-bank 0.04 0.06 0.09 0.13 --dt-jitter 0.15 \
  --warmup-bank 8 16 --sample-periods 4 \
  --softmax-beta-start 100 --softmax-beta-end 300 \
  --full-check-every 15 --full-gap 0.15 --full-sample-periods 16 --drift-gate 0.03 \
  --w-drift 60 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after 10 --a0-free \
  --cpu-every 40 --cpu-gap 0.15 --cpu-timestep-factors 0.5 --cpu-sample-periods 24 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage 120 --lr 4e-4 \
  --out "$OUT" >"$LOG" 2>&1
echo $? > "$DONE"
