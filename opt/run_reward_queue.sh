#!/bin/bash
# CAMPAIGN 2026-09-10: which reward can the WAVEFORM actually improve?
#
# Runs 7 (power) and 8 (energy) both stalled. opt/screen_targets.py explains why in
# closed form -- per unit step of the shared Fourier theta, over 6 (integrator, dt)
# cells on the locked it009p seed:
#
#   metric        d/d a0   d/d harmonics   consensus
#   tau_swing      203%        3258%         0.996     <- waveform owns it, robustly
#   tau_peak       137%        1576%         0.387     <- trap: authority, no agreement
#   power_crest      7%         218%         0.991
#   overlap        127%         197%         0.998
#   tau_rms        148%         195%         0.995
#   lift_ripple    372%          76%         0.988
#   spec_power     280%          54%         0.990
#   power           44%          15%         0.967     <- waveform-deaf: runs 7 and 8
#
# Each arm below minimizes ONE of those, normalized to ~1 at the seed, under the
# identical setup: locked wings, control-only genome (2K+1 = 7 params -- the
# question is about the control signal, and a narrow jacfwd is what makes 22
# iterations affordable on this contended box), rotating euler/implicitfast x dt
# batches, lift floor + amplitude band pinning the operating point, peak torque as
# a ceiling guard only.
#
# --control-after 10 --a0-free is the whole experiment: it000-009 moves ONLY a0
# (frequency), then the harmonics unlock. The optimizer dumps a `preharmonics/`
# snapshot with a full-matrix row at it010, so the certified harmonic gain is
# (obj at it022) - (obj at it010) and the frequency gain is (it010) - (baseline).
# A reward where harmonics help is one where the first difference is the big one.
#
# Sequential, not parallel: two jobs on this GPU halved each other's throughput.
#
# No in-loop full-matrix or CPU checks (--full-check-every 0 --cpu-every 0), and one
# rotating batch instead of two. XLA compile, not the jacfwd width, is what costs on
# this box -- ~30 min per distinct jitted rollout, so six arms x (2 grad + 6 bank)
# compiles is a day of pure compiling. Certification moves OUT of the training loop
# into opt/certify_arms.py, which builds the 6-cell bank ONCE and scores every arm's
# baseline / preharmonics / final theta through it, plus a CPU transfer check. Same
# gates, paid once for the campaign instead of once per arm.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NVLIBS="$(find "$PWD/.venv/lib" -path '*nvidia*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ARMS="${ARMS:-swing tau_rms power_crest overlap lift_ripple swing_rms}"
ITERS="${ITERS:-22}"
QDONE=opt/queue.done
rm -f "$QDONE"

for OBJ in $ARMS; do
  OUT="opt/runs_$OBJ"
  LOG="opt/run_$OBJ.log"
  [ -f "opt/run_$OBJ.done" ] && { echo "skip $OBJ (already done)"; continue; }
  echo "=== $(date -u +%H:%M) starting arm $OBJ -> $LOG" >> opt/queue.log
  uv run --no-sync python opt/optimize_control.py \
    --tg flight-orientations/hovering-soft-it009p/shoulder.tg \
    --lock-wings \
    --objective "$OBJ" --tau-guard 4 \
    --single-stage \
    --lift-floor 0.335 --lift-smooth 0.015 \
    --weights 10 10 8 0.03 --w-power 0 --w-overlap 0 --w-swing 0 --score-swing 0 \
    --rotate-batch 2 --n-batches 1 --batch-seed 17 \
    --bank-integrators euler implicitfast \
    --dt-bank 0.06 0.09 0.13 --dt-jitter 0.15 \
    --warmup-bank 4 8 --sample-periods 3 \
    --softmax-beta-start 100 --softmax-beta-end 300 \
    --full-check-every 0 --drift-gate 0.03 \
    --w-drift 60 \
    --holdout-dts 0.03 0.11 0.16 \
    --control-after 10 --a0-free \
    --cpu-every 0 \
    --integrator euler --timestep 0.06 --aero-substep 1 \
    --amp-band-pct 0.05 \
    --harmonics 3 --iters-per-stage "$ITERS" --lr 4e-4 \
    --out "$OUT" >"$LOG" 2>&1
  echo $? > "opt/run_$OBJ.done"
  echo "=== $(date -u +%H:%M) finished arm $OBJ exit=$(cat opt/run_$OBJ.done)" >> opt/queue.log
done
echo 0 > "$QDONE"
