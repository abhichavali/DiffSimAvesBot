#!/bin/bash
# RUN 10 "swing_rms, variable dt" (2026-09-10): close the hole the CPU check found.
#
# The 22-iteration queue found exactly one reward the harmonics can move: torque
# swing (and the swing+tau_rms composite). But CPU certification showed the gain is
# TIMESTEP-SPECIFIC -- swing_rms went -13.5% at the trained dt 0.06 ms and only
# -1.0% at a 0.03 ms holdout; peak torque went -5.4% at 0.06 and +1.2% at 0.03.
# The training bank was 0.06/0.09/0.13, so 0.03 was finer than anything the
# gradient ever saw, and the optimizer was free to buy swing with structure that
# only exists at coarse steps.
#
# So widen the bank DOWNWARD through the failure point: 0.03 0.045 0.06 0.09 0.13,
# still with 15% jitter and both integrators, and rotate-batch 3 x 2 batches so a
# single gradient step spans a fine and a coarse cell instead of two neighbours.
# Holdouts move outside the new bank entirely (0.02 0.075 0.11 0.16) so they stay
# honest tests. Everything else is identical to the queue arm: locked wings,
# control-only genome, --control-after 10 --a0-free for the harmonic-vs-frequency
# split, lift floor + amplitude band, peak torque as a ceiling guard.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NVLIBS="$(find "$PWD/.venv/lib" -path '*nvidia*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=opt/runs_swing_rms_vdt
LOG=opt/run_swing_rms_vdt.log
DONE=opt/run_swing_rms_vdt.done
rm -f "$DONE"
uv run --no-sync python opt/optimize_control.py \
  --tg flight-orientations/hovering-soft-it009p/shoulder.tg \
  --lock-wings \
  --objective swing_rms --tau-guard 4 \
  --single-stage \
  --lift-floor 0.335 --lift-smooth 0.015 \
  --weights 10 10 8 0.03 --w-power 0 --w-overlap 0 --w-swing 0 --score-swing 0 \
  --rotate-batch 3 --n-batches 2 --batch-seed 17 \
  --bank-integrators euler implicitfast \
  --dt-bank 0.03 0.045 0.06 0.09 0.13 --dt-jitter 0.15 \
  --warmup-bank 4 8 --sample-periods 3 \
  --softmax-beta-start 100 --softmax-beta-end 300 \
  --full-check-every 0 --drift-gate 0.03 \
  --w-drift 60 \
  --holdout-dts 0.02 0.075 0.11 0.16 \
  --control-after 10 --a0-free \
  --cpu-every 0 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage 22 --lr 4e-4 \
  --out "$OUT" >"$LOG" 2>&1
echo $? > "$DONE"
