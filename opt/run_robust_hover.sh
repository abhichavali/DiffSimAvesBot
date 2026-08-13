#!/usr/bin/env bash
# ROBUST CODESIGN from the hand-tuned hover seed: Adam on control (K=3 harmonics)
# + structural tendon stiffness/pretension + motor-tendon preload, under the
# ROTATING integrator x timestep mini-batch reward. Each iteration draws 3 random
# (integrator, dt) variants from a 3x4 bank and optimizes their robust envelope;
# every 25 iters the FULL 12-case matrix gates a transfer-checked robust_best;
# holdout dts {0.03, 0.11, 0.16} ms are evaluated ONLY at the end (generalization,
# not training). Tendon/motor knobs train alone for the first 80 iters (safer
# knobs first); harmonics unlock after.
#
# Startup is compile-heavy (12 forward jits at the baseline matrix + one jacfwd
# jit per batch on first draw) -- expect slow early iterations, then ~tri-run pace.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ITERS="${ITERS:-300}"
OUT="${OUT:-opt/runs_robust_hover}"
BATCH_SEED="${BATCH_SEED:-0}"

uv run python opt/optimize_control.py \
  --tg flight-orientations/hovering-tuned/shoulder.tg \
  --tune-tendons --tune-motor-tendons --single-stage \
  --lift-torque-tradeoff 5 \
  --rotate-batch 3 --n-batches 6 --batch-seed "$BATCH_SEED" \
  --dt-bank 0.04 0.06 0.09 0.13 \
  --dt-jitter 0.15 --precision-perturb 1e-7 \
  --softmax-beta-start 50 --softmax-beta-end 300 \
  --full-check-every 25 --full-gap 0.15 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after 80 \
  --cpu-every 50 --cpu-gap 0.15 --cpu-timestep-factors 0.5 1.5 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage "$ITERS" --lr 8e-4 \
  --out "$OUT"
code=$?
echo "=== [$(date -u +%H:%M:%S)] ROBUST HOVER RUN DONE exit $code ==="
exit $code
