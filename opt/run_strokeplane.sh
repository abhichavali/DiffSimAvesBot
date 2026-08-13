#!/usr/bin/env bash
# STROKE-PLANE codesign from the hand-tuned hover seed: the robust rotating-batch
# framework (run_robust_hover.sh), plus --tune-strokeplane — per DOF (disk tilt,
# pitch) a mean + 2 Fourier harmonics of the L_t1 crank angle, tanh-bounded at
# ±0.3 rad, applied kinematically via body_quat each step (MJX and CPU alike).
# Tendon/motor knobs train alone for the first CONTROL_AFTER iters; velocity
# harmonics AND stroke-plane unlock together after (both are the risky knobs).
# Startup baseline evals + it0 verification are removed: training starts cold.
# v2: stroke-AMPLITUDE BAND REMOVED (wK=0) — and amp-band-pct widened to 10 so the
# hidden single-stage/full-check amplitude GATES go vacuous too (wK=0 alone would
# leave every candidate ineligible if amplitude drifts).
# v5: objective = user's actual contract: LOWEST peak torque that reliably makes
# >= 345 mN lift. The 5:1 trade (anchored at the 372 mN warm-start baseline) is
# gone — it forbade spending the ~25 mN surplus the contract doesn't need; the
# static sweeps showed profitable floor-riding moves (e.g. −4% speed = −1% τ at
# 350 mN). Explicit --lift-floor 0.345 with wL=10; selection + full-check gate on
# robust-min lift >= floor (code change in optimize_control.py).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/workspace/jaxcache}"  # restarts skip the ~30min batch recompiles

ITERS="${ITERS:-300}"
OUT="${OUT:-opt/runs_strokeplane}"
BATCH_SEED="${BATCH_SEED:-0}"
CONTROL_AFTER="${CONTROL_AFTER:-10}"  # v3: unlock harmonics+stroke-plane at it010 (was 80)
INIT="${INIT:-}"                      # optional --init-codesign genome json (warm start)

# --no-sync: the box's PyPI route is ~dead; .venv is provisioned offline from
# relayed wheels (see memory: vast-instance-access), so never let uv touch the net.
uv run --no-sync python opt/optimize_control.py \
  --tg "${TG:-flight-orientations/hovering-tuned/shoulder.tg}" \
  --tune-tendons --tune-motor-tendons --tune-strokeplane --single-stage \
  --lift-floor "${LIFT_FLOOR:-0.345}" \
  --sp-max 0.3 --sp-lr-mult "${SP_LR_MULT:-10}" \
  --rotate-batch 3 --n-batches "${N_BATCHES:-4}" --batch-seed "$BATCH_SEED" \
  --dt-bank 0.04 0.06 0.09 0.13 \
  --dt-jitter 0.15 --precision-perturb 1e-7 \
  --softmax-beta-start 50 --softmax-beta-end 300 \
  --full-check-every 25 --full-gap 0.15 \
  --full-sample-periods 24 --cpu-sample-periods 24 \
  --w-drift 60 --drift-gate 0.03 --warmup-bank 8 24 \
  --score-swing 0 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after "$CONTROL_AFTER" ${INIT:+--init-codesign "$INIT"} \
  --cpu-every 50 --cpu-gap 0.15 --cpu-timestep-factors 0.5 1.5 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --weights 10 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
  --amp-floor "${AMP_FLOOR:-0.33}" --w-gap 40 \
  --harmonics 3 --iters-per-stage "$ITERS" --lr 8e-4 \
  --out "$OUT"
code=$?
echo "=== [$(date -u +%H:%M:%S)] STROKE-PLANE RUN DONE exit $code ==="
exit $code
