#!/usr/bin/env bash
# Overnight INTEGRATOR-AGNOSTIC sweep: optimize each EA tendon seed under the
# 3-integrator robust reward (euler + implicitfast + rk4, all differentiable in
# MJX). The loss scores the WORST-integrator peak torque + a cross-integrator
# spread penalty, so designs that only look good on euler can't win. ~3x the
# single-integrator cost, so PAR=1 on a 3080. cpu-every re-checks all three on
# MuJoCo CPU. Writes per-seed cpu_best/ + checkpoints; queue.done at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED_DIR="${SEED_DIR:-opt/tendon_seeds}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7}"
PAR="${PAR:-1}"
ITERS="${ITERS:-50}"

run_one() {
  local i=$1
  local seed="$SEED_DIR/seed_${i}.json"
  local log="opt/run_tt_seed_${i}.log"
  [ -f "$seed" ] || { echo "MISSING $seed"; return; }
  echo "=== [$(date -u +%H:%M:%S)] START run $i (seed $seed) ==="
  uv run python opt/optimize_control.py \
    --tg flight-orientations/hovering/shoulder.tg \
    --tune-tendons --tune-motor-tendons --single-stage \
    --lift-torque-tradeoff 5 \
    --tri-integrator --tri-timestep --w-gap 15 --cpu-every 10 --cpu-gap 0.15 \
    --integrator euler --timestep 0.06 --aero-substep 2 \
    --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
    --harmonics 5 --iters-per-stage "$ITERS" --lr 8e-4 \
    --init-tendons "$seed" \
    --out "opt/runs_tt_seed_${i}" > "$log" 2>&1
  echo "=== [$(date -u +%H:%M:%S)] DONE run $i exit $? ==="
}
export -f run_one; export SEED_DIR ITERS

for i in $SEEDS; do
  run_one "$i" &
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do wait -n; done
done
wait
echo "ALL DONE" > opt/queue_tt.done
echo "=== [$(date -u +%H:%M:%S)] TRI-INTEGRATOR+TIMESTEP SWEEP COMPLETE (seeds: $SEEDS) ==="
