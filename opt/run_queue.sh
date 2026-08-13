#!/usr/bin/env bash
# Run the 7 seeded codesign optimizations on the 5090, PAR at a time (default 2 --
# a 5090's 32 GB fits ~2 reverse-mode runs at ~8 GB each with prealloc off). Each
# is warm-started from an evolutionary tendon seed. Writes per-run logs + a
# queue.done sentinel when all finish, so the orchestrator can destroy the box.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 2 JAX procs share one GPU

SEED_DIR="${SEED_DIR:-opt/tendon_seeds}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7}"   # explicit seed list, in order
PAR="${PAR:-2}"

run_one() {
  local i=$1
  local seed="$SEED_DIR/seed_${i}.json"
  local log="opt/run_seed_${i}.log"
  [ -f "$seed" ] || { echo "MISSING $seed"; return; }
  echo "=== [$(date -u +%H:%M:%S)] START run $i (seed $seed) ==="
  uv run python opt/optimize_control.py \
    --tg flight-orientations/hovering/shoulder.tg \
    --tune-tendons --tune-motor-tendons --single-stage \
    --lift-torque-tradeoff 5 --cpu-every 5 \
    --integrator euler --timestep 0.06 --aero-substep 2 \
    --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
    --harmonics 5 --iters-per-stage "${ITERS:-70}" --lr 8e-4 \
    --init-tendons "$seed" \
    --out "opt/runs_seed_${i}" > "$log" 2>&1
  echo "=== [$(date -u +%H:%M:%S)] DONE run $i exit $? ==="
}
export -f run_one; export SEED_DIR

for i in $SEEDS; do
  run_one "$i" &
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do wait -n; done
done
wait
echo "ALL DONE" > opt/queue.done
echo "=== [$(date -u +%H:%M:%S)] QUEUE COMPLETE (seeds: $SEEDS) ==="
