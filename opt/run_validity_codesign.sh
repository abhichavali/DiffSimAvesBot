#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# All 50 Adam passes JIT the identical 5-way loss; persistent cache pays compile once.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/workspace/.jax_cache}"

OUT="${OUT:-opt/codesign_validity}"
SEEDS="${SEEDS:-50}"
POP="${POP:-512}"
BATCH="${BATCH:-256}"
GENS="${GENS:-80}"
PAR="${PAR:-2}"
EA_TIMESTEPS="${EA_TIMESTEPS:-0.06}"

mkdir -p "$OUT"

log() {
  printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
}

log "config OUT=$OUT SEEDS=$SEEDS POP=$POP BATCH=$BATCH GENS=$GENS PAR=$PAR EA_TIMESTEPS=$EA_TIMESTEPS"
log "gpu $(gpu_snapshot)"
if [ "${SKIP_EA:-0}" = "1" ]; then
  # Adam-phase-only mode: an EA launched elsewhere writes the seeds; wait for it.
  log "SKIP_EA=1 -- waiting for $OUT/seeds/seeds_summary.json"
  while [ ! -f "$OUT/seeds/seeds_summary.json" ]; do sleep 60; done
else
  log "EA start"
  uv run python opt/evolve_codesign.py \
    --tg flight-orientations/hovering/shoulder.tg \
    --out "$OUT/seeds" \
    --timesteps $EA_TIMESTEPS \
    --pop "$POP" --batch "$BATCH" --gens "$GENS" --n-seeds "$SEEDS" ${EA_EXTRA:-}
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "EA failed rc=$rc"
    exit "$rc"
  fi
fi
log "EA done seed_dirs=$(find "$OUT/seeds" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
log "gpu $(gpu_snapshot)"

run_one() {
  local i=$1
  local seed="$OUT/seeds/seed_${i}/codesign_seed.json"
  local log="$OUT/adam_seed_${i}.log"
  [ -f "$seed" ] || { echo "MISSING $seed"; return; }
  echo "[$(date -u +%H:%M:%S)] ADAM seed $i start seed=$seed"
  # alpha-stall/k-max: 1.05/1.15x the BASELINE alpha_pk=77deg, k=0.33 (hover is
  # inherently past textbook stall; C_valid only blocks going FURTHER invalid).
  uv run python opt/optimize_control.py \
    --tg flight-orientations/hovering/shoulder.tg \
    --tune-tendons --tune-motor-tendons --single-stage \
    --lift-torque-tradeoff 4 \
    --tri-integrator --tri-timestep --w-gap 15 \
    --cpu-every 3 --cpu-gap 0.15 --cpu-timestep-factors 0.5 \
    --integrator euler --timestep 0.06 --aero-substep 2 \
    --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 --w-valid 10 \
    --alpha-stall-deg 81 --k-max 0.38 \
    --lp-power 10 --avg-periods 3 --conv-eps 1e-4 \
    --harmonics 5 --iters-per-stage 15 --lr 8e-4 \
    --init-codesign "$seed" \
    --out "$OUT/adam_seed_${i}" > "$log" 2>&1
  local rc=$?
  echo "[$(date -u +%H:%M:%S)] ADAM seed $i done rc=$rc log=$log"
  return "$rc"
}
export -f run_one
export OUT

# Seed 1 runs ALONE first: N simultaneous cold XLA compiles exhaust process spawn
# ("Failed to launch ptxas", observed at PAR=6); one solo pass populates the JAX
# persistent cache so the parallel rest mostly skip compilation.
log "adam seed 1 (solo, warms compile cache)"
run_one 1
for i in $(seq 2 "$SEEDS"); do
  log "queue adam seed $i"
  run_one "$i" &
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do wait -n; done
done
wait
log "Adam phase done"
log "gpu $(gpu_snapshot)"
echo "ALL DONE" > "$OUT/queue.done"
