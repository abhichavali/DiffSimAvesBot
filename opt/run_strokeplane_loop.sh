#!/usr/bin/env bash
# Auto-resume wrapper for run_strokeplane.sh: the box's ~171 GB cgroup ceiling
# can OOM-kill the whole process group mid-XLA-compile (observed at v7 it011,
# no traceback, no sentinel). On unexpected death, reseed INIT from the newest
# checkpoint's theta_full.json and relaunch; a clean exit (sentinel written by
# the inner script's own echo) ends the loop.
set -u
cd "$(dirname "$0")/.."
LOG="${LOG:-opt/run_strokeplane.log}"
for attempt in 1 2 3 4 5; do
  echo "=== [$(date -u +%H:%M:%S)] strokeplane attempt $attempt (INIT=${INIT:-none}) ===" >> "$LOG"
  bash opt/run_strokeplane.sh >> "$LOG" 2>&1
  code=$?
  echo "=== [$(date -u +%H:%M:%S)] attempt $attempt exited code=$code ===" >> "$LOG"
  [ "$code" -eq 0 ] && break
  LATEST=$(ls -d "${OUT:-opt/runs_strokeplane}"/checkpoints/it0* 2>/dev/null | tail -1)
  if [ -n "$LATEST" ] && [ -f "$LATEST/theta_full.json" ]; then
    cp "$LATEST/theta_full.json" /workspace/init_resume.json
    export INIT=/workspace/init_resume.json
    echo "=== resuming from $LATEST ===" >> "$LOG"
  fi
  sleep 10
done
echo "$code" > "${LOG%.log}.done"
