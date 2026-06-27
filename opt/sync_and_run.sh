#!/usr/bin/env bash
# Sync this (private) repo to the vast.ai 5090 and launch the optimizer DETACHED.
# The remote job runs under setsid so an SSH drop can't SIGHUP it (that killed an
# earlier run at 233/240). This script returns immediately after launching; use
# opt/fetch_results.sh to poll for completion and pull results back.
#
#   opt/sync_and_run.sh --iters-per-stage 60                       # phase 1
#   opt/sync_and_run.sh --tune-tendons --freeze-control \          # phase 2
#       --init-control opt/runs/theta_best.json --out opt/runs_tendons
set -euo pipefail

# Direct SSH endpoint (the ssh7.vast.ai proxy was dead on this instance).
HOST="${HOST:-89.221.67.149}"
PORT="${PORT:-17108}"
KEY="${KEY:-$HOME/.ssh/claude_code_ed25519}"
REMOTE="${REMOTE:-/workspace/DiffSimAvesBot}"
LOG="${LOG:-opt/run.log}"   # remote log path (relative to REMOTE)
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

cd "$(dirname "$0")/.."

rsync -az --delete -e "$SSH" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'data' --exclude 'ltximg' --exclude 'opt/runs*' \
  ./ "root@$HOST:$REMOTE/"

# Sync deps (cached after first run), then launch detached. The `.done` sentinel
# carries the exit code so the poller can tell success from failure.
$SSH "root@$HOST" "cd $REMOTE && \
  (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
  export PATH=\$HOME/.local/bin:\$PATH && \
  uv sync -q && uv pip install -q 'jax[cuda12]' && \
  rm -f $LOG ${LOG%.log}.done && \
  setsid bash -c 'uv run python opt/optimize_control.py ${*:-}; echo \$? > ${LOG%.log}.done' \
    > $LOG 2>&1 < /dev/null & \
  sleep 1; echo LAUNCHED log=$REMOTE/$LOG"
echo "launched detached. poll with: opt/fetch_results.sh"
