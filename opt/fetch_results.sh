#!/usr/bin/env bash
# Poll the detached remote optimizer: show recent log; if the `.done` sentinel
# exists, rsync results back and report the exit code. Safe to run repeatedly
# (the remote job is detached and unaffected by this script).
#
#   opt/fetch_results.sh            # one check (tail + pull-if-done)
set -euo pipefail
HOST="${HOST:-89.221.67.149}"
PORT="${PORT:-17108}"
KEY="${KEY:-$HOME/.ssh/claude_code_ed25519}"
REMOTE="${REMOTE:-/workspace/DiffSimAvesBot}"
LOG="${LOG:-opt/run.log}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
cd "$(dirname "$0")/.."

DONE=$($SSH "root@$HOST" "cat $REMOTE/${LOG%.log}.done 2>/dev/null || echo RUNNING; \
  echo ---; tail -n 4 $REMOTE/$LOG 2>/dev/null" 2>/dev/null)
echo "$DONE"

code=$(printf '%s\n' "$DONE" | head -1)
if [ "$code" != "RUNNING" ]; then
  rsync -az -e "$SSH" --include='runs*/***' --include='runs*/' --exclude='*' \
    "root@$HOST:$REMOTE/opt/" ./opt/ 2>/dev/null || true
  echo "PULLED (exit=$code) -> opt/runs*/"
fi
