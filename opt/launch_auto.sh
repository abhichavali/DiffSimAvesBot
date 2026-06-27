#!/usr/bin/env bash
# Sync the repo to the current vast.ai 5090 and launch the autonomous reward
# tuner (opt/auto_tune.py) DETACHED. Resolves the instance via `vastai ssh-url`
# (the proxy sshN.vast.ai is usually dead -> use the direct host:port). Safe to
# re-run: auto_tune.py resumes from opt/auto/state.json.
#
#   opt/launch_auto.sh                 # resolve instance, sync, launch
#   HOST=1.2.3.4 PORT=22 opt/launch_auto.sh   # explicit endpoint
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${HOST:-}" ]; then
  ID="${VAST_ID:-$(vastai show instances --raw 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["id"])')}"
  URL=$(vastai ssh-url "$ID" 2>/dev/null)          # ssh://root@HOST:PORT
  HOST=$(echo "$URL" | sed -E 's#ssh://root@##' | cut -d: -f1)
  PORT=$(echo "$URL" | sed -E 's#ssh://root@##' | cut -d: -f2)
fi
KEY="${KEY:-$HOME/.ssh/claude_code_ed25519}"
REMOTE="${REMOTE:-/workspace/DiffSimAvesBot}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
echo "instance: root@$HOST:$PORT"

rsync -az --delete -e "$SSH" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'data' --exclude 'ltximg' --exclude 'opt/runs*' --exclude 'opt/auto/run_[0-9]*' --exclude 'opt/auto/run_baseline' --exclude 'opt/auto/state.json' --exclude 'opt/auto/*.jsonl' \
  ./ "root@$HOST:$REMOTE/"

# Absolutize the base design meshdir so a from-anywhere compile finds the mesh.
$SSH "root@$HOST" "cd $REMOTE && sed -i \"s#meshdir *= *assets#meshdir = $REMOTE/assets#\" opt/auto/base/shoulder.tg"

$SSH "root@$HOST" "cd $REMOTE && \
  (command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) && \
  export PATH=\$HOME/.local/bin:\$PATH && \
  uv sync -q && uv pip install -q 'jax[cuda12]' && \
  mkdir -p opt/auto && chmod +x opt/auto/run_tuner.sh && \
  setsid bash opt/auto/run_tuner.sh >> opt/auto/auto.log 2>&1 < /dev/null & \
  sleep 3"
echo "launch attempted; check opt/poll_auto.sh"
