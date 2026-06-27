#!/usr/bin/env bash
# Poll the autonomous tuner: show state (best design), recent ledger rows, and
# whether the driver is alive. Pulls state.json + ledger.jsonl back locally.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${HOST:-}" ]; then
  ID="${VAST_ID:-$(vastai show instances --raw 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["id"])')}"
  URL=$(vastai ssh-url "$ID" 2>/dev/null)
  HOST=$(echo "$URL" | sed -E 's#ssh://root@##' | cut -d: -f1)
  PORT=$(echo "$URL" | sed -E 's#ssh://root@##' | cut -d: -f2)
fi
KEY="${KEY:-$HOME/.ssh/claude_code_ed25519}"
REMOTE="${REMOTE:-/workspace/DiffSimAvesBot}"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

$SSH "root@$HOST" "cd $REMOTE && \
  (pgrep -f auto_tune.py >/dev/null && echo ALIVE || echo DEAD); \
  echo '--- state ---'; cat opt/auto/state.json 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);b=d.get(\"best\");print(\"experiments:\",d[\"counter\"]);print(\"baseline:\",d.get(\"baseline\"));print(\"best:\",b[\"cfg\"] if b else None, b[\"score\"] if b else None)' 2>/dev/null; \
  echo '--- last 5 experiments ---'; tail -5 opt/auto/ledger.jsonl 2>/dev/null | python3 -c 'import sys,json;[print(json.loads(l).get(\"i\"), json.loads(l).get(\"kind\"), json.loads(l).get(\"score\")) for l in sys.stdin]' 2>/dev/null; \
  echo '--- tail auto.log ---'; tail -4 opt/auto/auto.log 2>/dev/null" 2>&1 | grep -vE "Welcome|Have fun|AI agents|known hosts|mux_client|Warning"

mkdir -p opt/auto
rsync -az -e "$SSH" "root@$HOST:$REMOTE/opt/auto/state.json" "root@$HOST:$REMOTE/opt/auto/ledger.jsonl" ./opt/auto/ 2>/dev/null || true
