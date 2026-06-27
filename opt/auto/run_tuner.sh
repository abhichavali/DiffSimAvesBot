#!/usr/bin/env bash
# Remote entrypoint for the autonomous tuner. Single-instance via flock (no -c,
# so no nested-quote issues). Coexists with the competing agent's GPU process by
# not preallocating GPU memory.
cd /workspace/DiffSimAvesBot
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec flock -n opt/auto/auto.lock uv run python opt/auto_tune.py
