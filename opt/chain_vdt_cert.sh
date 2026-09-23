#!/bin/bash
# Wait for the variable-dt run, then certify it the same way as the queue arms:
# the 6-cell MJX bank, plus CPU at the trained dt (0.06), at the dt where the
# fixed-bank gain evaporated (0.03), and at 0.02 -- finer than anything in the
# widened bank, so it stays a genuine holdout.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NVLIBS="$(find "$PWD/.venv/lib" -path '*nvidia*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while [ ! -f opt/run_swing_rms_vdt.done ]; do sleep 60; done
JAX_PLATFORMS=cpu uv run --no-sync python opt/cpu_cert.py swing_rms_vdt \
  > opt/cpucert_vdt.log 2>&1
uv run --no-sync python opt/certify_arms.py --arms swing_rms_vdt --cpu-dts \
  --out opt/certify_vdt.json > opt/certify_vdt.log 2>&1
echo 0 > opt/vdtcert.done
