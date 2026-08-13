#!/usr/bin/env bash
# Wavelet co-design from the it009p flagship. Same robust/drift machinery as the
# Fourier campaign (ROBUST_CODESIGN_METHODOLOGY.md sec 11-13) -- ONLY the control
# basis is swapped: Fourier harmonics -> a dyadic Ricker WAVELET frame (localized
# crank-speed atoms). Warm-starts from hovering-soft-it009p/shoulder.tg (tendons +
# motor tendon + 6 deg crank lag baked in); control resets to a pure sinusoid (all
# wavelet atom amplitudes = 0 = the -10.4% flagship waveform). Question under test:
# do TIME-LOCAL crank-speed modulations transfer across integrator x dt where GLOBAL
# harmonics reproducibly never did (sec 13)?
set -euo pipefail
cd "$(dirname "$0")/.."

# jax[cuda13]: point the loader at the bundled nvidia libs in the venv (belt +
# suspenders; jaxlib usually self-locates, but nvvm/pjrt want the path on cu13).
if [ -d .venv/lib/python3.12/site-packages/nvidia ]; then
  export LD_LIBRARY_PATH="$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -path '*/lib' -type d | paste -sd: -):${LD_LIBRARY_PATH:-}"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

exec .venv/bin/python opt/optimize_control.py \
  --tg flight-orientations/hovering-soft-it009p/shoulder.tg \
  --tune-tendons --tune-motor-tendons --tune-pitch --single-stage \
  --wavelet --harmonics 3 \
  --lift-floor 0.330 --lift-torque-tradeoff 5 \
  --rotate-batch 3 --n-batches 6 --batch-seed 17 \
  --dt-bank 0.04 0.06 0.09 0.13 \
  --dt-jitter 0.15 --precision-perturb 1e-7 \
  --softmax-beta-start 60 --softmax-beta-end 300 \
  --full-check-every 25 --full-gap 0.15 --full-sample-periods 24 \
  --cpu-sample-periods 24 \
  --w-drift 60 --drift-gate 0.03 \
  --warmup-bank 8 16 24 \
  --holdout-dts 0.03 0.11 0.16 \
  --control-after 0 \
  --cpu-every 50 --cpu-gap 0.15 --cpu-timestep-factors 0.5 1.5 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --weights 0 10 8 0.03 --w-swing 5 --w-power 3 --w-overlap 0 \
  --score-swing 0 \
  --amp-floor 0.314 \
  --iters-per-stage 150 --lr 2e-4 \
  --out opt/runs_wavelet_it009p_5090
