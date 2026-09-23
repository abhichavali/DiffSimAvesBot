#!/bin/bash
# RUN 9 "swing" (2026-09-10): the target opt/screen_targets.py picked.
#
# Runs 7 and 8 both stalled because they optimized a quantity the CONTROL SIGNAL
# cannot move. The screen quantified it on the locked it009p seed, over 6
# (integrator, dt) cells, as elasticity per unit step of the shared Fourier theta:
#
#   metric        d/d a0    d/d harmonics   consensus   <- worst pairwise cosine of
#   tau_swing      203%         3258%         0.996        the harmonic gradient
#   tau_peak       137%         1576%         0.387        across the 6 cells
#   overlap        127%          197%         0.998
#   tau_rms        148%          195%         0.995
#   spec_power     280%           54%         0.990
#   power           44%           15%         0.967
#
# Two things fall out. Power is nearly waveform-DEAF (15%/unit, and a0 is 3x that),
# which is the whole run-7/run-8 verdict in one number: at fixed frequency there is
# nothing there to find. And peak torque is a TRAP -- the waveform has huge authority
# over it (1576%) but consensus 0.387 means each discretization wants it pushed a
# different way, so a gradient step is a bet on the integrator. That is precisely the
# MJX-overfit the CPU gate kept rejecting, now visible before spending a run on it.
#
# Torque SWING is the one target with both: 3258%/unit of waveform authority at
# consensus 0.996 -- every dt and both integrators agree on the direction. So:
# minimize the robust swing (m[7]), normalized to ~1 at the seed, with peak torque
# as a ceiling guard, the lift floor as a wall, and the amplitude band pinning the
# operating point so "less swing" cannot be bought by shrinking the machine.
#
# Same locked-wing seed and robustness machinery as run 8 (--lock-wings, rotate
# batch over euler/implicitfast x dt, full-matrix + CPU transfer gates, drift
# pricing). NO --control-after freeze: run 8 spent its first ten iterations (~80
# min on this contended box) on a frequency-only phase whose answer the screen
# already has in closed form, so the waveform is live from it000.
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
NVLIBS="$(find "$PWD/.venv/lib" -path '*nvidia*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVLIBS}:${LD_LIBRARY_PATH:-}"
# The box is shared: another tenant's sweep holds ~11 of the 16 GB. Take memory on
# demand instead of grabbing 75% up front, and do not fight them for it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=opt/runs_swing
LOG=opt/run_swing.log
DONE=opt/run_swing.done
rm -f "$DONE"
uv run --no-sync python opt/optimize_control.py \
  --tg flight-orientations/hovering-soft-it009p/shoulder.tg \
  --lock-wings \
  --objective swing --tau-guard 4 \
  --tune-tendons --tune-motor-tendons --tune-pitch --single-stage \
  --lift-floor 0.335 --lift-smooth 0.015 \
  --weights 10 10 8 0.03 --w-power 0 --w-overlap 0 --w-swing 0 --score-swing 0 \
  --rotate-batch 2 --n-batches 2 --batch-seed 17 \
  --bank-integrators euler implicitfast \
  --dt-bank 0.06 0.09 0.13 --dt-jitter 0.15 \
  --warmup-bank 4 8 --sample-periods 3 \
  --softmax-beta-start 100 --softmax-beta-end 300 \
  --full-check-every 15 --full-gap 0.15 --full-sample-periods 12 --drift-gate 0.03 \
  --w-drift 60 \
  --holdout-dts 0.03 0.11 0.16 \
  --a0-free \
  --cpu-every 30 --cpu-gap 0.15 --cpu-timestep-factors 0.5 --cpu-sample-periods 24 \
  --integrator euler --timestep 0.06 --aero-substep 1 \
  --amp-band-pct 0.05 \
  --harmonics 3 --iters-per-stage 80 --lr 4e-4 \
  --out "$OUT" >"$LOG" 2>&1
echo $? > "$DONE"
