# Co-design optimization (`opt/`)

Gradient and evolutionary co-design of the hummer rig on the differentiable MJX
simulation. The CLIs live here; the library is `src/codesign/`.

| File | What it is |
|---|---|
| `optimize_control.py` | Gradient co-design CLI (argparse -> `codesign.train.run`) |
| `evolve_codesign.py` | EA over the same genome, optional `.tg` geometry genes (`--geom-genes`) |
| `run_gui.py` | Tk codesign launcher: GA -> Adam chain over the two CLIs above (not the design-inspector GUI, `plot_fixed_frequency_gui.py`) |
| `design_io.py` | Replay or certify a design dir (optimizer output or committed `designs/`) on MuJoCo CPU (`codesign.artifacts.validate`) |
| `hover_policy.py` | Free-flight hover model + BPTT trainer (controller: `src/runtime/hover_control.py`) |
| `tg_scalars.py` | `.tg` scalar flatten/substitute for the GUI and geometry genes |
| `diag_mjx_cpu_parity.py`, `harmonic1_lift_sweep.py` | MJX<->CPU step diff; first-harmonic lift sweep |

`src/codesign/`: `rig.py` (compile + settle a `.tg`, integrator/dt variants, tendon
pairs), `genome.py` (the packed theta), `metrics.py` (theta -> `Metrics` rollout),
`loss.py` (robust reward, `--objective`), `cpu_check.py` (CPU replay, transfer gate),
`artifacts.py` (design dirs: the writer and the reader), `train.py` (the run).

## Run

Smoke test (CPU, ~15 s warm, up to a minute cold):
```sh
uv run python opt/optimize_control.py --tg designs/hummer/hover-soft-it009p/shoulder.tg \
    --warmup-periods 0.2 --sample-periods 0.2 --iters-per-stage 1 --harmonics 2 --out /tmp/oc
```
It only proves the pipeline runs: on 0.2-period windows the metrics are meaningless
(a ~37000 mNm `tau_hat` baseline, `best tau=inf`, `swing=nan` are expected).
The default `--tg` is the codesign seed, `designs/hummer/hover`. On a GPU box rsync the repo over and use `uv run --no-sync` (plain
`uv run` re-syncs and drops CUDA jax). The full campaign commands are in
`docs/ROBUST_CODESIGN_METHODOLOGY.md` section 6.

Compile time is the number of distinct graphs: one `jacfwd` per loss (per rotating batch
with `--rotate-batch`, the slowest, and slower still with rk4 in it), one forward per
full-matrix cell (`--bank-integrators` x `--dt-bank`) and per holdout cell (3 x
`--holdout-dts`). On a GPU these took 30-50 min per run. `export
JAX_COMPILATION_CACHE_DIR=~/.cache/jax` keeps the compiled graphs across runs with
identical results (a repeated CPU smoke run: 17 s -> 9 s).

## The genome

theta packs the active groups in this fixed order (`codesign.genome.Genome`):

```
[control 2K+1: a0_log, a_1..a_K, b_1..b_K]  unless --freeze-control
[f_log 1]                                  --tune-frequency
[stiff n][pre n]                           --tune-tendons (n = 24, or 12 with --lock-wings)
[pitch 4]                                  --tune-pitch
[motor 2]                                  --tune-motor-tendons
[sp 10]                                    --tune-strokeplane
```

a0 and every tendon/pitch/motor gene are log-factors of the design base (seeded 0 = 1x).
The command is `u(t) = a0 * (1 + sum a_n cos(n w t) + b_n sin(n w t))` on the crank-locked
clock `w = omega*exp(f_log + a0_log)`, which `theta_best.json` and `control.tg` record: inside theta
a_n/b_n are modulation DEPTHS; `theta_best.json` stores them ABSOLUTE (x a0). With
`--wavelet` the 2K coefficients are dyadic Ricker atom amplitudes instead.
`--init-codesign` reads a genome BY POSITION as `[control][tendons 2n][motor 2]...[sp 10
last]` (an `evolve_codesign` genome or a `theta_full.json`).

## The reward

`make_loss` returns `jit(jacfwd(loss))`: forward-mode gradients (MJX's solver loop is not
reverse-mode differentiable, and the parameter count is small). The loss aggregates the
variant rollouts of one design with a soft max/min (`--softmax-beta-start/end`):

```
R = wL*PhiL(soft-min lift) - wtau*Ptau(soft-max tau_hat) - wK*Ckin(amp band) - ws*S
    - w_swing*(swing/tau_max)^2 - w_lc*C_lc - w_power*P_inactive - w_overlap*C_overlap
    - w_gap*((max-min tau_hat)/tau_max)^2 - w_drift*max|drift|/tau_max          (maximized)
```

- **Lift**: one-sided soft floor `--lift-floor` (default 300 mN, smoothing
  `--lift-smooth`); optional `--lift-cap`. With `--lift-torque-tradeoff N` the floor is
  replaced by an N:1 peak-torque:lift trade anchored at the floor.
- **Amplitude**: two-sided band of `--amp-band-pct` around the seed's stroke amplitude, or
  the one-sided absolute `--amp-floor`.
- **Peak torque**: `tau_hat` is the soft-max over cycles of each cycle's Lp10 peak (the
  gradient surrogate); `tau_peak` is the exact per-cycle max, the certification metric.
  Judge designs by `tau_peak` on the full MJX matrix and on CPU replay, never by
  `tau_hat`.
- **Curriculum**: `tau_max` 300 -> 250 -> 200 -> 150 -> 100 mNm, clamped at `--tau-min`,
  `--iters-per-stage` each. `--single-stage` runs one stage at the baseline `tau_hat` and
  selects on objective (+ `--score-swing` x swing, converted to the objective's units
  as swing / baseline `tau_hat` x baseline objective) inside the amplitude band.
- `--w-lc` defaults to 0 (the cross-cycle variance gradient does not help); `C_lc`,
  `C_valid` and the rest of `Metrics` are logged either way.

### `--objective`

- **`torque`** (default): robust peak motor torque.
- **`power`**: robust mean SUPPLIED motor power (an exact window mean, so it can't be
  narrowed by duty-cycle tricks), normalized by the baseline. Peak torque drops to a soft
  ceiling guard (`--tau-guard`, default 4).
- **`energy`**: robust specific power P/L [W/N], lift soft-min in the denominator floored
  at `--lift-floor`. A linear `P - lambda*L` credit fixes the exchange rate, so any knob
  that buys lift below lambda W/N is a score fountain (run 7 farmed exactly that); the
  ratio prices lift at the design's own average P/L and is self-limiting. Do not add
  `--lift-torque-tradeoff` on top (make_loss zeroes the linear credit off `torque`).
- **`swing`, `overlap`, `tau_rms`, `power_crest`, `lift_ripple`**: that `Metrics` field,
  robust, over its baseline. **`swing_rms`**: 50/50 swing + tau_rms.

Selection scores and the full-matrix / CPU gates follow the same objective, and under a
non-torque objective a power spread wider than the gap threshold rejects a candidate the
way a torque spread does. Self-check: `uv run pytest -q tests/test_reward.py`.

## Robustness and certification

- `--tri-integrator` (euler + implicitfast + rk4) and `--tri-timestep` (euler at dt x
  `--tri-timestep-factors`, default 0.5 2.0): every loss evaluation rolls those variants.
- `--rotate-batch K`: a bank of (`--bank-integrators` x `--dt-bank`) cells; each iteration
  draws one of `--n-batches` random K-cell mini-batches (with `--dt-jitter`,
  `--warmup-bank`, `--precision-perturb`). Every `--full-check-every` iterations the full
  bank is evaluated on `--full-sample-periods` (set >= one torque-beat period, ~24):
  candidates within `--full-gap` spread, stationary (`--drift-gate`) and holding lift
  become `robust_best/`. `--holdout-dts` are evaluated only at the end.
- `--cpu-every N`: the CPU transfer gate. The current design runs on MuJoCo CPU under the
  run integrator (+ implicitfast/rk4 in tri/rotate mode, + `--cpu-timestep-factors`);
  within `--cpu-gap` and stationary it can become `cpu_best/`.
- `--control-after N` freezes the control genes (all but a0 with `--a0-free`) and the
  stroke plane for the first N iterations of each stage and snapshots `preharmonics/`
  (and the `*_preharmonics` bests) at the first stage's unlock.
- `--lock-wings` geometric-means each L/R tendon pair at load and tunes 12 mirrored
  pairs instead of 24 independent tendons, so both wings stay one design. Check:
  `uv run pytest -q tests/test_wing_lock.py`.
- `--aero-substep` (default 4, the CPU runtime's cadence) recomputes aero every N steps
  and holds it in between; euler (the default `--integrator`) is the integrator where MJX
  and CPU MuJoCo agree per step. `--added-mass` turns on the unsteady aero term in both.

Winner priority at the end of a run: robust_best > cpu_best > mjx best; the holdout uses
the first one that exists.

## Outputs (`--out`, default `opt/runs/`)

Each design dir (the run root = running best, `stage_<N>mNm/`, `checkpoints/itNNN/`
every 3 iterations counted across stages, `cpu_best/`, `robust_best/`, `*preharmonics/`) holds `shoulder.tg`
(control repointed), `control.tg`, `theta_full.json` (the raw theta: an
`--init-codesign` seed), `theta_best.json`, `tendon_optimized.json`,
`motor_tendon_optimized.json`, `pitch_optimized.json`, `strokeplane_optimized.json` and
`log.json`; the run root also gets `cpu_validation.json`. The layout is documented in
`src/codesign/artifacts.py`, which also reads it back:

```sh
uv run python opt/design_io.py opt/runs/cpu_best --integrator euler implicitfast --timestep 0.06
```

`design_io.py` also replays a committed design dir on its own `control.tg`; with
`--periods 24` it runs the certification doctrine of `docs/ROBUST_CODESIGN_METHODOLOGY.md`
section 12 (see its docstring). The seed `designs/hummer/hover` certifies at 193.1 mNm
worst case this way, at 187-210 mN lift: its passive pitch settles into a ~190 mN or a
~210 mN mode depending on dt and integrator. (Section 12's 174.7 mNm / 358-364 mN came
from the campaign's own scripts, on an aero that assumed the hinge edge always leads and
put the hinge on the leading edge of a wing starting at the hinge, not on the drawn wing.)

## Known issues (kept as-is for now; each fix changes outputs)

- `--lift-floor` (and `run_gui`'s `lift_floor`) default to 300 mN, picked when the aero
  assumed the hinge edge always leads and sat on the leading edge. The seed now makes
  187-210 mN, so the floor binds from the first step; pick a new one before the next run.
- MJX sizes its warmup/sample windows by the design period, not the crank-locked one
  (step counts are static, a0_log is not), so once a0 moves its window is not whole flap
  cycles while the CPU gate's is: at a0_log = 0.05, mean power reads 1.5% apart. (`C_lc`
  alone is resampled onto the crank's own cycles.)
- `--init-codesign` slices by position, so a `theta_full.json` from a `--tune-pitch` or
  `--tune-frequency` run is mis-decoded.
- `evolve_codesign.py` hard-codes K=5 harmonics.
- `--wavelet` writes a `control.tg` (`basis = wavelet`, `w_j = ...`) that no runtime tool
  loads (`load_control` rejects it), and `design_io` KeyErrors on its `theta_best.json`.
- The validity-penalized Adam stage (`--w-valid`, `--alpha-stall-deg`, `--k-max`, ...)
  and `--init-tendons` were removed with `opt/run_validity_codesign.sh`; `C_valid` is
  still computed and logged but no longer enters the loss. Restore from git history
  (before commit 0b4f12d) if that campaign is rerun.
