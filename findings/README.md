# DiffSim Aves — Optimization Findings

Differentiable co-design of a tensegrity hummingbird (MJX/JAX) — the work of tuning
the optimizer + reward function to produce hovering, low-torque, low-power designs
that **transfer from the MJX optimizer to CPU MuJoCo** (the validation reference).

Branch: `june-26-demo`. Test the result: `uv run python test_best.py`.

---

## TL;DR — Best result

Promoted design (`design/shoulder.tg` + `design/control.tg`, commit `b5cfc62`):

| metric | value | vs baseline (96 mN / 211 mNm) |
|---|---|---|
| mean lift (CPU implicitfast) | **222 mN** | **+131 %** |
| peak motor torque | **121 mNm** | **−43 %** |
| inactive (reactive/braking) power | **~0** | eliminated |

- Hovers for **~23 g** (target was 19.5 g).
- **Transfers**: MJX 208 ≈ CPU-euler 208 mN (ratio ~1.0).
- Pareto frontier mapped: 237 mN @ 155 mNm (max lift) … 205 mN @ 120 mNm (min torque);
  222/121 is the best balanced point.

The two fixes that made this possible were both **diagnoses, not features**: an aero bug
that broke sim-to-sim transfer, and a parameterization bug that crippled the optimizer.

---

## 1. Simulation-fidelity findings (why MJX ≠ CPU)

The central obstacle the whole time was that the MJX optimizer's numbers didn't match
CPU MuJoCo, so "optimized" designs didn't transfer. Three distinct causes, found in order:

### 1a. Aero substep cadence (2× lift error)
MJX recomputed the blade-element aero **every step** with a 1-step finite-difference
window; the CPU runtime recomputes **every 4 steps** and holds (FD window = 4·dt). The
every-step FD injected huge added-mass noise (RMS ~2 N vs ~0.2 N), **halving mean lift**
(106 vs 325 mN) and making gaits look chaotic. Fix: `--aero-substep 4` (recompute-and-hold,
matching CPU). *Implemented in `opt/optimize_control.py:build_rollout`.*

### 1b. Integrator mismatch (`implicitfast` diverges)
MJX (JAX) and CPU (C) implement the **implicit** solve differently, so under `implicitfast`
they diverge per-step (qvel diff ~0.17) and aggressive designs overfit MJX. **Euler is the
one integrator where they agree** (qvel diff ~3e-5) and it is stable for this stiff
tensegrity. Fix: `--integrator euler` (default). *Credit: the competing GPT agent
independently flagged the implicitfast issue.*

### 1c. Added-mass aero term — THE transfer killer (4× lift over-read)
Even with substep + Euler matched, MJX still over-read lift ~2× (and worse, 3.4×, the
harder the optimizer pushed). Step-by-step trace comparison showed: **the dynamics match
perfectly** (crank velocity + wing pitch identical MJX/CPU), so the gap was purely the
**aero force along the trajectory** — specifically the **unsteady/added-mass term**. Its
JAX and numpy implementations diverge ~**4× and with opposite sign** (MJX +211 mN, CPU
−67 mN on a ~160 mN quasi-steady base). Proof: disable it → **MJX 159.7 ≈ CPU 160.5 mN,
ratio 0.995**.

**Fix: `set_unsteady_aero(False)` (default).** This made MJX a faithful CPU proxy and
*also* dropped the limit-cycle variance `C_lc` from 0.6 → 0.007 (the added-mass FD noise
was destabilizing the cycle). Consequence: the honest, transferable lift is the
**quasi-steady ~90–230 mN**; the design's old "~200 mN" with added-mass on was inflation
that did not transfer. (Diagnostic: `opt/compare_mjx_cpu.py`.)

---

## 2. Optimizer-mechanics findings

### 2a. Parameter scaling silently crippled the optimization (the unlock)
`a0` (~63 rad/s) and the Fourier harmonics barely moved at a low LR — a 2e-3 Adam step on
a value of 63 is nothing, so the most important lever (`a_n`/`b_n`) was effectively frozen.
Fix: reparameterize `a0` and frequency as **log-factors**, and the **harmonics as
modulation depths of `a0`** (`u = a0·(1 + Σ aₙcos + bₙsin)`). After this, torque dropped
34 % in 16 iters instead of 7 % in 100. **The harmonics are the dominant control lever.**

### 2b. Frequency is NOT a tunable, and tuning it breaks `C_lc`
The spec allows K=5 Fourier coeffs at a **fixed** fundamental frequency. Tuning frequency
also silently corrupted the limit-cycle term (the cycle reshape used the base period while
the true period changed → artificially low `C_lc`). Removed `--tune-frequency` from the
sweep. *(GPT flagged this.)*

### 2c. Selection logic
Saving the global-min-loss design picked the loose-curriculum (near-baseline) result; fixed
to select the **lowest-peak-torque design that clears the lift floor**.

### 2d. The torque curriculum can collapse lift
Forcing `τ_max → 80 mNm` makes the optimizer cut flapping so hard that lift collapses,
regardless of smoothness. Fix: `--tau-min` makes the tightest budget a parameter. At
`τ_max=100` the design reaches 210–222 mN lift at ~120–135 mNm — a resonant regime where
lift actually *rises* as the budget tightens.

### 2e. Limit-cycle term `C_lc` is a weak gradient lever
Penalizing cross-cycle lift variance barely optimizes (chaos-corrupted variance gradient);
it's in the reward at low weight but the added-mass fix (1c) is what actually stabilized
the cycle.

---

## 3. The reward function (final form)

```
R = wL·Φ_L(L̄) − wτ·Pτ(τ̂) − wK·C_kin − ws·S − w_lc·C_lc − w_power·P_inactive   (maximize)
```
- **Φ_L** — smooth softplus lift floor (penalize below target), normalized.
- **Pτ** — `(max(0, τ̂−τ_max))²/τ_max²`, `τ̂` = soft-max (LSE) of `|τ|`.
- **C_kin** — stroke-amplitude band (forearm/lateral omitted; no clean DOFs).
- **S** — `Σ n²(aₙ²+bₙ²)` smoothness.
- **C_lc** — relative cross-cycle lift variance (stability; weak lever).
- **P_inactive** — **the inactive/reactive motor-power term** (added this round):
  `mean(max(0,−τ·ω)) / mean|τ·ω|` = braking-power fraction. Penalizing it drives elastic
  tendon storage to absorb deceleration instead of the motor → **resonance**. Higher
  `w_power` reliably produced more lift-efficient gaits (201→214→222 mN as w_power rose).
- **Curriculum** `τ_max`: 400 → 200 → (clamped to `--tau-min`, here 100 mNm).
- Run **added-mass OFF**, **Euler**, **fixed frequency**.

---

## 4. The promoted design

- `design/control.tg`: K=6 Fourier velocity-servo command at fixed 10 Hz.
- `design/shoulder.tg`: 12 per-edge structural-tendon **stiffness + pretension** overrides
  (L=R mirrored), via the `.tg` `overrides:` block added this effort.
- CPU-validated: 222 mN / 121 mNm / inactive power ~0, transfers (MJX≈CPU).
- **Reproduce: `uv run python test_best.py`** (disables added-mass automatically).

---

## 5. Tooling built (`opt/`)

- `optimize_control.py` — the differentiable optimizer (forward-mode jacfwd, Adam,
  curriculum, all reward terms, `set_unsteady_aero`, built-in CPU validation).
- `auto_tune.py` — autonomous reward-config sweep (seeds + hill-climb, CPU-gated).
- `cpu_validate.py` / `apply_design.py` / `test_design.py` — CPU validation + design
  materialization (apply optimizer output → compilable `.tg`).
- `compare_mjx_cpu.py` — the MJX-vs-CPU diagnostic that localized the added-mass bug.
- `launch_auto.sh` / `poll_auto.sh` — deploy/poll on the vast.ai GPU (detached, resumable).
- Morphology: per-tendon **color/pretension/stiffness** overrides in `.tg`
  (`src/morphology/`, test `tests/test_tendon_overrides.py`).

---

## 6. Caveats & next steps

- **Added-mass must be off** to reproduce the numbers; it's a capped FD hack whose two
  implementations are irreconcilable. A *correct* unsteady-aero model (matched in both, or
  a proper implicit added-mass state) would let us include real unsteady lift.
- **C_kin** is stroke-only; forearm/lateral excursions need distinct DOFs to measure.
  (Fallback discussed: *value* high flapping amplitude rather than band-cap it — not needed
  once the added-mass fix unlocked lift, but recorded for future.)
- **Residual implicitfast gap**: euler is the optimization/validation integrator; the
  implicitfast CPU number runs a bit higher (228 vs 215 mN) — a real-physics caveat.
- **Sim2Real** remains the headline open problem (per the original org doc).
- Frequency, motor-tendon, and pitch-deadband terms are in the allowed-tunable list but
  were held fixed / left to the morphology; future runs could open them.
