# Best design — metrics & provenance

Files: `shoulder.tg`, `control.tg` (copies of the promoted `design/` files, commit `b5cfc62`).
Reproduce: `uv run python test_best.py` (from repo root; disables added-mass automatically).

## Validated metrics (added-mass OFF — the transferable regime)

| integrator | mean lift | peak motor torque |
|---|---|---|
| MJX (optimizer) | 208 mN | 103 mNm |
| CPU euler | 208–215 mN | 111–119 mNm |
| CPU implicitfast | **222–228 mN** | **121–127 mNm** |

- Inactive (reactive/braking) motor power: **~0**.
- Hovers for **~23 g** (target 19.5 g).
- Transfer: MJX ≈ CPU-euler (ratio ~1.0).
- Baseline (same morphology, unoptimized control): 96 mN / 211 mNm → **+131 % lift, −43 % torque**.

## Reward config that produced it (exp12)

`τ_max = 100 mNm`, Euler, added-mass OFF, fixed frequency, tune control + 24 tendon params:
```
wL=32.2  wtau=27.6  wK=1  ws=0.005  w_lc≈0.65  w_power=6.74  K=6  lr=5e-3
```

## Pareto frontier (all 16 sweep designs hovered)

| point | lift | torque | note |
|---|---|---|---|
| exp13 | 237 mN | 155 mNm | max lift |
| **exp12** | **222 mN** | **121 mNm** | **best balanced (promoted)** |
| exp11 | 213 mN | 122 mNm | |
| exp14 | 205 mN | 120 mNm | min torque |
| exp6 | 203 mN | 123 mNm | |

Full ledger: `findings/results_ledger.jsonl`.
