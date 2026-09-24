---
name: golden-harness
description: Use when changing any code or design and you need to know whether behavior moved - before committing a refactor, move or rename (must stay all-PASS), when a golden item FAILs, when moving a design or module (edit the ADAPTER/PATHS only), or when a deliberate physics or design change requires recapturing tests/golden/baseline.json.
---

# Golden harness: the behavior contract

`tests/golden/harness.py` recomputes 35 items (parse, compile, CPU rollouts, aero, MJX,
the optimizer CLI) and diffs them with `tests/golden/baseline.json`: exactly for
CPU/numpy, within `JAX_RTOL` (1e-6) for JAX/MJX. Its docstring is the contract.

## Commands

```bash
uv run python -m tests.golden check --quick          # 32 items, no JIT-heavy tier; also run by pytest
uv run python -m tests.golden check                  # all 35 (+ mjx/metrics_jacfwd, mjx/simulate, optimizer/cli), ~40 s+
uv run python -m tests.golden check -k cpu/          # items whose name contains the pattern
uv run python -m tests.golden capture -k cpu/soft    # recompute and merge just those items
uv run pytest -q                                     # unit tests + the quick tier
```

List the items: `uv run python -c "import sys; sys.path[:0]=['.','src']; from tests.golden.harness import ITEMS; print(*ITEMS, sep='\n')"`.
Groups: `parse/<design>`, `model/<design>`, `helpers`, `rest/*`, `cpu/<design>/<integrator>`,
`drive/*` (runners, viewer, GUI, eagle), `aero/cpu`, `aero/mjx`, `mjx/*`, `optimizer/cli`.
A `WARNING: baseline captured with {...}` line means your mujoco/jax/numpy versions differ
from the baseline's; exact items may then fail for that reason alone.

## Rules

- **Refactor, move, rename, docs**: `check` (full, not just `--quick`) must be all-PASS.
  Never recapture to make a refactor pass: a FAIL means behavior moved; find out why.
- **Moving code or designs**: edit only the ADAPTER section of `harness.py` (imports, thin
  wrappers, and `PATHS`, which maps logical names like `shoulder`, `baseline`, `eagle`,
  `soft_it009p`, `tuned_hacked` to current `.tg` paths). Baseline keys are logical
  names, never paths; do not rename them.
- **Deliberate behavior change** (a physics fix, a design edit): in the same commit,
  run `check` to see exactly which items fail, recapture only those with `-k`, run full
  `check` again (all-PASS), and write in the commit body which items moved and how much.
  `capture` aborts and leaves the baseline untouched if any item errors.
- `capture --quick` / `-k` merge into the baseline; items not recomputed keep their old
  values. Never capture the whole baseline to hide an unexplained diff.

## A legitimate recapture commit (repo style)

Subject: imperative, what the change does (`Hold the aero's generalized force between
recomputes, as MJX does`). Body: why; the before -> after table from `opt/design_io.py`
(design x dt: lift / peak / power; see the `certify-design` skill); then e.g.

```
Golden items changed:
- model/* (all 8, eagle included): motor tendon_range only.
- Real dynamics change: cpu/baseline/{euler,implicitfast}, drive/run_fixed_frequency.
- Roundoff only (max rel <= 2e-9): rest/{shoulder,soft_it009p}.
- Unchanged: drive/eagle, aero/mjx, mjx/*.
Golden 35/35 after recapture. pytest NN.
```

Examples to copy: `git log --format='%h %s' -- tests/golden/baseline.json`, then
`git show <hash> --stat`.
