# Agent instructions for this repository

These instructions apply to all coding agents (Cursor, Claude, Gemini, Copilot, etc.) working in this repo.

## Environment & tooling

- Python is managed with **`uv`** (no manual venv activation needed).
  - Install/sync: `uv sync`
  - Run a script: `uv run python <script>`
  - Add a dependency: `uv add <package-name>`
- MuJoCo viewer runs on macOS require **`mjpython`** instead of `python`:
  - `mjpython src/run_hanging_poc.py`

## Common commands

- **Morphology compiler** (`.tg` DSL → MuJoCo XML):
  - Author: `shoulder/shoulder.tg`
  - `uv run python src/compile_morphology.py templates/generated-joint/scene.xml`
  - Optional: `--morphology path/to/file.tg`
  - Tests: `uv run pytest tests/test_morphology_compiler.py tests/test_tg_parser.py`
  - See `docs/MORPHOLOGY_COMPILER.md`
- Run the simulation viewer:
  - `mjpython src/run_hanging_poc.py`
- Use an alternate compiled scene at runtime:
  - `MUJOCO_XML_PATH=templates/shoulder-joint/scene.xml mjpython src/run_hanging_poc.py`
- Use an alternate morphology for runtime aero/wing params:
  - `MORPHOLOGY_PATH=path/to/custom.tg mjpython src/run_hanging_poc.py`

No automated lint suite is configured. Validate morphology changes with
`uv run pytest tests/test_morphology_compiler.py tests/test_tg_parser.py` and,
where relevant, the viewer.

## High-level architecture

The pipeline is **`shoulder/*.tg` → parser → compiler → MuJoCo simulation**.

- `shoulder/shoulder.tg` — authoritative design (geometry knobs, topology, materials)
- `src/morphology/tg_parser.py` — `.tg` → typed `Morphology` dataclass tree
- `src/morphology/` — geometry/symmetry derivation, MJCF emitter
- `src/compile_morphology.py` — CLI entry point
- `src/runtime/morphology_runtime.py` — loads `.tg` for sim-time aero/wing params
- `src/utils/aero/*` — quasi-steady aerodynamic forces (CPU/numpy reference)
- `src/run_hanging_poc.py` — loads compiled XML (`MUJOCO_XML_PATH`; default `templates/generated-joint/scene.xml`), viewer + actuators + aero
- `templates/generated-joint/` — compiled MJCF output target
- `src/runtime/mjx/` — **JAX/MJX port** of the simulation (GPU, jittable, differentiable). See the MJX section below.

## Codebase conventions

- Semantic tendon anchors: `M[station][face]` / `F[station][face]` in `.tg`; compiled to legacy node names (`t1`, `t1_2`, …) in MJCF.
- Six σ-generators expand to 12 structural tendons; R-side is an exact mirror of L.
- Quaternions follow MuJoCo's **w-first** ordering: `[w, x, y, z]`.
- Structural tendon rest lengths are **derived** from posed anchor distance × pretension — never hand-written.
- Do not hand-edit `templates/generated-joint/scene.xml` for design changes; edit `.tg` and recompile.
- Preferred execution: `uv` for compile scripts, `mjpython` for viewer-based MuJoCo runs on macOS.

## MJX / GPU / viewer (summary — see `docs/usage.md` for the full overview)

The simulation is also ported to **MJX** (`src/runtime/mjx/`): a pure-JAX rollout
that is `jit`/`vmap`-able and differentiable for gradient-based optimization. The
CPU path is unchanged and is the reference the port is validated against.

- **Modules**: `wing_pitch.py`, `aero.py` (+ `AeroParams` pytree), `aero_runtime.py`,
  `rollout.py` (`simulate(...)` / `rollout(...)`), `render.py`.
- **Autodiff**: forward mode (`jax.jacfwd`) works as-is; reverse mode (`jax.grad`)
  needs `rollout.set_reverse_mode_solver(model)` / `simulate(..., differentiable=True)`.
  **Geometry params (radius/attach/offsets) are not differentiable** (host-side compiler).
- **Batched**: `jax.vmap` a rollout over a swept param for parallel GPU runs.
- **Tests**: `uv run pytest -q tests/test_mjx_*.py`.
- **GPU server**: SSH alias `HanatServer`; project at `/workspace/Tensegrity-Playground`
  (own `.venv`). Repo is **private** → sync with `rsync` (not `git clone`).
  Instances get evicted often → **`docs/vast_migration.md`** is the quick runbook
  for spinning up a fresh box and re-syncing.
- **Viewer**: headless EGL render → inline video (`notebooks/mjx_viewer.ipynb`,
  kernel `tensegrity-mjx`); plus a **live** interactive WebSocket pixel-stream
  viewer (`src/runtime/mjx/live_viewer.py`, `notebooks/mjx_live_viewer.ipynb`)
  reached via SSH tunnel (no WebRTC — vast is TCP-only).

**Read `docs/usage.md` before doing MJX, GPU-server, or viewer work** — it has the
module APIs, autodiff/solver details, MJCF-prep rationale, gotchas, and exact commands.
This buffer is normal