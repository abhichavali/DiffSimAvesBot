# Agent instructions for this repository

These instructions apply to all coding agents (Cursor, Claude, Gemini, Copilot, etc.) working in this repo.

## Environment & tooling

- Python is managed with **`uv`** (no manual venv activation needed).
  - Install/sync: `uv sync`
  - Run a script: `uv run python <script>`
  - Add a dependency: `uv add <package-name>`
- MuJoCo viewer runs on macOS require **`mjpython`** instead of `python`:
  - `mjpython src/view_fixed_frequency.py`

## Common commands

- **Compile** a design (`.tg` DSL → MuJoCo XML):
  - `uv run python src/compile_tg.py --tg designs/hummer/hover/shoulder.tg --out scene.xml`
  - With no `--tg` it compiles `designs/hummer/legacy-baseline/` (negative lift, not a hover design).
- **Headless CPU run** (compiles, simulates, prints lift stats):
  - `uv run python src/run_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5`
  - MJX equivalent: `uv run python src/run_mjx_fixed_frequency.py --periods 5`
- **Viewer**: `mjpython src/view_fixed_frequency.py [--tg path/to/shoulder.tg]`
  (hover demos: `mjpython src/view_hover.py`; eagle: `mjpython src/view_eagle.py`)
- **Tests**: `uv run pytest -q` (unit tests + the golden quick tier).
- **Behavior check**: `uv run python -m tests.golden check [--quick]` must stay all-PASS
  after any refactor; see the docstring at the top of `tests/golden/harness.py`.

No automated lint suite is configured.

## High-level architecture

The pipeline is **`designs/<bird>/<name>/{shoulder,control}.tg` → parser → compiler → MuJoCo simulation**.

- `designs/` — the committed designs; **`designs/README.md`** lists each one (what it
  is, lineage, key numbers, which tool defaults to it). `shoulder.tg` is the whole
  morphology, `control.tg` the Fourier crank-velocity command beside it.
- `src/design_paths.py` — tool-default design paths (`DEFAULT_HUMMER_TG`,
  `HOVER_BASE_TG`, `LEGACY_BASELINE_TG`, `EAGLE_TG`). Nothing reads a design from the
  environment: every tool takes `--tg` and hands the parsed `Morphology` down.
- `src/morphology/` — the compiler (start at `compile.compile_file`): `tg_parser.py`
  (`.tg` → the `Morphology` dataclasses of `model.py`), `geometry.py` / `tendon_net.py`
  (strut poses, the 12 σ tendons) → `compile.py` (→ `compiled.py`), `motor_tendons.py`
  (crank tendon ranges, measured on a probe scene), `emit_mjcf.py` / `emit_eagle.py` /
  `mjcf_parts.py` (the XML), `wing_geometry.py` (wing planform from the STL)
- `src/compile_tg.py` — compile CLI (`compile_tg(tg, out)` is what every tool calls)
- `src/runtime/` — CPU simulation runtime: `sim.py` (`PlateWingSim` / `rest_state`, the
  plate-wing step every CPU loop uses), `aero.py` (quasi-steady blade-element aero, the
  numpy reference), `control_signal.py`, `crank.py`, `wing_pitch.py`, `eagle_drive.py`.
  Everything takes the `Morphology` the scene was compiled from.
- `src/run_fixed_frequency.py`, `src/view_fixed_frequency.py`, `src/view_hover.py`,
  `src/view_eagle.py` — CPU runners/viewers; `src/demos/render_*.py` — headless MP4 renders
- `src/runtime/mjx/` — **JAX/MJX port** of the simulation (GPU, jittable, differentiable). See the MJX section below.
- `src/codesign/` — the co-design library (`rig`, `genome`, `metrics`, `loss`, `train`,
  `cpu_check`, `artifacts`); `opt/` — its CLIs (`optimize_control.py`,
  `evolve_codesign.py`, `hover_policy.py`, ...). See `opt/README.md`.
- `plot_fixed_frequency_gui.py` — the Tk GUI; `src/gui/` — its simulations.
- `tests/` — pytest suite; `tests/golden/` — the golden behavior harness.
- `docs/ROBUST_CODESIGN_METHODOLOGY.md` — how designs are certified (CPU peak torque).

## Codebase conventions

- Semantic tendon anchors: `M[station][face]` / `F[station][face]` in `.tg`; compiled to legacy node names (`t1`, `t1_2`, …) in MJCF.
- Six σ-generators expand to 12 structural tendons; R-side is an exact mirror of L.
- Quaternions follow MuJoCo's **w-first** ordering: `[w, x, y, z]`.
- Structural tendon rest lengths are **derived** from posed anchor distance × pretension — never hand-written.
- Do not hand-edit a compiled `scene.xml` for design changes; edit the `.tg` and recompile.
- Preferred execution: `uv` for compile scripts, `mjpython` for viewer-based MuJoCo runs on macOS.

## MJX / GPU

The simulation is also ported to **MJX** (`src/runtime/mjx/`): a pure-JAX rollout
that is `jit`/`vmap`-able and differentiable for gradient-based optimization. The
CPU path is unchanged and is the reference the port is validated against.

- **Modules**: `wing_pitch.py`, `aero.py` (+ `AeroParams` pytree), `aero_runtime.py`,
  `rollout.py` (`physics_step`, `build_drive_config`, `simulate(...)`). The config
  builders take the `Morphology`; `codesign.rig.setup` wires them for the optimizer.
- **Autodiff**: forward mode (`jax.jacfwd`) works as-is; reverse mode (`jax.grad`)
  cannot trace MJX's solver `while_loop` unless `model.opt.iterations == 1`
  (`opt/hover_policy.py` does that). **Geometry params (radius/attach/offsets) are not
  differentiable** (host-side compiler).
- **Batched**: `jax.vmap` a rollout over a swept param for parallel GPU runs.
- **Tests**: `uv run pytest -q`; full behavior check: `uv run python -m tests.golden check`.
- **GPU server**: SSH alias `HanatServer`; project at `/workspace/Tensegrity-Playground`
  (own `.venv`). Repo is **private** → sync with `rsync` (not `git clone`).
- **Renders**: headless MP4 demos in `src/demos/render_*.py` (`--self-check` asserts
  without rendering).

