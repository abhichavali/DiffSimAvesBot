# Agent instructions for this repository

These instructions apply to all coding agents (Cursor, Claude, Gemini, Copilot, etc.) working in this repo.

## Environment & tooling

- Python is managed with **`uv`** (no manual venv activation needed).
  - Install/sync: `uv sync`
  - Run a script: `uv run python <script>`
  - Add a dependency: `uv add <package-name>`
- MuJoCo viewer runs on macOS need **`mjpython`**. It lives in `.venv/bin` (so `uv run
  mjpython`, not bare `mjpython`), and with uv's standalone Python it cannot find
  `libpython` until you point it there:
  - `DYLD_LIBRARY_PATH=$(uv run python -c 'import sys;print(sys.base_prefix)')/lib uv run mjpython src/view_fixed_frequency.py`
  - Headless paths need no mjpython: `view_eagle --gif`, `src/demos/render_*.py`, the
    golden viewer item. The design-inspector GUI's Viewer tab sets the variable itself.

## Common commands

Tools default `--scene` to the repo-root `scene.xml`: pass `--scene <tmp path>` when
running tools concurrently, or they overwrite each other's scene.

- **Compile** a design (`.tg` DSL → MuJoCo XML):
  - `uv run python src/compile_tg.py --tg designs/hummer/hover/shoulder.tg --out scene.xml`
  - With no `--tg` it compiles `designs/hummer/legacy-baseline/` (negative lift, not a hover design).
- **Headless CPU run** (compiles, simulates, prints lift stats):
  - `uv run python src/run_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5`
  - MJX lift stats: `uv run python src/run_mjx_fixed_frequency.py --periods 5` (same
    default design and printout; the numbers differ only by CPU vs MJX).
- **Viewer**: `mjpython src/view_fixed_frequency.py [--tg path/to/shoulder.tg]` (see
  above for the full mjpython command; hover demos: `src/view_hover.py`, eagle:
  `src/view_eagle.py`).
- **Certify** a design (CPU worst-case peak torque, `docs/ROBUST_CODESIGN_METHODOLOGY.md`
  section 12): `uv run python opt/design_io.py designs/hummer/hover --integrator euler implicitfast rk4 --periods 24`
  plus the same at `--timestep 0.03` (euler, implicitfast); the worst peak printed is the number.
- **Tests**: `uv run pytest -q` (unit tests + the golden quick tier).
- **Behavior check**: `uv run python -m tests.golden check [--quick]` must stay all-PASS
  after any refactor; see the docstring at the top of `tests/golden/harness.py`. When code
  or designs move, edit only the harness's ADAPTER section. `capture` rewrites
  `tests/golden/baseline.json`: run it only for a deliberate, reviewed behavior change, in
  its own commit.
- **Renders**: headless MP4 demos in `src/demos/render_*.py` (`--self-check` asserts
  without rendering). The `render_antiphase` and `render_bilateral_control` self-checks
  and the `render_configs --config collision|ceiling --dry-run` runs already fail their
  asserts; compare them before/after, do not expect PASS.

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
  `mjcf_parts.py` (the XML), `wing_geometry.py` (the wing mesh's pose, and the aero
  strips measured from the posed mesh)
- `src/compile_tg.py` — compile CLI (`compile_tg(tg, out)` is what every tool calls)
- `src/runtime/` — CPU simulation runtime: `sim.py` (`PlateWingSim` / `rest_state`, the
  plate-wing step every CPU loop uses, and the canonical force order), `aero.py`
  (quasi-steady blade-element aero, the numpy reference; also the one home of the aero
  coefficients, side signs and `set_unsteady_aero`), `control_signal.py`, `crank.py`,
  `wing_pitch.py`, `disk_orientation.py` (stroke-plane steering, `StrokePlaneSignal`),
  `hover_control.py` (free-flight hover wrench), `eagle_drive.py` / `eagle_poses.py` /
  `wing_cables.py` (eagle wing), `realtime.py` (viewer pacing). Everything takes the
  `Morphology` the scene was compiled from.
- `src/run_fixed_frequency.py`, `src/view_fixed_frequency.py`, `src/view_hover.py`,
  `src/view_eagle.py` — CPU runners/viewers; `src/flying.py` — MJCF surgery that turns
  the bench scene into a free-flying airframe (obstacles, ceiling, bird skin);
  `src/demos/render_*.py` — headless MP4 renders
- `src/runtime/mjx/` — **JAX/MJX port** of the simulation (GPU, jittable, differentiable). See the MJX section below.
- `src/codesign/` — the co-design library (`rig`, `genome`, `metrics`, `loss`, `train`,
  `cpu_check`, `artifacts`); `opt/` — its CLIs (`optimize_control.py`,
  `evolve_codesign.py`, `hover_policy.py`, ...). See `opt/README.md`.
- `plot_fixed_frequency_gui.py` — the Tk design inspector; `src/gui/` — its simulations.
- `tests/` — pytest suite; `tests/golden/` — the golden behavior harness.
- `docs/ROBUST_CODESIGN_METHODOLOGY.md` — how designs are certified (CPU peak torque;
  section 12 is the current doctrine, the rest is the campaign log).

Names that mislead:

| name | what it is |
|---|---|
| `shoulder.tg` | the whole morphology, not just the shoulder |
| `plot_fixed_frequency_gui.py` | the design-inspector GUI (`src/gui/` is its backend) |
| `opt/run_gui.py` | a second, unrelated Tk GUI: the codesign launcher |
| `opt/design_io.py` | CPU replay / certification of a design dir |
| `*_fixed_frequency` | run under the design's own control signal (harmonics included) |

## Codebase conventions

- Semantic tendon anchors: `M[station][face]` / `F[station][face]` in `.tg`; compiled to legacy node names (`t1`, `t1_2`, …) in MJCF.
- Six σ-generators expand to 12 structural tendons. The R-side geometry is an exact
  mirror of L; tendon materials need not be (`overrides:` are per tendon, and the
  flagship's L and R differ).
- Quaternions follow MuJoCo's **w-first** ordering: `[w, x, y, z]`.
- Structural tendon rest lengths are **derived** from posed anchor distance × pretension — never hand-written.
- Do not hand-edit a compiled `scene.xml` for design changes; edit the `.tg` and recompile.
- Preferred execution: `uv` for compile scripts, `mjpython` for viewer-based MuJoCo runs on macOS.

## MJX / GPU

The simulation is also ported to **MJX** (`src/runtime/mjx/`): a pure-JAX rollout
that is `jit`/`vmap`-able and differentiable for gradient-based optimization. The
CPU path is unchanged and is the reference the port is validated against.

- **Modules**: `wing_pitch.py`, `aero.py` (blade strips; coefficients from
  `runtime.aero.AeroParams`), `aero_runtime.py` (`aero_wrench`), `rollout.py`
  (`physics_step`, `build_drive_config`; `simulate(...)` is the golden harness's reference
  rollout). The config builders take the `Morphology`; `codesign.rig.setup` wires them
  for the optimizer, and `codesign.metrics.build_rollout` is the differentiable rollout.
- **Autodiff**: forward mode (`jax.jacfwd`) works as-is; reverse mode (`jax.grad`)
  cannot trace MJX's solver `while_loop` unless `model.opt.iterations == 1`
  (`opt/hover_policy.py` does that). **Geometry params (radius/attach/offsets) are not
  differentiable** (host-side compiler).
- **Batched**: `jax.vmap` a rollout over a swept param for parallel GPU runs.
- **Tests**: `uv run pytest -q`; full behavior check: `uv run python -m tests.golden check`.
- **GPU server**: no box is live. `docs/ROBUST_CODESIGN_METHODOLOGY.md` section 8 has the
  vast.ai setup (rsync to `/workspace/DiffSimAvesBot/`; the repo is private, so no
  `git clone`). On the box always `uv run --no-sync`: plain `uv run` re-syncs and drops
  the CUDA jax.

## Known bugs (left unfixed on purpose: each fix changes outputs)

- Codesign: see `opt/README.md` "Known issues".
- Morphology: degree-conversion heuristics; unknown `.tg` keys are silently accepted;
  the R fixed strut keeps the L site quats; the `attach_along_edge` default is 1.2 in
  one place and 0.0 in another.
- GUI: `run_gui`'s `RunConfig.loss` is never read; a leaked `mkdtemp`.
- `opt/hover_policy.py` writes an absolute `tg` and no `pd_gains` into `hover_meta.json`.
