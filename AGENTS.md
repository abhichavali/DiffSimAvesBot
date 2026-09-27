# Agent guide for DiffSimAvesBot

The canonical instructions for every coding agent (Claude, Codex, Cursor, Copilot, Gemini, ...)
working in this repo. `CLAUDE.md` imports this file; keep tool-specific notes out of here.

## What this is

A tensegrity flapping-wing robot in simulation. A `.tg` design (morphology + Fourier
crank-velocity control) compiles to MuJoCo XML and runs on a CPU reference simulator
(numpy quasi-steady blade-element aero), on a differentiable JAX/MJX port of it, and
under gradient and evolutionary co-design that tunes the control and the tendon net.

Pipeline: `designs/<bird>/<name>/{shoulder,control}.tg` -> `src/morphology` (parser +
compiler) -> MJCF -> `src/runtime` (CPU step; `mjx/` port) -> `src/codesign` + `opt/`.
Free flight: `src/flying.py` turns the bench rig into a free-standing bird and `src/hover/`
flies it (hover policy in `designs/hummer/free-hover/control.tg`).

## Documentation is part of the change

**Every decently sized change updates the docs in the same commit** (at the latest before
its branch merges). Decently sized = anything a new agent would otherwise get wrong:

- it adds, moves, renames or deletes a module, CLI, design, file format or workflow;
- it changes behavior, a default, a flag, or a design's numbers;
- it finds a gotcha, a known bug, or a result that overturns an earlier claim.

A result, decision or negative result (something tried that failed) gets its own
`docs/findings/` file (template: `docs/findings/README.md`). A finding that gets
overturned is not deleted: set its `status: superseded` and `superseded_by:`.

Update the one place that owns the fact (link to it elsewhere, don't copy it):

| doc | owns |
|---|---|
| `AGENTS.md` (this file) | the map: layout, commands, conventions, gotchas, known bugs. Keep it short |
| module docstrings | what the module does and the command that runs it |
| `.claude/skills/*/SKILL.md` | step-by-step workflows (see Skills) |
| `designs/README.md` | one row per design: what it is, lineage, certified numbers, which tools default to it |
| `opt/README.md` | optimizer genome, reward, flags, outputs; Known issues (synced with Known bugs here) |
| `docs/findings/` | one dated file per finding, decision or negative result, with a status (current / re-check / superseded) |
| `docs/flight_demos.org` | the free airframe, free-flight status and findings, the renders |
| `README.md` | the human quickstart |
| `docs/history/` | frozen campaign logs: never update them, never cite their numbers as current |

A doc claim that no longer matches the code is a bug: fix or delete it, don't annotate
around it. Run every command you document.

## Skills (step-by-step workflows)

Plain Markdown in `.claude/skills/<name>/SKILL.md`. Claude Code loads them by itself;
any other agent should read the matching one before starting that kind of task.

| skill | when |
|---|---|
| `certify-design` | trustworthy CPU numbers for a design or optimizer result, and why the rules are what they are |
| `run-codesign` | optimizer smoke test and real runs, reading outputs, promoting a result into `designs/` |
| `free-flight-policy` | the free-standing bird: how the hover policy works, re-deriving it after any change, verifying it |
| `gpu-box` | vast.ai boxes: rent, sync the private repo, CUDA jax, launch / monitor / fetch, destroy |
| `golden-harness` | check / capture rules, ADAPTER and `PATHS`, what a recapture commit says |
| `add-or-edit-design` | `.tg` dialect, strict parser, bird tags, overrides, flight-control keys, tool defaults |

`.claude/agents/librarian.md` is a read-only lookup subagent over all of the above plus
`docs/findings/` and git history: ask it instead of reading long docs into your own
context (any agent that can spawn subagents can use its prompt).

## Layout

| path | purpose | start at |
|---|---|---|
| `designs/` | committed designs: `shoulder.tg` = the whole morphology, `control.tg` = the crank command | `designs/README.md` (every design, lineage, certified numbers, which tool defaults to it) |
| `src/design_paths.py` | tool-default designs | `DEFAULT_HUMMER_TG` (hummer/hover), `FREE_HOVER_TG` (hummer/free-hover), `HOVER_BASE_TG`, `LEGACY_BASELINE_TG`, `EAGLE_TG` |
| `src/morphology/` | `.tg` -> MJCF compiler | `compile.compile_file`; `tg_parser.load_morphology` -> `model.py` dataclasses; `geometry.py`/`tendon_net.py` (strut poses, 12 sigma tendons); `motor_tendons.py`; `emit_mjcf.py`/`emit_eagle.py`/`mjcf_parts.py`; `wing_geometry.py` (posed wing mesh -> aero strips, hinge offset) |
| `src/compile_tg.py` | compile CLI | `compile_tg(tg, out)`, what every tool calls |
| `src/runtime/` | CPU runtime (the reference) | `sim.py` (`PlateWingSim`, `rest_state`: the canonical step and force order); `aero.py` (blade-element aero, the one home of aero coefficients and `set_unsteady_aero`); `control_signal.py`, `crank.py`, `wing_pitch.py`, `disk_orientation.py`, `eagle_*.py`, `wing_cables.py`, `realtime.py` |
| `src/runtime/mjx/` | JAX/MJX port (jit/vmap, differentiable) | `aero.py`, `aero_runtime.py` (`aero_wrench`), `rollout.py` (`physics_step`, `build_drive_config`, `simulate`) |
| `src/codesign/` | co-design library | `rig`, `genome`, `metrics` (`build_rollout`), `loss`, `train`, `cpu_check`, `artifacts` (design dirs: write + read back) |
| `src/hover/` | free-flight hover of the standalone bird | `launch` (phase-locked drive, launch cradle, strobed state, `fly`), `policy` (flight control `.tg` + MLP load/write), `trim` / `lqr` (CLIs), `env` (MJX closed loop); the `free-flight-policy` skill |
| `opt/` | co-design and free-flight CLIs | `optimize_control.py`, `evolve_codesign.py`, `design_io.py` (CPU replay/certify), `run_gui.py`, `diag_mjx_cpu_parity.py`; hover: `verify_hover.py`, `push_sweep.py`, `distill_hover.py`, `train_hover.py`; see `opt/README.md` |
| `src/*.py` | runners and viewers | `run_fixed_frequency.py`, `run_mjx_fixed_frequency.py`, `view_fixed_frequency.py`, `view_eagle.py`, `view_free_hover.py` (free flight under a flight control); `flying.py` (bench scene -> free-flying airframe) |
| `src/demos/` | headless MP4 renders | `render_*.py` (`--self-check` asserts without rendering) |
| `plot_fixed_frequency_gui.py`, `src/gui/` | Tk design inspector and its simulations | |
| `tests/`, `tests/golden/` | pytest suite; golden behavior harness | `tests/golden/harness.py` docstring |
| `docs/` | `findings/` (dated findings, decisions, negative results); `flight_demos.org` (free airframe, free-flight status and findings, renders); `history/` (frozen campaign logs, pre-2026-09 numbers) | |
| `assets/wing.stl` | the wing mesh the aero strips are measured from | |

Names that mislead:

| name | what it is |
|---|---|
| `shoulder.tg` | the whole morphology, not just the shoulder |
| `plot_fixed_frequency_gui.py` | the design-inspector GUI (`src/gui/` is its backend) |
| `opt/run_gui.py` | a second, unrelated Tk GUI: the codesign launcher |
| `opt/design_io.py` | CPU replay / certification of a design dir |
| `*_fixed_frequency` | run under the design's own control signal (harmonics included) |

## Commands

Python is managed with `uv`: `uv sync`, `uv run python <script>`, `uv add <pkg>`. No lint
suite is configured. Most CLIs take `--tg` (defaults live in `src/design_paths.py`) and `--help`;
exceptions: `opt/design_io.py` takes a design dir and `plot_fixed_frequency_gui.py`
parses no arguments (`--help` launches the GUI).

```bash
uv run python src/compile_tg.py --tg designs/hummer/hover/shoulder.tg --out scene.xml
uv run python src/run_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5   # CPU lift stats
uv run python src/run_mjx_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5  # same, on MJX
uv run python opt/design_io.py designs/hummer/hover --integrator euler   # CPU replay: lift / peak torque / power
uv run python plot_fixed_frequency_gui.py         # design inspector (Tk)
uv run python opt/optimize_control.py --help       # gradient co-design; see opt/README.md
uv run python opt/verify_hover.py                  # free-standing bird: fly its hover policy on CPU, with pushes
uv run pytest -q                                   # unit tests + the golden quick tier (~50 s)
uv run python -m tests.golden check [--quick]      # behavior check (see below)
```

- `compile_tg.py` with no `--tg` compiles `designs/hummer/legacy-baseline/` (the June
  demo baseline, not a hover design). Everything else defaults to `designs/hummer/hover`.
- **Certify** a design (CPU worst-case peak torque; the `certify-design` skill): run
  `opt/design_io.py <dir> --integrator euler implicitfast rk4 --periods 24` and the same
  with `--timestep 0.03 --integrator euler implicitfast`; the worst peak printed is the
  number. Compare lift per passive-pitch mode (see Gotchas).
- **Viewers** (`src/view_fixed_frequency.py`, `src/view_eagle.py`, `src/view_free_hover.py`) need `mjpython` on
  macOS; see Gotchas for the command. Headless alternatives:
  `view_eagle.py --gif out/eagle.gif`, `src/demos/render_*.py`.
- **Renders**: `src/demos/render_*.py --self-check`. The free-flight clips
  (`render_configs`, `render_flight_phases`, `render_stroke_plane`) apply no assist
  forces and do not run the hover policy yet, so the bird falls (Known bugs). The `render_antiphase` and `render_bilateral_control` self-checks and
  `render_configs --config collision|ceiling --dry-run` already fail their asserts;
  compare before/after, do not expect PASS.
- Tools default `--scene` to the repo-root `scene.xml`: pass `--scene <tmp path>` when
  running tools concurrently, or they overwrite each other's scene.

## Conventions

- **Designs**: a design is a directory `designs/<bird>/<name>/` with `shoulder.tg` and the
  `control.tg` it names (`control = control.tg`, resolved beside it; a relative
  `meshdir = assets` resolves against the repo root). Add a design as a new dir and a row
  in `designs/README.md`; point tool defaults at it only through `src/design_paths.py`.
  Nothing reads a design from the environment: tools take `--tg` and pass the parsed
  `Morphology` down.
- **Bird tags**: every `.tg` starts with `bird = hummer` (plate wing) or `bird = eagle`
  (three-linkage bone wing). Tools refuse a design of the wrong bird and a control of
  another bird (`require_bird`).
- **Strict `.tg` parser**: unknown keys, unparseable lines, brace parts without `:` and
  overrides that name no tendon all raise. Literals carry their unit: `mm`, `cm`, `g`, `ms`
  are scaled to SI; an angle converts to radians when its literal says `deg` (station
  angles and `arrange.angle` stay degrees by design; see the `tg_parser.py` docstring).
  Keys old files carry that nothing reads (`force_scale`, `body_torque`, `cop`, disk
  `outboard`, ...) are accepted and ignored so old files still parse.
- **Tendons**: `M[station][face]` / `F[station][face]` anchors in `.tg` compile to legacy
  node names (`t1`, `t1_2`, ...). Six sigma-generators expand to 12 structural tendons; the
  R-side geometry mirrors L exactly, materials need not (`overrides:` are per tendon; the
  flagship's L and R differ). Structural rest lengths are **derived** from posed anchor
  distance x pretension, never hand-written.
- **Quaternions** are MuJoCo **w-first**: `[w, x, y, z]`.
- **Harmonic clock**: `data.time` is reset to 0 after the rest settle, so the control's
  harmonics start at rollout start. CPU replay and export use the MJX rollout's
  crank-locked clock (`Genes.clock_factor()` = exp(f_log + a0_log)). Old claims that
  "harmonics are dt-specialized" were this clock bug. Harmonic artifacts written before
  the reset (649cab1, 2026-09-24) drove from time 2000*dt: only the four committed
  harmonic designs were rotated. Replay any other (a run dir's `theta_best.json` or
  `control.tg`) with `opt/design_io.py <dir> --pre-clock-fix-dt <the run's --timestep>`,
  and rotate an old `theta_full.json` / `codesign_seed.json` before `--init-codesign`:
  a' = a cos(th_k) + b sin(th_k), b' = b cos(th_k) - a sin(th_k), th_k = k*w*2000*dt
  (w = the file's `omega`, or 2 pi f*exp(f_log + a0_log) for a genome). New JSON artifacts
  carry `"time_origin": "settle_end_reset"`; loading a harmonic one without it warns.
- **Aero** follows the mounted wing mesh (strips, per-strip hinge offset, root offset come
  from `wing_geometry.py`); lift follows the chordwise flow whichever edge leads; aero is
  recomputed every `aero_substep` (default 4) steps and its generalized force held in
  between: CPU `PlateWingSim`, the codesign MJX rollout and `run_mjx_fixed_frequency`
  hold the same way. MJX's `build_aero_config(substep=)` only sizes the added-mass
  finite-difference window; the hold is the caller's, and `rollout.simulate` recomputes
  every step. Keep the CPU (`runtime/aero.py`) and MJX (`runtime/mjx/aero.py`)
  implementations consistent: change both or neither, then run
  `uv run python opt/diag_mjx_cpu_parity.py --steps 40` (per-step CPU vs MJX diff).
- **No fake forces**: aero and the wing-pitch limiter are the only terms in
  `qfrc_applied`; free flight gets no assist wrench, weight cancel or scaled gravity. The
  bird is flown only through its four crank motors.
- Do not hand-edit a compiled `scene.xml`; edit the `.tg` and recompile.
- Commits: one change per commit, imperative subject in the repo's style (`git log`).

## Free flight: flight control files (`src/hover/`)

- `designs/hummer/free-hover/` is the free-standing bird (226 mN; `flying.to_airframe`). A **flight control**
  `.tg` carries its whole policy: per-crank harmonics `a_<n>.<crank>` / `b_<n>.<crank>` (absolute rad/s,
  cranks `L_t1 L_t4 R_t1 R_t4`), `hover_attitude = w x y z` (release attitude), `phase_lock` (crank encoder
  lock, 1/s) and optionally `closed_loop = <mlp>.npz` (per-wingbeat MLP; omit = open loop). Plain controls
  are unchanged; `drive_command` returns one command per crank only when a file has per-crank keys.
- Load/write with `hover.policy` (`load_flight_policy`, `write_flight_control`). `hover.launch` = the
  phase-locked drive, launch cradle (holds the mean pose while the flapping settles; removed at release)
  and strobed per-wingbeat state. No force of any kind acts after release.
- Tools: `PYTHONPATH=src uv run python -m hover.trim --out new.tg` (open-loop trim),
  `PYTHONPATH=src uv run python -m hover.lqr --out lqr.npz`, `opt/distill_hover.py` (LQR -> MLP),
  `opt/verify_hover.py` / `opt/push_sweep.py` (CPU checks), `mjpython src/view_free_hover.py`
  (P = push, O = toggle closed loop). The GUI's Viewer tab launches it for a flight control.
  `opt/train_hover.py` (MJX BPTT) is kept as a negative result: gradients through more than one
  wingbeat of this model diverged, from scratch and from the LQR start.
- **Any change to the free bird or its physics invalidates its policy**: re-derive it
  (trim -> LQR -> distill -> verify, ~3 min on a laptop) with the `free-flight-policy` skill.

## Behavior contract: the golden harness

`tests/golden/` recomputes every piece of repo behavior (parse, compile, CPU rollouts,
aero, MJX, the optimizer CLI) and compares it with `tests/golden/baseline.json`: exactly
for CPU/numpy, within `JAX_RTOL` for JAX/MJX. The harness docstring is the contract.

- `uv run python -m tests.golden check [--quick] [-k PATTERN]` diffs against the
  baseline; `--quick` skips the JIT-heavy full tier (`mjx/*`, `optimizer/cli`), and
  `uv run pytest -q` runs the quick tier. After any refactor, `check` (full) must be
  all-PASS.
- `capture [--quick] [-k PATTERN]` merges recomputed items into `baseline.json`. A
  recapture is legitimate only for a deliberate, reviewed behavior change (a physics fix,
  a design edit): recapture just the affected items with `-k`, in the commit that makes
  the change, and list which items moved and by how much in the commit body. Never
  recapture to make a refactor pass.
- When code or designs **move**, edit only the harness's ADAPTER section (imports, thin
  wrappers, and `PATHS`, which maps logical design names like `shoulder`, `baseline`,
  `eagle`, `soft_it009p` to their current `.tg` paths). Baseline keys are logical names,
  never paths.

## Gotchas

- **mjpython on macOS**: the MuJoCo viewers need `mjpython`, which lives in `.venv/bin`
  (so `uv run mjpython`), and with uv's standalone Python it cannot find `libpython`
  ("failed to dlopen ... libpython3.12.dylib") until you point it there:
  `DYLD_LIBRARY_PATH=$(uv run python -c 'import sys;print(sys.base_prefix)')/lib uv run mjpython src/view_fixed_frequency.py`.
  The GUI's Viewer tab sets the variable itself.
- **Tk (GUIs)**: if `tk.Tk()` fails with "Can't find a usable init.tcl" (sandboxed shells),
  export `TCL_LIBRARY=$B/lib/tcl8.6` and `TK_LIBRARY=$B/lib/tk8.6` with
  `B=$(uv run python -c 'import sys;print(sys.base_prefix)')`.
- **Parallel runs on one design**: `codesign.rig.setup` holds `opt/.compile.lock` (flock)
  across compile + load, but `opt/design_io.py` / `artifacts.reconstruct_design` write
  `_recon_scene.xml` inside the design dir with no lock. Two `design_io` processes on the
  same design dir (e.g. the two certification commands) race on that file: run them one
  after the other, or on a copy of the dir. Likewise pass distinct `--scene` paths.
- **Golden harness off the capture platform**: the CPU items compare exactly, so on Linux x86
  (Python 3.13, mujoco 3.10.0, jax 0.10.2) the quick tier already fails 23 of 32 items at a clean
  checkout (model/rest/cpu/drive/aero). There, compare `check --quick` output before and after a change
  (it must be identical) instead of expecting all-PASS.
- **JIT cost**: every MJX entry point pays an XLA compile first (a few seconds for
  `run_mjx_fixed_frequency`, ~10-20 s per golden full-tier item on a laptop CPU, minutes
  for a tri-integrator / rotating-batch optimizer setup); a new shape, integrator, dt or
  `vmap` width recompiles. Iterate with CPU tools and `golden check --quick`.
- **GPU box** (vast.ai; the `gpu-box` skill; the repo is private, never `git clone` it
  there): run `.venv/bin/python ...` (or `uv run --no-sync`), never plain `uv run`: it
  re-syncs and drops the CUDA jax. `pkill -f '[o]ptimize_control'`, not
  `pkill -f optimize_control` (that kills your own ssh session).
  **Always run jobs on the box detached** (`setsid nohup sh -c '... > run_X.log 2>&1; echo $? > run_X.done'`,
  the skill's launch recipe), never in a foreground ssh call: closing the local agent session or
  dropping the ssh connection must not end them. Poll with short ssh calls and fetch results as you go.
- **Reverse-mode autodiff**: `jax.jacfwd` works as-is; `jax.grad` cannot trace MJX's
  solver `while_loop` unless `model.opt.iterations == 1`.
  Geometry (radius / attach / offsets) is compiled host-side and is not differentiable.
- **Passive pitch is bistable** on the seed `designs/hummer/hover`: it settles into a
  ~190 mN or a ~210 mN lift mode depending on dt and integrator. Compare per-cycle lift
  or per mode, never one mean against another. At the viewer/GUI cadence (0.13 ms, aero
  substep 4) the seed's passive pitch is not periodic at all; 0.06 and 0.03 ms are fine.
- **Stale numbers**: the 2026-09 aero fixes cut the seed's lift from ~363 to ~187-210 mN
  and raised its certified peak from 174.7 to 193.1 mNm; `legacy-baseline` now has
  positive lift (~110 mN). Any older lift/peak (`docs/history/`, pre-fix rows of
  `designs/README.md`, "negative lift" claims) is history: re-run `design_io` first.

## Known bugs (left unfixed on purpose: each fix changes outputs)

Keep this list and `opt/README.md` "Known issues" in sync.

- Codesign (details in `opt/README.md` "Known issues"): the `--lift-floor` default
  (300 mN, also `run_gui`'s `lift_floor`) is unreachable now that the seed lifts 187-210 mN,
  so it binds from the first step; MJX sizes its warmup/sample windows by the design
  period, not the crank-locked one (1.5% mean-power gap at a0_log = 0.05); `--wavelet`
  writes a `control.tg` no runtime tool loads; the validity-penalized Adam stage was
  removed (`C_valid` is logged, not in the loss).
- GUI plot cache (`src/gui/sims.py` `cached_sim`): the `src/` mtime is read per call but
  the code is imported once, so editing physics with the GUI open and pressing Cmd-U
  stores old-code results under the new key. Restart the GUI after physics edits, or
  `rm -r .gui_plot_cache`.
- Morphology: the R fixed strut keeps the L site quats (cosmetic). The motor-tendon
  ranges are swept on the compile probe's post-sag pose, so they are L/R asymmetric
  (seed `L_motor_t1` [34.88, 80.01] vs `R_motor_t1` [34.33, 78.66] mm) and `R_motor_t4`'s
  lower limit (52.99 mm) sits above its qpos0 length (52.30 mm): every rest settle starts
  with that limit active (5 tendon-limit rows at qpos0).
- GUI: `run_gui`'s `RunConfig.loss` is never read; the design inspector leaks one
  `mkdtemp` scene dir per tab.
- Free flight: the free-flight renders (`render_configs`, `render_flight_phases`,
  `render_stroke_plane`) drive the plain control, not the hover policy, so the bird falls.
