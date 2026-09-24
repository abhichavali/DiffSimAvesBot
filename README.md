# DiffSimAvesBot

A tensegrity flapping-wing robot in simulation. A `.tg` design (morphology + Fourier
crank-velocity control) compiles to MuJoCo XML and runs on a CPU reference simulator
(numpy blade-element aero), on a differentiable JAX/MJX port of it, and under
gradient and evolutionary co-design that tunes the control and the tendon net.

Pipeline: `designs/<bird>/<name>/{shoulder,control}.tg` -> `src/morphology` (parser +
compiler) -> MJCF -> `src/runtime` (CPU step, `mjx/` port) -> `src/codesign` + `opt/`.

## Run

```bash
uv sync
uv run python src/compile_tg.py --tg designs/hummer/hover/shoulder.tg --out scene.xml
uv run python src/run_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5
uv run python src/run_mjx_fixed_frequency.py --tg designs/hummer/hover/shoulder.tg --periods 5
uv run python plot_fixed_frequency_gui.py        # design inspector (Tk)
uv run pytest -q                                  # tests + golden quick tier
```

The viewers (`src/view_fixed_frequency.py`, `src/view_eagle.py`, `src/view_free_hover.py`)
need `mjpython` on macOS; see `AGENTS.md` for the exact command. Pass `--tg` to the
compiler: it defaults to `designs/hummer/legacy-baseline`, the June demo baseline, which
is not a hover design.

`designs/eagle/` is the AvesEagle spinoff: the same shoulder carrying a three-linkage
bone wing (humerus / ulna / manus). `uv run python src/view_eagle.py --gif out/eagle.gif`
renders it headless; stroke shaping is on the command line (`--fold`, `--twist`,
`--lead`, `--level`, `--attack`), design knobs in its `.tg` files.

`designs/hummer/free-hover/` is the free-standing bird (226 mN) hovering under its own
flight policy: a phase-locked Fourier open loop plus a per-wingbeat closed-loop MLP, both
in its `control.tg` (code in `src/hover/`). `uv run python opt/verify_hover.py` flies it
on CPU with pushes; `mjpython src/view_free_hover.py` (or the GUI's Viewer tab) shows it.

Every `.tg` starts with `bird = hummer` (the plate wing) or `bird = eagle` (the bone
wing); tools refuse a design of the wrong bird and a control of another bird.

## Where to read next

- `AGENTS.md` — layout, commands, conventions, gotchas, known bugs, the golden behavior
  check, and the index of workflow skills (`.claude/skills/`: certification, co-design
  runs, free-flight policy, GPU boxes, designs, the golden harness).
- `designs/README.md` — every committed design, its lineage and numbers.
- `opt/README.md` — co-design (genome, reward, robustness gates, outputs, known issues)
  and the free-flight CLIs.
- `docs/flight_demos.org` — the free-flight airframe, what was learned, the renders.
- `docs/history/` — frozen campaign logs (numbers predate the 2026-09 aero fixes).
