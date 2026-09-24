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

The viewers (`src/view_fixed_frequency.py`, `src/view_hover.py`, `src/view_eagle.py`)
need `mjpython` on macOS; see `AGENTS.md` for the exact command. Pass `--tg`: the
CPU runner and the compiler default to `designs/hummer/legacy-baseline`, which is not a
hover design (negative lift).

`designs/eagle/` is the AvesEagle spinoff: the same shoulder carrying a three-linkage
bone wing (humerus / ulna / manus). `uv run python src/view_eagle.py --gif out/eagle.gif`
renders it headless; stroke shaping is on the command line (`--fold`, `--twist`,
`--lead`, `--level`, `--attack`), design knobs in its `.tg` files.

Every `.tg` starts with `bird = hummer` (the plate wing) or `bird = eagle` (the bone
wing); tools refuse a design of the wrong bird and a control of another bird.

## Where to read next

- `AGENTS.md` — layout, commands, conventions, the golden behavior check.
- `designs/README.md` — every committed design, its lineage and numbers.
- `opt/README.md` — co-design: genome, reward, robustness gates, outputs, known issues.
- `docs/ROBUST_CODESIGN_METHODOLOGY.md` — how designs are certified (CPU peak torque);
  `docs/flight_demos.org` — the free-flight hover demos.
