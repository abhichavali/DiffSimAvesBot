# Tensegrity Barebones

This is the small version of `Tensegrity-Playground`: a `.tg` morphology file,
the compiler needed to emit MuJoCo XML, and runners driven by the design control
signal.

## Run

```bash
uv sync
uv run python src/compile_tg.py
uv run python src/run_fixed_frequency.py --periods 5
uv run python src/run_mjx_fixed_frequency.py --periods 5
mjpython src/view_fixed_frequency.py
```

`design/eagle.tg` is the AvesEagle spinoff: the same tensegrity shoulder
carrying a three-linkage avian wing (humerus / ulna / manus) instead of the flat
plate, stroking side to side the way the plate wing does.

```bash
mjpython src/view_eagle.py
uv run python src/view_eagle.py --gif out/eagle.gif
uv run pytest -q tests/test_eagle_wing.py
```

The default design is `design/shoulder.tg`; it points at `design/control.tg` for
the Fourier control signal. The default generated scene is `scene.xml`.
