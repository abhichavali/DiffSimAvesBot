# Tensegrity Barebones

This is the small version of `Tensegrity-Playground`: a `.tg` morphology file,
the compiler needed to emit MuJoCo XML, and runners driven by the design control
signal.

## Run

```bash
uv sync
uv run python compile_tg.py
uv run python run_fixed_frequency.py --periods 5
uv run python run_mjx_fixed_frequency.py --periods 5
mjpython view_fixed_frequency.py
```

The default design is `design/shoulder.tg`; it points at `design/control.tg` for
the Fourier control signal. The default generated scene is `scene.xml`.
