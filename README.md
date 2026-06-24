# Tensegrity Barebones

This is the small version of `Tensegrity-Playground`: a `.tg` morphology file,
the compiler needed to emit MuJoCo XML, and a fixed-frequency runner.

## Run

```bash
uv sync
uv run python compile_tg.py
uv run python run_fixed_frequency.py --frequency 10 --periods 5
uv run python run_mjx_fixed_frequency.py --frequency 10 --periods 5
```

The default design is `design/shoulder.tg`. The default generated scene is
`scene.xml`.
