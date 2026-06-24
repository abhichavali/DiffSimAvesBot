# Copilot instructions for this repository

## Build and run

- Install/sync: `uv sync`
- Compile morphology: `uv run python src/compile_morphology.py templates/generated-joint/scene.xml`
- Run viewer (macOS): `mjpython src/run_hanging_poc.py`
- Tests: `uv run pytest tests/test_morphology_compiler.py tests/test_tg_parser.py`

## Architecture

- **Design source**: `shoulder/shoulder.tg` (DSL describing free parameters and topology)
- **Compiler**: `src/morphology/` parses `.tg`, derives geometry, emits MuJoCo XML
- **Runtime**: loads compiled `templates/generated-joint/scene.xml`; aero/wing params from same `.tg` via `src/runtime/morphology_runtime.py`

## Key env vars

- `MUJOCO_XML_PATH` — override compiled scene path
- `MORPHOLOGY_PATH` — override `.tg` design file for runtime and compile (`--morphology`)
