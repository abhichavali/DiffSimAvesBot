"""The morphology compiler: a .tg design -> a MuJoCo MJCF scene.

tg_parser (.tg -> Morphology, model.py) -> compile (geometry, tendon_net ->
CompiledMorphology) -> emit_mjcf (+ emit_eagle) -> XML, with motor_tendons
measuring the crank tendons on a probe model in between. Start at
compile.compile_file.
"""
