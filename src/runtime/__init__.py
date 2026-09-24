"""The simulator's runtime: what happens to a compiled scene each physics step.

CPU (numpy + MuJoCo), the reference the JAX port in ``mjx/`` is validated against:
``sim`` (the plate-wing rollout step and rest state), ``aero`` (blade-element
forces), ``wing_pitch`` (rest calibration + pitch limiter), ``crank`` (crank sync,
phase-holding drive), ``control_signal`` (the Fourier drive), ``disk_orientation``
(stroke-plane steering), ``eagle_drive`` / ``wing_cables`` (eagle wing),
``realtime`` (viewer pacing), ``morphology_runtime`` (design paths).
"""
