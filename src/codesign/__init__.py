"""Gradient co-design of the hummer rig on the differentiable MJX simulation.

opt/optimize_control.py (CLI) -> train.run: rig (compile + settle a .tg, MJX variants),
genome (the packed parameter vector), metrics (theta -> rollout Metrics), loss (robust
reward + objectives), cpu_check (MuJoCo CPU replay / transfer gate), artifacts (design
dirs on disk: write and read back). opt/evolve_codesign.py runs an EA over the same
genome and rollout.
"""
