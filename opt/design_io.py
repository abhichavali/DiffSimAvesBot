#!/usr/bin/env python3
"""Canonical reconstruction from optimizer artifacts to a runnable MuJoCo design.

The optimizer (opt/optimize_control.py) saves a design as a *baseline* shoulder.tg
plus JSON overrides (theta_best / tendon_optimized / motor_tendon_optimized /
pitch_optimized). The .tg format only carries a GLOBAL tendon default, so the
per-tendon tuning CANNOT be baked back into a .tg -- it must be applied to the
compiled model. This module is the single place that knows the
optimizer<->artifact conventions, so every consumer (CPU validation, replay,
viewer) reconstructs the *same* design instead of re-deriving them:

  * tendon stiffness   : json 'stiffness' is absolute -> model.tendon_stiffness
  * tendon rest length : base_lengthspring * pretension_factor (BOTH range cols)
  * motor tendons      : one shared stiffness + pretension over the 4 motor ids
  * control            : theta_best a/b are ABSOLUTE (x a0); runners want DEPTHS (/ a0)
  * warmup             : 8 settling periods (the rig builds lift for ~8 cycles)
  * wing geometry      : MORPHOLOGY_PATH + absolute meshdir

Mirrors optimize_control.py's dump()/perturbed_model() -- keep the two in sync.

CLI: replay an optimizer checkpoint on MuJoCo CPU with its tuned tendons applied,
via the SAME cpu_validate_design the optimizer used (printed lift / peak torque
match the run's cpu_validation.json):

    uv run python opt/design_io.py <design_dir> [--integrator euler implicitfast] [--timestep 0.06]
    uv run python opt/design_io.py --selfcheck   # asserts it000 + cpu_best reproduce known peaks
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compile_tg import compile_tg  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from runtime.morphology_runtime import load_design_morphology  # noqa: E402


@dataclass
class DesignSpec:
    """Everything needed to instantiate the exact optimized design on MuJoCo CPU."""
    scene_path: Path
    ctrl_params: tuple                 # (a0, a_depth, b_depth, omega) -- runner convention
    tendon_arrays: tuple | None        # (stiffness(ntendon,), lengthspring(ntendon,2)) or None
    pitch_params: tuple | None         # (up_pitch, down_pitch, up_stiff, down_stiff) or None
    phase_lag: float
    warmup: float = 8.0
    periods: float = 3.0


def _compiled_scene(design_dir: Path) -> Path:
    """Compile the design's shoulder.tg to a scene, repointing meshdir + MORPHOLOGY_PATH.

    A checkpoint's shoulder.tg has a relative `meshdir = assets` (resolved next to
    the .tg, absent in a checkpoint) and the CPU aero loads wing geometry from the
    canonical MORPHOLOGY_PATH. Both are pinned to this design here.
    """
    src = re.sub(r"(meshdir\s*=\s*)assets\b", rf"\g<1>{ROOT / 'assets'}",
                 (design_dir / "shoulder.tg").read_text())
    tg = design_dir / "_recon_shoulder.tg"
    tg.write_text(src)
    os.environ["MORPHOLOGY_PATH"] = str(tg.resolve())
    load_design_morphology.cache_clear()  # lru-cached; must refresh per design
    scene = design_dir / "_recon_scene.xml"
    compile_tg(tg, scene)
    return tg, scene


def _tendon_arrays(scene: Path, design_dir: Path):
    """Full (stiffness, lengthspring) arrays = baseline model + JSON overrides, or
    None when the design has no tendon tuning."""
    tj = design_dir / "tendon_optimized.json"
    if not tj.exists():
        return None
    model = mujoco.MjModel.from_xml_path(str(scene))
    stiff = np.array(model.tendon_stiffness, dtype=float)
    lenspring = np.array(model.tendon_lengthspring, dtype=float)  # (ntendon, 2)
    name2id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i): i
               for i in range(model.ntendon)}

    for name, p in json.loads(tj.read_text())["pairs"].items():
        i = name2id[name]                          # KeyError => json stale vs scene
        stiff[i] = p["stiffness"]                  # absolute N/m
        lenspring[i, :] *= p["pretension_factor"]  # scale BOTH range cols (== perturbed_model)

    mj = design_dir / "motor_tendon_optimized.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        for name, i in name2id.items():
            if name and "motor" in name:
                stiff[i] = m["stiffness"]
                lenspring[i, :] *= m["pretension_factor"]
    return stiff, lenspring


def _ctrl_params(design_dir: Path):
    """(a0, a_depth, b_depth, omega). theta_best stores a/b ABSOLUTE (x a0); the CPU
    runner expects DEPTHS -- divide a0 back out or the harmonics get re-multiplied
    by a0 (~63x) and the rollout diverges."""
    th = json.loads((design_dir / "theta_best.json").read_text())
    a0 = th["a0"]
    return (a0, np.asarray(th["a"]) / a0, np.asarray(th["b"]) / a0, th["omega"])


def _pitch_params(design_dir: Path):
    pj = design_dir / "pitch_optimized.json"
    if not pj.exists():
        return None
    p = json.loads(pj.read_text())
    return (p["upstroke_pitch_rad"], p["downstroke_pitch_rad"],
            p["upstroke_stiffness"], p["downstroke_stiffness"])


def reconstruct_design(design_dir, *, warmup=8.0, periods=3.0) -> DesignSpec:
    """Reconstruct a DesignSpec from an optimizer output/checkpoint dir.

    ``design_dir`` must contain shoulder.tg + theta_best.json; tendon/motor/pitch
    JSONs are applied when present (absent => baseline for that component).
    """
    design_dir = Path(design_dir)
    tg, scene = _compiled_scene(design_dir)
    return DesignSpec(
        scene_path=scene,
        ctrl_params=_ctrl_params(design_dir),
        tendon_arrays=_tendon_arrays(scene, design_dir),
        pitch_params=_pitch_params(design_dir),
        phase_lag=load_morphology(tg).motors.crank_phase_lag_rad,
        warmup=warmup, periods=periods,
    )


def validate(design_dir, integrators=("euler", "implicitfast"), *, timestep_ms=0.06,
             warmup=8.0, periods=3.0):
    """Reconstruct and roll out on MuJoCo CPU -> {integrator: (lift_N, peak_Nm, power_W)}."""
    from optimize_control import cpu_validate_design
    spec = reconstruct_design(design_dir, warmup=warmup, periods=periods)
    out = {}
    for integ in integrators:
        out[integ] = cpu_validate_design(
            spec.scene_path, spec.ctrl_params, spec.tendon_arrays, integ,
            pitch_params=spec.pitch_params, warmup=spec.warmup, periods=spec.periods,
            phase_lag=spec.phase_lag, timestep_ms=timestep_ms)
    return out


def _demo():
    """Self-check: reconstruction reproduces the run's cpu_validation.json to ~1 mNm."""
    d = ROOT / "best_designs" / "seed1_transfers"
    base = validate(d / "checkpoints" / "it000", ["euler"])
    best = validate(d / "cpu_best", ["euler", "implicitfast"])
    assert abs(base["euler"][1] * 1e3 - 185.3) < 3, base["euler"][1]
    assert abs(best["euler"][1] * 1e3 - 163.4) < 3, best["euler"][1]
    assert abs(best["implicitfast"][1] * 1e3 - 169.7) < 3, best["implicitfast"][1]
    print("design_io self-check OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir", type=Path, nargs="?")
    ap.add_argument("--integrator", nargs="+", default=["euler", "implicitfast"])
    ap.add_argument("--timestep", type=float, default=0.06, help="ms (match the run)")
    ap.add_argument("--warmup", type=float, default=8.0, help="settling periods before sampling")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _demo()
        return
    if args.design_dir is None:
        ap.error("design_dir required (or --selfcheck)")
    res = validate(args.design_dir, args.integrator, timestep_ms=args.timestep, warmup=args.warmup)
    for integ, (L, T, P) in res.items():
        print(f"[{args.design_dir.name}/{integ:12s}] lift={L*1e3:7.1f} mN  "
              f"peak_torque={T*1e3:7.1f} mNm  power={P*1e3:6.0f} mW")


if __name__ == "__main__":
    main()
