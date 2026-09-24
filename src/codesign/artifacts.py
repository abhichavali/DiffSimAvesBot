"""A co-design result on disk: the one writer and the one reader of a design dir.

A design dir (a run's --out, and each checkpoints/itNNN, stage_*, cpu_best, robust_best,
*preharmonics subdir) holds:

  shoulder.tg                  the source .tg with `control =` repointed at control.tg
  control.tg                   the optimized control (control_optimized.tg: same text)
  theta_full.json              {"theta": the raw packed theta} -- an --init-codesign seed
  theta_best.json              a0 and a/b ABSOLUTE (x a0), omega, frequency_hz, ...
  tendon_optimized.json        per structural tendon: absolute stiffness + factors, keyed
                               by FULL tendon name (only L_ names under --lock-wings)
  motor_tendon_optimized.json  the shared motor-tendon stiffness + pretension factor
  pitch_optimized.json         the 4 wing-pitch deadband/stiffness values
  strokeplane_optimized.json   the learned crank-disk mount signal
  log.json                     the per-iteration optimizer log

The .tg format only carries a GLOBAL tendon default, so per-tendon tuning cannot be baked
back into a .tg: reconstruct_design() applies the JSON to the compiled model instead.
It does not yet read strokeplane_optimized.json or wavelet theta_best files.
strokeplane_optimized.json is decoded by runtime.disk_orientation.StrokePlaneSignal
(GUI, cpu_check), inline in view_fixed_frequency, and in JAX by codesign.metrics.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from codesign.cpu_check import CpuDesign, CpuResult, cpu_rollout
from codesign.genome import Genome, wavelet_dictionary
from compile_tg import compile_tg
from morphology.tg_parser import load_morphology


def repoint_control(tg_text: str) -> str:
    """The .tg text with its `control =` line pointing at control.tg beside it."""
    return re.sub(r"(?m)^(\s*control\s*=\s*).*$", r"\1control.tg", tg_text)


def fourier_control_tg(bird, K, freq_hz, a0, a, b, motor_dir=1.0) -> str:
    """control.tg text for a0 + K harmonics (absolute, actuator frame / motor_dir)."""
    lines = [f"bird = {bird}", f"harmonics = {K}", f"fundamental_frequency = {freq_hz:g} Hz",
             f"a0 = {a0 / motor_dir:.6g}"]
    for j in range(K):
        lines.append(f"a_{j+1} = {a[j] / motor_dir:.6g}")
        lines.append(f"b_{j+1} = {b[j] / motor_dir:.6g}")
    return "\n".join(lines) + "\n"


def tendon_pairs_json(model, tendons, stiff_log, pre_log) -> dict:
    """tendon_optimized.json "pairs": full tendon name -> absolute stiffness and factors."""
    l_ids, base_stiff, base_lenspring = (np.asarray(tendons[0]), np.asarray(tendons[2]),
                                         np.asarray(tendons[3]))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(k)) for k in l_ids]
    return {
        names[k]: {
            "stiffness": float(base_stiff[l_ids[k]] * math.exp(stiff_log[k])),
            "stiffness_factor": float(math.exp(stiff_log[k])),
            "pretension_factor": float(math.exp(pre_log[k])),
            "rest_length": float(base_lenspring[l_ids[k]][0] * math.exp(pre_log[k])),
        }
        for k in range(len(l_ids))
    }


def motor_json(tendons, motor_ids, motor_log) -> dict:
    """motor_tendon_optimized.json: the shared motor-tendon stiffness and factors."""
    mids = np.asarray(motor_ids)
    return {"stiffness": float(np.asarray(tendons[2])[mids[0]] * math.exp(motor_log[0])),
            "stiffness_factor": float(math.exp(motor_log[0])),
            "pretension_factor": float(math.exp(motor_log[1]))}


def write_json(path: Path, obj, indent=2) -> None:
    path.write_text(json.dumps(obj, indent=indent))


def write_design(out_dir: Path, design: dict, log, *, genome: Genome, control_seed, rig, tg: Path,
                 morph, strokeplane=None) -> None:
    """Write a selected candidate ({"theta", "Lbar", "tau", ...}) as a design dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(design["theta"])
    write_json(out_dir / "log.json", log)
    (out_dir / "theta_full.json").write_text(json.dumps({"theta": [float(x) for x in theta]}))
    (out_dir / "shoulder.tg").write_text(repoint_control(tg.read_text()))
    g = genome.split(theta, np.asarray(control_seed))
    K, ctrl = genome.K, g.control
    base_freq, motor_dir = rig.control.fundamental_frequency_hz, morph.motors.motor_direction
    a0 = float(rig.drive.a0) * math.exp(ctrl[0])  # a0 log-factor -> absolute (actuator frame)
    a_out, b_out = a0 * ctrl[1 : 1 + K], a0 * ctrl[1 + K :]  # depth -> absolute a_n
    freq_hz = base_freq * math.exp(float(g.f_log)) if genome.frequency else base_freq
    stats = {"motor_direction": motor_dir, "Lbar": design["Lbar"], "tau_hat": design["tau"]}
    if (genome.control or genome.frequency) and genome.wavelet:
        # Wavelet coeffs are dimensionless atom depths; theta_full.json is the warm start,
        # theta_best.json carries the dictionary so a certifier can rebuild the waveform.
        coeffs = [float(x) for x in ctrl[1:]]
        centers, sigmas, _ = wavelet_dictionary(2 * K)
        lines = [f"bird = {morph.bird}", "basis = wavelet", f"n_atoms = {2 * K}",
                 f"fundamental_frequency = {freq_hz:g} Hz", f"a0 = {a0 / motor_dir:.6g}"]
        lines += [f"w_{j} = {coeffs[j]:.6g}" for j in range(2 * K)]
        control_text = "\n".join(lines) + "\n"
        best = {"a0": float(a0), "basis": "wavelet", "n_atoms": 2 * K, "coeffs": coeffs,
                "centers": [float(x) for x in centers], "sigmas": [float(x) for x in sigmas]}
    elif genome.control or genome.frequency:
        control_text = fourier_control_tg(morph.bird, K, freq_hz, a0, a_out, b_out, motor_dir)
        best = {"a0": float(a0), "a": [float(x) for x in a_out], "b": [float(x) for x in b_out]}
    else:
        control_text = None
        (out_dir / "control.tg").write_text(_control_source(tg, morph).read_text())
    if control_text is not None:
        (out_dir / "control.tg").write_text(control_text)
        (out_dir / "control_optimized.tg").write_text(control_text)
        write_json(out_dir / "theta_best.json", {**best, "frequency_hz": float(freq_hz),
                                                 "omega": float(math.tau * freq_hz), **stats})
    if genome.tendons:
        write_json(out_dir / "tendon_optimized.json", {
            "tau_hat": design["tau"], "Lbar": design["Lbar"],
            "pairs": tendon_pairs_json(rig.model, rig.tendons, g.stiff, g.pre)})
    if genome.pitch:
        p = rig.pitch
        write_json(out_dir / "pitch_optimized.json", {
            "upstroke_pitch_rad": float(p.upstroke_pitch) * math.exp(g.pitch[0]),
            "downstroke_pitch_rad": float(p.downstroke_pitch) * math.exp(g.pitch[1]),
            "upstroke_stiffness": float(p.upstroke_stiffness) * math.exp(g.pitch[2]),
            "downstroke_stiffness": float(p.downstroke_stiffness) * math.exp(g.pitch[3]),
        })
    if genome.motor:
        write_json(out_dir / "motor_tendon_optimized.json", motor_json(rig.tendons, rig.motor_ids, g.motor))
    if genome.strokeplane:
        write_json(out_dir / "strokeplane_optimized.json", {
            "phase_ref_joint": "L_t1_disk_hinge",
            "tilt0_rad": strokeplane["tilt0"], "pitch0_rad": strokeplane["pitch0"],
            "max_delta_rad": strokeplane["max_rad"],
            "tilt_coeffs": [float(x) for x in g.sp[:5]],
            "pitch_coeffs": [float(x) for x in g.sp[5:]],
            "model": "angle = angle0 + max*tanh((c0 + c1*cos(th) + c2*sin(th) "
                     "+ c3*cos(2th) + c4*sin(2th))/max), th = L_t1 crank angle",
        })


def _control_source(tg: Path, morph) -> Path:
    path = Path(morph.motors.control_file)
    return path if path.is_absolute() else tg.parent / path


class DesignSpec(NamedTuple):
    """A design dir rebuilt for CPU replay."""
    scene_path: Path
    design: CpuDesign
    morph: object  # the design's Morphology


def reconstruct_design(design_dir) -> DesignSpec:
    """Rebuild a design dir (shoulder.tg + theta_best.json, plus tendon/motor/pitch JSON
    when present) as the CpuDesign the optimizer validated.

    theta_best stores a/b ABSOLUTE (x a0); the CPU runner wants DEPTHS, so a0 is divided
    back out. Writes _recon_scene.xml into ``design_dir``.
    """
    design_dir = Path(design_dir)
    tg, scene = design_dir / "shoulder.tg", design_dir / "_recon_scene.xml"
    compile_tg(tg, scene)
    th = json.loads((design_dir / "theta_best.json").read_text())
    a0 = th["a0"]
    ctrl = (a0, np.asarray(th["a"]) / a0, np.asarray(th["b"]) / a0, th["omega"])
    return DesignSpec(scene, CpuDesign(ctrl, _tendon_arrays(scene, design_dir),
                                       _pitch_params(design_dir)),
                      load_morphology(tg))


def validate(design_dir, integrators=("euler", "implicitfast"), *, timestep_ms=0.06,
             warmup=8.0, periods=3.0) -> dict[str, CpuResult]:
    """Reconstruct and roll out on MuJoCo CPU; warmup=8 periods is load-bearing (the rig
    builds lift for ~8 cycles)."""
    spec = reconstruct_design(design_dir)
    return {integ: cpu_rollout(spec.scene_path, spec.design, integ, morph=spec.morph, warmup=warmup,
                               periods=periods, phase_lag=spec.morph.motors.crank_phase_lag_rad,
                               timestep_ms=timestep_ms)
            for integ in integrators}


def _tendon_arrays(scene: Path, design_dir: Path):
    """Full (stiffness, lengthspring) arrays = compiled model + JSON overrides, or None
    when the design has no tendon tuning."""
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
        lenspring[i, :] *= p["pretension_factor"]  # scale BOTH range cols, like the MJX rollout
    mj = design_dir / "motor_tendon_optimized.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        for name, i in name2id.items():
            if name and "motor" in name:
                stiff[i] = m["stiffness"]
                lenspring[i, :] *= m["pretension_factor"]
    return stiff, lenspring


def _pitch_params(design_dir: Path):
    pj = design_dir / "pitch_optimized.json"
    if not pj.exists():
        return None
    p = json.loads(pj.read_text())
    return (p["upstroke_pitch_rad"], p["downstroke_pitch_rad"],
            p["upstroke_stiffness"], p["downstroke_stiffness"])
