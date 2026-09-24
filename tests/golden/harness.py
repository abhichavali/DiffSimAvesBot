"""Golden behavior harness: proves a refactor did not change what the code computes.

    uv run python -m tests.golden check [--quick] [-k PATTERN]    # diff vs baseline.json
    uv run python -m tests.golden capture [--quick] [-k PATTERN]  # (re)write those items

Every ITEM recomputes one piece of repo behavior (parse, compile, CPU rollouts, aero,
MJX, the optimizer CLI) and compares it with tests/golden/baseline.json: exactly for
CPU/numpy, to within JAX_RTOL for JAX/MJX. `--quick` skips the JIT-heavy full tier.
`capture` merges the recomputed items into the baseline -- only do that for a
deliberate, reviewed behavior change.

Refactor contract: when code or designs MOVE, update only the ADAPTER section below
(imports, thin wrappers, PATHS). Baseline keys are logical names, never paths.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest import mock

import numpy as np

# ================================== ADAPTER ==================================
# The ONLY place that names repo modules, functions and file paths.
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# logical design name -> current .tg path. Move a design: change its path here only.
PATHS = {
    "shoulder": "designs/hummer/hover/shoulder.tg",  # runtime / viewer / GUI default
    "baseline": "designs/hummer/legacy-baseline/shoulder.tg",  # compile_tg default
    "eagle": "designs/eagle/shoulder.tg",
    "soft_it009p": "designs/hummer/hover-soft-it009p/shoulder.tg",  # harmonics + stroke plane
    "tuned": "designs/hummer/hover/shoulder.tg",  # was a byte-identical copy of `shoulder`
    "hovering": "designs/hummer/hover-base/shoulder.tg",  # run_gui / EA default
    "it009p": "designs/hummer/hover/shoulder.tg",  # hover_policy default; was a byte-identical copy of `shoulder`
    "tuned_hacked": "designs/hummer/checkpoints/tuned-dt-hacked/shoulder.tg",  # diag_mjx_cpu_parity default
}
OPTIMIZER_CLI = ROOT / "opt" / "optimize_control.py"
STROKEPLANE_JSON = "strokeplane_optimized.json"  # learned stroke-plane artifact beside a design
PATH_FIELDS = {"meshdir", "mesh_file", "control_file", "mesh_path", "file"}  # compared by basename only

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import mujoco.mjx as mjx  # noqa: E402
import mujoco.viewer  # noqa: E402,F401  (patched headless below)

import run_fixed_frequency as _run_ff  # noqa: E402
import view_eagle as _view_eagle  # noqa: E402
import view_fixed_frequency as _view_ff  # noqa: E402
from codesign.cpu_check import CpuDesign, cpu_rollout  # noqa: E402
from codesign.genome import Genome, load_control_seed, ricker_atoms, wavelet_dictionary  # noqa: E402
from codesign.metrics import build_rollout  # noqa: E402
from codesign.rig import motor_tendon_ids, setup, structural_tendon_pairs, symmetrize_wings  # noqa: E402
from compile_tg import compile_tg  # noqa: E402
from gui.sims import phase_plot, phase_trace  # noqa: E402
from morphology.tg_parser import load_morphology  # noqa: E402
from morphology.wing_geometry import load_wing_geometry  # noqa: E402
from runtime.aero import DEFAULT_AERO_SUBSTEP, RHO_AIR, AeroBladeRuntime, set_unsteady_aero  # noqa: E402
from runtime.aero import aerodynamics as cpu_aero_strip  # noqa: E402
from runtime.control_signal import eval_control, load_control_for_morphology, ramp_factor  # noqa: E402
from runtime.crank import CRANK_ACTUATORS, sync_crank_disks_at_rest  # noqa: E402
from runtime.mjx import aero as _mjx_aero, aero_runtime as _mjx_aero_rt  # noqa: E402
from runtime.mjx import rollout as _mjx_rollout, wing_pitch as _mjx_pitch  # noqa: E402
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest  # noqa: E402


def mjx_aero_wing(*args, **kwargs):
    return _mjx_aero.aero_wing(*args, **kwargs)[:2]


_design = None  # the design use_design() last named; the runtime wrappers below are handed it


def tg(name: str) -> Path:
    return ROOT / PATHS[name]


def use_design(name: str) -> None:
    """Name the design the wrappers below (rest_state, CpuAero, mjx_*, ...) run."""
    global _design
    _design = name


def parse(name: str):
    return load_morphology(tg(name)), load_control_for_morphology(tg(name))


def design_morph():
    return load_morphology(tg(_design))


def wing_geometry():
    return load_wing_geometry(design_morph())


def CpuAero(model, **kw):
    return AeroBladeRuntime(model, design_morph(), **kw)


def CpuPitch(model, data):
    return WingPitchLimiter(model, data, design_morph())


def mjx_drive_config(model):
    return _mjx_rollout.build_drive_config(model, *parse(_design))


def mjx_simulate(model, data, n_steps: int):
    return _mjx_rollout.simulate(model, data, *parse(_design), n_steps)


def compile_model(name: str, scene: Path) -> mujoco.MjModel:
    use_design(name)
    compile_tg(tg(name), scene)
    return mujoco.MjModel.from_xml_path(str(scene))


def rest_state(model, data, lag=None) -> None:
    """Settle + neutral wing pitch + crank sync: the start of every CPU drive path."""
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data, design_morph(), lag)


def runtime_config() -> dict:
    """The design as the MJX configs once read it (the deleted runtime/runtime_config.py)."""
    morph, control = parse(_design)
    p, direction = morph.wing.pitch, morph.motors.motor_direction
    runtime = {
        "aero": {"num_blades": morph.wing.aero.blades, "substep": DEFAULT_AERO_SUBSTEP, "mirror_wings": False},
        "wing_pitch_parameters": {
            "upstroke_pitch": p.upstroke_pitch_rad, "downstroke_pitch": p.downstroke_pitch_rad,
            "upstroke_stiffness": p.upstroke_stiffness, "downstroke_stiffness": p.downstroke_stiffness,
            "free_angle": p.upstroke_pitch_rad, "limit_stiffness": p.upstroke_stiffness,
            "limit_damping": p.damping, "initial_pitch_angle": p.initial_rad},
        "assets": {"meshes": [{"file": morph.wing.mesh_file}]},
        "compiler": {"meshdir": morph.world.meshdir},
        "motors": {"crank_phase_lag_rad": morph.motors.crank_phase_lag_rad,
                   "tendon_rest_length": morph.motors.tendon_rest_length},
    }
    drive = {
        "control_frequency_rad_s": control.omega, "control_a0": direction * control.a0,
        "control_a": tuple(direction * v for v in control.a), "control_b": tuple(direction * v for v in control.b),
        "ramp_duration": 2.0, "flap_actuator_names": list(CRANK_ACTUATORS)}
    return {"runtime": runtime, "control": drive}


def cpu_validate(scene, ctrl, tendons, integrator, *, pitch_params=None, strokeplane_params=None,
                 cycle_stats=False, with_swing=False, **kw):
    """codesign.cpu_check.cpu_rollout as (lift, peak, power[, cycle_peaks][, swing])."""
    r = cpu_rollout(scene, CpuDesign(ctrl, tendons, pitch_params, strokeplane_params), integrator,
                    morph=design_morph(), **kw)
    return (r.lift, r.peak, r.power) + ((r.cycle_peaks,) if cycle_stats else ()) + (
        (r.swing,) if with_swing else ())


def ctrl_params(name: str):
    """(a0, a, b, omega) as the optimizer hands them to cpu_rollout, + crank lag."""
    morph, c = parse(name)
    a0 = morph.motors.motor_direction * c.a0
    return (a0, [x / c.a0 for x in c.a], [x / c.a0 for x in c.b], c.omega), morph.motors.crank_phase_lag_rad


def pitch_params(name: str, scale=(1.0, 1.0, 1.0, 1.0)):
    p = parse(name)[0].wing.pitch
    base = (p.upstroke_pitch_rad, p.downstroke_pitch_rad, p.upstroke_stiffness, p.downstroke_stiffness)
    return tuple(b * s for b, s in zip(base, scale))


def strokeplane_params(name: str):
    d = json.loads((tg(name).parent / STROKEPLANE_JSON).read_text())
    return d["tilt_coeffs"] + d["pitch_coeffs"], d["tilt0_rad"], d["pitch0_rad"], d["max_delta_rad"]


def run_fixed(name: str, scene: Path, periods: float, warmup: float) -> None:
    _run_ff.run(tg(name), scene, periods, warmup)


def view_fixed(name: str, scene: Path, frames: int) -> None:
    """view_fixed_frequency.main() headless: a fake passive viewer that runs `frames` frames."""

    class Viewer:
        left = frames

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def is_running(self):
            self.left -= 1
            return self.left >= 0

        def sync(self):
            pass

    argv = ["view_fixed_frequency", "--tg", str(tg(name)), "--scene", str(scene), "--speed", "0.2"]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(mujoco.viewer, "launch_passive", lambda m, d: Viewer()):
        _view_ff.main()


def gui_trace(name: str, scene: Path, plot_kind: str, periods: float, settle: float):
    return phase_plot(phase_trace(tg(name), scene, periods, settle_periods=settle), plot_kind)


def eagle_run(name: str, scene: Path, steps: int):
    """view_eagle's headless drive (build + step) with its CLI defaults."""
    args = argparse.Namespace(
        tg=tg(name), scene=scene, fold=0.35, twist=30.0, lead=35.0, level=1.0, attack=None,
        cables="slack", cable_only=False, elbow=None, elbow_sweep=None, glide=False,
        glide_park=20.0, paused=False)
    model, data, cranks, _control, drive, held = _view_eagle.build(args)
    for _ in range(steps):
        _view_eagle.step(model, data, cranks, drive, held)
        mujoco.mj_step(model, data)
    return model, data, drive


def mjx_aero_wrench(model, datas):
    """aero_wrench jitted as the rollout runs it; the aero state (added-mass FD) carries across calls."""
    cfg = _mjx_aero_rt.build_aero_config(model, design_morph(), substep=1)
    f = jax.jit(lambda m, d, s: _mjx_aero_rt.aero_wrench(m, d, cfg, s))
    mx, s, out = mjx.put_model(model), _mjx_aero_rt.init_aero_state(cfg), []
    for data in datas:
        qfrc, net, s, _, _ = f(mx, mjx.put_data(model, data), s)
        out.append((qfrc, net))
    return out


def mjx_pitch_qfrc(model, qpos, qvel, centers):
    data = mujoco.MjData(model)  # the dead-zone centers, as the rest pose build_wing_pitch_config reads
    for side, center in zip("LR", centers):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wing_pitch")]] = center
    cfg = _mjx_pitch.build_wing_pitch_config(model, design_morph(), data)
    return _mjx_pitch.wing_pitch_qfrc(jnp.asarray(qpos), jnp.asarray(qvel), cfg)


def mjx_metrics(name: str, scene: Path, K: int, warmup_steps: int, sample_steps: int, avg_periods: int):
    """build_rollout wired the way the optimizer wires it (control + frequency)."""
    rig = setup(tg(name), scene)
    model, control, drive = rig.model, rig.control, rig.drive
    base_a0 = float(drive.a0)
    seed = jnp.asarray(load_control_seed(None, K, drive, base_a0), dtype=jnp.float32)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in np.asarray(drive.flap_actuator_ids)]
    t1_mask = jnp.asarray([1.0 if "t1" in n else 0.0 for n in names], dtype=jnp.float32)
    metrics = build_rollout(
        rig, Genome(K, int(rig.tendons[0].shape[0]), control=True, frequency=True), float(control.omega),
        warmup_steps=warmup_steps, sample_steps=sample_steps,
        steps_per_cycle=max(1, round(control.period / model.opt.timestep)), aero_substep=4,
        control_fixed=seed, flap_t1_mask=t1_mask, avg_periods=avg_periods)
    return metrics, jnp.concatenate([seed, jnp.zeros(1, jnp.float32)])


def run_optimizer(argv: list[str]) -> str:
    r = subprocess.run([sys.executable, str(OPTIMIZER_CLI), *argv], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"optimizer exited {r.returncode}:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    return r.stdout
# ================================ END ADAPTER ================================

BASELINE = Path(__file__).with_name("baseline.json")
JAX_RTOL = 1e-6


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def vec(a, exact: bool = True):
    """Short arrays verbatim; long ones as summary stats (+ a hash when exact)."""
    a = np.asarray(a, dtype=float).ravel()
    if a.size <= 64:
        return a.tolist()
    d = {"n": a.size, "sum": float(a.sum()), "min": float(a.min()), "max": float(a.max()),
         "head": a[:3].tolist(), "tail": a[-3:].tolist()}
    return {**d, "sha": sha(a)} if exact else d


def canon(x, key: str = "", exact: bool = True):
    """JSON-able canonical dump of dataclass / NamedTuple / array trees (no type names)."""
    if key in PATH_FIELDS and isinstance(x, (str, Path)):
        return Path(x).name
    if isinstance(x, (mujoco.MjModel, mujoco.MjData)):
        return "<mujoco>"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: canon(getattr(x, f.name), f.name, exact) for f in dataclasses.fields(x)}
    if hasattr(x, "_asdict"):
        return {k: canon(v, k, exact) for k, v in x._asdict().items()}
    if isinstance(x, enum.Enum):
        return canon(x.value, key, exact)
    if isinstance(x, dict):
        return {str(k): canon(v, str(k), exact) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canon(v, "", exact) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    if x is None or isinstance(x, str):
        return x
    if hasattr(x, "shape"):  # numpy / jax array
        return vec(x, exact)
    raise TypeError(f"canon: no rule for {type(x).__name__} at {key!r}")


def fingerprint(m: mujoco.MjModel) -> dict:
    """Every non-empty numpy array of MjModel (shape + vec) + sizes, names, opt, stat.
    Asset paths are skipped so moving meshes changes nothing; any real change shows."""
    out = {}
    for k in sorted(dir(m)):
        if k.startswith("_") or "path" in k or k == "signature":
            continue
        v = getattr(m, k)
        if isinstance(v, np.ndarray) and v.size:
            out[k] = [list(v.shape), vec(v)]
        elif isinstance(v, int) and not isinstance(v, bool):
            out[k] = v
    out["names"] = m.names.decode().split("\x00")
    for sub in ("opt", "stat"):
        o = getattr(m, sub)
        for k in sorted(dir(o)):
            v = getattr(o, k)
            if not k.startswith("_") and isinstance(v, (int, float, np.ndarray)):
                out[f"{sub}.{k}"] = np.asarray(v).tolist()
    return out


def state(d) -> dict:
    return {"time": float(d.time), "qpos": vec(d.qpos), "qvel": vec(d.qvel)}


@contextlib.contextmanager
def last_step():
    """Record the MjData of the most recent mujoco.mj_step (to reach a rollout's end state)."""
    rec = argparse.Namespace(data=None)
    orig = mujoco.mj_step

    def step(m, d, *a, **k):
        orig(m, d, *a, **k)
        rec.data = d

    with mock.patch.object(mujoco, "mj_step", step):
        yield rec


_ABS_PATH = re.compile(r"""(?:/[^\s/'"]+)+/([^\s/'"]+)""")


def scrub(text: str, **placeholders) -> list[str]:
    """Printed text -> lines; `placeholders` paths -> <name>, any other absolute path -> its basename."""
    for k, p in placeholders.items():
        text = text.replace(str(p), f"<{k}>")
    return _ABS_PATH.sub(r"\1", text).splitlines()


def printed(fn, *args) -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()) as s:
        fn(*args)
    return scrub(s.getvalue())


# ================================== ITEMS ==================================
ITEMS: dict = {}  # name -> (fn(tmp) -> JSON-able, full_tier_only, rtol)


def item(name: str, fn, *, full: bool = False, rtol: float = 0.0) -> None:
    ITEMS[name] = (fn, full, rtol)


def parse_item(n, tmp):
    morph, control = parse(n)
    return {"morphology": canon(morph), "control": canon(control)}


for _n in PATHS:
    item(f"parse/{_n}", lambda tmp, n=_n: parse_item(n, tmp))
    item(f"model/{_n}", lambda tmp, n=_n: fingerprint(compile_model(n, tmp / "scene.xml")))


def helpers_item(tmp):
    ts = np.linspace(0.0, 0.2, 9)
    centers, sigmas, freqs = wavelet_dictionary(6)
    out = {
        "eval_control": {n: [eval_control(parse(n)[1], float(t)) for t in ts] for n in PATHS},
        "ramp": [ramp_factor(float(t), 0.01, 0.05) for t in ts] + [ramp_factor(0.1, 0.0, 0.0)],
        "wavelet": {"centers": vec(centers), "sigmas": vec(sigmas), "freqs": vec(freqs),
                    "ricker": [vec(ricker_atoms(p, centers, sigmas, np)) for p in np.linspace(0, 6.5, 7)]},
    }
    model = compile_model("soft_it009p", tmp / "scene.xml")
    out["runtime_config"] = canon(runtime_config())
    drive = mjx_drive_config(model)
    out["drive_config"] = canon(drive)
    out["control_seed"] = canon(load_control_seed(None, 3, drive, float(drive.a0)))
    out["tendon_pairs"] = canon(structural_tendon_pairs(model, individual=True))
    out["tendon_pairs_mirrored"] = canon(structural_tendon_pairs(model, individual=False))
    out["motor_tendons"] = canon(motor_tendon_ids(model))
    out["symmetrize_wings"] = [symmetrize_wings(model), vec(model.tendon_stiffness), vec(model.tendon_lengthspring)]
    return out


item("helpers", helpers_item)


def rest_item(n, tmp):
    model = compile_model(n, tmp / "scene.xml")
    data = mujoco.MjData(model)
    rest_state(model, data)
    return state(data)


for _n in ("shoulder", "soft_it009p"):
    item(f"rest/{_n}", lambda tmp, n=_n: rest_item(n, tmp))


def cpu_item(n, integrator, tmp, *, tendon_scale=None, ctrl=None, **kw):
    """codesign.cpu_check.cpu_rollout, the certification path (0.25 + 0.5 periods)."""
    scene = tmp / "scene.xml"
    model = compile_model(n, scene)
    cp, lag = ctrl_params(n)
    tendons = None if tendon_scale is None else (model.tendon_stiffness * tendon_scale[0],
                                                  model.tendon_lengthspring * tendon_scale[1])
    kw = {"phase_lag": lag, "aero_substep": 4, **kw}
    with last_step() as last:
        L, T, P, pc, swing = cpu_validate(scene, ctrl or cp, tendons, integrator, periods=0.5,
                                          warmup=0.25, cycle_stats=True, with_swing=True, **kw)
    return {"lift": L, "peak_torque": T, "power": P, "cycle_peaks": pc, "swing": swing, "final": state(last.data)}


for _n in ("baseline", "soft_it009p"):
    for _i in ("euler", "implicitfast"):
        item(f"cpu/{_n}/{_i}", lambda tmp, n=_n, i=_i: cpu_item(n, i, tmp))
item("cpu/soft_it009p/tendons_pitch_strokeplane_dt", lambda tmp: cpu_item(
    "soft_it009p", "euler", tmp, tendon_scale=(1.1, 1.002), timestep_ms=0.1,
    pitch_params=pitch_params("soft_it009p", (0.9, 1.1, 1.2, 0.8)),
    strokeplane_params=strokeplane_params("soft_it009p")))


def wavelet_item(tmp):
    (a0, _, _, omega), _ = ctrl_params("soft_it009p")
    wl = wavelet_dictionary(6)
    return cpu_item("soft_it009p", "euler", tmp, wavelet=wl,
                    ctrl=(a0, [0.02, -0.01, 0.015, 0.0, -0.02, 0.01], [], omega))


item("cpu/soft_it009p/wavelet", wavelet_item)


def run_fixed_item(tmp):
    with last_step() as last:
        out = printed(run_fixed, "baseline", tmp / "scene.xml", 0.2, 0.2)
    return {"stdout": out, "final": state(last.data)}


def viewer_item(tmp):
    with last_step() as last:
        out = printed(view_fixed, "soft_it009p", tmp / "scene.xml", 12)
    return {"stdout": [s for s in out if not s.startswith("viewer_speed actual")], "final": state(last.data)}


def gui_item(n, plot_kind, periods, tmp):
    with last_step() as last:
        x, series, *labels, colors = gui_trace(n, tmp / "scene.xml", plot_kind, periods, 0.3)
    return {"x": vec(x), "series": {k: vec(v) for k, v in series.items()}, "labels": labels,
            "colors": canon(colors), "final": state(last.data)}


def eagle_item(tmp):
    with contextlib.redirect_stdout(io.StringIO()) as s:
        _model, data, drive = eagle_run("eagle", tmp / "scene.xml", 1000)
    return {"stdout": scrub(s.getvalue()), "drive": canon(drive), "final": state(data)}


item("drive/run_fixed_frequency", run_fixed_item)
item("drive/view_fixed_frequency", viewer_item)
item("drive/gui_torque", lambda tmp: gui_item("soft_it009p", "Left actuator torque vs phase", 0.3, tmp))
item("drive/gui_lift", lambda tmp: gui_item("shoulder", "Mean lift per half-cycle", 1.0, tmp))
item("drive/eagle", eagle_item)

AERO_CASES = [  # (omega_w, vel_h, omega_w_dot, vel_h_prev) in the wing root-site frame
    ((12.0, -30.0, 5.0), (0.4, -1.1, 0.3), (200.0, -500.0, 80.0), (0.38, -1.05, 0.31)),
    ((-3.0, 45.0, -8.0), (-0.2, 0.9, -0.5), (-100.0, 300.0, 50.0), (-0.25, 0.85, -0.45)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]
AERO_DT = 4 * 0.13e-3  # the runtime's finite-difference window (substep 4)


def fixed_state(tmp):
    """shoulder at rest, then a deterministic nonzero qvel; plus a perturbed follow-up qvel."""
    model = compile_model("shoulder", tmp / "scene.xml")
    data = mujoco.MjData(model)
    rest_state(model, data)
    i = np.arange(model.nv)
    data.qvel[:] = 3.0 * np.sin(i + 1.0)
    mujoco.mj_forward(model, data)
    return model, data, 3.0 * np.sin(i + 1.0) + 0.5 * np.cos(i)


def aero_cpu_item(tmp):
    use_design("shoulder")
    g = wing_geometry()
    nb = len(g.chords)
    out = {"geometry": canon(g)}
    for side in "LR":
        for ci, (w, v, wd, vp) in enumerate(AERO_CASES):
            rows = [cpu_aero_strip(side, np.array(w), np.zeros(3), np.array(v), np.array(wd), (j + 0.5) / nb,
                                   AERO_DT, g.length, nb, RHO_AIR, float(c),
                                   vel_h_prev=np.array(vp), return_cop=True, x_rot=float(g.x_rot[j]),
                                   root=g.root_offset) for j, c in enumerate(g.chords)]
            out[f"strips/{side}{ci}"] = {k: [np.asarray(r[j]).tolist() for r in rows]
                                         for j, k in enumerate(("force", "torque", "cop"))}
    model, data, qvel2 = fixed_state(tmp)
    for unsteady in (False, True):
        set_unsteady_aero(unsteady)
        try:
            rt = CpuAero(model, substep=1)
            data.qvel[:] = 3.0 * np.sin(np.arange(model.nv) + 1.0)
            for step, qv in enumerate((data.qvel.copy(), qvel2)):
                data.qvel[:] = qv
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = 0.0
                rt.apply(model, data)
                out[f"runtime/unsteady={unsteady}/call{step}"] = {
                    "qfrc": vec(data.qfrc_applied),
                    **{f"{w.side}_{k}": vec(getattr(w, f"cached_{k}")) for w in rt.wings
                       for k in ("forces", "torques", "points")}}
        finally:
            set_unsteady_aero(False)
    pitch = CpuPitch(model, data)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wing_pitch") for s in "LR"]
    for jid, dev in zip(jids, (1.2, -1.1)):
        data.qpos[model.jnt_qposadr[jid]] += dev
    data.qfrc_applied[:] = 0.0
    pitch.apply(model, data)
    out["pitch_limiter_qfrc"] = vec(data.qfrc_applied)
    return out


def aero_mjx_item(tmp):
    use_design("shoulder")
    g = wing_geometry()
    out = {}
    for side in "LR":
        for ci, (w, v, wd, vp) in enumerate(AERO_CASES):
            f, t = mjx_aero_wing(*(jnp.asarray(x) for x in (w, (0.0, 0.0, 0.0), v, wd, vp)), side=side,
                                 chords=jnp.asarray(g.chords), x_rot=jnp.asarray(g.x_rot),
                                 length=g.length, root=g.root_offset, num_blade=len(g.chords), dt=AERO_DT)
            out[f"strips/{side}{ci}"] = {"force": np.asarray(f).tolist(), "torque": np.asarray(t).tolist()}
    model, data, qvel2 = fixed_state(tmp)
    centers = [float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wing_pitch")]])
               for s in "LR"]
    data2 = mujoco.MjData(model)  # same pose, perturbed qvel: the second (finite-difference) call
    data2.qpos[:], data2.qvel[:] = data.qpos, qvel2
    mujoco.mj_forward(model, data2)
    for unsteady in (False, True):
        set_unsteady_aero(unsteady)  # read at trace time; mjx_aero_wrench jits afresh
        try:
            for call, (qfrc, net) in enumerate(mjx_aero_wrench(model, (data, data2))):
                out[f"wrench/unsteady={unsteady}/call{call}"] = {
                    "qfrc": np.asarray(qfrc).tolist(), "net_force": np.asarray(net).tolist()}
        finally:
            set_unsteady_aero(False)
    qpos = data.qpos.copy()
    for s, dev in zip("LR", (1.2, -1.1)):
        qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wing_pitch")]] += dev
    out["pitch_qfrc"] = np.asarray(mjx_pitch_qfrc(model, qpos, data.qvel, centers)).tolist()
    return out


item("aero/cpu", aero_cpu_item)
item("aero/mjx", aero_mjx_item, rtol=JAX_RTOL)


def mjx_metrics_item(tmp):
    """Two whole cycles (744 steps each at 0.13 ms), metrics + d(metrics)/d(theta) in one jacfwd.
    avg_periods=1 puts the metric window on cycle 2, clear of the spin-up torque spike."""
    metrics, theta0 = mjx_metrics("shoulder", tmp / "scene.xml", 1, 744, 744, avg_periods=1)
    both = jax.jit(jax.jacfwd(lambda th: (lambda m: (m, m))(jnp.stack(metrics(th))), has_aux=True))
    jac, vals = both(theta0)
    return {"theta0": np.asarray(theta0).tolist(), "metrics": np.asarray(vals).tolist(),
            "jacobian": np.asarray(jac).tolist()}


def mjx_simulate_item(tmp):
    """rollout.simulate: the run_mjx_fixed_frequency path."""
    model = compile_model("shoulder", tmp / "scene.xml")
    data = mujoco.MjData(model)
    rest_state(model, data)
    final, traj = mjx_simulate(model, data, 300)
    return {"qpos": np.asarray(final.qpos).tolist(), "qvel": np.asarray(final.qvel).tolist(),
            "traj": vec(traj, exact=False)}


def optimizer_item(tmp):
    """opt/optimize_control.py end to end: 2 Adam iterations on a tiny window + CPU validation."""
    out, scene = tmp / "out", tmp / "scene.xml"
    stdout = run_optimizer([
        "--tg", str(tg("baseline")), "--scene", str(scene), "--out", str(out), "--harmonics", "1",
        "--warmup-periods", "0.05", "--sample-periods", "0.1", "--iters-per-stage", "2",
        "--single-stage", "--tune-tendons"])
    rec = {"stdout": [s for s in scrub(stdout, out=out, scene=scene) if not s.startswith("Failed to import")],
           "files": sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())}
    for f in ("log.json", "theta_full.json", "theta_best.json", "tendon_optimized.json", "cpu_validation.json"):
        rec[f] = json.loads((out / f).read_text())
    rec["control.tg"] = (out / "control.tg").read_text().splitlines()
    return rec


item("mjx/metrics_jacfwd", mjx_metrics_item, full=True, rtol=JAX_RTOL)
item("mjx/simulate", mjx_simulate_item, full=True, rtol=JAX_RTOL)
item("optimizer/cli", optimizer_item, full=True, rtol=JAX_RTOL)

# ================================== DIFF / CLI ==================================
_NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _close(a: float, b: float, rtol: float, ulp: float = 0.0) -> bool:
    if a == b or (a != a and b != b):
        return True
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), ulp)


def _close_text(a: str, b: str, rtol: float) -> bool:
    """Printed lines under rtol: same text, numbers equal to rtol or one printed digit."""
    if _NUM.split(a) != _NUM.split(b):
        return False
    na, nb = _NUM.findall(a), _NUM.findall(b)
    if len(na) != len(nb):
        return False
    for x, y in zip(na, nb):
        mant, _, exp = x.lower().partition("e")
        ulp = 1.01 * 10.0 ** (int(exp or 0) - len(mant.partition(".")[2]))
        if not _close(float(x), float(y), rtol, ulp):
            return False
    return True


def diff(a, b, rtol: float, path: str = "", out: list | None = None, limit: int = 8) -> list[str]:
    """First `limit` differences between baseline `a` and current `b`."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in b:
                out.append(f"{path}.{k}: missing now")
            elif k not in a:
                out.append(f"{path}.{k}: new, not in baseline")
            else:
                diff(a[k], b[k], rtol, f"{path}.{k}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, rtol, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                break
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
        if not _close(float(a), float(b), rtol):
            out.append(f"{path}: {a!r} -> {b!r}")
    elif isinstance(a, str) and isinstance(b, str) and rtol and _close_text(a, b, rtol):
        pass
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.golden", description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("check", "capture"))
    ap.add_argument("--quick", action="store_true", help="skip the full tier (JAX-compile-heavy items)")
    ap.add_argument("-k", dest="pattern", default="", help="only items whose name contains PATTERN")
    args = ap.parse_args(argv)
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {"meta": {}, "items": {}}
    meta = {"mujoco": mujoco.__version__, "jax": jax.__version__, "numpy": np.__version__}
    if args.mode == "check" and base["meta"] != meta:
        print(f"WARNING: baseline captured with {base['meta']}, running {meta}", flush=True)
    names = [n for n, (_, full, _) in ITEMS.items() if args.pattern in n and not (full and args.quick)]
    results, failed, t_all = {}, [], time.time()
    for name in names:
        fn, _, rtol = ITEMS[name]
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            try:
                cur, problems = json.loads(json.dumps(fn(Path(tmp)))), []
            except Exception:
                cur, problems = None, ["error: " + traceback.format_exc(limit=-4).strip().replace("\n", "\n    ")]
        if cur is not None and args.mode == "check":
            problems = (diff(base["items"][name], cur, rtol) if name in base["items"]
                        else ["not in baseline (run capture)"])
        results[name] = cur
        verdict = ("FAIL" if problems else "PASS") if args.mode == "check" else ("FAIL" if problems else "CAPT")
        print(f"{verdict}  {name:<48} {time.time() - t0:6.1f}s", flush=True)
        for p in problems:
            print(f"        {p}", flush=True)
        if problems:
            failed.append(name)
    tier = "quick" if args.quick else "full"
    if args.mode == "capture":
        if failed:
            print(f"capture aborted, {len(failed)} item(s) errored; baseline untouched")
            return 1
        items = {n: results.get(n, base["items"].get(n)) for n in ITEMS if n in results or n in base["items"]}
        BASELINE.write_text(json.dumps({"meta": meta, "items": items}, indent=1) + "\n")
        print(f"captured {len(results)} item(s) ({tier}) into {BASELINE.relative_to(ROOT)} in {time.time() - t_all:.0f}s")
        return 0
    print(f"{len(names) - len(failed)}/{len(names)} passed ({tier}) in {time.time() - t_all:.0f}s")
    return 1 if failed else 0
