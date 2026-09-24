"""Offline CPU rollouts behind the design-inspector GUI's plots, and its two caches.

``phase_trace`` (every phase-plot series in one rollout; ``phase_plot`` picks one),
``work_loop_data`` and ``wing_path_data`` each compile a design and settle it on the
ramped crank drive (``settled_drive``) before recording. ``cached_sim`` pickles
their results under .gui_plot_cache/ keyed on the design files' mtimes, and
``prepare_scene`` skips recompiling an unchanged .tg. ``plot_fixed_frequency_gui.py``
is the Tk front end that draws them.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import mujoco
import numpy as np

from compile_tg import compile_tg
from morphology.emit_mjcf import STRUCTURAL_TENDON_COLORS
from morphology.model import require_bird
from morphology.tg_parser import load_morphology
from runtime.control_signal import control_file_path, drive_command, load_control, load_control_for_morphology
from runtime.disk_orientation import StrokePlaneSignal
from runtime.sim import PlateWingSim

ROOT = Path(__file__).resolve().parents[2]

PLOT_ACTUATORS = ("L_motor_t1", "L_motor_t4")
# Phase-plot kind -> its y-axis label; the keys are the GUI's plot menu.
PLOT_YLABELS = {
    "Left actuator torque vs phase": "Actuator torque (mN m)",
    "Left motor tendon tension vs phase": "Motor tendon tension (N)",
    "Angular velocity vs phase": "|Angular velocity| (rad/s)",
    "Angular velocity vs time": "|Angular velocity| (rad/s)",
    "Structural tendon tension vs phase": "Structural tendon tension (N)",
    "Structural tendon displacement vs phase": "Structural tendon displacement (mm)",
    "Structural elastic energy vs phase": "Elastic energy (J)",
    "Wingtip stroke vs phase (3D)": "Wingtip excursion from mean (mm)",
    "Stroke plane deviation vs phase": "Stroke-plane deviation (deg)",
    "Mean lift per cycle": "Mean lift (mN)",
    "Mean lift per half-cycle": "Mean lift (mN)",
}
PLOT_CHOICES = tuple(PLOT_YLABELS)
MEAN_LIFT_PLOTS = ("Mean lift per cycle", "Mean lift per half-cycle")
SETTLE_PERIODS = 3
WINGTIP_SITE = "L_wing_aero_blade_04"
PLOT_CACHE_DIR = ROOT / ".gui_plot_cache"


def cached_sim(func, tg_path: Path, scene_path: Path, *args, control_path: Path | None = None,
               settle_periods: float = SETTLE_PERIODS):
    """Disk-cache sim results keyed by the design files' mtimes; unchanged designs plot instantly.

    scene_path is excluded from the key: it's a per-pane temp compile target and
    doesn't affect results.
    """
    tg = tg_path.resolve()
    control = (control_path or coupled_control_path(tg)).resolve()
    sp_json = tg.parent / "strokeplane_optimized.json"
    key = (func.__name__, Path(func.__code__.co_filename).stat().st_mtime_ns,
           str(tg), tg.stat().st_mtime_ns, str(control), control.stat().st_mtime_ns,
           sp_json.stat().st_mtime_ns if sp_json.exists() else 0, args, settle_periods)
    cache_file = PLOT_CACHE_DIR / (hashlib.sha1(repr(key).encode()).hexdigest() + ".pkl")
    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            pass  # corrupt entry: fall through and recompute
    result = func(tg_path, scene_path, *args, control_path=control_path, settle_periods=settle_periods)
    PLOT_CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_bytes(pickle.dumps(result))
    # ponytail: entries for old mtimes accumulate; `rm -r .gui_plot_cache` to clear.
    return result


def coupled_control_path(tg_path: Path) -> Path:
    """Control file referenced by a morphology, resolved relative to it."""
    morph = load_morphology(tg_path)
    require_bird(morph, "hummer", tool="the plotting GUI")
    return control_file_path(tg_path, morph)


def prepare_scene(tg_path: Path, scene_path: Path, *, color_structural_tendons: bool = False) -> Path:
    """Compile the design to ``scene_path``, only when the .tg changed."""
    tg_path = tg_path.resolve()
    scene_path = scene_path.resolve()
    stamp_file = scene_path.with_suffix(scene_path.suffix + (".color.stamp" if color_structural_tendons else ".stamp"))
    stamp_text = f"{tg_path}\n{tg_path.stat().st_mtime_ns}\n{int(color_structural_tendons)}\n"
    if not scene_path.exists() or not stamp_file.exists() or stamp_file.read_text() != stamp_text:
        compile_tg(tg_path, scene_path, color_structural_tendons=color_structural_tendons)
        stamp_file.write_text(stamp_text)
    return scene_path


def settled_drive(tg_path: Path, scene_path: Path, control_path: Path | None, settle_periods: float,
                  *, strokeplane: bool = False):
    """Compile, rest and settle a design on the ramped crank drive: the CPU loop behind every plot.

    Returns (sim, control, step, sp); step() is one physics step of the drive. With
    ``strokeplane`` a StrokePlaneSignal beside the .tg (sp, else None) is applied every
    step -- only the phase plots ask for it; work loops and the wing path run the plain
    drive.
    """
    prepare_scene(tg_path, scene_path)
    morph = load_morphology(tg_path)
    sim = PlateWingSim(mujoco.MjModel.from_xml_path(str(scene_path)), morph)
    model, data = sim.model, sim.data
    drive_start = data.time

    control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
    sp_file = tg_path.parent / "strokeplane_optimized.json"
    sp = StrokePlaneSignal.from_json(model, json.loads(sp_file.read_text())) if strokeplane and sp_file.exists() else None

    def step() -> None:
        if sp is not None:
            sp.apply(data)
        sim.step(drive_command(control, morph.motors.motor_direction, data.time, drive_start,
                               settle_periods * control.period))

    for _ in range(math.ceil(settle_periods * control.period / model.opt.timestep)):
        step()
    return sim, control, step, sp


def phase_trace(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> dict:
    """Settle the design, then record every phase-plot series over `periods` cycles.

    One rollout serves all PLOT_CHOICES; ``phase_plot`` picks one out. Series are
    per step except the mean lift, kept as per-(half-)cycle running sums.
    """
    sim, control, step, sp = settled_drive(tg_path, scene_path, control_path, settle_periods, strokeplane=True)
    model, data = sim.model, sim.data
    plot_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in PLOT_ACTUATORS}
    tendon_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) for name in PLOT_ACTUATORS}
    qvel_ids = {}
    for name in PLOT_ACTUATORS:
        prefix, station = name.split("_motor_")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{station}_disk_hinge")
        qvel_ids[name] = int(model.jnt_dofadr[jid])
    structural_ids, structural_names = structural_tendons(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)

    trace = {
        "elapsed": [], "phase": [], "U_tens": [], "tilt": [], "pitch": [], "x": [], "y": [], "z": [],
        "torque": {name: [] for name in PLOT_ACTUATORS},
        "motor_tension": {name: [] for name in PLOT_ACTUATORS},
        "speed": {name: [] for name in PLOT_ACTUATORS},
        "tension": {name: [] for name in structural_names},
        "displacement": {name: [] for name in structural_names},
        "lift": {kind: ({}, {}) for kind in MEAN_LIFT_PLOTS},  # kind -> (sums, counts) by (half-)cycle
    }
    start_time = data.time
    for _ in range(math.ceil(periods * control.period / model.opt.timestep)):
        step()
        elapsed = data.time - start_time
        phase = elapsed * control.fundamental_frequency_hz
        trace["elapsed"].append(elapsed)
        trace["phase"].append(phase)
        for name, aid in plot_ids.items():
            trace["torque"][name].append(1000.0 * float(data.actuator_force[aid]))
        for name, tid in tendon_ids.items():
            trace["motor_tension"][name].append(
                tendon_tension(model, data, tid, float(model.tendon_lengthspring[tid][0])))
        for name, dof in qvel_ids.items():
            trace["speed"][name].append(abs(float(data.qvel[dof])))
        energy = 0.0
        for name, tid in zip(structural_names, structural_ids):
            rest_length = float(model.tendon_range[tid][1])
            displacement = float(data.ten_length[tid]) - rest_length
            trace["tension"][name].append(tendon_tension(model, data, tid, rest_length))
            trace["displacement"][name].append(1000.0 * displacement)
            energy += 0.5 * float(model.tendon_stiffness[tid]) * displacement * displacement
        trace["U_tens"].append(energy)
        if sp is not None:
            tilt, pitv = sp.angles(data)
            dev_t, dev_p = tilt - sp.tilt0, pitv - sp.pitch0
        else:  # no learned signal next to this design: flat zero deviation
            dev_t = dev_p = 0.0
        trace["tilt"].append(math.degrees(dev_t))
        trace["pitch"].append(math.degrees(dev_p))
        pos = data.site_xpos[tip_id]
        trace["x"].append(1000.0 * float(pos[0]))
        trace["y"].append(1000.0 * float(pos[1]))
        trace["z"].append(1000.0 * float(pos[2]))
        if phase < periods:
            lift = sum(float(sum(force[2] for force in (wing.cached_forces or []))) for wing in sim.aero.wings)
            cycle = int(phase) + int(math.floor(settle_periods)) + 1
            for kind, key in (("Mean lift per cycle", cycle),
                              ("Mean lift per half-cycle", cycle + (0.0 if phase % 1.0 < 0.5 else 0.5))):
                lift_sums, lift_counts = trace["lift"][kind]
                lift_sums[key] = lift_sums.get(key, 0.0) + 1000.0 * lift
                lift_counts[key] = lift_counts.get(key, 0) + 1
    return trace


def phase_plot(trace: dict, plot_kind: str) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    """(x, series, title, xlabel, ylabel, colors) of one PLOT_CHOICES kind from a phase_trace."""
    x_values = trace["elapsed"] if plot_kind == "Angular velocity vs time" else trace["phase"]
    colors: dict[str, str] = {}
    if plot_kind in MEAN_LIFT_PLOTS:
        lift_sums, lift_counts = trace["lift"][plot_kind]
        x_values = sorted(lift_sums)
        series = {"mean_lift": [lift_sums[key] / lift_counts[key] for key in x_values]}
    elif plot_kind == "Left motor tendon tension vs phase":
        series = trace["motor_tension"]
    elif plot_kind in ("Angular velocity vs phase", "Angular velocity vs time"):
        series = trace["speed"]
    elif plot_kind in ("Structural tendon tension vs phase", "Structural tendon displacement vs phase"):
        series = trace["tension" if plot_kind == "Structural tendon tension vs phase" else "displacement"]
        colors = {name: STRUCTURAL_TENDON_COLORS[i] for i, name in enumerate(series)}
    elif plot_kind == "Structural elastic energy vs phase":
        series = {"U_tens": trace["U_tens"]}
    elif plot_kind == "Stroke plane deviation vs phase":
        series = {"tilt": trace["tilt"], "pitch": trace["pitch"]}
    elif plot_kind == "Wingtip stroke vs phase (3D)":
        # Center each axis on its cycle mean; flapping amplitude = peak-to-peak
        # excursion per axis, plus the resultant 3D radial excursion magnitude.
        centered = {k: np.array(trace[k]) - np.mean(trace[k]) for k in ("x", "y", "z")}
        mag = np.sqrt(sum(centered[k] ** 2 for k in ("x", "y", "z")))
        series = {
            f"{k} (pp {centered[k].max() - centered[k].min():.1f} mm)": centered[k].tolist()
            for k in ("x", "y", "z")
        }
        series[f"|3D| (max {mag.max():.1f} mm)"] = mag.tolist()
    else:
        series = trace["torque"]

    xlabel = "Crank phase (cycles)"
    if plot_kind in MEAN_LIFT_PLOTS:
        xlabel = "Cycle"
    elif plot_kind == "Angular velocity vs time":
        xlabel = "Time (s)"
    return x_values, series, plot_kind, xlabel, PLOT_YLABELS[plot_kind], colors


def data_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    plot_kind: str,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    """One phase plot straight from a fresh rollout (the GUI caches phase_trace instead)."""
    return phase_plot(phase_trace(tg_path, scene_path, periods, control_path, settle_periods), plot_kind)


def tendon_tension(model: mujoco.MjModel, data: mujoco.MjData, tid: int, rest_length: float) -> float:
    tension = (
        float(model.tendon_stiffness[tid]) * (float(data.ten_length[tid]) - rest_length)
        + float(model.tendon_damping[tid]) * float(data.ten_velocity[tid])
    )
    return max(0.0, tension)


def joint_angle_deg(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Missing joint: {joint_name}")
    return math.degrees(float(data.qpos[model.jnt_qposadr[jid]]))


def body_yaw(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Missing body: {body_name}")
    mat = data.xmat[bid]
    return math.atan2(float(mat[3]), float(mat[0]))


def structural_tendons(model: mujoco.MjModel) -> tuple[list[int], list[str]]:
    tendon_ids = [
        tid
        for tid in range(model.ntendon)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid))
        and name.startswith("L_")
        and "_motor_" not in name
    ][:12]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid) for tid in tendon_ids]
    return tendon_ids, names


def structural_tendon_stiffness_rows(tg_path: Path, scene_path: Path) -> list[tuple[str, str, float]]:
    """Return (color, tendon name, stiffness) for the left color-coded structural tendons."""
    prepare_scene(tg_path, scene_path, color_structural_tendons=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    tendon_ids, names = structural_tendons(model)
    return [
        (STRUCTURAL_TENDON_COLORS[i], name, float(model.tendon_stiffness[tid]))
        for i, (tid, name) in enumerate(zip(tendon_ids, names))
    ]


def work_loop_data(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> dict[str, object]:
    sim, control, step, _sp = settled_drive(tg_path, scene_path, control_path, settle_periods)
    model, data = sim.model, sim.data
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in PLOT_ACTUATORS}
    motor_tendon_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        for name in PLOT_ACTUATORS
    }
    structural_ids, structural_names = structural_tendons(model)

    shaft_loops = {
        "Downstroke motor (orange)": {"x": [], "y": [], "color": "#ff7f0e"},
        "Upstroke motor (blue)": {"x": [], "y": [], "color": "#1f77b4"},
    }
    motor_tendon_loops = {
        "Downstroke tendon (orange)": {"x": [], "y": [], "color": "#ff7f0e"},
        "Upstroke tendon (blue)": {"x": [], "y": [], "color": "#1f77b4"},
    }
    structural_loops = {
        name: {"x": [], "y": [], "color": STRUCTURAL_TENDON_COLORS[i]}
        for i, name in enumerate(structural_names)
    }
    wing_loops = {
        "Downstroke motor (orange)": {"x": [], "y": [], "color": "#ff7f0e"},
        "Upstroke motor (blue)": {"x": [], "y": [], "color": "#1f77b4"},
    }

    start_yaw = body_yaw(model, data, "L_part_1")
    previous_yaw = start_yaw
    unwrapped_yaw = start_yaw

    steps = math.ceil(periods * control.period / model.opt.timestep)
    for _ in range(steps):
        step()

        shaft_loops["Downstroke motor (orange)"]["x"].append(joint_angle_deg(model, data, "L_t4_disk_hinge") % 360.0)
        shaft_loops["Downstroke motor (orange)"]["y"].append(1000.0 * float(data.actuator_force[actuator_ids["L_motor_t4"]]))
        shaft_loops["Upstroke motor (blue)"]["x"].append(joint_angle_deg(model, data, "L_t1_disk_hinge") % 360.0)
        shaft_loops["Upstroke motor (blue)"]["y"].append(1000.0 * float(data.actuator_force[actuator_ids["L_motor_t1"]]))

        for label, tendon_name in (
            ("Downstroke tendon (orange)", "L_motor_t4"),
            ("Upstroke tendon (blue)", "L_motor_t1"),
        ):
            tid = motor_tendon_ids[tendon_name]
            rest = float(model.tendon_lengthspring[tid][0])
            motor_tendon_loops[label]["x"].append(1000.0 * (float(data.ten_length[tid]) - rest))
            motor_tendon_loops[label]["y"].append(tendon_tension(model, data, tid, rest))

        for name, tid in zip(structural_names, structural_ids):
            rest = float(model.tendon_range[tid][1])
            structural_loops[name]["x"].append(1000.0 * (float(data.ten_length[tid]) - rest))
            structural_loops[name]["y"].append(tendon_tension(model, data, tid, rest))

        yaw = body_yaw(model, data, "L_part_1")
        delta = yaw - previous_yaw
        if delta > math.pi:
            delta -= 2.0 * math.pi
        elif delta < -math.pi:
            delta += 2.0 * math.pi
        unwrapped_yaw += delta
        previous_yaw = yaw
        wing_angle = math.degrees(unwrapped_yaw - start_yaw)
        wing_loops["Downstroke motor (orange)"]["x"].append(wing_angle)
        wing_loops["Downstroke motor (orange)"]["y"].append(1000.0 * float(data.actuator_force[actuator_ids["L_motor_t4"]]))
        wing_loops["Upstroke motor (blue)"]["x"].append(wing_angle)
        wing_loops["Upstroke motor (blue)"]["y"].append(1000.0 * float(data.actuator_force[actuator_ids["L_motor_t1"]]))

    return {
        "control_frequency_hz": control.fundamental_frequency_hz,
        "shaft_loops": shaft_loops,
        "motor_tendon_loops": motor_tendon_loops,
        "structural_loops": structural_loops,
        "wing_loops": wing_loops,
    }



def wing_path_data(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> dict[str, list[float]]:
    """Record the wingtip site's 3D world trajectory (mm) over `periods` cycles."""
    sim, control, step, _sp = settled_drive(tg_path, scene_path, control_path, settle_periods)
    model, data = sim.model, sim.data
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)
    if tip_id < 0:
        raise ValueError(f"Missing wingtip site: {WINGTIP_SITE}")

    path: dict[str, list[float]] = {"phase": [], "x": [], "y": [], "z": []}
    start_time = data.time
    for _ in range(math.ceil(periods * control.period / model.opt.timestep)):
        step()
        path["phase"].append((data.time - start_time) * control.fundamental_frequency_hz)
        pos = data.site_xpos[tip_id]
        path["x"].append(1000.0 * float(pos[0]))
        path["y"].append(1000.0 * float(pos[1]))
        path["z"].append(1000.0 * float(pos[2]))
    return path


def wing_path_correlation(
    a: dict[str, list[float]], b: dict[str, list[float]]
) -> tuple[float, dict[str, float]]:
    """Phase-aligned similarity of two wingtip paths: overall + per-axis in [-1, 1]."""
    span = min(a["phase"][-1], b["phase"][-1])
    grid = np.linspace(0.0, span, 400)
    mats = []
    for p in (a, b):
        m = np.stack([np.interp(grid, p["phase"], p[k]) for k in "xyz"], axis=1)
        mats.append(m - m.mean(axis=0))
    pa, pb = mats

    def corr(u: np.ndarray, v: np.ndarray) -> float:
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        return float((u * v).sum() / denom) if denom else 0.0

    per_axis = {axis: corr(pa[:, i], pb[:, i]) for i, axis in enumerate("xyz")}
    return corr(pa.ravel(), pb.ravel()), per_axis
