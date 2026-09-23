from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from compile_tg import compile_tg
from morphology.emit_mjcf import STRUCTURAL_TENDON_COLORS
from morphology.model import PLATE_BIRDS, require_bird
from morphology.tg_parser import load_morphology
import runtime.aero as cpu_aero
from runtime.aero import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control, load_control_for_morphology, ramp_factor
from runtime.crank import sync_crank_disks_at_rest
from runtime.disk_orientation import DiskOrientation
from runtime.eagle_drive import CrankDrive, wing_plane_roll
from runtime.morphology_runtime import DEFAULT_HUMMER_TG, DESIGNS_DIR, load_design_morphology
from runtime.realtime import get_actuator_id
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")
PLOT_ACTUATORS = ("L_motor_t1", "L_motor_t4")
PLOT_CHOICES = (
    "Left actuator torque vs phase",
    "Left motor tendon tension vs phase",
    "Angular velocity vs phase",
    "Angular velocity vs time",
    "Structural tendon tension vs phase",
    "Structural tendon displacement vs phase",
    "Structural elastic energy vs phase",
    "Wingtip stroke vs phase (3D)",
    "Stroke plane deviation vs phase",
    "Mean lift per cycle",
    "Mean lift per half-cycle",
)
DEFAULT_PHASE_PLOTS = (
    "Left actuator torque vs phase",
    "Mean lift per cycle",
    "Left motor tendon tension vs phase",
    "Angular velocity vs phase",
)
SETTLE_PERIODS = 3
_COMPILE_CACHE: dict[tuple[Path, bool], tuple[Path, int]] = {}
PLOT_CACHE_DIR = Path(__file__).resolve().parent / ".gui_plot_cache"


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


def _short_path(path: str) -> str:
    """Path relative to the repo root when possible, else the bare name."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def coupled_control_path(tg_path: Path) -> Path:
    """Control file referenced by a morphology, resolved relative to it."""
    morph = load_morphology(tg_path)
    require_bird(morph, *PLATE_BIRDS, tool="the plotting GUI")
    control_path = Path(morph.motors.control_file)
    if not control_path.is_absolute():
        control_path = tg_path.parent / control_path
    return control_path


def design_tab_title(tg_path: str, bird: str = "") -> str:
    path = Path(tg_path)
    title = path.parent.name if path.name == "shoulder.tg" else path.stem
    return f"{title} · {bird}" if bird else title


def embed_figure(figure: Figure, master: ttk.Frame, **pack_options) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(figure, master=master)
    widget = canvas.get_tk_widget()
    widget.pack(**pack_options)
    size = None

    def resize_if_needed(event) -> None:
        nonlocal size
        new_size = (event.width, event.height)
        if new_size == size:
            return
        size = new_size
        canvas.resize(event)

    widget.bind("<Configure>", resize_if_needed)
    canvas.draw()
    return canvas


def prepare_scene(tg_path: Path, scene_path: Path, *, color_structural_tendons: bool = False) -> Path:
    """Compile only when the selected morphology changed."""
    tg_path = tg_path.resolve()
    scene_path = scene_path.resolve()
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    load_design_morphology.cache_clear()
    stamp = (tg_path, tg_path.stat().st_mtime_ns)
    key = (scene_path, color_structural_tendons)
    stamp_file = scene_path.with_suffix(scene_path.suffix + (".color.stamp" if color_structural_tendons else ".stamp"))
    stamp_text = f"{tg_path}\n{stamp[1]}\n{int(color_structural_tendons)}\n"
    if (
        _COMPILE_CACHE.get(key) != stamp
        and (not scene_path.exists() or not stamp_file.exists() or stamp_file.read_text() != stamp_text)
    ):
        compile_tg(tg_path, scene_path, color_structural_tendons=color_structural_tendons)
        stamp_file.write_text(stamp_text)
        _COMPILE_CACHE[key] = stamp
    return scene_path


def data_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    plot_kind: str,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    is_mean_lift_plot = plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle")
    prepare_scene(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    drive_start = data.time

    control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
    period = control.period
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}
    plot_ids = {name: actuator_ids[name] for name in PLOT_ACTUATORS}
    tendon_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) for name in PLOT_ACTUATORS}
    qvel_ids = {}
    for name in PLOT_ACTUATORS:
        prefix, station = name.split("_motor_")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{station}_disk_hinge")
        qvel_ids[name] = int(model.jnt_dofadr[jid])
    structural_tendon_ids = [
        tid
        for tid in range(model.ntendon)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid))
        and name.startswith("L_")
        and "_motor_" not in name
    ][:12]
    structural_tendon_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid) for tid in structural_tendon_ids
    ]
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)

    # Optimized designs carry a learned stroke-plane signal (disk-mount tilt/pitch
    # over crank phase, see optimize_control --tune-strokeplane). When the artifact
    # sits next to the tg, apply it every step so ALL plots reflect the design.
    sp_signal = None
    sp_file = tg_path.parent / "strokeplane_optimized.json"
    if sp_file.exists():
        spd = json.loads(sp_file.read_text())
        _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spd["phase_ref_joint"])
        sp_signal = (DiskOrientation(model), int(model.jnt_qposadr[_jid]),
                     np.asarray(spd["tilt_coeffs"] + spd["pitch_coeffs"], dtype=float),
                     float(spd["tilt0_rad"]), float(spd["pitch0_rad"]),
                     float(spd["max_delta_rad"]))

    def strokeplane_angles() -> tuple[float, float]:
        _, qadr, coef, tilt0, pitch0, mx = sp_signal
        th = float(data.qpos[qadr])
        basis = np.array([1.0, math.cos(th), math.sin(th), math.cos(2.0 * th), math.sin(2.0 * th)])
        return (tilt0 + mx * math.tanh(float(coef[:5] @ basis) / mx),
                pitch0 + mx * math.tanh(float(coef[5:] @ basis) / mx))

    def step() -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        caps = (cpu_aero.OMEGA_DOT_CAP, cpu_aero.LIN_ACC_CAP)
        if is_mean_lift_plot:
            cpu_aero.OMEGA_DOT_CAP = cpu_aero.LIN_ACC_CAP = 0.0
        try:
            aero.apply(model, data)
        finally:
            if is_mean_lift_plot:
                cpu_aero.OMEGA_DOT_CAP, cpu_aero.LIN_ACC_CAP = caps
        pitch.apply(model, data)
        ramp = ramp_factor(data.time, drive_start, settle_periods * control.period)
        base_ctrl = ramp * morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        if sp_signal is not None:
            sp_signal[0].set(*strokeplane_angles())
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(settle_periods * period / model.opt.timestep)):
        step()

    x_values: list[float] = []
    is_structural_plot = plot_kind in (
        "Structural tendon tension vs phase",
        "Structural tendon displacement vs phase",
    )
    is_wingtip = plot_kind == "Wingtip stroke vs phase (3D)"
    is_strokeplane = plot_kind == "Stroke plane deviation vs phase"
    names = ["x", "y", "z"] if is_wingtip else (
        ["tilt", "pitch"] if is_strokeplane else
        ["U_tens"] if plot_kind == "Structural elastic energy vs phase" else
        ["mean_lift"] if is_mean_lift_plot else
        structural_tendon_names if is_structural_plot else list(PLOT_ACTUATORS)
    )
    series = {name: [] for name in names}
    colors = (
        {name: STRUCTURAL_TENDON_COLORS[i] for i, name in enumerate(structural_tendon_names)}
        if is_structural_plot
        else {}
    )
    start_time = data.time
    steps = math.ceil(periods * period / model.opt.timestep)
    lift_sums: dict[int | float, float] = {}
    lift_counts: dict[int | float, int] = {}

    for _ in range(steps):
        step()
        elapsed = data.time - start_time
        phase = elapsed * control.fundamental_frequency_hz
        if plot_kind != "Mean lift per cycle":
            x_values.append(elapsed if plot_kind == "Angular velocity vs time" else phase)
        if is_mean_lift_plot:
            if phase >= periods:
                continue
            lift = sum(float(sum(force[2] for force in (wing.cached_forces or []))) for wing in aero.wings)
            cycle = int(phase) + int(math.floor(settle_periods)) + 1
            key = cycle if plot_kind == "Mean lift per cycle" else cycle + (0.0 if phase % 1.0 < 0.5 else 0.5)
            lift_sums[key] = lift_sums.get(key, 0.0) + 1000.0 * lift
            lift_counts[key] = lift_counts.get(key, 0) + 1
            ylabel = "Mean lift (mN)"
        elif plot_kind == "Left motor tendon tension vs phase":
            for name, tid in tendon_ids.items():
                rest_length = float(model.tendon_lengthspring[tid][0])
                tension = (
                    float(model.tendon_stiffness[tid]) * (float(data.ten_length[tid]) - rest_length)
                    + float(model.tendon_damping[tid]) * float(data.ten_velocity[tid])
                )
                series[name].append(max(0.0, tension))
            ylabel = "Motor tendon tension (N)"
        elif plot_kind in ("Angular velocity vs phase", "Angular velocity vs time"):
            for name, dof in qvel_ids.items():
                series[name].append(abs(float(data.qvel[dof])))
            ylabel = "|Angular velocity| (rad/s)"
        elif plot_kind == "Structural tendon tension vs phase":
            for name, tid in zip(structural_tendon_names, structural_tendon_ids):
                rest_length = float(model.tendon_range[tid][1])
                tension = (
                    float(model.tendon_stiffness[tid]) * (float(data.ten_length[tid]) - rest_length)
                    + float(model.tendon_damping[tid]) * float(data.ten_velocity[tid])
                )
                series[name].append(max(0.0, tension))
            ylabel = "Structural tendon tension (N)"
        elif plot_kind == "Structural tendon displacement vs phase":
            for name, tid in zip(structural_tendon_names, structural_tendon_ids):
                rest_length = float(model.tendon_range[tid][1])
                series[name].append(1000.0 * (float(data.ten_length[tid]) - rest_length))
            ylabel = "Structural tendon displacement (mm)"
        elif plot_kind == "Structural elastic energy vs phase":
            energy = 0.0
            for tid in structural_tendon_ids:
                rest_length = float(model.tendon_range[tid][1])
                displacement = float(data.ten_length[tid]) - rest_length
                energy += 0.5 * float(model.tendon_stiffness[tid]) * displacement * displacement
            series["U_tens"].append(energy)
            ylabel = "Elastic energy (J)"
        elif is_strokeplane:
            if sp_signal is not None:
                tilt, pitv = strokeplane_angles()
                dev_t, dev_p = tilt - sp_signal[3], pitv - sp_signal[4]
            else:  # no learned signal next to this design: flat zero deviation
                dev_t = dev_p = 0.0
            series["tilt"].append(math.degrees(dev_t))
            series["pitch"].append(math.degrees(dev_p))
            ylabel = "Stroke-plane deviation (deg)"
        elif is_wingtip:
            pos = data.site_xpos[tip_id]
            series["x"].append(1000.0 * float(pos[0]))
            series["y"].append(1000.0 * float(pos[1]))
            series["z"].append(1000.0 * float(pos[2]))
            ylabel = "Wingtip excursion from mean (mm)"
        else:
            for name, aid in plot_ids.items():
                series[name].append(1000.0 * float(data.actuator_force[aid]))
            ylabel = "Actuator torque (mN m)"

    if is_mean_lift_plot:
        x_values = sorted(lift_sums)
        series["mean_lift"] = [lift_sums[key] / lift_counts[key] for key in x_values]

    if is_wingtip:
        # Center each axis on its cycle mean; flapping amplitude = peak-to-peak
        # excursion per axis, plus the resultant 3D radial excursion magnitude.
        centered = {k: np.array(series[k]) - np.mean(series[k]) for k in ("x", "y", "z")}
        mag = np.sqrt(sum(centered[k] ** 2 for k in ("x", "y", "z")))
        series = {
            f"{k} (pp {centered[k].max() - centered[k].min():.1f} mm)": centered[k].tolist()
            for k in ("x", "y", "z")
        }
        series[f"|3D| (max {mag.max():.1f} mm)"] = mag.tolist()

    xlabel = "Crank phase (cycles)"
    if is_mean_lift_plot:
        xlabel = "Cycle"
    elif plot_kind == "Angular velocity vs time":
        xlabel = "Time (s)"
    return x_values, series, plot_kind, xlabel, ylabel, colors


def actuator_torque_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    settle_periods: float = SETTLE_PERIODS,
) -> tuple[list[float], dict[str, list[float]]]:
    phases, series, _title, _xlabel, _ylabel, _colors = data_by_phase(
        tg_path,
        scene_path,
        periods,
        "Left actuator torque vs phase",
        settle_periods=settle_periods,
    )
    return phases, series


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
    prepare_scene(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    drive_start = data.time

    control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}
    motor_tendon_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        for name in PLOT_ACTUATORS
    }
    structural_ids, structural_names = structural_tendons(model)

    def step() -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ramp = ramp_factor(data.time, drive_start, settle_periods * control.period)
        base_ctrl = ramp * morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(settle_periods * control.period / model.opt.timestep)):
        step()

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


WINGTIP_SITE = "L_wing_aero_blade_04"


def wing_path_data(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> dict[str, list[float]]:
    """Record the wingtip site's 3D world trajectory (mm) over `periods` cycles."""
    prepare_scene(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)
    drive_start = data.time

    control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)
    if tip_id < 0:
        raise ValueError(f"Missing wingtip site: {WINGTIP_SITE}")

    def step() -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ramp = ramp_factor(data.time, drive_start, settle_periods * control.period)
        base_ctrl = ramp * morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(settle_periods * control.period / model.opt.timestep)):
        step()

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


# ---------------------------------------------------------------------------
# Impulse response: spin the disks at one speed, kick a shaft, watch the phases.

REF_MOTOR = "L_motor_t1"
MOTOR_COLORS = {"L_motor_t1": "#1f77b4", "L_motor_t4": "#ff7f0e", "R_motor_t1": "#1f77b4", "R_motor_t4": "#ff7f0e"}
SIDE_COLORS = {"L": "#1f77b4", "R": "#d62728"}


def wrap_deg(angle, center: float = 180.0):
    """Wrap degrees into [center-180, center+180): center 180 -> [0, 360), 0 -> [-180, 180)."""
    return (angle - center + 180.0) % 360.0 + center - 180.0


def cut_wraps(phase: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where a [0, 360) phase trace wraps, so a line plot doesn't draw the flyback."""
    cut = np.flatnonzero(np.diff(phase) < -180.0) + 1
    return np.insert(phase, cut, np.nan), np.insert(values, cut, np.nan)


class ImpulseSim:
    """Live sim: crank disks at one constant speed, plus a torque pulse through chosen servos.

    Phase is each crank's travel from the synced rest angle in its OWN rotation
    sense (the actuator gear encodes the counter-rotation and the L/R mirror), so
    every motor reads 0 at sync and the .tg crank_phase_lag shows as the t4 motors
    sitting at -lag. Relative phase is against ``REF_MOTOR``.

    The motors are velocity servos, and a velocity loop cancels a torque added at
    its shaft within one step, so an additive disturbance never reaches the wing.
    The pulse therefore opens the loop: for its duration the pulsed servo outputs
    the torque-vs-phase profile it produced over its last cycle (the nominal
    drive) plus the pulse, and the wing feels the difference. Afterwards the
    velocity loop closes again, blended back in over the pulse's own duration so
    the re-lock is a finite-bandwidth recovery rather than a one-step slam.

    Nothing in the plain drive regulates phase: the t1 and t4 cranks carry
    different mean loads, so their servo slip differs and the .tg's crank lag
    walks off (~0.1 deg/cycle on the flagship), and any phase a pulse shifts
    stays shifted. ``hold_phase`` switches in the eagle's ``CrankDrive`` trim,
    which pins q_t1 + q_t4 (and the L/R pair) at the synced values, so the lag
    holds and a pulse's phase step decays back over ~1/GAIN seconds.
    """

    SAMPLE_EVERY = 4  # physics steps per recorded sample (~0.5 ms at the 0.13 ms step)
    MAX_CYCLES = 200  # history kept for the plots
    #: Per-wing kinematics in degrees, the same definitions the eagle's stroke
    #: calibration uses, mirrored so a symmetric pose reads the same on both wings:
    #:   sweep      - span direction in the horizontal plane: 0 straight out, + forward
    #:   elevation  - span direction above the horizontal, + up
    #:   roll       - plate rotation about the span (wing_plane_roll): 0 level, + nose-up
    #:   aoa        - angle between the plate and its own relative wind at 62% span (unsigned)
    #: The wingtip's fore-aft and height positions ride along (not plotted) so the
    #: stroke-plane angle can be read off the two stroke reversals, Ellington-style.
    KINEMATICS = ("sweep", "elevation", "roll", "aoa")
    FORWARD = np.array([-1.0, 0.0, 0.0])  # .tg convention: +x is the trailing edge, so -x is cranial
    AOA_BLADE = "aero_blade_03"  # (2 + 0.5) / 4 = 62.5% span, the nearest strip to the usual 70%
    TIP_SITE = "aero_blade_04"

    def __init__(self, tg_path: Path, scene_path: Path, control_path: Path | None,
                 settle_periods: float, speed: float | None = None) -> None:
        prepare_scene(tg_path, scene_path)
        self.morph = load_morphology(tg_path)
        self.model = model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = data = mujoco.MjData(model)
        calibrate_wing_pitch_at_rest(model, data)
        sync_crank_disks_at_rest(model, data)
        control = load_control(control_path) if control_path else load_control_for_morphology(tg_path)
        self.speed = control.a0 if speed is None else float(speed)  # rad/s commanded on every disk
        self.period = math.tau / abs(self.speed) if self.speed else control.period
        self.settle_periods = float(settle_periods)
        self.aero = AeroBladeRuntime(model)
        self.pitch = WingPitchLimiter(model, data)

        self.motors = tuple(name for name in ACTUATORS if get_actuator_id(model, name) >= 0)
        if not self.motors:
            raise ValueError("no crank motors in this design")
        self.aid = {name: get_actuator_id(model, name) for name in self.motors}
        self.ctrl_sign = math.copysign(1.0, self.morph.motors.motor_direction * self.speed) if self.speed else 1.0
        self.qadr: dict[str, int] = {}
        self.dof: dict[str, int] = {}
        self.direction: dict[str, float] = {}
        self.kv: dict[str, float] = {}
        self.gear: dict[str, float] = {}
        for name in self.motors:
            prefix, station = name.split("_motor_")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{station}_disk_hinge")
            if jid < 0:
                raise ValueError(f"missing disk hinge for {name}")
            aid = self.aid[name]
            self.qadr[name] = int(model.jnt_qposadr[jid])
            self.dof[name] = int(model.jnt_dofadr[jid])
            self.kv[name] = float(model.actuator_gainprm[aid][0]) or 1.0
            self.gear[name] = float(model.actuator_gear[aid][0]) or 1.0
            self.direction[name] = math.copysign(1.0, self.gear[name] * self.ctrl_sign)
        self.ref = REF_MOTOR if REF_MOTOR in self.aid else self.motors[0]
        self.theta_ref = float(data.qpos[self.qadr[self.ref]])
        self.hold_phase = False  # off: the plain drive every hummer runner uses
        self.lock = CrankDrive(model, self.morph, control)
        self.lock.capture(data)  # the synced rest phases are the ones to hold, however late it is switched on

        self.sides: dict[str, tuple[int, int, int]] = {}  # side -> (wing body, AoA blade site, tip site)
        self.outboard: dict[str, np.ndarray] = {}  # world direction the span points at rest
        for side in ("L", "R"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wing")
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_wing_{self.AOA_BLADE}")
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_wing_{self.TIP_SITE}")
            if bid >= 0 and sid >= 0 and tid >= 0:
                self.sides[side] = (bid, sid, tid)
                span = np.asarray(data.xmat[bid]).reshape(3, 3)[:, 2]
                self.outboard[side] = np.array([0.0, math.copysign(1.0, span[1]), 0.0])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self.pulse: dict[str, float] = {}  # motor -> pulse torque (N m) while it is on
        self.nominal: dict[str, Callable[[float], float]] = {}  # motor -> last cycle's torque vs phase (N m)
        self.pulse_end = -math.inf
        self.recovery = 0.0  # seconds after pulse_end over which the velocity loop is blended back in
        self.pulses: list[tuple[float, float, float, tuple[str, ...]]] = []  # (t0, t1, mN m, motors)
        self.applied = {name: 0.0 for name in self.motors}  # actuator-frame pulse torque this step (N m)
        self.reference: dict[str, np.ndarray | float] | None = None  # undisturbed stroke, see capture_reference

        keys = ["t"]
        for name in self.motors:
            keys += [f"phase:{name}", f"torque:{name}", f"applied:{name}"]
        for side in self.sides:
            keys += [f"{key}:{side}" for key in (*self.KINEMATICS, "tipf", "tipz")]
        maxlen = int(self.MAX_CYCLES * self.period / (self.SAMPLE_EVERY * model.opt.timestep)) + 1
        self.hist = {key: deque(maxlen=maxlen) for key in keys}

        self.drive_start = data.time
        self._steps = 0
        self._carry = 0.0
        for _ in range(math.ceil(self.settle_periods * self.period / model.opt.timestep)):
            self.step()
        self.t0 = data.time
        self.record()

    # -- physics ------------------------------------------------------------
    def step(self) -> None:
        model, data = self.model, self.data
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        self.aero.apply(model, data)
        self.pitch.apply(model, data)
        ramp = ramp_factor(data.time, self.drive_start, self.settle_periods * self.period)
        ctrl = ramp * self.morph.motors.motor_direction * self.speed
        t = data.time
        for name in self.motors:
            data.ctrl[self.aid[name]] = ctrl
        if self.hold_phase:
            self.lock.trim(data)
        for name in self.motors:
            aid = self.aid[name]
            tau = 0.0
            if name in self.pulse and t < self.pulse_end + self.recovery:
                # The servo's force is kv*(ctrl - gear*qvel): pick the ctrl that makes it
                # exactly what we want. Pulse: last cycle's profile plus the pulse (loop
                # open). Recovery: blend from that profile back to the velocity loop.
                gear_qvel = self.gear[name] * float(data.qvel[self.dof[name]])
                nominal = self.nominal[name](self.phase_deg(name))
                if t < self.pulse_end:
                    tau = self.pulse[name] * self.ctrl_sign  # actuator frame: + pushes forward
                    want = nominal + tau
                else:
                    s = (t - self.pulse_end) / self.recovery
                    want = (1.0 - s) * nominal + s * self.kv[name] * (ctrl - gear_qvel)
                data.ctrl[aid] = want / self.kv[name] + gear_qvel  # open loop: no trim on this shaft
            self.applied[name] = tau
        mujoco.mj_step(model, data)
        self._steps += 1

    def advance(self, seconds: float) -> None:
        """Step ``seconds`` of sim time, recording a sample every SAMPLE_EVERY steps."""
        dt = self.model.opt.timestep
        n = int((seconds + self._carry) / dt)
        self._carry = seconds + self._carry - n * dt
        for _ in range(n):
            self.step()
            if self._steps % self.SAMPLE_EVERY == 0:
                self.record()

    def inject(self, torque_mnm: float, duration_ms: float, motors) -> tuple[str, ...]:
        """Open the loop on ``motors`` for ``duration_ms`` and add ``torque_mnm`` (+ aids rotation)."""
        motors = tuple(name for name in motors if name in self.aid)
        self.nominal = {name: self.nominal_profile(name) for name in motors}
        self.pulse = {name: 1e-3 * torque_mnm for name in motors}
        self.pulse_end = self.data.time + 1e-3 * duration_ms
        self.recovery = 1e-3 * duration_ms
        self.pulses.append((self.time, self.pulse_end - self.t0, torque_mnm, motors))
        self.capture_reference()
        return motors

    def nominal_profile(self, name: str) -> Callable[[float], float]:
        """This servo's torque vs phase over its last cycle, as a periodic interpolant (N m)."""
        w = self.window(1.0)
        phase, torque = w[f"phase:{name}"], 1e-3 * w[f"torque:{name}"]
        if len(phase) < 8:  # barely started: hold what it is doing now
            hold = float(self.data.actuator_force[self.aid[name]])
            return lambda _phase: hold
        order = np.argsort(phase)
        phase, torque = phase[order], torque[order]
        return lambda p: float(np.interp(p, phase, torque, period=360.0))

    def capture_reference(self) -> None:
        """Keep the last full stroke before the pulse; the plots tile it ahead as the undisturbed motion."""
        t = np.array(self.hist["t"])
        phase = np.array(self.hist[f"phase:{self.ref}"])
        wraps = np.flatnonzero(np.diff(phase) < -180.0) + 1
        if len(wraps) < 2:
            self.reference = None
            return
        a, b = int(wraps[-2]), int(wraps[-1])
        self.reference = {"t0": float(t[a]), "period": float(t[b] - t[a]), "t": t[a:b] - t[a]}
        for side in self.sides:
            for key in self.KINEMATICS:
                self.reference[f"{key}:{side}"] = np.array(self.hist[f"{key}:{side}"])[a:b]

    # -- readings -----------------------------------------------------------
    @property
    def time(self) -> float:
        return self.data.time - self.t0

    def kinematics(self, side: str) -> dict[str, float]:
        model, data = self.model, self.data
        bid, sid, tid = self.sides[side]
        rot = np.asarray(data.xmat[bid]).reshape(3, 3)
        normal, span = rot[:, 1], rot[:, 2]
        out = {
            "sweep": math.degrees(math.atan2(float(span @ self.FORWARD), float(span @ self.outboard[side]))),
            "elevation": math.degrees(math.asin(float(np.clip(span[2], -1.0, 1.0)))),
            "roll": math.degrees(wing_plane_roll(rot, side)),
        }
        # relative wind at the blade, in the section plane perpendicular to the span
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, sid)
        vel = self._jacp @ data.qvel
        vel -= float(vel @ span) * span
        speed = float(np.linalg.norm(vel))
        # Blank around stroke reversal: with the plate nearly stopped its relative wind
        # has no direction and the reading pins at 90 deg (mid-stroke is ~2 m/s).
        out["aoa"] = (
            math.degrees(math.asin(min(1.0, abs(float(vel @ normal)) / speed))) if speed > 0.3 else math.nan
        )
        tip = data.site_xpos[tid]
        out["tipf"] = 1000.0 * float(tip @ self.FORWARD)
        out["tipz"] = 1000.0 * float(tip[2])
        return out

    def stroke_cycle(self, side: str) -> dict[str, float]:
        """Last period's stroke amplitude (deg, peak-to-peak sweep) and stroke-plane angle (deg).

        The stroke plane is the line through the wingtip at the two stroke reversals,
        measured from horizontal: + when the forward reversal is the higher one.
        Both are NaN until a full period is in the history.
        """
        w = self.window(1.0)
        if len(w["t"]) < 0.9 * self.period / (self.SAMPLE_EVERY * self.model.opt.timestep):
            return {"amplitude": math.nan, "plane": math.nan}
        sweep = w[f"sweep:{side}"]
        fwd, aft = int(np.argmax(sweep)), int(np.argmin(sweep))
        rise = w[f"tipz:{side}"][fwd] - w[f"tipz:{side}"][aft]
        run = w[f"tipf:{side}"][fwd] - w[f"tipf:{side}"][aft]
        return {"amplitude": float(sweep[fwd] - sweep[aft]), "plane": math.degrees(math.atan2(rise, run))}

    def phase_deg(self, name: str) -> float:
        travel = self.direction[name] * (float(self.data.qpos[self.qadr[name]]) - self.theta_ref)
        return wrap_deg(math.degrees(travel))

    def phases(self) -> dict[str, float]:
        return {name: self.phase_deg(name) for name in self.motors}

    def relative_deg(self, name: str) -> float:
        return wrap_deg(self.phase_deg(name) - self.phase_deg(self.ref), 0.0)

    def record(self) -> None:
        h, data = self.hist, self.data
        h["t"].append(self.time)
        for name in self.motors:
            h[f"phase:{name}"].append(self.phase_deg(name))
            h[f"torque:{name}"].append(1000.0 * float(data.actuator_force[self.aid[name]]))
            h[f"applied:{name}"].append(1000.0 * self.applied[name])
        for side in self.sides:
            for key, value in self.kinematics(side).items():
                h[f"{key}:{side}"].append(value)

    def window(self, cycles: float) -> dict[str, np.ndarray]:
        """The last ``cycles`` commanded periods of history, oldest first."""
        n = int(cycles * self.period / (self.SAMPLE_EVERY * self.model.opt.timestep)) + 1
        n = max(1, min(n, len(self.hist["t"])))
        return {
            key: np.fromiter(itertools.islice(reversed(dq), n), dtype=float, count=n)[::-1]
            for key, dq in self.hist.items()
        }

    def reference_tiled(self, key: str, t_from: float, t_to: float) -> tuple[np.ndarray, np.ndarray]:
        """The captured stroke repeated over [t_from, t_to] (NaN-separated), for overlaying."""
        ref = self.reference
        if ref is None:
            return np.array([]), np.array([])
        t0, period = ref["t0"], ref["period"]
        k_first = max(1, math.floor((t_from - t0) / period))  # the captured cycle itself is real data
        k_last = math.ceil((t_to - t0) / period)
        xs, ys = [], []
        for k in range(k_first, k_last + 1):
            xs += [t0 + k * period + ref["t"], [np.nan]]
            ys += [ref[key], [np.nan]]
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.array([]), np.array([]))


class DesignPane:
    def __init__(
        self,
        root: tk.Tk,
        parent: ttk.Frame,
        status: tk.StringVar,
        default_tg: Path,
        on_title_change: Callable[[str], None],
    ) -> None:
        self.root = root
        self.status = status
        self.on_title_change = on_title_change
        self.scene_dir = Path(tempfile.mkdtemp(prefix="diffsim_gui_", dir="/private/tmp"))
        self.scene_path = self.scene_dir / "scene.xml"
        self.tendon_scene_path = self.scene_dir / "scene_tendons.xml"
        self.viewer_scene_path = self.scene_dir / "scene_viewer.xml"

        self.tg_var = tk.StringVar(value=str(default_tg))
        self.control_var = tk.StringVar(value=str(coupled_control_path(default_tg)))
        self.bird = load_morphology(default_tg).bird
        self.tg_display = tk.StringVar()
        self.control_display = tk.StringVar()
        self.tg_var.trace_add("write", lambda *_: self.tg_display.set(_short_path(self.tg_var.get())))
        self.control_var.trace_add("write", lambda *_: self.control_display.set(_short_path(self.control_var.get())))
        self.tg_display.set(_short_path(self.tg_var.get()))
        self.control_display.set(_short_path(self.control_var.get()))

        self.settle_periods = tk.DoubleVar(value=float(SETTLE_PERIODS))

        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_tab = ttk.Frame(self.tabs)
        self.work_tab = ttk.Frame(self.tabs)
        self.wing_path_tab = ttk.Frame(self.tabs)
        self.impulse_tab = ttk.Frame(self.tabs)
        self.viewer_tab = ttk.Frame(self.tabs)
        self.settings_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.plot_tab, text="Phase plots")
        self.tabs.add(self.work_tab, text="Work loops")
        self.tabs.add(self.wing_path_tab, text="Wing path")
        self.tabs.add(self.impulse_tab, text="Impulse response")
        self.tabs.add(self.viewer_tab, text="Viewer")
        self.tabs.add(self.settings_tab, text="Settings")
        self.viewer_proc: subprocess.Popen | None = None
        self.wing_paths: dict[str, dict] = {}
        self.build_wing_path_tab()
        self.build_impulse_tab()
        self.build_viewer_tab()
        self.build_settings_tab()

        self.grid = ttk.Frame(self.plot_tab, padding=8)
        self.grid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.grid.columnconfigure(0, weight=1)
        self.grid.columnconfigure(1, weight=1)
        self.grid.rowconfigure(0, weight=1)
        self.grid.rowconfigure(1, weight=1)

        self.slots: list[ttk.Frame] = []
        self.slot_controls: dict[ttk.Frame, tuple[ttk.Combobox, tk.DoubleVar, ttk.Button]] = {}
        self.plotted_slots: set[ttk.Frame] = set()
        for plot_kind in DEFAULT_PHASE_PLOTS:
            self.add_prompt(plot_kind)
        self.build_work_tab()

    def build_wing_path_tab(self) -> None:
        controls = ttk.Frame(self.wing_path_tab, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="Periods").pack(side=tk.LEFT)
        self.wing_path_periods = tk.DoubleVar(value=2.0)
        ttk.Entry(controls, textvariable=self.wing_path_periods, width=6).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Button(controls, text="Capture as A", command=lambda: self.capture_wing_path("A")).pack(side=tk.LEFT)
        ttk.Button(controls, text="Capture as B", command=lambda: self.capture_wing_path("B")).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.wing_corr = tk.StringVar(value="Capture A and B (from the design in Settings) to compare.")
        ttk.Label(self.wing_path_tab, textvariable=self.wing_corr, padding=(12, 0)).pack(
            side=tk.TOP, anchor=tk.W
        )
        self.wing_path_plot = ttk.Frame(self.wing_path_tab, padding=12)
        self.wing_path_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def capture_wing_path(self, slot: str) -> None:
        periods = float(self.wing_path_periods.get())
        if periods <= 0:
            self.status.set("Periods must be positive.")
            return
        settle_periods = self.get_settle_periods()
        if settle_periods is None:
            return
        self.status.set(f"Tracing wing path {slot}...")
        self.root.update_idletasks()
        try:
            path = cached_sim(
                wing_path_data,
                Path(self.tg_var.get()),
                self.scene_path,
                periods,
                control_path=Path(self.control_var.get()),
                settle_periods=settle_periods,
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            raise
        path["label"] = f"{slot}: {_short_path(self.tg_var.get())}"
        self.wing_paths[slot] = path
        self.redraw_wing_paths()
        self.status.set(f"Captured wing path {slot} ({_short_path(self.tg_var.get())}).")

    def redraw_wing_paths(self) -> None:
        for child in self.wing_path_plot.winfo_children():
            child.destroy()
        figure = Figure(figsize=(7, 6), dpi=100)
        ax = figure.add_subplot(111, projection="3d")
        for slot, color in (("A", "#1f77b4"), ("B", "#ff7f0e")):
            path = self.wing_paths.get(slot)
            if path:
                ax.plot(path["x"], path["y"], path["z"], color=color, label=path["label"])
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_title("Wingtip path in 3D")
        if self.wing_paths:
            ax.legend(loc="upper left", fontsize=8)

        if "A" in self.wing_paths and "B" in self.wing_paths:
            overall, per_axis = wing_path_correlation(self.wing_paths["A"], self.wing_paths["B"])
            axes = ", ".join(f"{k}={v:+.3f}" for k, v in per_axis.items())
            self.wing_corr.set(f"Path correlation A↔B: overall {overall:+.3f}   (per axis: {axes})")

        embed_figure(figure, self.wing_path_plot, side=tk.TOP, fill=tk.BOTH, expand=True)

    IMPULSE_WINDOW_CYCLES = 4.0

    def build_impulse_tab(self) -> None:
        self.impulse_sim: ImpulseSim | None = None
        self.impulse_running = False
        self.impulse_wall = 0.0
        self.impulse_drawn = 0.0
        self.impulse_lines: dict[str, object] = {}
        self.impulse_rate = tk.DoubleVar(value=0.2)
        self.impulse_torque = tk.DoubleVar(value=100.0)
        self.impulse_duration = tk.DoubleVar(value=20.0)
        self.impulse_targets = {name: tk.BooleanVar(value=True) for name in ACTUATORS}
        self.impulse_hold = tk.BooleanVar(value=False)
        self.impulse_readout = tk.StringVar(value="")

        bar = ttk.Frame(self.impulse_tab, padding=(12, 10, 12, 0))
        bar.pack(side=tk.TOP, fill=tk.X)
        self.impulse_start = ttk.Button(bar, text="Start", width=7, command=self.impulse_toggle)
        self.impulse_start.pack(side=tk.LEFT)
        ttk.Button(bar, text="Reset", width=7, command=self.impulse_reset).pack(side=tk.LEFT, padx=(6, 24))
        ttk.Label(bar, text="Pulse").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.impulse_torque, width=7).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(bar, text="mN·m for").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.impulse_duration, width=5).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(bar, text="ms on").pack(side=tk.LEFT)
        for name, var in self.impulse_targets.items():
            ttk.Checkbutton(bar, text=name.replace("_motor_", " "), variable=var).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(bar, text="Inject  (Cmd-I)", command=self.impulse_inject).pack(side=tk.LEFT, padx=(16, 24))
        ttk.Label(bar, text="Playback ×").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.impulse_rate, width=5).pack(side=tk.LEFT, padx=(6, 24))
        ttk.Checkbutton(bar, text="Hold crank phase", variable=self.impulse_hold, command=self.impulse_set_hold).pack(
            side=tk.LEFT
        )
        ttk.Label(self.impulse_tab, textvariable=self.impulse_readout, font=("Menlo", 11), padding=(12, 4)).pack(
            side=tk.TOP, anchor=tk.W
        )

        plots = ttk.Frame(self.impulse_tab, padding=(12, 0, 12, 12))
        plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        left = ttk.Frame(plots)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right = ttk.Frame(plots)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        fig = Figure(figsize=(7, 7), dpi=100)
        self.ax_torque, self.ax_rel = fig.subplots(2, 1, height_ratios=[1.6, 1.0])
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.08, top=0.94, hspace=0.4)
        self.impulse_canvas_left = embed_figure(fig, left, side=tk.TOP, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(7, 7), dpi=100)
        self.ax_kin = dict(zip(ImpulseSim.KINEMATICS, fig.subplots(4, 1, sharex=True)))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.07, top=0.95, hspace=0.25)
        self.impulse_canvas_right = embed_figure(fig, right, side=tk.TOP, fill=tk.BOTH, expand=True)
        self.impulse_style_axes()

    def impulse_style_axes(self) -> None:
        ax = self.ax_torque
        ax.set_title("Servo torque vs crank phase")
        ax.set_xlabel("Crank phase (deg, own rotation, from synced rest)")
        ax.set_ylabel("Torque (mN·m)")
        ax.set_xlim(0.0, 360.0)
        ax.xaxis.set_major_locator(MultipleLocator(45.0))
        ax.grid(True, alpha=0.3)
        ax = self.ax_rel
        ax.set_title(f"Crank phase relative to {REF_MOTOR.replace('_motor_', ' ')}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        self.ax_kin["sweep"].set_title("Wing kinematics")
        self.ax_kin["sweep"].set_ylabel("Sweep (deg, + fwd)")
        self.ax_kin["elevation"].set_ylabel("Elevation (deg, + up)")
        self.ax_kin["roll"].set_ylabel("Roll about span (deg)")
        self.ax_kin["aoa"].set_ylabel("AoA at 62% span (deg)")
        self.ax_kin["aoa"].set_xlabel("Time (s)")
        for ax in self.ax_kin.values():
            ax.grid(True, alpha=0.3)

    def impulse_make_lines(self) -> None:
        sim = self.impulse_sim
        lines = self.impulse_lines = {}
        for ax in (self.ax_torque, self.ax_rel, *self.ax_kin.values()):
            for artist in list(ax.lines) + list(ax.patches):
                artist.remove()
        for name in sim.motors:
            color = MOTOR_COLORS.get(name, "k")
            right = name.startswith("R_")
            style = "--" if right else "-"
            label = name.replace("_motor_", " ")
            (lines[f"torque:{name}"],) = self.ax_torque.plot([], [], style, color=color, lw=1.2, label=label)
            (lines[f"applied:{name}"],) = self.ax_torque.plot([], [], ":", color=color, lw=2.2)
            (lines[f"now:{name}"],) = self.ax_torque.plot([], [], "o", color=color, ms=7, mfc="white" if right else color)
            if name != sim.ref:
                (lines[f"rel:{name}"],) = self.ax_rel.plot([], [], style, color=color, lw=1.2, label=label)
        for side in sim.sides:
            for key, ax in self.ax_kin.items():
                (lines[f"ref:{key}:{side}"],) = ax.plot([], [], ":", color="0.55", lw=1.0)
                (lines[f"{key}:{side}"],) = ax.plot([], [], color=SIDE_COLORS[side], lw=1.2, label=f"{side} wing")
        self.ax_torque.legend(loc="upper right", fontsize=8, ncol=len(sim.motors))
        if len(sim.motors) > 1:
            self.ax_rel.legend(loc="upper left", fontsize=8)
        if sim.sides:
            self.ax_kin["sweep"].legend(loc="upper right", fontsize=8)

    def impulse_set_hold(self) -> None:
        if self.impulse_sim is not None:
            self.impulse_sim.hold_phase = bool(self.impulse_hold.get())
            self.status.set(
                "Holding crank phase (CrankDrive trim)." if self.impulse_sim.hold_phase else "Plain drive: no phase hold."
            )

    def impulse_reset(self) -> bool:
        settle_periods = self.get_settle_periods()
        if settle_periods is None:
            return False
        self.impulse_running = False
        self.impulse_start.configure(text="Start")
        self.status.set("Compiling and settling the impulse sim...")
        self.root.update_idletasks()
        try:
            self.impulse_sim = ImpulseSim(
                Path(self.tg_var.get()), self.scene_path, Path(self.control_var.get()), settle_periods
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            raise
        self.impulse_sim.hold_phase = bool(self.impulse_hold.get())
        self.impulse_make_lines()
        self.impulse_redraw()
        self.status.set(
            f"Impulse sim ready: disks at {self.impulse_sim.speed:g} rad/s "
            f"({1.0 / self.impulse_sim.period:.2f} Hz) after {settle_periods:g} settle periods. Press Start."
        )
        return True

    def impulse_toggle(self) -> None:
        if self.impulse_sim is None and not self.impulse_reset():
            return
        self.impulse_running = not self.impulse_running
        self.impulse_start.configure(text="Pause" if self.impulse_running else "Start")
        if self.impulse_running:
            self.impulse_wall = time.perf_counter()
            self.impulse_tick()

    def impulse_tick(self) -> None:
        if not self.impulse_running or self.impulse_sim is None:
            return
        now = time.perf_counter()
        wall = min(now - self.impulse_wall, 0.25)
        self.impulse_wall = now
        try:
            rate = max(0.0, float(self.impulse_rate.get()))
        except (tk.TclError, ValueError):
            rate = 0.2
        try:
            self.impulse_sim.advance(wall * rate)
            if now - self.impulse_drawn >= 0.09:  # ~10 fps is plenty for plots; the sim keeps wall pace regardless
                self.impulse_redraw()
            self.root.after(15, self.impulse_tick)
        except tk.TclError:
            return  # window went away mid-tick

    def impulse_redraw(self) -> None:
        sim, lines = self.impulse_sim, self.impulse_lines
        w = sim.window(self.IMPULSE_WINDOW_CYCLES)
        t = w["t"]
        ref = w[f"phase:{sim.ref}"]
        for name in sim.motors:
            phase, torque, applied = w[f"phase:{name}"], w[f"torque:{name}"], w[f"applied:{name}"]
            lines[f"torque:{name}"].set_data(*cut_wraps(phase, torque))
            lines[f"applied:{name}"].set_data(*cut_wraps(phase, np.where(applied != 0.0, applied, np.nan)))
            lines[f"now:{name}"].set_data([phase[-1]], [torque[-1]])
            if name != sim.ref:
                lines[f"rel:{name}"].set_data(t, wrap_deg(phase - ref, 0.0))
        t_end = max(t[-1], t[0] + 0.5 * sim.period)
        for side in sim.sides:
            for key in sim.KINEMATICS:
                lines[f"{key}:{side}"].set_data(t, w[f"{key}:{side}"])
                lines[f"ref:{key}:{side}"].set_data(*sim.reference_tiled(f"{key}:{side}", t[0], t_end))
        for ax in (self.ax_torque, self.ax_rel, *self.ax_kin.values()):
            if ax is not self.ax_torque:
                ax.set_xlim(t[0], t_end)
            ax.relim()
            ax.autoscale_view(scalex=False)
        phases = sim.phases()
        relative = "  ".join(
            f"{name.replace('_motor_', ' ')} {wrap_deg(phases[name] - phases[sim.ref], 0.0):+.1f}°"
            for name in sim.motors
            if name != sim.ref
        )
        wings = "   ".join(
            f"{side}: amp {cyc['amplitude']:.0f}° plane {cyc['plane']:.0f}°"
            for side in sim.sides
            for cyc in (sim.stroke_cycle(side),)
        )
        self.impulse_readout.set(
            f"t = {sim.time:7.3f} s   cycle {sim.time / sim.period:5.1f}   "
            f"phase vs {sim.ref.replace('_motor_', ' ')}:  {relative}   |   stroke {wings}"
        )
        self.impulse_canvas_left.draw_idle()
        self.impulse_canvas_right.draw_idle()
        self.impulse_drawn = time.perf_counter()

    def impulse_inject(self) -> None:
        sim = self.impulse_sim
        if sim is None:
            self.status.set("Start (or Reset) the impulse sim first.")
            return
        try:
            torque = float(self.impulse_torque.get())
            duration = float(self.impulse_duration.get())
        except (tk.TclError, ValueError):
            self.status.set("Pulse torque and duration must be numbers.")
            return
        if duration <= 0:
            self.status.set("Duration must be positive.")
            return
        motors = sim.inject(torque, duration, [name for name, var in self.impulse_targets.items() if var.get()])
        if not motors:
            self.status.set("Pick at least one motor to pulse.")
            return
        for ax in (self.ax_rel, *self.ax_kin.values()):
            ax.axvline(sim.time, color="k", ls="--", lw=0.8, alpha=0.5)
        self.status.set(
            f"Pulse {torque:+g} mN·m for {duration:g} ms on "
            f"{', '.join(name.replace('_motor_', ' ') for name in motors)} at t = {sim.time:.3f} s, "
            f"{sim.ref.replace('_motor_', ' ')} phase {sim.phase_deg(sim.ref):.0f}°."
        )
        if not self.impulse_running:
            self.impulse_redraw()

    def build_viewer_tab(self) -> None:
        self.viewer_color = tk.BooleanVar(value=False)
        self.viewer_speed = tk.DoubleVar(value=0.2)

        controls = ttk.Frame(self.viewer_tab, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(controls, text="Launch / reload viewer", command=self.launch_viewer).pack(side=tk.LEFT)
        ttk.Button(controls, text="Close viewer", command=self.close_viewer).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(controls, text="Color structural tendons (--color)", variable=self.viewer_color).pack(
            side=tk.LEFT, padx=(16, 0)
        )
        ttk.Label(controls, text="Speed").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Entry(controls, textvariable=self.viewer_speed, width=6).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            self.viewer_tab,
            text="Launches the MuJoCo passive viewer for the selected morphology and control "
            "(set in Settings). Each launch re-reads the .tg files, so click again after editing "
            "to reload. A new launch closes the previous viewer window.",
            wraplength=720,
            justify=tk.LEFT,
        ).pack(side=tk.TOP, anchor=tk.W, padx=12)

    def launch_viewer(self) -> None:
        if self.viewer_proc and self.viewer_proc.poll() is None:
            self.viewer_proc.terminate()
        # macOS requires mujoco's mjpython (not plain python) for launch_passive.
        launcher = sys.executable
        if sys.platform == "darwin":
            mjpython = Path(sys.executable).with_name("mjpython")
            if not mjpython.exists():
                self.status.set(f"mjpython not found next to {sys.executable}; viewer needs it on macOS.")
                return
            launcher = str(mjpython)
        cmd = [
            launcher,
            str(ROOT / "src" / "view_fixed_frequency.py"),
            "--tg", self.tg_var.get(),
            "--control", self.control_var.get(),
            "--scene", str(self.viewer_scene_path),
            "--speed", str(float(self.viewer_speed.get())),
        ]
        if self.viewer_color.get():
            cmd.append("--color")
        self.viewer_proc = subprocess.Popen(cmd, cwd=str(ROOT))
        self.status.set(f"Launched viewer for {_short_path(self.tg_var.get())}")

    def close_viewer(self) -> None:
        if self.viewer_proc and self.viewer_proc.poll() is None:
            self.viewer_proc.terminate()
            self.status.set("Closed viewer.")
        else:
            self.status.set("No viewer running.")

    def build_settings_tab(self) -> None:
        bar = ttk.LabelFrame(self.settings_tab, text="Coupled design", padding=(12, 10))
        bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)
        bar.columnconfigure(2, weight=1)
        ttk.Button(bar, text="Morphology…", width=13, command=self.choose_morphology).grid(row=0, column=0, sticky="w")
        ttk.Label(bar, text="shoulder").grid(row=0, column=1, padx=(10, 8))
        ttk.Label(bar, textvariable=self.tg_display).grid(row=0, column=2, sticky="w")
        ttk.Button(bar, text="Control…", width=13, command=self.choose_control).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(bar, text="control").grid(row=1, column=1, padx=(10, 8), pady=(8, 0))
        ttk.Label(bar, textvariable=self.control_display).grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Label(bar, text="settle periods").grid(row=2, column=1, padx=(10, 8), pady=(8, 0))
        ttk.Entry(bar, textvariable=self.settle_periods, width=8).grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Label(
            self.settings_tab,
            text="Picking a morphology loads its coupled control automatically. "
            "Override the control to pair a design with a different signal.",
            wraplength=720,
            justify=tk.LEFT,
        ).pack(side=tk.TOP, anchor=tk.W, padx=12)
        self.tendon_table = ttk.LabelFrame(
            self.settings_tab,
            text="Color-coded structural tendon stiffness",
            padding=(12, 10),
        )
        self.tendon_table.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)
        self.refresh_tendon_table()

    def refresh_tendon_table(self) -> None:
        for child in self.tendon_table.winfo_children():
            child.destroy()
        try:
            rows = structural_tendon_stiffness_rows(
                Path(self.tg_var.get()),
                self.tendon_scene_path,
            )
        except Exception as exc:
            ttk.Label(self.tendon_table, text=f"Unable to load tendon stiffness: {exc}").grid(
                row=0, column=0, sticky="w"
            )
            return

        ttk.Label(self.tendon_table, text="").grid(row=0, column=0, sticky="w")
        ttk.Label(self.tendon_table, text="Tendon").grid(row=0, column=1, sticky="w", padx=(6, 16))
        ttk.Label(self.tendon_table, text="Stiffness (N/m)").grid(row=0, column=2, sticky="e")
        for row, (color, name, stiffness) in enumerate(rows, start=1):
            tk.Label(self.tendon_table, text="  ", bg=color, width=2).grid(row=row, column=0, sticky="w", pady=1)
            ttk.Label(self.tendon_table, text=name.removeprefix("L_")).grid(
                row=row, column=1, sticky="w", padx=(6, 16), pady=1
            )
            ttk.Label(self.tendon_table, text=f"{stiffness:g}").grid(row=row, column=2, sticky="e", pady=1)

    def choose_morphology(self) -> None:
        path = filedialog.askopenfilename(
            title="Select morphology (.tg)",
            initialdir=str(Path(self.tg_var.get()).parent),
            filetypes=[("Tensegrity morphology", "*.tg")],
        )
        if not path:
            return
        try:
            control = coupled_control_path(Path(path))  # also refuses non-plate-wing birds
        except Exception as exc:
            self.status.set(f"Not loaded: {exc}")
            return
        self.tg_var.set(path)
        self.control_var.set(str(control))
        self.bird = load_morphology(Path(path)).bird
        self.refresh_tendon_table()
        self.on_title_change(design_tab_title(path, self.bird))
        self.status.set(f"Design: {Path(path).name} ({self.bird}; coupled control loaded)")

    def choose_control(self) -> None:
        path = filedialog.askopenfilename(
            title="Select control (.tg)",
            initialdir=str(Path(self.control_var.get()).parent),
            filetypes=[("Control", "*.tg")],
        )
        if not path:
            return
        try:
            control_bird = load_control(Path(path)).bird
        except Exception as exc:
            self.status.set(f"Not loaded: {exc}")
            return
        if control_bird != self.bird:
            self.status.set(f"Not loaded: {Path(path).name} is tagged bird = {control_bird}, this design is {self.bird}.")
            return
        self.control_var.set(path)
        self.status.set(f"Control: {Path(path).name}")

    def reload_design(self) -> None:
        tg_path = Path(self.tg_var.get())
        control_path = Path(self.control_var.get())
        self.status.set("Reloading design files...")
        self.root.update_idletasks()
        try:
            load_control(control_path)
            prepare_scene(tg_path, self.scene_path)
            self.refresh_tendon_table()
            if self.viewer_proc and self.viewer_proc.poll() is None:
                self.viewer_proc.terminate()
                self.viewer_proc = None
                self.status.set(f"Reloaded {_short_path(str(tg_path))}; closed viewer.")
            else:
                self.status.set(f"Reloaded {_short_path(str(tg_path))}")
        except Exception as exc:
            self.status.set(f"Reload failed: {exc}")
            raise

    def get_settle_periods(self) -> float | None:
        try:
            settle_periods = float(self.settle_periods.get())
        except (tk.TclError, ValueError):
            self.status.set("Settle periods must be a number.")
            return None
        if settle_periods < 0:
            self.status.set("Settle periods must be non-negative.")
            return None
        return settle_periods

    def add_prompt(self, default_plot: str | None = None) -> None:
        if len(self.slots) >= 4:
            return

        index = len(self.slots)
        frame = ttk.Frame(self.grid, padding=8, relief=tk.GROOVE)
        frame.grid(row=index % 2, column=index // 2, sticky="nsew", padx=6, pady=6)
        self.slots.append(frame)

        controls = ttk.Frame(frame)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Plot").pack(side=tk.LEFT)
        plot_choice = ttk.Combobox(controls, values=PLOT_CHOICES, state="readonly", width=36)
        plot_choice.current(PLOT_CHOICES.index(default_plot) if default_plot in PLOT_CHOICES else 0)
        plot_choice.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(controls, text="Periods").pack(side=tk.LEFT)
        periods = tk.DoubleVar(value=5.0)
        ttk.Entry(controls, textvariable=periods, width=8).pack(side=tk.LEFT, padx=(6, 12))

        button = ttk.Button(
            controls,
            text="Plot",
            command=lambda: self.plot_in_slot(frame, plot_choice, periods, button),
        )
        button.pack(side=tk.LEFT)
        self.slot_controls[frame] = (plot_choice, periods, button)

    def update_phase_plots(self) -> None:
        self.reload_design()
        while len(self.slots) < len(DEFAULT_PHASE_PLOTS):
            self.add_prompt(DEFAULT_PHASE_PLOTS[len(self.slots)])
        errors = []
        for frame in list(self.slots):
            plot_choice, periods, button = self.slot_controls[frame]
            try:
                self.plot_in_slot(frame, plot_choice, periods, button)
            except Exception as exc:
                errors.append(f"{plot_choice.get()}: {exc}")
        if errors:
            self.status.set("Plot errors — " + "; ".join(errors))

    def update_current_tab_graphs(self) -> None:
        selected = self.tabs.select()
        if selected == str(self.work_tab):
            self.plot_work_loops()
        elif selected == str(self.impulse_tab):
            self.impulse_reset()
        elif selected == str(self.wing_path_tab):
            slots = list(self.wing_paths) or ["A"]
            for slot in slots:
                self.capture_wing_path(slot)
        else:
            # Viewer/Settings (where fresh tabs usually sit) have no graphs of
            # their own: jump to phase plots so Cmd+U always visibly does something.
            self.tabs.select(self.plot_tab)
            self.update_phase_plots()

    def plot_in_slot(
        self,
        frame: ttk.Frame,
        plot_choice: ttk.Combobox,
        periods_var: tk.DoubleVar,
        button: ttk.Button,
    ) -> None:
        periods = float(periods_var.get())
        if periods <= 0:
            self.status.set("Periods must be positive.")
            return
        settle_periods = self.get_settle_periods()
        if settle_periods is None:
            return

        button.configure(state=tk.DISABLED)
        self.status.set("Running simulation...")
        self.root.update_idletasks()
        try:
            x_values, series, title, xlabel, ylabel, colors = cached_sim(
                data_by_phase,
                Path(self.tg_var.get()),
                self.scene_path,
                periods,
                plot_choice.get(),
                control_path=Path(self.control_var.get()),
                settle_periods=settle_periods,
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            raise
        finally:
            button.configure(state=tk.NORMAL)

        for child in frame.winfo_children()[1:]:
            child.destroy()

        plot_area = ttk.Frame(frame)
        plot_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        figure = Figure(figsize=(7, 4), dpi=100)
        ax = figure.add_subplot(111)
        lines = {}
        if title in ("Mean lift per cycle", "Mean lift per half-cycle"):
            width = 0.8 if title == "Mean lift per cycle" else 0.4
            ax.bar(x_values, series["mean_lift"], width=width, color="#1f77b4")
            if title == "Mean lift per half-cycle":
                ax.set_xticks(x_values)
                ax.set_xticklabels(
                    [f"{int(x)} {'up' if x % 1.0 == 0.0 else 'down'}" for x in x_values],
                    rotation=45,
                    ha="right",
                )
        else:
            for name, values in series.items():
                (line,) = ax.plot(x_values, values, label=name, color=colors.get(name))
                if colors:
                    line.set_visible(False)
                lines[name] = line
        if title == "Left actuator torque vs phase":
            peak = max((max(map(abs, v)) for v in series.values() if v), default=0.0)
            title = f"{title} — peak {peak:.1f} mN·m"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title in ("Angular velocity vs phase", "Angular velocity vs time"):
            vmax = max((max(values) for values in series.values() if values), default=0.0)
            ax.set_ylim(0.0, max(35.0, math.ceil(vmax / 5.0) * 5.0))
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
            if title == "Angular velocity vs time":
                ax.set_xlim(50.0, 75.0)
        ax.grid(True, alpha=0.3)

        def update_legend() -> None:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            visible_lines = [line for line in lines.values() if line.get_visible()]
            if visible_lines:
                ax.legend(visible_lines, [line.get_label() for line in visible_lines])

        if colors:
            update_legend()
        elif lines:
            ax.legend()

        canvas = embed_figure(figure, plot_area, side=tk.LEFT, fill=tk.BOTH, expand=True)

        if colors:
            toggles = ttk.Frame(plot_area, padding=(8, 0, 0, 0))
            toggles.pack(side=tk.RIGHT, fill=tk.Y)
            for name, line in lines.items():
                visible = tk.BooleanVar(value=False)

                def toggle(line=line, visible=visible) -> None:
                    line.set_visible(visible.get())
                    update_legend()
                    canvas.draw_idle()

                row = ttk.Frame(toggles)
                row.pack(side=tk.TOP, fill=tk.X, anchor=tk.W)
                tk.Label(row, text="  ", bg=colors[name], width=2).pack(side=tk.LEFT, padx=(0, 4), pady=1)
                ttk.Checkbutton(
                    row,
                    text=name.removeprefix("L_"),
                    variable=visible,
                    command=toggle,
                ).pack(side=tk.LEFT, anchor=tk.W)

        self.plotted_slots.add(frame)
        button.configure(text="Update")
        if len(self.slots) < 4 and len(self.plotted_slots) == len(self.slots):
            self.add_prompt()
        control = load_control(Path(self.control_var.get()))
        self.status.set(
            f"Plotted {periods:g} periods at control f0 {control.fundamental_frequency_hz:g} Hz, "
            f"after {settle_periods:g} settle periods."
        )

    def build_work_tab(self) -> None:
        controls = ttk.Frame(self.work_tab, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Periods").pack(side=tk.LEFT)
        self.work_periods = tk.DoubleVar(value=5.0)
        ttk.Entry(controls, textvariable=self.work_periods, width=8).pack(side=tk.LEFT, padx=(6, 12))

        self.work_button = ttk.Button(controls, text="Plot work loops", command=self.plot_work_loops)
        self.work_button.pack(side=tk.LEFT)

        self.work_sets = ttk.Notebook(self.work_tab)
        self.work_sets.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.shaft_frame = ttk.Frame(self.work_sets, padding=8)
        self.motor_tendon_frame = ttk.Frame(self.work_sets, padding=8)
        self.structural_frame = ttk.Frame(self.work_sets, padding=8)
        self.wing_frame = ttk.Frame(self.work_sets, padding=8)

        self.work_sets.add(self.shaft_frame, text="Motor shafts")
        self.work_sets.add(self.motor_tendon_frame, text="Motor tendons")
        self.work_sets.add(self.structural_frame, text="Passive tendons")
        self.work_sets.add(self.wing_frame, text="Wing angle vs motor torque")

    def clear_frame(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def plot_single_loop(
        self,
        parent: ttk.Frame,
        loop: dict[str, object],
        title: str,
        xlabel: str,
        ylabel: str,
        row: int,
        column: int,
    ) -> None:
        frame = ttk.Frame(parent, padding=6)
        frame.grid(row=row, column=column, sticky="nsew")

        figure = Figure(figsize=(6, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(loop["x"], loop["y"], color=str(loop["color"]))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        embed_figure(figure, frame, side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_structural_work_loops(self, loops: dict[str, dict[str, object]]) -> None:
        plot_area = ttk.Frame(self.structural_frame)
        plot_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        figure = Figure(figsize=(8, 5), dpi=100)
        ax = figure.add_subplot(111)
        lines = {}
        for name, loop in loops.items():
            (line,) = ax.plot(loop["x"], loop["y"], label=name, color=str(loop["color"]))
            line.set_visible(False)
            lines[name] = line
        ax.set_title("Passive tensegrity tendon work loops")
        ax.set_xlabel("Tendon extension (mm)")
        ax.set_ylabel("Tendon tension (N)")
        ax.grid(True, alpha=0.3)

        def update_legend() -> None:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            visible_lines = [line for line in lines.values() if line.get_visible()]
            if visible_lines:
                ax.legend(visible_lines, [line.get_label() for line in visible_lines])

        canvas = embed_figure(figure, plot_area, side=tk.LEFT, fill=tk.BOTH, expand=True)

        toggles = ttk.Frame(plot_area, padding=(8, 0, 0, 0))
        toggles.pack(side=tk.RIGHT, fill=tk.Y)
        for name, line in lines.items():
            visible = tk.BooleanVar(value=False)
            color = str(loops[name]["color"])

            def toggle(line=line, visible=visible) -> None:
                line.set_visible(visible.get())
                update_legend()
                canvas.draw_idle()

            row = ttk.Frame(toggles)
            row.pack(side=tk.TOP, fill=tk.X, anchor=tk.W)
            tk.Label(row, text="  ", bg=color, width=2).pack(side=tk.LEFT, padx=(0, 4), pady=1)
            ttk.Checkbutton(
                row,
                text=name.removeprefix("L_"),
                variable=visible,
                command=toggle,
            ).pack(side=tk.LEFT, anchor=tk.W)

    def plot_work_loops(self) -> None:
        periods = float(self.work_periods.get())
        if periods <= 0:
            self.status.set("Periods must be positive.")
            return
        settle_periods = self.get_settle_periods()
        if settle_periods is None:
            return

        self.work_button.configure(state=tk.DISABLED)
        self.status.set("Running work-loop simulation...")
        self.root.update_idletasks()
        try:
            data = cached_sim(
                work_loop_data,
                Path(self.tg_var.get()),
                self.scene_path,
                periods,
                control_path=Path(self.control_var.get()),
                settle_periods=settle_periods,
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            raise
        finally:
            self.work_button.configure(state=tk.NORMAL)

        for frame in (self.shaft_frame, self.motor_tendon_frame, self.structural_frame, self.wing_frame):
            self.clear_frame(frame)

        for frame in (self.shaft_frame, self.motor_tendon_frame, self.wing_frame):
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            frame.rowconfigure(0, weight=1)

        shaft_loops = data["shaft_loops"]
        self.plot_single_loop(
            self.shaft_frame,
            shaft_loops["Downstroke motor (orange)"],
            "Downstroke motor shaft loop",
            "Motor shaft angle (deg)",
            "Actuator torque (mN m)",
            0,
            0,
        )
        self.plot_single_loop(
            self.shaft_frame,
            shaft_loops["Upstroke motor (blue)"],
            "Upstroke motor shaft loop",
            "Motor shaft angle (deg)",
            "Actuator torque (mN m)",
            0,
            1,
        )

        motor_tendon_loops = data["motor_tendon_loops"]
        self.plot_single_loop(
            self.motor_tendon_frame,
            motor_tendon_loops["Downstroke tendon (orange)"],
            "Downstroke motor tendon loop",
            "Motor tendon extension (mm)",
            "Tendon tension (N)",
            0,
            0,
        )
        self.plot_single_loop(
            self.motor_tendon_frame,
            motor_tendon_loops["Upstroke tendon (blue)"],
            "Upstroke motor tendon loop",
            "Motor tendon extension (mm)",
            "Tendon tension (N)",
            0,
            1,
        )

        self.plot_structural_work_loops(data["structural_loops"])
        wing_loops = data["wing_loops"]
        self.plot_single_loop(
            self.wing_frame,
            wing_loops["Downstroke motor (orange)"],
            "Downstroke motor torque vs wing angle",
            "Wing angle (deg)",
            "Motor torque (mN m)",
            0,
            0,
        )
        self.plot_single_loop(
            self.wing_frame,
            wing_loops["Upstroke motor (blue)"],
            "Upstroke motor torque vs wing angle",
            "Wing angle (deg)",
            "Motor torque (mN m)",
            0,
            1,
        )

        self.status.set(
            f"Plotted work loops for {periods:g} periods at control f0 "
            f"{data['control_frequency_hz']:g} Hz, after {settle_periods:g} settle periods."
        )

class PlotGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DiffSim Aves Plots")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))
        self.root.bind_all("<Command-r>", lambda _event: self.reload_design())
        self.root.bind_all("<Command-R>", lambda _event: self.reload_design())
        self.root.bind_all("<Command-t>", lambda _event: self.add_design_tab())
        self.root.bind_all("<Command-T>", lambda _event: self.add_design_tab())
        self.root.bind_all("<Command-i>", lambda _event: self.inject_impulse())
        self.root.bind_all("<Command-I>", lambda _event: self.inject_impulse())
        self.root.bind_all("<Command-u>", lambda _event: self.update_current_tab_graphs())
        self.root.bind_all("<Command-U>", lambda _event: self.update_current_tab_graphs())

        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New morphology tab", accelerator="Cmd-T", command=self.add_design_tab)
        file_menu.add_command(label="Reload current tab", accelerator="Cmd-R", command=self.reload_design)
        file_menu.add_command(label="Update current tab graphs", accelerator="Cmd-U", command=self.update_current_tab_graphs)
        file_menu.add_command(label="Inject impulse torque", accelerator="Cmd-I", command=self.inject_impulse)
        menu.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 0, 8, 8)).pack(side=tk.TOP, anchor=tk.W)

        self.design_tabs = ttk.Notebook(self.root)
        self.design_tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.panes: dict[str, DesignPane] = {}

        self.add_design_tab(DEFAULT_HUMMER_TG, select=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def active_pane(self) -> DesignPane | None:
        selected = self.design_tabs.select()
        return self.panes.get(selected) if selected else None

    def add_design_tab(self, tg_path: Path | None = None, *, select: bool = True) -> None:
        active = self.active_pane()
        if tg_path is None:
            initial = Path(active.tg_var.get()).parent if active else DESIGNS_DIR
            path = filedialog.askopenfilename(
                title="Select morphology for new tab",
                initialdir=str(initial),
                filetypes=[("Tensegrity morphology", "*.tg")],
            )
            if not path:
                return
            tg_path = Path(path)
        default_tg = tg_path
        frame = ttk.Frame(self.design_tabs)

        def rename(title: str, frame: ttk.Frame = frame) -> None:
            self.design_tabs.tab(frame, text=title)

        try:
            pane = DesignPane(self.root, frame, self.status, default_tg, rename)
        except ValueError as exc:  # wrong bird, or an untagged / malformed .tg
            frame.destroy()
            self.status.set(f"Not loaded: {exc}")
            return
        self.panes[str(frame)] = pane
        self.design_tabs.add(frame, text=design_tab_title(str(default_tg), pane.bird))
        if select:
            self.design_tabs.select(frame)

    def reload_design(self) -> None:
        pane = self.active_pane()
        if pane:
            pane.reload_design()

    def update_phase_plots(self) -> None:
        pane = self.active_pane()
        if pane:
            pane.update_phase_plots()

    def update_current_tab_graphs(self) -> None:
        pane = self.active_pane()
        if pane:
            pane.update_current_tab_graphs()

    def inject_impulse(self) -> None:
        pane = self.active_pane()
        if pane and pane.tabs.select() == str(pane.impulse_tab):
            pane.impulse_inject()

    def close(self) -> None:
        for pane in self.panes.values():
            if pane.viewer_proc and pane.viewer_proc.poll() is None:
                pane.viewer_proc.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlotGui().run()


if __name__ == "__main__":
    main()
