from __future__ import annotations

import math
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

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
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")
PLOT_ACTUATORS = ("L_motor_t1", "L_motor_t4")
PLOT_CHOICES = (
    "Left actuator torque vs phase",
    "Left motor tendon tension vs phase",
    "Angular velocity vs time",
    "Structural tendon tension vs phase",
    "Structural tendon displacement vs phase",
    "Structural elastic energy vs phase",
    "Wingtip stroke vs phase (3D)",
    "Mean lift per cycle",
    "Mean lift per half-cycle",
)
SETTLE_PERIODS = 3
FIRST_RECORDED_CYCLE = SETTLE_PERIODS + 1


def _short_path(path: str) -> str:
    """Path relative to the repo root when possible, else the bare name."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def coupled_control_path(tg_path: Path) -> Path:
    """Control file referenced by a morphology, resolved relative to it."""
    morph = load_morphology(tg_path)
    control_path = Path(morph.motors.control_file)
    if not control_path.is_absolute():
        control_path = tg_path.parent / control_path
    return control_path


def data_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    plot_kind: str,
    control_path: Path | None = None,
) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

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

    def step() -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        base_ctrl = morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(SETTLE_PERIODS * period / model.opt.timestep)):
        step()

    x_values: list[float] = []
    is_structural_plot = plot_kind in (
        "Structural tendon tension vs phase",
        "Structural tendon displacement vs phase",
    )
    is_wingtip = plot_kind == "Wingtip stroke vs phase (3D)"
    names = ["x", "y", "z"] if is_wingtip else (
        ["U_tens"] if plot_kind == "Structural elastic energy vs phase" else
        ["mean_lift"] if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle") else
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
        if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
            if phase >= periods:
                continue
            lift = sum(float(sum(force[2] for force in (wing.cached_forces or []))) for wing in aero.wings)
            cycle = int(phase) + FIRST_RECORDED_CYCLE
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
        elif plot_kind == "Angular velocity vs time":
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

    if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
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
    if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
        xlabel = "Cycle"
    elif plot_kind == "Angular velocity vs time":
        xlabel = "Time (s)"
    return x_values, series, plot_kind, xlabel, ylabel, colors


def actuator_torque_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
) -> tuple[list[float], dict[str, list[float]]]:
    phases, series, _title, _xlabel, _ylabel, _colors = data_by_phase(
        tg_path,
        scene_path,
        periods,
        "Left actuator torque vs phase",
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


def work_loop_data(
    tg_path: Path, scene_path: Path, periods: float, control_path: Path | None = None
) -> dict[str, object]:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

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
        base_ctrl = morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(SETTLE_PERIODS * control.period / model.opt.timestep)):
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
    wing_loop = {"x": [], "y": [], "color": "#2ca02c"}

    l_body_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_part_1_freejoint")
    if l_body_jid < 0:
        raise ValueError("Missing joint: L_part_1_freejoint")
    flap_dof = int(model.jnt_dofadr[l_body_jid]) + 5
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
        wing_loop["x"].append(math.degrees(unwrapped_yaw - start_yaw))
        wing_loop["y"].append(1000.0 * float(data.qfrc_passive[flap_dof]))

    return {
        "control_frequency_hz": control.fundamental_frequency_hz,
        "shaft_loops": shaft_loops,
        "motor_tendon_loops": motor_tendon_loops,
        "structural_loops": structural_loops,
        "wing_loop": wing_loop,
    }


WINGTIP_SITE = "L_wing_aero_blade_04"


def wing_path_data(
    tg_path: Path, scene_path: Path, periods: float, control_path: Path | None = None
) -> dict[str, list[float]]:
    """Record the wingtip site's 3D world trajectory (mm) over `periods` cycles."""
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

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
        base_ctrl = morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids.values():
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(SETTLE_PERIODS * control.period / model.opt.timestep)):
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


class PlotGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DiffSim Aves Plots")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))
        self.root.bind("<Command-r>", lambda _event: self.reload_design())
        self.root.bind("<Command-R>", lambda _event: self.reload_design())
        self.root.bind("<Command-u>", lambda _event: self.update_phase_plots())
        self.root.bind("<Command-U>", lambda _event: self.update_phase_plots())
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Reload files", accelerator="Cmd-R", command=self.reload_design)
        file_menu.add_command(label="Update phase plots", accelerator="Cmd-U", command=self.update_phase_plots)
        menu.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu)

        default_tg = ROOT / "design" / "baseline_shoulder.tg"
        self.tg_var = tk.StringVar(value=str(default_tg))
        self.control_var = tk.StringVar(value=str(coupled_control_path(default_tg)))
        self.tg_display = tk.StringVar()
        self.control_display = tk.StringVar()
        self.tg_var.trace_add("write", lambda *_: self.tg_display.set(_short_path(self.tg_var.get())))
        self.control_var.trace_add("write", lambda *_: self.control_display.set(_short_path(self.control_var.get())))
        self.tg_display.set(_short_path(self.tg_var.get()))
        self.control_display.set(_short_path(self.control_var.get()))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 0, 8, 8)).pack(side=tk.TOP, anchor=tk.W)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_tab = ttk.Frame(self.tabs)
        self.work_tab = ttk.Frame(self.tabs)
        self.wing_path_tab = ttk.Frame(self.tabs)
        self.viewer_tab = ttk.Frame(self.tabs)
        self.settings_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.plot_tab, text="Phase plots")
        self.tabs.add(self.work_tab, text="Work loops")
        self.tabs.add(self.wing_path_tab, text="Wing path")
        self.tabs.add(self.viewer_tab, text="Viewer")
        self.tabs.add(self.settings_tab, text="Settings")
        self.viewer_proc: subprocess.Popen | None = None
        self.wing_paths: dict[str, dict] = {}
        self.build_wing_path_tab()
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
        self.add_prompt()
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
        self.status.set(f"Tracing wing path {slot}...")
        self.root.update_idletasks()
        try:
            path = wing_path_data(
                Path(self.tg_var.get()), ROOT / "scene.xml", periods, Path(self.control_var.get())
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

        canvas = FigureCanvasTkAgg(figure, master=self.wing_path_plot)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

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
            str(ROOT / "view_fixed_frequency.py"),
            "--tg", self.tg_var.get(),
            "--control", self.control_var.get(),
            "--scene", str(ROOT / "scene_viewer.xml"),
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
        ttk.Label(
            self.settings_tab,
            text="Picking a morphology loads its coupled control automatically. "
            "Override the control to pair a design with a different signal.",
            wraplength=720,
            justify=tk.LEFT,
        ).pack(side=tk.TOP, anchor=tk.W, padx=12)

    def choose_morphology(self) -> None:
        path = filedialog.askopenfilename(
            title="Select morphology (.tg)",
            initialdir=str(Path(self.tg_var.get()).parent),
            filetypes=[("Tensegrity morphology", "*.tg")],
        )
        if not path:
            return
        self.tg_var.set(path)
        try:
            self.control_var.set(str(coupled_control_path(Path(path))))
            self.status.set(f"Design: {Path(path).name} (coupled control loaded)")
        except Exception as exc:
            self.status.set(f"Loaded morphology, but couldn't resolve coupled control: {exc}")

    def choose_control(self) -> None:
        path = filedialog.askopenfilename(
            title="Select control (.tg)",
            initialdir=str(Path(self.control_var.get()).parent),
            filetypes=[("Control", "*.tg")],
        )
        if path:
            self.control_var.set(path)
            self.status.set(f"Control: {Path(path).name}")

    def reload_design(self) -> None:
        tg_path = Path(self.tg_var.get())
        control_path = Path(self.control_var.get())
        self.status.set("Reloading design files...")
        self.root.update_idletasks()
        try:
            load_control(control_path)
            compile_tg(tg_path, ROOT / "scene.xml")
            if self.viewer_proc and self.viewer_proc.poll() is None:
                self.viewer_proc.terminate()
                self.viewer_proc = None
                self.status.set(f"Reloaded {_short_path(str(tg_path))}; closed viewer.")
            else:
                self.status.set(f"Reloaded {_short_path(str(tg_path))}")
        except Exception as exc:
            self.status.set(f"Reload failed: {exc}")
            raise

    def add_prompt(self) -> None:
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
        plot_choice.current(0)
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
        plotted = [frame for frame in self.slots if frame in self.plotted_slots]
        if not plotted:
            return
        for frame in plotted:
            plot_choice, periods, button = self.slot_controls[frame]
            self.plot_in_slot(frame, plot_choice, periods, button)

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

        button.configure(state=tk.DISABLED)
        self.status.set("Running simulation...")
        self.root.update_idletasks()
        try:
            x_values, series, title, xlabel, ylabel, colors = data_by_phase(
                Path(self.tg_var.get()),
                ROOT / "scene.xml",
                periods,
                plot_choice.get(),
                Path(self.control_var.get()),
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
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title == "Angular velocity vs time":
            vmax = max((max(values) for values in series.values() if values), default=0.0)
            ax.set_ylim(0.0, max(35.0, math.ceil(vmax / 5.0) * 5.0))
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
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

        canvas = FigureCanvasTkAgg(figure, master=plot_area)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.draw()

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
            f"after {SETTLE_PERIODS} settle periods."
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
        self.work_sets.add(self.wing_frame, text="Wing output")

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

        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

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

        canvas = FigureCanvasTkAgg(figure, master=plot_area)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.draw()

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

        self.work_button.configure(state=tk.DISABLED)
        self.status.set("Running work-loop simulation...")
        self.root.update_idletasks()
        try:
            data = work_loop_data(
                Path(self.tg_var.get()), ROOT / "scene.xml", periods, Path(self.control_var.get())
            )
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            raise
        finally:
            self.work_button.configure(state=tk.NORMAL)

        for frame in (self.shaft_frame, self.motor_tendon_frame, self.structural_frame, self.wing_frame):
            self.clear_frame(frame)

        for frame in (self.shaft_frame, self.motor_tendon_frame):
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
        self.plot_single_loop(
            self.wing_frame,
            data["wing_loop"],
            "Wing output loop",
            "Wing stroke angle theta_w (deg)",
            "Tensegrity generalized torque (mN m)",
            0,
            0,
        )
        self.wing_frame.columnconfigure(0, weight=1)
        self.wing_frame.rowconfigure(0, weight=1)

        self.status.set(
            f"Plotted work loops for {periods:g} periods at control f0 "
            f"{data['control_frequency_hz']:g} Hz, after {SETTLE_PERIODS} settle periods."
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlotGui().run()


if __name__ == "__main__":
    main()
