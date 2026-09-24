"""Offline CPU rollouts behind the design-inspector GUI's plots, and its two caches.

``data_by_phase``, ``work_loop_data`` and ``wing_path_data`` each compile a design,
settle it on the ramped crank drive and record one kind of trace; ``cached_sim``
pickles their results under .gui_plot_cache/ keyed on the design files' mtimes, and
``prepare_scene`` skips recompiling an unchanged .tg. ``plot_fixed_frequency_gui.py``
is the Tk front end that draws them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import mujoco
import numpy as np

from compile_tg import compile_tg
from morphology.emit_mjcf import STRUCTURAL_TENDON_COLORS
from morphology.model import PLATE_BIRDS, require_bird
from morphology.tg_parser import load_morphology
from runtime.aero import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control, load_control_for_morphology, ramp_factor
from runtime.crank import sync_crank_disks_at_rest
from runtime.disk_orientation import DiskOrientation
from runtime.morphology_runtime import load_design_morphology
from runtime.realtime import get_actuator_id
from runtime.wing_pitch import WingPitchLimiter, calibrate_wing_pitch_at_rest

ROOT = Path(__file__).resolve().parents[2]

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
    require_bird(morph, *PLATE_BIRDS, tool="the plotting GUI")
    control_path = Path(morph.motors.control_file)
    if not control_path.is_absolute():
        control_path = tg_path.parent / control_path
    return control_path


def prepare_scene(tg_path: Path, scene_path: Path, *, color_structural_tendons: bool = False) -> Path:
    """Point the runtime at this design (env + cache clear) and compile only when the .tg changed."""
    tg_path = tg_path.resolve()
    scene_path = scene_path.resolve()
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    load_design_morphology.cache_clear()
    stamp_file = scene_path.with_suffix(scene_path.suffix + (".color.stamp" if color_structural_tendons else ".stamp"))
    stamp_text = f"{tg_path}\n{tg_path.stat().st_mtime_ns}\n{int(color_structural_tendons)}\n"
    if not scene_path.exists() or not stamp_file.exists() or stamp_file.read_text() != stamp_text:
        compile_tg(tg_path, scene_path, color_structural_tendons=color_structural_tendons)
        stamp_file.write_text(stamp_text)
    return scene_path


class StrokePlaneSignal:
    """The learned disk-mount (tilt, pitch) over crank phase that optimize_control
    --tune-strokeplane saves beside a design as strokeplane_optimized.json."""

    def __init__(self, model: mujoco.MjModel, spec: dict) -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec["phase_ref_joint"])
        self.orient = DiskOrientation(model)
        self.qadr = int(model.jnt_qposadr[jid])
        self.coef = np.asarray(spec["tilt_coeffs"] + spec["pitch_coeffs"], dtype=float)
        self.tilt0 = float(spec["tilt0_rad"])
        self.pitch0 = float(spec["pitch0_rad"])
        self.max_delta = float(spec["max_delta_rad"])

    def angles(self, data: mujoco.MjData) -> tuple[float, float]:
        th = float(data.qpos[self.qadr])
        basis = np.array([1.0, math.cos(th), math.sin(th), math.cos(2.0 * th), math.sin(2.0 * th)])
        mx = self.max_delta
        return (self.tilt0 + mx * math.tanh(float(self.coef[:5] @ basis) / mx),
                self.pitch0 + mx * math.tanh(float(self.coef[5:] @ basis) / mx))


def settled_drive(tg_path: Path, scene_path: Path, control_path: Path | None, settle_periods: float,
                  *, strokeplane: bool = False):
    """Compile, rest and settle a design on the ramped crank drive: the CPU loop behind every plot.

    Returns (model, data, control, aero, step, sp); step() is one physics step of the
    drive. With ``strokeplane`` a StrokePlaneSignal beside the .tg (sp, else None) is
    applied every step -- only the phase plots ask for it; work loops and the wing
    path run the plain drive.
    """
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
    actuator_ids = [get_actuator_id(model, name) for name in ACTUATORS]
    sp_file = tg_path.parent / "strokeplane_optimized.json"
    sp = StrokePlaneSignal(model, json.loads(sp_file.read_text())) if strokeplane and sp_file.exists() else None

    def step() -> None:
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        aero.apply(model, data)
        pitch.apply(model, data)
        ramp = ramp_factor(data.time, drive_start, settle_periods * control.period)
        base_ctrl = ramp * morph.motors.motor_direction * eval_control(control, data.time)
        for aid in actuator_ids:
            if aid >= 0:
                data.ctrl[aid] = base_ctrl
        if sp is not None:
            sp.orient.set(*sp.angles(data))
        mujoco.mj_step(model, data)

    for _ in range(math.ceil(settle_periods * control.period / model.opt.timestep)):
        step()
    return model, data, control, aero, step, sp


def data_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    plot_kind: str,
    control_path: Path | None = None,
    settle_periods: float = SETTLE_PERIODS,
) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    is_mean_lift_plot = plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle")
    model, data, control, aero, step, sp_signal = settled_drive(
        tg_path, scene_path, control_path, settle_periods, strokeplane=True)
    period = control.period
    plot_ids = {name: get_actuator_id(model, name) for name in PLOT_ACTUATORS}
    tendon_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) for name in PLOT_ACTUATORS}
    qvel_ids = {}
    for name in PLOT_ACTUATORS:
        prefix, station = name.split("_motor_")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{station}_disk_hinge")
        qvel_ids[name] = int(model.jnt_dofadr[jid])
    structural_tendon_ids, structural_tendon_names = structural_tendons(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, WINGTIP_SITE)

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
                series[name].append(tendon_tension(model, data, tid, float(model.tendon_lengthspring[tid][0])))
            ylabel = "Motor tendon tension (N)"
        elif plot_kind in ("Angular velocity vs phase", "Angular velocity vs time"):
            for name, dof in qvel_ids.items():
                series[name].append(abs(float(data.qvel[dof])))
            ylabel = "|Angular velocity| (rad/s)"
        elif plot_kind == "Structural tendon tension vs phase":
            for name, tid in zip(structural_tendon_names, structural_tendon_ids):
                series[name].append(tendon_tension(model, data, tid, float(model.tendon_range[tid][1])))
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
                tilt, pitv = sp_signal.angles(data)
                dev_t, dev_p = tilt - sp_signal.tilt0, pitv - sp_signal.pitch0
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
    model, data, control, _aero, step, _sp = settled_drive(tg_path, scene_path, control_path, settle_periods)
    actuator_ids = {name: get_actuator_id(model, name) for name in PLOT_ACTUATORS}
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
    model, data, control, _aero, step, _sp = settled_drive(tg_path, scene_path, control_path, settle_periods)
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
