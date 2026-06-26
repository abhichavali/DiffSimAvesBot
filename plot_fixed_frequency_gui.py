from __future__ import annotations

import math
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from compile_tg import compile_tg
from morphology.emit_mjcf import STRUCTURAL_TENDON_COLORS
from morphology.tg_parser import load_morphology
from runtime.aero_runtime import AeroBladeRuntime
from runtime.control_signal import eval_control, load_control_for_morphology
from runtime.crank_sync import sync_crank_disks_at_rest
from runtime.viewer_runtime import get_actuator_id
from runtime.wing_pitch_calibration import calibrate_wing_pitch_at_rest
from runtime.wing_pitch_limit import WingPitchLimiter


ACTUATORS = ("L_motor_t1", "L_motor_t4", "R_motor_t1", "R_motor_t4")
PLOT_ACTUATORS = ("L_motor_t1", "L_motor_t4")
PLOT_CHOICES = (
    "Left actuator torque vs phase",
    "Left motor tendon tension vs phase",
    "Structural tendon tension vs phase",
    "Structural tendon displacement vs phase",
    "Structural elastic energy vs phase",
    "Mean lift per cycle",
    "Mean lift per half-cycle",
)


def data_by_phase(
    tg_path: Path,
    scene_path: Path,
    periods: float,
    plot_kind: str,
) -> tuple[list[float], dict[str, list[float]], str, str, str, dict[str, str]]:
    os.environ["MORPHOLOGY_PATH"] = str(tg_path)
    os.environ["MUJOCO_XML_PATH"] = str(scene_path)
    compile_tg(tg_path, scene_path)

    morph = load_morphology(tg_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    calibrate_wing_pitch_at_rest(model, data)
    sync_crank_disks_at_rest(model, data)

    control = load_control_for_morphology(tg_path)
    period = control.period
    aero = AeroBladeRuntime(model)
    pitch = WingPitchLimiter(model, data)
    actuator_ids = {name: get_actuator_id(model, name) for name in ACTUATORS}
    plot_ids = {name: actuator_ids[name] for name in PLOT_ACTUATORS}
    tendon_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) for name in PLOT_ACTUATORS}
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

    for _ in range(math.ceil(period / model.opt.timestep)):
        step()

    x_values: list[float] = []
    is_structural_plot = plot_kind in (
        "Structural tendon tension vs phase",
        "Structural tendon displacement vs phase",
    )
    names = ["U_tens"] if plot_kind == "Structural elastic energy vs phase" else (
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
            x_values.append(phase)
        if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
            if phase >= periods:
                continue
            lift = sum(float(sum(force[2] for force in (wing.cached_forces or []))) for wing in aero.wings)
            cycle = int(phase) + 2
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
        else:
            for name, aid in plot_ids.items():
                series[name].append(1000.0 * float(data.actuator_force[aid]))
            ylabel = "Actuator torque (mN m)"

    if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
        x_values = sorted(lift_sums)
        series["mean_lift"] = [lift_sums[key] / lift_counts[key] for key in x_values]

    xlabel = "Crank phase (cycles)"
    if plot_kind in ("Mean lift per cycle", "Mean lift per half-cycle"):
        xlabel = "Cycle"
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


class PlotGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DiffSim Aves Plots")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 0, 8, 8)).pack(side=tk.TOP, anchor=tk.W)

        self.grid = ttk.Frame(self.root, padding=8)
        self.grid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.grid.columnconfigure(0, weight=1)
        self.grid.columnconfigure(1, weight=1)
        self.grid.rowconfigure(0, weight=1)
        self.grid.rowconfigure(1, weight=1)

        self.slots: list[ttk.Frame] = []
        self.plotted_slots: set[ttk.Frame] = set()
        self.add_prompt()

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
                ROOT / "design" / "shoulder.tg",
                ROOT / "scene.xml",
                periods,
                plot_choice.get(),
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
        control = load_control_for_morphology(ROOT / "design" / "shoulder.tg")
        self.status.set(
            f"Plotted {periods:g} periods at control f0 {control.fundamental_frequency_hz:g} Hz, "
            "starting at cycle 2 after settle."
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PlotGui().run()


if __name__ == "__main__":
    main()
