"""Design inspector: a Tk window with one tab per design and six sub-tabs.

    uv run python plot_fixed_frequency_gui.py

Phase plots, Work loops, Wing path, Impulse response, Viewer and Settings. Cmd-T opens
a design tab, Cmd-R reloads it, Cmd-U updates the current sub-tab's graphs and Cmd-I
injects an impulse; Esc leaves fullscreen. This file is UI only: the rollouts and
caches live in src/gui/sims.py, the impulse sim in src/gui/impulse.py, and the Viewer
tab launches src/view_fixed_frequency.py under mjpython.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from gui.impulse import REF_MOTOR, ImpulseSim, cut_wraps, wrap_deg
from gui.sims import (
    ACTUATORS,
    PLOT_CHOICES,
    SETTLE_PERIODS,
    cached_sim,
    coupled_control_path,
    data_by_phase,
    prepare_scene,
    structural_tendon_stiffness_rows,
    wing_path_correlation,
    wing_path_data,
    work_loop_data,
)
from morphology.tg_parser import load_morphology
from runtime.control_signal import load_control
from runtime.morphology_runtime import DEFAULT_HUMMER_TG, DESIGNS_DIR


DEFAULT_PHASE_PLOTS = (
    "Left actuator torque vs phase",
    "Mean lift per cycle",
    "Left motor tendon tension vs phase",
    "Angular velocity vs phase",
)
MOTOR_COLORS = {"L_motor_t1": "#1f77b4", "L_motor_t4": "#ff7f0e", "R_motor_t1": "#1f77b4", "R_motor_t4": "#ff7f0e"}
SIDE_COLORS = {"L": "#1f77b4", "R": "#d62728"}


def _short_path(path: str) -> str:
    """Path relative to the repo root when possible, else the bare name."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


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


def add_line_toggles(ax, lines: dict, colors: dict[str, str], plot_area: ttk.Frame, canvas: FigureCanvasTkAgg) -> None:
    """A color-swatch checkbox per (initially hidden) line; the legend lists the visible ones."""

    def update_legend() -> None:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        visible_lines = [line for line in lines.values() if line.get_visible()]
        if visible_lines:
            ax.legend(visible_lines, [line.get_label() for line in visible_lines])

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
        ttk.Checkbutton(row, text=name.removeprefix("L_"), variable=visible, command=toggle).pack(
            side=tk.LEFT, anchor=tk.W
        )


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
        if lines and not colors:
            ax.legend()

        canvas = embed_figure(figure, plot_area, side=tk.LEFT, fill=tk.BOTH, expand=True)
        if colors:
            add_line_toggles(ax, lines, colors, plot_area, canvas)

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

        canvas = embed_figure(figure, plot_area, side=tk.LEFT, fill=tk.BOTH, expand=True)
        colors = {name: str(loop["color"]) for name, loop in loops.items()}
        add_line_toggles(ax, lines, colors, plot_area, canvas)

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

        for frame, loops, titles, xlabel, ylabel in (
            (self.shaft_frame, data["shaft_loops"], ("Downstroke motor shaft loop", "Upstroke motor shaft loop"),
             "Motor shaft angle (deg)", "Actuator torque (mN m)"),
            (self.motor_tendon_frame, data["motor_tendon_loops"],
             ("Downstroke motor tendon loop", "Upstroke motor tendon loop"),
             "Motor tendon extension (mm)", "Tendon tension (N)"),
            (self.wing_frame, data["wing_loops"],
             ("Downstroke motor torque vs wing angle", "Upstroke motor torque vs wing angle"),
             "Wing angle (deg)", "Motor torque (mN m)"),
        ):  # each loops dict is (downstroke, upstroke) in that order
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            frame.rowconfigure(0, weight=1)
            for column, (loop, title) in enumerate(zip(loops.values(), titles)):
                self.plot_single_loop(frame, loop, title, xlabel, ylabel, 0, column)
        self.plot_structural_work_loops(data["structural_loops"])

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
