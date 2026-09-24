"""Tk front-end for a codesign run: GA genes -> loss -> diff-sim params -> launch.

Four tabs, in the order you fill them in:

  1. GA genes   -- the scalars the EA may search, grouped by .tg section (control,
                   struts, arrangement, placement, motors, tendons). Two kinds: genes
                   that ride the existing genome as log-factors on MJX model arrays,
                   and geometry genes that are compiled into the model, for which the
                   EA writes a .tg variant and recompiles per genome.
  2. Loss       -- hover (the existing lift-floor/peak-torque/power objective) or
                   forward flight (no objective implemented yet, so it is disabled).
  3. Diff-sim   -- which theta blocks Adam may move in opt/optimize_control.py, and
                   how many of the GA's ranked seeds get a differentiable run (top N).
  4. Run        -- the exact argv for each stage, launch/stop, live log, results.

Run it with:  uv run python opt/run_gui.py
"""
from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk

from tg_scalars import Param, parse_tg

ROOT = Path(__file__).resolve().parent.parent

# Sections shown in tab 1, in display order. Everything else in the .tg (world,
# wing, aero) is deliberately hidden -- 66 rows of which most are not knobs you
# would ever hand an optimizer is noise, not choice.
GROUPS: tuple[tuple[str, str], ...] = (
    ("Control -- Fourier coefficients", "@control"),
    ("Struts", "strut."),
    ("Strut arrangement", "arrange."),
    ("Placement", "placement."),
    ("Motors", "motors."),
    ("Tendons", "tendons."),
)


def in_group(p: Param, prefix: str) -> bool:
    if prefix == "@control":
        return p.source == "control" and bool(_CONTROL_GENE.match(p.name))
    return p.name.startswith(prefix)


def grouped(params: list[Param]) -> list[tuple[str, list[Param]]]:
    """Params to show, bucketed by section. Only bare scalars: a list or an
    enum (`shape = arc`, `stations = [...]`, `ctrl: [-120, 120]`) cannot carry
    a continuous gene, so listing it would only be a checkbox that lies."""
    out = []
    for title, prefix in GROUPS:
        rows = [p for p in params if p.tunable and in_group(p, prefix)]
        if rows:
            out.append((title, rows))
    return out


# --- how a selected param becomes a gene ------------------------------------
#
# Two mechanisms. The four STATIC_BLOCKS ride the existing genome: they are
# log-factors applied to MJX model arrays inside the rollout, so they cost
# nothing extra. Everything else is a GEOMETRY gene -- baked into the MJCF at
# compile time, so the EA has to write a .tg variant and recompile per genome
# (cheap: ~0.2 s, and array shapes are unchanged, so the batched rollout still
# vmaps -- see evolve_codesign --geom-genes).

STATIC_BLOCKS: dict[str, tuple[str, ...]] = {
    "tendon_stiffness": ("tendons.defaults.stiffness",),
    "tendon_pretension": ("tendons.defaults.pretension",),
    "motor": ("motors.tendon_stiffness", "motors.tendon_rest_length"),
}
_CONTROL_GENE = re.compile(r"^(a0|[ab]_\d+)$")


def gene_blocks(params: list[Param]) -> dict[str, list[str]]:
    """Map EA genome block -> the param names it covers, for these .tg files."""
    blocks = {k: [n for n in v if any(p.name == n for p in params)]
              for k, v in STATIC_BLOCKS.items()}
    blocks["control"] = [p.name for p in params if _CONTROL_GENE.match(p.name)]
    return {k: v for k, v in blocks.items() if v}


def block_of(name: str, blocks: dict[str, list[str]]) -> str | None:
    return next((b for b, names in blocks.items() if name in names), None)


# --- run configuration ------------------------------------------------------

@dataclass
class RunConfig:
    tg: Path
    scene: Path
    out: Path
    genes: list[str] = field(default_factory=list)       # EA blocks (model-array genes)
    geom_genes: list[str] = field(default_factory=list)  # .tg scalars (recompiled per genome)
    loss: str = "hover"
    # loss knobs (hover)
    lift_floor: float = 0.300
    tau_min: float = 0.100
    w_power: float = 5.0
    w_overlap: float = 5.0
    # GA hyperparams
    pop: int = 96
    gens: int = 40
    sigma: float = 0.06
    timesteps: list[float] = field(default_factory=lambda: [0.06, 0.03])
    # differentiable-sim blocks
    tune_stiffness: bool = True
    tune_pretension: bool = True
    tune_fourier: bool = True
    harmonics: int = 5
    iters_per_stage: int = 40
    lr: float = 2e-3
    top_n: int = 1          # how many of the GA's ranked seeds get a diff-sim run
    # stages
    run_ga: bool = True
    run_adam: bool = True

    def to_json(self) -> str:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        return json.dumps(d, indent=2)


def ga_seed_path(out: Path, rank: int = 1) -> Path:
    """Genome handed from the EA to Adam for the rank-th seed. The EA materializes
    seed_1..seed_N only once it finishes, so rank 1 falls back to the
    every-generation checkpoint (which is all an interrupted run leaves)."""
    seed = out / "ga" / f"seed_{rank}" / "codesign_seed.json"
    if seed.exists() or rank > 1:
        return seed
    return out / "ga" / "best_genome.json"


def build_commands(cfg: RunConfig) -> list[tuple[str, list[str]]]:
    """(label, argv) per enabled stage. Raises ValueError on a config that would
    only fail once the subprocess is already up."""
    if not (cfg.run_ga or cfg.run_adam):
        raise ValueError("no stage selected -- tick 'GA' and/or 'differentiable sim'")
    py = [sys.executable, "-u"]
    stages: list[tuple[str, list[str]]] = []

    if cfg.top_n < 1:
        raise ValueError("top-N must be >= 1 (tab 3)")

    if cfg.run_ga:
        if not (cfg.genes or cfg.geom_genes):
            raise ValueError("GA stage: no tunable genes selected (tab 1)")
        stages.append(("GA", py + [
            str(ROOT / "opt" / "evolve_codesign.py"),
            "--tg", str(cfg.tg), "--scene", str(cfg.scene),
            "--out", str(cfg.out / "ga"),
            "--genes", ",".join(cfg.genes),
            *(["--geom-genes", ",".join(cfg.geom_genes)] if cfg.geom_genes else []),
            "--pop", str(cfg.pop), "--gens", str(cfg.gens), "--sigma", str(cfg.sigma),
            # the EA only materializes the seeds it is asked for, and top-N is
            # exactly how many of them the diff-sim stage will consume
            "--n-seeds", str(cfg.top_n),
            "--timesteps", *[str(t) for t in cfg.timesteps],
            "--harmonics", str(cfg.harmonics),  # the genome layout Adam will decode
        ]))

    if cfg.run_adam:
        if not (cfg.tune_stiffness or cfg.tune_pretension or cfg.tune_fourier):
            raise ValueError("diff-sim stage: nothing to optimize (tab 3)")
        # Cold start = one run; chained = one run per GA seed, ranked best-first.
        for rank in range(1, cfg.top_n + 1) if cfg.run_ga else (None,):
            sub = f"seed_{rank}" if rank else "cold"
            seed = ga_seed_path(cfg.out, rank) if rank else None
            # A materialized seed dir is the evolved design (geometry, a0, clock, harmonics),
            # and its codesign_seed.json is relative to it; best_genome.json to the base.
            tg = seed.parent / "shoulder.tg" if seed and seed.name == "codesign_seed.json" else cfg.tg
            argv = py + [
                str(ROOT / "opt" / "optimize_control.py"),
                "--tg", str(tg), "--scene", str(cfg.scene),
                "--out", str(cfg.out / "adam" / sub),
                "--harmonics", str(cfg.harmonics),
                "--iters-per-stage", str(cfg.iters_per_stage), "--lr", str(cfg.lr),
                "--lift-floor", str(cfg.lift_floor), "--tau-min", str(cfg.tau_min),
                "--w-power", str(cfg.w_power), "--w-overlap", str(cfg.w_overlap),
            ]
            if cfg.tune_stiffness or cfg.tune_pretension:
                argv += ["--tune-tendons"]
                if not cfg.tune_pretension:
                    argv += ["--tendon-params", "stiffness"]
                elif not cfg.tune_stiffness:
                    argv += ["--tendon-params", "pretension"]
            if "motor" in cfg.genes:
                argv += ["--tune-motor-tendons"]
            if not cfg.tune_fourier:
                argv += ["--freeze-control"]
            if seed:  # warm-start from that GA seed's genome
                argv += ["--init-codesign", str(seed)]
            stages.append((f"diff-sim {sub}", argv))

    return stages


def collect_results(out: Path) -> str:
    """Human-readable summary of whatever the run left on disk."""
    lines: list[str] = []
    best = out / "ga" / "best_genome.json"
    if best.exists():
        d = json.loads(best.read_text())
        lines.append(f"GA  best_genome.json: gen {d.get('gen')}  fitness {d.get('fitness'):.4f}")
    for dt in sorted((out / "ga").glob("seed_*/dt_transfer.json")):
        d = json.loads(dt.read_text())
        lines.append(f"GA  {dt.parent.name}: fit {d['fitness']:.3f}  "
                     f"tau/dt {d['tau_by_dt_mNm']}  spread {d['tau_spread_frac']}")
    for log in sorted((out / "adam").glob("**/log.json")):
        entries = json.loads(log.read_text())
        if entries:
            e = entries[-1]
            lines.append(f"ADAM {log.parent.relative_to(out)}: it{e.get('iter')} "
                         f"tau {e.get('tau_hat', 0) * 1e3:.1f} mNm  "
                         f"L {e.get('Lbar', 0) * 1e3:.1f} mN  loss {e.get('loss', 0):.4f}")
    for d in sorted(out.glob("**/")):
        if (d / "shoulder.tg").exists():
            lines.append(f"design  {d.relative_to(out)}/")
    return "\n".join(lines) if lines else f"nothing found under {out}"


# --- GUI --------------------------------------------------------------------

def _scrollable(master: ttk.Frame) -> ttk.Frame:
    """Vertically scrollable inner frame. Tk has no built-in, so: canvas + frame."""
    canvas = tk.Canvas(master, highlightthickness=0)
    bar = ttk.Scrollbar(master, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    # bind_all, but only while the pointer is over this canvas -- an unscoped
    # global binding steals the wheel from the log Text on the Run tab.
    wheel = lambda e: canvas.yview_scroll(-e.delta, "units")
    inner.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
    inner.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    return inner


class OptGui(ttk.Frame):
    def __init__(self, master: tk.Tk, tg: Path, control: Path, scene: Path, out: Path):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)
        self.tg, self.control, self.scene = tg, control, scene
        self.params = parse_tg(tg) + parse_tg(control)
        self.blocks = gene_blocks(self.params)
        self.vars: dict[str, tk.BooleanVar] = {}
        self.proc: subprocess.Popen | None = None
        self.running = False  # proc is None between stages; this covers the whole chain
        self.q: queue.Queue = queue.Queue()

        self.out = tk.StringVar(value=str(out))
        self.loss = tk.StringVar(value="hover")
        self.cfg_vars = {  # name -> (tk var, cast)
            "pop": (tk.StringVar(value="96"), int),
            "gens": (tk.StringVar(value="40"), int),
            "sigma": (tk.StringVar(value="0.06"), float),
            "timesteps": (tk.StringVar(value="0.06 0.03"), None),
            "lift_floor": (tk.StringVar(value="0.300"), float),
            "tau_min": (tk.StringVar(value="0.100"), float),
            "w_power": (tk.StringVar(value="5.0"), float),
            "w_overlap": (tk.StringVar(value="5.0"), float),
            "harmonics": (tk.StringVar(value="5"), int),
            "iters_per_stage": (tk.StringVar(value="40"), int),
            "lr": (tk.StringVar(value="2e-3"), float),
            "top_n": (tk.StringVar(value="1"), int),
        }
        self.tune = {k: tk.BooleanVar(value=True)
                     for k in ("tune_stiffness", "tune_pretension", "tune_fourier")}
        self.stages = {k: tk.BooleanVar(value=True) for k in ("run_ga", "run_adam")}

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        for title, build in (("1. GA genes", self._tab_genes), ("2. Loss", self._tab_loss),
                             ("3. Diff-sim params", self._tab_diffsim), ("4. Run", self._tab_run)):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text=title)
            build(f)
        self.after(120, self._drain)

    # -- tab 1
    def _tab_genes(self, f: ttk.Frame) -> None:
        groups = grouped(self.params)
        every = {p.name for _, rows in groups for p in rows}
        ttk.Label(f, text=f"Scalars from {self.tg.name} + {self.control.name} the EA may search. "
                          f"Rows marked 'geometry' are compiled into the model, so selecting them "
                          f"makes the EA write a .tg variant and recompile per genome -- correct, "
                          f"but slower per generation than the model-array genes.",
                  wraplength=780).pack(anchor="w", pady=(0, 6))
        row = ttk.Frame(f)
        row.pack(anchor="w", pady=(0, 6))
        ttk.Button(row, text="All", command=lambda: self._set_all(every, True)).pack(side="left")
        ttk.Button(row, text="None", command=lambda: self._set_all(every, False)).pack(side="left", padx=4)
        ttk.Button(row, text="Model-array genes only",
                   command=self._set_cheap).pack(side="left", padx=4)

        inner = _scrollable(f)
        for gi, (title, rows) in enumerate(groups):
            box = ttk.LabelFrame(inner, text=title, padding=(8, 4))
            box.pack(fill="x", expand=True, anchor="w", pady=(0 if gi == 0 else 10, 0), padx=2)
            for p in rows:
                blk = block_of(p.name, self.blocks)
                var = tk.BooleanVar(value=blk is not None)  # cheap genes on by default
                self.vars[p.name] = var
                line = ttk.Frame(box)
                line.pack(fill="x", anchor="w")
                ttk.Checkbutton(line, text=p.name, variable=var, width=42).pack(side="left")
                ttk.Label(line, text=f"= {p.value}", width=20).pack(side="left")
                ttk.Label(line, text=f"gene: {blk}" if blk else "geometry (recompiles)",
                          foreground="#444" if blk else "#8a5a00").pack(side="left")

    def _set_all(self, names: set[str], on: bool) -> None:
        for n in names:
            self.vars[n].set(on)

    def _set_cheap(self) -> None:
        """Back to the four model-array blocks: no recompile, fastest generation."""
        wired = {n for names in self.blocks.values() for n in names}
        for n, v in self.vars.items():
            v.set(n in wired)

    # -- tab 2
    def _tab_loss(self, f: ttk.Frame) -> None:
        ttk.Radiobutton(f, text="Hover", value="hover", variable=self.loss).pack(anchor="w")
        ttk.Radiobutton(f, text="Forward flight", value="forward", variable=self.loss,
                        state="disabled").pack(anchor="w", pady=(2, 12))
        box = ttk.LabelFrame(f, text="Hover weights", padding=8)
        box.pack(anchor="w", fill="x")
        for i, (key, label) in enumerate((("lift_floor", "--lift-floor [N]"),
                                          ("tau_min", "--tau-min [Nm]"),
                                          ("w_power", "--w-power"),
                                          ("w_overlap", "--w-overlap"))):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(box, textvariable=self.cfg_vars[key][0], width=12).grid(row=i, column=1, padx=6)

    # -- tab 3
    def _tab_diffsim(self, f: ttk.Frame) -> None:
        ttk.Label(f, text="Which theta blocks Adam may move in opt/optimize_control.py.",
                  wraplength=760).pack(anchor="w", pady=(0, 8))
        for key, label in (("tune_stiffness", "Tendon stiffness  (per-tendon log-factors)"),
                           ("tune_pretension", "Tendon pretension  (per-tendon rest-length log-factors)"),
                           ("tune_fourier", "Fourier coefficients for control  (a0 + 2K harmonics)")):
            ttk.Checkbutton(f, text=label, variable=self.tune[key]).pack(anchor="w")
        box = ttk.LabelFrame(f, text="Optimizer", padding=8)
        box.pack(anchor="w", fill="x", pady=10)
        for i, (key, label) in enumerate((("harmonics", "--harmonics K"),
                                          ("iters_per_stage", "--iters-per-stage"),
                                          ("lr", "--lr"))):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(box, textvariable=self.cfg_vars[key][0], width=12).grid(row=i, column=1, padx=6)

        seeds = ttk.LabelFrame(f, text="GA -> diff-sim handoff", padding=8)
        seeds.pack(anchor="w", fill="x")
        ttk.Label(seeds, text="Top N GA seeds").grid(row=0, column=0, sticky="w")
        ttk.Entry(seeds, textvariable=self.cfg_vars["top_n"][0], width=12).grid(row=0, column=1, padx=6)
        ttk.Label(seeds, text="one differentiable run per seed, ranked best-first, into "
                              "<out>/adam/seed_N/. Also sets the GA's --n-seeds.",
                  foreground="#444", wraplength=560).grid(row=1, column=0, columnspan=2, sticky="w")

    # -- tab 4
    def _tab_run(self, f: ttk.Frame) -> None:
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text="Output dir").pack(side="left")
        ttk.Entry(top, textvariable=self.out, width=52).pack(side="left", padx=6)
        ttk.Button(top, text="Browse", command=self._browse).pack(side="left")

        ga = ttk.LabelFrame(f, text="Stages", padding=8)
        ga.pack(fill="x", pady=8)
        ttk.Checkbutton(ga, text="GA (evolve_codesign)", variable=self.stages["run_ga"]
                        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(ga, text="Differentiable sim (optimize_control), warm-started from the GA winner",
                        variable=self.stages["run_adam"]).grid(row=1, column=0, sticky="w")
        for i, (key, label) in enumerate((("pop", "--pop"), ("gens", "--gens"),
                                          ("sigma", "--sigma"), ("timesteps", "--timesteps [ms]"))):
            ttk.Label(ga, text=label).grid(row=i, column=1, sticky="e", padx=(20, 4))
            ttk.Entry(ga, textvariable=self.cfg_vars[key][0], width=12).grid(row=i, column=2)

        bar = ttk.Frame(f)
        bar.pack(fill="x")
        for text, cmd in (("Preview command", self._preview), ("Save config", self._save),
                          ("Launch", self._launch), ("Stop", self._stop),
                          ("Collect results", self._collect)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=(0, 6))

        self.log = tk.Text(f, height=24, wrap="none", font=("Menlo", 10))
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        sb = ttk.Scrollbar(f, orient="vertical", command=self.log.yview)
        sb.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
        self.log.configure(yscrollcommand=sb.set)

    # -- config / actions
    def run_config(self) -> RunConfig:
        get = lambda k: self.cfg_vars[k][0].get()
        cast = lambda k: self.cfg_vars[k][1](get(k))
        wired = {n for names in self.blocks.values() for n in names}
        picked = {n for n, v in self.vars.items() if v.get()}
        genes = sorted({b for b in self.blocks
                        if any(n in picked for n in self.blocks[b])})
        return RunConfig(
            tg=self.tg, scene=self.scene, out=Path(self.out.get()),
            genes=genes, geom_genes=sorted(picked - wired), loss=self.loss.get(),
            lift_floor=cast("lift_floor"), tau_min=cast("tau_min"),
            w_power=cast("w_power"), w_overlap=cast("w_overlap"),
            pop=cast("pop"), gens=cast("gens"), sigma=cast("sigma"),
            timesteps=[float(x) for x in get("timesteps").split()],
            tune_stiffness=self.tune["tune_stiffness"].get(),
            tune_pretension=self.tune["tune_pretension"].get(),
            tune_fourier=self.tune["tune_fourier"].get(),
            harmonics=cast("harmonics"), iters_per_stage=cast("iters_per_stage"), lr=cast("lr"),
            top_n=cast("top_n"),
            run_ga=self.stages["run_ga"].get(), run_adam=self.stages["run_adam"].get(),
        )

    def _browse(self) -> None:
        if d := filedialog.askdirectory(initialdir=self.out.get()):
            self.out.set(d)

    def _write(self, text: str) -> None:
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 5000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")

    def _preview(self) -> None:
        try:
            cfg = self.run_config()
            self._write("")
            for label, argv in build_commands(cfg):
                self._write(f"[{label}] " + " ".join(argv))
            if cfg.geom_genes:
                self._write(f"[note] {len(cfg.geom_genes)} geometry gene(s) -> the GA "
                            f"recompiles a .tg variant per genome")
        except Exception as e:  # bad entry casts land here too
            self._write(f"[error] {e}")

    def _save(self) -> None:
        try:
            cfg = self.run_config()
            cfg.out.mkdir(parents=True, exist_ok=True)
            path = cfg.out / "run_config.json"
            path.write_text(cfg.to_json())
            self._write(f"[saved] {path}")
        except Exception as e:
            self._write(f"[error] {e}")

    def _launch(self) -> None:
        if self.running:
            self._write("[error] a run is already going -- Stop it first")
            return
        try:
            cfg = self.run_config()
            stages = build_commands(cfg)
        except Exception as e:
            self._write(f"[error] {e}")
            return
        self._save()
        self.running = True
        self.stopping = False
        threading.Thread(target=self._worker, args=(cfg, stages), daemon=True).start()

    def _worker(self, cfg: RunConfig, stages: list[tuple[str, list[str]]]) -> None:
        try:
            self._run_stages(cfg, stages)
        finally:
            self.running = False
            self.q.put(("results", cfg.out))

    def _run_stages(self, cfg: RunConfig, stages: list[tuple[str, list[str]]]) -> None:
        for label, argv in stages:
            if self.stopping:
                self.q.put(f"[stop] {label} and the stages after it skipped\n")
                break
            if label.startswith("diff-sim") and cfg.run_ga:
                # The GA writes seed_*/ only at the very end, so the argv built before
                # launch points at best_genome.json. Re-derive now that the GA has run.
                argv = dict(build_commands(cfg))[label]
                if not Path(argv[argv.index("--init-codesign") + 1]).exists():
                    self.q.put(f"[skip] {label}: the GA produced no such seed\n")
                    continue
            self.q.put(f"\n=== {label}: {' '.join(argv)}\n")
            try:
                self.proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1)
            except OSError as e:
                self.q.put(f"[error] cannot start {label}: {e}\n")
                break
            for line in self.proc.stdout:
                self.q.put(line)
            rc = self.proc.wait()
            self.proc = None
            self.q.put(f"=== {label} exited rc={rc}\n")
            if rc != 0 and not self.stopping:
                if not label.startswith("diff-sim"):
                    self.q.put("[stop] GA failed -- diff-sim stages skipped\n")
                    break
                # One seed diverging says nothing about the others; keep going.
                self.q.put(f"[warn] {label} failed -- continuing with the next seed\n")

    def _stop(self) -> None:
        self.stopping = True  # _run_stages starts no further stage
        if self.proc is None:
            self._write("[stop] nothing running")
            return
        self.proc.terminate()
        self._write("[stop] SIGTERM sent")

    def _collect(self) -> None:
        self._write("\n" + collect_results(Path(self.out.get())))

    def _drain(self) -> None:
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                self._write("\n" + collect_results(item[1]))
            else:
                self._write(item.rstrip("\n"))
        self.after(120, self._drain)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tg", type=Path, default=ROOT / "designs" / "hummer" / "hover-base" / "shoulder.tg")
    p.add_argument("--control", type=Path, default=None,
                   help="control .tg (default: the shoulder's sibling control.tg)")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--out", type=Path, default=ROOT / "opt" / "gui_runs")
    a = p.parse_args()
    control = a.control or a.tg.with_name("control.tg")
    root = tk.Tk()
    root.title(f"codesign run -- {a.tg.parent.name}/{a.tg.name}")
    root.geometry("900x760")
    OptGui(root, a.tg, control, a.scene, a.out)
    root.mainloop()


if __name__ == "__main__":
    main()
