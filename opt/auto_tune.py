"""Autonomous reward-function tuner for CPU-transferable flapping designs.

Runs on the GPU box, indefinitely. Each experiment: run the differentiable
optimizer (Euler — the MJX<->CPU matching integrator) with one reward config,
then read its built-in CPU validation. Designs are scored on the CPU (the
robustness gate): must keep hover lift, minimize CPU peak torque. The config
search seeds a small grid, then hill-climbs around the best.

State persists in opt/auto/: ledger.jsonl (one row per experiment), state.json
(best config/design, counter). Safe to kill/restart — it resumes from state.

    setsid uv run python opt/auto_tune.py > opt/auto/auto.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "opt" / "auto"
BASE_TG = AUTO / "base" / "shoulder.tg"
LEDGER = AUTO / "ledger.jsonl"
STATE = AUTO / "state.json"

HOVER_FRAC = 0.95          # design must keep >= this fraction of baseline lift
TENDON_BASE_STIFF = 200.0
EXPERIMENT_TIMEOUT = 3 * 3600

# Reward-config search space (the levers). wtau drives torque reduction; ws
# regularizes harmonic relative-amplitudes (overfit control); K is harmonic count.
SEEDS = [
    # tau_max=100 mNm regime (added-mass off, lift ~210 mN achievable). Achievable
    # floor (selection works), high wL (lift), w_power minimizes inactive power.
    {"K": 5, "wL": 30, "wtau": 20, "ws": 0.01, "lift_floor": 0.17, "tau_min": 0.15, "lr": 5e-3, "iters": 40, "sample": 4, "w_lc": 0.0, "w_power": 2.0},
    {"K": 5, "wL": 40, "wtau": 25, "ws": 0.01, "lift_floor": 0.18, "tau_min": 0.15, "lr": 5e-3, "iters": 40, "sample": 4, "w_lc": 0.0, "w_power": 5.0},
    {"K": 5, "wL": 30, "wtau": 20, "ws": 0.02, "lift_floor": 0.17, "tau_min": 0.15, "lr": 5e-3, "iters": 40, "sample": 4, "w_lc": 1.0, "w_power": 8.0},
    {"K": 4, "wL": 35, "wtau": 15, "ws": 0.02, "lift_floor": 0.16, "tau_min": 0.15, "lr": 5e-3, "iters": 40, "sample": 4, "w_lc": 0.0, "w_power": 3.0},
    {"K": 5, "wL": 45, "wtau": 30, "ws": 0.005, "lift_floor": 0.18, "tau_min": 0.15, "lr": 5e-3, "iters": 40, "sample": 4, "w_lc": 0.0, "w_power": 12.0},
]
PERTURB = {  # (min, max) multiplicative jitter when hill-climbing the best config
    "wtau": (0.6, 1.7), "ws": (0.4, 2.5), "lr": (0.6, 1.6),
}
INT_PERTURB = {"K": [3, 4, 5, 6], "iters": [40, 60], "sample": [4, 6]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"counter": 0, "best": None, "baseline": None}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def run_optimizer(cfg, out_dir, extra=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", "opt/optimize_control.py",
           "--tg", str(BASE_TG), "--scene", str(out_dir / "scene.xml"),
           "--integrator", "euler", "--w-lc", str(cfg.get("w_lc", 0.0)),
           "--w-power", str(cfg.get("w_power", 0.0)),
           "--out", str(out_dir),
           "--harmonics", str(cfg["K"]), "--lift-floor", str(cfg["lift_floor"]),
           "--weights", str(cfg.get("wL", 20)), str(cfg["wtau"]), "1", str(cfg["ws"]),
           "--tau-min", str(cfg.get("tau_min", 0.12)),
           "--lr", str(cfg["lr"]), "--iters-per-stage", str(cfg["iters"]),
           "--warmup-periods", "2", "--sample-periods", str(cfg["sample"]),
           *extra]
    subprocess.run(cmd, cwd=ROOT, timeout=EXPERIMENT_TIMEOUT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return json.loads((out_dir / "cpu_validation.json").read_text())


def baseline_metrics():
    """CPU metrics of the un-optimized control (0 iters, no tendons) = reference."""
    out = AUTO / "run_baseline"
    cfg = {"K": 2, "wtau": 25, "ws": 0.01, "lift_floor": 0.0, "lr": 5e-3, "iters": 0, "sample": 4}
    val = run_optimizer(cfg, out, extra=())  # no --tune-tendons -> control only, 0 iters = seed
    return val["cpu"].get("implicitfast", {})


def score(val, base_lift):
    cpu = val.get("cpu", {}).get("implicitfast", {})
    if "mean_lift_N" not in cpu:
        return None
    lift, tau = cpu["mean_lift_N"], cpu["peak_torque_Nm"]
    hover = lift >= HOVER_FRAC * base_lift
    return {"lift": lift, "tau": tau, "hover": hover,
            "obj": tau + (0.0 if hover else 0.5)}  # lower is better; non-hover penalized


def perturb(cfg):
    c = dict(cfg)
    for k, (lo, hi) in PERTURB.items():
        c[k] = cfg[k] * random.uniform(lo, hi)
    for k, choices in INT_PERTURB.items():
        if random.random() < 0.5:
            c[k] = random.choice(choices)
    c["lift_floor"] = round(min(0.20, max(0.14, cfg["lift_floor"] * random.uniform(0.92, 1.08))), 3)
    if random.random() < 0.5:
        c["w_lc"] = round(max(0.0, cfg.get("w_lc", 0.0) + random.uniform(-1.5, 2.0)), 2)
    c["tau_min"] = 0.15  # fixed: tau_max=100 mNm per spec
    c["wL"] = round(min(60, max(10, cfg.get("wL", 25) * random.uniform(0.8, 1.4))), 1)
    if random.random() < 0.6:
        c["w_power"] = round(max(0.0, cfg.get("w_power", 2.0) + random.uniform(-2.0, 3.0)), 2)
    return c


def main():
    AUTO.mkdir(parents=True, exist_ok=True)
    s = load_state()
    if s["baseline"] is None:
        log("measuring baseline...")
        s["baseline"] = baseline_metrics()
        save_state(s)
    base_lift = s["baseline"].get("mean_lift_N", 0.19)
    base_tau = s["baseline"].get("peak_torque_Nm", 0.257)
    log(f"baseline CPU: lift={base_lift*1e3:.1f} mN  tau={base_tau*1e3:.1f} mNm")

    while True:
        i = s["counter"]
        if i < len(SEEDS):
            cfg = SEEDS[i]
            kind = "seed"
        elif s["best"] is not None:
            cfg = perturb(s["best"]["cfg"])
            kind = "climb"
        else:  # no successful seed yet -> random restart from a seed
            cfg = perturb(random.choice(SEEDS))
            kind = "restart"
        out = AUTO / f"run_{i}"
        log(f"exp {i} ({kind}): {cfg}")
        try:
            val = run_optimizer(cfg, out, extra=["--tune-tendons", "--tune-pitch", "--tune-motor-tendons"])
            sc = score(val, base_lift)
        except Exception as e:
            sc, val = None, {"error": str(e)[:300]}
            log(f"exp {i} FAILED: {e}")
        row = {"i": i, "kind": kind, "cfg": cfg, "score": sc, "cpu": val.get("cpu"),
               "mjx": val.get("mjx"), "t": time.time()}
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if sc is not None:
            improved = (s["best"] is None) or (sc["hover"] and (
                not s["best"]["score"]["hover"] or sc["obj"] < s["best"]["score"]["obj"]))
            tag = ""
            if improved:
                s["best"] = {"cfg": cfg, "score": sc, "run": str(out)}
                tag = "  <-- NEW BEST"
            log(f"exp {i}: lift={sc['lift']*1e3:.1f} mN  tau={sc['tau']*1e3:.1f} mNm "
                f"hover={sc['hover']}{tag}")
        s["counter"] = i + 1
        save_state(s)


if __name__ == "__main__":
    main()
