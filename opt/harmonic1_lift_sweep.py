#!/usr/bin/env python3
"""First-harmonic lift sweep: is there an (a_1, b_1) that raises lift at EVERY discretization?

Injects a_1 = d*cos(phi), b_1 = d*sin(phi) (modulation depths, fraction of a0) on top
of the design's pure sinusoid -- no optimizer in the loop -- and rolls each point out on
MuJoCo CPU in every (integrator, dt) cell. A point only counts if its lift gain over the
zero-harmonic baseline holds in the WORST cell. Rows stream to --out (JSONL); re-running
skips rows already there.

    uv run python opt/harmonic1_lift_sweep.py
    uv run python opt/harmonic1_lift_sweep.py --summary-only
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "opt"):
    sys.path.insert(0, str(p))

from optimize_control import cpu_validate_design, set_unsteady_aero  # noqa: E402


def run(task):
    set_unsteady_aero(False)  # doctrine: quasi-steady lift only (added-mass FD is itself dt-dependent)
    scene, a0, omega, lag, integ, dt, d, phi, warmup, periods = task
    ph = math.radians(phi)
    L, T, P, pc = cpu_validate_design(
        scene, (a0, [d * math.cos(ph)], [d * math.sin(ph)], omega), None, integ,
        warmup=warmup, periods=periods, phase_lag=lag, timestep_ms=dt, aero_substep=1,
        cycle_stats=True)
    return dict(integ=integ, dt=dt, d=d, phi=phi, lift_mN=L * 1e3, peak_mNm=T * 1e3,
                power_W=P, drift_mNm=(pc[-1] - pc[0]) * 1e3)


def summarize(rows, cells):
    base = {(r["integ"], r["dt"]): r["lift_mN"] for r in rows if r["d"] == 0}
    pts = {}
    for r in rows:
        if r["d"] and (r["integ"], r["dt"]) in base:
            pts.setdefault((r["d"], r["phi"]), {})[(r["integ"], r["dt"])] = r
    full = {k: v for k, v in pts.items() if len(v) == len(cells)}
    print("baseline lift: " + "  ".join(f"{i}@{dt:g}={base[(i, dt)]:.1f}"
                                        for i, dt in cells if (i, dt) in base))
    print(f"{len(full)}/{len(pts)} points have every cell\n")
    hdr = "   d    phi  worst  " + " ".join(f"{i[:5]}@{dt:g}".rjust(12) for i, dt in cells)
    print(hdr + "   peak(worst)")
    ranked = sorted(full.items(), key=lambda kv: -min(r["lift_mN"] - base[c] for c, r in kv[1].items()))
    for (d, phi), cell in ranked[:25]:
        dl = [cell[c]["lift_mN"] - base[c] for c in cells]
        print(f"{d:5.3f} {phi:5.0f}  {min(dl):+5.1f}  " + " ".join(f"{x:+12.1f}" for x in dl)
              + f"   {max(cell[c]['peak_mNm'] for c in cells):6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tg", type=Path, default=ROOT / "design" / "shoulder.tg")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--phases", type=int, default=12, help="phi samples over [0, 360)")
    ap.add_argument("--cells", nargs="+",
                    default=["euler:0.015", "euler:0.03", "euler:0.06", "euler:0.09", "euler:0.13",
                             "implicitfast:0.03", "implicitfast:0.06", "rk4:0.06"],
                    help="integrator:dt_ms discretizations every point must survive")
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--periods", type=float, default=24.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "opt" / "harmonic1_lift_sweep.jsonl")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cells = [(c.split(":")[0], float(c.split(":")[1])) for c in args.cells]
    done = [json.loads(s) for s in args.out.read_text().splitlines()] if args.out.exists() else []

    if not args.summary_only:
        from optimize_control import setup
        from morphology.tg_parser import load_morphology
        scene = ROOT / "opt" / "scene_h1_sweep.xml"
        _, _, _, control, drive, *_ = setup(args.tg, scene)  # compiles the scene once
        a0, omega = float(drive.a0), float(control.omega)
        lag = load_morphology(args.tg).motors.crank_phase_lag_rad
        grid = [(0.0, 0.0)] + [(d, 360.0 * k / args.phases)
                               for d in args.depths for k in range(args.phases)]
        seen = {(r["integ"], r["dt"], r["d"], r["phi"]) for r in done}
        tasks = [(str(scene), a0, omega, lag, i, dt, d, phi, args.warmup, args.periods)
                 for i, dt in cells for d, phi in grid if (i, dt, d, phi) not in seen]
        print(f"a0={a0} omega={omega:.4f} lag={math.degrees(lag):.1f}deg  "
              f"{len(tasks)} rollouts ({len(done)} cached)", flush=True)
        with Pool(args.jobs) as pool, args.out.open("a") as f:
            for n, r in enumerate(pool.imap_unordered(run, tasks), 1):
                f.write(json.dumps(r) + "\n"); f.flush()
                done.append(r)
                print(f"[{n}/{len(tasks)}] {r['integ']}@{r['dt']:g} d={r['d']:.3f} phi={r['phi']:5.1f}"
                      f"  lift={r['lift_mN']:6.1f}  peak={r['peak_mNm']:6.1f}", flush=True)
    summarize(done, cells)


if __name__ == "__main__":
    main()
