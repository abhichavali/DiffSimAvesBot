"""Per-cycle mean lift and peak torque of a design dir, on the exact certification rollout.

    uv run python .claude/skills/certify-design/cycle_lift.py <design_dir> [--integrator euler] [--timestep 0.06]

Runs codesign.artifacts.validate (what opt/design_io.py runs: warmup 8, 24 sampled
cycles) with PlateWingSim.step wrapped to record lift, so the numbers are the same
rollout's; it only splits the sampled window into cycles. Writes the design dir's
_recon_scene.xml like design_io: do not run it beside another replay of the same dir.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from codesign import cpu_check  # noqa: E402
from codesign.artifacts import validate  # noqa: E402

lifts = []


class _Recording(cpu_check.PlateWingSim):
    def step(self, *a, **k):
        lift = super().step(*a, **k)
        lifts.append(lift)
        return lift


cpu_check.PlateWingSim = _Recording

ap = argparse.ArgumentParser()
ap.add_argument("design_dir", type=Path)
ap.add_argument("--integrator", nargs="+", default=["euler"])
ap.add_argument("--timestep", type=float, default=0.06, help="ms")
ap.add_argument("--periods", type=int, default=24)
args = ap.parse_args()
for integ in args.integrator:
    lifts.clear()
    r = validate(args.design_dir, [integ], timestep_ms=args.timestep, periods=args.periods)[integ]
    per = len(lifts) / (8 + args.periods)  # steps per cycle (to +-1 step of cpu_rollout's)
    n, spc = len(r.cycle_peaks), round(per)
    start = len(lifts) - round(args.periods * per)  # first sampled step
    L = np.asarray(lifts[start:start + n * spc]).reshape(n, spc).mean(axis=1)
    print(f"[{args.design_dir.name}/{integ} dt {args.timestep}] mean lift {r.lift*1e3:.1f} mN, "
          f"peak {r.peak*1e3:.1f} mNm, drift {(r.cycle_peaks[-1]-r.cycle_peaks[0])*1e3:+.2f} mNm")
    for i, (l, p) in enumerate(zip(L, r.cycle_peaks)):
        print(f"  cycle {i:2d}  lift {l*1e3:6.1f} mN  peak {p*1e3:6.1f} mNm")
