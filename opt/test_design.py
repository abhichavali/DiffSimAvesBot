"""CPU-test a self-contained design dir (morphology + control .tg).

For evaluating externally-provided designs (e.g. best_designs/...). Compiles the
morphology, runs the CPU reference pipeline (numpy aero, substep=4) under euler
and implicitfast, reports mean lift + peak motor torque.

    uv run python opt/test_design.py best_designs/codex_reward_2026-06-27/robust_seed_optimizer_best
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "opt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cpu_validate import cpu_metrics


def _find(d: Path, *names):
    for n in names:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"{names} not in {d}")


def test(design_dir, integrators=("euler", "implicitfast")) -> dict:
    d = Path(design_dir)
    morph = _find(d, "morphology_optimized.tg", "shoulder.tg")
    ctrl = _find(d, "control_optimized.tg", "control.tg")
    tmp = Path(tempfile.mkdtemp())
    src = re.sub(r"(meshdir\s*=\s*)assets", rf"\g<1>{ROOT / 'assets'}", morph.read_text())
    # Repoint the control file (may be an absolute path to the author's worktree).
    src = re.sub(r"(\n\s*control\s*=\s*).*", r"\g<1>control.tg", src)
    (tmp / "shoulder.tg").write_text(src)
    shutil.copyfile(ctrl, tmp / "control.tg")  # morphology's control_file defaults to control.tg
    out = {}
    for integ in integrators:
        try:
            out[integ] = cpu_metrics(tmp / "shoulder.tg", tmp / "scene.xml", integ)
        except Exception as e:
            out[integ] = {"error": str(e)[:200]}
    return out


if __name__ == "__main__":
    for dd in sys.argv[1:]:
        res = test(dd)
        print(dd)
        for integ, m in res.items():
            if "mean_lift_N" in m:
                print(f"  CPU {integ:13s}: lift={m['mean_lift_N']*1e3:6.1f} mN  "
                      f"peak_torque={m['peak_torque_Nm']*1e3:6.1f} mNm  finite={m['finite']}")
            else:
                print(f"  CPU {integ:13s}: {m}")
