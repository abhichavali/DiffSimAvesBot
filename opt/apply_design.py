"""Materialize an optimizer run's output into a compilable .tg design.

Writes a `shoulder.tg` (base topology + 12 per-edge structural-tendon overrides
from tendon_optimized.json) and a `control.tg` (= control_optimized.tg). Used by
cpu_validate.py and for promoting designs to the demo branch.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morphology.compile import compile_morphology
from morphology.emit_mjcf import anchor_node_name
from morphology.tendons import legacy_edge_name
from morphology.tg_parser import load_morphology


def _legacy_to_semantic(base_shoulder_tg: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Map each L structural edge's legacy name -> (M[..], F[..]) semantic anchors."""
    sh = compile_morphology(load_morphology(base_shoulder_tg)).shoulders[0]
    out: dict[str, tuple[str, str]] = {}
    for e in sh.structural_edges:
        a, b = e.anchor_a, e.anchor_b
        ma = a if a.side == "M" else b
        fa = b if a.side == "M" else a
        legacy = legacy_edge_name(sh.side, anchor_node_name(ma), anchor_node_name(fa))[2:]
        face = lambda x: x.face.value if hasattr(x.face, "value") else x.face
        out[legacy] = (f"M[{ma.station}{face(ma)}]", f"F[{fa.station}{face(fa)}]")
    return out


def write_design(run_dir: pathlib.Path, base_shoulder_tg: pathlib.Path,
                 out_shoulder: pathlib.Path, out_control: pathlib.Path) -> None:
    """Apply run_dir/{control_optimized.tg, tendon_optimized.json} onto a base .tg."""
    base = base_shoulder_tg.read_text()
    # meshdir is relative ("assets") -> absolutize so a temp-dir compile finds the mesh.
    base = re.sub(r"(meshdir\s*=\s*)assets", rf"\g<1>{ROOT / 'assets'}", base)
    # The optimizer's baseline is effective pretension 1.0 (original scene had no
    # springlength); use that so designs are referenced to the same baseline.
    base = base.replace("pretension: 0.827", "pretension: 1.0")
    out_control.write_text((run_dir / "control_optimized.tg").read_text())

    tj = run_dir / "tendon_optimized.json"
    if not tj.exists():  # control-only run: base topology at pretension 1.0
        out_shoulder.write_text(base)
        return

    mapping = _legacy_to_semantic(base_shoulder_tg)
    pairs = json.loads(tj.read_text())["pairs"]
    lines = ["    overrides:"]
    for legacy, (msem, fsem) in mapping.items():
        p = pairs[legacy]
        lines.append(f"        {msem} -- {fsem} "
                     f"{{ stiffness: {p['stiffness']:.4g}, pretension: {p['pretension_factor']:.4g} }}")
    # pretension default 1.0 = the optimizer's baseline (springlength = length).
    src = base.replace("pretension: 0.827", "pretension: 1.0")
    anchor = "        M[2B] -- F[1B]\n"
    src = src.replace(anchor, anchor + "\n" + "\n".join(lines) + "\n", 1)
    out_shoulder.write_text(src)


if __name__ == "__main__":  # tiny self-check on the generated structure
    import tempfile
    rd = ROOT / "opt" / "runs_joint"
    if (rd / "control_optimized.tg").exists():
        d = pathlib.Path(tempfile.mkdtemp())
        write_design(rd, ROOT / "design" / "shoulder.tg", d / "shoulder.tg", d / "control.tg")
        s = (d / "shoulder.tg").read_text()
        assert "overrides:" in s and "pretension: 1.0" in s, "override block missing"
        print("ok: wrote", d / "shoulder.tg")
    else:
        print("no run to apply; skipping")
