"""The hummer wing's planform, measured from its STL mesh.

The blade-element aero (CPU and MJX) needs the wing's span, a chord per blade
strip and the span cutoffs; the compiler needs the same span to place the aero
blade sites. Both read it here from the mesh the .tg names (`mesh` under
`meshdir`, a relative meshdir being taken from the repo root), so they agree by
construction. Loads are cached per (mesh, blade count, mtime).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from runtime.morphology_runtime import load_design_morphology

ROOT = Path(__file__).resolve().parents[2]
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


@dataclass(frozen=True)
class WingGeometry:
    length: float
    chords: np.ndarray
    span_cutoffs: np.ndarray
    span_axis: int
    chord_axis: int
    mesh_path: str


def wing_mesh_path(mesh_file: str, meshdir: str) -> Path:
    """The wing STL: `mesh_file` under `meshdir`; relative paths start at the repo root."""
    return ROOT / meshdir / mesh_file


def load_default_wing_geometry(
    num_blades: int | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> WingGeometry:
    """Wing geometry of the active design (runtime.morphology_runtime, MORPHOLOGY_PATH)."""
    del config_path
    morph = load_design_morphology()
    if num_blades is None:
        num_blades = morph.wing.aero.blades
    return load_wing_geometry_from_stl(wing_mesh_path(morph.wing.mesh_file, morph.world.meshdir), num_blades)


def load_wing_geometry_from_stl(mesh_path: str | os.PathLike[str], num_blades: int) -> WingGeometry:
    if not isinstance(num_blades, int):
        raise TypeError("num_blades must be an integer")
    if num_blades <= 0:
        raise ValueError("num_blades must be > 0")
    path = Path(mesh_path)
    return _load_wing_geometry_cached(str(path.resolve()), num_blades, path.stat().st_mtime_ns)


@lru_cache(maxsize=32)
def _load_wing_geometry_cached(mesh_path: str, num_blades: int, mtime_ns: int) -> WingGeometry:
    del mtime_ns  # part of the cache key only
    triangles = _read_stl_triangles(Path(mesh_path))
    vertices = triangles.reshape(-1, 3)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extents = maxs - mins
    span_axis = int(np.argmax(extents))
    chord_axis = int(np.argsort(extents)[-2])

    length = float(extents[span_axis])
    if length <= 0:
        raise ValueError(f"Wing mesh has zero span length: {mesh_path}")

    chord_length = float(extents[chord_axis])
    span_min = float(mins[span_axis])
    chord_max = float(maxs[chord_axis])

    chords = []
    for j in range(num_blades):
        values = _section_values(triangles, span_axis, span_min + ((j + 0.5) / num_blades) * length, chord_axis)
        chords.append(float(values.max() - values.min()) if values.size else 0.0)
    cutoffs = []
    for k in range(num_blades):
        values = _section_values(triangles, chord_axis, chord_max - ((k + 1) / num_blades) * chord_length, span_axis)
        cutoffs.append((float(values.max()) if values.size else 0.0) - span_min)

    chords = np.array(chords, dtype=float)
    span_cutoffs = np.maximum(np.array(cutoffs, dtype=float), 0.0)
    chords.setflags(write=False)
    span_cutoffs.setflags(write=False)
    return WingGeometry(
        length=length,
        chords=chords,
        span_cutoffs=span_cutoffs,
        span_axis=span_axis,
        chord_axis=chord_axis,
        mesh_path=mesh_path,
    )


def _read_stl_triangles(path: Path) -> np.ndarray:
    """(n, 3, 3) float64 triangle vertices from a binary or ASCII STL."""
    data = path.read_bytes()
    triangles = None
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + tri_count * _STL_RECORD.itemsize == len(data):
            triangles = np.frombuffer(data, _STL_RECORD, count=tri_count, offset=84)["vertices"].astype(float)
    if triangles is None:
        vertices = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        usable = len(vertices) - (len(vertices) % 3)
        triangles = np.array(vertices[:usable], dtype=float).reshape(-1, 3, 3)
    if triangles.size == 0:
        raise ValueError(f"No triangles found in STL mesh: {path}")
    return triangles


def _section_values(triangles: np.ndarray, plane_axis: int, plane_value: float, output_axis: int) -> np.ndarray:
    """`output_axis` coordinates where the triangle edges meet the plane x[plane_axis] = plane_value.

    A vertex within 1e-12 of the plane counts as on it; an edge crossing the
    plane strictly between its ends contributes the interpolated point.
    """
    v0 = triangles
    v1 = triangles[:, [1, 2, 0]]  # edges (0,1), (1,2), (2,0)
    d0 = v0[..., plane_axis] - plane_value
    d1 = v1[..., plane_axis] - plane_value
    on0 = np.abs(d0) <= 1e-12
    on1 = np.abs(d1) <= 1e-12
    cross = ~on0 & ~on1 & (d0 * d1 < 0)
    a, b = v0[cross], v1[cross]
    t = (plane_value - a[:, plane_axis]) / (b[:, plane_axis] - a[:, plane_axis])
    return np.concatenate([
        v0[..., output_axis][on0],
        v1[..., output_axis][on1],
        a[:, output_axis] + t * (b[:, output_axis] - a[:, output_axis]),
    ])
