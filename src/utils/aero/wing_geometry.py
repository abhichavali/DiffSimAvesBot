"""Wing planform geometry derived from the MuJoCo wing mesh."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from runtime.morphology_runtime import load_design_morphology

DEFAULT_GENERATED_XML_PATH = "templates/generated-joint/scene.xml"


@dataclass(frozen=True)
class WingGeometry:
    length: float
    chords: np.ndarray
    span_cutoffs: np.ndarray
    span_axis: int
    chord_axis: int
    mesh_path: str


def load_default_wing_geometry(
    num_blades: int | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> WingGeometry:
    """Load wing geometry from the mesh declared in the active morphology."""
    del config_path
    morph = load_design_morphology()
    if num_blades is None:
        num_blades = morph.wing.aero.blades
    mesh_path = _resolve_wing_mesh_path(morph.wing.mesh_file, morph.world.meshdir)
    return load_wing_geometry_from_stl(mesh_path, num_blades=num_blades)


def load_wing_geometry_from_config(
    config: dict,
    num_blades: int | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> WingGeometry:
    """Backward-compatible entry point; *config* is ignored in favour of morphology."""
    del config, config_path
    return load_default_wing_geometry(num_blades=num_blades)


def default_num_blades(config: dict | None = None) -> int:
    del config
    return _validate_num_blades(load_design_morphology().wing.aero.blades)


def _resolve_wing_mesh_path(mesh_file: str, meshdir: str) -> Path:
    override = os.getenv("MUJOCO_WING_MESH_PATH")
    if override:
        return Path(override)

    mesh_path = Path(mesh_file)
    if mesh_path.is_absolute():
        return mesh_path

    meshdir_path = Path(meshdir)
    scene_path = Path(os.getenv("MUJOCO_XML_PATH", DEFAULT_GENERATED_XML_PATH))
    candidate_roots = [
        scene_path.parent / meshdir_path,
        Path("templates/generated-joint") / meshdir_path,
        Path("templates/shoulder-joint") / meshdir_path,
        meshdir_path,
    ]

    for root in candidate_roots:
        candidate = root / mesh_path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root / mesh_path) for root in candidate_roots)
    raise FileNotFoundError(f"Could not find wing mesh '{mesh_path}'. Searched: {searched}")


def load_wing_geometry_from_stl(
    mesh_path: str | os.PathLike[str],
    num_blades: int,
    scale: Iterable[float] | None = None,
) -> WingGeometry:
    path = Path(mesh_path)
    num_blades = _validate_num_blades(num_blades)
    scale_tuple = tuple(float(v) for v in ((1.0, 1.0, 1.0) if scale is None else scale))
    if len(scale_tuple) != 3:
        raise ValueError("Wing mesh scale must have three components")
    return _load_wing_geometry_from_stl_cached(
        str(path.resolve()),
        num_blades,
        scale_tuple,
        path.stat().st_mtime_ns,
    )


def _validate_num_blades(num_blades: int) -> int:
    if not isinstance(num_blades, int):
        raise TypeError("num_blades must be an integer")
    if num_blades <= 0:
        raise ValueError("num_blades must be > 0")
    return num_blades


@lru_cache(maxsize=32)
def _load_wing_geometry_from_stl_cached(
    mesh_path: str,
    num_blades: int,
    scale: tuple[float, float, float],
    mtime_ns: int,
) -> WingGeometry:
    del mtime_ns
    triangles = _read_stl_triangles(Path(mesh_path), np.array(scale, dtype=float))
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

    chords = np.array(
        [
            _cross_section_extent(
                triangles,
                plane_axis=span_axis,
                plane_value=span_min + ((j + 0.5) / num_blades) * length,
                extent_axis=chord_axis,
            )
            for j in range(num_blades)
        ],
        dtype=float,
    )

    span_cutoffs = np.array(
        [
            _cross_section_max(
                triangles,
                plane_axis=chord_axis,
                plane_value=chord_max - ((k + 1) / num_blades) * chord_length,
                extent_axis=span_axis,
            )
            - span_min
            for k in range(num_blades)
        ],
        dtype=float,
    )
    span_cutoffs = np.maximum(span_cutoffs, 0.0)

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


def _read_stl_triangles(path: Path, scale: np.ndarray) -> np.ndarray:
    data = path.read_bytes()
    triangles = _read_binary_stl_triangles(data)
    if triangles is None:
        triangles = _read_ascii_stl_triangles(data)
    if triangles.size == 0:
        raise ValueError(f"No triangles found in STL mesh: {path}")
    return triangles * scale


def _read_binary_stl_triangles(data: bytes) -> np.ndarray | None:
    if len(data) < 84:
        return None

    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_len = 84 + tri_count * 50
    if expected_len != len(data):
        return None

    triangles = np.empty((tri_count, 3, 3), dtype=float)
    offset = 84
    for tri_index in range(tri_count):
        offset += 12
        for vertex_index in range(3):
            triangles[tri_index, vertex_index] = struct.unpack_from("<fff", data, offset)
            offset += 12
        offset += 2
    return triangles


def _read_ascii_stl_triangles(data: bytes) -> np.ndarray:
    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    usable_vertex_count = len(vertices) - (len(vertices) % 3)
    return np.array(vertices[:usable_vertex_count], dtype=float).reshape(-1, 3, 3)


def _cross_section_extent(
    triangles: np.ndarray,
    plane_axis: int,
    plane_value: float,
    extent_axis: int,
) -> float:
    values = _intersect_triangle_edges(triangles, plane_axis, plane_value, extent_axis)
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _cross_section_max(
    triangles: np.ndarray,
    plane_axis: int,
    plane_value: float,
    extent_axis: int,
) -> float:
    values = _intersect_triangle_edges(triangles, plane_axis, plane_value, extent_axis)
    if not values:
        return 0.0
    return float(max(values))


def _intersect_triangle_edges(
    triangles: np.ndarray,
    plane_axis: int,
    plane_value: float,
    output_axis: int,
) -> list[float]:
    values = []
    eps = 1e-12
    for tri in triangles:
        for i0, i1 in ((0, 1), (1, 2), (2, 0)):
            v0 = tri[i0]
            v1 = tri[i1]
            d0 = float(v0[plane_axis] - plane_value)
            d1 = float(v1[plane_axis] - plane_value)

            if abs(d0) <= eps and abs(d1) <= eps:
                values.append(float(v0[output_axis]))
                values.append(float(v1[output_axis]))
            elif abs(d0) <= eps:
                values.append(float(v0[output_axis]))
            elif abs(d1) <= eps:
                values.append(float(v1[output_axis]))
            elif d0 * d1 < 0:
                t = (plane_value - float(v0[plane_axis])) / float(v1[plane_axis] - v0[plane_axis])
                values.append(float(v0[output_axis] + t * (v1[output_axis] - v0[output_axis])))
    return values
