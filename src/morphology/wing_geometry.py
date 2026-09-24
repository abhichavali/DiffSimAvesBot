"""The hummer wing's planform, measured from its STL mesh where the compiler mounts it.

The blade-element aero (CPU and MJX) needs the wing's span, a chord per blade
strip, where the pitch hinge sits in each chord and how far out the wing starts;
the compiler needs the same stations to place the aero blade sites. Both read it
here from the mesh the .tg names (`mesh` under `meshdir`, a relative meshdir
being taken from the repo root), posed by :func:`wing_mesh_pose` -- the pose the
compiler gives the mesh geom -- so the aero wing is the drawn (and massed) wing
by construction. Loads are cached per (mesh, blade count, mtime, pose).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from morphology.geometry import quat_rotate

ROOT = Path(__file__).resolve().parents[2]
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
# The pitch-mount arms (T-pose) hook the wing at (±arm, mount_z). World-up is the
# body y-axis here (verified numerically: world-up = ∓y), i.e. along the arm reach,
# so a pure VERTICAL displacement of the wing is an offset in y, not z. Pushing the
# mesh out past the hub (|y| = arm + drop) lowers the wing in world so the hook
# rides higher up the wing (toward the top); the arms and hub stay fixed. Absolute
# metres (the STL is not world-scaled).
WING_ATTACH_DROP = 0.035  # +down in world; flip sign to raise the wing
_MESH_QUAT = {"L": (0.5, -0.5, -0.5, -0.5), "R": (0.5, 0.5, -0.5, 0.5)}


@dataclass(frozen=True)
class WingGeometry:
    """The posed L wing mesh, in its body frame (origin on the pitch hinge, span = +z).

    Strip j is centred ``root_offset + (j + 0.5) / N * length`` out from the hinge;
    ``chords[j]`` is the mesh's chord there and ``x_rot[j]`` the hinge's distance
    behind that chord's hinge-side (leading) edge, in chords. R is the exact mirror.
    """

    length: float
    root_offset: float
    chords: np.ndarray
    x_rot: np.ndarray
    mesh_path: str


def wing_mesh_path(mesh_file: str, meshdir: str) -> Path:
    """The wing STL: `mesh_file` under `meshdir`; relative paths start at the repo root."""
    return ROOT / meshdir / mesh_file


def wing_mesh_pose(morph, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(pos, w-first quat) of the ``side`` wing mesh geom in its wing body."""
    scale = morph.world.scale
    arm = morph.wing.mount_arm * scale
    mount_z = morph.wing.rod_length * scale + 0.000096
    y = (1.0 if side == "L" else -1.0) * (arm + WING_ATTACH_DROP)
    return np.array([0.0, y, mount_z]), np.array(_MESH_QUAT[side])


def load_wing_geometry(morph) -> WingGeometry:
    """``morph``'s wing mesh, posed on the L wing, cut into its ``wing.aero.blades`` strips."""
    num_blades = morph.wing.aero.blades
    if not isinstance(num_blades, int):
        raise TypeError("num_blades must be an integer")
    if num_blades <= 0:
        raise ValueError("num_blades must be > 0")
    path = wing_mesh_path(morph.wing.mesh_file, morph.world.meshdir)
    pos, quat = wing_mesh_pose(morph, "L")
    return _load_wing_geometry_cached(str(path.resolve()), num_blades, path.stat().st_mtime_ns,
                                      tuple(pos), tuple(quat))


@lru_cache(maxsize=32)
def _load_wing_geometry_cached(mesh_path: str, num_blades: int, mtime_ns: int,
                               pos: tuple, quat: tuple) -> WingGeometry:
    del mtime_ns  # part of the cache key only
    stl = _read_stl_triangles(Path(mesh_path))
    triangles = (quat_rotate(np.array(quat), stl.reshape(-1, 3).T).T + pos).reshape(stl.shape)
    vertices = triangles.reshape(-1, 3)
    # L wing body: span outboard along +z; chord along +y, from the hinge-side edge to the far edge.
    SPAN, CHORD = 2, 1

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    length = float(maxs[SPAN] - mins[SPAN])
    if length <= 0:
        raise ValueError(f"Wing mesh has zero span length: {mesh_path}")

    span_min = float(mins[SPAN])

    chords, x_rot = [], []
    for j in range(num_blades):
        values = _section_values(triangles, SPAN, span_min + ((j + 0.5) / num_blades) * length, CHORD)
        chord = float(values.max() - values.min()) if values.size else 0.0
        chords.append(chord)
        x_rot.append(-float(values.min()) / chord if chord > 0 else 0.0)

    chords = np.array(chords, dtype=float)
    x_rot = np.array(x_rot, dtype=float)
    for a in (chords, x_rot):
        a.setflags(write=False)
    return WingGeometry(
        length=length,
        root_offset=span_min,
        chords=chords,
        x_rot=x_rot,
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
