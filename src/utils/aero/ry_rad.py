"""Rotation about the body Y axis (`ry` in MATLAB)."""

from __future__ import annotations

import numpy as np


def ry(theta: float) -> np.ndarray:
    """
    Return the rotation matrix for a right-handed rotation about the Y axis.

    Parameters
    ----------
    theta : float
        Rotation angle in radians, matching the MATLAB helper signature.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Direction cosine matrix equivalent to MATLAB's `ry(theta)`.
    """

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


__all__ = ["ry"]
