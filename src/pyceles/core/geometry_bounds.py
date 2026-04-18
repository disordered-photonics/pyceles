"""Cheap conservative geometric bounds used for LUT sizing."""

from __future__ import annotations

import numpy as np


def conservative_set_diameter(points: np.ndarray) -> float:
    """Return O(N) AABB-diagonal bound on max in-set pair distance."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`points` must have shape (N, 3). Got {pts.shape}.")
    if pts.shape[0] == 0:
        return 0.0
    pmin = np.min(pts, axis=0)
    pmax = np.max(pts, axis=0)
    return float(np.linalg.norm(pmax - pmin))


def conservative_cross_set_max_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return O(N+M) bound on max pair distance between two point sets."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or aa.shape[1] != 3:
        raise ValueError(f"`a` must have shape (N, 3). Got {aa.shape}.")
    if bb.ndim != 2 or bb.shape[1] != 3:
        raise ValueError(f"`b` must have shape (M, 3). Got {bb.shape}.")
    if aa.shape[0] == 0 or bb.shape[0] == 0:
        return 0.0
    amin = np.min(aa, axis=0)
    amax = np.max(aa, axis=0)
    bmin = np.min(bb, axis=0)
    bmax = np.max(bb, axis=0)
    d_axis = np.maximum(np.abs(bmax - amin), np.abs(amax - bmin))
    return float(np.linalg.norm(d_axis))
