"""Canonical angular-quadrature and beam-frame helpers.

This module centralizes utilities shared by source projection and near-field
integration so that numerical-policy fixes apply consistently across code paths.
"""

from __future__ import annotations

import numpy as np


def uniform_polar_grid(n: int) -> np.ndarray:
    """Return a uniform polar-angle grid on [0, pi] with endpoint included."""
    n_i = int(n)
    if n_i < 2:
        raise ValueError(f"`n` must be >= 2 for polar grid. Got {n!r}.")
    return np.linspace(0.0, np.pi, n_i, endpoint=True, dtype=float)


def uniform_periodic_azimuth_grid(n: int) -> np.ndarray:
    """Return a uniform periodic azimuth grid on [0, 2*pi) (endpoint excluded)."""
    n_i = int(n)
    if n_i < 2:
        raise ValueError(f"`n` must be >= 2 for azimuth grid. Got {n!r}.")
    return np.linspace(0.0, 2.0 * np.pi, n_i, endpoint=False, dtype=float)


def trapezoidal_weights(x: np.ndarray) -> np.ndarray:
    """Return 1D trapezoidal integration weights for sample locations `x`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return np.zeros_like(x)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if x.size > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def is_uniform_periodic_azimuth(alpha: np.ndarray) -> bool:
    """Return True when azimuthal samples are uniform and span 2*pi periodically."""
    a = np.asarray(alpha, dtype=float).reshape(-1)
    if a.size < 3:
        return False
    d = np.diff(a)
    d0 = float(d[0])
    if not np.allclose(d, d0, rtol=1e-8, atol=1e-12):
        return False
    full_span = (a[-1] - a[0]) + d0
    return bool(np.isclose(full_span, 2.0 * np.pi, rtol=1e-7, atol=1e-10))


def periodic_azimuthal_weights(alpha: np.ndarray) -> np.ndarray:
    """Return azimuthal quadrature weights with periodic 2*pi shortcut."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.size < 2:
        return np.zeros_like(alpha)
    d = np.diff(alpha)
    if d.size == 0:
        return np.zeros_like(alpha)
    d0 = float(d[0])
    if np.allclose(d, d0, rtol=1e-8, atol=1e-12):
        full_span = (alpha[-1] - alpha[0]) + d0
        if np.isclose(full_span, 2.0 * np.pi, rtol=1e-7, atol=1e-10):
            return np.full_like(alpha, d0)
    return trapezoidal_weights(alpha)


def beam_axis_and_frame(
    polar_angle: float, azimuthal_angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return beam axis n0 and transverse orthonormal vectors (u, v)."""
    st = float(np.sin(polar_angle))
    ct = float(np.cos(polar_angle))
    ca = float(np.cos(azimuthal_angle))
    sa = float(np.sin(azimuthal_angle))
    n0 = np.array([st * ca, st * sa, ct], dtype=float)

    # Choose the Cartesian axis most orthogonal to n0 to avoid near-collinearity
    # without introducing arbitrary angular thresholds.
    basis = np.eye(3, dtype=float)
    ref = basis[int(np.argmin(np.abs(basis @ n0)))]
    u = ref - float(np.dot(ref, n0)) * n0
    nu = float(np.linalg.norm(u))
    u = np.array([0.0, 1.0, 0.0], dtype=float) if nu == 0.0 else u / nu
    v = np.cross(n0, u)
    return n0, u, v
