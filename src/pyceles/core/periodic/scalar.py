"""Shared scalar helper kernels for periodic Ewald/direct structural sums."""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from .special import upper_incomplete_gamma_int_or_halfint

Array = np.ndarray


def validate_shell_count(value: int, *, name: str) -> int:
    """Validate one nonnegative shell/window count."""
    shells = int(value)
    if shells < 0:
        raise ValueError(f"`{name}` must be >= 0. Got {value!r}.")
    return shells


def square_shell_indices(shell: int) -> tuple[tuple[int, int], ...]:
    """Return rectangular-lattice index pairs on one square shell."""
    s = int(shell)
    if s < 0:
        raise ValueError(f"`shell` must be >= 0. Got {shell!r}.")
    if s == 0:
        return ((0, 0),)
    out: list[tuple[int, int]] = []
    for p in (-s, s):
        for q in range(-s, s + 1):
            out.append((p, q))
    for p in range(-s + 1, s):
        for q in (-s, s):
            out.append((p, q))
    return tuple(out)


def factorial_int(value: int | np.integer) -> int:
    """Return factorial for one integer-like value."""
    return math.factorial(int(value))


def structural_sum_m_normalization(order: int) -> float:
    """Return the SMUTHI-to-pyceles scalar normalization for order `M`."""
    m = int(order)
    if m >= 0:
        return math.sqrt(2.0 * math.pi) * ((-1.0) ** (-m))
    return math.sqrt(2.0 * math.pi)


def reciprocal_gamma(k: float, rho: Array) -> Array:
    """Return reciprocal `gamma = sqrt(k^2 - rho^2)` with zero-guard branch."""
    gamma = np.sqrt((float(k) * float(k) - np.asarray(rho, dtype=float) ** 2) + 0.0j)
    gamma[np.where(gamma == 0.0)[0]] += 1.0e-10j
    return np.asarray(gamma, dtype=np.complex128)


def upper_gamma_sequence(max_index: int, z: Array) -> Array:
    """Return `Gamma(1/2-n, z)` for `n = 0..max_index` and vectorized `z`."""
    n_max = int(max_index)
    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.zeros((z_arr.size, n_max + 1), dtype=np.complex128)
    for idx, zc in enumerate(z_arr):
        for n in range(n_max + 1):
            out[idx, n] = upper_incomplete_gamma_int_or_halfint(0.5 - float(n), zc)
    return out


def real_integral_sequence(degree: int, eta: float, k: float, radii: Array) -> Array:
    """Evaluate the real-space integral sequence used by the Kambe summand."""
    l = int(degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")
    alpha = float(k) * float(k) / (4.0 * float(eta) * float(eta))
    root_alpha = np.sqrt(alpha)
    w = special.wofz(root_alpha + 1j * float(k) * r / (2.0 * root_alpha))
    exp_term = np.exp(alpha - (float(k) * r) ** 2 / (4.0 * alpha))

    vals = np.zeros((r.size, l + 2), dtype=np.complex128)
    vals[:, 0] = math.sqrt(math.pi) * exp_term * w.imag
    vals[:, 1] = math.sqrt(math.pi) * 2.0 / (float(k) * r) * exp_term * w.real
    for idx in range(2, l + 2):
        vals[:, idx] = (2.0 / (float(k) * r)) ** 2 * (
            0.5 * float(2 * (idx - 2) + 1) * vals[:, idx - 1]
            - vals[:, idx - 2]
            + alpha ** (-(idx - 2) - 0.5) * exp_term
        )
    return vals[:, -1]
