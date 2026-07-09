"""Spherical angular helper functions in CELES conventions.

This module intentionally stays **NumPy-first** and implements exactly one canonical
version of the spherical angular recurrences used throughout CELES:

- `legendre_normalized_trigon`: Doicu/Wriedt/Eremin-normalized associated Legendre
  functions P_l^m(cos(theta)) for m>=0, evaluated from cos/sin(theta).

- `spherical_functions_trigon`: Doicu-normalized pi_l^m and tau_l^m functions
  (m>=0) used in the vector spherical wave function (VSWF) formulas.

These functions are ports of the MATLAB routines:
- `legendre_normalized_trigon.m`
- `spherical_functions_trigon.m`

Notes
-----
- All inputs are broadcastable arrays.
- We keep an `xp` parameter for future CuPy acceleration, but today this code is
  written and validated primarily for NumPy.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np


@cache
def _legendre_scalar_tables(
    lmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute scalar recurrence coefficients for normalized Legendre values.

    The `c_mm` diagonal normalization is built through an overflow-safe
    recurrence, avoiding factorial/double-factorial intermediates that overflow
    when translation tables request high angular orders.
    """
    lmax = int(lmax)
    a0 = np.zeros((lmax + 1,), dtype=np.float64)
    b0 = np.zeros((lmax + 1,), dtype=np.float64)
    c_mm = np.zeros((lmax + 1,), dtype=np.float64)
    a_lm = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
    b_lm = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
    c_mm[0] = np.sqrt(2.0) / 2.0

    if lmax >= 1:
        for l in range(1, lmax):
            lp1 = l + 1
            a0[l] = (1.0 / lp1) * np.sqrt((2 * l + 1.0) * (2 * l + 3.0))
            b0[l] = (l / lp1) * np.sqrt((2 * l + 3.0) / (2 * l - 1.0))

    c_prev = c_mm[0]
    for m in range(1, lmax + 1):
        c_prev *= np.sqrt((2 * m + 1.0) / (2.0 * m))
        c_mm[m] = c_prev
        for l in range(m, lmax):
            lp1 = l + 1
            den = (lp1 - m) * (lp1 + m)
            a_lm[l, m] = np.sqrt((2 * l + 1.0) * (2 * l + 3.0) / den)
            b_lm[l, m] = np.sqrt((2 * l + 3.0) * (l - m) * (l + m) / ((2 * l - 1.0) * den))

    return a0, b0, c_mm, a_lm, b_lm


@cache
def _legendre_backend_tables(
    lmax: int,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Typed host recurrence tables for array backends."""

    dtype = np.dtype(dtype_name)
    a0, b0, c_mm, a_lm, b_lm = _legendre_scalar_tables(int(lmax))
    return (
        np.asarray(a0, dtype=dtype),
        np.asarray(b0, dtype=dtype),
        np.asarray(c_mm, dtype=dtype),
        np.asarray(a_lm, dtype=dtype),
        np.asarray(b_lm, dtype=dtype),
    )


def legendre_normalized_trigon_scalar(ct: float, st: float, lmax: int) -> np.ndarray:
    """Fast scalar version of CELES-normalized associated Legendre values."""
    lmax = int(lmax)
    ct_f = float(ct)
    st_f = float(st)
    plm = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)

    plm[0, 0] = np.sqrt(2.0) / 2.0
    if lmax == 0:
        return plm

    a0, b0, c_mm, a_lm, b_lm = _legendre_scalar_tables(lmax)
    plm[1, 0] = np.sqrt(3.0 / 2.0) * ct_f

    for l in range(1, lmax):
        plm[l + 1, 0] = a0[l] * ct_f * plm[l, 0] - b0[l] * plm[l - 1, 0]

    st_pow = st_f
    for m in range(1, lmax + 1):
        plm[m, m] = c_mm[m] * st_pow
        for l in range(m, lmax):
            plm[l + 1, m] = a_lm[l, m] * ct_f * plm[l, m] - b_lm[l, m] * plm[l - 1, m]
        st_pow *= st_f

    return plm


def _scalar_like(value: float, dtype: Any) -> Any:
    """Return ``value`` as a NumPy scalar matching an array real dtype."""

    return np.asarray(value, dtype=np.dtype(dtype))[()]


def _legendre_tables_for_backend(
    lmax: int, dtype: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _legendre_backend_tables(int(lmax), np.dtype(dtype).name)


def legendre_normalized_trigon(ct: Any, st: Any, lmax: int, xp=None):
    """CELES/Doicu-normalized associated Legendre values P_l^m(ct) for m>=0.

    Parameters
    ----------
    ct, st:
        cos(theta) and sin(theta) arrays (same shape / broadcastable).
    lmax:
        Maximum degree.
    xp:
        Array module (defaults to NumPy). Kept for future optional CuPy usage.

    Returns
    -------
    plm:
        Array of shape (lmax+1, lmax+1, *ct.shape) with valid entries for 0<=m<=l.
        Values are Doicu/Wriedt/Eremin normalized like in CELES.
    """
    if xp is None:
        xp = np

    ct = xp.asarray(ct)
    st = xp.asarray(st)
    lmax = int(lmax)
    a0, b0, c_mm, a_lm, b_lm = _legendre_tables_for_backend(lmax, ct.dtype)

    plm = xp.zeros((lmax + 1, lmax + 1, *ct.shape), dtype=ct.dtype)

    # m=0 base
    plm[0, 0] = xp.ones_like(ct) * _scalar_like(np.sqrt(2.0) / 2.0, ct.dtype)
    if lmax >= 1:
        plm[1, 0] = _scalar_like(np.sqrt(3.0 / 2.0), ct.dtype) * ct

    # m=0 recurrence for l>=2
    for l in range(1, lmax):
        plm[l + 1, 0] = a0[l] * ct * plm[l, 0] - b0[l] * plm[l - 1, 0]

    # m>=1
    st_pow = st
    for m in range(1, lmax + 1):
        plm[m, m] = c_mm[m] * st_pow
        for l in range(m, lmax):
            lp1 = l + 1
            plm[lp1, m] = a_lm[l, m] * ct * plm[l, m] - b_lm[l, m] * plm[l - 1, m]
        st_pow = st_pow * st

    return plm


def spherical_functions_trigon(ct: Any, st: Any, lmax: int, xp=None, *, return_plm: bool = False):
    """Compute Doicu-normalized pi_l^m and tau_l^m functions (m>=0).

    Port of CELES `spherical_functions_trigon.m`.

    Returns
    -------
    pi, tau:
        Arrays of shape (lmax+1, lmax+1, *ct.shape) for 0<=m<=l.
    """
    if xp is None:
        xp = np

    ct = xp.asarray(ct)
    st = xp.asarray(st)
    lmax = int(lmax)
    a0, b0, c_mm, a_lm, b_lm = _legendre_tables_for_backend(lmax, ct.dtype)

    plm = xp.zeros((lmax + 1, lmax + 1, *ct.shape), dtype=ct.dtype)
    pi = xp.zeros_like(plm)
    tau = xp.zeros_like(plm)
    pprimel0 = xp.zeros((lmax + 1, *ct.shape), dtype=ct.dtype)

    # base
    plm[0, 0] = xp.ones_like(ct) * _scalar_like(np.sqrt(2.0) / 2.0, ct.dtype)
    if lmax >= 1:
        plm[1, 0] = _scalar_like(np.sqrt(3.0 / 2.0), ct.dtype) * ct

    pprimel0[0] = xp.zeros_like(ct)
    if lmax >= 1:
        pprimel0[1] = _scalar_like(np.sqrt(3.0), ct.dtype) * plm[0, 0]

    tau[0, 0] = -st * pprimel0[0]
    if lmax >= 1:
        tau[1, 0] = -st * pprimel0[1]

    # m=0 recurrence for l>=2
    for l in range(1, lmax):
        lp1 = l + 1
        plm[lp1, 0] = a0[l] * ct * plm[l, 0] - b0[l] * plm[l - 1, 0]
        coeff = _scalar_like(np.sqrt((2 * lp1 + 1.0) / (2 * lp1 - 1.0)), ct.dtype)
        pprimel0[lp1] = lp1 * coeff * plm[l, 0] + coeff * ct * pprimel0[l]
        tau[lp1, 0] = -st * pprimel0[lp1]

    # m>=1
    st_pow_prev = xp.ones_like(st)
    st_pow = st
    for m in range(1, lmax + 1):
        coeff = c_mm[m]

        plm[m, m] = coeff * st_pow
        pi[m, m] = coeff * st_pow_prev
        tau[m, m] = m * ct * pi[m, m]

        for l in range(m, lmax):
            lp1 = l + 1
            coeff1 = a_lm[l, m] * ct
            coeff2 = b_lm[l, m]

            plm[lp1, m] = coeff1 * plm[l, m] - coeff2 * plm[l - 1, m]
            pi[lp1, m] = coeff1 * pi[l, m] - coeff2 * pi[l - 1, m]

            tau[lp1, m] = (
                lp1 * ct * pi[lp1, m]
                - (lp1 + m)
                * _scalar_like(
                    np.sqrt((2 * lp1 + 1.0) * (lp1 - m) / ((2 * lp1 - 1.0) * (lp1 + m))),
                    ct.dtype,
                )
                * pi[l, m]
            )
        st_pow_prev = st_pow
        st_pow = st_pow * st

    if return_plm:
        return pi, tau, plm
    return pi, tau


def clear_caches() -> None:
    """Clear process-global spherical recurrence caches."""

    _legendre_scalar_tables.cache_clear()
    _legendre_backend_tables.cache_clear()
