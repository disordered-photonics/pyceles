"""Wigner angular-momentum coefficients used by translation and rotation kernels.

This is a CPU-first, deterministic implementation intended for **precompute** use.
For CELES-style translation tables, we only need small (lmax ~ O(10)) values but
we need them to be numerically stable.

Implementation notes
--------------------
- Uses a Racah-style summation with log-factorials via SciPy `gammaln` for
  `wigner_3j`.
- `wigner_3j` is cached with `functools.cache` because translation-table calls
  revisit a constrained integer domain.
- `wigner_d` stays uncached here; rotation matrices cache at the block level
  because their keys include Euler angles from a continuous domain.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from scipy.special import gammaln


def _triangle(l1: int, l2: int, l3: int) -> bool:
    """Triangle-condition check for angular-momentum coupling."""
    return abs(l1 - l2) <= l3 <= (l1 + l2)


def _gammaln(x: float) -> float:
    """Scalar wrapper around `scipy.special.gammaln`."""
    return float(gammaln(x))


def _log_factorial(n: int) -> float:
    """Return `log(n!)` for non-negative integer `n`."""

    if n < 0:
        raise ValueError("factorial arguments must be non-negative")
    return _gammaln(n + 1.0)


# NOTE:
# This cache is intentionally unbounded. Here, wigner_3j is called only
# by translation-table assembly with CELES-style mode-coupling constraints:
# l1 and l2 come from mode indices (1..lmax), l3 is p or p-1, and m values are
# tied to those modes (including m3 = -m1-m2 in the translated convention).
# So reachable tuples are a constrained subset of all integers, not arbitrary.
# If many distinct lmax values are explored in one long process, call
# clear_caches() between runs to release memory.
@cache
def wigner_3j(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    """Compute the Wigner 3j symbol (l1 l2 l3; m1 m2 m3) for integer arguments."""

    l1 = int(l1)
    l2 = int(l2)
    l3 = int(l3)
    m1 = int(m1)
    m2 = int(m2)
    m3 = int(m3)

    if (m1 + m2 + m3) != 0:
        return 0.0
    if any(abs(m) > l for m, l in ((m1, l1), (m2, l2), (m3, l3))):
        return 0.0
    if not _triangle(l1, l2, l3):
        return 0.0

    # phase (-1)^(l1-l2-m3)
    phase = -1.0 if ((l1 - l2 - m3) % 2) else 1.0

    a = l1 + l2 - l3
    b = l1 - l2 + l3
    c = -l1 + l2 + l3
    d = l1 + l2 + l3 + 1
    if min(a, b, c) < 0:
        return 0.0

    log_delta = 0.5 * (
        _log_factorial(a) + _log_factorial(b) + _log_factorial(c) - _log_factorial(d)
    )
    log_m = 0.5 * (
        _log_factorial(l1 + m1)
        + _log_factorial(l1 - m1)
        + _log_factorial(l2 + m2)
        + _log_factorial(l2 - m2)
        + _log_factorial(l3 + m3)
        + _log_factorial(l3 - m3)
    )

    kmin = max(0, l2 - l3 - m1, l1 - l3 + m2)
    kmax = min(l1 + l2 - l3, l1 - m1, l2 + m2)
    if kmin > kmax:
        return 0.0

    s = 0.0
    for k in range(kmin, kmax + 1):
        denom_args = [
            k,
            (l1 + l2 - l3 - k),
            (l1 - m1 - k),
            (l2 + m2 - k),
            (l3 - l2 + m1 + k),
            (l3 - l1 - m2 + k),
        ]
        if min(denom_args) < 0:
            continue
        log_denom = sum(_log_factorial(arg) for arg in denom_args)
        sign = -1.0 if (k % 2) else 1.0
        s += sign * np.exp(-log_denom)

    if s == 0.0:
        return 0.0

    return float(phase * np.exp(log_delta + log_m) * s)


def wigner_d(l: int, m: int, m_prime: int, beta: float) -> float:
    """Return the real Wigner small-`d` coefficient `d^l_{m,m'}(beta)`.

    This closed-form sum follows the CELES/SMUTHI phase convention used by the
    SVWF rotation layer and is faster than the equivalent recurrence in the
    moderate-`lmax` regime targeted by `pyceles`.
    """

    l = int(l)
    m = int(m)
    m_prime = int(m_prime)
    beta = float(beta)
    if l < 0:
        raise ValueError("l must be non-negative.")
    if abs(m) > l or abs(m_prime) > l:
        return 0.0

    k_min = max(0, m - m_prime)
    k_max = min(l + m, l - m_prime)
    if k_min > k_max:
        return 0.0

    prefactor = math.sqrt(
        math.factorial(l + m)
        * math.factorial(l - m)
        * math.factorial(l + m_prime)
        * math.factorial(l - m_prime)
    )
    c_half = math.cos(0.5 * beta)
    s_half = math.sin(0.5 * beta)

    total = 0.0
    for k in range(k_min, k_max + 1):
        denom = (
            math.factorial(l + m - k)
            * math.factorial(k)
            * math.factorial(m_prime - m + k)
            * math.factorial(l - m_prime - k)
        )
        pow_c = 2 * l + m - m_prime - 2 * k
        pow_s = m_prime - m + 2 * k
        total += ((-1) ** k) * (prefactor / denom) * (c_half**pow_c) * (s_half**pow_s)

    return float(total)


def wigner_D(l: int, m: int, m_prime: int, alpha: float, beta: float, gamma: float) -> complex:
    """Return the Wigner-`D` coefficient in the CELES / Doicu rotation convention."""

    l = int(l)
    m = int(m)
    m_prime = int(m_prime)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)

    if m >= 0 and m_prime >= 0:
        delta = 1
    elif m >= 0 and m_prime < 0:
        delta = (-1) ** m_prime
    elif m < 0 and m_prime >= 0:
        delta = (-1) ** m
    else:
        delta = (-1) ** (m + m_prime)

    return (
        ((-1) ** (m + m_prime))
        * np.exp(1j * m * alpha)
        * delta
        * wigner_d(l, m, m_prime, beta)
        * np.exp(1j * m_prime * gamma)
    )


def clear_caches() -> None:
    """Clear process-global Wigner caches."""

    wigner_3j.cache_clear()
