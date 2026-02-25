"""Wigner 3j symbols (integer arguments), cached.

This is a CPU-first, deterministic implementation intended for **precompute** use.
For CELES-style translation tables, we only need small (lmax ~ O(10)) values but
we need them to be numerically stable.

Implementation notes
--------------------
- Uses a Racah-style summation with log-factorials via SciPy `gammaln`.
- Cached with `functools.cache` because many calls repeat.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.special import gammaln


def _triangle(l1: int, l2: int, l3: int) -> bool:
    """Triangle-condition check for angular-momentum coupling."""
    return abs(l1 - l2) <= l3 <= (l1 + l2)


def _gammaln(x: float) -> float:
    """Scalar wrapper around `scipy.special.gammaln`."""
    return float(gammaln(x))


# NOTE:
# This cache is intentionally unbounded. Here, wigner_3j is called only
# by translation-table assembly with CELES-style mode-coupling constraints:
# l1 and l2 come from mode indices (1..lmax), l3 is p or p-1, and m values are
# tied to those modes (including m3 = -m1-m2 in the translated convention).
# So reachable tuples are a constrained subset of all integers, not arbitrary.
# If many distinct lmax values are explored in one long process, call
# clear_cache() between runs to release memory.
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

    def log_fact(n: int) -> float:
        """Log-factorial helper used by Racah summation terms."""
        if n < 0:
            return float("nan")
        return _gammaln(n + 1.0)

    # phase (-1)^(l1-l2-m3)
    phase = -1.0 if ((l1 - l2 - m3) % 2) else 1.0

    a = l1 + l2 - l3
    b = l1 - l2 + l3
    c = -l1 + l2 + l3
    d = l1 + l2 + l3 + 1
    if min(a, b, c) < 0:
        return 0.0

    log_delta = 0.5 * (log_fact(a) + log_fact(b) + log_fact(c) - log_fact(d))
    log_m = 0.5 * (
        log_fact(l1 + m1)
        + log_fact(l1 - m1)
        + log_fact(l2 + m2)
        + log_fact(l2 - m2)
        + log_fact(l3 + m3)
        + log_fact(l3 - m3)
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
        log_denom = sum(log_fact(arg) for arg in denom_args)
        sign = -1.0 if (k % 2) else 1.0
        s += sign * np.exp(-log_denom)

    if s == 0.0:
        return 0.0

    return float(phase * np.exp(log_delta + log_m) * s)


def clear_cache() -> None:
    """Clear the global Wigner cache."""

    wigner_3j.cache_clear()
