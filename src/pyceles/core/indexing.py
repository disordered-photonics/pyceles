"""Indexing conventions for CELES-compatible VSWF coefficient vectors.

We adopt CELES ordering (single sphere):

For tau in {1,2} (tau-major):
  for l = 1..lmax:
    for m = -l..+l:
      append coefficient

tau mapping (as in CELES):
  tau=1 -> M (TE)
  tau=2 -> N (TM)

Total number of modes:
  n_modes(lmax) = 2 * sum_{l=1..lmax} (2l+1) = 2*lmax*(lmax+2)

Closed-form (0-based) index matches CELES multi2single_index.m (minus MATLAB +1):
  scalar_index(l,m) = (l-1)*(l+1) + (m + l)     # in [0, lmax*(lmax+2)-1]
  idx = (tau-1)*lmax*(lmax+2) + scalar_index(l,m)

This ordering avoids interleaving and matches CELES/SMUTHI convention more directly.
"""

from __future__ import annotations

from collections.abc import Iterator


def n_modes(lmax: int) -> int:
    """Total number of vector spherical-wave modes up to `lmax`."""
    lmax = int(lmax)
    if lmax < 1:
        return 0
    return 2 * lmax * (lmax + 2)


def n_scalar(lmax: int) -> int:
    """Modes per polarization block (`tau=1` or `tau=2`) up to `lmax`."""
    lmax = int(lmax)
    if lmax < 1:
        return 0
    return lmax * (lmax + 2)


def scalar_index(l: int, m: int) -> int:
    """0-based scalar index inside one polarization block for `(l,m)`."""
    l = int(l)
    m = int(m)
    if l < 1:
        raise ValueError("l must be >= 1")
    if abs(m) > l:
        raise ValueError("|m| must be <= l")
    return (l - 1) * (l + 1) + (m + l)


def index_vswf(l: int, m: int, tau: int, lmax: int) -> int:
    """Map `(l,m,tau)` to global CELES-order mode index."""
    l = int(l)
    m = int(m)
    tau = int(tau)
    lmax = int(lmax)
    if tau not in (1, 2):
        raise ValueError("tau must be 1 or 2")
    return (tau - 1) * n_scalar(lmax) + scalar_index(l, m)


def unindex_vswf(idx: int, lmax: int) -> tuple[int, int, int]:
    """Inverse map from global mode index to `(l,m,tau)`."""
    idx = int(idx)
    lmax = int(lmax)
    Ns = n_scalar(lmax)
    if idx < 0 or idx >= 2 * Ns:
        raise ValueError("idx out of range")
    tau = 1 if idx < Ns else 2
    s = idx if tau == 1 else idx - Ns
    # invert scalar_index: find l such that (l-1)(l+1) <= s < l(l+2)
    l = 1
    while l < lmax and (l * (l + 2)) <= s:
        l += 1
    base = (l - 1) * (l + 1)
    m = (s - base) - l
    return l, m, tau


def iter_modes(lmax: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield (tau, l, m, idx) in CELES order."""
    lmax = int(lmax)
    Ns = n_scalar(lmax)
    for tau in (1, 2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                idx = (tau - 1) * Ns + (l - 1) * (l + 1) + (m + l)
                yield tau, l, m, idx
