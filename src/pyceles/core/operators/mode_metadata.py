"""Shared SVWF flattened-mode metadata for operator kernels."""

from __future__ import annotations

from functools import cache

import numpy as np

from pyceles._arrays import expose_read_only_view

from pyceles.core.indexing import iter_modes, n_modes


@cache
def mode_metadata_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return read-only `(tau, l, m)` arrays indexed by flattened SVWF mode id."""
    order = int(lmax)
    tau = np.zeros((n_modes(order),), dtype=np.int32)
    ell = np.zeros_like(tau)
    m = np.zeros_like(tau)
    for tau_i, l_i, m_i, idx in iter_modes(order):
        tau[idx] = tau_i
        ell[idx] = l_i
        m[idx] = m_i
    return (
        expose_read_only_view(tau),
        expose_read_only_view(ell),
        expose_read_only_view(m),
    )


def mode_m_table(lmax: int) -> np.ndarray:
    """Return read-only azimuthal order `m` indexed by flattened SVWF mode id."""
    return mode_metadata_tables(int(lmax))[2]


def mode_tau_l_tables(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return read-only `(tau, l)` arrays indexed by flattened SVWF mode id."""
    tau, ell, _m = mode_metadata_tables(int(lmax))
    return tau, ell


@cache
def mode_pair_p_range_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact translation-order ranges for every `(out_mode, in_mode)` pair.

    The arrays encode the CELES/SMUTHI admissible `p` interval used by compact
    translation kernels:
    - `pair_offset[n1, n2]`: starting offset into flattened compact tables,
    - `pair_pmin[n1, n2]`: first admissible translation order,
    - `pair_pcount[n1, n2]`: number of consecutive orders to visit.
    """
    order = int(lmax)
    nmodes_total = n_modes(order)
    mode_tau, mode_l, mode_m = mode_metadata_tables(order)
    pair_offset = np.zeros((nmodes_total, nmodes_total), dtype=np.int32)
    pair_pmin = np.zeros_like(pair_offset)
    pair_pcount = np.zeros_like(pair_offset)
    offset = 0
    for n1 in range(nmodes_total):
        for n2 in range(nmodes_total):
            p_min = max(
                abs(int(mode_m[n1]) - int(mode_m[n2])),
                abs(int(mode_l[n1]) - int(mode_l[n2])) + abs(int(mode_tau[n1]) - int(mode_tau[n2])),
            )
            p_count = int(mode_l[n1]) + int(mode_l[n2]) - p_min + 1
            pair_offset[n1, n2] = offset
            pair_pmin[n1, n2] = p_min
            pair_pcount[n1, n2] = p_count
            offset += p_count
    return (
        expose_read_only_view(pair_offset),
        expose_read_only_view(pair_pmin),
        expose_read_only_view(pair_pcount),
    )


__all__ = [
    "mode_m_table",
    "mode_metadata_tables",
    "mode_pair_p_range_tables",
    "mode_tau_l_tables",
]
