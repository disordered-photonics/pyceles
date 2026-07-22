"""Backend-neutral projection data for periodic interior local fields."""

from __future__ import annotations

from functools import cache

import numpy as np

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.translation import translation_ab5_table


@cache
def l1_projection_data(lmax: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Precompute the minimal contraction tensor for local `l=1` fields.

    The returned kernel contracts periodic structural sums directly into the
    six destination `l=1` modes, avoiding full translation-block construction.
    """

    lmax_i = int(lmax)
    if lmax_i < 1:
        raise ValueError(f"`lmax` must be >= 1. Got {lmax!r}.")
    nm = n_modes(lmax_i)
    row_idx: list[int] = []
    m_dst: list[int] = []
    m_src = np.zeros((nm,), dtype=np.int32)
    for _tau, degree, order_m, idx in iter_modes(lmax_i):
        m_src[idx] = int(order_m)
        if degree == 1:
            row_idx.append(int(idx))
            m_dst.append(int(order_m))
    row_idx_arr = np.asarray(row_idx, dtype=np.int64)
    m_dst_arr = np.asarray(m_dst, dtype=np.int32)

    ab5 = np.asarray(
        translation_ab5_table(lmax_i, dtype=np.complex128),
        dtype=np.complex128,
    )
    ab5_l1 = np.asarray(ab5[row_idx_arr, :, :], dtype=np.complex128)
    max_degree = int(lmax_i + 1)
    lmax_struct = int((max_degree + 1) // 2)
    structural_order = 2 * lmax_struct
    p_count = max_degree + 1
    m_offset = structural_order
    kernel = np.zeros(
        (ab5_l1.shape[0], nm, 2 * structural_order + 1, p_count),
        dtype=np.complex128,
    )
    for row in range(ab5_l1.shape[0]):
        dm_idx = m_src - int(m_dst_arr[row]) + m_offset
        for col in range(nm):
            kernel[row, col, int(dm_idx[col]), :p_count] = ab5_l1[row, col, :p_count]
    kernel.setflags(write=False)
    row_idx_arr.setflags(write=False)
    return lmax_struct, m_offset, kernel, row_idx_arr


def reduce_structural_sums_to_l1(
    structural_sums: np.ndarray,
    coeffs: np.ndarray,
    *,
    kernel: np.ndarray,
) -> np.ndarray:
    """Contract batched periodic structural sums into local `l=1` coefficients."""

    sums = np.asarray(structural_sums, dtype=np.complex128)
    src_coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    p_count = int(kernel.shape[3])
    return np.asarray(
        np.einsum("npm,rcmp,c->nr", sums[:, :p_count, :], kernel, src_coeffs, optimize=True)
    )


__all__ = ["l1_projection_data", "reduce_structural_sums_to_l1"]
