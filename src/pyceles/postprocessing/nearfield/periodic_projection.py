"""Backend-neutral projection data for periodic interior local fields."""

from __future__ import annotations

from functools import cache

import numpy as np

from pyceles._arrays import expose_read_only_view

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.translation import translation_ab5_table


def _l1_required_structural_order(lmax: int) -> int:
    """Return the exact scalar-translation degree needed for local ``l=1`` output.

    SVWF translation coefficients obey the Wigner-3j triangle rule
    ``p <= l_src + l_dst``.  Point-centered field reconstruction needs only
    ``l_dst = 1``, hence sources truncated at ``lmax`` require exactly
    ``p <= lmax + 1``.
    """
    lmax_i = int(lmax)
    if lmax_i < 1:
        raise ValueError(f"`lmax` must be >= 1. Got {lmax!r}.")
    return lmax_i + 1


def _ewald_capacity_lmax(structural_order: int) -> int:
    """Return the legacy Ewald capacity whose ``2*lmax`` covers ``order``.

    Generic periodic coupling historically parameterizes scalar structural
    tables by a particle-like ``lmax_struct`` and therefore allocates through
    ``2*lmax_struct``.  Odd requested orders need the next even capacity; this
    helper makes that storage convention explicit rather than looking like a
    numerical rounding operation.
    """
    order = int(structural_order)
    if order < 0:
        raise ValueError(f"`structural_order` must be >= 0. Got {structural_order!r}.")
    return (order + 1) // 2


@cache
def l1_projection_data(lmax: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Precompute the dense reference contraction tensor for local ``l=1`` fields.

    The returned kernel contracts periodic structural sums directly into the
    six destination ``l=1`` modes, avoiding full translation-block construction.
    Its rectangular storage follows the historical even Ewald capacity; the
    production compact helper below removes any unused capacity degree.
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
    required_order = _l1_required_structural_order(lmax_i)
    lmax_struct = _ewald_capacity_lmax(required_order)
    # This dense helper is retained as a reference representation.  Its
    # historical Ewald-shaped storage has an even maximum order, so an odd
    # required order (for example p<=5 at lmax=4) owns one unused capacity row.
    # The production compact path below evaluates only ``required_order``.
    structural_order = 2 * lmax_struct
    p_count = required_order + 1
    m_offset = structural_order
    kernel = np.zeros(
        (ab5_l1.shape[0], nm, 2 * structural_order + 1, p_count),
        dtype=np.complex128,
    )
    for row in range(ab5_l1.shape[0]):
        dm_idx = m_src - int(m_dst_arr[row]) + m_offset
        for col in range(nm):
            kernel[row, col, int(dm_idx[col]), :p_count] = ab5_l1[row, col, :p_count]
    return (
        lmax_struct,
        m_offset,
        expose_read_only_view(kernel),
        expose_read_only_view(row_idx_arr),
    )


@cache
def l1_compact_projection_data(
    lmax: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact structural channels needed for point-local ``l=1`` fields.

    A destination mode with ``l=1`` coupled to source modes through ``lmax`` can
    require translation degree at most ``lmax + 1``.  The ordinary periodic
    coupling tables round structural order up to ``2*lmax_struct``; for even
    particle ``lmax`` that computes one entire unused degree.  This compact
    representation keeps only valid ``(p,m)`` channels through the exact limit.
    """
    lmax_i = int(lmax)
    if lmax_i < 1:
        raise ValueError(f"`lmax` must be >= 1. Got {lmax!r}.")
    lmax_struct, dense_offset, dense_kernel, _row_idx = l1_projection_data(lmax_i)
    structural_order = _l1_required_structural_order(lmax_i)
    degrees: list[int] = []
    orders: list[int] = []
    for degree in range(structural_order + 1):
        for order_m in range(-degree, degree + 1):
            degrees.append(degree)
            orders.append(order_m)
    degree_array = np.asarray(degrees, dtype=np.int32)
    order_array = np.asarray(orders, dtype=np.int32)
    compact_kernel = np.ascontiguousarray(
        dense_kernel[:, :, order_array + int(dense_offset), degree_array],
        dtype=np.complex128,
    )
    dense_order_columns = order_array + int(structural_order)
    return (
        int(lmax_struct),
        int(structural_order),
        expose_read_only_view(degree_array),
        expose_read_only_view(dense_order_columns),
        expose_read_only_view(compact_kernel),
    )


def reduce_compact_structural_sums_to_l1(
    structural_sums: np.ndarray,
    coeffs: np.ndarray,
    *,
    degree_indices: np.ndarray,
    order_indices: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Contract exact valid structural channels into local ``l=1`` coefficients."""
    sums = np.asarray(structural_sums)
    compact = sums[:, degree_indices, order_indices]
    src_coeffs = np.asarray(coeffs).reshape(-1)
    dtype = np.result_type(compact.dtype, kernel.dtype, src_coeffs.dtype, np.complex64)
    return np.asarray(
        np.einsum(
            "nk,rck,c->nr",
            np.asarray(compact, dtype=dtype),
            np.asarray(kernel, dtype=dtype),
            np.asarray(src_coeffs, dtype=dtype),
            optimize=True,
        )
    )


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


__all__ = [
    "l1_compact_projection_data",
    "l1_projection_data",
    "reduce_compact_structural_sums_to_l1",
    "reduce_structural_sums_to_l1",
]
