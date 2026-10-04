from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.indexing import index_vswf, n_modes
from pyceles.core.operators.mlfmm_cupy import _leaf_rect_pair_tables_and_ab
from pyceles.core.operators.mode_metadata import mode_pair_p_range_tables
from pyceles.core.translation import _translation_ab5_compact_tables, translation_ab5_table


@pytest.mark.parametrize("lmax", (1, 2, 3, 4))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_square_compact_tables_reconstruct_dense_reference(lmax, dtype) -> None:
    reference = translation_ab5_table(lmax, dtype=dtype)
    real, imag = _translation_ab5_compact_tables(lmax, dtype=dtype)
    offsets, starts, counts = mode_pair_p_range_tables(lmax)
    packed = real + 1j * imag
    actual = np.zeros_like(reference)
    cursor = 0
    for out_mode in range(n_modes(lmax)):
        for in_mode in range(n_modes(lmax)):
            count = int(counts[out_mode, in_mode])
            start = int(starts[out_mode, in_mode])
            assert int(offsets[out_mode, in_mode]) == cursor
            orders = start + 2 * np.arange(count)
            actual[out_mode, in_mode, orders] = packed[cursor : cursor + count]
            cursor += count
    assert cursor == packed.size
    np.testing.assert_array_equal(actual, reference)
    # This pair has no admissible p: cross polarization and maximal |delta m|.
    left = index_vswf(lmax, -lmax, 1, lmax)
    right = index_vswf(lmax, lmax, 2, lmax)
    assert counts[left, right] == 0


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("reverse", (False, True))
def test_rectangular_compact_tables_reconstruct_dense_reference(dtype, reverse) -> None:
    full_order = 4
    # Unequal, reordered subsets catch swapped modes and square-only indexing.
    out_modes = np.array([47, 0, 31, 6, 23], dtype=np.int32)
    in_modes = np.array([24, 7, 46], dtype=np.int32)
    if reverse:
        out_modes, in_modes = in_modes, out_modes
    real, imag, offsets, starts, counts, _, _ = _leaf_rect_pair_tables_and_ab(
        full_order=full_order,
        out_mode_indices=out_modes,
        in_mode_indices=in_modes,
        out_dtype=np.dtype(dtype),
    )
    reference = translation_ab5_table(full_order, dtype=dtype)[
        out_modes[:, None], in_modes[None, :], :
    ]
    actual = np.zeros_like(reference)
    packed = real + 1j * imag
    cursor = 0
    for i in range(out_modes.size):
        for j in range(in_modes.size):
            pair = i * in_modes.size + j
            count = int(counts[pair])
            assert int(offsets[pair]) == cursor
            orders = int(starts[pair]) + 2 * np.arange(count)
            actual[i, j, orders] = packed[cursor : cursor + count]
            cursor += count
    assert cursor == packed.size
    np.testing.assert_array_equal(actual, reference)
