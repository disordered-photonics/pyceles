from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.operators.mode_metadata import (
    mode_m_table,
    mode_metadata_tables,
    mode_pair_p_range_tables,
    mode_tau_l_tables,
)


def test_mode_metadata_tables_match_flattened_indexing() -> None:
    lmax = 4

    mode_tau, mode_l, mode_m = mode_metadata_tables(lmax)

    assert mode_tau.shape == (n_modes(lmax),)
    assert mode_l.shape == mode_tau.shape
    assert mode_m.shape == mode_tau.shape
    assert mode_tau.dtype == np.int32
    assert mode_l.dtype == np.int32
    assert mode_m.dtype == np.int32
    for tau, ell, m, idx in iter_modes(lmax):
        assert mode_tau[idx] == tau
        assert mode_l[idx] == ell
        assert mode_m[idx] == m


def test_mode_metadata_tables_are_read_only_and_cached() -> None:
    first_tau, first_l, first_m = mode_metadata_tables(3)
    second_tau, second_l, second_m = mode_metadata_tables(3)

    assert first_tau is second_tau
    assert first_l is second_l
    assert first_m is second_m
    assert mode_m_table(3) is first_m
    tau_only, l_only = mode_tau_l_tables(3)
    assert tau_only is first_tau
    assert l_only is first_l
    with pytest.raises(ValueError):
        first_tau[0] = 99


def test_mode_pair_p_range_tables_match_translation_formula() -> None:
    lmax = 3
    mode_tau, mode_l, mode_m = mode_metadata_tables(lmax)

    pair_offset, pair_pmin, pair_pcount = mode_pair_p_range_tables(lmax)

    assert pair_offset.shape == (n_modes(lmax), n_modes(lmax))
    assert pair_pmin.shape == pair_offset.shape
    assert pair_pcount.shape == pair_offset.shape
    offset = 0
    for n1 in range(n_modes(lmax)):
        for n2 in range(n_modes(lmax)):
            expected_pmin = max(
                abs(int(mode_m[n1]) - int(mode_m[n2])),
                abs(int(mode_l[n1]) - int(mode_l[n2])) + abs(int(mode_tau[n1]) - int(mode_tau[n2])),
            )
            p_max = int(mode_l[n1]) + int(mode_l[n2])
            cross = int(mode_tau[n1] != mode_tau[n2])
            expected = [p for p in range(expected_pmin, p_max + 1) if (p_max + p + cross) % 2 == 0]
            start = int(pair_pmin[n1, n2])
            count = int(pair_pcount[n1, n2])
            assert pair_offset[n1, n2] == offset
            assert list(range(start, start + 2 * count, 2)) == expected
            assert (p_max + start + cross) % 2 == 0
            assert start in (expected_pmin, expected_pmin + 1)
            offset += len(expected)


def test_mode_pair_p_range_tables_are_read_only_and_cached() -> None:
    first_offset, first_pmin, first_pcount = mode_pair_p_range_tables(2)
    second_offset, second_pmin, second_pcount = mode_pair_p_range_tables(2)

    assert first_offset is second_offset
    assert first_pmin is second_pmin
    assert first_pcount is second_pcount
    with pytest.raises(ValueError):
        first_offset[0, 0] = 99
