from __future__ import annotations

import numpy as np
import pytest

from pyceles.core import clear_caches as clear_core_caches
from pyceles.core.periodic.structural import rectangular_projection_data
from pyceles.core.spherical import _legendre_backend_tables, _legendre_scalar_tables
from pyceles.core.svwf import mode_indices_by_l
from pyceles.core.svwf_rotation import _scalar_rotation_block_cached
from pyceles.core.translation import (
    _rectangular_radial_lut_from_base,
    _translation_ab5_table_cached,
    _translation_mode_pair_tables,
)
from pyceles.core.wigner import wigner_3j
from pyceles.postprocessing.nearfield import clear_caches as clear_nearfield_caches


def test_core_clear_caches_empties_process_global_tables() -> None:
    _translation_ab5_table_cached(4, np.dtype(np.complex128).str)
    _translation_mode_pair_tables(4)
    _rectangular_radial_lut_from_base(4, 1.0, 5.0, 0.5, np.dtype(np.complex128).str)
    _legendre_scalar_tables(4)
    _legendre_backend_tables(4, np.dtype(np.float64).name)
    _scalar_rotation_block_cached(2, 0.1, 0.2, 0.3)
    wigner_3j(1, 1, 0, 0, 0, 0)
    rectangular_projection_data(3, 1)

    assert _translation_ab5_table_cached.cache_info().currsize > 0
    assert _translation_mode_pair_tables.cache_info().currsize > 0
    assert _rectangular_radial_lut_from_base.cache_info().currsize > 0
    assert _legendre_scalar_tables.cache_info().currsize > 0
    assert _legendre_backend_tables.cache_info().currsize > 0
    assert _scalar_rotation_block_cached.cache_info().currsize > 0
    assert wigner_3j.cache_info().currsize > 0

    assert rectangular_projection_data.cache_info().currsize > 0

    clear_core_caches()

    assert _translation_ab5_table_cached.cache_info().currsize == 0
    assert _translation_mode_pair_tables.cache_info().currsize == 0
    assert _rectangular_radial_lut_from_base.cache_info().currsize == 0
    assert _legendre_scalar_tables.cache_info().currsize == 0
    assert _legendre_backend_tables.cache_info().currsize == 0
    assert _scalar_rotation_block_cached.cache_info().currsize == 0
    assert wigner_3j.cache_info().currsize == 0
    assert rectangular_projection_data.cache_info().currsize == 0


def test_nearfield_clear_caches_empties_mode_index_table() -> None:
    mode_indices_by_l(4)
    assert mode_indices_by_l.cache_info().currsize > 0

    clear_nearfield_caches()

    assert mode_indices_by_l.cache_info().currsize == 0


def test_periodic_rectangular_projection_cache_is_read_only() -> None:
    _degrees, columns, _indptr, _inputs, kernel = rectangular_projection_data(3, 1)

    assert not kernel.flags.writeable
    assert not columns.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        kernel[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        columns[0] = 0
