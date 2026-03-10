from __future__ import annotations

import numpy as np

from pyceles.core import clear_caches as clear_core_caches
from pyceles.core.spherical import _legendre_scalar_tables
from pyceles.core.svwf_rotation import _scalar_rotation_block_cached
from pyceles.core.translation import _translation_ab5_table_cached, _translation_mode_pair_tables
from pyceles.core.wigner import wigner_3j
from pyceles.postprocessing.nearfield import clear_caches as clear_nearfield_caches
from pyceles.postprocessing.nearfield.common import mode_indices_by_l


def test_core_clear_caches_empties_process_global_tables() -> None:
    _translation_ab5_table_cached(4, np.dtype(np.complex128).str)
    _translation_mode_pair_tables(4)
    _legendre_scalar_tables(4)
    _scalar_rotation_block_cached(2, 0.1, 0.2, 0.3)
    wigner_3j(1, 1, 0, 0, 0, 0)

    assert _translation_ab5_table_cached.cache_info().currsize > 0
    assert _translation_mode_pair_tables.cache_info().currsize > 0
    assert _legendre_scalar_tables.cache_info().currsize > 0
    assert _scalar_rotation_block_cached.cache_info().currsize > 0
    assert wigner_3j.cache_info().currsize > 0

    clear_core_caches()

    assert _translation_ab5_table_cached.cache_info().currsize == 0
    assert _translation_mode_pair_tables.cache_info().currsize == 0
    assert _legendre_scalar_tables.cache_info().currsize == 0
    assert _scalar_rotation_block_cached.cache_info().currsize == 0
    assert wigner_3j.cache_info().currsize == 0


def test_nearfield_clear_caches_empties_mode_index_table() -> None:
    mode_indices_by_l(4)
    assert mode_indices_by_l.cache_info().currsize > 0

    clear_nearfield_caches()

    assert mode_indices_by_l.cache_info().currsize == 0
