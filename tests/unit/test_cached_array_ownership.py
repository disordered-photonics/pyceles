from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.operators.mode_metadata import mode_metadata_tables, mode_pair_p_range_tables
from pyceles.core.periodic.rayleigh import valid_structural_indices
from pyceles.core.periodic.structural import rectangular_projection_data
from pyceles.core.svwf import mode_indices_by_l
from pyceles.core.translation import translation_ab5_table


def _assert_sealed(array: np.ndarray) -> None:
    assert not array.flags.owndata
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def test_process_global_cached_tables_are_not_reopenable() -> None:
    arrays = [translation_ab5_table(1)]
    arrays.extend(mode_metadata_tables(1))
    arrays.extend(mode_pair_p_range_tables(1))
    arrays.extend(valid_structural_indices(1))
    arrays.extend(rectangular_projection_data(3, 1))
    arrays.extend(value for group in mode_indices_by_l(1) for value in group)

    for array in arrays:
        _assert_sealed(array)
