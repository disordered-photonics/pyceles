from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.sources.common import _validated_int


@pytest.mark.parametrize("value", (0, 1, -3, 2.0, np.int64(7)))
def test_source_integer_parameter_preserves_exact_integers(value) -> None:
    assert _validated_int("order", value) == int(value)


@pytest.mark.parametrize("value", (2.000001, -2.000001, 100000.1, np.inf, np.nan))
def test_source_integer_parameter_does_not_round_nearby_nonintegers(value: float) -> None:
    with pytest.raises(ValueError, match="integer"):
        _validated_int("order", value)


def test_source_integer_parameter_still_checks_lower_bound() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        _validated_int("order", -1, minimum=0)
