import math

import pytest

from pyceles.core.wigner import wigner_3j

pytestmark = pytest.mark.reference


def test_wigner_known_values():
    # (1 1 0; 0 0 0) = -1/sqrt(3)
    v = wigner_3j(1, 1, 0, 0, 0, 0)
    target = -1.0 / math.sqrt(3.0)
    # Host-float64 path should achieve ~1e-12 easily.
    assert abs(v - target) < 1e-12

    # selection rule (m sum != 0) => 0
    assert wigner_3j(1, 1, 1, 1, 0, 0) == 0.0


@pytest.mark.parametrize("l1", range(1, 7))
@pytest.mark.parametrize("l2", range(1, 7))
def test_wigner_zero_m_odd_perimeter_is_exact_zero(l1: int, l2: int) -> None:
    for l3 in range(abs(l1 - l2), l1 + l2 + 1):
        if (l1 + l2 + l3) % 2:
            assert wigner_3j(l1, l2, l3, 0, 0, 0) == 0.0


def test_wigner_odd_perimeter_with_nonzero_m_is_not_removed() -> None:
    assert wigner_3j(1, 1, 1, 1, -1, 0) == pytest.approx(1.0 / math.sqrt(6.0))
