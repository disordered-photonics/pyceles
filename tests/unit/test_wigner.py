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
