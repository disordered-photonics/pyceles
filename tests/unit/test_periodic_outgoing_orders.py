"""Physical boundaries of the source-independent periodic outgoing transform."""

from typing import Any

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.postprocessing.farfield import periodic_outgoing_amplitudes


def test_grazing_outgoing_order_requires_a_limiting_prescription():
    kwargs: dict[str, Any] = dict(
        lattice=RectangularLattice2D(2 * np.pi, 2 * np.pi),
        positions=np.zeros((1, 3)),
        lmax=1,
        k=1,
        k_parallel=np.zeros(2),
        output_bmax=1.1,
    )
    with pytest.raises(ValueError, match="grazing Rayleigh-threshold"):
        periodic_outgoing_amplitudes(coeffs=np.ones((1, 6)), **kwargs)
    zero = periodic_outgoing_amplitudes(coeffs=np.zeros((1, 6)), **kwargs)
    np.testing.assert_array_equal(zero.up_amplitudes, 0)
    np.testing.assert_array_equal(zero.down_amplitudes, 0)


def test_inactive_outgoing_sources_do_not_enter_evanescent_phases():
    kwargs: dict[str, Any] = dict(
        lattice=RectangularLattice2D(600, 650),
        lmax=1,
        k=2 * np.pi / 550 * (1.33 + 0.08j),
        k_parallel=np.array([0.0013, -0.0007]),
        output_bmax=0.04,
    )
    coefficients = np.ones((1, 6), dtype=complex)
    reference = periodic_outgoing_amplitudes(
        positions=np.zeros((1, 3)), coeffs=coefficients, **kwargs
    )
    # A zero expansion is absent physically, even far outside the reference
    # slab where evaluating a growing evanescent phase alone would overflow.
    with np.errstate(over="raise", invalid="raise"):
        mixed = periodic_outgoing_amplitudes(
            positions=np.array([[0, 0, 0], [0, 0, 1e6]]),
            coeffs=np.r_[coefficients, np.zeros((1, 6))],
            **kwargs,
        )
    np.testing.assert_array_equal(mixed.up_amplitudes, reference.up_amplitudes)
    np.testing.assert_array_equal(mixed.down_amplitudes, reference.down_amplitudes)
