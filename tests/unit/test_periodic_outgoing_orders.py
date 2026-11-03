"""Physical boundaries of the source-independent periodic outgoing transform."""

from typing import Any

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import rayleigh_report
from pyceles.postprocessing.farfield import periodic_outgoing_amplitudes
from pyceles.postprocessing.farfield.orders import enumerate_diffraction_orders_rectangular


@pytest.mark.parametrize("imaginary_zero", [0.0, -0.0], ids=["positive-zero", "negative-zero"])
@pytest.mark.parametrize("length_scale", [1e-3, 1.0, 1e3])
def test_evanescent_orders_near_grazing_preserve_dispersion_and_length_units(
    length_scale, imaginary_zero
):
    k = 0.01 / length_scale
    transverse = k * np.sqrt(1 + 2e-9)
    lattice = RectangularLattice2D(2 * np.pi / transverse, 2 * np.pi / transverse)
    outgoing = periodic_outgoing_amplitudes(
        lattice=lattice,
        k=complex(k, imaginary_zero),
        k_parallel=np.zeros(2),
        output_bmax=1.01 * k,
        positions=np.zeros((1, 3)),
        coeffs=np.ones((1, 6)),
        lmax=1,
    )
    assert np.all(np.isfinite(outgoing.up_amplitudes))
    assert np.all(np.isfinite(outgoing.down_amplitudes))
    idx = np.flatnonzero((outgoing.orders.m == 1) & (outgoing.orders.n == 0))[0]
    table_order = next(
        o
        for o in lattice.diffraction_orders(k_parallel=np.zeros(2), k=k, max_order=1)
        if (o.m, o.n) == (1, 0)
    )
    report = rayleigh_report(lattice, k=k, k_parallel=np.zeros(2), at_threshold=1e-3)
    assert report.warning_level == "at"  # A warning threshold is not a dispersion clamp.
    assert report.min_clearance > 0
    for kz, propagating, kp in (
        (
            outgoing.orders.kz[idx],
            outgoing.orders.propagating[idx],
            outgoing.orders.k_parallel[idx],
        ),
        (table_order.kz, table_order.propagating, table_order.k_parallel),
        *[(o.kz, o.propagating, o.k_parallel) for o in report.nearest_orders],
    ):
        assert not propagating
        assert kz.real == 0 and kz.imag > 0
        np.testing.assert_allclose(
            kz * kz + np.dot(kp, kp), k * k, rtol=32 * np.finfo(float).eps, atol=0
        )
    propagating_only = enumerate_diffraction_orders_rectangular(
        lattice=lattice,
        k=complex(k, imaginary_zero),
        k_parallel_incident=np.zeros(2),
    )
    assert np.all(propagating_only.propagating)
    assert not np.any((propagating_only.m == 1) & (propagating_only.n == 0))


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
