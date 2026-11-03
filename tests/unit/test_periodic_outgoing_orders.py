"""Physical boundaries of the source-independent periodic outgoing transform."""

from typing import Any

import numpy as np
import pytest

from pyceles.core.conversions import transformation_coefficients
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import rayleigh_report
from pyceles.core.spherical import spherical_functions_trigon
from pyceles.postprocessing.farfield import periodic_outgoing_amplitudes
from pyceles.postprocessing.farfield.orders import enumerate_diffraction_orders_rectangular


def _scalar_outgoing_reference(outgoing, positions, coefficients, *, lmax, k):
    """Independently contract one order and one SVWF mode at a time."""
    branches = []
    for sign, reference_z in zip((1, -1), outgoing.reference_z, strict=True):
        amplitudes = np.zeros((len(outgoing.orders.m), 2), dtype=complex)
        for i, (kp, kz) in enumerate(
            zip(outgoing.orders.k_parallel, outgoing.orders.kz, strict=True)
        ):
            alpha = np.arctan2(kp[1], kp[0])
            pi, tau = spherical_functions_trigon(
                np.asarray(sign * kz / k), np.asarray(np.linalg.norm(kp) / k), lmax
            )
            phase = np.exp(
                -1j * (positions[:, :2] @ kp + sign * kz * (positions[:, 2] - reference_z))
            )
            for svwf_pol, l, m, mode in iter_modes(lmax):
                for plane_pol in (1, 2):
                    conversion = transformation_coefficients(
                        pi, tau, svwf_pol, l, m, pol=plane_pol, dagger=False
                    ) * np.exp(1j * m * alpha)
                    amplitudes[i, plane_pol - 1] += conversion * np.sum(
                        coefficients[:, mode] * phase
                    )
            amplitudes[i] *= 2 * np.pi / (outgoing.unit_cell_area * k * kz)
        branches.append(amplitudes)
    return branches


@pytest.mark.parametrize("k", [0.015 + 0j, 0.015 + 0.002j])
@pytest.mark.parametrize("lmax,bmax", [(1, 0.04), (3, 0.04), (6, 0.11)])
def test_outgoing_order_batches_match_scalar_full_mode_contraction(k, lmax, bmax):
    rng = np.random.default_rng(218)
    positions = rng.uniform(-100, 100, (4, 3))
    coefficients = rng.standard_normal((4, n_modes(lmax))) + 1j * rng.standard_normal(
        (4, n_modes(lmax))
    )
    outgoing = periodic_outgoing_amplitudes(
        lattice=RectangularLattice2D(600, 650),
        positions=positions,
        coeffs=coefficients,
        lmax=lmax,
        k=k,
        k_parallel=np.array([0.0013, -0.0007]),
        output_bmax=bmax,
        reference_z=(150, -150),
    )
    reference = _scalar_outgoing_reference(outgoing, positions, coefficients, lmax=lmax, k=k)
    for actual, expected in zip(
        (outgoing.up_amplitudes, outgoing.down_amplitudes), reference, strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)


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
