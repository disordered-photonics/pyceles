from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy import integrate, special

from pyceles.core.periodic.scalar import reciprocal_gamma_with_zero_mask
from pyceles.core.periodic.special import (
    kambe_integral,
    reduced_incomplete_gamma_int_or_halfint,
    shifted_delta_sequence,
    shifted_reciprocal_regime,
    upper_gamma_sequence,
    upper_incomplete_gamma_int_or_halfint,
)


def _quad_kambe(order: int, z: complex, eta: float) -> complex:
    def integrand(t: float) -> complex:
        return t**order * cmath.exp(-0.5 * z * z * t * t + 0.5 / (t * t))

    real = integrate.quad(lambda t: float(np.real(integrand(t))), eta, np.inf, limit=200)[0]
    imag = integrate.quad(lambda t: float(np.imag(integrand(t))), eta, np.inf, limit=200)[0]
    return complex(real, imag)


def test_upper_incomplete_gamma_matches_known_positive_real_values() -> None:
    z = 1.7

    assert upper_incomplete_gamma_int_or_halfint(0.0, z) == pytest.approx(special.exp1(z))
    assert upper_incomplete_gamma_int_or_halfint(1.0, z) == pytest.approx(math.exp(-z))
    assert upper_incomplete_gamma_int_or_halfint(0.5, z) == pytest.approx(
        math.sqrt(math.pi) * special.erfc(math.sqrt(z))
    )

    for order in (-1.5, -0.5, 0.5, 1.0, 1.5, 2.0):
        lhs = upper_incomplete_gamma_int_or_halfint(order + 1.0, z)
        rhs = order * upper_incomplete_gamma_int_or_halfint(order, z) + z**order * math.exp(-z)
        assert lhs == pytest.approx(rhs)


def test_upper_incomplete_gamma_keeps_complex_branch_conjugacy() -> None:
    order = -0.5
    above = upper_incomplete_gamma_int_or_halfint(order, complex(-1.2, 1e-12))
    below = upper_incomplete_gamma_int_or_halfint(order, complex(-1.2, -1e-12))

    assert above == pytest.approx(np.conj(below))


def test_reduced_incomplete_gamma_matches_definition() -> None:
    order = -1.5
    z = complex(0.8, -0.3)

    assert reduced_incomplete_gamma_int_or_halfint(order, z) == pytest.approx(
        upper_incomplete_gamma_int_or_halfint(order, z) / ((-z) ** order)
    )


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_kambe_integral_matches_direct_quadrature(order: int) -> None:
    z = 1.2
    eta = 0.85

    assert kambe_integral(order, z, eta) == pytest.approx(
        _quad_kambe(order, z, eta),
        rel=5e-11,
        abs=5e-11,
    )


def test_kambe_integral_accepts_complex_argument() -> None:
    z = complex(1.1, 0.2)
    eta = 0.9

    assert kambe_integral(0, z, eta) == pytest.approx(
        _quad_kambe(0, z, eta),
        rel=1e-10,
        abs=1e-10,
    )


def test_shifted_delta_sequence_returns_finite_recurrence_values() -> None:
    gamma = np.array([0.7 + 0.1j, 1.1 + 0.2j])

    delta = shifted_delta_sequence(3, gamma, z_offset=0.6, eta=1.2)

    assert delta.shape == (2, 4)
    assert np.all(np.isfinite(delta.real))
    assert np.all(np.isfinite(delta.imag))


def test_shifted_delta_sequence_stays_finite_for_large_propagating_height() -> None:
    # The lower-half-plane Faddeeva branch and its exponentially small
    # prefactor otherwise form a 0*inf product for a perfectly valid tall-cell
    # reciprocal offset.
    k = 2.0 * math.pi / 550.0
    delta = shifted_delta_sequence(3, np.asarray([k + 0.0j]), 10550.4, 2.67125e-3)

    assert np.all(np.isfinite(delta.real))
    assert np.all(np.isfinite(delta.imag))
    np.testing.assert_allclose(
        delta,
        shifted_delta_sequence(3, np.asarray([k + 0.0j]), -10550.4, 2.67125e-3),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_shifted_delta_sequence_rejects_same_plane_limit() -> None:
    with pytest.raises(ValueError, match="same-plane reciprocal formula"):
        shifted_delta_sequence(2, np.array([0.7 + 0.1j]), z_offset=0.0, eta=1.2)


def test_shifted_delta_sequence_rejects_regular_recurrence_near_singularity() -> None:
    with pytest.raises(ValueError, match="Rayleigh-threshold"):
        shifted_delta_sequence(2, np.array([1.0e-16 + 0.0j]), z_offset=0.5, eta=1.2)


def test_shifted_reciprocal_regime_classifies_scaled_offsets() -> None:
    assert shifted_reciprocal_regime(np.array([1.0 + 0.0j]), 0.0) == "same_plane"
    assert shifted_reciprocal_regime(np.array([1.0e-16 + 0.0j]), 0.5) == "rayleigh_limit"
    assert shifted_reciprocal_regime(np.array([0.7 + 0.1j]), 0.6) == "shifted"


def test_upper_gamma_sequence_matches_individual_branch_values() -> None:
    arguments = np.asarray([-6.7 + 0.0j, 74.0 + 0.0j, 1.3 - 4.4j], dtype=np.complex128)
    got = upper_gamma_sequence(12, arguments)
    expected = np.asarray(
        [
            [upper_incomplete_gamma_int_or_halfint(0.5 - n, value) for n in range(13)]
            for value in arguments
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)


def test_shifted_delta_sequence_uses_exact_near_plane_series_at_high_order() -> None:
    eta = 1.1588807813266307e-3
    gamma = np.asarray([0.006 + 0.0j, 0.02j], dtype=np.complex128)
    z_offset = 2.9574860172942863e-4
    max_order = 12

    got = shifted_delta_sequence(max_order, gamma, z_offset=z_offset, eta=eta)

    x = -(gamma * gamma) / (4.0 * eta * eta)
    q = 0.25 * (gamma * z_offset) ** 2
    gamma_table = upper_gamma_sequence(max_order + 30, x)
    expected = np.zeros_like(got)
    term = np.ones_like(gamma)
    for power in range(31):
        if power:
            term *= q / float(power)
        expected += term[:, None] * gamma_table[:, power : power + max_order + 1]

    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert np.max(np.abs(got)) < 1.0e4


def test_shifted_delta_sequence_requests_cached_series_table_lazily() -> None:
    eta = 1.1588807813266307e-3
    gamma = np.asarray([0.006 + 0.0j, 0.02j], dtype=np.complex128)
    calls: list[int] = []

    def provider(max_index: int) -> np.ndarray:
        calls.append(int(max_index))
        x = -(gamma * gamma) / (4.0 * eta * eta)
        return upper_gamma_sequence(max_index, x)

    near = shifted_delta_sequence(
        6,
        gamma,
        z_offset=2.9574860172942863e-4,
        eta=eta,
        upper_gamma_provider=provider,
    )
    assert calls == [22]
    assert np.all(np.isfinite(near))

    calls.clear()
    shifted_delta_sequence(
        3,
        np.asarray([0.7 + 0.1j], dtype=np.complex128),
        z_offset=0.6,
        eta=1.2,
        upper_gamma_provider=provider,
    )
    assert calls == []


def test_shifted_delta_series_is_accurate_at_its_dimensionless_boundary() -> None:
    eta = 1.0
    z_offset = 0.5
    gamma = np.asarray([16.0 + 0.0j, 16.0j], dtype=np.complex128)
    max_order = 20

    got = shifted_delta_sequence(max_order, gamma, z_offset=z_offset, eta=eta)

    x = -(gamma * gamma) / (4.0 * eta * eta)
    q = 0.25 * (gamma * z_offset) ** 2
    gamma_table = upper_gamma_sequence(max_order + 40, x)
    expected = np.zeros_like(got)
    term = np.ones_like(gamma)
    for power in range(41):
        if power:
            term *= q / float(power)
        expected += term[:, None] * gamma_table[:, power : power + max_order + 1]

    np.testing.assert_allclose(got, expected, rtol=5e-13, atol=5e-13)


def test_reciprocal_gamma_marks_exact_rayleigh_zero_before_regularization() -> None:
    gamma, zero = reciprocal_gamma_with_zero_mask(2.0, np.asarray([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(zero, np.asarray([False, True, False]))
    assert gamma[1] == pytest.approx(1.0e-10j)


def test_shifted_delta_series_exclusion_preserves_rayleigh_zero_policy() -> None:
    gamma = np.asarray([1.0e-10j], dtype=np.complex128)
    provider_calls: list[int] = []

    def provider(max_index: int) -> np.ndarray:
        provider_calls.append(int(max_index))
        raise AssertionError("the near-plane series must not own a Rayleigh-zero term")

    got = shifted_delta_sequence(
        0,
        gamma,
        z_offset=1.0e-3,
        eta=1.0e-4,
        upper_gamma_provider=provider,
        series_exclusion=np.asarray([True]),
    )

    assert provider_calls == []
    assert np.all(np.isfinite(got))
