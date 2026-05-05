from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy import integrate, special

from pyceles.core.periodic.special import (
    kambe_integral,
    reduced_incomplete_gamma_int_or_halfint,
    shifted_delta_sequence,
    shifted_reciprocal_regime,
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
