from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy import integrate, special

from pyceles.core.periodic.scalar import (
    reciprocal_gamma_with_zero_mask,
    scaled_real_integral_sequences,
)
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


@pytest.mark.parametrize("k", [1.1 + 0.1j, 1.3 + 0.8j])
def test_scaled_radial_sequence_matches_scalar_kambe(k: complex) -> None:
    radii = np.asarray([0.2, 0.7, 1.3])
    eta = 0.9
    actual = scaled_real_integral_sequences(8, eta, k, radii)
    expected = np.asarray(
        [
            [
                np.sqrt(2.0)
                * (k * radius) ** (degree + 1)
                * kambe_integral(2 * degree, k * radius, np.sqrt(2.0) * eta / k)
                for degree in range(9)
            ]
            for radius in radii
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)


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


def test_upper_gamma_sequence_mixed_complex_routes() -> None:
    # A single Ewald shell can contain different reciprocal regimes. Each
    # argument must select its own stable sequence, regardless of its neighbors.
    arguments = np.asarray([3.0 + 0.5j, complex(-1.2, -0.0), 0.7 + 0.3j])
    combined = upper_gamma_sequence(16, arguments)
    for index, argument in enumerate(arguments):
        np.testing.assert_allclose(
            combined[index], upper_gamma_sequence(16, [argument])[0], rtol=1e-13, atol=1e-14
        )
    np.testing.assert_allclose(
        combined[0, 0],
        math.sqrt(math.pi) * special.erfc(np.sqrt(arguments[0])),
        rtol=1e-13,
        atol=1e-14,
    )


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


def test_shifted_delta_sequence_matches_complex_treams_reference() -> None:
    gamma = np.asarray([1.1 + 0.5j, 0.8 + 1.7j])
    expected = np.asarray(
        [
            [
                -0.7425786056416375 + 0.4839875594993020j,
                -0.30031023048789474 - 0.46581559671760486j,
                0.29047352509224080 - 0.11522747477948687j,
                0.00734189880039784 + 0.16747768793045767j,
                -0.08166218806534035 - 0.03436698221270592j,
                0.03765948480180489 - 0.02844007518958043j,
                0.00153568630219542 + 0.02561618198391143j,
            ],
            [
                -0.018700964384003215 - 0.0015831657150971335j,
                -0.0030576370992868394 - 0.0032097037284871252j,
                -0.0000028768597376933273 - 0.0010623365280800290j,
                0.00018173340420014157 - 0.00018330855129056460j,
                0.00006352809031128087 + 0.0000008574183637547930j,
                0.000010660235943723534 + 0.000011694366864730096j,
                -0.0000003599249584667997 + 0.000003967231597861686j,
            ],
        ],
        dtype=np.complex128,
    )

    actual = shifted_delta_sequence(6, gamma, z_offset=2.2, eta=0.5)

    # Frozen values from TREAMS' compiled Kambe integral at complex arguments.
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(
        actual,
        shifted_delta_sequence(6, gamma, z_offset=-2.2, eta=0.5),
        rtol=1e-13,
        atol=1e-14,
    )


def test_shifted_delta_complex_batch_preserves_real_argument_channels() -> None:
    gamma = np.asarray([1.1 + 0.0j, 1.1 + 0.5j, 2.0j])

    batched = shifted_delta_sequence(6, gamma, z_offset=0.4, eta=1.3)
    separate = np.asarray(
        [shifted_delta_sequence(6, [value], z_offset=0.4, eta=1.3)[0] for value in gamma]
    )

    np.testing.assert_allclose(batched, separate, rtol=3e-11, atol=1e-12)


def test_shifted_delta_near_plane_uses_stable_complex_gamma_series() -> None:
    # Direct upper-incomplete-gamma contour integral references for a complex
    # evanescent root. The dimensionless height is just above the former series
    # crossover, where the order-12 forward recurrence loses accuracy.
    gamma = 0.055265064000928434 + 9.952037692216555j
    expected = np.asarray(
        [
            8.392349480335680e-8 + 1.4242259453421969e-8j,
            1.4386156323019380e-12 + 3.0721225292740895e-13j,
            2.5605348908887340e-17 + 6.629169063017819e-18j,
            4.6788899715590250e-22 + 1.4304046054405500e-22j,
        ],
        dtype=np.complex128,
    )

    actual = shifted_delta_sequence(12, [gamma], z_offset=0.4, eta=1.3)[0]

    np.testing.assert_allclose(actual[[0, 4, 8, 12]], expected, rtol=3e-12, atol=1e-30)


def test_shifted_delta_complex_series_covers_recurrence_crossover() -> None:
    # Direct contour-integral references at |gamma*z|=9.6 and eta*|z|=1.
    # The raw order recurrence loses about 2.5e-5 relative accuracy by n=12.
    gamma = 0.1 + 24.0j
    expected = np.asarray(
        [
            7.559259787570088e-12 + 1.5033242536907253e-12j,
            2.2759731453722017e-17 + 5.291833995358668e-18j,
            6.984817967155483e-23 + 1.8646207182807603e-23j,
            2.1752332804790307e-28 + 6.573843869096256e-29j,
        ],
        dtype=np.complex128,
    )

    actual = shifted_delta_sequence(12, [gamma], z_offset=0.4, eta=2.5)[0]

    np.testing.assert_allclose(actual[[0, 4, 8, 12]], expected, rtol=3e-12, atol=1e-35)


def test_shifted_delta_sequence_matches_tall_propagating_reference() -> None:
    # High-precision reference for a tall-cell propagating reciprocal order.
    # Directly forming the original lower-half-plane Faddeeva product in
    # binary64 produces 0*inf even though these values are finite.
    k = 2.0 * math.pi / 550.0
    delta = shifted_delta_sequence(3, np.asarray([k + 0.0j]), 10550.4, 2.67125e-3)
    expected = np.asarray(
        [
            0.7289288169279475 + 1.6156284948718302j,
            -0.026809295123869023 + 0.012095644412897245j,
            -0.00019702080574782977 - 0.0004465313518843544j,
            7.463360361976309e-06 - 3.1461266502750974e-06j,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(delta[0], expected, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(
        delta,
        shifted_delta_sequence(3, np.asarray([k + 0.0j]), -10550.4, 2.67125e-3),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_shifted_delta_sequence_matches_evanescent_crossover_references() -> None:
    # The auxiliary Faddeeva argument moves through zero as eta*|z| crosses
    # |gamma|/(2*eta).  Both sides must be evaluated without ever visiting the
    # exponentially growing lower-half-plane branch.
    k = 2.0 * math.pi / 550.0
    gamma = 0.5j * k
    below = shifted_delta_sequence(3, np.asarray([gamma]), 275.0, 2.67125e-3)[0]
    above = shifted_delta_sequence(3, np.asarray([gamma]), 10550.4, 2.67125e-3)[0]
    expected_below = np.asarray(
        [
            0.16299582348779956,
            0.09088696568255164,
            0.05608578922986181,
            0.03718231201638018,
        ],
        dtype=np.complex128,
    )
    expected_above = np.asarray(
        [
            1.192239514709785e-26,
            3.9567389544870187e-28,
            1.3349306678422327e-29,
            4.578524664187957e-31,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(below, expected_below, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(above, expected_above, rtol=2.0e-14, atol=1.0e-40)


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
        eta=2.0,
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
