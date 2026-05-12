import math

import numpy as np
import pytest
from scipy.special import lpmv

from pyceles.core.spherical import legendre_normalized_trigon, spherical_functions_trigon

pytestmark = pytest.mark.reference


def _celes_legendre_norm(l: int, m: int) -> float:
    """Normalization inferred from CELES legendre_normalized_trigon.m.

    CELES uses (Doicu/Wriedt/Eremin) normalized associated Legendre functions:

        \bar{P}_l^m(x) = sqrt((2l+1)/2 * (l-m)!/(l+m)!) * P_l^m(x)

    and omits the Condon-Shortley phase (-1)^m.

    SciPy's lpmv includes the Condon-Shortley phase, so we multiply by (-1)^m.
    """

    return math.sqrt((2 * l + 1) / 2 * math.factorial(l - m) / math.factorial(l + m))


def test_legendre_normalized_matches_scipy_lpmv():
    lmax = 6
    theta = np.linspace(0.3, 2.8, 9)
    ct = np.cos(theta)
    st = np.sin(theta)

    P = legendre_normalized_trigon(ct, st, lmax, xp=np)

    for l in range(0, lmax + 1):
        for m in range(0, l + 1):
            norm = _celes_legendre_norm(l, m)
            expected = norm * ((-1) ** m) * lpmv(m, l, ct)
            got = P[l, m]
            assert np.allclose(got, expected, rtol=5e-12, atol=5e-12)


def test_spherical_functions_pi_tau_against_scipy():
    lmax = 6
    theta = np.linspace(0.35, 2.75, 7)
    ct = np.cos(theta)
    st = np.sin(theta)

    PI, TAU = spherical_functions_trigon(ct, st, lmax, xp=np)
    P = legendre_normalized_trigon(ct, st, lmax, xp=np)

    # PI definition in CELES: pi_l^0 = 0, and for m>=1: pi_l^m = P_l^m/sin(theta)
    for l in range(1, lmax + 1):
        # m=0 special-case
        assert np.allclose(PI[l, 0], 0.0, atol=0.0)
        for m in range(1, l + 1):
            assert np.allclose(PI[l, m], P[l, m] / st, rtol=5e-12, atol=5e-12)

    # TAU should equal d/dtheta P_l^m(cos theta).
    # Compute via SciPy baseline: tau = -sin(theta) * d/dx P(x), x=cos(theta)
    eps = 1e-7
    x = ct
    x_plus = x + eps
    x_minus = x - eps

    # Keep within [-1,1] (we chose theta away from endpoints)
    assert np.all(np.abs(x_plus) < 1.0)
    assert np.all(np.abs(x_minus) < 1.0)

    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            norm = _celes_legendre_norm(l, m)
            Pp = norm * ((-1) ** m) * lpmv(m, l, x_plus)
            Pm = norm * ((-1) ** m) * lpmv(m, l, x_minus)
            dPdx = (Pp - Pm) / (2 * eps)
            expected_tau = -st * dPdx
            got_tau = TAU[l, m]
            assert np.allclose(got_tau, expected_tau, rtol=5e-7, atol=5e-7)


def test_spherical_functions_against_smuthi_prototype_values():
    """Ported from SMUTHI's spherical-function prototype test."""
    lmax = 3
    k0 = 2 * 3.14 / 550
    kp = k0 * np.array([0.01, 0.2, 0.7, 0.99, 1.2, 2 - 0.5j], dtype=np.complex128)
    kz = np.sqrt(k0**2 - kp**2 + 0j)
    kz[kz.imag < 0] = -kz[kz.imag < 0]
    ct = kz / k0
    st = kp / k0

    PI, TAU, P = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

    np.testing.assert_allclose(
        P[3, 0],
        np.array(
            [
                1.870267465826245,
                1.649727250184103,
                -0.300608757357466,
                -0.382739631607606,
                -3.226515147957620j,
                -25.338383323084539 - 22.141864985871653j,
            ],
            dtype=np.complex128,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        P[2, 1],
        np.array(
            [
                0.019363948460993,
                0.379473319220206,
                0.968052168015753,
                0.270444009917026,
                1.541427909439815j,
                3.906499971729346 + 6.239600710712296j,
            ],
            dtype=np.complex128,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        PI[2, 1],
        np.array(
            [
                1.936394846099318,
                1.897366596101028,
                1.382931668593933,
                0.273175767592955,
                1.284523257866512j,
                1.104282256024128 + 3.395870919362181j,
            ],
            dtype=np.complex128,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        TAU[3, 2],
        np.array(
            [
                0.051227068616724,
                0.963213372000203,
                0.950404683542753,
                -2.384713931794872,
                -7.131877733107878,
                -39.706934218093430 + 42.588889121019569j,
            ],
            dtype=np.complex128,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
