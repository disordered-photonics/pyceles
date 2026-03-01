import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pyceles.core.translation import spherical_bessel_jy


def _dx_xj(n: int, z: np.ndarray) -> np.ndarray:
    jn = spherical_jn(n, z)
    djn = spherical_jn(n, z, derivative=True)
    return jn + z * djn


def _dx_xh(n: int, z: np.ndarray) -> np.ndarray:
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    hn = jn + 1j * yn
    djn = spherical_jn(n, z, derivative=True)
    dyn = spherical_yn(n, z, derivative=True)
    dhn = djn + 1j * dyn
    return hn + z * dhn


def test_spherical_bessel_and_hankel_against_smuthi_prototype_subset():
    """Port of stable entries from SMUTHI's spherical-function prototype test."""
    n = 4
    z = np.array([0.01, 2, 5, 2 + 0.1j, 3 - 0.2j, 20 + 20j], dtype=np.complex128)
    j, y = spherical_bessel_jy(n, z)
    jn = j[n]
    hn = j[n] + 1j * y[n]

    np.testing.assert_allclose(jn[0], 1.058196248205502e-11, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jn[1], 0.014079392762915, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jn[2], 0.187017655344890, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jn[3], 0.013925330885893 + 0.002550081632129j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jn[4], 0.055554281414152 - 0.011718427699962j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jn[5], 5.430299683226971e06 - 3.884383001639664e06j, rtol=1e-10, atol=1e-6
    )

    # Skip z=0.01 and z=20+20j prototype values (SMUTHI marks first as unreliable; last is backend-sensitive).
    np.testing.assert_allclose(
        hn[1], 0.014079392762917 - 4.461291526363127j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        hn[2], 0.187017655344889 - 0.186615531479296j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        hn[3], -0.937540374646528 - 4.322684701489512j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        hn[4], 0.254757423766403 - 0.894658828739464j, rtol=1e-12, atol=1e-12
    )


def test_dx_xj_and_dx_xh_against_smuthi_prototype_subset():
    """Port of stable entries from SMUTHI's dx_xj/dx_xh prototype tests."""
    n = 4
    z = np.array([0.01, 2, 5, 2 + 0.1j, 3 - 0.2j, 20 + 20j], dtype=np.complex128)
    dxxj = _dx_xj(n, z)
    dxxh = _dx_xh(n, z)

    np.testing.assert_allclose(dxxj[0], 5.290971621054867e-11, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dxxj[1], 0.065126624274088, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dxxj[2], 0.401032469441925, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        dxxj[3], 0.064527362367182 + 0.011261758715092j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dxxj[4], 0.230875079050277 - 0.041423344864749j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dxxj[5], 3.329872586039824e07 - 1.858505295737451e08j, rtol=1e-10, atol=1e-2
    )

    # Skip z=0.01 and z=20+20j prototype values (sensitive in different numerical backends).
    np.testing.assert_allclose(
        dxxh[1], 0.065126624274084 + 14.876432990566345j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dxxh[2], 0.401032469441923 + 0.669247576352214j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dxxh[3], 3.574345443512018 + 14.372976166070977j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        dxxh[4], -0.423459406818455 + 1.976243655979050j, rtol=1e-12, atol=1e-12
    )


def test_dx_consistency_with_finite_differences():
    """Port of SMUTHI consistency checks for d/dx(x*z_l)."""
    n = 3
    z0 = 0.5

    eps_j = 1e-8
    zj = np.array([z0, z0 + eps_j, z0 - eps_j], dtype=np.complex128)
    dxxj = _dx_xj(n, zj)
    jn = spherical_jn(n, zj)
    fd_j = ((z0 + eps_j) * jn[1] - (z0 - eps_j) * jn[2]) / (2 * eps_j)
    np.testing.assert_allclose(dxxj[0], fd_j, rtol=1e-8, atol=1e-10)

    eps_h = 1e-10
    zh = np.array([z0, z0 + eps_h, z0 - eps_h], dtype=np.complex128)
    dxxh = _dx_xh(n, zh)
    hn = spherical_jn(n, zh) + 1j * spherical_yn(n, zh)
    fd_h = ((z0 + eps_h) * hn[1] - (z0 - eps_h) * hn[2]) / (2 * eps_h)
    np.testing.assert_allclose(dxxh[0], fd_h, rtol=1e-5, atol=1e-8)
