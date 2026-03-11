import numpy as np

from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.spherical import legendre_normalized_trigon
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
    spherical_bessel_jy,
    translation_ab5_table,
    translation_block,
)


def test_translation_z_axis_preserves_m():
    lmax = 3
    k = 2 * np.pi
    rvec = np.array([0.0, 0.0, 0.7])

    ab5 = translation_ab5_table(lmax)
    W = translation_block(lmax, k, rvec, ab5=ab5)

    modes = list(iter_modes(lmax))
    for tau1, l1, m1, _ in modes:
        i = index_vswf(l1, m1, tau1, lmax)
        for tau2, l2, m2, _ in modes:
            j = index_vswf(l2, m2, tau2, lmax)
            if m1 != m2:
                assert abs(W[i, j]) < 1e-8


def test_translation_block_shape():
    lmax = 4
    k = 3.0
    rvec = np.array([0.2, -0.1, 0.4])
    ab5 = translation_ab5_table(lmax)
    W = translation_block(lmax, k, rvec, ab5=ab5)
    assert W.shape == (n_modes(lmax), n_modes(lmax))


def test_translation_ab5_against_smuthi_prototype_values():
    """Ported from SMUTHI's `test_ab5_versus_prototype`."""
    lmax = 6
    ab5 = translation_ab5_table(lmax)

    # a5(3, -1, 2, 2, 3) = 0.235702260395516
    dst = index_vswf(3, -1, 1, lmax)
    src = index_vswf(2, 2, 1, lmax)
    np.testing.assert_allclose(ab5[src, dst, 3], 0.235702260395516 + 0j, rtol=1e-12, atol=1e-12)

    # b5(4, 3, 2, 2, 3) = 0.912870929175277j
    dst = index_vswf(4, 3, 1, lmax)
    src = index_vswf(2, 2, 2, lmax)
    np.testing.assert_allclose(ab5[src, dst, 3], 0.912870929175277j, rtol=1e-12, atol=1e-12)


def test_translation_block_entry_matches_direct_formula():
    lmax = 3
    k = 2 * np.pi / 550.0
    rvec = np.array([120.0, 35.0, -60.0], dtype=float)
    ab5 = translation_ab5_table(lmax)
    W = translation_block(lmax, k, rvec, ab5=ab5)

    # Pick an off-axis, cross-polarized entry to catch src/dst axis mixups.
    dst = index_vswf(2, 1, 1, lmax)  # tau=1,l=2,m=1
    src = index_vswf(3, -1, 2, lmax)  # tau=2,l=3,m=-1

    m_dst = 1
    m_src = -1
    dm = m_src - m_dst

    r = float(np.linalg.norm(rvec))
    ct = float(rvec[2] / r)
    st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
    phi = float(np.arctan2(rvec[1], rvec[0]))

    j, y = spherical_bessel_jy(2 * lmax, np.asarray(k * r, dtype=np.complex128))
    h = (j + 1j * y).reshape((2 * lmax + 1,))
    plm = legendre_normalized_trigon(np.asarray(ct), np.asarray(st), 2 * lmax, xp=np)

    expected = np.exp(1j * dm * phi) * np.sum(ab5[dst, src, :] * h * plm[:, abs(dm)])
    np.testing.assert_allclose(W[dst, src], expected, rtol=1e-12, atol=1e-12)


def test_translation_plm_coeff_table_matches_scalar_legendre_values():
    lmax = 3
    max_degree = 2 * lmax
    ct = 0.37
    st = float(np.sqrt(1.0 - ct * ct))
    plm = legendre_normalized_trigon(np.asarray(ct), np.asarray(st), max_degree, xp=np)

    for coeff_dtype, rtol, atol in ((np.float32, 1e-6, 1e-7), (np.float64, 1e-12, 1e-12)):
        coeff_tab = _translation_plm_coeff_table(lmax, dtype=coeff_dtype)
        for l in range(max_degree + 1):
            for m in range(l + 1):
                got = 0.0
                jj = 0
                for lam in range(l - m, -1, -2):
                    got += (st**m) * (ct**lam) * float(coeff_tab[jj, m, l])
                    jj += 1
                np.testing.assert_allclose(got, float(plm[l, m]), rtol=rtol, atol=atol)


def test_translation_compact_ab5_tables_follow_requested_precision():
    re64, im64 = _translation_ab5_compact_tables(3, dtype=np.complex64)
    re128, im128 = _translation_ab5_compact_tables(3, dtype=np.complex128)

    assert re64.dtype == np.float32
    assert im64.dtype == np.float32
    assert re128.dtype == np.float64
    assert im128.dtype == np.float64


def test_radial_lut_is_linear_interpolation_not_nearest():
    lmax = 3
    k = 2.3
    lut = RadialLUT(lmax=lmax, k=k, r_max=5.0, dr=0.5)

    r = 1.35  # intentionally off-grid
    t = r / lut.dr
    i0 = int(np.floor(t))
    frac = t - i0

    got = lut.hankel_all_p(r)
    expected = (1.0 - frac) * lut.h[:, i0] + frac * lut.h[:, i0 + 1]
    nearest = lut.h[:, i0]

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
    assert np.linalg.norm(got - nearest) > 0.0


def test_radial_lut_nearest_mode_matches_nearest_grid_sample():
    lmax = 3
    k = 2.3
    lut = RadialLUT(lmax=lmax, k=k, r_max=5.0, dr=0.5, interpolation="nearest")
    r = 1.35
    idx = int(np.round(r / lut.dr))
    idx = max(0, min(idx, lut.r_grid.size - 1))
    got = lut.hankel_all_p(r)
    np.testing.assert_allclose(got, lut.h[:, idx], rtol=0.0, atol=0.0)


def test_radial_lut_remains_finite_near_zero():
    """Regression: singular Hankel sample at r=0 must not pollute interpolation for r<dr."""
    lut = RadialLUT(lmax=4, k=1.0, r_max=4.0, dr=0.5)
    for r in (0.0, 1e-12, 1e-8, 1e-4, 0.1, 0.49):
        h = lut.hankel_all_p(r)
        assert np.all(np.isfinite(h))


def test_translation_block_lookup_vs_direct_coupling_agree():
    """Kernel-level regression: LUT interpolation must match direct radial evaluation."""
    lmax = 3
    k = 2 * np.pi / 550.0
    ab5 = translation_ab5_table(lmax)
    rvecs = np.array(
        [
            [120.0, 35.0, -60.0],
            [-90.0, 140.0, 50.0],
            [30.0, -70.0, 190.0],
        ],
        dtype=float,
    )
    r_max = float(np.max(np.linalg.norm(rvecs, axis=1)))
    lut = RadialLUT(lmax=lmax, k=k, r_max=r_max, dr=0.05)

    for rvec in rvecs:
        W_lut = translation_block(lmax, k, rvec, ab5=ab5, radial_lut=lut)
        W_direct = translation_block(lmax, k, rvec, ab5=ab5, radial_lut=None)
        np.testing.assert_allclose(W_lut, W_direct, rtol=1e-6, atol=1e-9)
