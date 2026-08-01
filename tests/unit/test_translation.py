import numpy as np
import pytest
from scipy.special import sph_harm_y

from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.spherical import legendre_normalized_trigon
from pyceles.core.svwf_rotation import svwf_rotation_matrix
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
    spherical_bessel_jy,
    translation_ab5_table,
    translation_block,
)
from pyceles.core.wigner import wigner_3j


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


@pytest.mark.reference
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
                for jj, lam in enumerate(range(l - m, -1, -2)):
                    got += (st**m) * (ct**lam) * float(coeff_tab[jj, m, l])
                np.testing.assert_allclose(got, float(plm[l, m]), rtol=rtol, atol=atol)


def test_translation_plm_coeff_table_high_order_remains_finite() -> None:
    """Regression: high-order table build should avoid factorial-overflow paths."""

    lmax = 64
    coeff_tab = _translation_plm_coeff_table(lmax, dtype=np.float64)
    assert coeff_tab.shape == (lmax + 1, 2 * lmax + 1, 2 * lmax + 1)
    assert np.isfinite(coeff_tab).all()


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


def test_high_order_float32_translation_lut_guards_origin_samples():
    """Complex64 translation tables keep valid low-order guard samples."""
    k = 2.0 * np.pi / 700.0
    lut = RadialLUT(lmax=12, k=k, r_max=100.0, dr=0.5, dtype=np.complex64)
    assert np.isfinite(lut.h).all()

    j0, y0 = spherical_bessel_jy(0, np.asarray(k * lut.r_grid[1]))
    expected = np.asarray(j0 + 1j * y0, dtype=np.complex64).reshape(-1)[0]
    np.testing.assert_allclose(lut.h[0, 1], expected, rtol=1e-6, atol=0.0)
    assert lut.h[0, 1] != lut.h[0, 103]


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


def _scalar_out_to_in_coeff(
    l_src: int,
    m_src: int,
    l_dst: int,
    m_dst: int,
    *,
    rvec: np.ndarray,
    k: float,
    hankel: np.ndarray,
    theta: float,
    phi: float,
) -> complex:
    """Scalar out-to-in coefficient via vector-addition-theorem scalar kernel.

    Reference:
    Dufva et al., Progress In Electromagnetics Research B 4 (2008) 79-99,
    Sec. 4, Eq. (49).
    """
    total = 0j
    for q in range(abs(l_src - l_dst), l_src + l_dst + 1):
        if (l_src + l_dst + q) % 2 != 0:
            continue
        dm = m_src - m_dst
        if abs(dm) > q:
            continue
        pref = np.sqrt((2 * l_src + 1) * (2 * l_dst + 1) * (2 * q + 1) / (4.0 * np.pi))
        gaunt_star = (
            ((-1) ** m_src)
            * pref
            * wigner_3j(l_src, l_dst, q, 0, 0, 0)
            * wigner_3j(l_src, l_dst, q, m_src, -m_dst, m_dst - m_src)
        )
        psi_q = hankel[q] * sph_harm_y(q, dm, theta, phi)
        total += ((-1j) ** (l_src - l_dst - q)) * psi_q * gaunt_star
    return 4.0 * np.pi * total


def _scalar_translation_table(
    lmax: int, k: float, rvec: np.ndarray
) -> dict[tuple[int, int, int, int], complex]:
    max_order = lmax + 1
    r = float(np.linalg.norm(rvec))
    theta = float(np.arccos(np.clip(rvec[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(rvec[1], rvec[0]))
    if phi < 0.0:
        phi += 2.0 * np.pi

    j, y = spherical_bessel_jy(2 * max_order, np.asarray(k * r, dtype=np.complex128))
    hankel = np.asarray(j + 1j * y, dtype=np.complex128).reshape((2 * max_order + 1,))

    table: dict[tuple[int, int, int, int], complex] = {}
    for l_src in range(0, max_order + 1):
        for m_src in range(-l_src, l_src + 1):
            for l_dst in range(0, max_order + 1):
                for m_dst in range(-l_dst, l_dst + 1):
                    table[(l_src, m_src, l_dst, m_dst)] = _scalar_out_to_in_coeff(
                        l_src,
                        m_src,
                        l_dst,
                        m_dst,
                        rvec=rvec,
                        k=k,
                        hankel=hankel,
                        theta=theta,
                        phi=phi,
                    )
    return table


def _scalar_lookup(
    table: dict[tuple[int, int, int, int], complex],
    l_src: int,
    m_src: int,
    l_dst: int,
    m_dst: int,
) -> complex:
    return table.get((l_src, m_src, l_dst, m_dst), 0j)


def _vector_A_from_scalar_coeffs(
    table: dict[tuple[int, int, int, int], complex],
    l_src: int,
    m_src: int,
    l_dst: int,
    m_dst: int,
) -> complex:
    """Vector-coupling A reconstructed from scalar coefficients.

    Reference:
    Dufva et al., PIER B 4 (2008) 79-99, Sec. 5, Eq. (67).
    """
    return complex(
        0.5
        * np.sqrt((l_src - m_src) * (l_src + m_src + 1) * (l_dst - m_dst) * (l_dst + m_dst + 1))
        * _scalar_lookup(table, l_src, m_src + 1, l_dst, m_dst + 1)
        + 0.5
        * np.sqrt((l_src + m_src) * (l_src - m_src + 1) * (l_dst + m_dst) * (l_dst - m_dst + 1))
        * _scalar_lookup(table, l_src, m_src - 1, l_dst, m_dst - 1)
        + m_src * m_dst * _scalar_lookup(table, l_src, m_src, l_dst, m_dst)
    ) / (l_dst * (l_dst + 1))


def _vector_B_from_scalar_coeffs(
    table: dict[tuple[int, int, int, int], complex],
    l_src: int,
    m_src: int,
    l_dst: int,
    m_dst: int,
) -> complex:
    """Vector-coupling B reconstructed from scalar coefficients.

    Reference:
    Dufva et al., PIER B 4 (2008) 79-99, Sec. 5, Eq. (68).
    """
    term_a = (
        -0.5
        * np.sqrt((l_src - m_src) * (l_src + m_src + 1))
        * (
            np.sqrt((l_dst + m_dst + 1) * (l_dst + m_dst + 2) / ((2 * l_dst + 1) * (2 * l_dst + 3)))
            / (l_dst + 1)
            * _scalar_lookup(table, l_src, m_src + 1, l_dst + 1, m_dst + 1)
            + np.sqrt((l_dst - m_dst - 1) * (l_dst - m_dst) / ((2 * l_dst - 1) * (2 * l_dst + 1)))
            / l_dst
            * _scalar_lookup(table, l_src, m_src + 1, l_dst - 1, m_dst + 1)
        )
    )
    term_b = (
        0.5
        * np.sqrt((l_src + m_src) * (l_src - m_src + 1))
        * (
            np.sqrt((l_dst - m_dst + 1) * (l_dst - m_dst + 2) / ((2 * l_dst + 1) * (2 * l_dst + 3)))
            / (l_dst + 1)
            * _scalar_lookup(table, l_src, m_src - 1, l_dst + 1, m_dst - 1)
            + np.sqrt((l_dst + m_dst - 1) * (l_dst + m_dst) / ((2 * l_dst - 1) * (2 * l_dst + 1)))
            / l_dst
            * _scalar_lookup(table, l_src, m_src - 1, l_dst - 1, m_dst - 1)
        )
    )
    term_c = m_src * (
        np.sqrt((l_dst + m_dst + 1) * (l_dst - m_dst + 1) / ((2 * l_dst + 1) * (2 * l_dst + 3)))
        / (l_dst + 1)
        * _scalar_lookup(table, l_src, m_src, l_dst + 1, m_dst)
        - np.sqrt((l_dst + m_dst) * (l_dst - m_dst) / ((2 * l_dst - 1) * (2 * l_dst + 1)))
        / l_dst
        * _scalar_lookup(table, l_src, m_src, l_dst - 1, m_dst)
    )
    return complex(term_a + term_b + term_c)


@pytest.mark.reference
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_scalar_to_vector_oracle_matches_axial_translation_block(lmax: int) -> None:
    """Low-order oracle from scalar translation to vector A/B couplings.

    References:
    Dufva et al., PIER B 4 (2008) 79-99.
    Scalar out-to-in coefficient: Sec. 4, Eq. (49).
    Vector reconstruction A/B: Sec. 5, Eqs. (67)-(68).
    """
    k = 2.0 * np.pi / 550.0
    rvec = np.array([0.0, 0.0, 100.0 + 20.0 * lmax], dtype=float)
    table = _scalar_translation_table(lmax, k, rvec)
    W = translation_block(lmax, k, rvec, ab5=translation_ab5_table(lmax))

    W_oracle = np.zeros_like(W)
    for tau_dst, l_dst, m_dst, idx_dst in iter_modes(lmax):
        for tau_src, l_src, m_src, idx_src in iter_modes(lmax):
            scale = np.sqrt((l_dst * (l_dst + 1)) / (l_src * (l_src + 1)))
            A_from_scalar = _vector_A_from_scalar_coeffs(table, l_src, m_src, l_dst, m_dst)
            B_from_scalar = _vector_B_from_scalar_coeffs(table, l_src, m_src, l_dst, m_dst)

            if tau_dst == tau_src:
                # CELES-compatible normalization factor for same-polarization coupling.
                W_oracle[idx_dst, idx_src] = scale * A_from_scalar
            else:
                # CELES-compatible normalization factor for cross-polarization coupling.
                W_oracle[idx_dst, idx_src] = 1j * (2 * l_dst + 1) * scale * B_from_scalar

    np.testing.assert_allclose(W, W_oracle, rtol=2e-11, atol=2e-11)


@pytest.mark.reference
@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_rotation_through_z_reproduces_general_translation_block(lmax: int) -> None:
    """Rotate-to-z translation oracle against direct off-axis translation.

    Reference:
    Dufva et al., PIER B 4 (2008) 79-99, Sec. 6
    (rotation-through-z translation construction).
    """
    k = 2.0 * np.pi / 550.0
    rvec = np.array([120.0, 35.0, -60.0], dtype=float)
    r = float(np.linalg.norm(rvec))
    theta = float(np.arccos(np.clip(rvec[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(rvec[1], rvec[0]))
    if phi < 0.0:
        phi += 2.0 * np.pi

    ab5 = translation_ab5_table(lmax)
    W_direct = translation_block(lmax, k, rvec, ab5=ab5)
    W_axial = translation_block(lmax, k, np.array([0.0, 0.0, r], dtype=float), ab5=ab5)
    D = svwf_rotation_matrix(lmax, phi, theta, 0.0)

    # Convention-compatible form in this codebase: W(r) = D* @ Wz(|r|) @ D^T.
    W_rot = np.asarray(D.conjugate() @ W_axial @ D.T, dtype=np.complex128)
    np.testing.assert_allclose(W_direct, W_rot, rtol=1e-12, atol=1e-12)
