import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pyceles.core.spheroid_ebcm import (
    SpheroidShapeProfile,
    assemble_axisymmetric_pq_block,
    axisymmetric_angular_functions,
    axisymmetric_shape_quadrature,
    modified_bessel_products,
    spheroid_geometry_quadrature,
)


def test_spheroid_geometry_quadrature_reduces_to_sphere():
    profile = SpheroidShapeProfile(equatorial_radius=120.0, polar_radius=120.0)
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=120.0,
        polar_radius=120.0,
    )

    np.testing.assert_allclose(geom.radius, 120.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(geom.dr_dtheta, 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(geom.mu, np.cos(geom.theta), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(geom.sin_theta, np.sin(geom.theta), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(geom.dr_dmu, 0.0, rtol=0.0, atol=1e-13)
    assert profile.aspect_ratio == 1.0
    np.testing.assert_allclose(profile.equivalent_volume_radius, 120.0, rtol=0.0, atol=1e-13)
    assert geom.equivalent_volume_radius is not None
    np.testing.assert_allclose(geom.equivalent_volume_radius, 120.0, rtol=0.0, atol=1e-13)


def test_spheroid_geometry_quadrature_half_space_weights_cover_full_polar_range():
    geom = spheroid_geometry_quadrature(
        n_theta=12,
        equatorial_radius=90.0,
        polar_radius=150.0,
    )

    assert geom.uses_gauss_half_space is True
    assert np.all(geom.mu >= 0.0)
    assert np.all(geom.mu <= 1.0)
    assert np.all(geom.theta > 0.0)
    assert np.all(geom.theta < 0.5 * np.pi)
    np.testing.assert_allclose(np.sum(geom.weights), np.pi, rtol=0.0, atol=1e-13)


def test_axisymmetric_shape_quadrature_preserves_mu_theta_derivative_relation():
    profile = SpheroidShapeProfile(equatorial_radius=90.0, polar_radius=150.0)
    mu = np.array([0.1, 0.4, 0.8], dtype=float)
    geom = axisymmetric_shape_quadrature(profile, n_theta=0, mu=mu)

    np.testing.assert_allclose(geom.mu, mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(geom.theta, np.arccos(mu), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        geom.dr_dtheta,
        -np.sin(geom.theta) * geom.dr_dmu,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(geom.radius, profile.radius_from_mu(mu), rtol=0.0, atol=1e-13)


def test_modified_bessel_products_match_direct_base_products_on_even_parity():
    nmax = 3
    s = 1.37 + 0.12j
    x = np.array([0.8, 1.6], dtype=np.complex128)
    products = modified_bessel_products(nmax=nmax, relative_refractive_index=s, x=x)

    max_order = nmax + 1
    for n in range(max_order + 1):
        for k in range(max_order + 1):
            if (n + k) % 2 == 0:
                np.testing.assert_allclose(
                    products.psipsi[n, k, :],
                    products.psi_x[:, n] * products.psi_sx[:, k],
                    rtol=1e-13,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    products.xipsi[n, k, :],
                    products.xi_x[:, n] * products.psi_sx[:, k],
                    rtol=1e-13,
                    atol=1e-13,
                )
            else:
                np.testing.assert_allclose(products.psipsi[n, k, :], 0.0, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(products.xipsi[n, k, :], 0.0, rtol=0.0, atol=0.0)


def test_modified_bessel_products_match_direct_mixed_derivatives_on_odd_parity():
    nmax = 4
    s = 1.21 + 0.05j
    x = np.array([1.1, 2.3], dtype=np.complex128)
    products = modified_bessel_products(nmax=nmax, relative_refractive_index=s, x=x)

    max_order = nmax + 1
    inv_sx2 = 1.0 / (s * (x * x))
    for n in range(max_order + 1):
        dpsi_n = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
        dchi_n = spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True)
        dxi_n = dpsi_n + 1j * dchi_n
        for k in range(max_order + 1):
            if (n + k) % 2 == 1:
                dpsi_k = spherical_jn(k, s * x) + (s * x) * spherical_jn(k, s * x, derivative=True)
                np.testing.assert_allclose(
                    products.xiprimepsi[n, k, :],
                    dxi_n * products.psi_sx[:, k],
                    rtol=1e-13,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    products.xipsiprime[n, k, :],
                    products.xi_x[:, n] * dpsi_k,
                    rtol=1e-13,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    products.psiprimepsi[n, k, :],
                    dpsi_n * products.psi_sx[:, k],
                    rtol=1e-13,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    products.psipsiprime[n, k, :],
                    products.psi_x[:, n] * dpsi_k,
                    rtol=1e-13,
                    atol=1e-13,
                )
            else:
                xi_psi_over_sx2 = products.xipsi[n, k, :] * inv_sx2
                psi_psi_over_sx2 = products.psipsi[n, k, :] * inv_sx2
                np.testing.assert_allclose(
                    products.xipsi_over_sx2[n, k, :],
                    xi_psi_over_sx2,
                    rtol=1e-13,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    products.psipsi_over_sx2[n, k, :],
                    psi_psi_over_sx2,
                    rtol=1e-13,
                    atol=1e-13,
                )


def test_modified_bessel_products_diagonal_helpers_match_direct_formulas():
    nmax = 4
    s = 1.18 + 0.03j
    x = np.array([0.9, 1.7], dtype=np.complex128)
    products = modified_bessel_products(nmax=nmax, relative_refractive_index=s, x=x)

    for n in range(1, nmax + 1):
        psi_n_x = products.psi_x[:, n]
        psi_np1_x = products.psi_x[:, n + 1]
        psi_n_sx = products.psi_sx[:, n]
        psi_np1_sx = products.psi_sx[:, n + 1]
        xi_n_x = products.xi_x[:, n]
        xi_np1_x = products.xi_x[:, n + 1]
        pref = ((s - 1.0) * (s + 1.0) / s) * (n + 1) / x

        np.testing.assert_allclose(
            products.p11_diag_kernel[n - 1, :],
            s * psi_n_x * psi_np1_sx - psi_np1_x * psi_n_sx,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            products.q11_diag_kernel[n - 1, :],
            s * xi_n_x * psi_np1_sx - xi_np1_x * psi_n_sx,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            products.p22_diag_kernel[n - 1, :],
            psi_n_x * psi_np1_sx - s * psi_np1_x * psi_n_sx + pref * psi_n_x * psi_n_sx,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            products.q22_diag_kernel[n - 1, :],
            xi_n_x * psi_np1_sx - s * xi_np1_x * psi_n_sx + pref * xi_n_x * psi_n_sx,
            rtol=1e-13,
            atol=1e-13,
        )


def test_axisymmetric_angular_functions_for_m_zero_match_legendre_relations():
    geom = spheroid_geometry_quadrature(
        n_theta=6,
        equatorial_radius=90.0,
        polar_radius=140.0,
    )
    ang = axisymmetric_angular_functions(nmax=4, m=0, quadrature=geom)

    assert ang.m == 0
    np.testing.assert_array_equal(ang.n_values, np.array([1, 2, 3, 4]))
    assert ang.pi_nm.shape == (4, geom.theta.size)
    assert ang.tau_nm.shape == (4, geom.theta.size)
    assert ang.d_nm.shape == (4, geom.theta.size)
    np.testing.assert_allclose(ang.pi_nm, 0.0, rtol=0.0, atol=0.0)


def test_assemble_axisymmetric_pq_block_returns_finite_square_m_block():
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=100.0,
        polar_radius=130.0,
    )
    x = 0.013 * geom.radius.astype(np.complex128)
    radial = modified_bessel_products(nmax=4, relative_refractive_index=1.4 + 0.0j, x=x)
    angular = axisymmetric_angular_functions(nmax=4, m=1, quadrature=geom)
    block = assemble_axisymmetric_pq_block(1.4 + 0.0j, geom, angular, radial)

    assert block.m == 1
    np.testing.assert_array_equal(block.n_values, np.array([1, 2, 3, 4]))
    for matrix in (
        block.Q11,
        block.Q12,
        block.Q21,
        block.Q22,
        block.P11,
        block.P12,
        block.P21,
        block.P22,
    ):
        assert matrix.shape == (4, 4)
        assert np.all(np.isfinite(matrix.real))
        assert np.all(np.isfinite(matrix.imag))
