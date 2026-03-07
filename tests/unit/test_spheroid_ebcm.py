import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pyceles.core.particles import Sphere
from pyceles.core.spheroid_ebcm import (
    SpheroidShapeProfile,
    assemble_axisymmetric_pq_block,
    assemble_axisymmetric_tmatrix_block,
    axisymmetric_angular_functions,
    axisymmetric_shape_quadrature,
    combine_axisymmetric_parity_blocks,
    modified_bessel_products,
    solve_axisymmetric_tmatrix_blocks,
    solve_axisymmetric_tr_block,
    spheroid_geometry_quadrature,
    spheroid_tmatrix_and_internal_block,
    split_axisymmetric_pq_block_by_parity,
)
from pyceles.core.tmatrix import particle_T_matrix_block, sphere_internal_ratios


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
    np.testing.assert_allclose(np.sum(geom.weights), 2.0, rtol=0.0, atol=1e-13)


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


def test_modified_bessel_products_match_direct_regular_products_on_even_parity():
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
            else:
                np.testing.assert_allclose(products.psipsi[n, k, :], 0.0, rtol=0.0, atol=0.0)
                np.testing.assert_allclose(products.xipsi[n, k, :], 0.0, rtol=0.0, atol=0.0)

def test_modified_bessel_products_use_direct_outgoing_products_in_non_cancelling_region():
    nmax = 4
    s = 1.21 + 0.05j
    x = np.array([1.1, 2.3], dtype=np.complex128)
    products = modified_bessel_products(nmax=nmax, relative_refractive_index=s, x=x)

    max_order = nmax + 1
    for n in range(max_order + 1):
        for k in range(max_order + 1):
            if (n + k) % 2 == 0 and n <= k + 2:
                np.testing.assert_allclose(
                    products.xipsi[n, k, :],
                    products.xi_x[:, n] * products.psi_sx[:, k],
                    rtol=1e-13,
                    atol=1e-13,
                )


def test_modified_bessel_products_keep_regular_derivative_identities_and_finite_outgoing_terms():
    nmax = 4
    s = 1.21 + 0.05j
    x = np.array([1.1, 2.3], dtype=np.complex128)
    products = modified_bessel_products(nmax=nmax, relative_refractive_index=s, x=x)

    max_order = nmax + 1
    inv_sx2 = 1.0 / (s * (x * x))
    for n in range(max_order + 1):
        dpsi_n = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
        for k in range(max_order + 1):
            if (n + k) % 2 == 1:
                dpsi_k = spherical_jn(k, s * x) + (s * x) * spherical_jn(k, s * x, derivative=True)
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
                assert np.all(np.isfinite(products.xiprimepsi[n, k, :].real))
                assert np.all(np.isfinite(products.xiprimepsi[n, k, :].imag))
                assert np.all(np.isfinite(products.xipsiprime[n, k, :].real))
                assert np.all(np.isfinite(products.xipsiprime[n, k, :].imag))
            else:
                if 1 <= n <= nmax and 1 <= k <= nmax:
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
    block = assemble_axisymmetric_pq_block(
        1.4 + 0.0j, geom, angular, radial, k_medium=0.013
    )

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


def test_split_axisymmetric_pq_block_by_parity_partitions_n_indices():
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=100.0,
        polar_radius=130.0,
    )
    x = 0.013 * geom.radius.astype(np.complex128)
    radial = modified_bessel_products(nmax=4, relative_refractive_index=1.4 + 0.0j, x=x)
    angular = axisymmetric_angular_functions(nmax=4, m=1, quadrature=geom)
    block = assemble_axisymmetric_pq_block(
        1.4 + 0.0j, geom, angular, radial, k_medium=0.013
    )
    even_odd, odd_even = split_axisymmetric_pq_block_by_parity(block)

    np.testing.assert_array_equal(even_odd.even_indices, np.array([1, 3]))
    np.testing.assert_array_equal(even_odd.odd_indices, np.array([0, 2]))
    np.testing.assert_array_equal(odd_even.even_indices, np.array([0, 2]))
    np.testing.assert_array_equal(odd_even.odd_indices, np.array([1, 3]))
    assert even_odd.Q11.shape == (2, 2)
    assert even_odd.Q22.shape == (2, 2)
    assert odd_even.Q11.shape == (2, 2)
    assert odd_even.Q22.shape == (2, 2)


def test_solve_axisymmetric_tr_block_satisfies_block_equations_for_m_nonzero():
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=100.0,
        polar_radius=130.0,
    )
    x = 0.013 * geom.radius.astype(np.complex128)
    radial = modified_bessel_products(nmax=4, relative_refractive_index=1.4 + 0.0j, x=x)
    angular = axisymmetric_angular_functions(nmax=4, m=1, quadrature=geom)
    block = assemble_axisymmetric_pq_block(
        1.4 + 0.0j, geom, angular, radial, k_medium=0.013
    )
    even_odd, _ = split_axisymmetric_pq_block_by_parity(block)
    tr = solve_axisymmetric_tr_block(even_odd, include_internal=True)

    np.testing.assert_allclose(
        tr.T11 @ even_odd.Q11 + tr.T12 @ even_odd.Q21,
        -even_odd.P11,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tr.T11 @ even_odd.Q12 + tr.T12 @ even_odd.Q22,
        -even_odd.P12,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tr.T21 @ even_odd.Q11 + tr.T22 @ even_odd.Q21,
        -even_odd.P21,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tr.T21 @ even_odd.Q12 + tr.T22 @ even_odd.Q22,
        -even_odd.P22,
        rtol=1e-10,
        atol=1e-10,
    )
    assert tr.R11 is not None
    assert tr.R12 is not None
    assert tr.R21 is not None
    assert tr.R22 is not None
    np.testing.assert_allclose(
        tr.R11 @ even_odd.Q11 + tr.R12 @ even_odd.Q21,
        np.eye(even_odd.Q11.shape[0]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tr.R11 @ even_odd.Q12 + tr.R12 @ even_odd.Q22,
        np.zeros_like(even_odd.Q12),
        rtol=1e-10,
        atol=1e-10,
    )


def test_solve_axisymmetric_tr_block_for_m_zero_has_zero_off_diagonals():
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=100.0,
        polar_radius=130.0,
    )
    x = 0.013 * geom.radius.astype(np.complex128)
    radial = modified_bessel_products(nmax=4, relative_refractive_index=1.4 + 0.0j, x=x)
    angular = axisymmetric_angular_functions(nmax=4, m=0, quadrature=geom)
    block = assemble_axisymmetric_pq_block(
        1.4 + 0.0j, geom, angular, radial, k_medium=0.013
    )
    even_odd, _ = split_axisymmetric_pq_block_by_parity(block)
    tr = solve_axisymmetric_tr_block(even_odd, include_internal=True)

    np.testing.assert_allclose(tr.T12, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tr.T21, 0.0, rtol=0.0, atol=0.0)
    assert tr.R12 is not None and tr.R21 is not None
    np.testing.assert_allclose(tr.R12, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tr.R21, 0.0, rtol=0.0, atol=0.0)


def test_spheroid_ebcm_reference_kernels_keep_full_precision_dtypes():
    geom = spheroid_geometry_quadrature(
        n_theta=6,
        equatorial_radius=100.0,
        polar_radius=135.0,
    )
    x = np.asarray(0.02 * geom.radius, dtype=np.complex64)
    radial = modified_bessel_products(
        nmax=4,
        relative_refractive_index=complex(np.complex64(1.4 + 0.1j)),
        x=x,
    )
    angular = axisymmetric_angular_functions(nmax=4, m=1, quadrature=geom)
    block = assemble_axisymmetric_pq_block(
        1.4 + 0.1j, geom, angular, radial, k_medium=0.02
    )
    even_odd, _ = split_axisymmetric_pq_block_by_parity(block)
    tr = solve_axisymmetric_tr_block(even_odd, include_internal=True)

    assert geom.mu.dtype == np.float64
    assert geom.radius.dtype == np.float64
    assert angular.pi_nm.dtype == np.float64
    assert angular.tau_nm.dtype == np.float64
    assert radial.xipsi.dtype == np.complex128
    assert radial.q11_diag_kernel.dtype == np.complex128
    assert block.Q11.dtype == np.complex128
    assert block.P22.dtype == np.complex128
    assert tr.T11.dtype == np.complex128
    assert tr.R11 is not None and tr.R11.dtype == np.complex128


def test_combine_axisymmetric_parity_blocks_reconstructs_full_m_shapes():
    geom = spheroid_geometry_quadrature(
        n_theta=8,
        equatorial_radius=100.0,
        polar_radius=130.0,
    )
    x = 0.013 * geom.radius.astype(np.complex128)
    radial = modified_bessel_products(nmax=4, relative_refractive_index=1.4 + 0.0j, x=x)
    angular = axisymmetric_angular_functions(nmax=4, m=1, quadrature=geom)
    pq = assemble_axisymmetric_pq_block(
        1.4 + 0.0j, geom, angular, radial, k_medium=0.013
    )
    even_odd_pq, odd_even_pq = split_axisymmetric_pq_block_by_parity(pq)
    even_odd = solve_axisymmetric_tr_block(even_odd_pq, include_internal=False)
    odd_even = solve_axisymmetric_tr_block(odd_even_pq, include_internal=False)
    full = combine_axisymmetric_parity_blocks(even_odd, odd_even)

    np.testing.assert_array_equal(full.n_values, np.array([1, 2, 3, 4]))
    assert full.T11.shape == (4, 4)
    assert full.T12.shape == (4, 4)
    assert full.T21.shape == (4, 4)
    assert full.T22.shape == (4, 4)
    np.testing.assert_allclose(full.T11[np.ix_([0, 2], [1, 3])], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(full.T22[np.ix_([1, 3], [0, 2])], 0.0, rtol=0.0, atol=0.0)


def test_axisymmetric_tmatrix_block_reduces_to_sphere_diagonal_limit():
    lmax = 3
    radius = 100.0
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    n_particle = 1.5 + 0.0j
    geom = spheroid_geometry_quadrature(
        n_theta=64,
        equatorial_radius=radius,
        polar_radius=radius,
    )
    radial = modified_bessel_products(
        nmax=lmax,
        relative_refractive_index=n_particle / n_medium,
        x=(k_medium * geom.radius).astype(np.complex128),
    )
    solved_blocks = solve_axisymmetric_tmatrix_blocks(
        lmax,
        n_particle / n_medium,
        k_medium,
        geom,
        radial,
        include_internal=False,
    )
    T_axis = assemble_axisymmetric_tmatrix_block(lmax, solved_blocks)
    T_sphere = particle_T_matrix_block(
        lmax=lmax,
        k_medium=k_medium,
        particle=Sphere(position=(0.0, 0.0, 0.0), radius=radius, refractive_index=n_particle),
        n_medium=n_medium,
    )

    np.testing.assert_allclose(T_axis, T_sphere, rtol=5e-5, atol=5e-7)


def test_axisymmetric_internal_block_reduces_to_sphere_internal_ratios():
    lmax = 3
    radius = 100.0
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    n_particle = 1.5 + 0.0j

    T_axis, C_axis = spheroid_tmatrix_and_internal_block(
        lmax=lmax,
        k_medium=k_medium,
        equatorial_radius=radius,
        polar_radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
        n_theta=64,
    )
    ratios = sphere_internal_ratios(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    repeats = 2 * np.arange(1, lmax + 1) + 1
    diag = np.concatenate([np.repeat(ratios[1][1:], repeats), np.repeat(ratios[2][1:], repeats)])

    np.testing.assert_allclose(np.diag(C_axis), diag, rtol=5e-5, atol=5e-7)
    np.testing.assert_allclose(C_axis, np.diag(diag), rtol=5e-5, atol=5e-7)
    assert T_axis.shape == C_axis.shape
