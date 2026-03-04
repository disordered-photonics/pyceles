import numpy as np

from pyceles.core.tmatrix import (
    layered_internal_ab_ratios,
    layered_mie_ab,
    layered_sphere_T_diagonal,
    mie_ab,
    sphere_internal_ratios,
    sphere_T_diagonal,
)


def test_layered_single_layer_matches_homogeneous_sphere():
    lmax = 6
    radius = 130.0
    n_particle = 1.45 + 0.02j
    n_medium = 1.0 + 0j
    k_medium = 2.0 * np.pi / 550.0

    a_ref, b_ref = mie_ab(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    a_l, b_l = layered_mie_ab(
        lmax=lmax,
        k_medium=k_medium,
        layer_radii=(radius,),
        layer_refractive_indices=(n_particle,),
        n_medium=n_medium,
    )
    np.testing.assert_allclose(a_l, a_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b_l, b_ref, rtol=1e-10, atol=1e-12)

    td_ref = sphere_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    td_l = layered_sphere_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        layer_radii=(radius,),
        layer_refractive_indices=(n_particle,),
        n_medium=n_medium,
    )
    np.testing.assert_allclose(td_l[1], td_ref[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(td_l[2], td_ref[2], rtol=1e-10, atol=1e-12)

    ratio_ref = sphere_internal_ratios(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    ratio_l = layered_internal_ab_ratios(
        lmax=lmax,
        k_medium=k_medium,
        layer_radii=(radius,),
        layer_refractive_indices=(n_particle,),
        n_medium=n_medium,
    )
    np.testing.assert_allclose(ratio_l[1]["A"][0, :], ratio_ref[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ratio_l[2]["A"][0, :], ratio_ref[2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ratio_l[1]["B"][0, :], 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(ratio_l[2]["B"][0, :], 0.0, rtol=0.0, atol=1e-14)


def test_layered_internal_ab_shapes_and_finiteness():
    out = layered_internal_ab_ratios(
        lmax=5,
        k_medium=2.0 * np.pi / 550.0,
        layer_radii=(80.0, 120.0, 180.0),
        layer_refractive_indices=(1.8 + 0j, 1.5 + 0.01j, 1.2 + 0j),
        n_medium=1.0 + 0j,
    )
    for tau in (1, 2):
        A = np.asarray(out[tau]["A"])
        B = np.asarray(out[tau]["B"])
        assert A.shape == (3, 6)
        assert B.shape == (3, 6)
        assert np.all(np.isfinite(A[1:, 1:]))
        assert np.all(np.isfinite(B[1:, 1:]))
