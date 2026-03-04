import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.particles import LayeredSphere
from pyceles.postprocessing.nearfield import compute_internal_field


def test_internal_field_particles_single_layer_matches_homogeneous_kernel():
    lmax = 4
    nm = n_modes(lmax)
    rng = np.random.default_rng(7)
    coeffs = (rng.standard_normal((1, nm)) + 1j * rng.standard_normal((1, nm))).astype(
        np.complex128
    )

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 15.0, -20.0],
            [90.0, 0.0, 0.0],
            [130.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    radius = 100.0
    n_particle = 1.5 + 0.05j
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j

    E_ref, H_ref, inside_ref = compute_internal_field(
        points,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([radius], dtype=float),
        coeffs=coeffs,
        k=k_medium,
        lmax=lmax,
        n_particle=np.array([n_particle], dtype=np.complex128),
        n_medium=n_medium,
        show_progress=False,
    )
    layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(radius,),
        layer_refractive_indices=(n_particle,),
    )
    E_new, H_new, inside_new = compute_internal_field(
        points,
        positions=None,
        radii=None,
        coeffs=coeffs,
        k=k_medium,
        lmax=lmax,
        n_particle=None,
        n_medium=n_medium,
        show_progress=False,
        particles=[layered],
    )

    np.testing.assert_array_equal(inside_new, inside_ref)
    np.testing.assert_allclose(E_new, E_ref, rtol=2e-8, atol=1e-9)
    np.testing.assert_allclose(H_new, H_ref, rtol=2e-8, atol=1e-9)


def test_internal_field_particles_layered_masks_and_finiteness():
    lmax = 3
    nm = n_modes(lmax)
    coeffs = np.zeros((1, nm), dtype=np.complex128)
    coeffs[0, 0] = 1.0 + 0.0j

    layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(50.0, 100.0),
        layer_refractive_indices=(1.8 + 0.0j, 1.3 + 0.02j),
    )
    points = np.array(
        [
            [10.0, 0.0, 0.0],  # core
            [70.0, 0.0, 0.0],  # shell
            [120.0, 0.0, 0.0],  # outside
        ],
        dtype=float,
    )
    E, H, inside = compute_internal_field(
        points,
        positions=None,
        radii=None,
        coeffs=coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        n_particle=None,
        n_medium=1.0 + 0j,
        show_progress=False,
        particles=[layered],
    )
    np.testing.assert_array_equal(inside, np.array([True, True, False]))
    assert np.all(np.isfinite(E[inside]))
    assert np.all(np.isfinite(H[inside]))
    np.testing.assert_allclose(E[~inside], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(H[~inside], 0.0, rtol=0.0, atol=0.0)
