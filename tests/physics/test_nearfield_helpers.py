import numpy as np

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import LayeredSphere
from pyceles.postprocessing.nearfield import compute_near_field_components


def test_compute_near_field_components_without_internal_returns_consistent_total():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    positions = np.array([[0.0, 0.0, 100.0]], dtype=float)
    coeffs = np.zeros((1, 6), dtype=np.complex128)
    beam = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    out = compute_near_field_components(
        pts,
        positions=positions,
        coeffs=coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        beam=beam,
        polar_angles=np.linspace(0.0, np.pi, 21),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False),
        n_medium=1.0 + 0j,
        show_progress=False,
    )

    np.testing.assert_allclose(out.E_total, out.E_initial + out.E_scattered)
    np.testing.assert_allclose(out.H_total, out.H_initial + out.H_scattered)
    assert not np.any(out.inside_mask)


def test_compute_near_field_components_zeroes_scattered_inside_particles():
    """Inside points must not carry exterior scattered-field values."""
    pts = np.array([[0.0, 0.0, 0.0], [220.0, 0.0, 0.0]], dtype=float)
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    radii = np.array([120.0], dtype=float)
    n_particle = np.array([1.5 + 0.1j], dtype=np.complex128)
    lmax = 1
    # Coefficients are synthetic but non-zero to exercise scattered-field path.
    coeffs = (1.0 + 0.3j) * np.ones((1, 6), dtype=np.complex128)
    beam = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    out = compute_near_field_components(
        pts,
        positions=positions,
        coeffs=coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        beam=beam,
        polar_angles=np.linspace(0.0, np.pi, 21),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False),
        radii=radii,
        n_particle=n_particle,
        n_medium=1.0 + 0j,
        show_progress=False,
    )

    assert out.inside_mask.tolist() == [True, False]
    np.testing.assert_allclose(out.E_scattered[0], np.zeros((3,), dtype=out.E_scattered.dtype))
    np.testing.assert_allclose(out.H_scattered[0], np.zeros((3,), dtype=out.H_scattered.dtype))


def test_compute_near_field_components_supports_particle_dispatch_for_internal_fields():
    pts = np.array([[10.0, 0.0, 0.0], [140.0, 0.0, 0.0]], dtype=float)
    particles = [
        LayeredSphere(
            position=(0.0, 0.0, 0.0),
            layer_radii=(50.0, 100.0),
            layer_refractive_indices=(1.8 + 0j, 1.3 + 0.01j),
        )
    ]
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    coeffs = np.ones((1, 6), dtype=np.complex128) * (0.8 + 0.2j)
    beam = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    out = compute_near_field_components(
        pts,
        positions=positions,
        coeffs=coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        beam=beam,
        polar_angles=np.linspace(0.0, np.pi, 21),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False),
        particles=particles,
        n_medium=1.0 + 0j,
        show_progress=False,
    )
    assert out.inside_mask.tolist() == [True, False]
    np.testing.assert_allclose(out.E_scattered[0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.H_scattered[0], 0.0, rtol=0.0, atol=0.0)
