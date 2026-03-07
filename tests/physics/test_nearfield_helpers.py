from typing import Any, cast

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.fields import PlaneWave
from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.postprocessing.nearfield import compute_near_field_components


def test_compute_near_field_components_without_internal_returns_consistent_total():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    particles = [Sphere(position=(0.0, 0.0, 100.0), radius=50.0, refractive_index=1.4 + 0.0j)]
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

    np.testing.assert_allclose(out.E_total, out.E_initial + out.E_scattered)
    np.testing.assert_allclose(out.H_total, out.H_initial + out.H_scattered)
    assert not np.any(out.inside_mask)


def test_compute_near_field_components_zeroes_scattered_inside_particles():
    """Inside points must not carry exterior scattered-field values."""
    pts = np.array([[0.0, 0.0, 0.0], [220.0, 0.0, 0.0]], dtype=float)
    lmax = 1
    particles = [Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.1j)]
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
        coeffs=coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        beam=beam,
        polar_angles=np.linspace(0.0, np.pi, 21),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False),
        particles=particles,
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


def test_compute_near_field_components_rejects_legacy_positions_kwarg():
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    particles = [Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.0j)]
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
    with np.testing.assert_raises(TypeError):
        fn = cast(Any, compute_near_field_components)
        fn(
            pts,
            positions=np.array([[10.0, 0.0, 0.0]], dtype=float),
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


def test_compute_near_field_components_supports_source_only_empty_particles():
    pts = np.array([[10.0, 0.0, 0.0], [180.0, 0.0, 0.0]], dtype=float)
    particles: list[Sphere] = []
    coeffs = np.zeros((0, 6), dtype=np.complex128)
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
    assert not np.any(out.inside_mask)


@pytest.mark.parametrize("n_medium", [1.0, 1.33])
def test_index_matched_sphere_total_field_matches_incident_plane_wave(n_medium: float):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [6.0, -8.0, 5.0],
            [0.0, 0.0, 80.0],
            [20.0, -15.0, 70.0],
        ],
        dtype=float,
    )
    source = PlaneWave(
        wavelength=550.0,
        medium_n=n_medium + 0j,
        polarization="TM",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    run = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=550.0,
            n_medium=n_medium + 0j,
            lmax=4,
            source=source,
            solver_method="direct",
            verbose=False,
        ),
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=40.0, refractive_index=n_medium + 0j)],
    ).run(include_farfield=False)

    nf = pcl.compute_near_field(run, points=points, channel="mixed", show_progress=False)

    phase = np.exp(1j * (2.0 * np.pi / 550.0 * n_medium) * points[:, 2])
    e_expected = np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=1)
    h_expected = np.stack(
        [np.zeros_like(phase), n_medium * phase, np.zeros_like(phase)],
        axis=1,
    )

    assert not np.any(nf.inside_mask)
    np.testing.assert_allclose(nf.E_total, e_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nf.H_total, h_expected, rtol=1e-12, atol=1e-12)
