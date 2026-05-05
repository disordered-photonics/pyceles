from typing import Any, cast

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.fields import PlaneWave
from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.core.periodic import plane_wave_k_parallel
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


def _make_periodic_run(*, polar_angle: float = 0.0, azimuthal_angle: float = 0.0):
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=float(polar_angle),
        azimuthal_angle=float(azimuthal_angle),
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(ax=700.0, ay=700.0)),
        solver_method="direct",
        verbose=False,
    )
    sim = pcl.Simulation(
        cfg,
        particles=[
            Sphere(position=(220.0, 180.0, 140.0), radius=90.0, refractive_index=1.5 + 0.0j)
        ],
    )
    return sim.run(include_farfield=False)


def test_compute_periodic_near_field_rejects_nonperiodic_run():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=80.0, refractive_index=1.5 + 0.0j)],
    ).run(include_farfield=False)

    with pytest.raises(ValueError, match="requires a periodic simulation result"):
        pcl.compute_periodic_near_field(run, points=np.array([[0.0, 0.0, 200.0]]))


def test_compute_periodic_near_field_exterior_returns_finite_components():
    run = _make_periodic_run()
    points = np.array(
        [
            [120.0, 180.0, -250.0],
            [320.0, 250.0, 860.0],
            [610.0, 410.0, 900.0],
        ],
        dtype=float,
    )

    nf_periodic = pcl.compute_periodic_near_field(
        run,
        points=points,
        channel="mixed",
        field_bmax=0.05,
    )

    assert not np.any(nf_periodic.inside_mask)
    assert np.all(np.isfinite(nf_periodic.E_total))
    assert np.all(np.isfinite(nf_periodic.H_total))


def test_compute_periodic_near_field_requires_explicit_output_basis():
    run = _make_periodic_run()
    with pytest.raises(ValueError, match="requires an explicit reciprocal output basis"):
        pcl.compute_periodic_near_field(
            run,
            points=np.array([[120.0, 180.0, 900.0]], dtype=float),
            channel="mixed",
        )


def test_compute_periodic_near_field_accepts_configured_output_bmax():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.2,
        azimuthal_angle=0.3,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        periodic=pcl.PeriodicSpec(
            lattice=pcl.RectangularLattice2D(ax=700.0, ay=700.0),
            options=pcl.PeriodicOptions(output_bmax=0.05),
        ),
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=[
            Sphere(position=(220.0, 180.0, 140.0), radius=90.0, refractive_index=1.5 + 0.0j)
        ],
    ).run(include_farfield=False)
    nf = pcl.compute_periodic_near_field(
        run,
        points=np.array([[120.0, 180.0, 900.0]], dtype=float),
        channel="mixed",
    )
    assert np.all(np.isfinite(nf.E_total))
    assert np.all(np.isfinite(nf.H_total))


def test_compute_periodic_near_field_interior_returns_finite_outside_circumspheres():
    run = _make_periodic_run(polar_angle=0.4, azimuthal_angle=0.7)
    nf = pcl.compute_periodic_near_field(
        run,
        points=np.array([[0.0, 0.0, 120.0]], dtype=float),
        channel="mixed",
    )
    assert nf.inside_mask.tolist() == [False]
    assert np.all(np.isfinite(nf.E_total))
    assert np.all(np.isfinite(nf.H_total))


def test_compute_periodic_near_field_marks_points_inside_periodic_circumspheres():
    run = _make_periodic_run(polar_angle=0.4, azimuthal_angle=0.7)
    nf = pcl.compute_periodic_near_field(
        run,
        points=np.array([[220.0, 180.0, 140.0]], dtype=float),
        channel="mixed",
    )
    assert nf.inside_mask.tolist() == [True]
    assert not np.all(np.isfinite(nf.E_total))
    assert not np.all(np.isfinite(nf.H_total))


def test_compute_periodic_near_field_interior_requires_ewald_method():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.4,
        azimuthal_angle=0.7,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        periodic=pcl.PeriodicSpec(
            lattice=pcl.RectangularLattice2D(ax=700.0, ay=700.0),
            options=pcl.PeriodicOptions(method="directsum", output_bmax=0.05),
        ),
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=[
            Sphere(position=(220.0, 180.0, 140.0), radius=90.0, refractive_index=1.5 + 0.0j)
        ],
    ).run(include_farfield=False)
    with pytest.raises(NotImplementedError, match=r"requires `periodic\.options\.method='ewald'`"):
        pcl.compute_periodic_near_field(
            run,
            points=np.array([[0.0, 0.0, 120.0]], dtype=float),
            channel="mixed",
        )


def test_periodic_supercell_replication_matches_fundamental_cell_observables():
    wavelength = 550.0
    n_medium = 1.0 + 0j
    ax = 680.0
    ay = 620.0
    beta = 0.35
    alpha = 0.6

    source_fund = PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=beta,
        azimuthal_angle=alpha,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    run_fund = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=wavelength,
            n_medium=n_medium,
            lmax=2,
            source=source_fund,
            periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(ax=ax, ay=ay)),
            solver_method="direct",
            verbose=False,
        ),
        particles=[
            Sphere(position=(210.0, 190.0, 160.0), radius=70.0, refractive_index=1.5 + 0.0j)
        ],
    ).run(include_farfield=False)

    source_super = PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=beta,
        azimuthal_angle=alpha,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    run_super = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=wavelength,
            n_medium=n_medium,
            lmax=2,
            source=source_super,
            periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(ax=2.0 * ax, ay=ay)),
            solver_method="direct",
            verbose=False,
        ),
        particles=[
            Sphere(position=(210.0, 190.0, 160.0), radius=70.0, refractive_index=1.5 + 0.0j),
            Sphere(
                position=(210.0 + ax, 190.0, 160.0),
                radius=70.0,
                refractive_index=1.5 + 0.0j,
            ),
        ],
    ).run(include_farfield=False)

    coeff_super = np.asarray(run_super.coeffs, dtype=np.complex128)
    k_parallel = plane_wave_k_parallel(source_super)
    phase = np.exp(1j * float(k_parallel[0]) * float(ax))
    np.testing.assert_allclose(coeff_super[1], phase * coeff_super[0], rtol=1e-11, atol=1e-11)

    if run_fund.periodic is None or run_super.periodic is None:
        raise RuntimeError("Periodic runs must populate SimulationResult.periodic.")
    np.testing.assert_allclose(
        [
            float(run_super.periodic.reflectance),
            float(run_super.periodic.transmittance),
            float(run_super.periodic.absorptance),
        ],
        [
            float(run_fund.periodic.reflectance),
            float(run_fund.periodic.transmittance),
            float(run_fund.periodic.absorptance),
        ],
        rtol=1e-10,
        atol=1e-10,
    )
