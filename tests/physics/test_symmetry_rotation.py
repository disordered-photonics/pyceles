from __future__ import annotations

from typing import Any, cast

import numpy as np

import pyceles as pcl
from pyceles.core.particles import spheres_from_arrays


def _scattered_intensity(run: pcl.SimulationResult) -> np.ndarray:
    ff = run.farfield
    return cast(
        np.ndarray,
        np.abs(np.asarray(ff.scattered_te["coeff"])) ** 2
        + np.abs(np.asarray(ff.scattered_tm["coeff"])) ** 2,
    )


def _rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return cast(np.ndarray, np.asarray(points, dtype=float) @ rot.T)


def test_scattered_field_far_zone_obeys_inverse_radius_scaling():
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=spheres_from_arrays(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            radii=np.array([90.0], dtype=float),
            refractive_indices=np.array([1.55 + 0.01j], dtype=np.complex128),
        ),
    ).run()

    r1 = 30_000.0
    r2 = 60_000.0
    pts = np.array([[r1, 0.0, 0.0], [r2, 0.0, 0.0]], dtype=float)
    nf = pcl.compute_near_field(run, points=pts, channel="mixed", show_progress=False)

    e1 = float(np.linalg.norm(nf.E_scattered[0]))
    e2 = float(np.linalg.norm(nf.E_scattered[1]))
    h1 = float(np.linalg.norm(nf.H_scattered[0]))
    h2 = float(np.linalg.norm(nf.H_scattered[1]))

    assert e1 > 0.0 and e2 > 0.0
    assert h1 > 0.0 and h2 > 0.0
    np.testing.assert_allclose(e1 * r1, e2 * r2, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(h1 * r1, h2 * r2, rtol=1e-3, atol=0.0)


def test_single_sphere_normal_incidence_unpolarized_scattering_is_azimuthally_symmetric():
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.2 + 0.1j),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 161, endpoint=True),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False),
        solver_method="direct",
        verbose=False,
    )
    sim = pcl.Simulation(
        cfg,
        particles=spheres_from_arrays(
            positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
            radii=np.array([90.0], dtype=float),
            refractive_indices=np.array([1.48 + 0.01j], dtype=np.complex128),
        ),
    )
    solved = sim.solve_sources(
        {
            "te": source.with_polarization("TE"),
            "tm": source.with_polarization("TM"),
        }
    )
    multi = sim.postprocess_sources(solved)
    I = 0.5 * (_scattered_intensity(multi["te"]) + _scattered_intensity(multi["tm"]))

    mean_beta = np.mean(I, axis=0)
    global_scale = float(np.max(mean_beta))
    active = mean_beta > (1e-8 * global_scale)
    if not np.any(active):
        raise AssertionError("Unexpected zero scattering across all angles.")

    # For a sphere under normal incidence, unpolarized intensity is azimuth-independent.
    variation = np.max(np.abs(I[:, active] - mean_beta[None, active])) / global_scale
    assert variation < 1e-3


def test_global_z_rotation_covariance_for_plane_wave_cluster():
    n_alpha = 96
    shift = 11
    delta = 2.0 * np.pi * shift / n_alpha

    positions = np.array(
        [
            [-220.0, -80.0, -30.0],
            [130.0, -140.0, 60.0],
            [180.0, 170.0, -90.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 75.0, 70.0], dtype=float)
    n_particle = np.array([1.50 + 0.00j, 1.62 + 0.01j, 1.46 + 0.02j], dtype=np.complex128)

    source_1 = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.3 + 0.2j),
        polar_angle=0.47,
        azimuthal_angle=0.21,
        amplitude=1.0,
    )
    source_2 = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.3 + 0.2j),
        polar_angle=0.47,
        azimuthal_angle=0.21 + delta,
        amplitude=1.0,
    )

    common: dict[str, Any] = dict(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        polar_angles=np.linspace(0.0, np.pi, 121, endpoint=True),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, n_alpha, endpoint=False),
        solver_method="direct",
        verbose=False,
    )
    run_1 = pcl.Simulation(
        pcl.SimulationConfig(source=source_1, **common),
        particles=spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        ),
    ).run()
    run_2 = pcl.Simulation(
        pcl.SimulationConfig(source=source_2, **common),
        particles=spheres_from_arrays(
            positions=_rotate_z(positions, delta),
            radii=radii,
            refractive_indices=n_particle,
        ),
    ).run()

    I1 = _scattered_intensity(run_1)
    I2 = _scattered_intensity(run_2)
    np.testing.assert_allclose(I2, np.roll(I1, shift=shift, axis=0), rtol=1e-3, atol=0.0)

    cs1 = run_1.cross_sections
    cs2 = run_2.cross_sections
    if cs1 is None or cs2 is None:
        raise AssertionError("Plane-wave runs must expose cross sections.")
    np.testing.assert_allclose(cs1.scattering, cs2.scattering, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(cs1.extinction, cs2.extinction, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(cs1.local_absorption, cs2.local_absorption, rtol=1e-3, atol=0.0)
