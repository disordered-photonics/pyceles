"""Regression: AR=1 spheroid near field matches canonical sphere path.

This guards against convention or assembly drift in the axisymmetric
implementation by comparing near fields against the analytically equivalent
sphere representation for the same physical particle.
"""

from __future__ import annotations

import numpy as np

import pyceles as pcl
from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere, Spheroid
from pyceles.simulation import Simulation, SimulationConfig


def _run_single_particle(
    particle: Sphere | Spheroid,
    *,
    source: PlaneWave,
    lmax: int,
) -> pcl.SimulationResult:
    cfg = SimulationConfig(
        wavelength=float(source.wavelength),
        n_medium=complex(source.medium_n),
        lmax=int(lmax),
        source=source,
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    return Simulation(cfg, particles=[particle]).run(include_farfield=False)


def test_ar1_spheroid_matches_sphere_nearfield_along_hotspot_line() -> None:
    wavelength = 1.0
    radius = 25.0 / 532.0
    n_particle = 0.5 + 2.3j
    lmax = 10

    source = PlaneWave(
        wavelength=wavelength,
        medium_n=1.0 + 0.0j,
        polarization="TM",
        polar_angle=float(np.deg2rad(16.4)),
        azimuthal_angle=0.0,
        amplitude=1.0,
    )

    sphere = Sphere(
        position=(0.0, 0.0, 0.0),
        radius=radius,
        refractive_index=n_particle,
    )
    spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=radius,
        polar_radius=radius,
        refractive_index=n_particle,
        euler_angles=(0.0, 0.0, 0.0),
    )

    run_sphere = _run_single_particle(sphere, source=source, lmax=lmax)
    run_spheroid = _run_single_particle(spheroid, source=source, lmax=lmax)

    x = np.linspace(-1.2 * radius, 1.2 * radius, 401)
    y = np.zeros_like(x)
    # Offset towards the illumination-facing hemisphere where gradients are stronger.
    z = np.full_like(x, -0.75 * radius)
    points = np.stack([x, y, z], axis=1)

    nf_sphere = pcl.compute_near_field(
        run_sphere, points=points, channel="mixed", show_progress=False
    )
    nf_spheroid = pcl.compute_near_field(
        run_spheroid, points=points, channel="mixed", show_progress=False
    )

    np.testing.assert_allclose(
        np.asarray(nf_spheroid.E_total),
        np.asarray(nf_sphere.E_total),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(nf_spheroid.H_total),
        np.asarray(nf_sphere.H_total),
        rtol=1e-9,
        atol=1e-11,
    )
