from __future__ import annotations

import numpy as np

import pyceles as pcl
from pyceles.core.particles import spheres_from_arrays
from pyceles.core.tmatrix import pec_mie_cross_sections


def test_lossless_cluster_plane_wave_has_negligible_absorption():
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.2 + 0.1j),
        polar_angle=0.35,
        azimuthal_angle=0.25,
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 241, endpoint=True),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False),
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=spheres_from_arrays(
            positions=np.array([[-180.0, 0.0, -20.0], [170.0, 0.0, 40.0]], dtype=float),
            radii=np.array([90.0, 75.0], dtype=float),
            refractive_indices=np.array([1.46 + 0.0j, 1.61 + 0.0j], dtype=np.complex128),
        ),
    ).run()

    cs = run.cross_sections
    if cs is None:
        raise AssertionError("Plane-wave run must provide cross sections.")

    c_ext = float(cs["C_ext"])
    c_sca = float(cs["C_sca"])
    c_abs = float(cs["C_abs"])

    assert c_ext > 0.0
    assert c_sca > 0.0
    assert c_abs >= -1e-8
    assert abs(c_abs) / c_ext < 1e-3
    np.testing.assert_allclose(c_ext, c_sca, rtol=1e-3, atol=0.0)


def test_pec_sphere_plane_wave_has_zero_local_absorption():
    wavelength = 550.0
    n_medium = 1.0 + 0j
    radius = 80.0
    source = pcl.PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.45,
        azimuthal_angle=0.25,
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=5,
        source=source,
        solver_method="direct",
        verbose=False,
    )

    run = pcl.Simulation(
        cfg,
        particles=[pcl.PECSphere(position=(0.0, 0.0, 0.0), radius=radius)],
    ).run()

    cs = run.cross_sections
    if cs is None:
        raise AssertionError("Plane-wave PEC run must provide cross sections.")

    mie = pec_mie_cross_sections(
        lmax=cfg.lmax,
        k_medium=run.k0 * complex(n_medium),
        radius=radius,
    )
    np.testing.assert_allclose(cs["C_ext"], mie["C_ext"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cs["C_abs"], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(cs["C_abs_local"], 0.0, rtol=0.0, atol=0.0)
    assert float(cs["C_sca"]) > 0.0
