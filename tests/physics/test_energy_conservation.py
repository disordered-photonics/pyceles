from __future__ import annotations

import numpy as np

import pyceles as pcl


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
        positions=np.array([[-180.0, 0.0, -20.0], [170.0, 0.0, 40.0]], dtype=float),
        radii=np.array([90.0, 75.0], dtype=float),
        n_particle=np.array([1.46 + 0.0j, 1.61 + 0.0j], dtype=np.complex128),
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
    assert abs(c_abs) / c_ext < 2e-2
    np.testing.assert_allclose(c_ext, c_sca, rtol=2e-2, atol=0.0)
