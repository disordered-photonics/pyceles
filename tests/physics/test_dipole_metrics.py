from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.particles import spheres_from_arrays


def _run_no_scatterers(cfg: pcl.SimulationConfig) -> pcl.SimulationResult:
    return pcl.Simulation(cfg, particles=[]).run()


def test_dipole_power_ldos_no_scatterers_single_dipole():
    source = pcl.DipoleSource(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        position=(10.0, 0.0, -20.0),
        dipole_moment=(1.0 + 0.2j, -0.3 + 0.1j, 0.4 - 0.2j),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = _run_no_scatterers(cfg)
    out = pcl.compute_dipole_power_ldos(run)

    np.testing.assert_allclose(out.E_scattered_at_dipoles, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.delta_power, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.power_total, out.power_homogeneous, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(out.enhancement, 1.0, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        pcl.compute_dipole_ldos_enhancement(run),
        1.0,
        rtol=1e-13,
        atol=0.0,
    )


def test_dipole_power_ldos_no_scatterers_collection_matches_background_formula():
    source = pcl.DipoleCollection(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        positions=np.array([[0.0, 0.0, 0.0], [120.0, -30.0, 80.0]], dtype=float),
        dipole_moments=np.array(
            [[1.0 + 0.0j, 0.0 + 0.0j, 0.2 + 0.1j], [0.0 + 0.0j, -0.8 + 0.2j, 0.1 - 0.1j]],
            dtype=np.complex128,
        ),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = _run_no_scatterers(cfg)
    out = pcl.compute_dipole_power_ldos(run)

    p0_ref = source.dissipated_power_homogeneous_background_per_dipole()
    np.testing.assert_allclose(out.power_homogeneous, p0_ref, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(out.power_total, p0_ref, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(out.enhancement, np.ones_like(p0_ref), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        pcl.compute_dipole_ldos_enhancement(run, squeeze=False),
        np.ones_like(p0_ref),
        rtol=1e-13,
        atol=0.0,
    )


def test_dipole_power_ldos_inside_particle_requires_explicit_override():
    source = pcl.DipoleSource(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        position=(50.0, 0.0, 0.0),  # inside the sphere, but not at the center
        dipole_moment=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )

    with pytest.warns(UserWarning, match="Untested configuration: dipole center lies inside"):
        run = pcl.Simulation(
            cfg,
            particles=spheres_from_arrays(
                positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                radii=np.array([100.0], dtype=float),
                refractive_indices=np.array([1.5 + 0.01j], dtype=np.complex128),
            ),
        ).run()

    with pytest.raises(ValueError, match="inside particle index"):
        _ = pcl.compute_dipole_power_ldos(run)

    with pytest.warns(UserWarning, match="inside particle index"):
        out = pcl.compute_dipole_power_ldos(run, allow_inside_particle=True)
    assert out.enhancement.shape == (1,)


def test_dipole_near_field_masks_exact_source_position():
    source = pcl.DipoleSource(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        position=(0.0, 0.0, 0.0),
        dipole_moment=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = _run_no_scatterers(cfg)
    points = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=float)

    nf = pcl.compute_near_field(run, points=points, channel="mixed", show_progress=False)
    assert np.all(np.isnan(nf.E_initial[0]))
    assert np.all(np.isnan(nf.H_initial[0]))
    assert np.all(np.isnan(nf.E_total[0]))
    assert np.all(np.isnan(nf.H_total[0]))
    assert np.all(np.isfinite(nf.E_total[1]))
    assert np.all(np.isfinite(nf.H_total[1]))


def test_dipole_ldos_uses_channel_source_from_postprocess_sources():
    base_source = pcl.DipoleCollection(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        positions=np.array([[0.0, 0.0, 0.0], [120.0, 0.0, 0.0]], dtype=float),
        dipole_moments=np.array(
            [[1.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j, 0.0 + 0j]],
            dtype=np.complex128,
        ),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=base_source,
        solver_method="direct",
        verbose=False,
    )
    sim = pcl.Simulation(cfg, particles=[])

    probe = pcl.DipoleSource(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        position=(30.0, 0.0, -40.0),
        dipole_moment=(1.0 + 0j, 0.0 + 0j, 0.0 + 0j),
    )
    solved = sim.solve_sources(probe.cartesian_basis_sources())
    multi = sim.postprocess_sources(solved, include_farfield=False)

    assert isinstance(multi["px"].config.source, pcl.DipoleSource)
    assert isinstance(multi["py"].config.source, pcl.DipoleSource)
    assert isinstance(multi["pz"].config.source, pcl.DipoleSource)
    np.testing.assert_allclose(
        pcl.compute_dipole_ldos_enhancement(multi["px"]),
        1.0,
        rtol=1e-13,
        atol=0.0,
    )
