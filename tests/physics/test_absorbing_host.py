"""Finite-reference source and channel contracts in an absorbing host."""

from dataclasses import replace

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.sources import JonesPolarizedSource, Source
from pyceles.postprocessing.farfield import scattering_cross_section


def _two_sphere_simulation(method="direct"):
    return pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=600,
            n_medium=1.33 + 0.1j,
            lmax=2,
            solver_method=method,
            solver_rtol=1e-10,
            radial_lut_dr=600 / 4096,
            polar_angles=np.array([0, 0.4, np.pi / 2, np.pi]),
            azimuthal_angles=np.arange(16) * np.pi / 8,
            verbose=False,
        ),
        particles=[
            pcl.Sphere((0, 0, 0), 40, 1.7 + 0.03j),
            pcl.Sphere((160, 20, 30), 35, 1.6 + 0.02j),
        ],
    )


@pytest.mark.parametrize("method", ["direct", "gmres", "bicgstab"])
def test_plane_wave_reference_shift_and_polarization_channels(method):
    simulation = _two_sphere_simulation(method)
    source = pcl.PlaneWave(
        600,
        medium_n=simulation.config.n_medium,
        polarization=(0.6, 0.8j),
        polar_angle=0.4,
        azimuthal_angle=0.7,
        focal_point=(20, -10, 30),
        amplitude=0.7 - 0.2j,
    )
    run = simulation.run(source)
    basis = simulation.run_polarizations(source)
    np.testing.assert_allclose(basis.mixed.coeffs, run.coeffs, rtol=1e-8, atol=1e-12)
    assert basis.unpolarized.power is None and basis.unpolarized.cross_sections is None
    for channel in (run, basis.te, basis.tm, basis.mixed):
        assert channel.power is None and channel.cross_sections is None
        assert channel.decomposition_forward is None and channel.decomposition_backward is None
        assert channel.k == 2 * np.pi / 600 * source.medium_n

    direction = np.array([np.sin(0.4) * np.cos(0.7), np.sin(0.4) * np.sin(0.7), np.cos(0.4)])
    shift = np.array([5, -8, 12])
    equivalent = replace(
        source,
        focal_point=tuple(np.asarray(source.focal_point) + shift),
        amplitude=source.amplitude * np.exp(1j * run.k * (direction @ shift)),
    )
    solved = simulation.solve_sources({"original": source, "shifted": equivalent})
    assert solved.k == run.k
    block = simulation.postprocess_sources(solved)
    for channel in block.channels.values():
        np.testing.assert_allclose(channel.coeffs, run.coeffs, rtol=1e-8, atol=1e-12)
    field = pcl.compute_near_field(
        run, points=np.array([[0, 0, 0], [160, 20, 30], [90, 30, 80]]), show_progress=False
    )
    np.testing.assert_array_equal(field.inside_mask, [True, True, False])
    for total, internal in ((field.E_total, field.E_internal), (field.H_total, field.H_internal)):
        assert np.isfinite(total).all()
        np.testing.assert_array_equal(total[:2], internal[:2])
    np.testing.assert_allclose(
        field.H_initial, source.medium_n * np.cross(direction, field.E_initial), rtol=1e-14, atol=0
    )
    with pytest.raises(ValueError, match="absorbing"):
        scattering_cross_section(
            source, run.farfield.scattered, k0=run.k0, n_medium=source.medium_n
        )


@pytest.mark.parametrize("kind", ["plane", "dipole"])
def test_absorbing_host_single_rhs_lsqr_matches_direct(kind):
    simulation = _two_sphere_simulation("lsqr")
    host = simulation.config.n_medium
    source = (
        pcl.PlaneWave(600, medium_n=host, polar_angle=0.4)
        if kind == "plane"
        else pcl.DipoleSource(600, medium_n=host, position=(0, 0, -170))
    )
    actual = simulation.run(source, include_farfield=False)
    expected = _two_sphere_simulation("direct").run(source, include_farfield=False)
    assert np.linalg.norm(actual.coeffs - expected.coeffs) / np.linalg.norm(expected.coeffs) < 1e-8


@pytest.mark.parametrize("kind", ["jones", "cartesian", "slm", "slm_jones"])
def test_finite_center_ring_sources_retain_complex_angular_amplitudes(kind):
    simulation = _two_sphere_simulation()
    host = simulation.config.n_medium
    beam = (
        pcl.CartesianPolarizedBesselBeam(600, medium_n=host, cone_angle=0.4)
        if kind == "cartesian"
        else pcl.BesselBeam(600, medium_n=host, cone_angle=0.4, polarization=(0.6, 0.8j))
    )
    source: Source = beam
    if kind == "slm":
        source = pcl.AngularSpectrumSLMSource(beam, modulation=0.4 + 0.7j)
    elif kind == "slm_jones":
        assert isinstance(beam, pcl.BesselBeam)
        source = pcl.SLMSource(beam, modulation=0.4 + 0.7j)
    run = simulation.run(source)
    assert run.farfield.initial is not None and run.farfield.total is not None
    assert run.power is None and run.cross_sections is None
    assert np.any(run.farfield.initial.kz.imag)
    field = pcl.compute_near_field(
        run, points=np.array([[0, 0, 0], [100, 20, 30]]), show_progress=False
    )
    assert np.isfinite(field.E_total).all() and np.isfinite(field.H_total).all()
    if kind in {"slm", "slm_jones"}:
        plain = simulation.run(beam)
        np.testing.assert_allclose(run.coeffs, (0.4 + 0.7j) * plain.coeffs, rtol=1e-12, atol=1e-14)
    if isinstance(source, JonesPolarizedSource):
        polarized = simulation.run_polarizations(source)
        np.testing.assert_allclose(polarized.mixed.coeffs, run.coeffs, rtol=1e-12, atol=1e-14)
