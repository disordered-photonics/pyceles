from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.particles import spheres_from_arrays
from pyceles.core.tmatrix import pec_mie_cross_sections
from pyceles.postprocessing.farfield.periodic import periodic_plane_wave_orders
from pyceles.postprocessing.farfield.power import local_absorbed_power_from_exciting
from pyceles.postprocessing.nearfield import poynting


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.2j])
@pytest.mark.parametrize("kind", ["plane", "dipole"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_host_shell_flux_loss_equals_volume_absorption(host, kind, dtype):
    """Poynting's theorem in a source-free shell, with stored Z0*H and exp(-i*omega*t).

    div Re(E cross conj(Z0*H))/2 = -k0*Im(n_host**2)*|E|**2/2.
    Both particles and the dipole emitter lie inside the inner surface.
    """
    wavelength = 2 * np.pi  # k0=1 in the same arbitrary length unit as the geometry.
    source = (
        pcl.PlaneWave(wavelength, medium_n=host, polar_angle=0.4, amplitude=0.7 - 0.2j)
        if kind == "plane"
        else pcl.DipoleSource(
            wavelength,
            medium_n=host,
            position=(0, -0.4, -0.1),
            dipole_moment=(1, 0.3j, -0.2),
            radial_lut_dr=wavelength / 32768,
        )
    )
    run = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=wavelength,
            n_medium=host,
            lmax=2,
            solver_method="direct",
            compute_dtype=dtype,
            radial_lut_dr=wavelength / 32768,
            verbose=False,
        ),
        particles=[
            pcl.Sphere((-0.22, 0, 0), 0.12, 1.7 + 0.05j),
            pcl.Sphere((0.3, 0.1, 0), 0.1, 1.6 + 0.03j),
        ],
    ).run(source, include_farfield=False)
    mu, w_mu = np.polynomial.legendre.leggauss(16)
    phi = np.arange(32) * (2 * np.pi / 32)
    normals = np.stack(
        np.broadcast_arrays(
            np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi),
            np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi),
            mu[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular_weights = np.repeat(w_mu, len(phi)) * (2 * np.pi / len(phi))
    inner, outer = 0.9, 1.6
    nodes, weights = np.polynomial.legendre.leggauss(16)
    radii = np.r_[inner, outer, (inner + outer) / 2 + (outer - inner) / 2 * nodes]
    fields = pcl.compute_near_field(
        run, points=radii[:, None, None] * normals[None, :, :], show_progress=False
    )
    assert not np.any(fields.inside_mask)
    radial_flux = np.sum(poynting(fields.E_total[:2], fields.H_total[:2]) * normals, axis=-1)
    flux = radii[:2] ** 2 * (radial_flux @ angular_weights)
    electric_integral = np.sum(np.abs(fields.E_total[2:]) ** 2, axis=-1) @ angular_weights
    absorption = (
        0.5
        * run.k0
        * (complex(host) ** 2).imag
        * (outer - inner)
        / 2
        * np.dot(weights * radii[2:] ** 2, electric_integral)
    )
    scale = max(np.max(np.abs(flux)), abs(absorption))
    tolerance = 1e-5 if dtype == "complex64" else 1e-7
    assert abs(flux[0] - flux[1] - absorption) / scale < tolerance
    if complex(host).imag:
        assert absorption > 0 and flux[0] > flux[1]


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
    ).run(source)

    cs = run.cross_sections
    if cs is None:
        raise AssertionError("Plane-wave run must provide cross sections.")

    c_ext = float(cs.extinction)
    c_sca = float(cs.scattering)
    c_abs = float(cs.local_absorption)

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
        solver_method="direct",
        verbose=False,
    )

    run = pcl.Simulation(
        cfg,
        particles=[pcl.PECSphere(position=(0.0, 0.0, 0.0), radius=radius)],
    ).run(source)

    cs = run.cross_sections
    if cs is None:
        raise AssertionError("Plane-wave PEC run must provide cross sections.")

    mie = pec_mie_cross_sections(
        lmax=cfg.lmax,
        k_medium=run.k0 * complex(n_medium),
        radius=radius,
    )
    np.testing.assert_allclose(cs.extinction, mie["C_ext"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cs.local_absorption, 0.0, rtol=0.0, atol=0.0)
    assert float(cs.scattering) > 0.0


def test_periodic_raw_flux_defect_matches_local_power_defect_for_exact_ewald():
    """An arbitrary coefficient defect must close through the exact periodic identity."""
    wavelength = 632.8
    source = pcl.PlaneWave(
        wavelength=wavelength,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(900.0, 950.0),
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=3.0e-3,
            real_shells=6,
            reciprocal_shells=6,
        ),
    )
    particles = pcl.pec_spheres_from_arrays(
        positions=np.array(
            [[-120.0, -80.0, -35.0], [135.0, 95.0, 90.0]],
            dtype=float,
        ),
        radii=np.array([55.0, 60.0], dtype=float),
    )
    cfg = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=periodic,
        solver_method="direct",
        verbose=False,
    )
    sim = pcl.Simulation(cfg, particles=particles)
    solved = sim.solve_sources({"source": source})
    prepared = sim._prepared_operator_cache
    if prepared is None:
        raise AssertionError("Periodic solve must retain its prepared operator.")

    exact = np.asarray(solved.coeffs["source"], dtype=np.complex128).reshape(-1)
    rng = np.random.default_rng(20260729)
    perturbation = rng.normal(size=exact.size) + 1j * rng.normal(size=exact.size)
    perturbed = exact + 2.0e-3 * np.linalg.norm(exact) * perturbation / np.linalg.norm(perturbation)
    incident = np.asarray(solved.initial_coeffs["source"], dtype=np.complex128).reshape(-1)
    exciting = incident + np.asarray(prepared.apply_W(perturbed)).reshape(-1)

    local_power = local_absorbed_power_from_exciting(
        exciting,
        perturbed,
        k0=float(solved.k0),
        n_medium=cfg.n_medium,
    )
    payload = periodic_plane_wave_orders(
        source=source,
        lattice=periodic.lattice,
        positions=sim.positions,
        coeffs=perturbed.reshape(sim.positions.shape[0], -1),
        lmax=cfg.lmax,
        k=solved.k.real,
        n_medium=cfg.n_medium,
    )
    assert payload.power.incident_power is not None
    local_absorptance = local_power / payload.power.incident_power

    assert payload.power.flux_defect_fraction is not None
    assert abs(payload.power.flux_defect_fraction) > 1.0e-8
    np.testing.assert_allclose(
        payload.power.flux_defect_fraction,
        local_absorptance,
        rtol=2.0e-10,
        atol=2.0e-13,
    )
