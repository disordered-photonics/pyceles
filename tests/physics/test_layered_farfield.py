"""Physics-side invariants for layered far-field/power behavior.

These tests complement external-oracle regression by checking exact internal
consistency identities that should hold independently of MSTM references.
"""

from __future__ import annotations

import numpy as np

from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.core.sources import GaussianBeam, PlaneWave
from pyceles.simulation import Simulation, SimulationConfig


def test_layered_single_layer_farfield_matches_homogeneous_sphere():
    wavelength = 550.0
    n_medium = 1.0 + 0j
    radius = 90.0
    n_particle = 1.52 + 0.015j

    source = PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.6,
        azimuthal_angle=0.9,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=7,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 161),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 201, endpoint=False),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )

    sim_sphere = Simulation(
        cfg,
        particles=[
            Sphere(position=(0.0, 0.0, 0.0), radius=radius, refractive_index=n_particle),
        ],
    )
    sim_layered = Simulation(
        cfg,
        particles=[
            LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(radius,),
                layer_refractive_indices=(n_particle,),
            )
        ],
    )

    run_sphere = sim_sphere.run()
    run_layered = sim_layered.run()

    assert run_layered.initial_coeffs is not None
    assert run_sphere.initial_coeffs is not None
    np.testing.assert_allclose(
        run_layered.initial_coeffs, run_sphere.initial_coeffs, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(run_layered.coeffs, run_sphere.coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        run_layered.farfield.scattered_te["coeff"],
        run_sphere.farfield.scattered_te["coeff"],
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        run_layered.farfield.scattered_tm["coeff"],
        run_sphere.farfield.scattered_tm["coeff"],
        rtol=1e-11,
        atol=1e-11,
    )
    assert run_sphere.cross_sections is not None
    assert run_layered.cross_sections is not None
    for key in ("C_ext", "C_sca", "C_abs"):
        np.testing.assert_allclose(
            run_layered.cross_sections[key], run_sphere.cross_sections[key], rtol=1e-11, atol=1e-11
        )


def test_layered_single_layer_finite_beam_power_matches_homogeneous_sphere():
    wavelength = 550.0
    n_medium = 1.0 + 0j
    radius = 100.0
    n_particle = 1.45 + 0.01j

    source = GaussianBeam(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1500.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=5,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 181),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )

    run_sphere = Simulation(
        cfg,
        particles=[
            Sphere(position=(0.0, 0.0, 0.0), radius=radius, refractive_index=n_particle),
        ],
    ).run()
    run_layered = Simulation(
        cfg,
        particles=[
            LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(radius,),
                layer_refractive_indices=(n_particle,),
            )
        ],
    ).run()

    assert run_sphere.power is not None
    assert run_layered.power is not None
    for key in ("P_initial", "P_transmitted", "P_reflected", "T", "R"):
        np.testing.assert_allclose(
            run_layered.power[key], run_sphere.power[key], rtol=1e-11, atol=1e-11
        )
