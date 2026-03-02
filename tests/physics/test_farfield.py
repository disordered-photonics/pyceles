from typing import Any

import numpy as np
import pytest

from pyceles.core.fields import GaussianBeam, PlaneWave
from pyceles.core.tmatrix import mie_cross_sections
from pyceles.postprocessing.farfield import (
    absorption_cross_section,
    extinction_cross_section,
    finite_beam_power_fractions,
    incident_power_from_pwp,
    plane_wave_cross_sections,
    pwp_power_decomposition,
    pwp_power_flux,
    scattering_cross_section,
    total_field_plane_wave_pattern,
    total_scattering_cross_section,
)
from pyceles.simulation import Simulation, SimulationConfig


def _dummy_pwp(alpha: np.ndarray, beta: np.ndarray, coeff: np.ndarray) -> dict:
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": np.sin(bgrid) * np.cos(agrid),
        "ky": np.sin(bgrid) * np.sin(agrid),
        "kz": np.cos(bgrid) * np.ones_like(agrid),
        "coeff": coeff,
    }


def test_power_decomposition_identity_forward():
    alpha = np.linspace(0.0, 2 * np.pi, 21)
    beta = np.linspace(0.0, np.pi, 41)
    rng = np.random.default_rng(3)

    gi_te = rng.standard_normal((alpha.size, beta.size)) + 1j * rng.standard_normal(
        (alpha.size, beta.size)
    )
    gi_tm = rng.standard_normal((alpha.size, beta.size)) + 1j * rng.standard_normal(
        (alpha.size, beta.size)
    )
    gs_te = 0.1 * (
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size))
    )
    gs_tm = 0.1 * (
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size))
    )

    p_i_te = _dummy_pwp(alpha, beta, gi_te)
    p_i_tm = _dummy_pwp(alpha, beta, gi_tm)
    p_s_te = _dummy_pwp(alpha, beta, gs_te)
    p_s_tm = _dummy_pwp(alpha, beta, gs_tm)

    d = pwp_power_decomposition(
        p_i_te,
        p_i_tm,
        p_s_te,
        p_s_tm,
        k0=1.3,
        k_medium=2.1,
        direction="forward",
    )
    assert np.isclose(d["P_total"], d["P_initial"] + d["P_scattered"] + d["P_interference"])

    p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)
    direct = pwp_power_flux(p_t_te, k0=1.3, k_medium=2.1, direction="forward") + pwp_power_flux(
        p_t_tm,
        k0=1.3,
        k_medium=2.1,
        direction="forward",
    )
    assert np.isclose(d["P_total"], direct)


def test_incident_power_from_pwp_matches_forward_plus_backward_flux():
    alpha = np.linspace(0.0, 2 * np.pi, 21, endpoint=False)
    beta = np.linspace(0.0, np.pi, 31)
    rng = np.random.default_rng(9)
    p_i_te = _dummy_pwp(
        alpha,
        beta,
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size)),
    )
    p_i_tm = _dummy_pwp(
        alpha,
        beta,
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size)),
    )
    k0 = 2.0 * np.pi / 550.0
    k_medium = 2.0 * np.pi / 550.0

    p_ref = (
        pwp_power_flux(p_i_te, k0=k0, k_medium=k_medium, direction="forward")
        + pwp_power_flux(p_i_tm, k0=k0, k_medium=k_medium, direction="forward")
        + pwp_power_flux(p_i_te, k0=k0, k_medium=k_medium, direction="backward")
        + pwp_power_flux(p_i_tm, k0=k0, k_medium=k_medium, direction="backward")
    )
    p_inc = incident_power_from_pwp(p_i_te, p_i_tm, k0=k0, k_medium=k_medium)
    np.testing.assert_allclose(p_inc, p_ref, rtol=1e-13, atol=1e-13)


def test_power_decomposition_rejects_plane_wave_source():
    alpha = np.linspace(0.0, 2 * np.pi, 9)
    beta = np.linspace(0.0, np.pi, 17)
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p = _dummy_pwp(alpha, beta, coeff)
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )

    with pytest.raises(ValueError, match="undefined"):
        pwp_power_decomposition(
            p,
            p,
            p,
            p,
            k0=2.0 * np.pi / 550.0,
            k_medium=2.0 * np.pi / 550.0,
            direction="forward",
            source=source,
        )


def test_finite_beam_power_fractions_rejects_plane_wave_source():
    alpha = np.linspace(0.0, 2 * np.pi, 9)
    beta = np.linspace(0.0, np.pi, 17)
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p = _dummy_pwp(alpha, beta, coeff)
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )

    with pytest.raises(ValueError, match="undefined"):
        finite_beam_power_fractions(
            source,
            p,
            p,
            p,
            p,
            k0=2.0 * np.pi / 550.0,
            k_medium=2.0 * np.pi / 550.0,
        )


def test_finite_beam_power_fractions_finite_beam_returns_finite_fraction():
    alpha = np.linspace(0.0, 2 * np.pi, 9)
    beta = np.linspace(0.0, np.pi, 17)
    coeff_initial = np.ones((alpha.size, beta.size), dtype=np.complex128)
    coeff_scattered = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p_initial = _dummy_pwp(alpha, beta, coeff_initial)
    p_scattered = _dummy_pwp(alpha, beta, coeff_scattered)
    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1000.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )

    out = finite_beam_power_fractions(
        source,
        p_initial,
        p_scattered,
        p_scattered,
        p_scattered,
        k0=2.0 * np.pi / 550.0,
        k_medium=2.0 * np.pi / 550.0,
    )
    assert np.isfinite(out["P_initial"])
    assert np.isfinite(out["T"])
    assert np.isfinite(out["R"])


def test_finite_beam_power_fractions_uses_initial_pwp_for_tilted_sources():
    alpha = np.linspace(0.0, 2 * np.pi, 25, endpoint=False)
    beta = np.linspace(0.0, np.pi, 37)
    k0 = 2.0 * np.pi / 550.0
    k_medium = 2.0 * np.pi / 550.0

    # Deliberately synthetic initial spectrum so the reference is fully defined
    # by the provided PWP (not by beam-width/tilt analytic shortcuts).
    initial_coeff = np.ones((alpha.size, beta.size), dtype=np.complex128)
    zero_coeff = np.zeros_like(initial_coeff)
    p_i_te = _dummy_pwp(alpha, beta, initial_coeff)
    p_i_tm = _dummy_pwp(alpha, beta, zero_coeff)
    p_s_te = _dummy_pwp(alpha, beta, zero_coeff)
    p_s_tm = _dummy_pwp(alpha, beta, zero_coeff)

    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.7,
        azimuthal_angle=0.4,
        beam_width=1300.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    out = finite_beam_power_fractions(
        source,
        p_i_te,
        p_i_tm,
        p_s_te,
        p_s_tm,
        k0=k0,
        k_medium=k_medium,
    )
    p_forward = pwp_power_flux(p_i_te, k0=k0, k_medium=k_medium, direction="forward")
    p_initial = incident_power_from_pwp(p_i_te, p_i_tm, k0=k0, k_medium=k_medium)
    np.testing.assert_allclose(out["P_initial"], p_initial, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(out["P_transmitted"], p_forward, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(out["P_reflected"], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out["T"], p_forward / p_initial, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(out["R"], 0.0, rtol=0.0, atol=0.0)


def test_finite_beam_power_fractions_rejects_plane_wave_limit_gaussian():
    alpha = np.linspace(0.0, 2 * np.pi, 9, endpoint=False)
    beta = np.linspace(0.0, np.pi, 17)
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p = _dummy_pwp(alpha, beta, coeff)
    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=np.inf,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )

    with pytest.raises(ValueError, match="plane-wave limit"):
        finite_beam_power_fractions(
            source,
            p,
            p,
            p,
            p,
            k0=2.0 * np.pi / 550.0,
            k_medium=2.0 * np.pi / 550.0,
        )


def test_pwp_power_flux_periodic_alpha_endpoint_handling():
    beta = np.linspace(0.0, np.pi, 401)
    alpha_open = np.linspace(0.0, 2 * np.pi, 401, endpoint=False)
    alpha_closed = np.linspace(0.0, 2 * np.pi, 402, endpoint=True)

    coeff_open = np.exp(1j * alpha_open[:, None]) * np.cos(beta)[None, :]
    coeff_closed = np.exp(1j * alpha_closed[:, None]) * np.cos(beta)[None, :]

    p_open = _dummy_pwp(alpha_open, beta, coeff_open)
    p_closed = _dummy_pwp(alpha_closed, beta, coeff_closed)

    f_open = pwp_power_flux(p_open, k0=1.7, k_medium=2.3, direction="forward")
    f_closed = pwp_power_flux(p_closed, k0=1.7, k_medium=2.3, direction="forward")
    assert np.isclose(f_open, f_closed, rtol=1e-10, atol=1e-12)


def test_total_scattering_cross_section_requires_plane_wave_source():
    alpha = np.linspace(0.0, 2 * np.pi, 9)
    beta = np.linspace(0.0, np.pi, 17)
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p = _dummy_pwp(alpha, beta, coeff)
    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1000.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    with pytest.raises(ValueError, match="PlaneWave"):
        total_scattering_cross_section(
            source,
            p,
            p,
            k0=2.0 * np.pi / 550.0,
            n_medium=1.0 + 0j,
        )


def test_total_scattering_cross_section_plane_wave_zero_scatter_is_zero():
    alpha = np.linspace(0.0, 2 * np.pi, 9)
    beta = np.linspace(0.0, np.pi, 17)
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    p = _dummy_pwp(alpha, beta, coeff)
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cs = total_scattering_cross_section(
        source,
        p,
        p,
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
    )
    assert np.isclose(cs, 0.0)


def test_scattering_cross_section_density_integrates_to_total():
    alpha = np.linspace(0.0, 2 * np.pi, 81, endpoint=False)
    beta = np.linspace(0.0, np.pi, 121)
    rng = np.random.default_rng(11)
    gte = 0.1 * (
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size))
    )
    gtm = 0.1 * (
        rng.standard_normal((alpha.size, beta.size))
        + 1j * rng.standard_normal((alpha.size, beta.size))
    )

    pte = _dummy_pwp(alpha, beta, gte)
    ptm = _dummy_pwp(alpha, beta, gtm)
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TM",
        polar_angle=0.4,
        azimuthal_angle=0.9,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.2,
    )
    k0 = 2.0 * np.pi / 550.0
    dcs = scattering_cross_section(
        source,
        pte,
        ptm,
        k0=k0,
        n_medium=1.0 + 0j,
    )
    assert dcs["te"].shape == gte.shape
    assert dcs["tm"].shape == gtm.shape
    assert dcs["total"].shape == gte.shape
    assert np.all(dcs["te"] >= 0.0)
    assert np.all(dcs["tm"] >= 0.0)

    c_sca = total_scattering_cross_section(
        source,
        pte,
        ptm,
        k0=k0,
        n_medium=1.0 + 0j,
    )
    sinb = np.sin(beta)[None, :]
    alpha_ext = np.concatenate([alpha, [alpha[0] + 2.0 * np.pi]])
    values_ext = np.concatenate([dcs["total"] * sinb, (dcs["total"] * sinb)[0:1, :]], axis=0)
    int_alpha = np.trapezoid(values_ext, alpha_ext, axis=0)
    c_ref = np.trapezoid(int_alpha, beta)
    np.testing.assert_allclose(c_sca, c_ref, rtol=1e-12, atol=1e-12)


def test_plane_wave_cross_sections_are_invariant_to_global_incident_scale():
    alpha = np.linspace(0.0, 2 * np.pi, 31, endpoint=False)
    beta = np.linspace(0.0, np.pi, 25)
    k0 = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    rng = np.random.default_rng(23)

    b_ref = rng.standard_normal((3, 7)) + 1j * rng.standard_normal((3, 7))
    x_ref = rng.standard_normal((3, 7)) + 1j * rng.standard_normal((3, 7))
    gte_ref = rng.standard_normal((alpha.size, beta.size)) + 1j * rng.standard_normal(
        (alpha.size, beta.size)
    )
    gtm_ref = rng.standard_normal((alpha.size, beta.size)) + 1j * rng.standard_normal(
        (alpha.size, beta.size)
    )

    pte_ref = _dummy_pwp(alpha, beta, gte_ref)
    ptm_ref = _dummy_pwp(alpha, beta, gtm_ref)

    source_ref = PlaneWave(
        wavelength=550.0,
        medium_n=n_medium,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.4,
        azimuthal_angle=0.9,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    source_scaled = PlaneWave(
        wavelength=550.0,
        medium_n=n_medium,
        polarization=(3.0 + 0.0j, 3.0j),
        polar_angle=0.4,
        azimuthal_angle=0.9,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=2.0,
    )
    scale = 6.0

    cs_ref = plane_wave_cross_sections(
        source_ref,
        b_ref,
        x_ref,
        k0=k0,
        n_medium=n_medium,
        scattered_pwp_te=pte_ref,
        scattered_pwp_tm=ptm_ref,
    )
    cs_scaled = plane_wave_cross_sections(
        source_scaled,
        scale * b_ref,
        scale * x_ref,
        k0=k0,
        n_medium=n_medium,
        scattered_pwp_te=_dummy_pwp(alpha, beta, scale * gte_ref),
        scattered_pwp_tm=_dummy_pwp(alpha, beta, scale * gtm_ref),
    )

    for key in ("C_ext", "C_sca", "C_abs"):
        np.testing.assert_allclose(cs_scaled[key], cs_ref[key], rtol=1e-12, atol=1e-12)


def test_plane_wave_cross_sections_from_coefficients_match_single_sphere_mie():
    wavelength = 550.0
    n_medium = 1.0 + 0j
    radius = 80.0
    n_particle = 1.5 + 0.02j

    source = PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.7,
        azimuthal_angle=1.1,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=8,
        source=source,
        polar_angles=np.linspace(0.0, np.pi, 181),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    sim = Simulation(
        cfg,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([radius], dtype=float),
        n_particle=np.array([n_particle], dtype=np.complex128),
    )
    run = sim.run()
    assert run.cross_sections is not None

    cs = plane_wave_cross_sections(
        source,
        run.initial_coeffs,
        run.coeffs,
        k0=run.k0,
        n_medium=n_medium,
        scattered_pwp_te=run.farfield.scattered_te,
        scattered_pwp_tm=run.farfield.scattered_tm,
    )
    mie = mie_cross_sections(
        lmax=cfg.lmax,
        k_medium=run.k0 * complex(n_medium),
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )

    np.testing.assert_allclose(cs["C_ext"], mie["C_ext"], rtol=2e-11, atol=1e-11)
    np.testing.assert_allclose(cs["C_sca"], mie["C_sca"], rtol=5e-3, atol=5e-2)
    np.testing.assert_allclose(cs["C_abs"], mie["C_abs"], rtol=5e-3, atol=5e-2)

    c_ext = extinction_cross_section(
        source,
        run.initial_coeffs,
        run.coeffs,
        k0=run.k0,
        n_medium=n_medium,
    )
    c_abs = absorption_cross_section(
        source,
        run.initial_coeffs,
        run.coeffs,
        run.farfield.scattered_te,
        run.farfield.scattered_tm,
        k0=run.k0,
        n_medium=n_medium,
    )
    np.testing.assert_allclose(c_ext, cs["C_ext"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(c_abs, cs["C_abs"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(run.cross_sections["C_ext"], cs["C_ext"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(run.cross_sections["C_sca"], cs["C_sca"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(run.cross_sections["C_abs"], cs["C_abs"], rtol=1e-13, atol=1e-13)


def test_plane_wave_cross_sections_require_scattered_pwps():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.2,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    b = np.zeros((1, 8), dtype=np.complex128)
    x = np.zeros((1, 8), dtype=np.complex128)
    with pytest.raises(TypeError):
        plane_wave_cross_sections(
            source,
            b,
            x,
            k0=2.0 * np.pi / 550.0,
            n_medium=1.0 + 0j,
        )  # type: ignore[call-arg]


def test_simulation_dual_basis_jones_mixing_consistency():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="direct",
        solve_polarization_basis=True,
        verbose=False,
    )
    sim = Simulation(
        cfg,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    )
    run = sim.run()
    assert run.coeffs_basis is not None
    np.testing.assert_allclose(
        run.coeffs,
        run.coeffs_basis["te"] + 1.0j * run.coeffs_basis["tm"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert run.cross_sections_basis is not None
    assert "te" in run.cross_sections_basis and "tm" in run.cross_sections_basis
    assert run.unpolarized is not None
    assert "cross_sections" in run.unpolarized


def test_simulation_dual_basis_mixed_precision_runs_without_numpy2_copy_errors():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.3,
        azimuthal_angle=0.2,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
        solver_method="direct",
        solve_polarization_basis=True,
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=False,
    )
    sim = Simulation(
        cfg,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([50.0], dtype=float),
        n_particle=np.array([1.45 + 0.01j], dtype=np.complex128),
    )
    run = sim.run()
    assert run.coeffs_basis is not None
    assert run.polarization_jones is not None
    np.testing.assert_allclose(
        run.coeffs,
        run.polarization_jones[0] * run.coeffs_basis["te"]
        + run.polarization_jones[1] * run.coeffs_basis["tm"],
        rtol=1e-11,
        atol=1e-11,
    )


def test_simulation_dual_basis_farfield_matches_single_channel_run():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    common: dict[str, Any] = dict(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="direct",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=False,
    )
    sim_args = dict(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    )

    run_single = Simulation(SimulationConfig(**common), **sim_args).run()
    run_basis = Simulation(
        SimulationConfig(**common, solve_polarization_basis=True), **sim_args
    ).run()

    np.testing.assert_allclose(
        run_basis.farfield.scattered_te["coeff"],
        run_single.farfield.scattered_te["coeff"],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        run_basis.farfield.scattered_tm["coeff"],
        run_single.farfield.scattered_tm["coeff"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_simulation_dual_basis_avoids_redundant_mixed_solve_and_farfield(monkeypatch):
    import pyceles.simulation as simulation_module

    solve_rhs_shapes = []
    farfield_calls = 0

    solve_linear_system_real = simulation_module.solve_linear_system
    compute_far_field_patterns_real = simulation_module.compute_far_field_patterns

    def _solve_linear_system_wrapped(A_mv, b, **kwargs):
        solve_rhs_shapes.append(np.asarray(b).shape)
        return solve_linear_system_real(A_mv, b, **kwargs)

    def _compute_far_field_patterns_wrapped(*args, **kwargs):
        nonlocal farfield_calls
        farfield_calls += 1
        return compute_far_field_patterns_real(*args, **kwargs)

    monkeypatch.setattr(simulation_module, "solve_linear_system", _solve_linear_system_wrapped)
    monkeypatch.setattr(
        simulation_module,
        "compute_far_field_patterns",
        _compute_far_field_patterns_wrapped,
    )

    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="direct",
        solve_polarization_basis=True,
        verbose=False,
    )
    run = Simulation(
        cfg,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    ).run()

    assert len(solve_rhs_shapes) == 1
    assert len(solve_rhs_shapes[0]) == 2 and solve_rhs_shapes[0][1] == 2
    assert farfield_calls == 2
    assert run.solver_result_basis is not None
    assert int(run.solver_result.rhs_count) == 2


def test_solve_sources_te_tm_matches_single_runs():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="direct",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=False,
    )
    sim = Simulation(
        cfg,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    )
    solved = sim.solve_sources(
        {
            "te": source.with_polarization("TE"),
            "tm": source.with_polarization("TM"),
        }
    )
    multi = sim.postprocess_sources(solved)
    run_te_single = Simulation(
        SimulationConfig(**{**cfg.__dict__, "source": source.with_polarization("TE")}),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    ).run()
    run_tm_single = Simulation(
        SimulationConfig(**{**cfg.__dict__, "source": source.with_polarization("TM")}),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([60.0], dtype=float),
        n_particle=np.array([1.5 + 0.01j], dtype=np.complex128),
    ).run()

    assert int(multi.solver_result.rhs_count) == 2
    np.testing.assert_allclose(multi["te"].coeffs, run_te_single.coeffs, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(multi["tm"].coeffs, run_tm_single.coeffs, rtol=1e-6, atol=1e-7)


def test_solve_sources_sequence_labels():
    src0 = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    src1 = src0.with_polarization("TM")
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=2,
            source=src0,
            solver_method="direct",
            verbose=False,
        ),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([40.0], dtype=float),
        n_particle=np.array([1.45 + 0.01j], dtype=np.complex128),
    )
    solved = sim.solve_sources([src0, src1], labels=["first", "second"])
    multi = sim.postprocess_sources(solved)
    assert tuple(multi.labels) == ("first", "second")
    assert set(multi.runs) == {"first", "second"}


def test_postprocess_sources_include_farfield_false_keeps_solve_outputs():
    src0 = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.4,
        azimuthal_angle=0.3,
        amplitude=1.0,
    )
    src1 = src0.with_polarization("TM")
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=2,
            source=src0,
            solver_method="direct",
            verbose=False,
        ),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        radii=np.array([40.0], dtype=float),
        n_particle=np.array([1.45 + 0.01j], dtype=np.complex128),
    )

    solved = sim.solve_sources([src0, src1], labels=["first", "second"])
    multi_solve_only = sim.postprocess_sources(solved, include_farfield=False)
    multi_full = sim.postprocess_sources(solved)

    for label in ("first", "second"):
        run0 = multi_solve_only[label]
        run1 = multi_full[label]
        np.testing.assert_allclose(run0.coeffs, run1.coeffs, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(run0.rhs, run1.rhs, rtol=1e-7, atol=1e-9)
        assert run0.power is None
        assert run0.cross_sections is None
        assert run0.farfield.initial_te is None
        assert run0.farfield.initial_tm is None
        assert run0.farfield.total_te is None
        assert run0.farfield.total_tm is None
        assert int(run0.farfield.scattered_te["coeff"].size) == 0
        assert int(run0.farfield.scattered_tm["coeff"].size) == 0


def test_simulation_supports_no_particle_source_only_run():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.2,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        solver_method="gmres",
        solver_rtol=1e-6,
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=False,
    )
    run = Simulation(
        cfg,
        positions=np.zeros((0, 3), dtype=float),
        radii=np.zeros((0,), dtype=float),
        n_particle=np.zeros((0,), dtype=np.complex128),
    ).run()

    assert run.coeffs.shape[0] == 0
    assert run.solver_result.info == 0
    assert int(run.solver_result.iterations) == 0
    np.testing.assert_allclose(run.farfield.scattered_te["coeff"], 0.0, atol=0.0)
    np.testing.assert_allclose(run.farfield.scattered_tm["coeff"], 0.0, atol=0.0)
    assert run.cross_sections is not None
    np.testing.assert_allclose(run.cross_sections["C_sca"], 0.0, atol=0.0)
    np.testing.assert_allclose(run.cross_sections["C_ext"], 0.0, atol=0.0)
    np.testing.assert_allclose(run.cross_sections["C_abs"], 0.0, atol=0.0)
