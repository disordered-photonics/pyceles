from __future__ import annotations

import numpy as np

from pyceles.core.conversions import pwp_to_svwf_regular
from pyceles.core.fields import (
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
    PlaneWave,
    SLMSource,
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
    project_source_basis_to_svwf,
    project_source_to_svwf,
)


def test_project_source_to_svwf_matches_planewave_formula():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, -20.0, 40.0],
            [-80.0, 35.0, 15.0],
        ],
        dtype=float,
    )
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.21,
        azimuthal_angle=0.7,
        focal_point=(3.0, -5.0, 2.0),
        amplitude=1.2,
    )
    lmax = 3
    polar = np.linspace(0.0, np.pi, 121)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 61, endpoint=False)

    got = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    ref = incident_coeffs_planewave(positions, lmax, source)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_project_source_to_svwf_matches_gaussian_formula():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, -20.0, 40.0],
            [-80.0, 35.0, 15.0],
        ],
        dtype=float,
    )
    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.4,
        beam_width=2000.0,
        focal_point=(3.0, -5.0, 2.0),
        amplitude=1.2,
    )
    lmax = 3
    polar = np.linspace(0.0, np.pi, 301)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 61, endpoint=False)

    got = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    ref = incident_coeffs_wavebundle_normal_incidence(positions, lmax, source, polar)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_gaussian_angular_spectrum_projection_parity_normal():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, -20.0, 40.0],
            [-80.0, 35.0, 15.0],
        ],
        dtype=float,
    )
    source_ref = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.4,
        beam_width=2000.0,
        focal_point=(3.0, -5.0, 2.0),
        amplitude=1.2,
    )
    lmax = 3
    polar = np.linspace(0.0, np.pi, 1001)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 121, endpoint=False)

    ref = incident_coeffs_wavebundle_normal_incidence(positions, lmax, source_ref, polar)
    got = project_source_to_svwf(
        positions,
        lmax,
        source_ref,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_pwp_to_svwf_regular_rejects_te_tm_grid_mismatch():
    alpha = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    beta = np.linspace(0.0, np.pi, 21)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    k = 2.0 * np.pi / 550.0
    pwp_te = {
        "alpha": alpha,
        "beta": beta,
        "kx": k * np.sin(bgrid) * np.cos(agrid),
        "ky": k * np.sin(bgrid) * np.sin(agrid),
        "kz": np.broadcast_to(k * np.cos(bgrid), (alpha.size, beta.size)),
        "coeff": np.ones((alpha.size, beta.size), dtype=np.complex128),
    }
    pwp_tm = dict(pwp_te)
    pwp_tm["alpha"] = alpha + 1e-3

    with np.testing.assert_raises_regex(ValueError, "azimuth grids must be identical"):
        pwp_to_svwf_regular(
            np.array([[0.0, 0.0, 0.0]], dtype=float),
            1,
            k=k,
            pwp_te=pwp_te,
            pwp_tm=pwp_tm,
        )


def test_pwp_to_svwf_regular_rejects_wavevector_k_inconsistency():
    alpha = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    beta = np.linspace(0.0, np.pi, 21)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    k = 2.0 * np.pi / 550.0
    pwp_te = {
        "alpha": alpha,
        "beta": beta,
        "kx": k * np.sin(bgrid) * np.cos(agrid),
        "ky": k * np.sin(bgrid) * np.sin(agrid),
        "kz": np.broadcast_to(k * np.cos(bgrid), (alpha.size, beta.size)),
        "coeff": np.ones((alpha.size, beta.size), dtype=np.complex128),
    }
    pwp_tm = dict(pwp_te)

    bad_te = dict(pwp_te)
    bad_tm = dict(pwp_tm)
    bad_te["kz"] = np.asarray(pwp_te["kz"], dtype=float) + 0.05 * k
    bad_tm["kz"] = np.asarray(pwp_tm["kz"], dtype=float) + 0.05 * k
    with np.testing.assert_raises_regex(ValueError, "inconsistent with `k`"):
        pwp_to_svwf_regular(
            np.array([[0.0, 0.0, 0.0]], dtype=float),
            1,
            k=k,
            pwp_te=bad_te,
            pwp_tm=bad_tm,
        )


def test_gaussian_angular_spectrum_tilted_runs():
    source = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.35,
        azimuthal_angle=0.9,
        beam_width=1500.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 101)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 61, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    assert te["coeff"].shape == (azimuthal.size, polar.size)
    assert tm["coeff"].shape == (azimuthal.size, polar.size)


def test_planewave_jones_mixes_basis_linearly():
    positions = np.array([[0.0, 0.0, 0.0], [15.0, -7.0, 4.0]], dtype=float)
    src = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 1.0j),
        polar_angle=0.2,
        azimuthal_angle=0.7,
        amplitude=0.9,
    )
    basis = project_source_basis_to_svwf(
        positions,
        3,
        src,
        polar_angles=np.linspace(0.0, np.pi, 41),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 21, endpoint=False),
    )
    got = project_source_to_svwf(
        positions,
        3,
        src,
        polar_angles=np.linspace(0.0, np.pi, 41),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 21, endpoint=False),
    )
    ref = basis["te"] + 1.0j * basis["tm"]
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_slm_source_identity_modulation_matches_base_projection():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, -20.0, 40.0],
            [-80.0, 35.0, 15.0],
        ],
        dtype=float,
    )
    base = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, -0.5j),
        polar_angle=0.33,
        azimuthal_angle=0.5,
        beam_width=1600.0,
        focal_point=(5.0, -3.0, 12.0),
        amplitude=1.1,
    )
    source = SLMSource(base_source=base, modulation=1.0 + 0.0j)
    lmax = 3
    polar = np.linspace(0.0, np.pi, 201)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 121, endpoint=False)

    got = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    ref = project_source_to_svwf(
        positions,
        lmax,
        base,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    np.testing.assert_allclose(got, ref, rtol=5e-12, atol=5e-12)


def test_slm_source_phase_ramp_matches_focal_shift_in_angular_spectrum():
    base = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 0.7 - 0.2j),
        polar_angle=0.2,
        azimuthal_angle=0.7,
        beam_width=1800.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    k = 2.0 * np.pi / base.wavelength * np.real(base.medium_n)
    dx = 120.0

    def modulation(alpha_grid: np.ndarray, beta_grid: np.ndarray) -> np.ndarray:
        return np.exp(-1j * k * np.sin(beta_grid) * np.cos(alpha_grid) * dx)

    slm = SLMSource(base_source=base, modulation=modulation)
    shifted = GaussianBeam(
        wavelength=base.wavelength,
        medium_n=base.medium_n,
        polarization=base.polarization,
        polar_angle=base.polar_angle,
        azimuthal_angle=base.azimuthal_angle,
        beam_width=base.beam_width,
        focal_point=(dx, 0.0, 0.0),
        amplitude=base.amplitude,
    )

    polar = np.linspace(0.0, np.pi, 181)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 161, endpoint=False)
    te_slm, tm_slm = slm.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    te_shift, tm_shift = shifted.angular_spectrum(
        k=k, polar_angles=polar, azimuthal_angles=azimuthal
    )

    np.testing.assert_allclose(te_slm["coeff"], te_shift["coeff"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(tm_slm["coeff"], tm_shift["coeff"], rtol=1e-11, atol=1e-11)


def test_bessel_beam_angular_spectrum_support_concentrates_on_cone():
    source = BesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=2,
        cone_angle=0.63,
        polarization="TE",
        amplitude=1.0,
        azimuthal_phase=0.2,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 361)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    beta_activity = np.sum(amp, axis=0)
    active = np.where(beta_activity > (1e-10 * np.max(beta_activity)))[0]
    assert active.size <= 2
    beta_mean = float(np.sum(polar[active] * beta_activity[active]) / np.sum(beta_activity[active]))
    assert abs(beta_mean - source.cone_angle) <= float(np.max(np.diff(polar)))


def test_bessel_beam_tilted_ring_support_matches_local_cone():
    source = BesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=1,
        cone_angle=0.58,
        polar_angle=0.41,
        azimuthal_angle=0.73,
        polarization="TE",
        amplitude=1.0,
        azimuthal_phase=0.1,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 361)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    thresh = 1e-12 * np.max(amp)
    active = amp > thresh
    # Sparse cone discretization: only few active beta bins per alpha.
    active_per_alpha = np.sum(active, axis=1)
    assert float(np.median(active_per_alpha)) <= 6.0

    sx = np.asarray(te["kx"], dtype=float) / float(k)
    sy = np.asarray(te["ky"], dtype=float) / float(k)
    sz = np.asarray(te["kz"], dtype=float) / float(k)
    n0 = np.array(
        [
            np.sin(source.polar_angle) * np.cos(source.azimuthal_angle),
            np.sin(source.polar_angle) * np.sin(source.azimuthal_angle),
            np.cos(source.polar_angle),
        ],
        dtype=float,
    )
    mu = sx * n0[0] + sy * n0[1] + sz * n0[2]
    mu_w = float(np.sum(mu[active] * amp[active]) / np.sum(amp[active]))
    beta_local_w = float(np.arccos(np.clip(mu_w, -1.0, 1.0)))
    assert abs(beta_local_w - source.cone_angle) <= 2.5 * float(np.max(np.diff(polar)))


def test_bessel_beam_oam_phase_advances_with_order_m():
    source = BesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=3,
        cone_angle=0.52,
        polarization="TE",
        amplitude=1.0,
        azimuthal_phase=0.0,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 401)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, _ = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    beta_activity = np.sum(np.abs(te["coeff"]), axis=0)
    j = int(np.argmax(beta_activity))
    c = np.asarray(te["coeff"][:, j], dtype=np.complex128)
    da = float(azimuthal[1] - azimuthal[0])
    expected = np.exp(1j * source.order_m * da)
    ratio = c[1:] / c[:-1]
    np.testing.assert_allclose(ratio, expected, rtol=1e-10, atol=1e-10)


def test_bessel_beam_m0_is_cylindrically_symmetric_in_alpha():
    source = BesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=0,
        cone_angle=0.44,
        polarization=(1.0 + 0.0j, 0.6 - 0.3j),
        amplitude=1.0,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 321)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    beta_activity = np.sum(amp, axis=0)
    active = np.where(beta_activity > (1e-10 * np.max(beta_activity)))[0]
    profile = np.sum(amp[:, active], axis=1)
    assert float(np.std(profile)) <= 1e-10 * float(np.max(np.abs(profile)))


def test_cartesian_bessel_beam_angular_spectrum_support_concentrates_on_cone():
    source = CartesianPolarizedBesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=2,
        cone_angle=0.63,
        global_polarization=(1.0 + 0.0j, 0.4j, 0.0 + 0.0j),
        amplitude=1.0,
        azimuthal_phase=0.2,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 361)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    beta_activity = np.sum(amp, axis=0)
    active = np.where(beta_activity > (1e-10 * np.max(beta_activity)))[0]
    assert active.size <= 2
    beta_mean = float(np.sum(polar[active] * beta_activity[active]) / np.sum(beta_activity[active]))
    assert abs(beta_mean - source.cone_angle) <= float(np.max(np.diff(polar)))


def test_cartesian_bessel_beam_transverse_projection_enforces_maxwell_constraint():
    source = CartesianPolarizedBesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=1,
        cone_angle=0.58,
        polar_angle=0.37,
        azimuthal_angle=0.61,
        global_polarization=(1.0 + 0.0j, 0.2 + 0.1j, 0.3j),
        amplitude=1.0,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 321)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    agrid, bgrid = np.meshgrid(azimuthal, polar, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    sx = sb * ca
    sy = sb * sa
    sz = cb

    ephi = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta = np.stack([cb * ca, cb * sa, -sb], axis=2)
    ex = te["coeff"] * ephi[..., 0] + tm["coeff"] * etheta[..., 0]
    ey = te["coeff"] * ephi[..., 1] + tm["coeff"] * etheta[..., 1]
    ez = te["coeff"] * ephi[..., 2] + tm["coeff"] * etheta[..., 2]
    dot = ex * sx + ey * sy + ez * sz
    amp = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    active = amp > (1e-12 * float(np.max(amp)))
    assert np.any(active)
    err = float(np.max(np.abs(dot[active])))
    scale = float(np.max(amp[active]))
    assert err <= 1e-10 * scale


def test_cartesian_bessel_beam_polarization_vector_is_directional_not_amplitude_scaling():
    src_ref = CartesianPolarizedBesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=0,
        cone_angle=0.44,
        global_polarization=(1.0 + 0.0j, 0.5j, 0.0 + 0.0j),
        amplitude=1.0,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    src_scaled = CartesianPolarizedBesselBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        order_m=0,
        cone_angle=0.44,
        global_polarization=(3.0 + 0.0j, 1.5j, 0.0 + 0.0j),
        amplitude=1.0,
        center=(0.0, 0.0, 0.0),
        forward_only=True,
    )
    k = 2.0 * np.pi / src_ref.wavelength * np.real(src_ref.medium_n)
    polar = np.linspace(0.0, np.pi, 241)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    te_ref, tm_ref = src_ref.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    te_s, tm_s = src_scaled.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    np.testing.assert_allclose(te_ref["coeff"], te_s["coeff"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm_ref["coeff"], tm_s["coeff"], rtol=1e-12, atol=1e-12)


def test_cartesian_focused_laguerre_transverse_projection_enforces_maxwell_constraint():
    source = CartesianPolarizedFocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=1,
        polar_angle=0.2,
        azimuthal_angle=0.7,
        global_polarization=(1.0 + 0.0j, 0.3 + 0.1j, 0.2j),
        beam_width=1000.0,
        focal_length=900.0,
        numerical_aperture=0.72,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 321)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    agrid, bgrid = np.meshgrid(azimuthal, polar, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    sx = sb * ca
    sy = sb * sa
    sz = cb

    ephi = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta = np.stack([cb * ca, cb * sa, -sb], axis=2)
    ex = te["coeff"] * ephi[..., 0] + tm["coeff"] * etheta[..., 0]
    ey = te["coeff"] * ephi[..., 1] + tm["coeff"] * etheta[..., 1]
    ez = te["coeff"] * ephi[..., 2] + tm["coeff"] * etheta[..., 2]
    dot = ex * sx + ey * sy + ez * sz
    amp = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    active = amp > (1e-12 * float(np.max(amp)))
    assert np.any(active)
    err = float(np.max(np.abs(dot[active])))
    scale = float(np.max(amp[active]))
    assert err <= 1e-10 * scale


def test_cartesian_focused_laguerre_spectrum_respects_numerical_aperture_cutoff():
    source = CartesianPolarizedFocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=0,
        polar_angle=0.0,
        azimuthal_angle=0.0,
        global_polarization=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
        beam_width=1200.0,
        focal_length=1000.0,
        numerical_aperture=0.45,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 501)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp_beta = np.sum(np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2), axis=0)
    active = np.where(amp_beta > (1e-12 * np.max(amp_beta)))[0]
    beta_max = float(np.max(polar[active]))
    beta_cut = float(np.arcsin(source.numerical_aperture / np.real(source.medium_n)))
    assert beta_max <= beta_cut + 2.0 * float(np.max(np.diff(polar)))


def test_cartesian_focused_laguerre_polarization_vector_is_directional_not_amplitude_scaling():
    src_ref = CartesianPolarizedFocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=1,
        polar_angle=0.0,
        azimuthal_angle=0.0,
        global_polarization=(1.0 + 0.0j, 0.2j, 0.0 + 0.0j),
        beam_width=1200.0,
        focal_length=1000.0,
        numerical_aperture=0.7,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    src_scaled = CartesianPolarizedFocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=1,
        polar_angle=0.0,
        azimuthal_angle=0.0,
        global_polarization=(3.0 + 0.0j, 0.6j, 0.0 + 0.0j),
        beam_width=1200.0,
        focal_length=1000.0,
        numerical_aperture=0.7,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    k = 2.0 * np.pi / src_ref.wavelength * np.real(src_ref.medium_n)
    polar = np.linspace(0.0, np.pi, 301)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te_ref, tm_ref = src_ref.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    te_s, tm_s = src_scaled.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    np.testing.assert_allclose(te_ref["coeff"], te_s["coeff"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm_ref["coeff"], tm_s["coeff"], rtol=1e-12, atol=1e-12)


def test_laguerre_gaussian_l0_p0_matches_gaussian_angular_spectrum():
    lg = LaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=0,
        polarization=(1.0 + 0.0j, -0.4j),
        polar_angle=0.3,
        azimuthal_angle=0.7,
        beam_width=1800.0,
        focal_point=(4.0, -3.0, 2.0),
        amplitude=1.2,
        azimuthal_phase=0.0,
    )
    g = GaussianBeam(
        wavelength=lg.wavelength,
        medium_n=lg.medium_n,
        polarization=lg.polarization,
        polar_angle=lg.polar_angle,
        azimuthal_angle=lg.azimuthal_angle,
        beam_width=lg.beam_width,
        focal_point=lg.focal_point,
        amplitude=lg.amplitude,
    )
    k = 2.0 * np.pi / lg.wavelength * np.real(lg.medium_n)
    polar = np.linspace(0.0, np.pi, 301)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 161, endpoint=False)

    te_lg, tm_lg = lg.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
    te_g, tm_g = g.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    np.testing.assert_allclose(te_lg["coeff"], te_g["coeff"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm_lg["coeff"], tm_g["coeff"], rtol=1e-12, atol=1e-12)


def test_laguerre_gaussian_oam_phase_winding_tracks_azimuthal_order():
    source = LaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=3,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1500.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.2,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 401)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    beta_activity = np.sum(amp, axis=0)
    j = int(np.argmax(beta_activity))
    # For normal incidence and TE basis, (te - i tm) removes polarization-driven +/-1 azimuth term.
    c = np.asarray(te["coeff"][:, j] - 1j * tm["coeff"][:, j], dtype=np.complex128)
    demod = c * np.exp(1j * azimuthal)
    da = float(azimuthal[1] - azimuthal[0])
    expected = np.exp(1j * source.azimuthal_order_l * da)
    ratio = demod[1:] / demod[:-1]
    np.testing.assert_allclose(ratio, expected, rtol=5e-10, atol=5e-10)


def test_laguerre_gaussian_m0_is_cylindrically_symmetric_in_alpha():
    source = LaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=1,
        azimuthal_order_l=0,
        polarization=(1.0 + 0.0j, 0.5 - 0.2j),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1400.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    k = 2.0 * np.pi / source.wavelength * np.real(source.medium_n)
    polar = np.linspace(0.0, np.pi, 321)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)
    te, tm = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)

    amp = np.sqrt(np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2)
    profile = np.sum(amp, axis=1)
    assert float(np.std(profile)) <= 1e-10 * float(np.max(np.abs(profile)))


def test_laguerre_gaussian_projection_matches_high_resolution_reference():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [80.0, -20.0, 30.0],
            [-65.0, 45.0, 10.0],
        ],
        dtype=float,
    )
    source = LaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=1,
        azimuthal_order_l=2,
        polarization=(1.0 + 0.0j, -0.3 + 0.4j),
        polar_angle=0.25,
        azimuthal_angle=0.55,
        beam_width=1700.0,
        focal_point=(10.0, -5.0, 2.0),
        amplitude=1.1,
        azimuthal_phase=0.15,
    )
    lmax = 3
    coarse_polar = np.linspace(0.0, np.pi, 481)
    coarse_az = np.linspace(0.0, 2.0 * np.pi, 321, endpoint=False)
    fine_polar = np.linspace(0.0, np.pi, 1001)
    fine_az = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)

    got = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=coarse_polar,
        azimuthal_angles=coarse_az,
    )
    ref = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=fine_polar,
        azimuthal_angles=fine_az,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_focused_laguerre_projection_matches_high_resolution_reference():
    positions = np.array([[0.0, 0.0, 0.0], [45.0, -30.0, 15.0]], dtype=float)
    source = FocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=1,
        polarization=(1.0 + 0.0j, 0.2j),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1200.0,
        focal_length=1000.0,
        numerical_aperture=0.25,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    lmax = 3
    coarse_polar = np.linspace(0.0, np.pi, 501)
    coarse_az = np.linspace(0.0, 2.0 * np.pi, 301, endpoint=False)
    fine_polar = np.linspace(0.0, np.pi, 1001)
    fine_az = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)

    got = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=coarse_polar,
        azimuthal_angles=coarse_az,
    )
    ref = project_source_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=fine_polar,
        azimuthal_angles=fine_az,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)
