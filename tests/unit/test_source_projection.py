from __future__ import annotations

import numpy as np

from pyceles.core.fields import (
    GaussianBeam,
    PlaneWave,
    SLMSource,
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
    initial_field_plane_wave_pattern_normal_incidence,
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


def test_gaussian_angular_spectrum_normal_matches_legacy_pwp():
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
    polar = np.linspace(0.0, np.pi, 301)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 121, endpoint=False)
    k = 2.0 * np.pi / source_ref.wavelength * np.real(source_ref.medium_n)

    te_ref, tm_ref = initial_field_plane_wave_pattern_normal_incidence(
        beam=source_ref,
        k=k,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    te_new, tm_new = source_ref.angular_spectrum(
        k=k,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )

    np.testing.assert_allclose(te_new["coeff"], te_ref["coeff"], rtol=5e-7, atol=5e-9)
    np.testing.assert_allclose(tm_new["coeff"], tm_ref["coeff"], rtol=5e-7, atol=5e-9)


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
    np.testing.assert_allclose(got, ref, rtol=3e-3, atol=3e-6)


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
