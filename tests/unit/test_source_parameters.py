"""Construction contracts for the Gaussian source family."""

import numpy as np
import pytest

from pyceles.core.sources import (
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
)
from pyceles.postprocessing.nearfield import compute_initial_field

GAUSSIAN_SOURCES = (
    GaussianBeam,
    LaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
)


@pytest.mark.parametrize("source_type", GAUSSIAN_SOURCES)
@pytest.mark.parametrize(
    "parameter,invalid_values",
    [
        ("wavelength", [-1.0, 0.0, -np.inf, np.inf, np.nan]),
        ("beam_width", [-1.0, 0.0, -np.inf, np.inf, np.nan]),
        ("polar_angle", [-0.1, np.pi + 0.1, -np.inf, np.inf, np.nan]),
        ("azimuthal_angle", [-np.inf, np.inf, np.nan]),
        ("amplitude", [-np.inf, np.inf, np.nan]),
    ],
)
def test_gaussian_sources_reject_invalid_parameters(source_type, parameter, invalid_values):
    for value in invalid_values:
        with pytest.raises(ValueError, match=parameter):
            source_type(**{"wavelength": 550.0, parameter: value})


@pytest.mark.parametrize("source_type", GAUSSIAN_SOURCES)
def test_gaussian_defaults_produce_finite_spectra_projection_and_fields(source_type):
    source = source_type(wavelength=550.0)
    assert source.has_finite_incident_power()
    polar, azimuthal = np.linspace(0, np.pi, 61), np.arange(16) * (np.pi / 8)
    k = 2 * np.pi / source.wavelength
    points = np.array([[0, 0, 0], [10, -20, 30]])
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        spectrum = source.angular_spectrum(k=k, polar_angles=polar, azimuthal_angles=azimuthal)
        coefficients = source.incident_coeffs(
            points, 1, polar_angles=polar, azimuthal_angles=azimuthal
        )
        fields = compute_initial_field(
            points,
            k=k,
            n_medium=1,
            beam=source,
            polar_angles=polar,
            azimuthal_angles=azimuthal,
            show_progress=False,
        )
    for values in (spectrum.coeff_te, spectrum.coeff_tm, coefficients, *fields):
        assert np.isfinite(values).all()


@pytest.mark.parametrize("source_type", GAUSSIAN_SOURCES)
def test_positive_gaussian_width_has_finite_power_without_a_near_zero_cutoff(source_type):
    assert source_type(wavelength=550.0, beam_width=1e-9).has_finite_incident_power()


@pytest.mark.parametrize("source_type", GAUSSIAN_SOURCES)
def test_gaussian_sources_allow_axial_angles_zero_and_signed_amplitudes(source_type):
    for angle, amplitude in ((0.0, 0.0), (np.pi, -0.5)):
        source = source_type(
            wavelength=550.0, polar_angle=angle, azimuthal_angle=-4 * np.pi, amplitude=amplitude
        )
        assert source.has_finite_incident_power()
