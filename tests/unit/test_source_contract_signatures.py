from __future__ import annotations

import inspect

import numpy as np

from pyceles.core.sources import (
    AngularSpectrumSource,
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    DipoleCollection,
    DipoleSource,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    JonesPolarizedSource,
    LaguerreGaussianBeam,
    PlaneWave,
    SLMSource,
    Source,
)


def _gaussian_base() -> GaussianBeam:
    return GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.2,
        azimuthal_angle=0.4,
        beam_width=1400.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )


def _all_builtin_sources() -> list[Source]:
    return [
        PlaneWave(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            polarization="TE",
            polar_angle=0.2,
            azimuthal_angle=0.3,
            focal_point=(0.0, 0.0, 0.0),
            amplitude=1.0,
        ),
        _gaussian_base(),
        LaguerreGaussianBeam(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            radial_order_p=1,
            azimuthal_order_l=2,
            polarization="TE",
            polar_angle=0.2,
            azimuthal_angle=0.3,
            beam_width=1300.0,
            focal_point=(0.0, 0.0, 0.0),
            amplitude=1.0,
            azimuthal_phase=0.0,
        ),
        FocusedLaguerreGaussianBeam(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            radial_order_p=0,
            azimuthal_order_l=1,
            polarization="TE",
            polar_angle=0.0,
            azimuthal_angle=0.0,
            beam_width=1000.0,
            focal_length=900.0,
            numerical_aperture=0.75,
            focal_point=(0.0, 0.0, 0.0),
            amplitude=1.0,
            azimuthal_phase=0.0,
            sine_condition_apodization=True,
        ),
        CartesianPolarizedFocusedLaguerreGaussianBeam(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            radial_order_p=0,
            azimuthal_order_l=1,
            polar_angle=0.0,
            azimuthal_angle=0.0,
            global_polarization=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
            beam_width=1000.0,
            focal_length=900.0,
            numerical_aperture=0.75,
            focal_point=(0.0, 0.0, 0.0),
            amplitude=1.0,
            azimuthal_phase=0.0,
            sine_condition_apodization=True,
        ),
        BesselBeam(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            order_m=1,
            cone_angle=0.5,
            polar_angle=0.2,
            azimuthal_angle=0.3,
            polarization="TE",
            amplitude=1.0,
            azimuthal_phase=0.0,
            center=(0.0, 0.0, 0.0),
            forward_only=True,
        ),
        CartesianPolarizedBesselBeam(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            order_m=1,
            cone_angle=0.5,
            polar_angle=0.2,
            azimuthal_angle=0.3,
            global_polarization=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
            amplitude=1.0,
            azimuthal_phase=0.0,
            center=(0.0, 0.0, 0.0),
            forward_only=True,
        ),
        SLMSource(base_source=_gaussian_base(), modulation=1.0 + 0.0j),
        DipoleSource(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            dipole_moment=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
            position=(0.0, 0.0, 0.0),
        ),
        DipoleCollection(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            positions=np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=float),
            dipole_moments=np.array(
                [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j]],
                dtype=np.complex128,
            ),
        ),
    ]


def test_incident_coeffs_signature_is_uniform_across_sources() -> None:
    """All source classes should expose one canonical incident projection signature."""
    for source in _all_builtin_sources():
        sig = inspect.signature(source.incident_coeffs)
        params = sig.parameters
        names = list(params)
        assert names[:2] == ["positions", "lmax"], (
            f"{type(source).__name__}: unexpected leading args"
        )
        assert "polar_angles" in params and "azimuthal_angles" in params and "dtype" in params
        assert params["polar_angles"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["azimuthal_angles"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["dtype"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["polar_angles"].default is None
        assert params["azimuthal_angles"].default is None
        assert params["dtype"].default is not inspect.Parameter.empty


def test_angular_spectrum_signature_is_uniform_for_angular_sources() -> None:
    """Angular-spectrum-capable sources must expose consistent `angular_spectrum(...)` API."""
    for source in _all_builtin_sources():
        if not isinstance(source, AngularSpectrumSource):
            continue
        sig = inspect.signature(source.angular_spectrum)
        params = sig.parameters
        names = list(params)
        assert names == ["k", "polar_angles", "azimuthal_angles"], (
            f"{type(source).__name__}: unexpected angular_spectrum parameters {names}"
        )
        assert params["k"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["polar_angles"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["azimuthal_angles"].kind is inspect.Parameter.KEYWORD_ONLY


def test_jones_methods_exist_only_on_jones_sources() -> None:
    """Jones-capable protocol is reflected explicitly in callable method surface."""
    for source in _all_builtin_sources():
        has_jones = isinstance(source, JonesPolarizedSource)
        has_jones_coeff = callable(getattr(source, "jones_coefficients", None))
        has_with_pol = callable(getattr(source, "with_polarization", None))
        assert has_jones_coeff is has_jones, (
            f"{type(source).__name__}: unexpected presence of jones_coefficients"
        )
        assert has_with_pol is has_jones, (
            f"{type(source).__name__}: unexpected presence of with_polarization"
        )
        if has_jones:
            jsrc = source
            assert isinstance(jsrc, JonesPolarizedSource)
            sig = inspect.signature(jsrc.with_polarization)
            assert list(sig.parameters) == ["polarization"], (
                f"{type(source).__name__}: with_polarization signature drift"
            )
