"""Incident-source models and source-side field helpers."""

from __future__ import annotations

from .base import (
    AngularSpectrumSource,
    FinitePlaneSpectrumSource,
    JonesPolarizedSource,
    LocalExpansionSource,
    Polarization,
    PolarizationInput,
    Source,
    ensure_finite_power_diagnostics_supported,
    finite_power_policy_error,
    is_normal_incidence,
    polarization_to_jones,
)
from .bessel import BesselBeam, CartesianPolarizedBesselBeam
from .bloch import BlochPlaneWave
from .dipole import DipoleCollection, DipoleSource
from .finite_plane import FinitePlaneSource
from .gaussian import (
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
)
from .plane_wave import PlaneWave
from .slm import AngularSpectrumSLMSource, SLMSource

__all__ = [
    "AngularSpectrumSLMSource",
    "AngularSpectrumSource",
    "BesselBeam",
    "BlochPlaneWave",
    "CartesianPolarizedBesselBeam",
    "CartesianPolarizedFocusedLaguerreGaussianBeam",
    "DipoleCollection",
    "DipoleSource",
    "FinitePlaneSource",
    "FinitePlaneSpectrumSource",
    "FocusedLaguerreGaussianBeam",
    "GaussianBeam",
    "JonesPolarizedSource",
    "LaguerreGaussianBeam",
    "LocalExpansionSource",
    "PlaneWave",
    "Polarization",
    "PolarizationInput",
    "SLMSource",
    "Source",
    "ensure_finite_power_diagnostics_supported",
    "finite_power_policy_error",
    "is_normal_incidence",
    "polarization_to_jones",
]
