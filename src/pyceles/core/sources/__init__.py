"""Incident-source models and source-side field helpers."""

from __future__ import annotations

from .base import (
    AngularSpectrumSource,
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
from .dipole import DipoleCollection, DipoleSource
from .gaussian import (
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
)
from .plane_wave import PlaneWave
from .slm import SLMSource

__all__ = [
    "AngularSpectrumSource",
    "BesselBeam",
    "CartesianPolarizedBesselBeam",
    "CartesianPolarizedFocusedLaguerreGaussianBeam",
    "DipoleCollection",
    "DipoleSource",
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
