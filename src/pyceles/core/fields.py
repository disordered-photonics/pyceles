from __future__ import annotations

"""Compatibility facade for source models and projection helpers.

Canonical implementations live in:
- `pyceles.core.sources`
- `pyceles.core.conversions`
- `pyceles.core.projection`
"""

from .conversions import (
    angular_spectrum_to_svwf_regular,
    pwp_to_svwf_regular,
    svwf_outgoing_to_pwp,
    svwf_regular_to_pwp,
    transformation_coefficients,
)
from .projection import (
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
    project_source_basis_to_svwf,
    project_source_to_svwf,
)
from .sources import (
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
    Polarization,
    PolarizationInput,
    SLMSource,
    Source,
    is_normal_incidence,
    polarization_to_jones,
)

__all__ = [
    "AngularSpectrumSource",
    "BesselBeam",
    "CartesianPolarizedBesselBeam",
    "CartesianPolarizedFocusedLaguerreGaussianBeam",
    "DipoleSource",
    "DipoleCollection",
    "FocusedLaguerreGaussianBeam",
    "GaussianBeam",
    "JonesPolarizedSource",
    "LaguerreGaussianBeam",
    "PlaneWave",
    "SLMSource",
    "Polarization",
    "PolarizationInput",
    "Source",
    "pwp_to_svwf_regular",
    "angular_spectrum_to_svwf_regular",
    "svwf_regular_to_pwp",
    "svwf_outgoing_to_pwp",
    "incident_coeffs_planewave",
    "incident_coeffs_wavebundle_normal_incidence",
    "is_normal_incidence",
    "polarization_to_jones",
    "project_source_basis_to_svwf",
    "project_source_to_svwf",
    "transformation_coefficients",
]
