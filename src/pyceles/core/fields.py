from __future__ import annotations

"""Compatibility facade for source models and projection helpers.

Canonical implementations live in:
- `pyceles.core.sources`
- `pyceles.core.projection`
"""

from .projection import (
    incident_coeffs_from_angular_spectrum,
    incident_coeffs_from_pwp,
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
    project_source_basis_to_svwf,
    project_source_to_svwf,
    transformation_coefficients,
)
from .sources import (
    AngularSpectrumSource,
    DipoleCollection,
    DipoleSource,
    GaussianBeam,
    PlaneWave,
    Polarization,
    PolarizationInput,
    Source,
    initial_field_plane_wave_pattern_normal_incidence,
    is_normal_incidence,
    polarization_to_jones,
    source_jones,
)

__all__ = [
    "AngularSpectrumSource",
    "DipoleSource",
    "DipoleCollection",
    "GaussianBeam",
    "PlaneWave",
    "Polarization",
    "PolarizationInput",
    "Source",
    "incident_coeffs_from_angular_spectrum",
    "incident_coeffs_from_pwp",
    "incident_coeffs_planewave",
    "incident_coeffs_wavebundle_normal_incidence",
    "initial_field_plane_wave_pattern_normal_incidence",
    "is_normal_incidence",
    "polarization_to_jones",
    "project_source_basis_to_svwf",
    "project_source_to_svwf",
    "source_jones",
    "transformation_coefficients",
]
