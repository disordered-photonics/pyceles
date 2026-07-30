"""Public far-field package for PWPs, power diagnostics, and cross sections.

The package surface is organized by physical ownership:
- `patterns`: assemble scattered/initial/total plane-wave spectra
- `power`: finite-power beam flux and decomposition diagnostics
- `cross_sections`: plane-wave-normalized scattering/extinction/absorption
"""

from __future__ import annotations

from .cross_sections import (
    CrossSectionBalance,
    extinction_cross_section,
    local_absorption_cross_section_from_exciting,
    plane_wave_cross_section_balance,
    scattering_cross_section,
    total_scattering_cross_section,
)
from .orders import DiffractionOrders, enumerate_diffraction_orders_rectangular
from .patterns import (
    FarFieldPatterns,
    compute_far_field_patterns,
    scattered_field_plane_wave_pattern,
    total_field_plane_wave_pattern,
)
from .periodic import (
    PeriodicFarFieldPayload,
    PeriodicOrderAmplitudes,
    mix_periodic_farfield_payloads,
    periodic_order_amplitudes,
    periodic_plane_wave_orders,
)
from .power import (
    PowerBalance,
    finite_beam_power_balance,
    incident_power_from_pwp,
    local_absorbed_power_from_exciting,
    local_power_balance_from_exciting,
    pwp_power_decomposition,
    pwp_power_flux,
)

__all__ = [
    "CrossSectionBalance",
    "DiffractionOrders",
    "FarFieldPatterns",
    "PeriodicFarFieldPayload",
    "PeriodicOrderAmplitudes",
    "PowerBalance",
    "compute_far_field_patterns",
    "enumerate_diffraction_orders_rectangular",
    "extinction_cross_section",
    "finite_beam_power_balance",
    "incident_power_from_pwp",
    "local_absorbed_power_from_exciting",
    "local_absorption_cross_section_from_exciting",
    "local_power_balance_from_exciting",
    "mix_periodic_farfield_payloads",
    "periodic_order_amplitudes",
    "periodic_plane_wave_orders",
    "plane_wave_cross_section_balance",
    "pwp_power_decomposition",
    "pwp_power_flux",
    "scattered_field_plane_wave_pattern",
    "scattering_cross_section",
    "total_field_plane_wave_pattern",
    "total_scattering_cross_section",
]
