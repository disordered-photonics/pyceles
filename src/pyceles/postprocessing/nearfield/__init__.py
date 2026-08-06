"""Canonical near-field implementation package.

The package surface mirrors the physical decomposition used throughout
`pyceles`: initial, scattered, internal, and total fields, plus slice helpers
used to inspect those contributions at arbitrary observation points.
"""

from __future__ import annotations

from .common import clear_caches
from .components import (
    ElectricFieldComponents,
    NearFieldComponents,
    compute_electric_field_components,
    compute_near_field_components,
    compute_total_field,
    poynting,
)
from .initial import compute_initial_field
from .internal import compute_internal_field
from .periodic import (
    compute_periodic_near_field,
    compute_periodic_near_field_slice,
)
from .scattered import (
    NearFieldRadialLUT,
    compute_scattered_electric_field,
    compute_scattered_field,
)
from .slice import NearFieldSlice
from .workflows import (
    compute_electric_field,
    compute_near_field,
    compute_near_field_slice,
    mix_near_field_components,
    mix_near_field_slices,
)

__all__ = [
    "ElectricFieldComponents",
    "NearFieldComponents",
    "NearFieldRadialLUT",
    "NearFieldSlice",
    "clear_caches",
    "compute_electric_field",
    "compute_electric_field_components",
    "compute_initial_field",
    "compute_internal_field",
    "compute_near_field",
    "compute_near_field_components",
    "compute_near_field_slice",
    "compute_periodic_near_field",
    "compute_periodic_near_field_slice",
    "compute_scattered_electric_field",
    "compute_scattered_field",
    "compute_total_field",
    "mix_near_field_components",
    "mix_near_field_slices",
    "poynting",
]
