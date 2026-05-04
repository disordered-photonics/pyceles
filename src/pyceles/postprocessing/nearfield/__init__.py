"""Canonical near-field implementation package.

The package surface mirrors the physical decomposition used throughout
`pyceles`: initial, scattered, internal, and total fields, plus slice helpers
and point-classification utilities used to decide which contribution is valid
at each observation point.
"""

from __future__ import annotations

from .classification import InternalPointClassification, classify_internal_points
from .common import clear_caches
from .components import (
    NearFieldComponents,
    compute_near_field_components,
    compute_total_field,
    poynting,
)
from .initial import compute_initial_field
from .internal import compute_internal_field
from .periodic import compute_periodic_near_field, compute_periodic_near_field_slice
from .scattered import NearFieldRadialLUT, compute_scattered_field
from .slice import NearFieldSlice
from .workflows import (
    compute_near_field,
    compute_near_field_slice,
    mix_near_field_components,
    mix_near_field_slices,
)

__all__ = [
    "InternalPointClassification",
    "NearFieldComponents",
    "NearFieldRadialLUT",
    "NearFieldSlice",
    "classify_internal_points",
    "clear_caches",
    "compute_initial_field",
    "compute_internal_field",
    "compute_near_field",
    "compute_near_field_components",
    "compute_near_field_slice",
    "compute_periodic_near_field",
    "compute_periodic_near_field_slice",
    "compute_scattered_field",
    "compute_total_field",
    "mix_near_field_components",
    "mix_near_field_slices",
    "poynting",
]
