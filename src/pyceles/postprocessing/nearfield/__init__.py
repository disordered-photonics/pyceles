from __future__ import annotations

"""Canonical near-field implementation package."""

from .classification import InternalPointClassification, classify_internal_points
from .components import (
    NearFieldComponents,
    compute_near_field_components,
    compute_total_field,
    poynting,
)
from .initial import compute_initial_field
from .internal import compute_internal_field
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
    "compute_initial_field",
    "compute_internal_field",
    "compute_near_field",
    "compute_near_field_components",
    "compute_near_field_slice",
    "compute_scattered_field",
    "compute_total_field",
    "mix_near_field_components",
    "mix_near_field_slices",
    "poynting",
]
