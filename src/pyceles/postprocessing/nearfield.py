from __future__ import annotations

"""Compatibility facade for near-field kernels and workflows."""

from .nearfield_kernels import (
    NearFieldRadialLUT,
    compute_initial_field,
    compute_internal_field,
    compute_scattered_field,
)
from .nearfield_workflows import (
    NearFieldComponents,
    compute_near_field_components,
    compute_total_field,
    poynting,
)

__all__ = [
    "NearFieldComponents",
    "NearFieldRadialLUT",
    "compute_initial_field",
    "compute_internal_field",
    "compute_near_field_components",
    "compute_scattered_field",
    "compute_total_field",
    "poynting",
]
