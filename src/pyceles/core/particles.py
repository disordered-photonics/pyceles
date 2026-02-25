from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Particle:
    """Base particle descriptor.

    Notes
    -----
    Only `Sphere` is currently supported in the active numerical kernels.
    Other shapes are included as explicit placeholders to expose planned API.
    """

    position: Tuple[float, float, float]

    def pos_array(self, dtype=float) -> np.ndarray:
        """Return particle center as numeric array for kernel consumption."""
        return np.asarray(self.position, dtype=dtype)

    @property
    def kind(self) -> str:
        """Lower-case particle type label used by dispatch utilities."""
        return type(self).__name__.lower()

    def circumscribing_radius(self) -> float:
        """Radius of a sphere enclosing the particle (for broad-phase geometry)."""
        raise NotImplementedError(f"{type(self).__name__} must implement circumscribing_radius().")


@dataclass(frozen=True)
class Sphere(Particle):
    """Homogeneous sphere supported by current scattering kernels."""

    radius: float
    refractive_index: complex = 1.5 + 0j

    def circumscribing_radius(self) -> float:
        """For spheres the circumscribing radius is the physical radius."""
        return float(self.radius)


@dataclass(frozen=True)
class LayeredSphere(Particle):
    """Placeholder for future multilayer Mie support."""

    layer_radii: Tuple[float, ...]
    layer_refractive_indices: Tuple[complex, ...]

    def __post_init__(self) -> None:
        if len(self.layer_radii) == 0:
            raise ValueError("layer_radii must be non-empty.")
        if len(self.layer_radii) != len(self.layer_refractive_indices):
            raise ValueError("layer_radii and layer_refractive_indices must have the same length.")
        if any(float(r) <= 0.0 for r in self.layer_radii):
            raise ValueError("All layer radii must be positive.")
        if any(
            self.layer_radii[i] <= self.layer_radii[i - 1] for i in range(1, len(self.layer_radii))
        ):
            raise ValueError("layer_radii must be strictly increasing (inner to outer).")

    def circumscribing_radius(self) -> float:
        """Outermost shell radius, useful for overlap checks and bounding boxes."""
        return float(self.layer_radii[-1])


@dataclass(frozen=True)
class Ellipsoid(Particle):
    """Placeholder for future ellipsoidal T-matrix support."""

    semi_axes: Tuple[float, float, float]
    refractive_index: complex = 1.5 + 0j
    euler_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.semi_axes) != 3:
            raise ValueError("semi_axes must be a 3-tuple (a, b, c).")
        if any(float(a) <= 0.0 for a in self.semi_axes):
            raise ValueError("All semi-axes must be positive.")

    def circumscribing_radius(self) -> float:
        """Largest semi-axis, i.e. radius of the minimal enclosing sphere."""
        return float(max(self.semi_axes))
