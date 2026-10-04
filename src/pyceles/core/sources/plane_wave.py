from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..projection import incident_coeffs_planewave
from .base import PolarizationInput, _canonicalize_polarization_input, polarization_to_jones
from .common import _float_triplet_tuple


@dataclass(frozen=True)
class PlaneWave:
    """Monochromatic plane-wave source in a homogeneous medium."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0

    def __post_init__(self):
        n = complex(self.medium_n)
        if abs(n.imag) > 0:
            raise ValueError(
                "Embedding medium refractive index must be real for an incident field coming from infinity. "
                f"Got medium_n={n!r}"
            )
        if not (n.real > 0):
            raise ValueError(f"medium_n must be positive. Got {n!r}")
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return the TE/TM Jones weights, including amplitude and phase."""
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> PlaneWave:
        """Clone plane wave with new polarization and unchanged propagation."""
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        """Plane waves carry infinite incident power in homogeneous media."""
        return False

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Project plane-wave source to incident SVWF coefficients."""
        return incident_coeffs_planewave(positions, lmax, self, dtype=dtype)
