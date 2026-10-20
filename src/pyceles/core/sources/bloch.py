from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..projection import incident_coeffs_planewave
from .base import PolarizationInput, _canonicalize_polarization_input, polarization_to_jones
from .common import _float_triplet_tuple


@dataclass(frozen=True)
class BlochPlaneWave:
    """Plane-wave component with an explicitly prescribed real Bloch vector.

    ``k_parallel`` sets the lateral phase directly. ``direction`` selects the
    sign of the outgoing longitudinal root, and the field amplitude is defined
    at the finite ``reference_point``. This specifies a transverse-Fourier
    boundary condition without inferring a Bloch vector from an angle in the
    embedding medium. TE/TM polarizations use the bilinear Maxwell convention
    (no complex conjugation in ``k dot E``); at zero transverse wavevector the
    transverse axes are fixed to Cartesian x/y.
    """

    wavelength: float
    k_parallel: tuple[float, float] = (0.0, 0.0)
    medium_n: complex = 1.0 + 0.0j
    polarization: PolarizationInput = "TE"
    direction: Literal["up", "down"] = "down"
    reference_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if not np.isfinite(n) or n.real <= 0.0 or n.imag < 0.0:
            raise ValueError(
                "medium_n must be finite with positive real and nonnegative imaginary parts"
            )
        wavelength = float(self.wavelength)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength must be finite and positive")
        amplitude = complex(self.amplitude)
        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite")

        q_raw = np.asarray(self.k_parallel)
        if np.iscomplexobj(q_raw):
            if np.any(q_raw.imag != 0.0):
                raise ValueError("k_parallel must be real")
            q_raw = q_raw.real
        q = np.asarray(q_raw, dtype=float).reshape(-1)
        if q.size != 2:
            raise ValueError(
                "k_parallel must contain exactly two values. "
                f"Got shape {np.asarray(self.k_parallel).shape}."
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("k_parallel must contain finite values")

        direction = str(self.direction).lower()
        if direction not in {"up", "down"}:
            raise ValueError("direction must be either 'up' or 'down'")

        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "medium_n", n)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "k_parallel", (float(q[0]), float(q[1])))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        object.__setattr__(
            self, "reference_point", _float_triplet_tuple("reference_point", self.reference_point)
        )

    @property
    def direction_sign(self) -> float:
        """Return ``+1`` for upward propagation and ``-1`` for downward."""
        return 1.0 if self.direction == "up" else -1.0

    @property
    def longitudinal_wavenumber(self) -> complex:
        """Return the outgoing root with nonnegative imaginary part."""
        k = 2.0 * np.pi / self.wavelength * complex(self.medium_n)
        qx, qy = self.k_parallel
        # The complex square root selects positive imaginary gamma at cutoffs.
        return complex(np.sqrt(k * k - qx * qx - qy * qy + 0.0j))

    @property
    def wavevector(self) -> np.ndarray:
        """Return the complex wavevector for the selected vertical direction."""
        qx, qy = self.k_parallel
        return np.asarray(
            [qx, qy, self.direction_sign * self.longitudinal_wavenumber],
            dtype=np.complex128,
        )

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return the TE/TM Jones weights, without the overall amplitude."""
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> BlochPlaneWave:
        """Clone the source with new Jones weights and unchanged wavevector."""
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        """Plane-wave components carry infinite incident power in homogeneous media."""
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
        """Project this component to incident regular SVWF coefficients."""
        del polar_angles, azimuthal_angles
        return incident_coeffs_planewave(positions, lmax, self, dtype=dtype)


__all__ = ["BlochPlaneWave"]
