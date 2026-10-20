from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .base import PolarizationInput, _canonicalize_polarization_input, polarization_to_jones
from .common import _float_triplet_tuple

if TYPE_CHECKING:
    from .bloch import BlochPlaneWave


@dataclass(frozen=True)
class PlaneWave:
    """Homogeneous plane wave with amplitude specified at ``focal_point``.

    The amplitude and Jones weights define E at this finite reference point.
    Real propagation angles specify parallel phase and attenuation directions
    in a passive complex host; illumination at infinity is not prescribed.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0

    def __post_init__(self):
        n = complex(self.medium_n)
        if not np.isfinite(n) or n.real <= 0 or n.imag < 0:
            raise ValueError(
                "medium_n must be finite with positive real and nonnegative imaginary parts"
            )
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        if not np.isfinite(self.polar_angle) or not 0 <= self.polar_angle <= np.pi:
            raise ValueError("polar_angle must be finite and between 0 and pi")
        if not np.isfinite(self.azimuthal_angle):
            raise ValueError("azimuthal_angle must be finite")
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return the TE/TM Jones weights, without the overall amplitude."""
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
        from ..projection import incident_coeffs_planewave

        return incident_coeffs_planewave(positions, lmax, self, dtype=dtype)


@dataclass(frozen=True)
class _PlaneWaveGeometry:
    """Analytic direction and reference frame shared by projections and fields."""

    wavevector: np.ndarray
    cos_beta: complex | float
    sin_beta: complex | float
    azimuth: float
    reference_point: tuple[float, float, float]
    direction_sign: float


def _plane_wave_geometry(
    source: PlaneWave | BlochPlaneWave, *, k: complex | None = None
) -> _PlaneWaveGeometry:
    wave_number = 2 * np.pi / source.wavelength * source.medium_n if k is None else k
    if isinstance(source, PlaneWave):
        beta = float(source.polar_angle)
        alpha = float(source.azimuthal_angle)
        # Exact poles have exactly zero transverse wavevector; the source's
        # azimuth still defines its TE/TM axes there.
        cb = (1.0 if beta == 0.0 else -1.0) if beta in (0.0, np.pi) else np.cos(beta)
        sb = 0.0 if beta in (0.0, np.pi) else np.sin(beta)
        vector = wave_number * np.asarray([sb * np.cos(alpha), sb * np.sin(alpha), cb])
        return _PlaneWaveGeometry(
            vector, cb, sb, alpha, source.focal_point, 1.0 if cb >= 0 else -1.0
        )

    qx, qy = source.k_parallel
    rho = float(np.hypot(qx, qy))
    kz = source.direction_sign * np.sqrt(complex(wave_number) ** 2 - rho**2 + 0j)
    return _PlaneWaveGeometry(
        np.asarray([qx, qy, kz], dtype=np.complex128),
        kz / wave_number,
        rho / wave_number,
        float(np.arctan2(qy, qx)) if rho != 0 else 0.0,
        source.reference_point,
        source.direction_sign,
    )
