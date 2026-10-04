from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..conversions import angular_spectrum_to_svwf_regular
from ..plane_wave_spectrum import PlaneWaveSpectrum
from ..polarization import normalize_global_polarization_vector
from .base import PolarizationInput, _canonicalize_polarization_input, polarization_to_jones
from .beam_kernels import _bessel_angular_spectrum_coeffs, _bessel_cartesian_angular_spectrum_coeffs
from .common import _float_triplet_tuple, _validated_int


@dataclass(frozen=True)
class BesselBeam:
    """Exact non-paraxial ideal Bessel beam via ring angular spectrum."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    order_m: int = 0
    cone_angle: float = 0.2
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    polarization: PolarizationInput = "TE"
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_only: bool = True

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "center", _float_triplet_tuple("center", self.center))
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        if not np.isfinite(float(self.polar_angle)):
            raise ValueError(f"`polar_angle` must be finite. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.azimuthal_angle)):
            raise ValueError(f"`azimuthal_angle` must be finite. Got {self.azimuthal_angle!r}.")
        if float(self.polar_angle) < 0.0 or float(self.polar_angle) > np.pi:
            raise ValueError(f"`polar_angle` must lie in [0, pi]. Got {self.polar_angle!r}.")
        object.__setattr__(self, "order_m", _validated_int("order_m", self.order_m))
        cone = float(self.cone_angle)
        if not (cone > 0.0 and cone < np.pi):
            raise ValueError(f"`cone_angle` must lie in (0, pi). Got {self.cone_angle!r}.")
        if bool(self.forward_only) and cone >= (0.5 * np.pi):
            raise ValueError(
                "`forward_only=True` requires `cone_angle < pi/2` (positive kz cone). "
                f"Got cone_angle={self.cone_angle!r}."
            )

    def jones_coefficients(self) -> tuple[complex, complex]:
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> BesselBeam:
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        return False

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _bessel_angular_spectrum_coeffs(
            beam=self,
            k=float(k),
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
            polarization_override=None,
        )

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        if polar_angles is None or azimuthal_angles is None:
            raise ValueError(
                "BesselBeam projection requires both `polar_angles` and `azimuthal_angles`."
            )
        k = 2.0 * np.pi / float(self.wavelength) * float(np.real(complex(self.medium_n)))
        return angular_spectrum_to_svwf_regular(
            positions,
            lmax,
            self,
            k=k,
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
            dtype=dtype,
        )


@dataclass(frozen=True)
class CartesianPolarizedBesselBeam:
    """Ideal Bessel beam with one global polarization state across the cone."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    order_m: int = 0
    cone_angle: float = 0.2
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    global_polarization: tuple[complex, complex, complex] = (1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_only: bool = True

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if abs(n.imag) > 0:
            raise ValueError(
                "Embedding medium refractive index must be real for an incident field coming from infinity. "
                f"Got medium_n={n!r}"
            )
        if not (n.real > 0):
            raise ValueError(f"medium_n must be positive. Got {n!r}")
        polarization = normalize_global_polarization_vector(self.global_polarization)
        object.__setattr__(
            self,
            "global_polarization",
            tuple(complex(value) for value in polarization),
        )
        object.__setattr__(self, "center", _float_triplet_tuple("center", self.center))
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        if not np.isfinite(float(self.polar_angle)):
            raise ValueError(f"`polar_angle` must be finite. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.azimuthal_angle)):
            raise ValueError(f"`azimuthal_angle` must be finite. Got {self.azimuthal_angle!r}.")
        if float(self.polar_angle) < 0.0 or float(self.polar_angle) > np.pi:
            raise ValueError(f"`polar_angle` must lie in [0, pi]. Got {self.polar_angle!r}.")
        object.__setattr__(self, "order_m", _validated_int("order_m", self.order_m))
        cone = float(self.cone_angle)
        if not (cone > 0.0 and cone < np.pi):
            raise ValueError(f"`cone_angle` must lie in (0, pi). Got {self.cone_angle!r}.")
        if bool(self.forward_only) and cone >= (0.5 * np.pi):
            raise ValueError(
                "`forward_only=True` requires `cone_angle < pi/2` (positive kz cone). "
                f"Got cone_angle={self.cone_angle!r}."
            )

    def has_finite_incident_power(self) -> bool:
        return False

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _bessel_cartesian_angular_spectrum_coeffs(
            beam=self,
            k=float(k),
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        )

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        if polar_angles is None or azimuthal_angles is None:
            raise ValueError(
                "CartesianPolarizedBesselBeam projection requires both "
                "`polar_angles` and `azimuthal_angles`."
            )
        k = 2.0 * np.pi / float(self.wavelength) * float(np.real(complex(self.medium_n)))
        return angular_spectrum_to_svwf_regular(
            positions,
            lmax,
            self,
            k=k,
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
            dtype=dtype,
        )
