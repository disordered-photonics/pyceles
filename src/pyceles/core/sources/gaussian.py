from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..conversions import angular_spectrum_to_svwf_regular
from ..plane_wave_spectrum import PlaneWaveSpectrum
from ..polarization import normalize_global_polarization_vector
from ..projection import incident_coeffs_wavebundle_normal_incidence
from .base import (
    PolarizationInput,
    _canonicalize_polarization_input,
    is_normal_incidence,
    polarization_to_jones,
)
from .beam_kernels import (
    _focused_laguerre_cartesian_angular_spectrum_coeffs,
    _focused_laguerre_gaussian_angular_spectrum_coeffs,
    _gaussian_angular_spectrum_coeffs,
    _laguerre_gaussian_angular_spectrum_coeffs,
)
from .common import _float_triplet_tuple, _validated_int


@dataclass(frozen=True)
class GaussianBeam:
    """Gaussian wavebundle source in a homogeneous medium."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    beam_width: float = np.inf
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0

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
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )

    def jones_coefficients(self) -> tuple[complex, complex]:
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> GaussianBeam:
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        w = float(self.beam_width)
        return bool(np.isfinite(w) and (not np.isclose(w, 0.0)))

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _gaussian_angular_spectrum_coeffs(
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
        if polar_angles is None:
            raise ValueError("GaussianBeam projection requires `polar_angles`.")
        if is_normal_incidence(float(self.polar_angle)):
            return incident_coeffs_wavebundle_normal_incidence(
                positions,
                lmax,
                self,
                np.asarray(polar_angles, float),
                dtype=dtype,
            )
        if azimuthal_angles is None:
            raise ValueError(
                "Tilted GaussianBeam projection requires both `polar_angles` and `azimuthal_angles`."
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
class LaguerreGaussianBeam:
    """Collimated Maxwell-consistent Laguerre-Gaussian beam via angular spectrum."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    radial_order_p: int = 0
    azimuthal_order_l: int = 0
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    beam_width: float = 2000.0
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if abs(n.imag) > 0:
            raise ValueError(
                "Embedding medium refractive index must be real for an incident field coming "
                f"from infinity. Got medium_n={n!r}"
            )
        if not (n.real > 0):
            raise ValueError(f"medium_n must be positive. Got {n!r}")
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        if not np.isfinite(float(self.polar_angle)):
            raise ValueError(f"`polar_angle` must be finite. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.azimuthal_angle)):
            raise ValueError(f"`azimuthal_angle` must be finite. Got {self.azimuthal_angle!r}.")
        if float(self.polar_angle) < 0.0 or float(self.polar_angle) > np.pi:
            raise ValueError(f"`polar_angle` must lie in [0, pi]. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )
        w = float(self.beam_width)
        if (not np.isfinite(w)) or (w <= 0.0):
            raise ValueError(f"`beam_width` must be finite and > 0. Got {self.beam_width!r}.")

    def jones_coefficients(self) -> tuple[complex, complex]:
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> LaguerreGaussianBeam:
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        return True

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _laguerre_gaussian_angular_spectrum_coeffs(
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
                "LaguerreGaussianBeam projection requires both `polar_angles` and "
                "`azimuthal_angles`."
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
class FocusedLaguerreGaussianBeam:
    """Debye/aplanatic focused Laguerre-Gaussian beam via angular spectrum."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    radial_order_p: int = 0
    azimuthal_order_l: int = 0
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    beam_width: float = 1000.0
    focal_length: float = 1000.0
    numerical_aperture: float = 0.8
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0
    sine_condition_apodization: bool = True

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if abs(n.imag) > 0:
            raise ValueError(
                "Embedding medium refractive index must be real for an incident field coming "
                f"from infinity. Got medium_n={n!r}"
            )
        if not (n.real > 0):
            raise ValueError(f"medium_n must be positive. Got {n!r}")
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        if not np.isfinite(float(self.polar_angle)):
            raise ValueError(f"`polar_angle` must be finite. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.azimuthal_angle)):
            raise ValueError(f"`azimuthal_angle` must be finite. Got {self.azimuthal_angle!r}.")
        if float(self.polar_angle) < 0.0 or float(self.polar_angle) > np.pi:
            raise ValueError(f"`polar_angle` must lie in [0, pi]. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )
        w = float(self.beam_width)
        if (not np.isfinite(w)) or (w <= 0.0):
            raise ValueError(f"`beam_width` must be finite and > 0. Got {self.beam_width!r}.")
        f = float(self.focal_length)
        if (not np.isfinite(f)) or (f <= 0.0):
            raise ValueError(f"`focal_length` must be finite and > 0. Got {self.focal_length!r}.")
        na = float(self.numerical_aperture)
        if (not np.isfinite(na)) or (na <= 0.0):
            raise ValueError(
                f"`numerical_aperture` must be finite and > 0. Got {self.numerical_aperture!r}."
            )
        if na >= float(n.real):
            raise ValueError(
                "`numerical_aperture` must be smaller than medium refractive index "
                f"({float(n.real)!r}). Got {self.numerical_aperture!r}."
            )

    def jones_coefficients(self) -> tuple[complex, complex]:
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> FocusedLaguerreGaussianBeam:
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        return True

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _focused_laguerre_gaussian_angular_spectrum_coeffs(
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
                "FocusedLaguerreGaussianBeam projection requires both `polar_angles` and "
                "`azimuthal_angles`."
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
class CartesianPolarizedFocusedLaguerreGaussianBeam:
    """Debye/aplanatic focused LG beam with one lab-frame polarization state."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    radial_order_p: int = 0
    azimuthal_order_l: int = 0
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    global_polarization: tuple[complex, complex, complex] = (
        1.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
    )
    beam_width: float = 1000.0
    focal_length: float = 1000.0
    numerical_aperture: float = 0.8
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0
    sine_condition_apodization: bool = True

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if abs(n.imag) > 0:
            raise ValueError(
                "Embedding medium refractive index must be real for an incident field coming "
                f"from infinity. Got medium_n={n!r}"
            )
        if not (n.real > 0):
            raise ValueError(f"medium_n must be positive. Got {n!r}")
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        if not np.isfinite(float(self.polar_angle)):
            raise ValueError(f"`polar_angle` must be finite. Got {self.polar_angle!r}.")
        if not np.isfinite(float(self.azimuthal_angle)):
            raise ValueError(f"`azimuthal_angle` must be finite. Got {self.azimuthal_angle!r}.")
        if float(self.polar_angle) < 0.0 or float(self.polar_angle) > np.pi:
            raise ValueError(f"`polar_angle` must lie in [0, pi]. Got {self.polar_angle!r}.")
        polarization = normalize_global_polarization_vector(self.global_polarization)
        object.__setattr__(
            self,
            "global_polarization",
            tuple(complex(value) for value in polarization),
        )
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )
        w = float(self.beam_width)
        if (not np.isfinite(w)) or (w <= 0.0):
            raise ValueError(f"`beam_width` must be finite and > 0. Got {self.beam_width!r}.")
        f = float(self.focal_length)
        if (not np.isfinite(f)) or (f <= 0.0):
            raise ValueError(f"`focal_length` must be finite and > 0. Got {self.focal_length!r}.")
        na = float(self.numerical_aperture)
        if (not np.isfinite(na)) or (na <= 0.0):
            raise ValueError(
                f"`numerical_aperture` must be finite and > 0. Got {self.numerical_aperture!r}."
            )
        if na >= float(n.real):
            raise ValueError(
                "`numerical_aperture` must be smaller than medium refractive index "
                f"({float(n.real)!r}). Got {self.numerical_aperture!r}."
            )

    def has_finite_incident_power(self) -> bool:
        return True

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _focused_laguerre_cartesian_angular_spectrum_coeffs(
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
                "CartesianPolarizedFocusedLaguerreGaussianBeam projection requires both "
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
