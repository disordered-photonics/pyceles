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
from .common import _float_triplet_tuple, _real_beam_wavenumber, _validated_int


def _validate_gaussian_parameters(
    *,
    wavelength: float,
    beam_width: float,
    polar_angle: float,
    azimuthal_angle: float,
    amplitude: float,
) -> None:
    """Validate the physical scalars shared by Gaussian source constructors."""
    for name, value in (("wavelength", wavelength), ("beam_width", beam_width)):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"`{name}` must be finite and > 0. Got {value!r}.")
    if not np.isfinite(float(polar_angle)) or not 0.0 <= float(polar_angle) <= np.pi:
        raise ValueError(f"`polar_angle` must be finite and lie in [0, pi]. Got {polar_angle!r}.")
    if not np.isfinite(float(azimuthal_angle)):
        raise ValueError(f"`azimuthal_angle` must be finite. Got {azimuthal_angle!r}.")
    if not np.isfinite(float(amplitude)):
        raise ValueError(f"`amplitude` must be finite. Got {amplitude!r}.")


@dataclass(frozen=True)
class GaussianBeam:
    """Gaussian wavebundle with incident-plane TE/TM Jones polarization.

    The axis-frame x/y directions are e_theta/e_phi, as for PlaneWave;
    azimuth fixes these axes even at normal or backward incidence.
    ``beam_width`` must be finite and positive; use PlaneWave for plane-wave
    illumination.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    beam_width: float = 2000.0
    focal_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        _real_beam_wavenumber(self.medium_n)
        _validate_gaussian_parameters(
            wavelength=self.wavelength,
            beam_width=self.beam_width,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            amplitude=self.amplitude,
        )
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
        return True

    def angular_spectrum(
        self,
        *,
        k: complex,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _gaussian_angular_spectrum_coeffs(
            beam=self,
            k=_real_beam_wavenumber(k),
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
    """Collimated Maxwell-consistent LG beam in the incident-plane frame.

    TE/TM polarization follows GaussianBeam. The vortex azimuth is measured
    from e_theta towards e_phi, with additive phase ``azimuthal_phase``.
    """

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
        _real_beam_wavenumber(self.medium_n)
        _validate_gaussian_parameters(
            wavelength=self.wavelength,
            beam_width=self.beam_width,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            amplitude=self.amplitude,
        )
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )

    def jones_coefficients(self) -> tuple[complex, complex]:
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> LaguerreGaussianBeam:
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        return True

    def angular_spectrum(
        self,
        *,
        k: complex,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _laguerre_gaussian_angular_spectrum_coeffs(
            beam=self,
            k=_real_beam_wavenumber(k),
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
    """Debye/aplanatic focused LG beam in the incident-plane frame.

    Polarization axes and the vortex phase reference follow LaguerreGaussianBeam.
    """

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
        _real_beam_wavenumber(self.medium_n)
        _validate_gaussian_parameters(
            wavelength=self.wavelength,
            beam_width=self.beam_width,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            amplitude=self.amplitude,
        )
        object.__setattr__(
            self, "polarization", _canonicalize_polarization_input(self.polarization)
        )
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )
        f = float(self.focal_length)
        if (not np.isfinite(f)) or (f <= 0.0):
            raise ValueError(f"`focal_length` must be finite and > 0. Got {self.focal_length!r}.")
        na = float(self.numerical_aperture)
        if (not np.isfinite(na)) or (na <= 0.0):
            raise ValueError(
                f"`numerical_aperture` must be finite and > 0. Got {self.numerical_aperture!r}."
            )
        if na >= self.medium_n.real:
            raise ValueError(
                "`numerical_aperture` must be smaller than medium refractive index "
                f"({self.medium_n.real!r}). Got {self.numerical_aperture!r}."
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
        k: complex,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _focused_laguerre_gaussian_angular_spectrum_coeffs(
            beam=self,
            k=_real_beam_wavenumber(k),
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
    """Focused LG beam with lab-frame polarization projected per ray.

    The vortex phase reference follows LaguerreGaussianBeam's incident-plane frame.
    """

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
        _real_beam_wavenumber(self.medium_n)
        _validate_gaussian_parameters(
            wavelength=self.wavelength,
            beam_width=self.beam_width,
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            amplitude=self.amplitude,
        )
        _validated_int("radial_order_p", self.radial_order_p, minimum=0)
        _validated_int("azimuthal_order_l", self.azimuthal_order_l)
        polarization = normalize_global_polarization_vector(self.global_polarization)
        object.__setattr__(
            self,
            "global_polarization",
            tuple(complex(value) for value in polarization),
        )
        if not np.isfinite(float(self.azimuthal_phase)):
            raise ValueError(f"`azimuthal_phase` must be finite. Got {self.azimuthal_phase!r}.")
        object.__setattr__(
            self, "focal_point", _float_triplet_tuple("focal_point", self.focal_point)
        )
        f = float(self.focal_length)
        if (not np.isfinite(f)) or (f <= 0.0):
            raise ValueError(f"`focal_length` must be finite and > 0. Got {self.focal_length!r}.")
        na = float(self.numerical_aperture)
        if (not np.isfinite(na)) or (na <= 0.0):
            raise ValueError(
                f"`numerical_aperture` must be finite and > 0. Got {self.numerical_aperture!r}."
            )
        if na >= self.medium_n.real:
            raise ValueError(
                "`numerical_aperture` must be smaller than medium refractive index "
                f"({self.medium_n.real!r}). Got {self.numerical_aperture!r}."
            )

    def has_finite_incident_power(self) -> bool:
        return True

    def angular_spectrum(
        self,
        *,
        k: complex,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum:
        return _focused_laguerre_cartesian_angular_spectrum_coeffs(
            beam=self,
            k=_real_beam_wavenumber(k),
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
