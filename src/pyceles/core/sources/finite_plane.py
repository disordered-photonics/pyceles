"""Sources defined by tangential Fourier data on a finite plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyceles._arrays import owned_read_only_view

from ..conversions import fourier_plane_to_svwf_regular
from ..fourier_plane import FourierPlaneSpectrum
from .common import _float_triplet_tuple


@dataclass(frozen=True, eq=False)
class FinitePlaneSource:
    """Tangential electric Fourier data fixed at a finite reference plane.

    The plane passes through ``reference_point`` with normal specified by
    ``polar_angle`` and ``azimuthal_angle``. Its local frame is Rz(azimuth)
    Ry(polar): local x is e_theta and local y is e_phi of the normal.
    ``transverse_wavenumbers`` are real radial frequencies q in inverse
    length units; ``spectrum_azimuths`` sample a full plane-frame period.
    TE/TM densities prescribe E_phi/E_rho with measure q*dq*dalpha.

    For example E_y=exp(-rho**2/w**2) has g_TE=cos(alpha)*G and
    g_TM=sin(alpha)*G, with G=w**2/(4*pi)*exp(-w**2*q**2/4), before
    truncation and quadrature. Pupil and phase masks act on these arrays.

    Passive longitudinal roots propagate towards local +z. Finite upstream
    evaluation is analytic continuation and can grow for evanescent waves.
    No material sheet or reflection is modeled. Incident far-angle spectra
    and power at infinity are not inferred from these boundary data.
    """

    wavelength: float
    transverse_wavenumbers: np.ndarray
    spectrum_azimuths: np.ndarray
    coeff_te: np.ndarray
    coeff_tm: np.ndarray
    medium_n: complex = 1.0
    reference_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    amplitude: complex = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive")
        if not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        if not np.isfinite(self.polar_angle) or not 0 <= self.polar_angle <= np.pi:
            raise ValueError("polar_angle must be finite and between 0 and pi")
        if not np.isfinite(self.azimuthal_angle):
            raise ValueError("azimuthal_angle must be finite")
        object.__setattr__(
            self, "reference_point", _float_triplet_tuple("reference_point", self.reference_point)
        )
        spectrum = FourierPlaneSpectrum(
            self.spectrum_azimuths,
            self.transverse_wavenumbers,
            2 * np.pi / self.wavelength * self.medium_n,
            self.coeff_te,
            self.coeff_tm,
        )
        for name, values in (
            ("transverse_wavenumbers", spectrum.q),
            ("spectrum_azimuths", spectrum.alpha),
            ("coeff_te", spectrum.coeff_te),
            ("coeff_tm", spectrum.coeff_tm),
        ):
            object.__setattr__(self, name, owned_read_only_view(values, dtype=values.dtype))

    def has_finite_incident_power(self) -> bool:
        """Sampled Fourier densities prescribe a finite plane profile."""
        return True

    def plane_spectrum(self) -> FourierPlaneSpectrum:
        """Return the boundary including its complex amplitude."""
        return FourierPlaneSpectrum(
            self.spectrum_azimuths,
            self.transverse_wavenumbers,
            2 * np.pi / self.wavelength * self.medium_n,
            self.amplitude * self.coeff_te,
            self.amplitude * self.coeff_tm,
        )

    def plane_frame(self) -> np.ndarray:
        """Return the plane-to-lab Cartesian rotation."""
        ca, sa = np.cos(self.azimuthal_angle), np.sin(self.azimuthal_angle)
        cb, sb = np.cos(self.polar_angle), np.sin(self.polar_angle)
        return np.array([[ca * cb, -sa, ca * sb], [sa * cb, ca, sa * sb], [-sb, 0, cb]])

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        # Boundary sampling belongs to the source, independently of far-field bins.
        local = (
            np.asarray(positions, dtype=np.float64) - self.reference_point
        ) @ self.plane_frame()
        return fourier_plane_to_svwf_regular(
            local,
            lmax,
            spectrum=self.plane_spectrum(),
            polar_angle=self.polar_angle,
            azimuthal_angle=self.azimuthal_angle,
            dtype=dtype,
        )
