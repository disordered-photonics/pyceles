"""Tangential electric spectra prescribed on a finite reference plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .angular import trapezoidal_weights
from .radial import passive_wavenumber


@dataclass(frozen=True, eq=False)
class FourierPlaneSpectrum:
    """Real transverse Fourier grid with measure q*dq*dalpha.

    ``coeff_te`` and ``coeff_tm`` are tangential electric densities along
    e_phi and e_rho in the plane frame. Thus
    E = integral (g_TE*e_phi + g_TM*(e_rho-q/kz*e_z))*exp(i*K.r) q*dq*dalpha.
    The passive root kz=sqrt(k**2-q**2) decays along local +z and includes
    evanescent waves. Alpha spans one full period with an optional repeated
    endpoint; nonuniform samples use periodic trapezoidal weights.

    Arrays are mutable numerical payloads, as in PlaneWaveSpectrum. Source
    descriptors take ownership of their prescribed boundary arrays.
    """

    alpha: np.ndarray
    q: np.ndarray
    k: complex
    coeff_te: np.ndarray
    coeff_tm: np.ndarray

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=np.float64).reshape(-1)
        q = np.asarray(self.q, dtype=np.float64).reshape(-1)
        if len(alpha) < 2 or not np.all(np.isfinite(alpha)) or not np.all(np.diff(alpha) > 0):
            raise ValueError("alpha must contain at least two finite, increasing samples")
        if alpha[-1] - alpha[0] > 2 * np.pi:
            raise ValueError("alpha must span at most one azimuthal period")
        if len(q) < 2 or not np.all(np.isfinite(q)) or q[0] < 0 or not np.all(np.diff(q) > 0):
            raise ValueError("q must contain at least two finite, increasing, nonnegative samples")
        object.__setattr__(self, "k", passive_wavenumber(self.k))
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "q", q)
        for name in ("coeff_te", "coeff_tm"):
            values = np.asarray(getattr(self, name), dtype=np.complex128)
            if values.shape != (alpha.size, q.size) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {(alpha.size, q.size)}")
            object.__setattr__(self, name, values)
        if np.any(self.kz == 0):
            raise ValueError("Tangential plane data do not determine the grazing kz=0 mode")

    @property
    def kz(self) -> np.ndarray:
        """Passive longitudinal roots, evaluated in complex128."""
        return np.sqrt(self.k**2 - self.q**2 + 0j)

    @property
    def weights(self) -> np.ndarray:
        """Periodic polar Fourier measure, independent of the host index."""
        gaps = np.diff(np.append(self.alpha, self.alpha[0] + 2 * np.pi))
        azimuthal = (gaps + np.roll(gaps, 1)) / 2
        return np.asarray(azimuthal[:, None] * (self.q * trapezoidal_weights(self.q)))

    def wavevectors(self) -> np.ndarray:
        """Plane-frame wavevectors with real transverse components."""
        return np.stack(
            np.broadcast_arrays(
                np.cos(self.alpha)[:, None] * self.q,
                np.sin(self.alpha)[:, None] * self.q,
                self.kz[None, :],
            ),
            axis=-1,
        )

    def electric_vectors(self) -> np.ndarray:
        """Full electric densities transverse to each complex wavevector."""
        ca, sa = np.cos(self.alpha)[:, None], np.sin(self.alpha)[:, None]
        return np.stack(
            (
                -sa * self.coeff_te + ca * self.coeff_tm,
                ca * self.coeff_te + sa * self.coeff_tm,
                -self.q / self.kz * self.coeff_tm,
            ),
            axis=-1,
        )
