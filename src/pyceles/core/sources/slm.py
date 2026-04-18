from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..conversions import angular_spectrum_to_svwf_regular
from .base import AngularSpectrumSource, JonesPolarizedSource, PolarizationInput, Source

_SLMModulation = complex | np.ndarray | Callable[[np.ndarray, np.ndarray], npt.ArrayLike]


@dataclass(frozen=True)
class SLMSource:
    """Angular-spectrum wrapper that applies an SLM-style complex modulation."""

    base_source: JonesPolarizedSource
    modulation: _SLMModulation = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if not isinstance(self.base_source, Source):
            raise TypeError(
                "`base_source` must satisfy the pyceles Source protocol "
                "(wavelength/medium_n + incident_coeffs + has_finite_incident_power APIs)."
            )
        if not isinstance(self.base_source, JonesPolarizedSource):
            raise TypeError(
                "`base_source` must expose Jones metadata and `with_polarization(...)` for SLM wrapping."
            )
        if not isinstance(self.base_source, AngularSpectrumSource):
            raise TypeError("`base_source` must expose `angular_spectrum(...)` for SLM modulation.")
        if not callable(self.modulation):
            weights = np.asarray(self.modulation, dtype=np.complex128)
            if not np.all(np.isfinite(weights.real)) or not np.all(np.isfinite(weights.imag)):
                raise ValueError("`modulation` array/scalar must be finite.")

    @property
    def wavelength(self) -> float:
        return float(self.base_source.wavelength)

    @property
    def medium_n(self) -> complex:
        return complex(self.base_source.medium_n)

    @property
    def amplitude(self) -> float:
        return float(self.base_source.amplitude)

    @property
    def polarization(self) -> PolarizationInput:
        return self.base_source.polarization

    @property
    def beam_width(self) -> float:
        """Delegate beam width for finite-power diagnostics when available."""
        return float(getattr(self.base_source, "beam_width", np.inf))

    def jones_coefficients(self) -> tuple[complex, complex]:
        return self.base_source.jones_coefficients()

    def with_polarization(self, polarization: PolarizationInput) -> SLMSource:
        return replace(self, base_source=self.base_source.with_polarization(polarization))

    def has_finite_incident_power(self) -> bool:
        return bool(self.base_source.has_finite_incident_power())

    def _modulation_weights(
        self, *, alpha: np.ndarray, beta: np.ndarray, dtype: npt.DTypeLike
    ) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        agrid, bgrid = np.meshgrid(alpha_arr, beta_arr, indexing="ij")
        w_raw = self.modulation(agrid, bgrid) if callable(self.modulation) else self.modulation
        w = np.asarray(w_raw, dtype=np.dtype(dtype))
        try:
            w = np.broadcast_to(w, agrid.shape)
        except ValueError as exc:
            raise ValueError(
                "`modulation` must be broadcastable to angular grid shape "
                f"{agrid.shape}. Got {w.shape}."
            ) from exc
        if not np.all(np.isfinite(w.real)) or not np.all(np.isfinite(w.imag)):
            raise ValueError("`modulation` must evaluate to finite complex weights.")
        return np.asarray(w, dtype=np.dtype(dtype))

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> tuple[dict, dict]:
        base = self.base_source
        if not isinstance(base, AngularSpectrumSource):
            raise TypeError("SLMSource requires a base source with `angular_spectrum(...)`.")
        pwp_te, pwp_tm = base.angular_spectrum(
            k=float(k),
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        )
        coeff_dtype = np.result_type(pwp_te["coeff"], pwp_tm["coeff"], np.complex64)
        weights = self._modulation_weights(
            alpha=np.asarray(pwp_te["alpha"], dtype=float),
            beta=np.asarray(pwp_te["beta"], dtype=float),
            dtype=coeff_dtype,
        )
        out_te = dict(pwp_te)
        out_tm = dict(pwp_tm)
        out_te["coeff"] = np.asarray(np.asarray(pwp_te["coeff"]) * weights, dtype=coeff_dtype)
        out_tm["coeff"] = np.asarray(np.asarray(pwp_tm["coeff"]) * weights, dtype=coeff_dtype)
        return out_te, out_tm

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
                "SLMSource projection requires both `polar_angles` and `azimuthal_angles`."
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
