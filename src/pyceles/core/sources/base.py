from __future__ import annotations

from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..plane_wave_spectrum import PlaneWaveSpectrum

Polarization = Literal["TE", "TM"]
PolarizationInput = Polarization | tuple[complex, complex] | list[complex] | np.ndarray


def polarization_to_jones(polarization: PolarizationInput) -> tuple[complex, complex]:
    """Validate TE/TM Jones weights without changing their amplitude or phase."""
    if isinstance(polarization, str):
        pol = polarization.strip().upper()
        if pol == "TE":
            return 1.0 + 0.0j, 0.0 + 0.0j
        if pol == "TM":
            return 0.0 + 0.0j, 1.0 + 0.0j
        raise ValueError(f"Unsupported polarization string {polarization!r}. Use 'TE' or 'TM'.")

    arr = np.asarray(polarization, dtype=np.complex128).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            "Jones polarization input must contain exactly two complex entries (a_te, a_tm)."
        )
    a_te = complex(arr[0])
    a_tm = complex(arr[1])
    if not (
        np.isfinite(a_te.real)
        and np.isfinite(a_te.imag)
        and np.isfinite(a_tm.real)
        and np.isfinite(a_tm.imag)
    ):
        raise ValueError("Jones polarization entries must be finite complex numbers.")
    if a_te == 0.0 and a_tm == 0.0:
        raise ValueError("At least one Jones polarization entry must be non-zero.")
    return a_te, a_tm


def _canonicalize_polarization_input(
    polarization: PolarizationInput,
) -> Polarization | tuple[complex, complex]:
    """Validate polarization and retain an immutable canonical value."""
    if isinstance(polarization, str):
        pol = cast(Polarization, polarization.strip().upper())
        polarization_to_jones(pol)
        return pol
    return polarization_to_jones(polarization)


def is_normal_incidence(polar_angle: float, *, atol: float = 1e-12) -> bool:
    """Return True when `polar_angle` corresponds to +/- z propagation."""
    return bool(np.isclose(np.sin(float(polar_angle)), 0.0, atol=float(atol)))


@runtime_checkable
class AngularSpectrumSource(Protocol):
    """Source interface exposing a TE/TM angular spectrum."""

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> PlaneWaveSpectrum: ...

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray: ...


@runtime_checkable
class LocalExpansionSource(Protocol):
    """Source interface exposing homogeneous-host local outgoing SVWF emitters.

    This capability is intentionally narrow: it is for sources whose emitted
    field can be represented in the host medium by one or more centered
    outgoing SVWF coefficient blocks plus explicit emitter positions.

    It should not be treated as a universal abstraction for every future local
    emitter workflow. For example, embedded dipoles inside particles may need a
    richer capability if their physically relevant representation involves
    interior regular expansions, shell-dependent coefficients, or reaction-field
    evaluations rather than a plain host-medium outgoing expansion.
    """

    @property
    def wavelength(self) -> float: ...

    @property
    def medium_n(self) -> complex: ...

    @property
    def amplitude(self) -> float: ...

    def has_finite_incident_power(self) -> bool: ...

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray: ...

    def source_positions(self) -> np.ndarray: ...

    def outgoing_coeffs(
        self,
        lmax: int,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray: ...


@runtime_checkable
class Source(Protocol):
    """Unified incident-source protocol."""

    @property
    def wavelength(self) -> float: ...

    @property
    def medium_n(self) -> complex: ...

    @property
    def amplitude(self) -> float: ...

    def has_finite_incident_power(self) -> bool: ...

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray: ...


@runtime_checkable
class JonesPolarizedSource(Source, Protocol):
    """Source interface exposing TE/TM Jones metadata and channel cloning."""

    @property
    def polarization(self) -> PolarizationInput: ...

    def jones_coefficients(self) -> tuple[complex, complex]: ...

    def with_polarization(self, polarization: PolarizationInput) -> JonesPolarizedSource: ...


def finite_power_policy_error(source: Source, *, diagnostic: str) -> ValueError:
    """Build a consistent error for diagnostics requiring finite incident power."""
    cls = type(source).__name__
    method = getattr(source, "has_finite_incident_power", None)
    if cls == "PlaneWave":
        reason = "PlaneWave excitation has infinite incident power."
    elif callable(method):
        beam_width = getattr(source, "beam_width", None)
        if beam_width is not None:
            try:
                w = float(beam_width)
            except (TypeError, ValueError):
                w = np.nan
            if (not np.isfinite(w)) or np.isclose(w, 0.0):
                reason = f"{cls} is in a plane-wave limit (beam_width={beam_width!r})."
            else:
                reason = f"{cls}.has_finite_incident_power() reported non-finite incident power."
        else:
            reason = f"{cls}.has_finite_incident_power() reported non-finite incident power."
    else:
        beam_width = getattr(source, "beam_width", None)
        if beam_width is not None:
            reason = f"{cls} is in a plane-wave limit (beam_width={beam_width!r})."
        else:
            reason = (
                f"{cls} does not advertise finite incident power for beam-normalized diagnostics."
            )
    return ValueError(
        f"{diagnostic} is undefined for infinite-power sources. "
        f"{reason} Use plane-wave cross sections when applicable."
    )


def ensure_finite_power_diagnostics_supported(source: Source, *, diagnostic: str) -> None:
    """Raise a consistent error when a finite-power-only diagnostic is requested."""
    if source.has_finite_incident_power():
        return
    raise finite_power_policy_error(source, diagnostic=diagnostic)
