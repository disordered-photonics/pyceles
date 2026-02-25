from __future__ import annotations

"""Incident-source models and source-side field helpers."""

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, Tuple

import numpy as np
import numpy.typing as npt

from .angular import beam_axis_and_frame
from .projection import (
    incident_coeffs_from_angular_spectrum,
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
)

Polarization = Literal["TE", "TM"]
PolarizationInput = Polarization | tuple[complex, complex] | list[complex] | np.ndarray


def polarization_to_jones(polarization: PolarizationInput) -> tuple[complex, complex]:
    """Normalize user polarization input to Jones-like TE/TM weights.

    Accepted input forms:
    - ``"TE"`` or ``"TM"``
    - complex pair ``(a_te, a_tm)`` (or length-2 list/array)
    """
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
    if np.isclose(abs(a_te), 0.0) and np.isclose(abs(a_tm), 0.0):
        raise ValueError("At least one Jones polarization entry must be non-zero.")
    return a_te, a_tm


def source_jones(source: Any) -> tuple[complex, complex]:
    """Return source polarization as Jones-like TE/TM weights."""
    if hasattr(source, "jones_coefficients"):
        return source.jones_coefficients()
    return polarization_to_jones(getattr(source, "polarization", "TE"))


def _pure_polarization_label(
    a_te: complex, a_tm: complex, *, atol: float = 1e-15
) -> Polarization | None:
    """Return pure-channel label when Jones weights represent TE-only or TM-only."""
    if abs(a_tm) <= atol and abs(a_te) > atol:
        return "TE"
    if abs(a_te) <= atol and abs(a_tm) > atol:
        return "TM"
    return None


def is_normal_incidence(polar_angle: float, *, atol: float = 1e-12) -> bool:
    """Return True when `polar_angle` corresponds to +/- z propagation."""
    return bool(np.isclose(np.sin(float(polar_angle)), 0.0, atol=float(atol)))


class AngularSpectrumSource(Protocol):
    """Source interface exposing a TE/TM angular spectrum."""

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> tuple[dict, dict]:
        """Return `(pwp_te, pwp_tm)` with `coeff` arrays shaped (Na, Nb)."""
        ...

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Project this source directly to incident SVWF coefficients."""
        ...


class Source(Protocol):
    """Unified incident-source protocol (channel-aware TE/TM excitation)."""

    @property
    def wavelength(self) -> float: ...

    @property
    def medium_n(self) -> complex: ...

    @property
    def amplitude(self) -> float: ...

    @property
    def polarization(self) -> PolarizationInput: ...

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return source polarization weights in the TE/TM basis."""
        ...

    def with_polarization(self, polarization: PolarizationInput) -> "Source":
        """Return a source copy with updated polarization state."""
        ...

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Project this source to incident SVWF coefficients on all spheres."""
        ...


def _gaussian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> tuple[dict, dict]:
    """Evaluate Gaussian-beam TE/TM angular-spectrum coefficients on a grid."""
    if polarization_override is None:
        a_te, a_tm = source_jones(beam)
        pure = _pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_te, te_tm = _gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_te, tm_tm = _gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TM",
            )
            pwp_te = dict(te_te)
            pwp_tm = dict(te_tm)
            pwp_te["coeff"] = a_te * np.asarray(te_te["coeff"]) + a_tm * np.asarray(tm_te["coeff"])
            pwp_tm["coeff"] = a_te * np.asarray(te_tm["coeff"]) + a_tm * np.asarray(tm_tm["coeff"])
            return pwp_te, pwp_tm
        polarization_override = pure

    beta = np.asarray(polar_angles, float)
    alpha = np.asarray(azimuthal_angles, float)

    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb

    sx = sb * ca
    sy = sb * sa
    sz = cb
    s = np.stack([sx, sy, sz], axis=2)  # (Na,Nb,3)

    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)

    rho2_l = np.maximum(0.0, 1.0 - sz_l**2)
    alpha_l = np.arctan2(sy_l, sx_l)
    sin_beta_l = np.sqrt(rho2_l)
    cos_beta_l = sz_l

    RG = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), float)
    E0 = float(getattr(beam, "amplitude", 1.0))
    w = float(getattr(beam, "beam_width", np.inf))
    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()

    if pol == "te":
        alpha_pol = float(getattr(beam, "azimuthal_angle", 0.0))
    else:
        alpha_pol = float(getattr(beam, "azimuthal_angle", 0.0)) - np.pi / 2.0

    phase = np.exp(-1j * (kx * RG[0] + ky * RG[1] + kz * RG[2]))
    pref = E0 * (k**2) * (w**2) / (4.0 * np.pi)
    envelope = pref * cos_beta_l * np.exp(-(w**2) / 4.0 * (k**2) * rho2_l)
    envelope = envelope * (cos_beta_l > 0.0)

    g_te_l = np.cos(alpha_l - alpha_pol) * envelope
    g_tm_l = np.sin(alpha_l - alpha_pol) * envelope

    # Global TE/TM basis relative to z axis.
    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)

    # Local TE/TM basis (TE=e_phi_l, TM=e_theta_l) rotated to global frame.
    sin_alpha_l = np.sin(alpha_l)
    cos_alpha_l = np.cos(alpha_l)
    ephi_l = (-sin_alpha_l)[..., None] * u[None, None, :] + cos_alpha_l[..., None] * v[
        None, None, :
    ]
    etheta_l = (
        (cos_beta_l * cos_alpha_l)[..., None] * u[None, None, :]
        + (cos_beta_l * sin_alpha_l)[..., None] * v[None, None, :]
        - sin_beta_l[..., None] * n0[None, None, :]
    )

    m11 = np.einsum("abi,abi->ab", ephi_g, ephi_l)
    m12 = np.einsum("abi,abi->ab", ephi_g, etheta_l)
    m21 = np.einsum("abi,abi->ab", etheta_g, ephi_l)
    m22 = np.einsum("abi,abi->ab", etheta_g, etheta_l)

    coeff_te = (m11 * g_te_l + m12 * g_tm_l) * phase
    coeff_tm = (m21 * g_te_l + m22 * g_tm_l) * phase

    pwp_te = {"beta": beta, "alpha": alpha, "kx": kx, "ky": ky, "kz": kz, "coeff": coeff_te}
    pwp_tm = {"beta": beta, "alpha": alpha, "kx": kx, "ky": ky, "kz": kz, "coeff": coeff_tm}
    return pwp_te, pwp_tm


@dataclass(frozen=True)
class GaussianBeam:
    """Gaussian wavebundle source in a homogeneous medium.

    Notes
    -----
    This source supports arbitrary beam-axis tilts.
    For normal incidence (`polar_angle` equal to 0 or pi), the code keeps a
    dedicated optimized projection path.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    beam_width: float = np.inf
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        polarization_to_jones(self.polarization)

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return normalized TE/TM Jones weights for this beam state."""
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> "GaussianBeam":
        """Clone beam with a new polarization while keeping geometric settings."""
        return replace(self, polarization=polarization)

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> tuple[dict, dict]:
        """Return TE/TM plane-wave spectrum sampled on (`alpha`,`beta`) grids."""
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
        """Project Gaussian source to incident SVWF coefficients for all spheres."""
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
        return incident_coeffs_from_angular_spectrum(
            positions,
            lmax,
            self,
            k=k,
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
            dtype=dtype,
        )


@dataclass(frozen=True)
class PlaneWave:
    """Monochromatic plane-wave source in a homogeneous medium."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    polarization: PolarizationInput = "TE"
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        polarization_to_jones(self.polarization)

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return normalized TE/TM Jones weights for this plane wave."""
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> "PlaneWave":
        """Clone plane wave with new polarization and unchanged propagation."""
        return replace(self, polarization=polarization)

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


def initial_field_plane_wave_pattern_normal_incidence(
    *,
    beam,
    k: float,
    polar_angles,
    azimuthal_angles,
    polarization_override: Polarization | None = None,
):
    """Initial field plane-wave pattern for a normal-incidence Gaussian beam.

    Returns
    -------
    tuple[dict, dict]
        `(pwp_te, pwp_tm)` where each has keys `beta, alpha, kx, ky, kz, coeff`
        and `coeff` has shape `(Na, Nb)`.
    """
    beta = np.asarray(polar_angles, float)
    alpha = np.asarray(azimuthal_angles, float)

    agrid = alpha[:, None]
    bgrid = beta[None, :]
    kx = k * np.sin(bgrid) * np.cos(agrid)
    ky = k * np.sin(bgrid) * np.sin(agrid)
    kz = np.broadcast_to(k * np.cos(bgrid), kx.shape)

    RG = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), float)
    E0 = float(getattr(beam, "amplitude", 1.0))
    w = float(getattr(beam, "beam_width", np.inf))
    polar_angle = float(getattr(beam, "polar_angle", 0.0))
    az_angle = float(getattr(beam, "azimuthal_angle", 0.0))
    if polarization_override is None:
        a_te, a_tm = source_jones(beam)
        pure = _pure_polarization_label(a_te, a_tm)
        if pure is None:
            common_kwargs = {
                "beam": beam,
                "k": k,
                "polar_angles": polar_angles,
                "azimuthal_angles": azimuthal_angles,
            }
            te_te, te_tm = initial_field_plane_wave_pattern_normal_incidence(
                polarization_override="TE",
                **common_kwargs,
            )
            tm_te, tm_tm = initial_field_plane_wave_pattern_normal_incidence(
                polarization_override="TM",
                **common_kwargs,
            )
            pwp_te = dict(te_te)
            pwp_tm = dict(te_tm)
            pwp_te["coeff"] = a_te * np.asarray(te_te["coeff"]) + a_tm * np.asarray(tm_te["coeff"])
            pwp_tm["coeff"] = a_te * np.asarray(te_tm["coeff"]) + a_tm * np.asarray(tm_tm["coeff"])
            return pwp_te, pwp_tm
        polarization_override = pure

    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()

    emnikrg = np.exp(-1j * (kx * RG[0] + ky * RG[1] + kz * RG[2]))
    pref_scalar = E0 * (k**2) * (w**2) / (4 * np.pi)
    sin_beta = np.sin(beta)[None, :]
    gaussian_envelope = np.exp(-(w**2) / 4 * (k**2) * (sin_beta**2))
    pref = pref_scalar * np.cos(beta)[None, :] * gaussian_envelope
    pref = pref * (np.sign(np.cos(beta))[None, :] == np.sign(np.cos(polar_angle)))
    eikrg_pref = emnikrg * pref

    if pol == "te":
        alphaG = az_angle
    else:
        alphaG = az_angle - np.pi / 2

    coeff_te = np.cos(alpha[:, None] - alphaG) * eikrg_pref
    coeff_tm = np.sign(np.cos(polar_angle)) * (np.sin(alpha[:, None] - alphaG) * eikrg_pref)

    pwp_te = {"beta": beta, "alpha": alpha, "kx": kx, "ky": ky, "kz": kz, "coeff": coeff_te}
    pwp_tm = {"beta": beta, "alpha": alpha, "kx": kx, "ky": ky, "kz": kz, "coeff": coeff_tm}
    return pwp_te, pwp_tm
