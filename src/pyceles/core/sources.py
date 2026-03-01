from __future__ import annotations

"""Incident-source models and source-side field helpers."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, Tuple

import numpy as np
import numpy.typing as npt

from .angular import beam_axis_and_frame
from .geometry_bounds import conservative_cross_set_max_distance
from .indexing import index_vswf, n_modes
from .projection import (
    incident_coeffs_from_angular_spectrum,
    incident_coeffs_planewave,
    incident_coeffs_wavebundle_normal_incidence,
)
from .translation import RadialLUT, translation_ab5_table, translation_block

Polarization = Literal["TE", "TM"]
PolarizationInput = Polarization | tuple[complex, complex] | list[complex] | np.ndarray


def _as_complex_triplet(
    name: str, values: tuple[complex, complex, complex] | np.ndarray
) -> np.ndarray:
    """Normalize one 3-component complex vector and validate finiteness."""
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"`{name}` must have exactly 3 entries. Got shape {np.asarray(values).shape}."
        )
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr


def _as_float_triplet(name: str, values: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Normalize one 3-component float vector and validate finiteness."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"`{name}` must have exactly 3 entries. Got shape {np.asarray(values).shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr


def _normalize_dipole_collection_inputs(
    positions: np.ndarray,
    dipole_moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize dipole collection arrays."""
    pos = np.asarray(positions, dtype=float)
    mom = np.asarray(dipole_moments, dtype=np.complex128)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (Nd, 3). Got {pos.shape}.")
    if mom.ndim != 2 or mom.shape[1] != 3:
        raise ValueError(f"`dipole_moments` must have shape (Nd, 3). Got {mom.shape}.")
    if pos.shape[0] != mom.shape[0]:
        raise ValueError(
            "`positions` and `dipole_moments` must have the same first dimension. "
            f"Got {pos.shape[0]} and {mom.shape[0]}."
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("`positions` must contain only finite values.")
    if not np.all(np.isfinite(mom.real)) or not np.all(np.isfinite(mom.imag)):
        raise ValueError("`dipole_moments` must contain only finite values.")
    return pos, mom


def _dipole_outgoing_coeff_vector(
    *,
    lmax: int,
    k0: float,
    k_medium: float,
    dipole_moment: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Outgoing SVWF coefficients for one electric point dipole.

    This follows SMUTHI's homogeneous-medium dipole normalization:
    the source is represented as an outgoing SVWF expansion centered at the
    dipole location, with only electric (`tau=2`) `l=1` modes populated.
    """
    Nm = n_modes(int(lmax))
    out = np.zeros((Nm,), dtype=dtype)
    if int(lmax) < 1:
        return out

    mx, my, mz = dipole_moment
    c_xy = 1.0 / (2.0 * np.sqrt(3.0))
    c_z = 1.0 / np.sqrt(6.0)
    pref = (1j * float(k_medium) * (float(k0) ** 2)) / np.pi

    out[index_vswf(1, -1, 2, int(lmax))] = pref * c_xy * (mx + 1j * my)
    out[index_vswf(1, 0, 2, int(lmax))] = pref * c_z * mz
    out[index_vswf(1, 1, 2, int(lmax))] = pref * c_xy * (mx - 1j * my)
    return out


def _dipole_incident_coeffs_from_outgoing(
    *,
    receiver_positions: np.ndarray,
    dipole_positions: np.ndarray,
    outgoing_coeffs: np.ndarray,
    lmax: int,
    k_medium: float,
    radial_lut_dr: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Translate outgoing dipole multipoles to regular SVWFs at receiver centers."""
    pos_rcv = np.asarray(receiver_positions, dtype=float)
    pos_dip = np.asarray(dipole_positions, dtype=float)
    coeffs_dip = np.asarray(outgoing_coeffs, dtype=dtype)
    Ns = int(pos_rcv.shape[0])
    Nm = n_modes(int(lmax))
    out = np.zeros((Ns, Nm), dtype=dtype)
    if Ns == 0 or pos_dip.shape[0] == 0:
        return out

    if np.any(np.all(np.isclose(pos_rcv[:, None, :], pos_dip[None, :, :], atol=1e-12), axis=2)):
        raise ValueError(
            "Dipole and receiver center coincide for at least one pair. "
            "Dipole-to-center translation is singular at zero separation."
        )

    ab5 = translation_ab5_table(int(lmax), dtype=dtype)
    r_max = conservative_cross_set_max_distance(pos_rcv, pos_dip)
    radial_lut = RadialLUT(
        lmax=int(lmax),
        k=float(k_medium),
        r_max=float(r_max),
        dr=float(radial_lut_dr),
        dtype=dtype,
    )

    for j in range(pos_dip.shape[0]):
        c_out = coeffs_dip[j]
        for i in range(Ns):
            rvec = pos_rcv[i] - pos_dip[j]
            Wij = translation_block(
                int(lmax),
                float(k_medium),
                rvec,
                ab5=ab5,
                radial_lut=radial_lut,
            )
            out[i] += np.asarray(Wij @ c_out, dtype=dtype)

    return out


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
    """Return source polarization as Jones-like TE/TM weights.

    Local dipole sources do not define TE/TM Jones metadata; this helper raises
    for `DipoleSource`/`DipoleCollection`.
    """
    if isinstance(source, (DipoleSource, DipoleCollection)):
        raise TypeError(
            "Dipole sources do not define TE/TM Jones polarization metadata. "
            "Use dipole-moment vectors/orientations instead."
        )
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


@dataclass(frozen=True)
class DipoleSource:
    """Single electric point dipole in a homogeneous medium.

    The dipole is represented internally by an outgoing SVWF expansion centered
    at `position` with only `l=1` electric modes populated.

    Note
    ----
    The current implementation still enforces real `medium_n`. This is a
    solver-policy legacy inherited from the original beam-only CELES-style
    workflow, not a fundamental limitation of local dipole sources.

    Magnitude convention
    --------------------
    `dipole_moment` follows the same length-unit system as geometry/wavelength.
    A useful reference magnitude is `|p| ~ k0^-3 = (wavelength/(2*pi))^3`.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    dipole_moment: tuple[complex, complex, complex] = (1.0 + 0j, 0.0 + 0j, 0.0 + 0j)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0
    radial_lut_dr: float = 1.0
    polarization: PolarizationInput = "TE"

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if not (n.real > 0):
            raise ValueError(f"medium_n must have positive real part. Got {n!r}")
        if abs(n.imag) > 0:
            raise ValueError(
                "DipoleSource currently requires real `medium_n` in this homogeneous solver path "
                "(legacy policy from the original beam-only workflow, not a dipole-physics limit)."
            )
        _as_complex_triplet("dipole_moment", self.dipole_moment)
        _as_float_triplet("position", self.position)
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if float(self.radial_lut_dr) <= 0.0:
            raise ValueError(f"`radial_lut_dr` must be > 0. Got {self.radial_lut_dr!r}.")

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Dipoles do not define TE/TM Jones polarization states."""
        raise TypeError(
            "DipoleSource does not define TE/TM Jones polarization metadata. "
            "Use `dipole_moment` orientation/components instead."
        )

    def with_polarization(self, polarization: PolarizationInput) -> "DipoleSource":
        """Dipoles are not TE/TM sources; this method is unsupported by design."""
        raise TypeError(
            "DipoleSource does not define TE/TM polarization states. "
            "Use `dipole_moment` orientation/components instead."
        )

    def dipole_positions(self) -> np.ndarray:
        """Return dipole center array with shape (1, 3)."""
        return _as_float_triplet("position", self.position).reshape(1, 3)

    def dipole_moments(self) -> np.ndarray:
        """Return amplitude-scaled dipole moment array with shape (1, 3)."""
        mu = _as_complex_triplet("dipole_moment", self.dipole_moment)
        return (complex(self.amplitude) * mu).reshape(1, 3)

    def angular_frequency(self) -> float:
        """Return omega = 2*pi/lambda in pyceles unit conventions."""
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background(self) -> float:
        """Power radiated in equivalent homogeneous background (SMUTHI convention)."""
        omega = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * omega
        mu = self.dipole_moments().reshape(3)
        mu2 = float(np.sum(np.abs(mu) ** 2))
        return float(mu2 * k * (omega**3) / (12.0 * np.pi))

    def cartesian_basis_sources(
        self,
        *,
        labels: tuple[str, str, str] = ("px", "py", "pz"),
        moment_magnitude: complex = 1.0 + 0j,
    ) -> dict[str, "DipoleSource"]:
        """Return three orthogonal dipole-orientation sources at same position."""
        if len(labels) != 3:
            raise ValueError(f"`labels` must have length 3. Got {labels!r}.")
        m = complex(moment_magnitude)
        if not np.isfinite(m.real) or not np.isfinite(m.imag):
            raise ValueError(f"`moment_magnitude` must be finite. Got {moment_magnitude!r}.")
        return {
            str(labels[0]): replace(self, dipole_moment=(m, 0.0 + 0j, 0.0 + 0j)),
            str(labels[1]): replace(self, dipole_moment=(0.0 + 0j, m, 0.0 + 0j)),
            str(labels[2]): replace(self, dipole_moment=(0.0 + 0j, 0.0 + 0j, m)),
        }

    def outgoing_coeffs(
        self,
        lmax: int,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Return outgoing SVWF coefficients at the dipole center (shape (1, Nm))."""
        ctype = np.dtype(dtype)
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        coeff = _dipole_outgoing_coeff_vector(
            lmax=int(lmax),
            k0=float(k0),
            k_medium=float(k),
            dipole_moment=self.dipole_moments().reshape(3),
            dtype=ctype,
        )
        return coeff.reshape(1, -1)

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Project dipole field to regular SVWF coefficients at receiver centers."""
        del polar_angles, azimuthal_angles
        ctype = np.dtype(dtype)
        pos_rcv = np.asarray(positions, dtype=float).reshape(-1, 3)
        Nm = n_modes(int(lmax))
        if pos_rcv.shape[0] == 0:
            return np.zeros((0, Nm), dtype=ctype)

        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        return _dipole_incident_coeffs_from_outgoing(
            receiver_positions=pos_rcv,
            dipole_positions=self.dipole_positions(),
            outgoing_coeffs=self.outgoing_coeffs(int(lmax), dtype=ctype),
            lmax=int(lmax),
            k_medium=float(k),
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )


@dataclass(frozen=True)
class DipoleCollection:
    """Collection of electric point dipoles in a homogeneous medium.

    Note
    ----
    The current implementation still enforces real `medium_n`. This is a
    solver-policy legacy inherited from the original beam-only CELES-style
    workflow, not a fundamental limitation of local dipole sources.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    dipole_moments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.complex128)
    )
    amplitude: float = 1.0
    radial_lut_dr: float = 1.0
    polarization: PolarizationInput = "TE"

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if not (n.real > 0):
            raise ValueError(f"medium_n must have positive real part. Got {n!r}")
        if abs(n.imag) > 0:
            raise ValueError(
                "DipoleCollection currently requires real `medium_n` in this homogeneous solver path "
                "(legacy policy from the original beam-only workflow, not a dipole-physics limit)."
            )
        _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if float(self.radial_lut_dr) <= 0.0:
            raise ValueError(f"`radial_lut_dr` must be > 0. Got {self.radial_lut_dr!r}.")

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Dipoles do not define TE/TM Jones polarization states."""
        raise TypeError(
            "DipoleCollection does not define TE/TM Jones polarization metadata. "
            "Use vector dipole moments/orientations instead."
        )

    def with_polarization(self, polarization: PolarizationInput) -> "DipoleCollection":
        """Dipoles are not TE/TM sources; this method is unsupported by design."""
        raise TypeError(
            "DipoleCollection does not define TE/TM polarization states. "
            "Use vector dipole moments/orientations instead."
        )

    def dipole_positions(self) -> np.ndarray:
        """Return dipole centers with shape (Nd, 3)."""
        pos, _ = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return pos

    def dipole_moments_array(self) -> np.ndarray:
        """Return amplitude-scaled dipole moments with shape (Nd, 3)."""
        _, mom = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return complex(self.amplitude) * mom

    def angular_frequency(self) -> float:
        """Return omega = 2*pi/lambda in pyceles unit conventions."""
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background_per_dipole(self) -> np.ndarray:
        """Per-dipole homogeneous-background dissipated power (SMUTHI convention)."""
        omega = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * omega
        mu = self.dipole_moments_array()
        mu2 = np.sum(np.abs(mu) ** 2, axis=1)
        return np.asarray(mu2 * k * (omega**3) / (12.0 * np.pi), dtype=float)

    def dissipated_power_homogeneous_background(self) -> float:
        """Total homogeneous-background dissipated power for collection."""
        return float(np.sum(self.dissipated_power_homogeneous_background_per_dipole()))

    def outgoing_coeffs(
        self,
        lmax: int,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Return outgoing SVWF coefficients per dipole, shape (Nd, Nm)."""
        ctype = np.dtype(dtype)
        pos = self.dipole_positions()
        mom = self.dipole_moments_array()
        Nm = n_modes(int(lmax))
        out = np.zeros((pos.shape[0], Nm), dtype=ctype)
        if pos.shape[0] == 0:
            return out
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        for j in range(pos.shape[0]):
            out[j] = _dipole_outgoing_coeff_vector(
                lmax=int(lmax),
                k0=float(k0),
                k_medium=float(k),
                dipole_moment=mom[j],
                dtype=ctype,
            )
        return out

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        """Project dipole-collection field to regular SVWF coefficients."""
        del polar_angles, azimuthal_angles
        ctype = np.dtype(dtype)
        pos_rcv = np.asarray(positions, dtype=float).reshape(-1, 3)
        Nm = n_modes(int(lmax))
        if pos_rcv.shape[0] == 0:
            return np.zeros((0, Nm), dtype=ctype)
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        return _dipole_incident_coeffs_from_outgoing(
            receiver_positions=pos_rcv,
            dipole_positions=self.dipole_positions(),
            outgoing_coeffs=self.outgoing_coeffs(int(lmax), dtype=ctype),
            lmax=int(lmax),
            k_medium=float(k),
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )


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
