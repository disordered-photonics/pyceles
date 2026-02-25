from __future__ import annotations

"""Incident-field source models and SVWF coefficient generators.

Supported source models for the solver:
- `PlaneWave` (arbitrary incidence angle)
- `GaussianBeam` (Gaussian wavebundle, arbitrary incidence; normal incidence uses
  optimized kernels where available)

Coefficient layout per sphere uses the SVWF ordering:
  [tau=1 (M/TE) all (l,m), tau=2 (N/TM) all (l,m)]
with `n_modes(lmax) = 2*lmax*(lmax+2)`.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import jv

from .angular import beam_axis_and_frame, periodic_azimuthal_weights, trapezoidal_weights
from .indexing import n_modes, n_scalar, scalar_index
from .spherical import spherical_functions_trigon

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


def transformation_coefficients(
    pilm: np.ndarray,
    taulm: np.ndarray,
    tau: int,
    l: int,
    m: int,
    pol: int,
    dagger: bool,
) -> np.ndarray:
    """Transformation coefficients B or B^dagger between PVWF and SVWF bases.

    Parameters
    ----------
    tau :
        1=TE (M), 2=TM (N) for the SVWF basis.
    pol :
        1=TE, 2=TM for the PVWF basis.
    dagger :
        If True, compute B^dagger.
    """
    ifac = (-1j) if dagger else (1j)
    mabs = abs(int(m))
    if int(tau) == int(pol):
        spher_fun = taulm[l, mabs]
    else:
        spher_fun = int(m) * pilm[l, mabs]
    return (
        -1
        / (ifac ** (l + 1))
        / np.sqrt(2 * l * (l + 1))
        * (ifac * (pol == 1) + (pol == 2))
        * spher_fun
    )


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


def incident_coeffs_from_pwp(
    positions: np.ndarray,
    lmax: int,
    *,
    k: float,
    pwp_te: dict,
    pwp_tm: dict,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Project TE/TM angular spectrum PWPs to incident SVWF coefficients."""
    pos = np.asarray(positions, dtype=float)
    lmax = int(lmax)
    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    Nscl = n_scalar(lmax)
    ctype = np.dtype(dtype)

    alpha = np.asarray(pwp_te["alpha"], dtype=float).reshape(-1)
    beta = np.asarray(pwp_te["beta"], dtype=float).reshape(-1)
    gte = np.asarray(pwp_te["coeff"], dtype=ctype)
    gtm = np.asarray(pwp_tm["coeff"], dtype=ctype)
    kx = np.asarray(pwp_te["kx"], dtype=float)
    ky = np.asarray(pwp_te["ky"], dtype=float)
    kz = np.asarray(pwp_te["kz"], dtype=float)

    if gte.shape != (alpha.size, beta.size) or gtm.shape != (alpha.size, beta.size):
        raise ValueError("PWP coefficient arrays must have shape (len(alpha), len(beta)).")

    wa = periodic_azimuthal_weights(alpha).astype(np.float64)
    wb = trapezoidal_weights(beta).astype(np.float64) * np.sin(beta)

    cb = np.cos(beta)
    sb = np.sin(beta)
    PI, TAU = spherical_functions_trigon(cb, sb, lmax, xp=np)

    Bdag_pol1 = np.zeros((Nm, beta.size), dtype=ctype)
    Bdag_pol2 = np.zeros((Nm, beta.size), dtype=ctype)
    m_of_mode = np.zeros((Nm,), dtype=np.int32)
    for tau in (1, 2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                sidx = scalar_index(l, m)
                idx = (tau - 1) * Nscl + sidx
                m_of_mode[idx] = m
                Bdag_pol1[idx, :] = transformation_coefficients(PI, TAU, tau, l, m, 1, dagger=True)
                Bdag_pol2[idx, :] = transformation_coefficients(PI, TAU, tau, l, m, 2, dagger=True)

    aI = np.zeros((Ns, Nm), dtype=ctype)
    for ia, alpha_a in enumerate(alpha):
        if wa[ia] == 0.0:
            continue

        phase = np.exp(
            1j
            * (
                pos[:, 0][:, None] * kx[ia, :][None, :]
                + pos[:, 1][:, None] * ky[ia, :][None, :]
                + pos[:, 2][:, None] * kz[ia, :][None, :]
            )
        )  # (Ns,Nb)

        g1 = gte[ia, :][None, :] * Bdag_pol1
        g2 = gtm[ia, :][None, :] * Bdag_pol2
        mode_beta = (g1 + g2) * wb[None, :]  # (Nm,Nb)

        mode_weight = np.exp(-1j * m_of_mode * alpha_a) * wa[ia]
        contrib = phase @ mode_beta.T  # (Ns,Nm)
        aI += contrib * mode_weight[None, :]

    return np.asarray(4.0 * aI, dtype=ctype)


def incident_coeffs_from_angular_spectrum(
    positions: np.ndarray,
    lmax: int,
    source: AngularSpectrumSource,
    *,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Project any angular-spectrum source to incident SVWF coefficients."""
    pwp_te, pwp_tm = source.angular_spectrum(
        k=k,
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    return incident_coeffs_from_pwp(
        positions,
        lmax,
        k=k,
        pwp_te=pwp_te,
        pwp_tm=pwp_tm,
        dtype=dtype,
    )


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


def project_source_to_svwf(
    positions: np.ndarray,
    lmax: int,
    source: Source,
    *,
    polar_angles: np.ndarray | None = None,
    azimuthal_angles: np.ndarray | None = None,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Canonical incident-source projector to SVWF coefficients.

    This is the single high-level entry point for source projection.
    Source-specific optimized kernels are used internally.

    Parameters
    ----------
    polar_angles, azimuthal_angles:
        Angular quadrature grid for source projection (RHS assembly) when the
        source requires angular-spectrum integration (for example Gaussian
        wavebundles). Plane-wave projection is analytic and does not use these
        arrays.
    """
    return _project_source_single_to_svwf(
        positions,
        lmax,
        source,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=dtype,
    )


def _project_source_single_to_svwf(
    positions: np.ndarray,
    lmax: int,
    source: Source,
    *,
    polar_angles: np.ndarray | None = None,
    azimuthal_angles: np.ndarray | None = None,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Low-level projector for one concrete source state."""
    if not hasattr(source, "incident_coeffs"):
        raise TypeError(f"Unsupported source type: {type(source).__name__}")
    coeffs = source.incident_coeffs(
        positions,
        lmax,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=np.dtype(dtype),
    )
    return np.asarray(coeffs, dtype=np.dtype(dtype))


def project_source_basis_to_svwf(
    positions: np.ndarray,
    lmax: int,
    source: Source,
    *,
    polar_angles: np.ndarray | None = None,
    azimuthal_angles: np.ndarray | None = None,
    dtype: npt.DTypeLike = np.complex128,
) -> dict[str, np.ndarray]:
    """Project unit TE/TM source basis to SVWF coefficients.

    Returns
    -------
    dict
        ``{"te": b_te, "tm": b_tm}``, each shaped ``(Ns, Nm)``.

    Notes
    -----
    This complements `project_source_to_svwf`:
    - `project_source_to_svwf` returns the mixed excitation selected by source
      polarization.
    - `project_source_basis_to_svwf` always returns both pure basis channels.
    - `polar_angles`/`azimuthal_angles` are source-projection quadrature nodes;
      they are conceptually independent from any far-field display grid.
    """
    src_te = source.with_polarization("TE")
    src_tm = source.with_polarization("TM")
    b_te = _project_source_single_to_svwf(
        positions,
        lmax,
        src_te,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=dtype,
    )
    b_tm = _project_source_single_to_svwf(
        positions,
        lmax,
        src_tm,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=dtype,
    )
    return {"te": b_te, "tm": b_tm}


def incident_coeffs_planewave(
    positions: np.ndarray,
    lmax: int,
    source: PlaneWave,
    *,
    dtype: npt.DTypeLike = np.complex128,
    polarization_override: Polarization | None = None,
) -> np.ndarray:
    """Compute incident regular SVWF coefficients for a plane wave.

    Returns
    -------
    np.ndarray
        Shape `(Ns, n_modes(lmax))`.
    """
    lmax = int(lmax)
    pos = np.asarray(positions, dtype=float)
    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    Nscl = n_scalar(lmax)
    ctype = np.dtype(dtype)

    wavelength = float(source.wavelength)
    k0 = 2 * np.pi / wavelength
    nM = complex(source.medium_n)
    if abs(nM.imag) > 0:
        raise ValueError(
            f"Embedding medium refractive index must be real for plane-wave incidence. Got {nM}."
        )
    k = k0 * float(nM.real)

    E0 = float(source.amplitude)
    beta = float(source.polar_angle)
    alpha = float(source.azimuthal_angle)
    cb = np.cos(beta)
    sb = np.sin(beta)

    if polarization_override is None:
        a_te, a_tm = source.jones_coefficients()
        pure = _pure_polarization_label(a_te, a_tm)
        if pure is None:
            a_te_field = incident_coeffs_planewave(
                positions,
                lmax,
                source.with_polarization("TE"),
                dtype=dtype,
                polarization_override="TE",
            )
            a_tm_field = incident_coeffs_planewave(
                positions,
                lmax,
                source.with_polarization("TM"),
                dtype=dtype,
                polarization_override="TM",
            )
            return np.asarray(a_te * a_te_field + a_tm * a_tm_field, dtype=ctype)
        polarization_override = pure

    PI, TAU = spherical_functions_trigon(np.asarray(cb), np.asarray(sb), lmax, xp=np)
    pol = 1 if str(polarization_override).upper() == "TE" else 2

    fp = np.asarray(source.focal_point, dtype=float).reshape(3)
    rel = pos - fp
    kvec = k * np.array([sb * np.cos(alpha), sb * np.sin(alpha), cb], dtype=float)
    eikr = np.exp(1j * (rel @ kvec))

    aI = np.zeros((Ns, Nm), dtype=ctype)
    for m in range(-lmax, lmax + 1):
        phase_m = np.exp(-1j * m * alpha)
        for tau in (1, 2):
            for l in range(max(1, abs(m)), lmax + 1):
                sidx = scalar_index(l, m)
                idx = (tau - 1) * Nscl + sidx
                Bdag = transformation_coefficients(PI, TAU, tau, l, m, pol, dagger=True)
                aI[:, idx] = 4.0 * E0 * phase_m * eikr * Bdag
    return np.asarray(aI, dtype=ctype)


def incident_coeffs_wavebundle_normal_incidence(
    positions: np.ndarray,
    lmax: int,
    beam: GaussianBeam,
    polar_angles_array: np.ndarray,
    *,
    dtype: npt.DTypeLike = np.complex128,
    polarization_override: Polarization | None = None,
) -> np.ndarray:
    """Compute incident regular SVWF coefficients for a normal-incidence Gaussian wavebundle.

    Returns
    -------
    np.ndarray
        Shape `(Ns, n_modes(lmax))`.
    """
    lmax = int(lmax)
    pos = np.asarray(positions, dtype=float)
    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    Nscl = n_scalar(lmax)
    ctype = np.dtype(dtype)

    wavelength = float(beam.wavelength)
    k0 = 2 * np.pi / wavelength
    nM = complex(beam.medium_n)
    if abs(nM.imag) > 0:
        raise ValueError(
            f"Embedding medium refractive index must be real for wavebundle incidence. Got {nM}."
        )
    k = k0 * float(nM.real)
    E0 = float(beam.amplitude)
    w = float(beam.beam_width)
    if not np.isclose(np.sin(float(beam.polar_angle)), 0.0, atol=1e-12):
        raise ValueError(
            "incident_coeffs_wavebundle_normal_incidence requires polar_angle = 0 or pi."
        )

    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = _pure_polarization_label(a_te, a_tm)
        if pure is None:
            a_te_field = incident_coeffs_wavebundle_normal_incidence(
                positions,
                lmax,
                beam.with_polarization("TE"),
                polar_angles_array,
                dtype=dtype,
                polarization_override="TE",
            )
            a_tm_field = incident_coeffs_wavebundle_normal_incidence(
                positions,
                lmax,
                beam.with_polarization("TM"),
                polar_angles_array,
                dtype=dtype,
                polarization_override="TM",
            )
            return np.asarray(a_te * a_te_field + a_tm * a_tm_field, dtype=ctype)
        polarization_override = pure

    prefac = E0 * (k**2) * (w**2) / np.pi

    if str(polarization_override).upper() == "TE":
        alphaG = float(beam.azimuthal_angle)
    else:
        alphaG = float(beam.azimuthal_angle) - np.pi / 2

    full_beta = np.asarray(polar_angles_array, dtype=float).reshape(-1)
    direction = np.sign(np.cos(float(beam.polar_angle)))
    mask = np.sign(np.cos(full_beta)) == direction
    beta = full_beta[mask]
    d_beta = float(np.mean(np.diff(beta)))
    cb = np.cos(beta)
    sb = np.sin(beta)

    gaussfac = np.exp(-(w**2) / 4 * (k**2) * (sb**2))
    gaussfac_sincos = gaussfac * cb * sb

    pilm, taulm = spherical_functions_trigon(cb, sb, lmax, xp=np)

    Nk = cb.size
    Bdag_pol1 = np.zeros((Nm, Nk), dtype=ctype)
    Bdag_pol2 = np.zeros((Nm, Nk), dtype=ctype)
    mode_indices_by_m: list[list[int]] = [[] for _ in range(2 * lmax + 1)]
    for tau in (1, 2):
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                sidx = scalar_index(l, m)
                idx = (tau - 1) * Nscl + sidx
                Bdag_pol1[idx, :] = transformation_coefficients(
                    pilm, taulm, tau, l, m, 1, dagger=True
                )
                Bdag_pol2[idx, :] = transformation_coefficients(
                    pilm, taulm, tau, l, m, 2, dagger=True
                )
                mode_indices_by_m[m + lmax].append(idx)

    g1_modes = Bdag_pol1 * gaussfac_sincos[None, :]
    g2_modes = Bdag_pol2 * gaussfac_sincos[None, :]

    fp = np.asarray(beam.focal_point, dtype=float).reshape(3)
    rel = pos - fp
    rho = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    phiG = np.arctan2(rel[:, 1], rel[:, 0])
    zG = rel[:, 2]

    aI = np.zeros((Ns, Nm), dtype=ctype)

    exp_ikz = np.exp(1j * (zG[:, None] * k) * cb[None, :])
    krho_sb = (rho[:, None] * k) * sb[None, :]

    for m in range(-lmax, lmax + 1):
        Jm1 = jv(abs(m - 1), krho_sb)
        Jp1 = jv(abs(m + 1), krho_sb)

        term_m1 = np.exp(-1j * (m - 1) * phiG)[:, None] * (exp_ikz * Jm1)
        term_p1 = np.exp(-1j * (m + 1) * phiG)[:, None] * (exp_ikz * Jp1)

        eikzI1 = np.pi * (
            np.exp(-1j * alphaG) * (1j ** abs(m - 1)) * term_m1
            + np.exp(+1j * alphaG) * (1j ** abs(m + 1)) * term_p1
        )
        eikzI2 = (
            np.pi
            * 1j
            * (
                -np.exp(-1j * alphaG) * (1j ** abs(m - 1)) * term_m1
                + np.exp(+1j * alphaG) * (1j ** abs(m + 1)) * term_p1
            )
        )

        idx_m = np.asarray(mode_indices_by_m[m + lmax], dtype=np.intp)
        if idx_m.size == 0:
            continue

        g1 = g1_modes[idx_m, :]
        g2 = g2_modes[idx_m, :]

        contrib = (
            eikzI1[:, 1:] @ g1[:, 1:].T
            + eikzI1[:, :-1] @ g1[:, :-1].T
            + eikzI2[:, 1:] @ g2[:, 1:].T
            + eikzI2[:, :-1] @ g2[:, :-1].T
        )
        aI[:, idx_m] = prefac * contrib * (d_beta / 2.0)

    return np.asarray(aI, dtype=ctype)


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
