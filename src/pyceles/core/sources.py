from __future__ import annotations

"""Incident-source models and source-side field helpers."""

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Protocol, Tuple, runtime_checkable

import numpy as np
import numpy.typing as npt

from .angular import beam_axis_and_frame, trapezoidal_weights
from .conversions import angular_spectrum_to_svwf_regular
from .geometry_bounds import conservative_cross_set_max_distance
from .indexing import index_vswf, n_modes
from .polarization import pure_polarization_label
from .projection import (
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


def _incident_coeffs_from_outgoing_expansion(
    *,
    receiver_positions: np.ndarray,
    source_positions: np.ndarray,
    outgoing_coeffs: np.ndarray,
    lmax: int,
    k_medium: float,
    radial_lut_dr: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Translate outgoing multipoles to regular SVWFs at receiver centers."""
    pos_rcv = np.asarray(receiver_positions, dtype=float)
    pos_src = np.asarray(source_positions, dtype=float)
    coeffs_src = np.asarray(outgoing_coeffs, dtype=dtype)
    Ns = int(pos_rcv.shape[0])
    Nm = n_modes(int(lmax))
    out = np.zeros((Ns, Nm), dtype=dtype)
    if Ns == 0 or pos_src.shape[0] == 0:
        return out

    if np.any(np.all(np.isclose(pos_rcv[:, None, :], pos_src[None, :, :], atol=1e-12), axis=2)):
        raise ValueError(
            "Outgoing source center and receiver center coincide for at least one pair. "
            "Outgoing-to-regular translation is singular at zero separation."
        )

    ab5 = translation_ab5_table(int(lmax), dtype=dtype)
    r_max = conservative_cross_set_max_distance(pos_rcv, pos_src)
    radial_lut = RadialLUT(
        lmax=int(lmax),
        k=float(k_medium),
        r_max=float(r_max),
        dr=float(radial_lut_dr),
        dtype=dtype,
    )

    for j in range(pos_src.shape[0]):
        c_out = coeffs_src[j]
        for i in range(Ns):
            rvec = pos_rcv[i] - pos_src[j]
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


@runtime_checkable
class Source(Protocol):
    """Unified incident-source protocol.

    Capability contract
    -------------------
    New source implementations should satisfy this protocol explicitly.
    In particular, `has_finite_incident_power()` drives finite-beam-only
    diagnostics (transmitted/reflected fractions, power decompositions).
    """

    @property
    def wavelength(self) -> float: ...

    @property
    def medium_n(self) -> complex: ...

    @property
    def amplitude(self) -> float: ...

    def has_finite_incident_power(self) -> bool:
        """Return True when incident-power-normalized beam diagnostics are valid."""
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


@runtime_checkable
class JonesPolarizedSource(Source, Protocol):
    """Source interface exposing TE/TM Jones metadata and channel cloning."""

    @property
    def polarization(self) -> PolarizationInput: ...

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return source polarization weights in the TE/TM basis."""
        ...

    def with_polarization(self, polarization: PolarizationInput) -> "JonesPolarizedSource":
        """Return a source copy with updated polarization state."""
        ...


def finite_power_policy_error(source: Source, *, diagnostic: str) -> ValueError:
    """Build a consistent error for diagnostics requiring finite incident power."""
    cls = type(source).__name__
    method = getattr(source, "has_finite_incident_power", None)
    if isinstance(source, PlaneWave):
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
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
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


def _bessel_ring_beta_kernel(beta: np.ndarray, beta0: float) -> np.ndarray:
    """Return a narrow beta-kernel approximating delta(beta-beta0).

    The returned kernel is scaled so that:
        sum_j kernel[j] * wb[j] = 1
    where wb are the same beta quadrature weights used in source projection:
        wb = trapezoidal_weights(beta) * sin(beta).
    This keeps ring-strength approximately grid-invariant.
    """
    b = np.asarray(beta, dtype=float).reshape(-1)
    if b.size < 2:
        raise ValueError("`polar_angles` must contain at least two samples for BesselBeam.")
    if float(beta0) < float(b[0]) or float(beta0) > float(b[-1]):
        raise ValueError(
            f"`cone_angle`={beta0!r} must lie within supplied polar grid range "
            f"[{float(b[0])!r}, {float(b[-1])!r}]."
        )

    wb = trapezoidal_weights(b) * np.sin(b)
    ker = np.zeros_like(b, dtype=float)
    atol = 1e-14

    if np.isclose(beta0, b[0], atol=atol, rtol=0.0):
        if wb[0] <= 0.0:
            raise ValueError(
                "BesselBeam cone intersects beta endpoint with zero quadrature weight."
            )
        ker[0] = 1.0 / wb[0]
        return ker
    if np.isclose(beta0, b[-1], atol=atol, rtol=0.0):
        if wb[-1] <= 0.0:
            raise ValueError(
                "BesselBeam cone intersects beta endpoint with zero quadrature weight."
            )
        ker[-1] = 1.0 / wb[-1]
        return ker

    j1 = int(np.searchsorted(b, beta0, side="left"))
    if j1 <= 0:
        j1 = 1
    if j1 >= b.size:
        j1 = b.size - 1
    j0 = j1 - 1
    b0 = float(b[j0])
    b1 = float(b[j1])
    if not (b1 > b0):
        raise ValueError("`polar_angles` must be strictly increasing for BesselBeam.")
    t = (float(beta0) - b0) / (b1 - b0)
    p0 = 1.0 - t
    p1 = t
    if wb[j0] <= 0.0 or wb[j1] <= 0.0:
        raise ValueError(
            "BesselBeam cone too close to beta endpoints for current polar grid; "
            "use a denser grid away from 0/pi."
        )
    ker[j0] = p0 / wb[j0]
    ker[j1] = p1 / wb[j1]
    return ker


def _bessel_add_beta_spike(
    *,
    kernel: np.ndarray,
    beta: np.ndarray,
    wb: np.ndarray,
    beta_star: float,
    scale: float,
) -> None:
    """Deposit one weighted beta spike onto the discrete beta quadrature grid."""
    if scale == 0.0:
        return
    b = np.asarray(beta, dtype=float).reshape(-1)
    atol = 1e-14
    if np.isclose(beta_star, b[0], atol=atol, rtol=0.0):
        if wb[0] <= 0.0:
            raise ValueError(
                "Tilted Bessel ring intersects beta endpoint with zero quadrature weight."
            )
        kernel[0] += scale / wb[0]
        return
    if np.isclose(beta_star, b[-1], atol=atol, rtol=0.0):
        if wb[-1] <= 0.0:
            raise ValueError(
                "Tilted Bessel ring intersects beta endpoint with zero quadrature weight."
            )
        kernel[-1] += scale / wb[-1]
        return

    j1 = int(np.searchsorted(b, beta_star, side="left"))
    j1 = min(max(1, j1), b.size - 1)
    j0 = j1 - 1
    b0 = float(b[j0])
    b1 = float(b[j1])
    if not (b1 > b0):
        raise ValueError("`polar_angles` must be strictly increasing for BesselBeam.")
    if wb[j0] <= 0.0 or wb[j1] <= 0.0:
        raise ValueError(
            "Tilted Bessel ring intersects beta samples with zero quadrature weight; "
            "use a denser polar grid away from 0/pi."
        )
    t = (float(beta_star) - b0) / (b1 - b0)
    p0 = 1.0 - t
    p1 = t
    kernel[j0] += scale * p0 / wb[j0]
    kernel[j1] += scale * p1 / wb[j1]


def _bessel_beta_roots_for_alpha(
    *,
    alpha_value: float,
    polar_angle: float,
    azimuthal_angle: float,
    cone_cos: float,
) -> list[tuple[float, float]]:
    """Return `(beta_root, jacobian_weight)` intersections for one alpha slice."""
    theta0 = float(polar_angle)
    phi0 = float(azimuthal_angle)
    d_alpha = float(alpha_value - phi0)
    A = float(np.sin(theta0) * np.cos(d_alpha))
    B = float(np.cos(theta0))
    R = float(np.hypot(A, B))
    if R <= 1e-15:
        return []
    c = float(cone_cos)
    if abs(c) > R + 1e-12:
        return []
    ratio = float(np.clip(c / R, -1.0, 1.0))
    gamma = float(np.arccos(ratio))
    delta = float(np.arctan2(A, B))

    roots: list[tuple[float, float]] = []
    for base in (delta - gamma, delta + gamma):
        for k in (-1, 0, 1):
            beta_star = float(base + 2.0 * np.pi * k)
            if beta_star < -1e-12 or beta_star > np.pi + 1e-12:
                continue
            beta_star = float(np.clip(beta_star, 0.0, np.pi))
            fp = float(A * np.cos(beta_star) - B * np.sin(beta_star))
            if abs(fp) <= 1e-14:
                continue
            # delta(beta_local-cone) = sin(cone) * delta(f-c) / |f'(beta)|
            w = float(np.sin(np.arccos(abs(c))) / abs(fp))
            roots.append((beta_star, w))

    deduped: list[tuple[float, float]] = []
    for beta_star, w in sorted(roots, key=lambda t: t[0]):
        if deduped and abs(beta_star - deduped[-1][0]) < 1e-10:
            deduped[-1] = (deduped[-1][0], deduped[-1][1] + w)
        else:
            deduped.append((beta_star, w))
    return deduped


def _bessel_tilted_ring_kernel(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    polar_angle: float,
    azimuthal_angle: float,
    cone_angle: float,
    forward_only: bool,
) -> np.ndarray:
    """Return sparse `(Na,Nb)` ring kernel for a tilted Bessel cone."""
    alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    wb = trapezoidal_weights(beta_arr) * np.sin(beta_arr)
    ring = np.zeros((alpha_arr.size, beta_arr.size), dtype=float)

    cone_cos = float(np.cos(float(cone_angle)))
    cone_signs = (1.0,) if bool(forward_only) else (1.0, -1.0)
    for ia, alpha_value in enumerate(alpha_arr):
        row = ring[ia]
        for sign in cone_signs:
            roots = _bessel_beta_roots_for_alpha(
                alpha_value=float(alpha_value),
                polar_angle=float(polar_angle),
                azimuthal_angle=float(azimuthal_angle),
                cone_cos=sign * cone_cos,
            )
            for beta_star, w in roots:
                _bessel_add_beta_spike(
                    kernel=row,
                    beta=beta_arr,
                    wb=wb,
                    beta_star=beta_star,
                    scale=float(w),
                )
    return ring


def _bessel_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> tuple[dict, dict]:
    """Evaluate exact non-paraxial Bessel-beam ring spectrum on one alpha-beta grid."""
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_te, te_tm = _bessel_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_te, tm_tm = _bessel_angular_spectrum_coeffs(
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

    beta = np.asarray(polar_angles, dtype=float).reshape(-1)
    alpha = np.asarray(azimuthal_angles, dtype=float).reshape(-1)
    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb

    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    s = np.stack([sb * ca, sb * sa, cb], axis=2)  # (Na,Nb,3)
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)
    alpha_l = np.arctan2(sy_l, sx_l)
    cos_beta_l = np.clip(sz_l, -1.0, 1.0)
    sin_beta_l = np.sqrt(np.maximum(0.0, 1.0 - cos_beta_l**2))

    if is_normal_incidence(float(beam.polar_angle)):
        ring_1d = _bessel_ring_beta_kernel(beta, float(beam.cone_angle))
        if not bool(beam.forward_only):
            ring_1d = ring_1d + _bessel_ring_beta_kernel(
                beta, float(np.pi - float(beam.cone_angle))
            )
        ring = np.broadcast_to(ring_1d[None, :], (alpha.size, beta.size))
    else:
        ring = _bessel_tilted_ring_kernel(
            alpha=alpha,
            beta=beta,
            polar_angle=float(beam.polar_angle),
            azimuthal_angle=float(beam.azimuthal_angle),
            cone_angle=float(beam.cone_angle),
            forward_only=bool(beam.forward_only),
        )

    m = int(beam.order_m)
    az_phase = float(beam.azimuthal_phase)
    phase_mode = np.exp(1j * (m * alpha_l + az_phase))
    cx, cy, cz = _as_float_triplet("center", beam.center)
    phase_center = np.exp(-1j * (kx * cx + ky * cy + kz * cz))
    envelope = float(beam.amplitude) * phase_mode * phase_center * ring

    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()
    if pol == "te":
        g_te_l = envelope
        g_tm_l = np.zeros_like(envelope)
    else:
        g_te_l = np.zeros_like(envelope)
        g_tm_l = envelope

    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)
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

    coeff_te = m11 * g_te_l + m12 * g_tm_l
    coeff_tm = m21 * g_te_l + m22 * g_tm_l

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

    def has_finite_incident_power(self) -> bool:
        """Return True when this beam has finite incident power."""
        w = float(self.beam_width)
        return bool(np.isfinite(w) and (not np.isclose(w, 0.0)))

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
class BesselBeam:
    """Exact non-paraxial ideal Bessel beam via ring angular spectrum.

    This source is represented by a cone-supported plane-wave spectrum at
    `beta_local=cone_angle` around its beam axis with azimuthal phase factor
    `exp(i*m*alpha_local)`. The representation is non-paraxial and TE/TM
    Jones-compatible.

    Notes
    -----
    - This is an ideal Bessel beam with infinite total incident power.
      Accordingly, finite-beam power-fraction diagnostics are not defined.
    - `forward_only=True` keeps the +axis cone only and requires
      `cone_angle < pi/2` in the local beam frame.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    order_m: int = 0
    cone_angle: float = 0.2
    polar_angle: float = 0.0
    azimuthal_angle: float = 0.0
    polarization: PolarizationInput = "TE"
    amplitude: float = 1.0
    azimuthal_phase: float = 0.0
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        polarization_to_jones(self.polarization)
        _as_float_triplet("center", self.center)
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
        try:
            m_float = float(self.order_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`order_m` must be an integer. Got {self.order_m!r}.") from exc
        if not np.isfinite(m_float) or (not np.isclose(m_float, round(m_float), atol=0.0)):
            raise ValueError(f"`order_m` must be an integer. Got {self.order_m!r}.")
        cone = float(self.cone_angle)
        if not (cone > 0.0 and cone < np.pi):
            raise ValueError(f"`cone_angle` must lie in (0, pi). Got {self.cone_angle!r}.")
        if bool(self.forward_only) and cone >= (0.5 * np.pi):
            raise ValueError(
                "`forward_only=True` requires `cone_angle < pi/2` (positive kz cone). "
                f"Got cone_angle={self.cone_angle!r}."
            )

    def jones_coefficients(self) -> tuple[complex, complex]:
        """Return normalized TE/TM Jones weights for this Bessel beam state."""
        return polarization_to_jones(self.polarization)

    def with_polarization(self, polarization: PolarizationInput) -> "BesselBeam":
        """Clone beam with updated TE/TM polarization state."""
        return replace(self, polarization=polarization)

    def has_finite_incident_power(self) -> bool:
        """Ideal Bessel beams are infinite-power sources."""
        return False

    def angular_spectrum(
        self,
        *,
        k: float,
        polar_angles: np.ndarray,
        azimuthal_angles: np.ndarray,
    ) -> tuple[dict, dict]:
        """Return TE/TM angular spectrum concentrated on the Bessel cone ring."""
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
        """Project Bessel beam to incident SVWF coefficients via angular spectrum."""
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


_SLMModulation = complex | np.ndarray | Callable[[np.ndarray, np.ndarray], npt.ArrayLike]


@dataclass(frozen=True)
class SLMSource:
    """Angular-spectrum wrapper that applies an SLM-style complex modulation.

    The modulation is applied directly to TE/TM plane-wave amplitudes
    (per angular bin), which preserves polarization-basis semantics and keeps
    compatibility with existing source-projection and near-field quadratures.

    Parameters
    ----------
    base_source:
        Underlying propagating source exposing `angular_spectrum(...)`, for
        example `GaussianBeam`.
    modulation:
        Either:
        - callable `(alpha_grid, beta_grid) -> complex weights`, or
        - array/scalar broadcastable to shape `(len(alpha), len(beta))`.
        The same complex weight multiplies both TE and TM amplitudes at each
        angular sample.
    """

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

    def with_polarization(self, polarization: PolarizationInput) -> "SLMSource":
        return replace(self, base_source=self.base_source.with_polarization(polarization))

    def has_finite_incident_power(self) -> bool:
        """Delegate finite-power capability to base source."""
        return bool(self.base_source.has_finite_incident_power())

    def _modulation_weights(
        self, *, alpha: np.ndarray, beta: np.ndarray, dtype: npt.DTypeLike
    ) -> np.ndarray:
        """Evaluate/broadcast modulation weights on one `(alpha,beta)` grid."""
        alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        agrid, bgrid = np.meshgrid(alpha_arr, beta_arr, indexing="ij")
        if callable(self.modulation):
            w_raw = self.modulation(agrid, bgrid)
        else:
            w_raw = self.modulation
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
        """Return base TE/TM angular spectrum multiplied by SLM complex weights."""
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
        """Project SLM-modulated angular spectrum to incident SVWF coefficients."""
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

    def has_finite_incident_power(self) -> bool:
        """Plane waves carry infinite incident power in homogeneous media."""
        return False

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

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    def dipole_positions(self) -> np.ndarray:
        """Return dipole center array with shape (1, 3)."""
        return _as_float_triplet("position", self.position).reshape(1, 3)

    def dipole_moments(self) -> np.ndarray:
        """Return amplitude-scaled dipole moment array with shape (1, 3)."""
        mu = _as_complex_triplet("dipole_moment", self.dipole_moment)
        return (complex(self.amplitude) * mu).reshape(1, 3)

    def angular_frequency(self) -> float:
        """Return `k0 = 2*pi/lambda` in pyceles unit conventions.

        The method name is kept for historical compatibility.
        """
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background(self) -> float:
        """Power radiated in equivalent homogeneous background (SMUTHI convention)."""
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments().reshape(3)
        mu2 = float(np.sum(np.abs(mu) ** 2))
        return float(mu2 * k * (k0**3) / (12.0 * np.pi))

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
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.dipole_positions(),
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

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    def dipole_positions(self) -> np.ndarray:
        """Return dipole centers with shape (Nd, 3)."""
        pos, _ = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return pos

    def dipole_moments_array(self) -> np.ndarray:
        """Return amplitude-scaled dipole moments with shape (Nd, 3)."""
        _, mom = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return complex(self.amplitude) * mom

    def angular_frequency(self) -> float:
        """Return `k0 = 2*pi/lambda` in pyceles unit conventions.

        The method name is kept for historical compatibility.
        """
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background_per_dipole(self) -> np.ndarray:
        """Per-dipole homogeneous-background dissipated power (SMUTHI convention)."""
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments_array()
        mu2 = np.sum(np.abs(mu) ** 2, axis=1)
        return np.asarray(mu2 * k * (k0**3) / (12.0 * np.pi), dtype=float)

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
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.dipole_positions(),
            outgoing_coeffs=self.outgoing_coeffs(int(lmax), dtype=ctype),
            lmax=int(lmax),
            k_medium=float(k),
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )
