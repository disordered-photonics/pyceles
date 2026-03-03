from __future__ import annotations

"""SVWF projection routines (PWP -> incident coefficients)."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.special import jv

from .angular import periodic_azimuthal_weights, trapezoidal_weights
from .indexing import n_modes, n_scalar, scalar_index
from .spherical import spherical_functions_trigon

if TYPE_CHECKING:
    from .sources import AngularSpectrumSource, GaussianBeam, PlaneWave, Source


Polarization = Literal["TE", "TM"]


def _pure_polarization_label(
    a_te: complex, a_tm: complex, *, atol: float = 1e-15
) -> Polarization | None:
    """Return pure-channel label when Jones weights represent TE-only or TM-only."""
    if abs(a_tm) <= atol and abs(a_te) > atol:
        return "TE"
    if abs(a_te) <= atol and abs(a_tm) > atol:
        return "TM"
    return None


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
        gte_row = gte[ia, :]
        gtm_row = gtm[ia, :]
        active_beta = np.flatnonzero((gte_row != 0) | (gtm_row != 0))
        if active_beta.size == 0:
            continue

        phase = np.exp(
            1j
            * (
                pos[:, 0][:, None] * kx[ia, active_beta][None, :]
                + pos[:, 1][:, None] * ky[ia, active_beta][None, :]
                + pos[:, 2][:, None] * kz[ia, active_beta][None, :]
            )
        )  # (Ns,Nb)

        g1 = gte_row[active_beta][None, :] * Bdag_pol1[:, active_beta]
        g2 = gtm_row[active_beta][None, :] * Bdag_pol2[:, active_beta]
        mode_beta = (g1 + g2) * wb[active_beta][None, :]  # (Nm,Nb_active)

        mode_weight = np.exp(-1j * m_of_mode * alpha_a) * wa[ia]
        contrib = phase @ mode_beta.T  # (Ns,Nm), reduced Nb when sparse
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
