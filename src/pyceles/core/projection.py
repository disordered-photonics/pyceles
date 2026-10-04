"""SVWF projection routines (PWP -> incident coefficients)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.special import jv

from .conversions import transformation_coefficients
from .indexing import n_modes, n_scalar, scalar_index
from .polarization import pure_polarization_label
from .spherical import spherical_functions_trigon

if TYPE_CHECKING:
    from .sources import GaussianBeam, JonesPolarizedSource, PlaneWave, Source


Polarization = Literal["TE", "TM"]


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
    source: JonesPolarizedSource,
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
    b_te = project_source_to_svwf(
        positions,
        lmax,
        src_te,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=dtype,
    )
    b_tm = project_source_to_svwf(
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

    jones_weight = 1.0 + 0.0j
    if polarization_override is None:
        a_te, a_tm = source.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
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
        jones_weight = a_te if pure == "TE" else a_tm

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
    if jones_weight != 1.0:
        aI *= jones_weight
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

    jones_weight = 1.0 + 0.0j
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
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
        jones_weight = a_te if pure == "TE" else a_tm

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

    if jones_weight != 1.0:
        aI *= jones_weight
    return np.asarray(aI, dtype=ctype)
