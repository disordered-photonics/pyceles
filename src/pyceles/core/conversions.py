"""Canonical basis-conversion helpers for SVWF/PVWF workflows."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .angular import periodic_azimuthal_weights, trapezoidal_weights
from .indexing import iter_modes, n_modes, n_scalar, scalar_index
from .plane_wave_spectrum import PlaneWaveSpectrum
from .spherical import spherical_functions_trigon

if TYPE_CHECKING:
    from .sources import AngularSpectrumSource


def _array_cache_key(arr: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    """Return a hashable cache key for a small angular grid."""

    a = np.ascontiguousarray(np.asarray(arr))
    return tuple(a.shape), a.dtype.str, a.tobytes()


def _array_from_cache_key(key: tuple[tuple[int, ...], str, bytes]) -> np.ndarray:
    """Rebuild an array stored through `_array_cache_key`."""

    shape, dtype_str, payload = key
    return np.frombuffer(payload, dtype=np.dtype(dtype_str)).reshape(shape).copy()


@cache
def _cached_pwp_conversion_tables(
    lmax: int,
    alpha_key: tuple[tuple[int, ...], str, bytes],
    beta_key: tuple[tuple[int, ...], str, bytes],
    dtype_str: str,
) -> tuple[np.ndarray, ...]:
    """Return reusable SVWF<->PWP angular basis tables for one grid.

    The non-MLFMM code paths call the same conversions repeatedly on fixed
    angular grids. Caching the expensive transformation-coefficient tables here
    keeps the generic helpers readable while avoiding redundant setup work in
    the shipped conversion paths.
    """

    alpha = _array_from_cache_key(alpha_key).astype(float, copy=False).reshape(-1)
    beta = _array_from_cache_key(beta_key).astype(float, copy=False).reshape(-1)
    ctype = np.dtype(dtype_str)
    lmax_i = int(lmax)
    Nm = n_modes(lmax_i)
    Nscl = n_scalar(lmax_i)

    wa = periodic_azimuthal_weights(alpha).astype(np.float64, copy=False)
    wb = trapezoidal_weights(beta).astype(np.float64, copy=False) * np.sin(beta)
    cb = np.cos(beta)
    sb = np.sin(beta)
    PI, TAU = spherical_functions_trigon(cb, sb, lmax_i, xp=np)

    Bdag_pol1 = np.zeros((Nm, beta.size), dtype=ctype)
    Bdag_pol2 = np.zeros((Nm, beta.size), dtype=ctype)
    B_te = np.zeros((Nm, beta.size), dtype=ctype)
    B_tm = np.zeros((Nm, beta.size), dtype=ctype)
    m_of_mode = np.zeros((Nm,), dtype=np.int32)
    for tau, l, m, n in iter_modes(lmax_i):
        m_of_mode[n] = m
        sidx = scalar_index(l, m)
        idx = (tau - 1) * Nscl + sidx
        Bdag_pol1[idx, :] = transformation_coefficients(PI, TAU, tau, l, m, 1, dagger=True)
        Bdag_pol2[idx, :] = transformation_coefficients(PI, TAU, tau, l, m, 2, dagger=True)
        B_te[n, :] = transformation_coefficients(PI, TAU, tau, l, m, pol=1, dagger=False)
        B_tm[n, :] = transformation_coefficients(PI, TAU, tau, l, m, pol=2, dagger=False)

    eima = np.exp(1j * alpha[:, None] * m_of_mode[None, :]).astype(ctype, copy=False)
    mode_weight = np.exp(-1j * alpha[:, None] * m_of_mode[None, :]).astype(ctype, copy=False)
    return wa, wb, Bdag_pol1, Bdag_pol2, B_te, B_tm, m_of_mode, eima, mode_weight


def transformation_coefficients(
    pilm: np.ndarray,
    taulm: np.ndarray,
    tau: int,
    l: int,
    m: int,
    pol: int,
    dagger: bool,
) -> np.ndarray:
    """Transformation coefficient between PVWF and SVWF bases.

    Parameters
    ----------
    tau
        SVWF polarization index (`1`=M, `2`=N).
    pol
        PVWF polarization index (`1`=TE, `2`=TM).
    dagger
        If True, returns the adjoint-side coefficient (`B^dagger`), otherwise
        the forward-side coefficient (`B`).
    """
    ifac = (-1j) if dagger else (1j)
    mabs = abs(int(m))
    spher_fun = taulm[l, mabs] if int(tau) == int(pol) else int(m) * pilm[l, mabs]
    return np.asarray(
        -1
        / (ifac ** (l + 1))
        / np.sqrt(2 * l * (l + 1))
        * (ifac * (pol == 1) + (pol == 2))
        * spher_fun,
        dtype=np.result_type(spher_fun, np.complex128),
    )


def pwp_to_svwf_regular(
    positions: np.ndarray,
    lmax: int,
    *,
    k: float,
    spectrum: PlaneWaveSpectrum,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Project one TE/TM plane-wave spectrum to regular SVWF coefficients."""
    pos = np.asarray(positions, dtype=float)
    lmax = int(lmax)
    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    ctype = np.dtype(dtype)

    k = float(k)
    if (not np.isfinite(k)) or (k <= 0.0):
        raise ValueError(f"`k` must be finite and > 0. Got {k!r}.")

    alpha = spectrum.alpha
    beta = spectrum.beta
    gte = np.asarray(spectrum.coeff_te, dtype=ctype)
    gtm = np.asarray(spectrum.coeff_tm, dtype=ctype)
    kx = spectrum.kx
    ky = spectrum.ky
    kz = spectrum.kz

    agrid = alpha[:, None]
    bgrid = beta[None, :]
    expected_kx = k * np.sin(bgrid) * np.cos(agrid)
    expected_ky = k * np.sin(bgrid) * np.sin(agrid)
    expected_kz = np.broadcast_to(k * np.cos(bgrid), expected_kx.shape)
    k_atol = max(1e-12, 1e-10 * abs(k))
    if not np.allclose(kx, expected_kx, rtol=1e-10, atol=k_atol):
        raise ValueError("Spectrum `kx` is inconsistent with `k`, `alpha`, and `beta`.")
    if not np.allclose(ky, expected_ky, rtol=1e-10, atol=k_atol):
        raise ValueError("Spectrum `ky` is inconsistent with `k`, `alpha`, and `beta`.")
    if not np.allclose(kz, expected_kz, rtol=1e-10, atol=k_atol):
        raise ValueError("Spectrum `kz` is inconsistent with `k`, `alpha`, and `beta`.")

    wa, wb, Bdag_pol1, Bdag_pol2, _B_te, _B_tm, _m_of_mode, _eima, mode_weight = (
        _cached_pwp_conversion_tables(
            lmax,
            _array_cache_key(alpha),
            _array_cache_key(beta),
            ctype.str,
        )
    )

    aI = np.zeros((Ns, Nm), dtype=ctype)
    for ia, _alpha_a in enumerate(alpha):
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
        )

        g1 = gte_row[active_beta][None, :] * Bdag_pol1[:, active_beta]
        g2 = gtm_row[active_beta][None, :] * Bdag_pol2[:, active_beta]
        mode_beta = (g1 + g2) * wb[active_beta][None, :]

        weight_row = mode_weight[ia] * wa[ia]
        contrib = phase @ mode_beta.T
        aI += contrib * weight_row[None, :]

    return np.asarray(4.0 * aI, dtype=ctype)


def angular_spectrum_to_svwf_regular(
    positions: np.ndarray,
    lmax: int,
    source: AngularSpectrumSource,
    *,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Project any angular-spectrum source to regular SVWF coefficients."""
    spectrum = source.angular_spectrum(
        k=float(k),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    return pwp_to_svwf_regular(
        positions,
        lmax,
        k=float(k),
        spectrum=spectrum,
        dtype=dtype,
    )


def _svwf_to_pwp_common(
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    prefactor: float,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> PlaneWaveSpectrum:
    """Shared SVWF->PWP kernel used by regular/outgoing conversion helpers."""
    ctype = np.dtype(dtype)
    positions = np.asarray(positions, dtype=float)
    coeffs = np.asarray(coeffs, dtype=ctype)
    beta = np.asarray(polar_angles, dtype=float)
    alpha = np.asarray(azimuthal_angles, dtype=float)

    Ns = positions.shape[0]
    if coeffs.shape[0] != Ns:
        raise ValueError("coeffs must have shape (Ns, Nm)")

    Nb = beta.size
    Na = alpha.size

    agrid = alpha[:, None]
    bgrid = beta[None, :]

    kx = (k * np.sin(bgrid) * np.cos(agrid)).astype(float)
    ky = (k * np.sin(bgrid) * np.sin(agrid)).astype(float)
    kz = np.broadcast_to(k * np.cos(beta), kx.shape).astype(float)

    _wa, _wb, _Bdag_pol1, _Bdag_pol2, B_te, B_tm, _m_of_mode, eima, _mode_weight = (
        _cached_pwp_conversion_tables(
            lmax,
            _array_cache_key(alpha),
            _array_cache_key(beta),
            ctype.str,
        )
    )

    coeff_te = np.zeros((Na, Nb), dtype=ctype)
    coeff_tm = np.zeros((Na, Nb), dtype=ctype)

    sphere_iter: Iterable[int] = range(Ns)
    if show_progress:
        sphere_iter = tqdm(sphere_iter, desc="PWP (SVWF->PWP)")

    for jS in sphere_iter:
        rj = positions[jS]
        phase = np.asarray(np.exp(-1j * (rj[0] * kx + rj[1] * ky + rj[2] * kz)), dtype=ctype)
        cj = coeffs[jS, :]
        beima = eima * cj[None, :]
        coeff_te += np.asarray((beima @ B_te) * phase * prefactor, dtype=ctype)
        coeff_tm += np.asarray((beima @ B_tm) * phase * prefactor, dtype=ctype)

    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def svwf_outgoing_to_pwp(
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> PlaneWaveSpectrum:
    """Convert outgoing SVWF coefficients to TE/TM plane-wave spectrum."""
    return _svwf_to_pwp_common(
        positions,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        prefactor=1.0 / (2.0 * np.pi),
        dtype=dtype,
        show_progress=show_progress,
    )


def svwf_regular_to_pwp(
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> PlaneWaveSpectrum:
    """Convert regular SVWF coefficients to TE/TM plane-wave spectrum."""
    return _svwf_to_pwp_common(
        positions,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        prefactor=1.0 / (4.0 * np.pi),
        dtype=dtype,
        show_progress=show_progress,
    )
