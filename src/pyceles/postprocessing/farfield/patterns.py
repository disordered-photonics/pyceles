from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.conversions import svwf_outgoing_to_pwp, transformation_coefficients
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.sources import AngularSpectrumSource, DipoleCollection, DipoleSource, Source
from pyceles.core.spherical import spherical_functions_trigon

from .common import cast_pwp_coeff_dtype


@dataclass(frozen=True)
class FarFieldPatterns:
    """TE/TM plane-wave spectra on the common `(alpha,beta)` angular grid."""

    initial_te: dict | None
    initial_tm: dict | None
    scattered_te: dict
    scattered_tm: dict
    total_te: dict | None
    total_tm: dict | None


def scattered_field_plane_wave_pattern(
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    *,
    backend: str = "numpy",
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> tuple[dict, dict]:
    """Compute far-field plane-wave patterns of the scattered field.

    The shipped CuPy path accelerates only the outgoing SVWF-to-PWP assembly
    and still returns canonical NumPy PWP dictionaries. This keeps all
    downstream far-field diagnostics on the reference code path while moving
    the dominant particle/angle contraction off CPU.

    Precision policy note:
    this helper currently follows the historical far-field API and exposes a
    single `dtype` knob, so both device-side spectrum accumulation and returned
    PWP coefficients use the same complex dtype. This matches the existing CPU
    helper contract, but it is narrower than the solve path's explicit
    `compute_dtype` / `accum_dtype` split and can be revisited if far-field
    accuracy or stability requires it.
    """
    backend_name = str(backend).lower()
    if backend_name == "cupy":
        return _scattered_field_plane_wave_pattern_cupy(
            positions=positions,
            coeffs=coeffs,
            k=k,
            lmax=lmax,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            dtype=dtype,
            show_progress=show_progress,
        )
    return svwf_outgoing_to_pwp(
        positions=np.asarray(positions, dtype=float),
        coeffs=np.asarray(coeffs, dtype=np.dtype(dtype)),
        k=float(k),
        lmax=int(lmax),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        dtype=np.dtype(dtype),
        show_progress=bool(show_progress),
    )


def _scattered_field_plane_wave_pattern_cupy(
    *,
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    dtype: npt.DTypeLike,
    show_progress: bool,
) -> tuple[dict, dict]:
    """CuPy-accelerated outgoing SVWF-to-PWP assembly.

    The CPU reference path loops over spheres. For realistic CuPy solves that
    loop dominates far-field postprocessing, so this backend batches small
    groups of spheres on device and accumulates their angular spectra there.

    We still build the spherical-function coefficient tables on CPU. They are
    small, setup-only data and the hot cost is the repeated particle/angle
    contraction, not the table construction.

    The implementation stays in this shared owner module for now because there
    is only one accelerated branch and it shares the public far-field contract
    with the NumPy reference path. If near-field/far-field GPU postprocessing
    grows into a larger family of backend-specific kernels, splitting into
    dedicated `*_cupy.py` modules will become more attractive.
    """
    cupy, _ = import_cupy()

    ctype = np.dtype(dtype)
    positions_np = np.asarray(positions, dtype=float)
    coeffs_np = np.asarray(coeffs, dtype=ctype)
    beta_np = np.asarray(polar_angles, dtype=float).reshape(-1)
    alpha_np = np.asarray(azimuthal_angles, dtype=float).reshape(-1)
    Ns = positions_np.shape[0]
    if coeffs_np.shape[0] != Ns:
        raise ValueError("coeffs must have shape (Ns, Nm)")

    real_dtype = cupy.float32 if ctype == np.dtype(np.complex64) else cupy.float64
    complex_dtype = cupy.complex64 if ctype == np.dtype(np.complex64) else cupy.complex128
    k_real = real_dtype(float(k))

    beta = cupy.asarray(beta_np, dtype=real_dtype)
    alpha = cupy.asarray(alpha_np, dtype=real_dtype)
    positions_gpu = cupy.asarray(positions_np, dtype=real_dtype)
    coeffs_gpu = cupy.asarray(coeffs_np, dtype=complex_dtype)

    agrid = alpha[:, None]
    bgrid = beta[None, :]
    sb = cupy.sin(beta)
    cb = cupy.cos(beta)
    kx = k_real * cupy.sin(bgrid) * cupy.cos(agrid)
    ky = k_real * cupy.sin(bgrid) * cupy.sin(agrid)
    kz = cupy.broadcast_to(k_real * cupy.cos(beta), kx.shape)

    PI, TAU = spherical_functions_trigon(asnumpy(cb), asnumpy(sb), int(lmax), xp=np)
    Nm = n_modes(int(lmax))
    Nb = beta_np.size
    Na = alpha_np.size
    B_te = np.zeros((Nm, Nb), dtype=ctype)
    B_tm = np.zeros((Nm, Nb), dtype=ctype)
    m_of_n = np.zeros(Nm, dtype=np.int32)
    for tau, l, m, n in iter_modes(int(lmax)):
        m_of_n[n] = m
        B_te[n, :] = transformation_coefficients(PI, TAU, tau, l, m, pol=1, dagger=False)
        B_tm[n, :] = transformation_coefficients(PI, TAU, tau, l, m, pol=2, dagger=False)

    B_te_gpu = cupy.asarray(B_te, dtype=complex_dtype)
    B_tm_gpu = cupy.asarray(B_tm, dtype=complex_dtype)
    m_gpu = cupy.asarray(m_of_n, dtype=real_dtype)
    eima = cupy.exp(1j * agrid * m_gpu[None, :]).astype(complex_dtype, copy=False)

    pwp_te_coeff = cupy.zeros((Na, Nb), dtype=complex_dtype)
    pwp_tm_coeff = cupy.zeros((Na, Nb), dtype=complex_dtype)

    sphere_iter = range(0, Ns, 16)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            sphere_iter = tqdm(sphere_iter, desc="PWP (SVWF->PWP)")
        except Exception:
            pass

    # Batch a modest number of spheres per launch. This avoids the original
    # Python sphere loop while keeping the `(chunk, alpha, beta)` workspace
    # bounded and easy to reason about.
    for start in sphere_iter:
        stop = min(start + 16, Ns)
        pos_chunk = positions_gpu[start:stop]
        coeff_chunk = coeffs_gpu[start:stop]

        beima = coeff_chunk[:, None, :] * eima[None, :, :]
        spec_te = cupy.einsum("can,nb->cab", beima, B_te_gpu, optimize=True)
        spec_tm = cupy.einsum("can,nb->cab", beima, B_tm_gpu, optimize=True)

        phase = cupy.exp(
            -1j
            * (
                pos_chunk[:, 0][:, None, None] * kx[None, :, :]
                + pos_chunk[:, 1][:, None, None] * ky[None, :, :]
                + pos_chunk[:, 2][:, None, None] * kz[None, :, :]
            )
        ).astype(complex_dtype, copy=False)

        pwp_te_coeff += cupy.sum(spec_te * phase, axis=0) / (2.0 * np.pi)
        pwp_tm_coeff += cupy.sum(spec_tm * phase, axis=0) / (2.0 * np.pi)

    return (
        {
            "beta": beta_np,
            "alpha": alpha_np,
            "kx": asnumpy(kx),
            "ky": asnumpy(ky),
            "kz": asnumpy(kz),
            "coeff": asnumpy(pwp_te_coeff).astype(ctype, copy=False),
        },
        {
            "beta": beta_np,
            "alpha": alpha_np,
            "kx": asnumpy(kx),
            "ky": asnumpy(ky),
            "kz": asnumpy(kz),
            "coeff": asnumpy(pwp_tm_coeff).astype(ctype, copy=False),
        },
    )


def total_field_plane_wave_pattern(
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
) -> tuple[dict, dict]:
    """Combine initial and scattered PWPs into total-field PWPs."""
    tot_te = dict(initial_pwp_te)
    tot_tm = dict(initial_pwp_tm)
    tot_te["coeff"] = initial_pwp_te["coeff"] + scattered_pwp_te["coeff"]
    tot_tm["coeff"] = initial_pwp_tm["coeff"] + scattered_pwp_tm["coeff"]
    return tot_te, tot_tm


def compute_far_field_patterns(
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    source: Source | None = None,
    backend: str = "numpy",
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> FarFieldPatterns:
    """Compute scattered PWPs and, when available, initial/total PWPs."""
    ctype = np.dtype(dtype)

    p_s_te, p_s_tm = scattered_field_plane_wave_pattern(
        positions=positions,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        backend=backend,
        dtype=ctype,
        show_progress=show_progress,
    )

    p_i_te = None
    p_i_tm = None
    p_t_te = None
    p_t_tm = None

    if isinstance(source, (DipoleSource, DipoleCollection)):
        dip_pos = np.asarray(source.dipole_positions(), dtype=float).reshape(-1, 3)
        dip_coeffs = np.asarray(source.outgoing_coeffs(1, dtype=ctype), dtype=ctype)
        p_i_te, p_i_tm = scattered_field_plane_wave_pattern(
            positions=dip_pos,
            coeffs=dip_coeffs,
            k=k,
            lmax=1,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            backend=backend,
            dtype=ctype,
            show_progress=False,
        )
        p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)
    elif isinstance(source, AngularSpectrumSource):
        p_i_te, p_i_tm = source.angular_spectrum(
            k=float(k),
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        )
        p_i_te = cast_pwp_coeff_dtype(p_i_te, ctype)
        p_i_tm = cast_pwp_coeff_dtype(p_i_tm, ctype)
        p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)

    return FarFieldPatterns(
        initial_te=p_i_te,
        initial_tm=p_i_tm,
        scattered_te=p_s_te,
        scattered_tm=p_s_tm,
        total_te=p_t_te,
        total_tm=p_t_tm,
    )


__all__ = [
    "FarFieldPatterns",
    "compute_far_field_patterns",
    "scattered_field_plane_wave_pattern",
    "total_field_plane_wave_pattern",
]
