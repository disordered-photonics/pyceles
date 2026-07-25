from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import jv
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.angular import (
    beam_axis_and_frame,
    is_uniform_periodic_azimuth,
    periodic_azimuthal_weights,
    trapezoidal_weights,
)
from pyceles.core.sources import (
    GaussianBeam,
    LocalExpansionSource,
    PlaneWave,
    PolarizationInput,
    is_normal_incidence,
    polarization_to_jones,
)

from .scattered import compute_scattered_field


def _compute_initial_field_gaussian_normal_incidence_analytic(
    field_points_local: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam_width: float,
    amplitude: float,
    polarization: PolarizationInput,
    azimuthal_angle: float,
    propagation_sign: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    show_progress: bool,
    backend: str,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fast initial-field kernel for Gaussian wavebundles aligned with local +z."""
    if not np.isfinite(float(beam_width)) or float(beam_width) <= 0.0:
        return None
    if not is_uniform_periodic_azimuth(azimuthal_angles):
        return None

    pts = np.asarray(field_points_local, dtype=np.float64)
    beta = np.asarray(polar_angles, dtype=float).reshape(-1)
    if beta.size < 2:
        return None

    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    real_compute_dtype: np.dtype[np.float32] | np.dtype[np.float64]
    if compute_dtype == np.dtype(np.complex64):
        real_compute_dtype = np.dtype(np.float32)
    else:
        real_compute_dtype = np.dtype(np.float64)
    n_medium_c = complex(n_medium)

    real_array = npt.NDArray[np.floating[Any]]
    x: real_array = pts[:, 0].astype(real_compute_dtype, copy=False)
    y: real_array = pts[:, 1].astype(real_compute_dtype, copy=False)
    z: real_array = pts[:, 2].astype(real_compute_dtype, copy=False)
    rho: real_array = np.hypot(x, y).astype(real_compute_dtype, copy=False)
    phi: real_array = np.arctan2(y, x)
    sin_phi: real_array = np.sin(phi).astype(real_compute_dtype, copy=False)
    cos_phi: real_array = np.cos(phi).astype(real_compute_dtype, copy=False)
    sin2_phi: real_array = (2.0 * sin_phi * cos_phi).astype(real_compute_dtype, copy=False)
    cos2_phi: real_array = (cos_phi * cos_phi - sin_phi * sin_phi).astype(
        real_compute_dtype, copy=False
    )

    sb: real_array = np.sin(beta).astype(real_compute_dtype, copy=False)
    cb: real_array = np.cos(beta).astype(real_compute_dtype, copy=False)
    beta_w: real_array = trapezoidal_weights(beta).astype(real_compute_dtype, copy=False)

    e0 = float(amplitude)
    w = float(beam_width)
    pref = e0 * (k**2) * (w**2) / (4.0 * np.pi)
    envelope: real_array = pref * cb * np.exp(-(w**2) / 4.0 * (k**2) * (sb**2))
    envelope *= np.sign(cb) == np.sign(float(propagation_sign))
    beta_weighted: real_array = envelope * sb * beta_w

    a_te, a_tm = polarization_to_jones(polarization)
    if not (np.isclose(abs(a_te), 0.0) or np.isclose(abs(a_tm), 0.0)):
        out_te = _compute_initial_field_gaussian_normal_incidence_analytic(
            field_points_local,
            k=k,
            n_medium=n_medium,
            beam_width=beam_width,
            amplitude=amplitude,
            polarization="TE",
            azimuthal_angle=azimuthal_angle,
            propagation_sign=propagation_sign,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        out_tm = _compute_initial_field_gaussian_normal_incidence_analytic(
            field_points_local,
            k=k,
            n_medium=n_medium,
            beam_width=beam_width,
            amplitude=amplitude,
            polarization="TM",
            azimuthal_angle=azimuthal_angle,
            propagation_sign=propagation_sign,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            show_progress=False,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        if out_te is None or out_tm is None:
            return None
        e_te, h_te = out_te
        e_tm, h_tm = out_tm
        return (
            np.asarray(a_te * e_te + a_tm * e_tm, dtype=accum_dtype),
            np.asarray(a_te * h_te + a_tm * h_tm, dtype=accum_dtype),
        )

    pol = "te" if abs(a_te) > 0 else "tm"
    alpha_g = float(azimuthal_angle) if pol == "te" else float(azimuthal_angle) - np.pi / 2.0
    c = float(np.cos(alpha_g))
    s = float(np.sin(alpha_g))
    propagation = float(np.sign(propagation_sign))
    if propagation == 0.0:
        propagation = 1.0

    e = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    h = np.zeros_like(e)

    u1: real_array = (c * sin2_phi - s * cos2_phi).astype(real_compute_dtype, copy=False)
    u2: real_array = (c * cos2_phi + s * sin2_phi).astype(real_compute_dtype, copy=False)
    u3: real_array = (s * cos_phi - c * sin_phi).astype(real_compute_dtype, copy=False)
    u4: real_array = (c * cos_phi + s * sin_phi).astype(real_compute_dtype, copy=False)

    pi_r = float(np.pi)
    two_i_pi = np.asarray(2j * np.pi, dtype=compute_dtype)
    c_plus: real_array = (1.0 + propagation * cb).astype(real_compute_dtype, copy=False)
    c_minus: real_array = (1.0 - propagation * cb).astype(real_compute_dtype, copy=False)
    c_splus: real_array = (propagation + cb).astype(real_compute_dtype, copy=False)
    c_sminus: real_array = (propagation - cb).astype(real_compute_dtype, copy=False)

    if propagation >= 0.0:
        hemi_mask = cb > 0.0
        hemi_label = "fwd"
    else:
        hemi_mask = cb < 0.0
        hemi_label = "bwd"
    nonzero_mask = beta_weighted != 0.0
    active_beta = np.flatnonzero(hemi_mask & nonzero_mask)
    if str(backend).lower() == "cupy":
        cupy, _ = import_cupy()
        cupyx_special = import_module("cupyx.scipy.special")
        compute_dtype_cp = (
            cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
        )
        accum_dtype_cp = (
            cupy.complex64 if accum_dtype == np.dtype(np.complex64) else cupy.complex128
        )
        real_dtype_cp = cupy.float32 if compute_dtype == np.dtype(np.complex64) else cupy.float64
        two_i_pi_cp = compute_dtype_cp(2j * np.pi)

        z_gpu = cupy.asarray(z, dtype=real_dtype_cp)
        rho_gpu = cupy.asarray(rho, dtype=real_dtype_cp)
        u1_gpu = cupy.asarray(u1, dtype=real_dtype_cp)
        u2_gpu = cupy.asarray(u2, dtype=real_dtype_cp)
        u3_gpu = cupy.asarray(u3, dtype=real_dtype_cp)
        u4_gpu = cupy.asarray(u4, dtype=real_dtype_cp)
        sb_gpu = cupy.asarray(sb[active_beta], dtype=real_dtype_cp)
        cb_gpu = cupy.asarray(cb[active_beta], dtype=real_dtype_cp)
        bw_gpu = cupy.asarray(beta_weighted[active_beta], dtype=compute_dtype_cp)
        c_plus_gpu = cupy.asarray(c_plus[active_beta], dtype=real_dtype_cp)
        c_minus_gpu = cupy.asarray(c_minus[active_beta], dtype=real_dtype_cp)
        c_splus_gpu = cupy.asarray(c_splus[active_beta], dtype=real_dtype_cp)
        c_sminus_gpu = cupy.asarray(c_sminus[active_beta], dtype=real_dtype_cp)

        e_gpu = cupy.zeros((pts.shape[0], 3), dtype=accum_dtype_cp)
        h_gpu = cupy.zeros_like(e_gpu)
        beta_batch = 32
        beta_pbar = None
        if show_progress:
            hemi_total = int(np.count_nonzero(hemi_mask))
            desc = (
                f"Initial field (non-zero {hemi_label} betas {int(active_beta.size)}/{hemi_total})"
            )
            beta_pbar = tqdm(total=int(active_beta.size), desc=desc)

        for start in range(0, int(active_beta.size), beta_batch):
            stop = min(int(active_beta.size), start + beta_batch)
            sb_i = sb_gpu[start:stop][:, None]
            cb_i = cb_gpu[start:stop][:, None]
            bw_i = bw_gpu[start:stop][:, None]
            c_plus_i = c_plus_gpu[start:stop][:, None]
            c_minus_i = c_minus_gpu[start:stop][:, None]
            c_splus_i = c_splus_gpu[start:stop][:, None]
            c_sminus_i = c_sminus_gpu[start:stop][:, None]

            q = (real_dtype_cp(k) * sb_i) * rho_gpu[None, :]
            j0 = cupyx_special.j0(q).astype(real_dtype_cp, copy=False)
            j1 = cupyx_special.j1(q).astype(real_dtype_cp, copy=False)
            # CuPy exposes j0/j1 but not the generic cylindrical J_v ufunc.
            # Build J2 from the standard recurrence so the whole Gaussian
            # initial-field kernel stays on device.
            j2 = cupy.where(
                cupy.abs(q) > real_dtype_cp(1e-12),
                (real_dtype_cp(2.0) / q) * j1 - j0,
                real_dtype_cp(0.0),
            ).astype(real_dtype_cp, copy=False)
            phase_z = cupy.exp(1j * (real_dtype_cp(k) * cb_i) * z_gpu[None, :]).astype(
                compute_dtype_cp, copy=False
            )
            wgt = (bw_i * phase_z).astype(compute_dtype_cp, copy=False)

            ex = (-s * pi_r * c_plus_i) * j0 + (pi_r * c_minus_i) * (u1_gpu[None, :] * j2)
            ey = (c * pi_r * c_plus_i) * j0 - (pi_r * c_minus_i) * (u2_gpu[None, :] * j2)
            ez = (two_i_pi_cp * (propagation * sb_i)) * (u3_gpu[None, :] * j1)
            hx = (-c * pi_r * c_splus_i) * j0 - (pi_r * c_sminus_i) * (u2_gpu[None, :] * j2)
            hy = (-s * pi_r * c_splus_i) * j0 - (pi_r * c_sminus_i) * (u1_gpu[None, :] * j2)
            hz = (two_i_pi_cp * sb_i) * (u4_gpu[None, :] * j1)

            e_gpu[:, 0] += cupy.sum((wgt * ex).astype(accum_dtype_cp, copy=False), axis=0)
            e_gpu[:, 1] += cupy.sum((wgt * ey).astype(accum_dtype_cp, copy=False), axis=0)
            e_gpu[:, 2] += cupy.sum((wgt * ez).astype(accum_dtype_cp, copy=False), axis=0)
            h_gpu[:, 0] += n_medium_c * cupy.sum(
                (wgt * hx).astype(accum_dtype_cp, copy=False), axis=0
            )
            h_gpu[:, 1] += n_medium_c * cupy.sum(
                (wgt * hy).astype(accum_dtype_cp, copy=False), axis=0
            )
            h_gpu[:, 2] += n_medium_c * cupy.sum(
                (wgt * hz).astype(accum_dtype_cp, copy=False), axis=0
            )
            if beta_pbar is not None:
                beta_pbar.update(stop - start)

        if beta_pbar is not None:
            beta_pbar.close()
        return asnumpy(e_gpu).astype(accum_dtype, copy=False), asnumpy(h_gpu).astype(
            accum_dtype, copy=False
        )

    if show_progress:
        hemi_total = int(np.count_nonzero(hemi_mask))
        desc = f"Initial field (non-zero {hemi_label} betas {int(active_beta.size)}/{hemi_total})"
        beta_iter: Iterable[int] = (
            int(ib) for ib in tqdm(active_beta, desc=desc, total=int(active_beta.size))
        )
    else:
        beta_iter = (int(ib) for ib in active_beta)

    for ib_raw in beta_iter:
        ib = ib_raw
        bw = beta_weighted[ib]
        if bw == 0.0:
            continue
        sb_i = sb[ib]
        cb_i = cb[ib]
        c_plus_i = float(c_plus[ib])
        c_minus_i = float(c_minus[ib])
        c_splus_i = float(c_splus[ib])
        c_sminus_i = float(c_sminus[ib])

        q = (k * sb_i) * rho
        j0 = jv(0, q).astype(real_compute_dtype, copy=False)
        j1 = jv(1, q).astype(real_compute_dtype, copy=False)
        j2 = jv(2, q).astype(real_compute_dtype, copy=False)
        phase_z = np.exp(1j * (k * cb_i) * z).astype(compute_dtype, copy=False)
        wgt = (bw * phase_z).astype(compute_dtype, copy=False)

        ex = (-s * pi_r * c_plus_i) * j0 + (pi_r * c_minus_i) * (u1 * j2)
        ey = (c * pi_r * c_plus_i) * j0 - (pi_r * c_minus_i) * (u2 * j2)
        ez = (two_i_pi * (propagation * sb_i)) * (u3 * j1)

        hx = (-c * pi_r * c_splus_i) * j0 - (pi_r * c_sminus_i) * (u2 * j2)
        hy = (-s * pi_r * c_splus_i) * j0 - (pi_r * c_sminus_i) * (u1 * j2)
        hz = (two_i_pi * sb_i) * (u4 * j1)

        e[:, 0] += (wgt * ex).astype(accum_dtype, copy=False)
        e[:, 1] += (wgt * ey).astype(accum_dtype, copy=False)
        e[:, 2] += (wgt * ez).astype(accum_dtype, copy=False)
        h[:, 0] += n_medium_c * (wgt * hx).astype(accum_dtype, copy=False)
        h[:, 1] += n_medium_c * (wgt * hy).astype(accum_dtype, copy=False)
        h[:, 2] += n_medium_c * (wgt * hz).astype(accum_dtype, copy=False)

    return e, h


def _compute_initial_field_gaussian_rotated_fast(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam: GaussianBeam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    show_progress: bool,
    backend: str,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rotate tilted Gaussian beams to local +z, evaluate, rotate back."""
    pts = np.asarray(field_points, dtype=np.float64)
    fp = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
    rel = pts - fp[None, :]

    if is_normal_incidence(float(beam.polar_angle)):
        local = rel
        rot_back = np.eye(3, dtype=float)
        prop_sign = float(np.sign(np.cos(float(beam.polar_angle))))
    else:
        n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
        q = np.stack([u, v, n0], axis=0)
        local = rel @ q.T
        rot_back = q
        prop_sign = 1.0

    out_local = _compute_initial_field_gaussian_normal_incidence_analytic(
        local,
        k=float(k),
        n_medium=complex(n_medium),
        beam_width=float(beam.beam_width),
        amplitude=float(beam.amplitude),
        polarization=beam.polarization,
        azimuthal_angle=float(beam.azimuthal_angle),
        propagation_sign=prop_sign,
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        show_progress=show_progress,
        backend=backend,
        compute_dtype=np.dtype(compute_dtype),
        accum_dtype=np.dtype(accum_dtype),
    )
    if out_local is None:
        return None
    e_local, h_local = out_local
    e_global = np.asarray(e_local, dtype=np.dtype(accum_dtype)) @ rot_back
    h_global = np.asarray(h_local, dtype=np.dtype(accum_dtype)) @ rot_back
    return e_global, h_global


def compute_initial_field(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    batch_size: int = 2048,
    show_progress: bool = True,
    force_general_initial_field: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the incident field for the configured source model."""
    if (not bool(force_general_initial_field)) and isinstance(beam, GaussianBeam):
        fast = _compute_initial_field_gaussian_rotated_fast(
            field_points,
            k=float(k),
            n_medium=complex(n_medium),
            beam=beam,
            polar_angles=np.asarray(polar_angles, float),
            azimuthal_angles=np.asarray(azimuthal_angles, float),
            show_progress=show_progress,
            backend=backend,
            compute_dtype=np.dtype(compute_dtype),
            accum_dtype=np.dtype(accum_dtype),
        )
        if fast is not None:
            return fast

    return _compute_initial_field_general(
        field_points,
        k=float(k),
        n_medium=complex(n_medium),
        beam=beam,
        polar_angles=np.asarray(polar_angles, float),
        azimuthal_angles=np.asarray(azimuthal_angles, float),
        batch_size=int(batch_size),
        show_progress=show_progress,
        backend=backend,
        compute_dtype=np.dtype(compute_dtype),
        accum_dtype=np.dtype(accum_dtype),
    )


def _compute_initial_field_general(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    batch_size: int,
    show_progress: bool,
    backend: str,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """General alpha-beta quadrature evaluator kept for flexible fallback paths."""
    pts = np.asarray(field_points, np.float64)
    n_medium_c = complex(n_medium)

    if isinstance(beam, LocalExpansionSource):
        src_pos = np.asarray(beam.source_positions(), dtype=float).reshape(-1, 3)
        src_coeffs = np.asarray(beam.outgoing_coeffs(1, dtype=compute_dtype), dtype=compute_dtype)
        e, h = compute_scattered_field(
            pts,
            src_pos,
            src_coeffs,
            k=float(k),
            lmax=1,
            n_medium=n_medium_c,
            particle_distance_resolution=float(getattr(beam, "radial_lut_dr", 0.0)),
            batch_size=int(batch_size),
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        if src_pos.shape[0] > 0 and pts.shape[0] > 0:
            hit = np.any(
                np.all(np.isclose(pts[:, None, :], src_pos[None, :, :], atol=1e-12), axis=2),
                axis=1,
            )
            if np.any(hit):
                e = np.asarray(e, dtype=accum_dtype).copy()
                h = np.asarray(h, dtype=accum_dtype).copy()
                e[hit] = np.nan + 0j
                h[hit] = np.nan + 0j
        return np.asarray(e, dtype=accum_dtype), np.asarray(h, dtype=accum_dtype)

    if isinstance(beam, PlaneWave):
        alpha_pw = float(beam.azimuthal_angle)
        beta_pw = float(beam.polar_angle)
        amp = complex(beam.amplitude)
        fp = np.asarray(beam.focal_point, dtype=np.float64).reshape(3)

        ca = np.cos(alpha_pw)
        sa = np.sin(alpha_pw)
        cb = np.cos(beta_pw)
        sb = np.sin(beta_pw)
        khat = np.array([sb * ca, sb * sa, cb], dtype=np.float64)
        kvec = float(k) * khat

        ehat_te = np.array([-sa, ca, 0.0], dtype=np.float64)
        ehat_tm = np.array([cb * ca, cb * sa, -sb], dtype=np.float64)
        a_te, a_tm = polarization_to_jones(beam.polarization)
        ehat = a_te * ehat_te + a_tm * ehat_tm
        hhat = np.cross(khat, ehat)

        phase = np.exp(1j * ((pts - fp[None, :]) @ kvec))
        e = amp * phase[:, None] * ehat[None, :]
        h = amp * phase[:, None] * (n_medium_c * hhat)[None, :]
        return e.astype(accum_dtype, copy=False), h.astype(accum_dtype, copy=False)

    if hasattr(beam, "angular_spectrum"):
        pwp_te, pwp_tm = beam.angular_spectrum(
            k=float(k),
            polar_angles=np.asarray(polar_angles, float),
            azimuthal_angles=np.asarray(azimuthal_angles, float),
        )
    else:
        raise TypeError(
            "Initial-field evaluation is unavailable for source type "
            f"{type(beam).__name__}. Supported initial-field sources are PlaneWave, "
            "local outgoing-expansion sources, and angular-spectrum sources."
        )

    e = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    alpha_weights = periodic_azimuthal_weights(np.asarray(azimuthal_angles, float))

    beta = np.asarray(pwp_te["beta"], dtype=float)
    alpha = np.asarray(pwp_te["alpha"], dtype=float)
    coeff_te = np.asarray(pwp_te["coeff"], dtype=compute_dtype)
    coeff_tm = np.asarray(pwp_tm["coeff"], dtype=compute_dtype)

    sinb = np.sin(beta).astype(np.float64)
    cosb = np.cos(beta).astype(np.float64)
    beta_weights = trapezoidal_weights(beta).astype(np.float64)

    kx_all = np.asarray(pwp_te["kx"], dtype=np.float64)
    ky_all = np.asarray(pwp_te["ky"], dtype=np.float64)
    kz_all = np.asarray(pwp_te["kz"], dtype=np.float64)

    n_points = pts.shape[0]
    n_alpha = alpha.shape[0]
    if alpha_weights.shape[0] != n_alpha:
        raise ValueError("azimuthal angle grid mismatch in alpha integration.")
    if beta.size == 0:
        return e, h

    batch_size_eff = int(batch_size)
    target_phase_bytes = 128 * 1024**2
    max_by_memory = max(
        1, target_phase_bytes // (int(beta.size) * int(np.dtype(compute_dtype).itemsize))
    )
    batch_size_eff = max(batch_size_eff, int(max_by_memory))

    sin_a = np.sin(alpha)[:, None]
    cos_a = np.cos(alpha)[:, None]
    sinb_row = sinb[None, :]
    cosb_row = cosb[None, :]

    weighted_beta = (sinb_row * beta_weights[None, :]).astype(compute_dtype, copy=False)
    w_te_all = coeff_te * weighted_beta
    w_tm_all = coeff_tm * weighted_beta

    khx_all = kx_all / float(k)
    khy_all = ky_all / float(k)
    khz_all = kz_all / float(k)

    ex_te = -sin_a
    ey_te = cos_a
    ez_te = np.zeros_like(ex_te)
    ex_tm = cosb_row * cos_a
    ey_tm = cosb_row * sin_a
    ez_tm = -sinb_row

    hx_te = khy_all * ez_te - khz_all * ey_te
    hy_te = khz_all * ex_te - khx_all * ez_te
    hz_te = khx_all * ey_te - khy_all * ex_te

    hx_tm = khy_all * ez_tm - khz_all * ey_tm
    hy_tm = khz_all * ex_tm - khx_all * ez_tm
    hz_tm = khx_all * ey_tm - khy_all * ex_tm

    v_all = np.empty((n_alpha, beta.size, 6), dtype=compute_dtype)
    v_all[:, :, 0] = w_te_all * ex_te + w_tm_all * ex_tm
    v_all[:, :, 1] = w_te_all * ey_te + w_tm_all * ey_tm
    v_all[:, :, 2] = w_te_all * ez_te + w_tm_all * ez_tm
    v_all[:, :, 3] = w_te_all * hx_te + w_tm_all * hx_tm
    v_all[:, :, 4] = w_te_all * hy_te + w_tm_all * hy_tm
    v_all[:, :, 5] = w_te_all * hz_te + w_tm_all * hz_tm

    if str(backend).lower() == "cupy":
        cupy, _ = import_cupy()
        compute_dtype_cp = (
            cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
        )
        accum_dtype_cp = (
            cupy.complex64 if accum_dtype == np.dtype(np.complex64) else cupy.complex128
        )
        pts_gpu = cupy.asarray(pts, dtype=cupy.float64)
        e_gpu = cupy.zeros((pts.shape[0], 3), dtype=accum_dtype_cp)
        h_gpu = cupy.zeros_like(e_gpu)
        alpha_iter: Iterable[int] = range(n_alpha)
        if show_progress:
            alpha_iter = tqdm(
                range(n_alpha),
                desc="Initial field (alpha)",
                total=n_alpha,
            )
        for ja in alpha_iter:
            alpha_w = alpha_weights[ja]
            if alpha_w == 0.0:
                continue
            active_beta = np.flatnonzero((w_te_all[ja, :] != 0) | (w_tm_all[ja, :] != 0))
            if active_beta.size == 0:
                continue
            kx = cupy.asarray(kx_all[ja, active_beta], dtype=cupy.float64)
            ky = cupy.asarray(ky_all[ja, active_beta], dtype=cupy.float64)
            kz = cupy.asarray(kz_all[ja, active_beta], dtype=cupy.float64)
            v = cupy.asarray(v_all[ja, active_beta, :], dtype=compute_dtype_cp)
            for s in range(0, n_points, batch_size_eff):
                e_idx = min(n_points, s + batch_size_eff)
                p = pts_gpu[s:e_idx, :]
                phase_arg = (
                    p[:, 0, None] * kx[None, :]
                    + p[:, 1, None] * ky[None, :]
                    + p[:, 2, None] * kz[None, :]
                )
                phase = cupy.exp(1j * phase_arg).astype(compute_dtype_cp, copy=False)
                weighted = phase @ v
                weighted_acc = (alpha_w * weighted).astype(accum_dtype_cp, copy=False)
                e_gpu[s:e_idx, 0] += weighted_acc[:, 0]
                e_gpu[s:e_idx, 1] += weighted_acc[:, 1]
                e_gpu[s:e_idx, 2] += weighted_acc[:, 2]
                h_gpu[s:e_idx, 0] += n_medium_c * weighted_acc[:, 3]
                h_gpu[s:e_idx, 1] += n_medium_c * weighted_acc[:, 4]
                h_gpu[s:e_idx, 2] += n_medium_c * weighted_acc[:, 5]
        return asnumpy(e_gpu).astype(accum_dtype, copy=False), asnumpy(h_gpu).astype(
            accum_dtype, copy=False
        )

    for ja in tqdm(
        range(n_alpha), desc="Initial field (alpha)", total=n_alpha, disable=not show_progress
    ):
        alpha_w = alpha_weights[ja]
        if alpha_w == 0.0:
            continue
        active_beta = np.flatnonzero((w_te_all[ja, :] != 0) | (w_tm_all[ja, :] != 0))
        if active_beta.size == 0:
            continue
        kx = kx_all[ja, active_beta]
        ky = ky_all[ja, active_beta]
        kz = kz_all[ja, active_beta]
        v = v_all[ja, active_beta, :]
        for s in range(0, n_points, batch_size_eff):
            e_idx = min(n_points, s + batch_size_eff)
            p = pts[s:e_idx, :]
            phase_arg = (
                p[:, 0, None] * kx[None, :]
                + p[:, 1, None] * ky[None, :]
                + p[:, 2, None] * kz[None, :]
            )
            phase = np.exp(1j * phase_arg).astype(compute_dtype, copy=False)
            weighted = phase @ v
            weighted_acc = (alpha_w * weighted).astype(accum_dtype, copy=False)
            e[s:e_idx, 0] += weighted_acc[:, 0]
            e[s:e_idx, 1] += weighted_acc[:, 1]
            e[s:e_idx, 2] += weighted_acc[:, 2]
            h[s:e_idx, 0] += n_medium_c * weighted_acc[:, 3]
            h[s:e_idx, 1] += n_medium_c * weighted_acc[:, 4]
            h[s:e_idx, 2] += n_medium_c * weighted_acc[:, 5]

    return e, h


__all__ = ["compute_initial_field"]
