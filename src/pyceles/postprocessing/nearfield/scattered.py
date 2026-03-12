from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.geometry_bounds import conservative_cross_set_max_distance
from pyceles.core.indexing import n_modes
from pyceles.core.spherical import spherical_functions_trigon

from .common import contract_modes, dx_xz_hankel1, mode_indices_by_l, sph_hankel1


@cache
def _mode_indices_by_l_cupy(
    lmax: int,
    *,
    real_dtype_name: str,
) -> tuple[tuple[Any, Any, Any, Any], ...]:
    """Cache per-``l`` mode tables on device for repeated CuPy near-field calls.

    The index tables themselves are tiny, but rebuilding and re-uploading them
    on every scattered-field call is unnecessary. Caching them once per
    ``(lmax, real_dtype)`` keeps the hot path focused on particle/point work.
    """
    cupy, _ = import_cupy()
    real_dtype_cp = getattr(cupy, real_dtype_name)
    out = []
    for m_vals, abs_m, n1_idx, n2_idx in mode_indices_by_l(int(lmax)):
        out.append(
            (
                cupy.asarray(m_vals, dtype=real_dtype_cp),
                cupy.asarray(abs_m, dtype=cupy.int32),
                cupy.asarray(n1_idx, dtype=cupy.int32),
                cupy.asarray(n2_idx, dtype=cupy.int32),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class NearFieldRadialLUT:
    """Near-field lookup table for radial outgoing SVWF factors on uniform radius."""

    lmax: int
    k: float
    dr: float
    dtype: np.dtype
    ri: np.ndarray
    h: list[np.ndarray]
    dxxz: list[np.ndarray]
    _ri_gpu: Any
    _h_gpu: tuple[Any, ...]
    _dxxz_gpu: tuple[Any, ...]

    def __init__(
        self,
        *,
        lmax: int,
        k: float,
        r_max: float,
        dr: float = 1.0,
        dtype: npt.DTypeLike = np.complex128,
    ):
        object.__setattr__(self, "lmax", int(lmax))
        object.__setattr__(self, "k", float(k))
        object.__setattr__(self, "dr", float(dr))
        object.__setattr__(self, "dtype", np.dtype(dtype))
        ri = np.arange(0.0, r_max + dr, dr, dtype=np.float64)
        x = self.k * ri
        if x.size > 1:
            x = x.copy()
            x[0] = x[1]
        h = [np.zeros_like(x, dtype=self.dtype) for _ in range(self.lmax + 1)]
        dxxz = [np.zeros_like(x, dtype=self.dtype) for _ in range(self.lmax + 1)]
        for l in range(1, self.lmax + 1):
            h[l] = sph_hankel1(l, x).astype(self.dtype, copy=False)
            dxxz[l] = dx_xz_hankel1(l, x).astype(self.dtype, copy=False)
        object.__setattr__(self, "ri", ri)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "dxxz", dxxz)
        object.__setattr__(self, "_ri_gpu", None)
        object.__setattr__(self, "_h_gpu", tuple([None] * (self.lmax + 1)))
        object.__setattr__(self, "_dxxz_gpu", tuple([None] * (self.lmax + 1)))

    def interp(self, l: int, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate precomputed radial kernels at requested radii."""
        r = np.asarray(r, np.float64)
        idx = np.clip((r / self.dr).astype(np.int64), 0, len(self.ri) - 2)
        t = (r - self.ri[idx]) / self.dr
        z0 = self.h[l][idx]
        z1 = self.h[l][idx + 1]
        d0 = self.dxxz[l][idx]
        d1 = self.dxxz[l][idx + 1]
        z = (1 - t) * z0 + t * z1
        d = (1 - t) * d0 + t * d1
        return z, d

    def interp_cupy(self, l: int, r) -> tuple[Any, Any]:
        """Linearly interpolate precomputed radial kernels on device.

        The LUT tables are built once on CPU because they are setup-only data.
        When a CuPy near-field path is active we upload them lazily and keep the
        device copies resident on the LUT object so repeated batches do not
        bounce radial tables through host memory.
        """
        cupy, _ = import_cupy()
        if self._ri_gpu is None:
            object.__setattr__(self, "_ri_gpu", cupy.asarray(self.ri, dtype=cupy.float64))
        h_gpu = list(self._h_gpu)
        dxxz_gpu = list(self._dxxz_gpu)
        if h_gpu[l] is None:
            h_gpu[l] = cupy.asarray(self.h[l])
            dxxz_gpu[l] = cupy.asarray(self.dxxz[l])
            object.__setattr__(self, "_h_gpu", tuple(h_gpu))
            object.__setattr__(self, "_dxxz_gpu", tuple(dxxz_gpu))

        r_arr = cupy.asarray(r, dtype=cupy.float64)
        idx = cupy.clip((r_arr / self.dr).astype(cupy.int64), 0, len(self.ri) - 2)
        t = (r_arr - self._ri_gpu[idx]) / self.dr
        z0 = h_gpu[l][idx]
        z1 = h_gpu[l][idx + 1]
        d0 = dxxz_gpu[l][idx]
        d1 = dxxz_gpu[l][idx + 1]
        z = (1 - t) * z0 + t * z1
        d = (1 - t) * d0 + t * d1
        return z, d


def compute_scattered_field(
    field_points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_medium: complex = 1.0 + 0j,
    particle_distance_resolution: float = 0.0,
    lut: NearFieldRadialLUT | None = None,
    active_mask: np.ndarray | None = None,
    batch_size: int = 8192,
    show_progress: bool = True,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exterior scattered field from outgoing SVWF coefficients.

    The CuPy branch accelerates only the scattered contribution. Initial and
    internal near-field components still follow their existing reference
    implementations, so backend inheritance currently gives a partial
    near-field speedup rather than a fully GPU-native decomposition.
    """
    backend_name = str(backend).lower()
    if backend_name == "cupy":
        return _compute_scattered_field_cupy(
            field_points=field_points,
            positions=positions,
            coeffs=coeffs,
            k=k,
            lmax=lmax,
            n_medium=n_medium,
            particle_distance_resolution=particle_distance_resolution,
            lut=lut,
            active_mask=active_mask,
            batch_size=batch_size,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
    pts = np.asarray(field_points, np.float64)
    pos = np.asarray(positions, np.float64)
    lmax = int(lmax)
    k = float(k)
    n_medium_c = complex(n_medium)
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)

    n_spheres = pos.shape[0]
    n_modes_total = n_modes(lmax)
    c = np.asarray(coeffs, compute_dtype).reshape(n_spheres, n_modes_total)
    e = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    if n_spheres == 0:
        return e, h

    idx_eval: np.ndarray | None = None
    if active_mask is None:
        pts_eval = pts
        e_eval = e
        h_eval = h
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != pts.shape[0]:
            raise ValueError(f"`active_mask` must have length {pts.shape[0]}. Got {mask.shape[0]}.")
        idx_eval = np.flatnonzero(mask)
        if idx_eval.size == 0:
            return e, h
        pts_eval = pts[idx_eval]
        e_eval = np.zeros((pts_eval.shape[0], 3), dtype=accum_dtype)
        h_eval = np.zeros_like(e_eval)

    dr = float(particle_distance_resolution)
    if dr < 0.0:
        raise ValueError(
            f"`particle_distance_resolution` must be >= 0. Got {particle_distance_resolution!r}."
        )
    k_abs = float(abs(k))
    if k_abs <= 0.0:
        raise ValueError(f"`k` must be non-zero for near-field radial LUT setup. Got {k!r}.")
    dr = (1.0e-2 / k_abs) if dr == 0.0 else dr

    if lut is None:
        rmax = conservative_cross_set_max_distance(pos, pts_eval)
        lut = NearFieldRadialLUT(
            lmax=lmax,
            k=k,
            r_max=rmax,
            dr=dr,
            dtype=compute_dtype,
        )

    mode_by_l = mode_indices_by_l(lmax)
    sphere_iter = tqdm(
        range(n_spheres),
        desc="Scattered field (spheres)",
        total=n_spheres,
        disable=not show_progress,
    )
    for j_sphere in sphere_iter:
        for s in range(0, pts_eval.shape[0], batch_size):
            e_idx = min(pts_eval.shape[0], s + batch_size)
            p = pts_eval[s:e_idx]
            rvec = p - pos[j_sphere]

            r = np.linalg.norm(rvec, axis=1)
            r_safe = np.where(r == 0, 1e-30, r)
            e_r = rvec / r_safe[:, None]
            ct = np.clip(e_r[:, 2], -1.0, 1.0)
            st = np.sqrt(np.maximum(0.0, 1.0 - ct**2))
            phi = np.arctan2(rvec[:, 1], rvec[:, 0])

            e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
            e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
            kr = k * r_safe
            pi_all, tau_all, p_all = spherical_functions_trigon(
                ct, st, lmax, xp=np, return_plm=True
            )

            for l in range(1, lmax + 1):
                m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
                pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))
                z, dxxz = lut.interp(l, r)
                z_over_kr = z / kr
                dxxz_over_kr = dxxz / kr

                p_lm = p_all[l, abs_m, :].T
                pi_lm = pi_all[l, abs_m, :].T
                tau_lm = tau_all[l, abs_m, :].T
                eimphi = np.asarray(
                    np.exp(1j * phi[:, None] * m_vals[None, :]), dtype=compute_dtype
                )
                impi = np.asarray((1j * m_vals[None, :]) * pi_lm, dtype=compute_dtype)

                a_vec = c[j_sphere, n1_idx].astype(compute_dtype, copy=False)
                b_vec = c[j_sphere, n2_idx].astype(compute_dtype, copy=False)

                theta_phi_m = (
                    impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
                )
                m_all = pref * z[:, None, None] * theta_phi_m * eimphi[:, :, None]

                radial_er = (l * (l + 1.0) * z_over_kr)[:, None] * p_lm
                mix_theta_phi = (
                    tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
                )
                n_all = (
                    pref
                    * (
                        radial_er[:, :, None] * e_r[:, None, :]
                        + dxxz_over_kr[:, None, None] * mix_theta_phi
                    )
                    * eimphi[:, :, None]
                )

                e_eval[s:e_idx] += contract_modes(a_vec, m_all)
                e_eval[s:e_idx] += contract_modes(b_vec, n_all)
                h_eval[s:e_idx] += (-1j * n_medium_c) * contract_modes(a_vec, n_all)
                h_eval[s:e_idx] += (-1j * n_medium_c) * contract_modes(b_vec, m_all)

    if idx_eval is not None:
        e[idx_eval] = e_eval
        h[idx_eval] = h_eval
        return e, h
    return e_eval, h_eval


def _compute_scattered_field_cupy(
    *,
    field_points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    n_medium: complex,
    particle_distance_resolution: float,
    lut: NearFieldRadialLUT | None,
    active_mask: np.ndarray | None,
    batch_size: int,
    show_progress: bool,
    compute_dtype: npt.DTypeLike,
    accum_dtype: npt.DTypeLike,
) -> tuple[np.ndarray, np.ndarray]:
    """CuPy-accelerated exterior scattered field assembly.

    This keeps the same near-field formulas as the NumPy reference path but
    batches spheres and field points on device. We deliberately accelerate only
    the scattered component first because it is the dominant sphere loop in the
    current profile and maps cleanly onto CuPy tensor contractions.
    """
    cupy, _ = import_cupy()

    pts = np.asarray(field_points, np.float64)
    pos = np.asarray(positions, np.float64)
    lmax = int(lmax)
    k = float(k)
    n_medium_c = complex(n_medium)
    compute_dtype_np = np.dtype(compute_dtype)
    accum_dtype_np = np.dtype(accum_dtype)
    compute_dtype_cp = (
        cupy.complex64 if compute_dtype_np == np.dtype(np.complex64) else cupy.complex128
    )
    accum_dtype_cp = cupy.complex64 if accum_dtype_np == np.dtype(np.complex64) else cupy.complex128
    real_dtype_cp = cupy.float32 if compute_dtype_np == np.dtype(np.complex64) else cupy.float64

    n_spheres = pos.shape[0]
    n_modes_total = n_modes(lmax)
    c = np.asarray(coeffs, compute_dtype_np).reshape(n_spheres, n_modes_total)
    e = np.zeros((pts.shape[0], 3), dtype=accum_dtype_np)
    h = np.zeros_like(e)
    if n_spheres == 0:
        return e, h

    idx_eval: np.ndarray | None = None
    if active_mask is None:
        pts_eval = pts
        e_eval = e
        h_eval = h
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != pts.shape[0]:
            raise ValueError(f"`active_mask` must have length {pts.shape[0]}. Got {mask.shape[0]}.")
        idx_eval = np.flatnonzero(mask)
        if idx_eval.size == 0:
            return e, h
        pts_eval = pts[idx_eval]
        e_eval = np.zeros((pts_eval.shape[0], 3), dtype=accum_dtype_np)
        h_eval = np.zeros_like(e_eval)

    dr = float(particle_distance_resolution)
    if dr < 0.0:
        raise ValueError(
            f"`particle_distance_resolution` must be >= 0. Got {particle_distance_resolution!r}."
        )
    k_abs = float(abs(k))
    if k_abs <= 0.0:
        raise ValueError(f"`k` must be non-zero for near-field radial LUT setup. Got {k!r}.")
    dr = (1.0e-2 / k_abs) if dr == 0.0 else dr

    if lut is None:
        rmax = conservative_cross_set_max_distance(pos, pts_eval)
        lut = NearFieldRadialLUT(
            lmax=lmax,
            k=k,
            r_max=rmax,
            dr=dr,
            dtype=compute_dtype_np,
        )

    pos_gpu = cupy.asarray(pos, dtype=real_dtype_cp)
    coeffs_gpu = cupy.asarray(c, dtype=compute_dtype_cp)
    pts_gpu = cupy.asarray(pts_eval, dtype=real_dtype_cp)
    e_gpu = cupy.zeros((pts_eval.shape[0], 3), dtype=accum_dtype_cp)
    h_gpu = cupy.zeros_like(e_gpu)

    mode_by_l = _mode_indices_by_l_cupy(lmax, real_dtype_name=real_dtype_cp.__name__)
    # The fixed sphere chunk is the leanest policy we have shipped so far.
    # A more elaborate hardware-aware chunk heuristic was benchmarked here, but
    # the gains were not robust across near-field canvases and regressed the
    # dense `dx=10` profile. Keep the simple default until a clearer win
    # justifies more policy surface.
    sphere_chunk_size = 8
    sphere_pbar = None
    if show_progress:
        sphere_pbar = tqdm(total=n_spheres, desc="Scattered field (spheres)")

    medium_factor = accum_dtype_cp(np.asarray(-1j * n_medium_c, dtype=accum_dtype_np))

    for j_sphere in range(0, n_spheres, sphere_chunk_size):
        j_stop = min(n_spheres, j_sphere + sphere_chunk_size)
        pos_chunk = pos_gpu[j_sphere:j_stop]
        coeff_chunk = coeffs_gpu[j_sphere:j_stop]

        for s in range(0, pts_eval.shape[0], batch_size):
            e_idx = min(pts_eval.shape[0], s + batch_size)
            p = pts_gpu[s:e_idx]
            rvec = p[None, :, :] - pos_chunk[:, None, :]

            r = cupy.linalg.norm(rvec, axis=2)
            r_safe = cupy.where(r == 0, real_dtype_cp(1e-30), r)
            e_r = rvec / r_safe[:, :, None]
            ct = cupy.clip(e_r[:, :, 2], -1.0, 1.0)
            st = cupy.sqrt(cupy.maximum(0.0, 1.0 - ct**2))
            phi = cupy.arctan2(rvec[:, :, 1], rvec[:, :, 0])

            e_theta = cupy.stack([ct * cupy.cos(phi), ct * cupy.sin(phi), -st], axis=2)
            e_phi = cupy.stack([-cupy.sin(phi), cupy.cos(phi), cupy.zeros_like(phi)], axis=2)
            kr = real_dtype_cp(k) * r_safe
            pi_all, tau_all, p_all = spherical_functions_trigon(
                ct, st, lmax, xp=cupy, return_plm=True
            )

            e_chunk = cupy.zeros((e_idx - s, 3), dtype=accum_dtype_cp)
            h_chunk = cupy.zeros_like(e_chunk)

            for l in range(1, lmax + 1):
                m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
                pref = real_dtype_cp(1.0 / np.sqrt(2.0 * l * (l + 1.0)))
                z, dxxz = lut.interp_cupy(l, r)
                z = z.astype(compute_dtype_cp, copy=False)
                dxxz = dxxz.astype(compute_dtype_cp, copy=False)
                z_over_kr = z / kr.astype(compute_dtype_cp, copy=False)
                dxxz_over_kr = dxxz / kr.astype(compute_dtype_cp, copy=False)

                p_lm = cupy.transpose(p_all[l][abs_m], (1, 2, 0)).astype(
                    compute_dtype_cp, copy=False
                )
                pi_lm = cupy.transpose(pi_all[l][abs_m], (1, 2, 0)).astype(
                    compute_dtype_cp, copy=False
                )
                tau_lm = cupy.transpose(tau_all[l][abs_m], (1, 2, 0)).astype(
                    compute_dtype_cp, copy=False
                )
                eimphi = cupy.exp(1j * phi[:, :, None] * m_vals[None, None, :]).astype(
                    compute_dtype_cp, copy=False
                )
                impi = ((1j * m_vals[None, None, :]) * pi_lm).astype(compute_dtype_cp, copy=False)

                a_vec = coeff_chunk[:, n1_idx].astype(compute_dtype_cp, copy=False)
                b_vec = coeff_chunk[:, n2_idx].astype(compute_dtype_cp, copy=False)
                # Contract over the mode index first so we do not materialize
                # the full `(sphere, batch, mode, xyz)` M/N tensors. The
                # algebra is the same as the reference formulation, but the GPU
                # path only keeps a handful of scalar fields per `(sphere,point)`
                # pair alive before combining them with the basis vectors.
                a_phase = a_vec[:, None, :] * eimphi
                b_phase = b_vec[:, None, :] * eimphi

                a_impi = cupy.sum(a_phase * impi, axis=2)
                a_tau = cupy.sum(a_phase * tau_lm, axis=2)
                a_p = cupy.sum(a_phase * p_lm, axis=2)
                b_impi = cupy.sum(b_phase * impi, axis=2)
                b_tau = cupy.sum(b_phase * tau_lm, axis=2)
                b_p = cupy.sum(b_phase * p_lm, axis=2)

                m_from_a = (
                    pref
                    * z[:, :, None]
                    * (a_impi[:, :, None] * e_theta - a_tau[:, :, None] * e_phi)
                )
                m_from_b = (
                    pref
                    * z[:, :, None]
                    * (b_impi[:, :, None] * e_theta - b_tau[:, :, None] * e_phi)
                )

                radial_pref = real_dtype_cp(l * (l + 1.0))
                n_from_a = pref * (
                    (radial_pref * z_over_kr * a_p)[:, :, None] * e_r
                    + dxxz_over_kr[:, :, None]
                    * (a_tau[:, :, None] * e_theta + a_impi[:, :, None] * e_phi)
                )
                n_from_b = pref * (
                    (radial_pref * z_over_kr * b_p)[:, :, None] * e_r
                    + dxxz_over_kr[:, :, None]
                    * (b_tau[:, :, None] * e_theta + b_impi[:, :, None] * e_phi)
                )

                e_chunk += cupy.sum(
                    m_from_a + n_from_b,
                    axis=0,
                ).astype(accum_dtype_cp, copy=False)
                h_chunk += medium_factor * cupy.sum(
                    n_from_a + m_from_b,
                    axis=0,
                ).astype(accum_dtype_cp, copy=False)

            e_gpu[s:e_idx] += e_chunk
            h_gpu[s:e_idx] += h_chunk
        if sphere_pbar is not None:
            sphere_pbar.update(j_stop - j_sphere)

    if sphere_pbar is not None:
        sphere_pbar.close()

    e_eval[:] = asnumpy(e_gpu).astype(accum_dtype_np, copy=False)
    h_eval[:] = asnumpy(h_gpu).astype(accum_dtype_np, copy=False)
    if idx_eval is not None:
        e[idx_eval] = e_eval
        h[idx_eval] = h_eval
        return e, h
    return e_eval, h_eval


__all__ = ["NearFieldRadialLUT", "compute_scattered_field"]
