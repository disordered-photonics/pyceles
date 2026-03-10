from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles.core.geometry_bounds import conservative_cross_set_max_distance
from pyceles.core.indexing import n_modes
from pyceles.core.spherical import spherical_functions_trigon

from .common import contract_modes, dx_xz_hankel1, mode_indices_by_l, sph_hankel1


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
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exterior scattered field from outgoing SVWF coefficients."""
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


__all__ = ["NearFieldRadialLUT", "compute_scattered_field"]
