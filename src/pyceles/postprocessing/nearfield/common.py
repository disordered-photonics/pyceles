from __future__ import annotations

from functools import cache
from typing import cast

import numpy as np

from pyceles._arrays import expose_read_only_view
from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.indexing import index_vswf


@cache
def mode_indices_by_l(
    lmax: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Cache per-l arrays for vectorized mode contractions."""
    lmax = int(lmax)
    out = []
    for l in range(1, lmax + 1):
        m_vals = np.arange(-l, l + 1, dtype=np.int32)
        abs_m = np.abs(m_vals).astype(np.int32)
        n1_idx = np.array([index_vswf(l, int(m), 1, lmax) for m in m_vals], dtype=np.int32)
        n2_idx = np.array([index_vswf(l, int(m), 2, lmax) for m in m_vals], dtype=np.int32)
        out.append(
            (
                expose_read_only_view(m_vals),
                expose_read_only_view(abs_m),
                expose_read_only_view(n1_idx),
                expose_read_only_view(n2_idx),
            )
        )
    return tuple(out)


def contract_modes(mode_coeffs: np.ndarray, mode_tensor: np.ndarray) -> np.ndarray:
    """Contract mode axis: (M,) x (B,M,3) -> (B,3)."""
    return cast(np.ndarray, np.matmul(np.transpose(mode_tensor, (0, 2, 1)), mode_coeffs))


def build_internal_mode_tensors(
    *,
    l: int,
    m_vals: np.ndarray,
    abs_m: np.ndarray,
    phi: np.ndarray,
    e_r: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    pi_all: np.ndarray,
    tau_all: np.ndarray,
    p_all: np.ndarray,
    z_l: np.ndarray,
    z_over_kr: np.ndarray,
    dxxz_over_kr: np.ndarray,
    compute_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(batch, 2*l+1, 3)` M/N tensors from normalized radial factors."""
    if is_cupy_array(phi) or is_cupy_array(z_l):
        cupy, _ = import_cupy()
        xp = cupy
        dtype = cupy.dtype(compute_dtype)
        m_vals_arr = cupy.asarray(m_vals, dtype=cupy.int32)
        abs_m_arr = cupy.asarray(abs_m, dtype=cupy.int32)
    else:
        xp = np
        dtype = np.dtype(compute_dtype)
        m_vals_arr = np.asarray(m_vals, dtype=np.int32)
        abs_m_arr = np.asarray(abs_m, dtype=np.int32)

    real_scalar = np.float32 if np.dtype(compute_dtype) == np.dtype(np.complex64) else np.float64
    pref = real_scalar(1.0 / np.sqrt(2.0 * l * (l + 1.0)))
    ll = real_scalar(l * (l + 1.0))

    p_lm = p_all[l][abs_m_arr].T
    pi_lm = pi_all[l][abs_m_arr].T
    tau_lm = tau_all[l][abs_m_arr].T

    eimphi = xp.asarray(xp.exp(1j * phi[:, None] * m_vals_arr[None, :]), dtype=dtype)
    impi = xp.asarray((1j * m_vals_arr[None, :]) * pi_lm, dtype=dtype)

    theta_phi_m = impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
    m_all = pref * z_l[:, None, None] * theta_phi_m * eimphi[:, :, None]

    radial_er = (ll * z_over_kr)[:, None] * p_lm
    mix_theta_phi = tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
    n_all = (
        pref
        * (radial_er[:, :, None] * e_r[:, None, :] + dxxz_over_kr[:, None, None] * mix_theta_phi)
        * eimphi[:, :, None]
    )
    return m_all, n_all


def clear_caches() -> None:
    """Clear process-global near-field helper caches."""

    mode_indices_by_l.cache_clear()


__all__ = [
    "build_internal_mode_tensors",
    "clear_caches",
    "contract_modes",
    "mode_indices_by_l",
]
