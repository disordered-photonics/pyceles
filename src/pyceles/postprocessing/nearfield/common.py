from __future__ import annotations

from functools import cache

import numpy as np
from scipy.special import spherical_jn, spherical_yn

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
        out.append((m_vals, abs_m, n1_idx, n2_idx))
    return tuple(out)


def sph_hankel1(l: int, x: np.ndarray) -> np.ndarray:
    """Spherical Hankel function of first kind h_l^(1)(x)."""
    return spherical_jn(l, x) + 1j * spherical_yn(l, x)


def dx_xz_hankel1(l: int, x: np.ndarray) -> np.ndarray:
    """Return d/dx (x z_l(x)) for z_l = h_l^(1)."""
    z = sph_hankel1(l, x)
    dzdx = spherical_jn(l, x, derivative=True) + 1j * spherical_yn(l, x, derivative=True)
    return z + x * dzdx


def contract_modes(mode_coeffs: np.ndarray, mode_tensor: np.ndarray) -> np.ndarray:
    """Contract mode axis: (M,) x (B,M,3) -> (B,3)."""
    return np.matmul(np.transpose(mode_tensor, (0, 2, 1)), mode_coeffs)


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
    dxxz: np.ndarray,
    kr: np.ndarray,
    compute_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(M_l, N_l)` mode tensors for one degree and radial basis."""
    pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))
    z_over_kr = z_l / kr
    dxxz_over_kr = dxxz / kr

    p_lm = p_all[l, abs_m, :].T
    pi_lm = pi_all[l, abs_m, :].T
    tau_lm = tau_all[l, abs_m, :].T

    eimphi = np.asarray(np.exp(1j * phi[:, None] * m_vals[None, :]), dtype=compute_dtype)
    impi = np.asarray((1j * m_vals[None, :]) * pi_lm, dtype=compute_dtype)

    theta_phi_m = impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
    m_all = pref * z_l[:, None, None] * theta_phi_m * eimphi[:, :, None]

    radial_er = (l * (l + 1.0) * z_over_kr)[:, None] * p_lm
    mix_theta_phi = tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
    n_all = (
        pref
        * (radial_er[:, :, None] * e_r[:, None, :] + dxxz_over_kr[:, None, None] * mix_theta_phi)
        * eimphi[:, :, None]
    )
    return m_all, n_all


__all__ = [
    "build_internal_mode_tensors",
    "contract_modes",
    "dx_xz_hankel1",
    "mode_indices_by_l",
    "sph_hankel1",
]
