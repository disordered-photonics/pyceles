"""SVWF translation operator blocks (CELES-compatible)."""

from __future__ import annotations

from functools import cache
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn, spherical_yn

from .indexing import index_vswf, iter_modes, n_modes
from .spherical import legendre_normalized_trigon_scalar
from .wigner import wigner_3j

# SciPy spherical_yn is defined for real arguments; CELES assumes real k in the embedding medium.


def spherical_bessel_jy(lmax: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays j_l(z), y_l(z) for l=0..lmax.

    Parameters
    ----------
    lmax:
        Maximum order.
    z:
        Array-like argument (typically real).

    Returns
    -------
    j, y:
        Arrays of shape (lmax+1, *z.shape).
    """
    z = np.asarray(z)
    j = np.zeros((lmax + 1,) + z.shape, dtype=np.complex128)
    y = np.zeros_like(j)
    for l in range(0, lmax + 1):
        j[l] = spherical_jn(l, z)
        y[l] = spherical_yn(l, z)
    return j, y


class RadialLUT:
    """Lookup table for spherical Hankel h_p^(1)(k r), p=0..2*lmax.

    Parameters
    ----------
    interpolation:
        `linear` (default) or `nearest`.
    """

    def __init__(
        self,
        lmax: int,
        k,
        r_max: float,
        dr: float,
        interpolation: str = "linear",
        dtype: npt.DTypeLike = np.complex128,
    ):
        self.lmax = int(lmax)
        k_c = complex(k)
        if abs(k_c.imag) > 0:
            raise ValueError(
                f"Translation RadialLUT requires real k (embedding medium must be non-absorbing). Got k={k_c}."
            )
        self.k = float(k_c.real)
        self.dr = float(dr)
        self.dtype = np.dtype(dtype)
        interp = str(interpolation).strip().lower()
        if interp not in {"linear", "nearest"}:
            raise ValueError(f"interpolation must be 'linear' or 'nearest', got {interpolation!r}.")
        self.interpolation = interp
        # Uniform radial grid up to r_max plus one guard interval so i0+1 is always valid.
        maxdist = float(r_max) + self.dr
        n_grid = int(np.ceil(maxdist / self.dr)) + 1
        self.r_grid = self.dr * np.arange(n_grid, dtype=float)
        self._inv_dr = 1.0 / self.dr
        self._last_index = self.r_grid.size - 1
        z = (self.k * self.r_grid).astype(np.complex128)
        # Avoid evaluating spherical Hankel at exactly kr=0 (singular y_l term).
        # We mirror the near-field LUT policy: replace the first sample with the
        # first positive-radius sample so interpolation for 0 < r < dr remains finite.
        if z.size > 1:
            z = z.copy()
            z[0] = z[1]
        j, y = spherical_bessel_jy(2 * self.lmax, z)
        self.h = (j + 1j * y).astype(self.dtype, copy=False)  # (p, Nr)

    def hankel_all_p(self, r: float) -> np.ndarray:
        """Return interpolated `h_p^(1)(k r)` for all `p=0..2*lmax` at radius `r`."""
        r = float(r)
        if r <= 0.0:
            return self.h[:, 0]
        t = r * self._inv_dr
        if self.interpolation == "nearest":
            i = int(np.round(t))
            i = max(0, min(i, self._last_index))
            return self.h[:, i]
        i0 = int(np.floor(t))
        frac = t - i0
        i0 = max(0, min(i0, self._last_index - 1))
        return (1.0 - frac) * self.h[:, i0] + frac * self.h[:, i0 + 1]


@cache
def _translation_ab5_table_cached(lmax: int, dtype_str: str) -> np.ndarray:
    """Port of CELES `translation_table_ab.m`.

    Returns an array `ab5` of shape `(Nm, Nm, 2*lmax+1)` used directly by
    `translation_block` in the CELES/SMUTHI contraction order.
    """
    lmax = int(lmax)
    dtype = np.dtype(dtype_str)
    Nm = n_modes(lmax)
    ab5 = np.zeros((Nm, Nm, 2 * lmax + 1), dtype=dtype)

    for tau1, l1, m1, _ in iter_modes(lmax):
        j1 = index_vswf(l1, m1, tau1, lmax)
        for tau2, l2, m2, _ in iter_modes(lmax):
            j2 = index_vswf(l2, m2, tau2, lmax)
            phase_exp_base = abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1
            sign_dm = -1.0 if ((m1 - m2) % 2) else 1.0  # (-1)^(m1-m2)
            pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1)))

            for p in range(0, 2 * lmax + 1):
                if tau1 == tau2:
                    i_phase = (1j) ** (phase_exp_base + p)
                    factor = (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1)
                    w = wigner_3j(l1, l2, p, m1, -m2, -m1 + m2) * wigner_3j(l1, l2, p, 0, 0, 0)
                    ab5[j2, j1, p] = i_phase * sign_dm * pref * factor * w
                else:
                    if p == 0:
                        continue
                    i_phase = (1j) ** (phase_exp_base + p)
                    inside = (
                        (l1 + l2 + 1 + p)
                        * (l1 + l2 + 1 - p)
                        * (p + l1 - l2)
                        * (p - l1 + l2)
                        * (2 * p + 1)
                    )
                    if inside < 0:
                        continue
                    factor = np.sqrt(inside)
                    w = wigner_3j(l1, l2, p, m1, -m2, -m1 + m2) * wigner_3j(l1, l2, p - 1, 0, 0, 0)
                    ab5[j2, j1, p] = i_phase * sign_dm * pref * factor * w
    ab5.setflags(write=False)
    return ab5


def translation_ab5_table(lmax: int, dtype=np.complex128) -> np.ndarray:
    """Cached translation `ab5` table.

    Results are cached by `(lmax, dtype)` and returned read-only.

    Notes
    -----
    This cache is intentionally unbounded. It is safe for the intended
    usage where `lmax` takes only a small set of values. If many distinct
    `(lmax, dtype)` combinations are used in a single process, memory grows
    monotonically.
    """
    return _translation_ab5_table_cached(int(lmax), np.dtype(dtype).str)


def clear_caches() -> None:
    """Clear process-global translation precompute caches.

    This is intended for long interactive sessions and exploratory parameter
    sweeps where many distinct `(lmax, dtype)` combinations may be visited in a
    single Python process. `pyceles` does not clear these caches automatically
    because repeated solves often benefit from keeping translation tables warm.
    """

    _translation_ab5_table_cached.cache_clear()
    _translation_mode_pair_tables.cache_clear()


@cache
def _translation_mode_pair_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute mode-pair tables for fast translation block assembly.

    Returns
    -------
    dm, absdm, dm_lookup
        dm and absdm have shape (dst, src). `dm_lookup` indexes a compact
        phase lookup table built on m=-2*lmax..2*lmax.
    """
    lmax = int(lmax)
    Nm = n_modes(lmax)

    m_of_idx = np.zeros((Nm,), dtype=np.int32)
    for _tau, _l, m, idx in iter_modes(lmax):
        m_of_idx[idx] = m

    dm = m_of_idx[None, :] - m_of_idx[:, None]  # (dst, src): m_src - m_dst
    absdm = np.abs(dm).astype(np.int32)
    dm_i = dm.astype(np.int32, copy=False)
    dm_lookup = dm_i + (2 * lmax)
    return dm_i, absdm, dm_lookup.astype(np.int32, copy=False)


def translation_block(
    lmax: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray,
    radial_lut: Optional[RadialLUT] = None,
) -> np.ndarray:
    """Dense translation block W_{ij} for two centers separated by rvec=r_i-r_j."""
    lmax = int(lmax)
    out_dtype = np.asarray(ab5).dtype
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    Nm = n_modes(lmax)

    r = float(np.linalg.norm(rvec))
    if r == 0.0:
        return np.zeros((Nm, Nm), dtype=out_dtype)

    ct = rvec[2] / r
    st = np.sqrt(max(0.0, 1.0 - ct * ct))
    phi = np.arctan2(rvec[1], rvec[0])

    if radial_lut is None:
        j, y = spherical_bessel_jy(2 * lmax, np.asarray(k * r, dtype=np.complex128))
        h = np.asarray(j + 1j * y, dtype=out_dtype).reshape((2 * lmax + 1,))
    else:
        h = np.asarray(radial_lut.hankel_all_p(r), dtype=out_dtype)

    # Scalar recurrence is materially faster here than generic vectorized variants
    # for the tiny per-pair angular workload in matrix-free solver matvec loops.
    plm = legendre_normalized_trigon_scalar(ct, st, 2 * lmax)
    dm, absdm, dm_lookup = _translation_mode_pair_tables(lmax)  # (dst,src)

    # g[m,p] = P_p^m(cos theta) * h_p(kr)
    g_mp = (plm * h[:, None]).T.astype(out_dtype, copy=False)  # (m,p)

    # Gather the m-dependent angular-radial factors for each mode pair.
    # gp[dst,src,p] = g_mp[absdm[dst,src], p]
    gp = g_mp[absdm]

    # `ab5` and angular-radial terms are already aligned for (dst, src, p)
    # contraction in the CELES/SMUTHI convention.
    acc = np.sum(ab5 * gp, axis=2)  # (dst,src)
    # Compute only the small set of phase factors for m=-2*lmax..2*lmax,
    # then gather to (dst,src) by precomputed lookup indices.
    m_phase = np.arange(-2 * lmax, 2 * lmax + 1, dtype=np.int32)
    phase_lut = np.exp(1j * phi * m_phase).astype(out_dtype, copy=False)
    phase = phase_lut[dm_lookup]
    W = acc * phase
    return W.astype(out_dtype, copy=False)
