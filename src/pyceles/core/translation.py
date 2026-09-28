"""SVWF translation operator blocks (CELES-compatible)."""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn, spherical_yn

from pyceles._arrays import expose_read_only_view
from .indexing import index_vswf, iter_modes, n_modes
from .spherical import _legendre_scalar_tables, legendre_normalized_trigon_scalar
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
    j = np.zeros((lmax + 1, *z.shape), dtype=np.complex128)
    y = np.zeros_like(j)
    for l in range(0, lmax + 1):
        j[l] = spherical_jn(l, z)
        y[l] = spherical_yn(l, z)
    return j, y


def _cast_radial_table_with_finite_guard(
    values: np.ndarray,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    """Cast radial rows while replacing only their singular guard prefixes.

    Different spherical orders become representable at different radii. A
    high-order overflow must therefore not flatten otherwise-valid low-order
    samples. For each radial row independently, retain every representable
    sample and repeat its first representable value only across that row's
    leading singular prefix. Non-finite values after the prefix indicate a
    genuine table-construction failure and are rejected.
    """
    source = np.asarray(values)
    target = np.dtype(dtype)
    if source.ndim == 0:
        return source.astype(target, copy=False)
    if source.dtype == target and np.isfinite(source).all():
        return source

    real_dtype = np.empty((), dtype=target).real.dtype
    max_finite = np.finfo(real_dtype).max
    rows = source.reshape(-1, source.shape[-1])
    out_rows = np.empty(rows.shape, dtype=target)
    for row, out_row in zip(rows, out_rows, strict=True):
        representable = (
            np.isfinite(row.real)
            & np.isfinite(row.imag)
            & (np.abs(row.real) <= max_finite)
            & (np.abs(row.imag) <= max_finite)
        )
        finite_indices = np.flatnonzero(representable)
        if finite_indices.size == 0:
            raise FloatingPointError(
                f"Radial table for dtype {target} has no finite sample in the requested range."
            )
        first_finite = int(finite_indices[0])
        if not np.all(representable[first_finite:]):
            raise FloatingPointError(
                "Radial table contains non-finite samples beyond its guard prefix."
            )
        out_row[first_finite:] = row[first_finite:].astype(target, copy=False)
        out_row[:first_finite] = out_row[first_finite]
    return out_rows.reshape(source.shape)


def spherical_bessel_j(lmax: int, z: np.ndarray) -> np.ndarray:
    """Return `j_l(z)` for `l=0..lmax`."""

    z = np.asarray(z)
    j = np.zeros((lmax + 1, *z.shape), dtype=np.complex128)
    for l in range(0, lmax + 1):
        j[l] = spherical_jn(l, z)
    return j


class RadialLUT:
    """Lookup table for radial spherical families up to `p=2*lmax`.

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
        self.h = _cast_radial_table_with_finite_guard(j + 1j * y, self.dtype)
        self.j = spherical_bessel_j(2 * self.lmax, self.k * self.r_grid).astype(
            self.dtype, copy=False
        )

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

    def bessel_j_all_p(self, r: float) -> np.ndarray:
        """Return interpolated `j_p(k r)` for all `p=0..2*lmax` at radius `r`."""

        r = float(r)
        if r <= 0.0:
            return self.j[:, 0]
        t = r * self._inv_dr
        if self.interpolation == "nearest":
            i = int(np.round(t))
            i = max(0, min(i, self._last_index))
            return self.j[:, i]
        i0 = int(np.floor(t))
        frac = t - i0
        i0 = max(0, min(i0, self._last_index - 1))
        return (1.0 - frac) * self.j[:, i0] + frac * self.j[:, i0 + 1]


@cache
def _mode_indices_within_larger_lmax(small_lmax: int, large_lmax: int) -> tuple[int, ...]:
    """Return CELES-order mode indices for `small_lmax` embedded in `large_lmax`."""

    small = int(small_lmax)
    large = int(large_lmax)
    if small > large:
        raise ValueError(f"small_lmax={small} cannot exceed large_lmax={large}.")
    return tuple(index_vswf(l, m, tau, large) for tau, l, m, _ in iter_modes(small))


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
    return expose_read_only_view(ab5)


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
    _translation_ab5_compact_tables_cached.cache_clear()
    _translation_plm_coeff_table_cached.cache_clear()
    _translation_mode_pair_tables.cache_clear()
    _rectangular_radial_lut_from_base.cache_clear()


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


def _poly_linear_combo(
    lhs: np.ndarray, rhs: np.ndarray, *, lhs_scale: float, rhs_scale: float
) -> np.ndarray:
    """Return `lhs_scale * lhs + rhs_scale * rhs` for 1D polynomial-coefficient arrays."""
    out_size = max(lhs.size, rhs.size)
    out = np.zeros((out_size,), dtype=np.float64)
    out[: lhs.size] += lhs_scale * lhs
    out[: rhs.size] += rhs_scale * rhs
    return out


@cache
def _translation_ab5_compact_tables_cached(
    lmax: int, dtype_str: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return CELES-style flattened real/imag `ab5` tables for direct GPU kernels."""
    lmax = int(lmax)
    ab5 = np.asarray(_translation_ab5_table_cached(lmax, dtype_str))
    real_dtype = np.float32 if np.dtype(dtype_str) == np.dtype(np.complex64) else np.float64
    re_entries: list[float] = []
    im_entries: list[float] = []
    for tau1, l1, m1, n1 in iter_modes(lmax):
        for tau2, l2, m2, n2 in iter_modes(lmax):
            p_min = max(abs(m1 - m2), abs(l1 - l2) + abs(tau1 - tau2))
            for p in range(p_min, l1 + l2 + 1):
                entry = ab5[n1, n2, p]
                re_entries.append(float(np.real(entry)))
                im_entries.append(float(np.imag(entry)))
    return (
        np.asarray(re_entries, dtype=real_dtype),
        np.asarray(im_entries, dtype=real_dtype),
    )


def _translation_ab5_compact_tables(lmax: int, dtype=np.complex64) -> tuple[np.ndarray, np.ndarray]:
    """Return CELES-style flattened real/imag `ab5` tables for direct GPU kernels."""
    return _translation_ab5_compact_tables_cached(int(lmax), np.dtype(dtype).str)


@cache
def _rectangular_radial_lut_from_base(
    lmax: int,
    k: float,
    r_max: float,
    dr: float,
    dtype_str: str,
) -> RadialLUT:
    """Return a higher-order radial LUT compatible with a base LUT."""

    return RadialLUT(
        lmax=int(lmax),
        k=float(k),
        r_max=float(r_max),
        dr=float(dr),
        dtype=np.dtype(dtype_str),
    )


def _translation_radial_values(
    lmax: int,
    k: float,
    r: float,
    *,
    out_dtype: np.dtype,
    family: str,
    radial_lut: RadialLUT | None,
) -> np.ndarray:
    """Return the radial sequence for one SVWF translation family."""

    fam = str(family).strip().lower()
    if fam == "outgoing_to_regular":
        if radial_lut is None:
            j, y = spherical_bessel_jy(2 * lmax, np.asarray(k * r, dtype=np.complex128))
            return np.asarray(j + 1j * y, dtype=out_dtype).reshape((2 * lmax + 1,))
        return np.asarray(radial_lut.hankel_all_p(r), dtype=out_dtype)[: 2 * lmax + 1]
    if fam in {"interior", "regular_to_regular"}:
        if radial_lut is None:
            return np.asarray(
                spherical_bessel_j(2 * lmax, np.asarray(k * r, dtype=np.complex128)),
                dtype=out_dtype,
            ).reshape((2 * lmax + 1,))
        return np.asarray(radial_lut.bessel_j_all_p(r), dtype=out_dtype)[: 2 * lmax + 1]
    raise ValueError(
        f"Unsupported translation family {family!r}. "
        "Expected 'outgoing_to_regular', 'interior', or 'regular_to_regular'."
    )


def _translation_block_family(
    lmax: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray,
    radial_lut: RadialLUT | None,
    family: str,
) -> np.ndarray:
    """Shared SVWF translation-block assembly for one radial family."""

    lmax = int(lmax)
    out_dtype = np.asarray(ab5).dtype
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    nmodes = n_modes(lmax)

    r = float(np.linalg.norm(rvec))
    if r == 0.0:
        if str(family).strip().lower() in {"interior", "regular_to_regular"}:
            return np.eye(nmodes, dtype=out_dtype)
        return np.zeros((nmodes, nmodes), dtype=out_dtype)

    ct = rvec[2] / r
    st = np.sqrt(max(0.0, 1.0 - ct * ct))
    phi = np.arctan2(rvec[1], rvec[0])
    radial = _translation_radial_values(
        lmax,
        k,
        r,
        out_dtype=out_dtype,
        family=family,
        radial_lut=radial_lut,
    )

    plm = legendre_normalized_trigon_scalar(ct, st, 2 * lmax)
    _dm, absdm, dm_lookup = _translation_mode_pair_tables(lmax)
    g_mp = (plm * radial[:, None]).T.astype(out_dtype, copy=False)
    gp = g_mp[absdm]
    acc = np.sum(ab5 * gp, axis=2)
    m_phase = np.arange(-2 * lmax, 2 * lmax + 1, dtype=np.int32)
    phase_lut = np.exp(1j * phi * m_phase).astype(out_dtype, copy=False)
    phase = phase_lut[dm_lookup]
    return np.asarray(acc * phase, dtype=out_dtype)


@cache
def _translation_plm_coeff_table_cached(lmax: int, dtype_str: str) -> np.ndarray:
    """Return CELES-style trigonometric Legendre coefficient table up to degree `2*lmax`."""
    lmax = int(lmax)
    out_dtype = np.float32 if np.dtype(dtype_str) == np.dtype(np.float32) else np.float64
    max_degree = 2 * lmax
    max_terms = lmax + 1
    q_poly: list[list[np.ndarray]] = [
        [np.zeros((0,), dtype=np.float64) for _ in range(max_degree + 1)]
        for _ in range(max_degree + 1)
    ]

    q_poly[0][0] = np.asarray([np.sqrt(2.0) / 2.0], dtype=np.float64)
    if max_degree >= 1:
        q_poly[1][0] = np.asarray([0.0, np.sqrt(3.0 / 2.0)], dtype=np.float64)

    a0, b0, c_mm, a_lm, b_lm = _legendre_scalar_tables(max_degree)
    for l in range(1, max_degree):
        q_poly[l + 1][0] = _poly_linear_combo(
            np.pad(q_poly[l][0], (1, 0)),
            q_poly[l - 1][0],
            lhs_scale=float(a0[l]),
            rhs_scale=-float(b0[l]),
        )

    for m in range(1, max_degree + 1):
        q_poly[m][m] = np.asarray([c_mm[m]], dtype=np.float64)
        for l in range(m, max_degree):
            q_poly[l + 1][m] = _poly_linear_combo(
                np.pad(q_poly[l][m], (1, 0)),
                q_poly[l - 1][m],
                lhs_scale=float(a_lm[l, m]),
                rhs_scale=-float(b_lm[l, m]),
            )

    coeffs = np.zeros((max_terms, max_degree + 1, max_degree + 1), dtype=out_dtype)
    for l in range(max_degree + 1):
        for m in range(l + 1):
            poly = q_poly[l][m]
            for jj, lam in enumerate(range(l - m, -1, -2)):
                coeffs[jj, m, l] = out_dtype(poly[lam]) if lam < poly.size else out_dtype(0.0)
    coeffs.setflags(write=False)
    return coeffs


def _translation_plm_coeff_table(lmax: int, dtype=np.float32) -> np.ndarray:
    """Return CELES-style trigonometric Legendre coefficient table up to degree `2*lmax`."""
    return _translation_plm_coeff_table_cached(int(lmax), np.dtype(dtype).str)


def translation_block(
    lmax: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray,
    radial_lut: RadialLUT | None = None,
) -> np.ndarray:
    """Return the outgoing-to-regular SVWF coupling block `W_ij`.

    This is the canonical free-space pair block used in the direct many-body
    operator, mapping outgoing coefficients at source center `j` to regular
    coefficients at destination center `i`.
    """
    return _translation_block_family(
        lmax,
        k,
        rvec,
        ab5=ab5,
        radial_lut=radial_lut,
        family="outgoing_to_regular",
    )


def translation_block_regular(
    lmax: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray,
) -> np.ndarray:
    """Return a regular-to-regular SVWF recentering block.

    This interior shift is used when one box expansion is re-expanded about a
    different center without switching from regular to outgoing families.
    """

    return translation_block_interior(lmax, k, rvec, ab5=ab5)


def translation_block_interior(
    lmax: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray,
) -> np.ndarray:
    """Return the dense interior `j_l` recentering block for two SVWF centers."""

    return _translation_block_family(
        lmax,
        k,
        rvec,
        ab5=ab5,
        radial_lut=None,
        family="interior",
    )


def translation_block_rect(
    lmax_out: int,
    lmax_in: int,
    k: float,
    rvec: np.ndarray,
    *,
    ab5: np.ndarray | None = None,
    radial_lut: RadialLUT | None = None,
    family: str = "outgoing_to_regular",
) -> np.ndarray:
    """Return a rectangular SVWF translation block between different orders.

    This is the helper used when parent and child boxes, or particles and
    boxes, do not share the same truncation order but still need one exact
    CELES-order translation block.
    """

    out_order = int(lmax_out)
    in_order = int(lmax_in)
    full_order = max(out_order, in_order)
    lut = radial_lut
    if radial_lut is not None and radial_lut.lmax < full_order:
        lut = _rectangular_radial_lut_from_base(
            full_order,
            radial_lut.k,
            float(radial_lut.r_grid[-1]),
            radial_lut.dr,
            radial_lut.dtype.str,
        )
    full_block = _translation_block_family(
        full_order,
        float(k),
        np.asarray(rvec, dtype=float),
        ab5=(
            np.asarray(ab5, dtype=np.complex128)
            if ab5 is not None
            else translation_ab5_table(full_order, dtype=np.complex128)
        ),
        radial_lut=lut,
        family=family,
    )
    row_idx = np.asarray(_mode_indices_within_larger_lmax(out_order, full_order), dtype=np.int64)
    col_idx = np.asarray(_mode_indices_within_larger_lmax(in_order, full_order), dtype=np.int64)
    return np.asarray(full_block[np.ix_(row_idx, col_idx)], dtype=full_block.dtype, copy=False)
