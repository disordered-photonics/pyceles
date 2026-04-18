"""Directional sampled-basis helpers for pyceles MLFMM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import scipy.sparse

from pyceles.core.indexing import n_scalar, scalar_index
from pyceles.core.spherical import legendre_normalized_trigon_scalar

Array = np.ndarray


@dataclass(frozen=True)
class MLFMMDirectionalGrid:
    """Shared sampled angular grid for one MLFMM directional level.

    The sampled channels live on the physical propagation directions stored in
    `directions`. The `reflection_permutation` maps samples under
    `(alpha, beta) -> (alpha, pi - beta)` so the physical convention stays
    centralized here rather than leaking into geometry offsets elsewhere.
    """

    order: int
    alpha: Array
    beta: Array
    beta_weights: Array
    directions: Array
    weights: Array
    reflection_permutation: Array


@dataclass(frozen=True)
class MLFMMDirectionalTransforms:
    """SVWF <-> sampled-direction transform operators for one shared grid.

    Canonical storage keeps only `Fth/Fph`. `G` operators and all adjoints are
    derived lazily from directional-basis identities when needed.
    """

    box_order: int
    grid: MLFMMDirectionalGrid
    Fth: Array
    Fph: Array

    @cached_property
    def Gth(self) -> Array:
        return np.asarray(1j * self.Fph, dtype=np.complex128)

    @cached_property
    def Gph(self) -> Array:
        return np.asarray(-1j * self.Fth, dtype=np.complex128)

    @cached_property
    def Fth_adj(self) -> Array:
        return np.asarray(np.conjugate(self.Fth.T), dtype=np.complex128)

    @cached_property
    def Fph_adj(self) -> Array:
        return np.asarray(np.conjugate(self.Fph.T), dtype=np.complex128)

    @cached_property
    def Gth_adj(self) -> Array:
        return np.asarray(-1j * self.Fph_adj, dtype=np.complex128)

    @cached_property
    def Gph_adj(self) -> Array:
        return np.asarray(1j * self.Fth_adj, dtype=np.complex128)


@dataclass(frozen=True)
class MLFMMDirectionalInterpolation:
    """Interpolation matrix between two sampled directional grids."""

    source_order: int
    target_order: int
    matrix: scipy.sparse.csr_matrix


def _cache_key_factor(value: float) -> int:
    return round(float(value) * 1_000_000)


def _from_cache_key_factor(value: int) -> float:
    return float(value) / 1_000_000.0


def _legendre2(n: int, x: float) -> Array:
    """Return stable 4pi-normalized associated Legendre values for one degree.

    The directional transform builder only needs the Legendre values after the
    spherical-harmonic normalization is applied. Reusing the canonical
    normalized CELES recurrence keeps the shallow/high-order path finite
    without introducing a second angular-recurrence implementation here.
    """

    n_i = int(n)
    if n_i < 0:
        raise ValueError(f"n must be >= 0. Got {n}.")
    x_f = float(np.clip(x, -1.0, 1.0))
    st_f = math.sqrt(max(0.0, 1.0 - x_f * x_f))
    plm = legendre_normalized_trigon_scalar(x_f, st_f, n_i)
    phase = np.where(np.arange(n_i + 1, dtype=np.int32) & 1, -1.0, 1.0)
    return np.asarray(
        phase * plm[n_i, : n_i + 1] / math.sqrt(2.0 * np.pi),
        dtype=np.float64,
    )


def _directional_reflection_permutation(
    alpha: Array, beta: Array, *, tol: float = 1.0e-12
) -> Array:
    """Return sampled-direction reindexing for `(alpha, beta) -> (alpha, pi - beta)`."""

    alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    n_alpha = int(alpha_arr.size)
    n_beta = int(beta_arr.size)
    beta_targets = np.pi - beta_arr
    beta_perm = np.empty((n_beta,), dtype=np.int64)

    for ib, target in enumerate(beta_targets):
        idx = int(np.argmin(np.abs(beta_arr - float(target))))
        if abs(float(beta_arr[idx]) - float(target)) > float(tol):
            raise ValueError("beta grid does not support reflection mapping to pi-beta.")
        beta_perm[ib] = idx

    if np.unique(beta_perm).size != n_beta:
        raise ValueError("beta reflection mapping is not a permutation on the sampled grid.")

    perm = np.empty((n_alpha * n_beta,), dtype=np.int64)
    for ia in range(n_alpha):
        for ib in range(n_beta):
            perm[ia * n_beta + ib] = ia * n_beta + int(beta_perm[ib])
    if not np.array_equal(perm[perm], np.arange(perm.size, dtype=np.int64)):
        raise ValueError("reflection permutation is not involutive.")
    return perm


def apply_directional_reflection(
    permutation: Array,
    a_theta: Array,
    a_phi: Array,
    b_theta: Array,
    b_phi: Array,
) -> tuple[Array, Array, Array, Array]:
    """Reindex sampled channels under the physical `(alpha, beta) -> (alpha, pi-beta)` map."""

    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    size = int(perm.size)
    channels = [
        np.asarray(a_theta, dtype=np.complex128).reshape(-1),
        np.asarray(a_phi, dtype=np.complex128).reshape(-1),
        np.asarray(b_theta, dtype=np.complex128).reshape(-1),
        np.asarray(b_phi, dtype=np.complex128).reshape(-1),
    ]
    if any(channel.size != size for channel in channels):
        raise ValueError("Directional channels must match the permutation size.")
    return (
        np.asarray(channels[0][perm], dtype=np.complex128, copy=False),
        np.asarray(channels[1][perm], dtype=np.complex128, copy=False),
        np.asarray(channels[2][perm], dtype=np.complex128, copy=False),
        np.asarray(channels[3][perm], dtype=np.complex128, copy=False),
    )


@cache
def _cached_directional_grid(
    order: int,
    alpha_factor_key: int,
    beta_factor_key: int,
) -> MLFMMDirectionalGrid:
    order_i = max(1, int(order))
    alpha_factor = max(1.0, _from_cache_key_factor(alpha_factor_key))
    beta_factor = max(1.0, _from_cache_key_factor(beta_factor_key))
    n_beta = max(2, int(np.ceil(beta_factor * (order_i + 1))))
    n_alpha = max(4, int(np.ceil(alpha_factor * (2 * (order_i + 1)))))
    if n_alpha % 2 != 0:
        n_alpha += 1

    cos_beta, w_beta = np.polynomial.legendre.leggauss(n_beta)
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0)).astype(float, copy=False)
    beta_weights = np.asarray(w_beta, dtype=float)
    alpha = ((2.0 * np.pi * np.arange(n_alpha, dtype=float) / n_alpha) + (np.pi / n_alpha)).astype(
        float, copy=False
    )

    agrid = alpha[:, None]
    bgrid = beta[None, :]
    directions = np.stack(
        (
            np.sin(bgrid) * np.cos(agrid),
            np.sin(bgrid) * np.sin(agrid),
            np.broadcast_to(np.cos(beta), (n_alpha, n_beta)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = np.broadcast_to(
        (beta_weights * (2.0 * np.pi / float(n_alpha)))[None, :],
        (n_alpha, n_beta),
    ).reshape(-1)

    return MLFMMDirectionalGrid(
        order=order_i,
        alpha=np.asarray(alpha, dtype=float),
        beta=np.asarray(beta, dtype=float),
        beta_weights=np.asarray(beta_weights, dtype=float),
        directions=np.asarray(directions, dtype=float),
        weights=np.asarray(weights, dtype=float),
        reflection_permutation=_directional_reflection_permutation(alpha, beta),
    )


def directional_grid(
    order: int,
    *,
    alpha_factor: float = 1.0,
    beta_factor: float = 1.0,
) -> MLFMMDirectionalGrid:
    """Return the shared sampled angular grid for one directional level."""

    return _cached_directional_grid(
        int(order),
        _cache_key_factor(alpha_factor),
        _cache_key_factor(beta_factor),
    )


@cache
def _cached_directional_transforms(
    box_order: int,
    grid_order: int,
    alpha_factor_key: int,
    beta_factor_key: int,
) -> MLFMMDirectionalTransforms:
    grid = directional_grid(
        int(grid_order),
        alpha_factor=_from_cache_key_factor(alpha_factor_key),
        beta_factor=_from_cache_key_factor(beta_factor_key),
    )
    ndir = int(grid.directions.shape[0])
    nscl = n_scalar(int(box_order))
    fth = np.zeros((ndir, nscl), dtype=np.complex128)
    fph = np.zeros_like(fth)

    for idir, direction in enumerate(np.asarray(grid.directions, dtype=float)):
        theta = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
        phi = float(np.arctan2(direction[1], direction[0]))
        if phi < 0.0:
            phi += 2.0 * np.pi
        sin_theta = max(1.0e-15, float(np.sin(theta)))
        exp_cache: dict[int, complex] = {}
        legendre_by_n = {
            n: _legendre2(n, float(np.cos(theta))) for n in range(0, int(box_order) + 2)
        }
        for l in range(1, int(box_order) + 1):
            l_arr = legendre_by_n[l]
            l1_arr = legendre_by_n[l + 1]
            l2_arr = legendre_by_n[l - 1] if l > 1 else legendre_by_n[0]
            q = math.sqrt(l * (l + 1.0)) / ((2.0 * l + 1.0) * sin_theta)
            for m in range(-l, l + 1):
                mm = abs(m)
                if m not in exp_cache:
                    exp_cache[m] = complex(np.exp(1j * m * phi))
                phase = exp_cache[m]
                y = complex(l_arr[mm] * phase)
                y1 = complex(
                    math.sqrt(
                        (2.0 * l + 1.0)
                        / (2.0 * l + 3.0)
                        * (((l + 1.0) * (l + 1.0) - mm * mm) / ((l + 1.0) * (l + 1.0)))
                    )
                    * l1_arr[mm]
                    * phase
                )
                y2 = (
                    0.0j
                    if mm == l
                    else complex(
                        math.sqrt((2.0 * l + 1.0) / (2.0 * l - 1.0) * ((l * l - mm * mm) / (l * l)))
                        * l2_arr[mm]
                        * phase
                    )
                )
                b_theta = (y1 - y2) * q
                b_phi = (1j * m * (2.0 * l + 1.0) / (l * (l + 1.0)) * y) * q
                idx = scalar_index(l, m)
                fth[idir, idx] = 1j * (1j**l) * b_theta
                fph[idir, idx] = 1j * (1j**l) * b_phi

    return MLFMMDirectionalTransforms(
        box_order=int(box_order),
        grid=grid,
        Fth=fth,
        Fph=fph,
    )


def directional_transforms(
    box_order: int,
    *,
    grid_order: int | None = None,
    alpha_factor: float = 1.0,
    beta_factor: float = 1.0,
) -> MLFMMDirectionalTransforms:
    """Return cached SVWF <-> sampled-direction transforms for one box order."""

    effective_grid_order = int(box_order) if grid_order is None else int(grid_order)
    return _cached_directional_transforms(
        int(box_order),
        effective_grid_order,
        _cache_key_factor(alpha_factor),
        _cache_key_factor(beta_factor),
    )


def box_outgoing_to_directional(
    transforms: MLFMMDirectionalTransforms,
    box_state: Array,
) -> tuple[Array, Array, Array, Array]:
    """Map one outgoing box SVWF state to sampled physical directional channels."""

    coeffs = np.asarray(box_state, dtype=np.complex128).reshape(-1)
    nscl = transforms.Fth.shape[1]
    if coeffs.size != 2 * nscl:
        raise ValueError(f"box_state must have length {2 * nscl}, got {coeffs.size}.")
    a_box = coeffs[:nscl]
    b_box = coeffs[nscl:]
    a_theta = transforms.Fth @ a_box
    a_phi = transforms.Fph @ a_box
    b_theta = transforms.Fth @ b_box
    b_phi = transforms.Fph @ b_box
    return apply_directional_reflection(
        transforms.grid.reflection_permutation,
        np.asarray(a_theta, dtype=np.complex128, copy=False),
        np.asarray(a_phi, dtype=np.complex128, copy=False),
        np.asarray(b_theta, dtype=np.complex128, copy=False),
        np.asarray(b_phi, dtype=np.complex128, copy=False),
    )


def directional_to_box_regular(
    transforms: MLFMMDirectionalTransforms,
    a_theta: Array,
    a_phi: Array,
    b_theta: Array,
    b_phi: Array,
) -> Array:
    """Map sampled physical directional channels to one regular box SVWF state."""

    a_theta_arr, a_phi_arr, b_theta_arr, b_phi_arr = apply_directional_reflection(
        transforms.grid.reflection_permutation,
        np.asarray(a_theta, dtype=np.complex128).reshape(-1),
        np.asarray(a_phi, dtype=np.complex128).reshape(-1),
        np.asarray(b_theta, dtype=np.complex128).reshape(-1),
        np.asarray(b_phi, dtype=np.complex128).reshape(-1),
    )
    fth_adj = transforms.Fth_adj
    fph_adj = transforms.Fph_adj
    top = (
        fth_adj @ a_theta_arr
        + fph_adj @ a_phi_arr
        - 1j * (fph_adj @ b_theta_arr)
        + 1j * (fth_adj @ b_phi_arr)
    )
    bottom = (
        fth_adj @ b_theta_arr
        + fph_adj @ b_phi_arr
        - 1j * (fph_adj @ a_theta_arr)
        + 1j * (fth_adj @ a_phi_arr)
    )
    return np.concatenate((top, bottom)).astype(np.complex128, copy=False)


def _grid_key_bytes(values: Array) -> bytes:
    """Return a stable cache key payload for one sampled angular grid."""

    return np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)).tobytes()


def _build_sparse_directional_interpolation(
    *,
    source_alpha: Array,
    source_beta: Array,
    target_alpha: Array,
    target_beta: Array,
    stencil_half_width: int = 2,
) -> scipy.sparse.csr_matrix:
    """Build the validated sparse theta/phi interpolation used in multilevel HF transfers."""

    alpha_src = np.asarray(source_alpha, dtype=float).reshape(-1)
    beta_src = np.asarray(source_beta, dtype=float).reshape(-1)
    alpha_dst = np.asarray(target_alpha, dtype=float).reshape(-1)
    beta_dst = np.asarray(target_beta, dtype=float).reshape(-1)
    n_alpha_src = int(alpha_src.size)
    n_beta_src = int(beta_src.size)
    n_alpha_dst = int(alpha_dst.size)
    n_beta_dst = int(beta_dst.size)
    source_size = n_alpha_src * n_beta_src
    target_size = n_alpha_dst * n_beta_dst

    if (
        n_alpha_src == n_alpha_dst
        and n_beta_src == n_beta_dst
        and np.allclose(alpha_src, alpha_dst)
        and np.allclose(beta_src, beta_dst)
    ):
        rows = np.arange(target_size, dtype=np.int64)
        return scipy.sparse.csr_matrix(
            (np.ones((target_size,), dtype=np.float64), (rows, rows)),
            shape=(target_size, source_size),
        )

    p = max(1, int(stencil_half_width))
    half_turn = n_alpha_src // 2
    if 2 * half_turn != n_alpha_src:
        raise ValueError("Directional alpha grid size must be even for FaSTMM2-style shifts.")

    theta_ext = np.concatenate((beta_src + np.pi, beta_src, beta_src - np.pi), dtype=float)
    phi_ext = np.concatenate(
        (alpha_src - 2.0 * np.pi, alpha_src, alpha_src + 2.0 * np.pi), dtype=float
    )

    theta_rows: list[int] = []
    theta_cols: list[int] = []
    theta_data: list[float] = []
    for ia_src in range(n_alpha_src):
        phi_1b = ia_src + 1
        for ib_tgt in range(n_beta_dst):
            theta_target = float(beta_dst[ib_tgt])
            tt = n_beta_src + 1
            for t in range(1, n_beta_src + 1):
                if theta_target > float(beta_src[t - 1]):
                    tt = t
                    break
            t = tt - 1
            row = ia_src * n_beta_dst + ib_tgt
            i_min = t - p + 1
            i_max = t + p
            for i1 in range(i_min, i_max + 1):
                weight = 1.0
                x_i1 = float(theta_ext[(i1 + n_beta_src) - 1])
                for i2 in range(i_min, i_max + 1):
                    if i2 == i1:
                        continue
                    x_i2 = float(theta_ext[(i2 + n_beta_src) - 1])
                    weight *= (theta_target - x_i2) / (x_i1 - x_i2)
                theta_index = i1
                phi_index = phi_1b
                sign = 1.0
                if theta_index > n_beta_src:
                    theta_index = 2 * n_beta_src - theta_index + 1
                    phi_index += half_turn
                    if phi_index > n_alpha_src:
                        phi_index -= n_alpha_src
                    sign = -1.0
                if theta_index < 1:
                    theta_index = 1 - theta_index
                    phi_index += half_turn
                    if phi_index > n_alpha_src:
                        phi_index -= n_alpha_src
                    sign = -1.0
                col = (phi_index - 1) * n_beta_src + (theta_index - 1)
                theta_rows.append(int(row))
                theta_cols.append(int(col))
                theta_data.append(float(sign * weight))

    theta_matrix = scipy.sparse.coo_matrix(
        (
            np.asarray(theta_data, dtype=np.float64),
            (np.asarray(theta_rows, dtype=np.int64), np.asarray(theta_cols, dtype=np.int64)),
        ),
        shape=(n_alpha_src * n_beta_dst, n_alpha_src * n_beta_src),
    ).tocsr()
    theta_matrix.sum_duplicates()

    phi_rows: list[int] = []
    phi_cols: list[int] = []
    phi_data: list[float] = []
    for ia_tgt in range(n_alpha_dst):
        phi_target = float(alpha_dst[ia_tgt])
        ss = n_alpha_src + 1
        for s in range(1, n_alpha_src + 1):
            if phi_target < float(alpha_src[s - 1]):
                ss = s
                break
        s = ss - 1
        i_min = s - p + 1
        i_max = s + p
        weights: list[tuple[int, float]] = []
        for i1 in range(i_min, i_max + 1):
            weight = 1.0
            x_i1 = float(phi_ext[(i1 + n_alpha_src) - 1])
            for i2 in range(i_min, i_max + 1):
                if i2 == i1:
                    continue
                x_i2 = float(phi_ext[(i2 + n_alpha_src) - 1])
                weight *= (phi_target - x_i2) / (x_i1 - x_i2)
            phi_index = i1
            if phi_index > n_alpha_src:
                phi_index -= n_alpha_src
            if phi_index < 1:
                phi_index += n_alpha_src
            weights.append((int(phi_index - 1), float(weight)))
        for ib_tgt in range(n_beta_dst):
            row = ia_tgt * n_beta_dst + ib_tgt
            for ia_src, weight in weights:
                col = ia_src * n_beta_dst + ib_tgt
                phi_rows.append(int(row))
                phi_cols.append(int(col))
                phi_data.append(float(weight))

    phi_matrix = scipy.sparse.coo_matrix(
        (
            np.asarray(phi_data, dtype=np.float64),
            (np.asarray(phi_rows, dtype=np.int64), np.asarray(phi_cols, dtype=np.int64)),
        ),
        shape=(n_alpha_dst * n_beta_dst, n_alpha_src * n_beta_dst),
    ).tocsr()
    phi_matrix.sum_duplicates()

    matrix = (phi_matrix @ theta_matrix).tocsr()
    matrix.sum_duplicates()
    return matrix


@cache
def _cached_directional_interpolation(
    source_order: int,
    target_order: int,
    alpha_factor_key: int,
    beta_factor_key: int,
) -> MLFMMDirectionalInterpolation:
    source_grid = directional_grid(
        int(source_order),
        alpha_factor=_from_cache_key_factor(alpha_factor_key),
        beta_factor=_from_cache_key_factor(beta_factor_key),
    )
    target_grid = directional_grid(
        int(target_order),
        alpha_factor=_from_cache_key_factor(alpha_factor_key),
        beta_factor=_from_cache_key_factor(beta_factor_key),
    )
    matrix = _build_sparse_directional_interpolation(
        source_alpha=source_grid.alpha,
        source_beta=source_grid.beta,
        target_alpha=target_grid.alpha,
        target_beta=target_grid.beta,
    )
    return MLFMMDirectionalInterpolation(
        source_order=int(source_order),
        target_order=int(target_order),
        matrix=matrix.astype(np.complex128),
    )


def directional_interpolation(
    source_order: int,
    target_order: int,
    *,
    alpha_factor: float = 1.0,
    beta_factor: float = 1.0,
) -> MLFMMDirectionalInterpolation:
    """Return a cached interpolation operator between two directional grids."""

    return _cached_directional_interpolation(
        int(source_order),
        int(target_order),
        _cache_key_factor(alpha_factor),
        _cache_key_factor(beta_factor),
    )


def directional_anterpolation(
    source_order: int,
    target_order: int,
    *,
    alpha_factor: float = 1.0,
    beta_factor: float = 1.0,
) -> MLFMMDirectionalInterpolation:
    """Return the transpose back-projection paired with `directional_interpolation`."""

    interpolation = directional_interpolation(
        source_order,
        target_order,
        alpha_factor=alpha_factor,
        beta_factor=beta_factor,
    )
    return MLFMMDirectionalInterpolation(
        source_order=int(target_order),
        target_order=int(source_order),
        matrix=interpolation.matrix.T.tocsr().astype(np.complex128),
    )


__all__ = [
    "MLFMMDirectionalGrid",
    "MLFMMDirectionalInterpolation",
    "MLFMMDirectionalTransforms",
    "apply_directional_reflection",
    "box_outgoing_to_directional",
    "directional_anterpolation",
    "directional_grid",
    "directional_interpolation",
    "directional_to_box_regular",
    "directional_transforms",
]
