from __future__ import annotations

"""Directional sampled-basis helpers for pyceles MLFMM."""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from pyceles.core.indexing import n_scalar, scalar_index

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
    """SVWF <-> sampled-direction transform operators for one shared grid."""

    box_order: int
    grid: MLFMMDirectionalGrid
    Fth: Array
    Fph: Array
    Gth: Array
    Gph: Array
    Fth_adj: Array
    Fph_adj: Array
    Gth_adj: Array
    Gph_adj: Array
    sampled_matrix: Array
    sampled_pinv: Array


@dataclass(frozen=True)
class MLFMMDirectionalInterpolation:
    """Interpolation matrix between two sampled directional grids."""

    source_order: int
    target_order: int
    matrix: Array


def _cache_key_factor(value: float) -> int:
    return int(round(float(value) * 1_000_000))


def _from_cache_key_factor(value: int) -> float:
    return float(value) / 1_000_000.0


def _legendre2(n: int, x: float) -> Array:
    """Return unnormalized associated Legendre values `P_n^m(x)` for one degree."""

    n_i = int(n)
    if n_i < 0:
        raise ValueError(f"n must be >= 0. Got {n}.")
    pm = np.zeros((n_i + 2, n_i + 2), dtype=np.float64)
    pm[0, 0] = 1.0

    x_f = float(x)
    ls = 1.0 if abs(x_f) <= 1.0 else -1.0
    xq = math.sqrt(max(0.0, ls * (1.0 - x_f * x_f)))

    for m in range(1, n_i + 1):
        pm[m, m] = -ls * (2.0 * m - 1.0) * xq * pm[m - 1, m - 1]

    for m in range(0, n_i + 1):
        pm[m, m + 1] = (2.0 * m + 1.0) * x_f * pm[m, m]

    for m in range(0, n_i + 1):
        for ell in range(m + 2, n_i + 1):
            pm[m, ell] = (
                (2.0 * ell - 1.0) * x_f * pm[m, ell - 1] - (m + ell - 1.0) * pm[m, ell - 2]
            ) / float(ell - m)

    return pm[: n_i + 1, n_i].copy()


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
    beta = np.arccos(np.clip(cos_beta[::-1], -1.0, 1.0)).astype(float, copy=False)
    beta_weights = np.asarray(w_beta[::-1], dtype=float)
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
    gth = np.zeros_like(fth)
    gph = np.zeros_like(fth)

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
            l2_arr = legendre_by_n[l - 1] if l > 1 else np.asarray([1.0], dtype=float)
            q = math.sqrt(l * (l + 1.0)) / ((2.0 * l + 1.0) * sin_theta)
            for m in range(-l, l + 1):
                mm = abs(m)
                if m not in exp_cache:
                    exp_cache[m] = complex(np.exp(1j * m * phi))
                phase = exp_cache[m]
                log_norm = (
                    math.log((2.0 * l + 1.0) / (4.0 * np.pi))
                    + math.lgamma(l - mm + 1.0)
                    - math.lgamma(l + mm + 1.0)
                )
                cc = (1j**l) * math.exp(0.5 * log_norm)
                y = complex(l_arr[mm] * phase)
                y1 = complex(l1_arr[mm] * phase)
                y2 = 0.0j if mm == l else complex(l2_arr[mm] * phase)
                y1 *= (l - mm + 1.0) / (l + 1.0)
                y2 *= (l + mm) / l
                b_theta = (y1 - y2) * q
                b_phi = (1j * m * (2.0 * l + 1.0) / (l * (l + 1.0)) * y) * q
                c_theta = b_phi
                c_phi = -b_theta
                idx = scalar_index(l, m)
                fth[idir, idx] = 1j * cc * b_theta
                fph[idir, idx] = 1j * cc * b_phi
                gth[idir, idx] = -cc * c_theta
                gph[idir, idx] = -cc * c_phi

    reflection = np.asarray(grid.reflection_permutation, dtype=np.int64)
    fth_phys = np.asarray(fth[reflection], dtype=np.complex128)
    fph_phys = np.asarray(fph[reflection], dtype=np.complex128)
    zero = np.zeros_like(fth_phys)
    sampled_matrix = np.block(
        [
            [fth_phys, zero],
            [fph_phys, zero],
            [zero, fth_phys],
            [zero, fph_phys],
        ]
    )
    sampled_pinv = np.linalg.pinv(sampled_matrix, rcond=1.0e-12)

    return MLFMMDirectionalTransforms(
        box_order=int(box_order),
        grid=grid,
        Fth=fth,
        Fph=fph,
        Gth=gth,
        Gph=gph,
        Fth_adj=np.conjugate(fth.T),
        Fph_adj=np.conjugate(fph.T),
        Gth_adj=np.conjugate(gth.T),
        Gph_adj=np.conjugate(gph.T),
        sampled_matrix=np.asarray(sampled_matrix, dtype=np.complex128),
        sampled_pinv=np.asarray(sampled_pinv, dtype=np.complex128),
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
    top = (
        transforms.Fth_adj @ a_theta_arr
        + transforms.Fph_adj @ a_phi_arr
        + transforms.Gth_adj @ b_theta_arr
        + transforms.Gph_adj @ b_phi_arr
    )
    bottom = (
        transforms.Fth_adj @ b_theta_arr
        + transforms.Fph_adj @ b_phi_arr
        + transforms.Gth_adj @ a_theta_arr
        + transforms.Gph_adj @ a_phi_arr
    )
    return np.concatenate((top, bottom)).astype(np.complex128, copy=False)


def _periodic_linear_interpolation_matrix(source_alpha: Array, target_alpha: Array) -> Array:
    """Return a periodic linear interpolation matrix for the uniform azimuth grid."""

    source = np.asarray(source_alpha, dtype=float).reshape(-1)
    target = np.asarray(target_alpha, dtype=float).reshape(-1)
    n_source = int(source.size)
    step = 2.0 * np.pi / float(n_source)
    base = float(source[0])
    matrix = np.zeros((target.size, n_source), dtype=np.float64)

    for i, value in enumerate(target):
        scaled = ((float(value) - base) / step) % n_source
        left = int(np.floor(scaled)) % n_source
        frac = float(scaled - np.floor(scaled))
        matrix[i, left] += 1.0 - frac
        matrix[i, (left + 1) % n_source] += frac
    return matrix


def _linear_interpolation_matrix(source_nodes: Array, target_nodes: Array) -> Array:
    """Return a piecewise-linear interpolation matrix on a monotone node set."""

    source = np.asarray(source_nodes, dtype=float).reshape(-1)
    target = np.asarray(target_nodes, dtype=float).reshape(-1)
    matrix = np.zeros((target.size, source.size), dtype=np.float64)

    for i, value in enumerate(target):
        x = float(value)
        if x <= float(source[0]):
            matrix[i, 0] = 1.0
            continue
        if x >= float(source[-1]):
            matrix[i, -1] = 1.0
            continue
        right = int(np.searchsorted(source, x, side="right"))
        left = right - 1
        x0 = float(source[left])
        x1 = float(source[right])
        frac = (x - x0) / (x1 - x0)
        matrix[i, left] = 1.0 - frac
        matrix[i, right] = frac
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
    alpha_matrix = _periodic_linear_interpolation_matrix(source_grid.alpha, target_grid.alpha)
    beta_matrix = _linear_interpolation_matrix(source_grid.beta, target_grid.beta)
    matrix = np.kron(alpha_matrix, beta_matrix).astype(np.complex128, copy=False)
    return MLFMMDirectionalInterpolation(
        source_order=int(source_order),
        target_order=int(target_order),
        matrix=matrix,
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
        matrix=np.asarray(interpolation.matrix.T, dtype=np.complex128),
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
