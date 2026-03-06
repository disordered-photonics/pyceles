"""SVWF rotation helpers in CELES mode ordering.

Axisymmetric particles are prepared in a body-fixed frame and then rotated into
the lab frame. In CELES ordering, the SVWF rotation operator is diagonal in
polarization and block-diagonal in degree `l`, with dense coupling only across
azimuthal orders `m=-l..l` inside each degree block.

This module keeps two entry points:
- `svwf_rotation_matrix(...)` assembles the full dense operator for inspection,
  tests, and external use.
- `rotate_svwf_tmatrix_block(...)` applies the same rotation blockwise, which
  avoids materializing the full sparse-by-structure operator in the common
  particle-local T-matrix path.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .indexing import n_scalar
from .wigner import wigner_D

Array = np.ndarray


def _scalar_rotation_block(l: int, alpha: float, beta: float, gamma: float) -> Array:
    """Return the scalar `(2l+1, 2l+1)` rotation block for one degree `l`."""

    m_vals = np.arange(-l, l + 1, dtype=np.int64)
    block = np.empty((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for row, m in enumerate(m_vals):
        for col, m_prime in enumerate(m_vals):
            block[row, col] = wigner_D(l, int(m), int(m_prime), alpha, beta, gamma)
    return block


@lru_cache(maxsize=2048)
def _scalar_rotation_block_cached(l: int, alpha: float, beta: float, gamma: float) -> Array:
    """Cached scalar rotation block for one degree and Euler-angle triplet.

    The key space includes continuous Euler angles, so this cache is bounded on
    purpose. A moderate size still helps when many particles reuse a discrete
    orientation set, while avoiding unbounded growth for large disordered
    clusters with effectively unique angles.
    """

    return _scalar_rotation_block(int(l), float(alpha), float(beta), float(gamma))


def _degree_block_bounds(lmax: int) -> list[tuple[int, int]]:
    """Return CELES scalar-block bounds for degrees `l=1..lmax`."""

    lmax = int(lmax)
    return [((l - 1) * (l + 1), (l - 1) * (l + 1) + (2 * l + 1)) for l in range(1, lmax + 1)]


def svwf_rotation_matrix(lmax: int, alpha: float, beta: float, gamma: float) -> Array:
    """Return the CELES-ordered SVWF rotation matrix for one Euler-angle triplet.

    This dense matrix is kept mainly as a reference/inspection helper. The
    production particle-T path uses `rotate_svwf_tmatrix_block(...)` directly,
    so caching the full dense matrix is not important enough to justify keeping
    another angle-keyed cache alive.
    """

    lmax = int(lmax)
    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    Ns = n_scalar(lmax)
    Nm = 2 * Ns
    R = np.zeros((Nm, Nm), dtype=np.complex128)

    for start, stop in _degree_block_bounds(lmax):
        block = _scalar_rotation_block_cached(
            l=(stop - start - 1) // 2, alpha=alpha, beta=beta, gamma=gamma
        )
        R[start:stop, start:stop] = block
        R[Ns + start : Ns + stop, Ns + start : Ns + stop] = block

    return R


def rotate_svwf_tmatrix_block(
    T: Array, lmax: int, euler_angles: tuple[float, float, float]
) -> Array:
    """Rotate one particle-local spherical-basis `T` block into the lab frame.

    The rotated block is `D(-gamma,-beta,-alpha)^T @ T @ D(alpha,beta,gamma)^T`
    in the CELES/SMUTHI convention. We apply the left and right block-diagonal
    factors degree-by-degree instead of forming dense `(Nm, Nm)` rotation
    matrices, which keeps this one-time particle-local preparation step lighter
    on memory.

    Repeated or intentionally quantized orientations reduce preparation cost,
    not iterative solver cost: once the prepared operator stores the rotated
    particle `T` block, the cluster matvec reuses that block directly.
    """

    alpha, beta, gamma = (float(v) for v in euler_angles)
    if (alpha, beta, gamma) == (0.0, 0.0, 0.0):
        return np.asarray(T, dtype=np.complex128)

    T_arr = np.asarray(T, dtype=np.complex128)
    lmax = int(lmax)
    Ns = n_scalar(lmax)
    Nm = 2 * Ns
    if T_arr.shape != (Nm, Nm):
        raise ValueError(f"T must have shape {(Nm, Nm)} for lmax={lmax}, got {T_arr.shape}.")

    degree_bounds = _degree_block_bounds(lmax)
    rotated_cols = np.empty_like(T_arr)
    for tau_offset in (0, Ns):
        for l, (start, stop) in enumerate(degree_bounds, start=1):
            block = _scalar_rotation_block_cached(l=l, alpha=alpha, beta=beta, gamma=gamma)
            col_slice = slice(tau_offset + start, tau_offset + stop)
            rotated_cols[:, col_slice] = T_arr[:, col_slice] @ block.T

    rotated = np.empty_like(T_arr)
    for tau_offset in (0, Ns):
        for l, (start, stop) in enumerate(degree_bounds, start=1):
            block = _scalar_rotation_block_cached(l=l, alpha=-gamma, beta=-beta, gamma=-alpha)
            row_slice = slice(tau_offset + start, tau_offset + stop)
            rotated[row_slice, :] = block.T @ rotated_cols[row_slice, :]

    return rotated
