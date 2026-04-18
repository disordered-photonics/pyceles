"""Dense-assembly helpers built on the prepared-operator boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles.core.indexing import n_modes
from pyceles.core.translation import translation_block

from .base import PreparedOperator
from .coupling_pairwise import require_pairwise_coupling


def estimate_translation_cache_bytes(
    N: int, lmax: int, *, dtype: npt.DTypeLike = np.complex128
) -> int:
    """Estimate memory of storing all pair blocks W_ij (i!=j)."""
    N = int(N)
    lmax = int(lmax)
    if N < 0:
        raise ValueError("N must be non-negative")
    nm = n_modes(lmax)
    pairs = N * (N - 1)
    block_entries = nm * nm
    return pairs * block_entries * np.dtype(dtype).itemsize


def make_prepared_A_and_rhs(
    prepared: PreparedOperator, b: np.ndarray
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Return `(A_mv, rhs)` from a prepared system and incident coefficients."""
    rhs = prepared.rhs_Tb(b)

    def A_mv(x: np.ndarray) -> np.ndarray:
        return prepared.apply_A(x)

    return A_mv, rhs


def assemble_dense_A_numpy(
    prepared: PreparedOperator,
    *,
    show_progress: bool = False,
    use_cache: bool = False,
    store_blocks: bool = False,
) -> np.ndarray:
    """Assemble dense A = I - T W from a prepared system."""
    ns = prepared.positions.shape[0]
    nm = n_modes(prepared.lmax)
    n = ns * nm
    A = np.zeros((n, n), dtype=prepared.dtype)
    A[np.arange(n), np.arange(n)] = 1.0 + 0.0j

    pair_iter: Iterable[tuple[int, int]] = ((i, j) for i in range(ns) for j in range(ns) if i != j)
    if show_progress:
        pair_iter = tqdm(pair_iter, total=ns * (ns - 1), desc="Assemble A (blockwise)")

    pairwise = require_pairwise_coupling(prepared.coupling)
    cache = pairwise._W_cache if use_cache and pairwise.cache_translation_blocks else None
    for i, j in pair_iter:
        key = (i, j)
        wij = cache.get(key) if cache is not None else None
        if wij is None:
            rvec = prepared.positions[i] - prepared.positions[j]
            wij = translation_block(
                prepared.lmax,
                prepared.k,
                rvec,
                ab5=pairwise.ab5,
                radial_lut=pairwise.radial_lut,
            )
            if store_blocks and pairwise.cache_translation_blocks:
                pairwise._W_cache[key] = wij

        blk = -prepared.apply_particle_block(i, wij)
        rs = slice(i * nm, (i + 1) * nm)
        cs = slice(j * nm, (j + 1) * nm)
        A[rs, cs] = blk

    return A


__all__ = ["assemble_dense_A_numpy", "estimate_translation_cache_bytes", "make_prepared_A_and_rhs"]
