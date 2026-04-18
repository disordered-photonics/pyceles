"""Pairwise free-space coupling backend and low-level pairwise kernels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles.core.indexing import n_modes
from pyceles.core.translation import RadialLUT, translation_ab5_table, translation_block

from .base import CouplingOperator

COMPLEX128_DTYPE = np.dtype(np.complex128)


@dataclass
class PairwiseCouplingOperator:
    """Prepared pairwise free-space coupling operator `W`."""

    lmax: int
    k: float
    positions: np.ndarray
    ab5: np.ndarray
    radial_lut: RadialLUT
    dtype: np.dtype = COMPLEX128_DTYPE
    cache_translation_blocks: bool = False
    _W_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def apply_W(self, x: np.ndarray) -> np.ndarray:
        return apply_W_numpy(
            self.lmax,
            self.k,
            self.positions,
            x,
            self.ab5,
            dtype=self.dtype,
            radial_lut=self.radial_lut,
            block_cache=self._W_cache if self.cache_translation_blocks else None,
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.apply_W(x)

    def populate(self, *, show_progress: bool = False) -> None:
        if not self.cache_translation_blocks:
            return
        ns = self.positions.shape[0]
        pair_iter: Iterable[tuple[int, int]] = (
            (i, j) for i in range(ns) for j in range(ns) if i != j
        )
        if show_progress:
            pair_iter = tqdm(pair_iter, total=ns * (ns - 1), desc="Populate W cache")
        for i, j in pair_iter:
            key = (i, j)
            if key in self._W_cache:
                continue
            rvec = self.positions[i] - self.positions[j]
            self._W_cache[key] = translation_block(
                self.lmax,
                self.k,
                rvec,
                ab5=self.ab5,
                radial_lut=self.radial_lut,
            )


def require_pairwise_coupling(coupling: CouplingOperator) -> PairwiseCouplingOperator:
    """Return the free-space pairwise coupling backend or raise a clear error."""
    if not isinstance(coupling, PairwiseCouplingOperator):
        raise TypeError(
            "This path currently requires the PairwiseCouplingOperator backend. "
            f"Got {type(coupling).__name__}."
        )
    return coupling


def apply_W_numpy(
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    ab5: np.ndarray,
    *,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: RadialLUT | None,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute y = W x (NumPy), excluding self-interaction."""
    out_dtype = np.dtype(dtype)
    ns = positions.shape[0]
    nm = n_modes(lmax)
    arr = np.asarray(x, dtype=out_dtype).reshape(ns, nm)
    y = np.zeros_like(arr, dtype=out_dtype)

    for i in range(ns):
        for j in range(ns):
            if i == j:
                continue
            key = (i, j)
            wij = block_cache.get(key) if block_cache is not None else None
            if wij is None:
                rvec = positions[i] - positions[j]
                wij = translation_block(lmax, k, rvec, ab5=ab5, radial_lut=radial_lut)
                if block_cache is not None:
                    block_cache[key] = wij
            y[i] += wij @ arr[j]

    return y.reshape(ns * nm)


def apply_A_numpy(
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    *,
    T_M: np.ndarray,
    T_N: np.ndarray,
    T_diag: np.ndarray | None = None,
    ab5: np.ndarray | None = None,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: RadialLUT | None,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute y = (I - T W) x with precomputed diagonal T entries."""
    from .prepare import build_T_mode_diagonal

    out_dtype = np.dtype(dtype)
    ns = positions.shape[0]
    nm = n_modes(lmax)
    if ab5 is None:
        ab5 = translation_ab5_table(lmax)

    wx = apply_W_numpy(
        lmax,
        k,
        positions,
        x,
        ab5,
        dtype=out_dtype,
        radial_lut=radial_lut,
        block_cache=block_cache,
    ).reshape(ns, nm)

    y = np.asarray(x, dtype=out_dtype).reshape(ns, nm).copy()
    diag = (
        build_T_mode_diagonal(lmax, T_M, T_N)
        if T_diag is None
        else np.asarray(T_diag, dtype=out_dtype)
    )
    y -= diag * wx
    return y.reshape(ns * nm)


__all__ = [
    "PairwiseCouplingOperator",
    "apply_A_numpy",
    "apply_W_numpy",
    "require_pairwise_coupling",
]
