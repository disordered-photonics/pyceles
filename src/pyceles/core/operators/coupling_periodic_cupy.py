"""CuPy periodic coupling operator for rectangular two-dimensional lattices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.ewald import (
    ewald_self_correction,
    resolve_ewald_eta,
    resolve_ewald_shell_counts,
)
from pyceles.core.periodic.ewald_cupy import (
    CupyEwaldShellWorkspace,
    ewald_structural_sums_2d_fixed_cupy,
)
from pyceles.core.periodic.scalar import structural_sum_m_normalization
from pyceles.core.periodic.structural import translation_contraction_tensor

Array = np.ndarray


@dataclass
class CuPyPeriodicCouplingOperator:
    """GPU-backed Bloch-reduced periodic coupling descriptor.

    The current implementation keeps the Ewald structural-sum evaluation in
    vectorized CuPy operations with fixed shell counts resolved once from the
    periodic options.  It supports both matrix-free source-batched matvecs and
    optional explicit dense block caching for workflows where the memory cost is
    acceptable.
    """

    lmax: int
    k: float
    positions: Array
    ab5: Array
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype
    cache_blocks: bool = False
    _positions_gpu: Any | None = field(default=None, init=False, repr=False)
    _workspace: CupyEwaldShellWorkspace | None = field(default=None, init=False, repr=False)
    _resolved_ewald_eta: float | None = field(default=None, init=False, repr=False)
    _resolved_shell_counts: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _contraction_tensor_gpu: Any | None = field(default=None, init=False, repr=False)
    _source_block_cache: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy periodic coupling supports only complex64 and complex128. "
                f"Got {self.dtype!r}."
            )
        if self.periodic.options.method != "ewald":
            raise NotImplementedError("CuPy periodic coupling currently supports only Ewald sums.")

    @property
    def n_particles(self) -> int:
        return int(np.asarray(self.positions).reshape(-1, 3).shape[0])

    @property
    def n_modes(self) -> int:
        return n_modes(int(self.lmax))

    def _cupy(self) -> Any:
        cupy, _ = import_cupy()
        return cupy

    def _positions_device(self) -> Any:
        cp = self._cupy()
        if self._positions_gpu is None:
            self._positions_gpu = cp.asarray(
                np.asarray(self.positions, dtype=np.float64).reshape(-1, 3),
                dtype=cp.float64,
            )
        return self._positions_gpu

    def _ewald_eta(self) -> float:
        eta = self.periodic.options.eta
        if eta is not None:
            return float(eta)
        if self._resolved_ewald_eta is None:
            self._resolved_ewald_eta = resolve_ewald_eta(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=float).reshape(2),
                positions=np.asarray(self.positions, dtype=float).reshape(-1, 3),
                lmax=int(self.lmax),
            )
        return float(self._resolved_ewald_eta)

    def _shell_counts(self) -> tuple[int, int]:
        if self._resolved_shell_counts is None:
            counts = resolve_ewald_shell_counts(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=float).reshape(2),
                positions=np.asarray(self.positions, dtype=float).reshape(-1, 3),
                lmax=int(self.lmax),
                eta=self._ewald_eta(),
            )
            self._resolved_shell_counts = (int(counts.real_shells), int(counts.reciprocal_shells))
        return self._resolved_shell_counts

    def _workspace_device(self) -> CupyEwaldShellWorkspace:
        if self._workspace is None:
            self._workspace = CupyEwaldShellWorkspace(
                cupy=self._cupy(),
                lattice=self.periodic.lattice,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=np.float64).reshape(2),
                eta=self._ewald_eta(),
            )
        return self._workspace

    def _contraction_tensor_device(self) -> Any:
        cp = self._cupy()
        if self._contraction_tensor_gpu is None:
            tensor_np = translation_contraction_tensor(
                lmax=int(self.lmax),
                ab5=np.asarray(self.ab5),
                dtype=self.dtype,
            )
            self._contraction_tensor_gpu = cp.asarray(tensor_np, dtype=self.dtype)
        return self._contraction_tensor_gpu

    def _add_self_correction(self, sums: Any, *, local_destination_index: int) -> None:
        idx = int(local_destination_index)
        if idx < 0 or idx >= int(sums.shape[0]):
            return
        order = int(sums.shape[1] - 1)
        cp = self._cupy()
        correction = ewald_self_correction(float(self.k), self._ewald_eta())
        value = structural_sum_m_normalization(0) * correction
        sums[idx, 0, order] = sums[idx, 0, order] + cp.asarray(value, dtype=cp.complex128)

    def _structural_sums_for_source(self, source_index: int) -> Any:
        cp = self._cupy()
        pos = self._positions_device()
        src = pos[int(source_index)]
        rel = src[None, :] - pos
        real_count, reciprocal_count = self._shell_counts()
        sums = ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=rel,
            lmax_struct=int(self.lmax),
            k=float(self.k),
            eta=self._ewald_eta(),
            workspace=self._workspace_device(),
            real_shell_count=int(real_count),
            reciprocal_shell_count=int(reciprocal_count),
            coordinate_scale=float(np.max(np.abs(np.asarray(self.positions, dtype=float)))),
        )
        self._add_self_correction(sums, local_destination_index=int(source_index))
        return cp.asarray(sums, dtype=cp.complex128)

    def _blocks_for_source(self, source_index: int) -> Any:
        cached = self._source_block_cache.get(int(source_index))
        if cached is not None:
            return cached
        cp = self._cupy()
        sums = self._structural_sums_for_source(int(source_index))
        tensor = self._contraction_tensor_device()
        blocks = cp.einsum(
            "dpm,ijpm->dij",
            sums.astype(self.dtype, copy=False),
            tensor,
            optimize=True,
        ).astype(self.dtype, copy=False)
        if self.cache_blocks:
            self._source_block_cache[int(source_index)] = blocks
        return blocks

    def populate(self, *, show_progress: bool = False) -> None:
        """Eagerly populate the optional device-side dense block cache."""
        if not self.cache_blocks:
            return
        sources: Iterable[int] = range(self.n_particles)
        if show_progress:
            sources = tqdm(sources, total=self.n_particles, desc="Populate periodic W cache [cupy]")
        for source_index in sources:
            self._blocks_for_source(int(source_index))

    def _apply_gpu(self, x: Array | object) -> Any:
        cp = self._cupy()
        arr_raw = coerce_array(x, dtype=self.dtype, prefer_cupy=True)
        squeezed = False
        if int(arr_raw.ndim) == 1:
            if int(arr_raw.size) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input length must match n_particles * n_modes. "
                    f"Got {int(arr_raw.size)} for {self.n_particles * self.n_modes}."
                )
            arr = cp.asarray(arr_raw, dtype=self.dtype).reshape(self.n_particles, self.n_modes, 1)
            squeezed = True
        elif int(arr_raw.ndim) == 2:
            if int(arr_raw.shape[0]) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input first dimension must match n_particles * n_modes. "
                    f"Got {int(arr_raw.shape[0])} for {self.n_particles * self.n_modes}."
                )
            arr = cp.asarray(arr_raw, dtype=self.dtype).reshape(
                self.n_particles,
                self.n_modes,
                int(arr_raw.shape[1]),
            )
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(arr_raw.shape)}.")

        y = cp.zeros_like(arr, dtype=self.dtype)
        tensor = self._contraction_tensor_device()
        for source_index in range(self.n_particles):
            src = arr[source_index]
            if self.cache_blocks:
                blocks = self._blocks_for_source(source_index)
                y += cp.einsum("dij,jr->dir", blocks, src, optimize=True)
                continue
            sums = self._structural_sums_for_source(source_index)
            y += cp.einsum(
                "dpm,ijpm,jr->dir",
                sums.astype(self.dtype, copy=False),
                tensor,
                src,
                optimize=True,
            )
        if squeezed:
            return y.reshape(self.n_particles * self.n_modes)
        return y.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))

    def apply(self, x: Array | object) -> Array | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPeriodicCouplingOperator"]
