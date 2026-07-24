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
from pyceles.core.periodic.scalar import same_plane_z_tolerance, structural_sum_m_normalization
from pyceles.core.periodic.structural import translation_contraction_tensor

from .base import SourceBlockBatch

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
    _coordinate_scale: float | None = field(default=None, init=False, repr=False)
    _contraction_tensor_gpu: Any | None = field(default=None, init=False, repr=False)
    _self_correction_gpu: Any | None = field(default=None, init=False, repr=False)
    _dense_w_cache_gpu: Any | None = field(default=None, init=False, repr=False)
    _source_index_gpu_cache: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _same_plane_index_gpu_cache: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )

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

    def _coordinate_scale_value(self) -> float:
        if self._coordinate_scale is None:
            self._coordinate_scale = float(np.max(np.abs(np.asarray(self.positions, dtype=float))))
        return float(self._coordinate_scale)

    def _same_plane_pair_indices_device(self, source_indices: tuple[int, ...]) -> Any:
        """Return flat pair indices needing the exact same-plane reciprocal formula."""
        key = tuple(int(i) for i in source_indices)
        cached = self._same_plane_index_gpu_cache.get(key)
        if cached is not None:
            return cached
        cp = self._cupy()
        if len(key) == 0:
            out = cp.zeros((0,), dtype=cp.int64)
            self._same_plane_index_gpu_cache[key] = out
            return out
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        src_z = pos[np.asarray(key, dtype=np.int64), 2]
        atol = same_plane_z_tolerance(
            float(self.k), coordinate_scale=self._coordinate_scale_value()
        )
        mask = np.abs(src_z[:, None] - pos[None, :, 2]) <= float(atol)
        out = cp.asarray(np.flatnonzero(mask).astype(np.int64, copy=False), dtype=cp.int64)
        self._same_plane_index_gpu_cache[key] = out
        return out

    def _self_correction_device(self) -> Any:
        cp = self._cupy()
        if self._self_correction_gpu is None:
            correction = ewald_self_correction(float(self.k), self._ewald_eta())
            value = structural_sum_m_normalization(0) * correction
            self._self_correction_gpu = cp.asarray(value, dtype=cp.complex128)
        return self._self_correction_gpu

    def _source_batch_size(self) -> int:
        """Return an internal source chunk size bounded by temporary device memory."""
        if self.cache_blocks:
            bytes_per_source = self.n_particles * self.n_modes * self.n_modes * self.dtype.itemsize
            target_bytes = 256 * 1024**2
            max_sources = 16
        else:
            order = 2 * int(self.lmax)
            bytes_per_source = (
                self.n_particles * (order + 1) * (2 * order + 1) * np.dtype(np.complex128).itemsize
            )
            target_bytes = 256 * 1024**2
            # Cache-off matvecs retain no pair data between calls, so let the
            # temporary-memory budget set the launch size rather than imposing
            # an additional fixed source-count cap.
            max_sources = self.n_particles
        return max(1, min(self.n_particles, max_sources, target_bytes // max(bytes_per_source, 1)))

    def _source_batches(self) -> Iterable[tuple[int, ...]]:
        batch_size = int(self._source_batch_size())
        for start in range(0, self.n_particles, batch_size):
            stop = min(self.n_particles, start + batch_size)
            yield tuple(range(start, stop))

    def _source_index_device(self, source_indices: tuple[int, ...]) -> Any:
        key = tuple(int(i) for i in source_indices)
        cached = self._source_index_gpu_cache.get(key)
        if cached is not None:
            return cached
        cp = self._cupy()
        out = cp.asarray(key, dtype=cp.int64)
        self._source_index_gpu_cache[key] = out
        return out

    def _add_self_corrections(self, sums: Any, *, source_indices: tuple[int, ...]) -> None:
        if len(source_indices) == 0:
            return
        order = int(sums.shape[2] - 1)
        cp = self._cupy()
        rows = cp.arange(len(source_indices), dtype=cp.int64)
        cols = cp.asarray(source_indices, dtype=cp.int64)
        sums[rows, cols, 0, order] = sums[rows, cols, 0, order] + self._self_correction_device()

    def _structural_sums_for_sources(self, source_indices: tuple[int, ...]) -> Any:
        key = tuple(int(i) for i in source_indices)
        if len(key) == 0:
            cp = self._cupy()
            order = 2 * int(self.lmax)
            return cp.zeros((0, self.n_particles, order + 1, 2 * order + 1), dtype=cp.complex128)
        cp = self._cupy()
        pos = self._positions_device()
        src = pos[self._source_index_device(key)]
        rel = src[:, None, :] - pos[None, :, :]
        real_count, reciprocal_count = self._shell_counts()
        sums = ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=rel.reshape(-1, 3),
            lmax_struct=int(self.lmax),
            workspace=self._workspace_device(),
            real_shell_count=int(real_count),
            reciprocal_shell_count=int(reciprocal_count),
            coordinate_scale=self._coordinate_scale_value(),
            same_plane_pair_indices=self._same_plane_pair_indices_device(key),
        )
        sums = cp.asarray(sums, dtype=cp.complex128).reshape(
            len(key),
            self.n_particles,
            2 * int(self.lmax) + 1,
            4 * int(self.lmax) + 1,
        )
        self._add_self_corrections(sums, source_indices=key)
        return sums

    def _compute_blocks_for_sources(self, source_indices: tuple[int, ...]) -> Any:
        key = tuple(int(i) for i in source_indices)
        cp = self._cupy()
        sums = self._structural_sums_for_sources(key)
        tensor = self._contraction_tensor_device()
        return cp.einsum(
            "sdpm,ijpm->sdij",
            sums.astype(self.dtype, copy=False),
            tensor,
            optimize=True,
        ).astype(self.dtype, copy=False)

    def _dense_blocks_for_sources(self, source_indices: tuple[int, ...]) -> Any:
        """Recover source-major blocks from the flattened dense cache."""
        key = tuple(int(i) for i in source_indices)
        matrix = self._dense_w_cache_gpu
        if matrix is None:
            raise RuntimeError("Periodic dense block cache has not been populated.")
        cp = self._cupy()
        nm = int(self.n_modes)
        if len(key) == 0:
            return cp.empty((0, self.n_particles, nm, nm), dtype=self.dtype)
        if key == tuple(range(key[0], key[0] + len(key))):
            columns = matrix[:, key[0] * nm : (key[-1] + 1) * nm]
        else:
            source_index = cp.asarray(key, dtype=cp.int64)
            mode_index = cp.arange(nm, dtype=cp.int64)
            columns = matrix[:, (source_index[:, None] * nm + mode_index).reshape(-1)]
        return columns.reshape(self.n_particles, nm, len(key), nm).transpose(2, 0, 1, 3)

    def populate(self, *, show_progress: bool = False) -> None:
        """Eagerly populate one contiguous device-side dense ``W`` cache.

        Cache-on periodic solves already accept the full ``O(N^2 n_mode^2)``
        storage cost.  Keeping that payload as source-major 4-D chunks made
        every Krylov matvec launch one general ``einsum`` per chunk.  Flattening
        the same blocks once into ``W[destination_mode, source_mode]`` lets the
        hot path use a single cuBLAS matrix-vector or matrix-matrix product.
        """
        if not self.cache_blocks or self._dense_w_cache_gpu is not None:
            return
        cp = self._cupy()
        nm = int(self.n_modes)
        n = int(self.n_particles * nm)
        matrix = cp.empty((n, n), dtype=self.dtype, order="C")
        progress = (
            tqdm(total=self.n_particles, desc="Populate periodic W cache (CuPy)")
            if show_progress
            else None
        )
        try:
            for source_indices in self._source_batches():
                key = tuple(int(i) for i in source_indices)
                blocks = self._compute_blocks_for_sources(key)
                columns = blocks.transpose(1, 2, 0, 3).reshape(n, len(key) * nm)
                start = key[0] * nm
                stop = (key[-1] + 1) * nm
                matrix[:, start:stop] = columns
                if progress is not None:
                    progress.update(len(key))
        finally:
            if progress is not None:
                progress.close()
        self._dense_w_cache_gpu = matrix

    def iter_source_block_batches(
        self, *, show_progress: bool = False
    ) -> Iterable[SourceBlockBatch]:
        """Yield cached or ephemeral source-major blocks for dense assembly."""
        batches: Iterable[tuple[int, ...]] = self._source_batches()
        if show_progress:
            batches = tqdm(
                batches,
                total=(self.n_particles + self._source_batch_size() - 1)
                // self._source_batch_size(),
                desc="Build periodic source blocks (CuPy)",
            )
        for source_indices in batches:
            key = tuple(int(i) for i in source_indices)
            blocks = (
                self._dense_blocks_for_sources(key)
                if self._dense_w_cache_gpu is not None
                else self._compute_blocks_for_sources(key)
            )
            yield SourceBlockBatch(source_indices=key, blocks=blocks)

    def supports_source_block_dense_assembly(self) -> bool:
        """Return whether Ewald source blocks are available for this operator."""
        return self.periodic.options.method == "ewald"

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

        if self.cache_blocks:
            self.populate(show_progress=False)
            matrix = self._dense_w_cache_gpu
            if matrix is None:
                raise RuntimeError("Periodic dense block cache population failed.")
            if squeezed:
                return matrix @ arr.reshape(self.n_particles * self.n_modes)
            return matrix @ arr.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))

        y = cp.zeros_like(arr, dtype=self.dtype)
        tensor = self._contraction_tensor_device()
        for source_indices in self._source_batches():
            source_indexer = self._source_index_device(source_indices)
            sums = self._structural_sums_for_sources(source_indices)
            y += cp.einsum(
                "sdpm,ijpm,sjr->dir",
                sums.astype(self.dtype, copy=False),
                tensor,
                arr[source_indexer],
                optimize=True,
            )
        if squeezed:
            return y.reshape(self.n_particles * self.n_modes)
        return y.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))

    def apply(self, x: Array | object) -> Array | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPeriodicCouplingOperator"]
