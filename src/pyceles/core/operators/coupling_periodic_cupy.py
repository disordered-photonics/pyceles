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
from pyceles.core.periodic.rayleigh import (
    RayleighPlan,
    near_pair_csr,
    prepare_rayleigh_plan,
    resolve_rayleigh_mode_chunk_size,
    valid_structural_indices,
)
from pyceles.core.periodic.rayleigh_cupy import scan_far_cupy, scatter_add_complex
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
    circumscribing_radii: Array | None = None
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
    _rayleigh_plan_cache: RayleighPlan | None = field(default=None, init=False, repr=False)
    _rayleigh_arrays_gpu: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _near_indptr: Array | None = field(default=None, init=False, repr=False)
    _near_destinations: Array | None = field(default=None, init=False, repr=False)
    _near_sources: Array | None = field(default=None, init=False, repr=False)
    _near_structural_sums_gpu: Any | None = field(default=None, init=False, repr=False)
    _self_block_gpu: Any | None = field(default=None, init=False, repr=False)
    _near_contraction_tensor_gpu: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy periodic coupling supports only complex64 and complex128. "
                f"Got {self.dtype!r}."
            )
        if self.periodic.options.method not in {"ewald", "rayleigh"}:
            raise NotImplementedError(
                "CuPy periodic coupling supports periodic methods 'ewald' and 'rayleigh'."
            )
        if self.periodic.options.method == "rayleigh" and self.cache_blocks:
            raise ValueError(
                "`cache_translation_blocks=True` is incompatible with periodic "
                "method='rayleigh'; the hybrid operator already caches only sparse near data."
            )

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

    def _rayleigh_plan(self) -> RayleighPlan:
        plan = self._rayleigh_plan_cache
        if plan is not None:
            return plan
        plan = prepare_rayleigh_plan(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            circumscribing_radii=self.circumscribing_radii,
            periodic=self.periodic,
            k_parallel=self.k_parallel,
            dtype=self.dtype,
        )
        self._rayleigh_plan_cache = plan
        return plan

    def _rayleigh_device_array(self, name: str, value: Any, *, dtype: Any | None = None) -> Any:
        cached = self._rayleigh_arrays_gpu.get(name)
        if cached is not None:
            return cached
        cp = self._cupy()
        out = cp.asarray(value, dtype=dtype)
        self._rayleigh_arrays_gpu[name] = out
        return out

    def _reshape_input_device(self, x: Array | object) -> tuple[Any, bool]:
        """Normalize a stacked vector or RHS matrix to ``(N, Nm, nrhs)``."""
        cp = self._cupy()
        arr_raw = coerce_array(x, dtype=self.dtype, prefer_cupy=True)
        expected = self.n_particles * self.n_modes
        if int(arr_raw.ndim) == 1:
            if int(arr_raw.size) != expected:
                raise ValueError(
                    "Input length must match n_particles * n_modes. "
                    f"Got {int(arr_raw.size)} for {expected}."
                )
            return cp.asarray(arr_raw, dtype=self.dtype).reshape(
                self.n_particles, self.n_modes, 1
            ), True
        if int(arr_raw.ndim) == 2:
            if int(arr_raw.shape[0]) != expected:
                raise ValueError(
                    "Input first dimension must match n_particles * n_modes. "
                    f"Got {int(arr_raw.shape[0])} for {expected}."
                )
            return cp.asarray(arr_raw, dtype=self.dtype).reshape(
                self.n_particles, self.n_modes, int(arr_raw.shape[1])
            ), False
        raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(arr_raw.shape)}.")

    @staticmethod
    def _memory_bounded_batch_size(
        *, total: int, bytes_per_item: int, workspace_bytes: int = 256 * 1024**2
    ) -> int:
        """Return a nonzero batch size within one temporary-memory budget."""
        return max(
            1,
            min(
                int(total),
                int(workspace_bytes) // max(1, int(bytes_per_item)),
            ),
        )

    def _near_apply_batch_size(self, *, total: int, n_rhs: int) -> int:
        """Bound exact-near contraction intermediates by a predictable budget.

        A three-operand ``einsum`` over every near pair can legally choose the
        contraction ``structural @ tensor`` first.  That materializes one dense
        ``(Nm, Nm)`` block per pair, even though the final contribution has only
        ``Nm`` entries.  Count that block (and one equally sized library
        workspace) explicitly so dense vertical bands cannot create multi-GiB
        temporaries before the final scatter.
        """
        degrees, _orders = valid_structural_indices(int(self.lmax))
        nm = int(self.n_modes)
        rhs = max(1, int(n_rhs))
        itemsize = int(self.dtype.itemsize)
        bytes_per_pair = itemsize * (
            int(degrees.size)  # complex128 -> compute-dtype structural cast
            + 2 * nm * nm  # dense block plus contraction workspace
            + 3 * nm * rhs  # gathered source, contribution, and matmul workspace
        )
        return self._memory_bounded_batch_size(
            total=int(total),
            bytes_per_item=int(bytes_per_pair),
        )

    def _near_structure(self) -> tuple[Array, Array, Array]:
        if (
            self._near_indptr is None
            or self._near_destinations is None
            or self._near_sources is None
        ):
            indptr, destinations, sources = near_pair_csr(
                self.positions, self._rayleigh_plan().z_cut
            )
            self._near_indptr = indptr
            self._near_destinations = destinations
            self._near_sources = sources
        return self._near_indptr, self._near_destinations, self._near_sources

    def _near_contraction_tensor_device(self) -> Any:
        """Return the device tensor restricted to valid structural channels."""
        if self._near_contraction_tensor_gpu is not None:
            return self._near_contraction_tensor_gpu
        cp = self._cupy()
        degrees, orders = valid_structural_indices(int(self.lmax))
        tensor_np = translation_contraction_tensor(
            lmax=int(self.lmax),
            ab5=np.asarray(self.ab5),
            dtype=self.dtype,
        )[:, :, degrees, orders]
        self._near_contraction_tensor_gpu = cp.asarray(tensor_np, dtype=self.dtype)
        return self._near_contraction_tensor_gpu

    def _self_block_device(self) -> Any:
        """Return the exact periodic self block shared by every particle."""
        if self._self_block_gpu is not None:
            return self._self_block_gpu
        cp = self._cupy()
        order = 2 * int(self.lmax)
        relative = cp.zeros((1, 3), dtype=cp.float64)
        same_plane = cp.zeros((1,), dtype=cp.int64)
        real_count, reciprocal_count = self._shell_counts()
        structural = ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=relative,
            lmax_struct=int(self.lmax),
            workspace=self._workspace_device(),
            real_shell_count=int(real_count),
            reciprocal_shell_count=int(reciprocal_count),
            coordinate_scale=self._coordinate_scale_value(),
            same_plane_pair_indices=same_plane,
        )
        structural = cp.asarray(structural, dtype=cp.complex128).reshape(
            1, order + 1, 2 * order + 1
        )[0]
        structural[0, order] += self._self_correction_device()
        self._self_block_gpu = cp.einsum(
            "pm,ijpm->ij",
            structural.astype(self.dtype, copy=False),
            self._contraction_tensor_device(),
            optimize=True,
        ).astype(self.dtype, copy=False)
        return self._self_block_gpu

    def _populate_near_structural_sums_device(self, *, show_progress: bool = False) -> None:
        if self._near_structural_sums_gpu is not None:
            return
        cp = self._cupy()
        _indptr, destinations, sources = self._near_structure()
        order = 2 * int(self.lmax)
        degrees, orders = valid_structural_indices(int(self.lmax))
        total = int(destinations.size)
        sums = cp.empty((total, degrees.size), dtype=cp.complex128)
        if total == 0:
            self._near_structural_sums_gpu = sums
            return
        src_gpu = self._rayleigh_device_array("near_sources", sources, dtype=cp.int32)
        dst_gpu = self._rayleigh_device_array("near_destinations", destinations, dtype=cp.int32)
        degree_gpu = self._rayleigh_device_array("near_structural_degrees", degrees, dtype=cp.int32)
        order_gpu = self._rayleigh_device_array("near_structural_orders", orders, dtype=cp.int32)
        pos = self._positions_device()
        real_count, reciprocal_count = self._shell_counts()
        structural_width = (order + 1) * (2 * order + 1)
        pair_batch = self._memory_bounded_batch_size(
            total=total,
            bytes_per_item=structural_width * np.dtype(np.complex128).itemsize,
        )
        batches: Iterable[tuple[int, int]] = (
            (start, min(total, start + pair_batch)) for start in range(0, total, pair_batch)
        )
        if show_progress:
            batches = tqdm(
                batches,
                total=(total + pair_batch - 1) // pair_batch,
                desc="Build periodic near Ewald cache (CuPy)",
            )
        atol = same_plane_z_tolerance(
            float(self.k), coordinate_scale=self._coordinate_scale_value()
        )
        for start, stop in batches:
            src = src_gpu[start:stop]
            dst = dst_gpu[start:stop]
            rel = pos[src] - pos[dst]
            local_same = cp.flatnonzero(cp.abs(rel[:, 2]) <= float(atol)).astype(cp.int64)
            local = ewald_structural_sums_2d_fixed_cupy(
                relative_source_minus_destination=rel,
                lmax_struct=int(self.lmax),
                workspace=self._workspace_device(),
                real_shell_count=int(real_count),
                reciprocal_shell_count=int(reciprocal_count),
                coordinate_scale=self._coordinate_scale_value(),
                same_plane_pair_indices=local_same,
            )
            local = cp.asarray(local, dtype=cp.complex128).reshape(
                stop - start, order + 1, 2 * order + 1
            )
            sums[start:stop] = local[:, degree_gpu, order_gpu]
        self._near_structural_sums_gpu = sums

    def _apply_rayleigh_gpu(self, x: Array | object) -> Any:
        cp = self._cupy()
        arr, squeezed = self._reshape_input_device(x)

        plan = self._rayleigh_plan()
        order = self._rayleigh_device_array("sort_order", plan.sort_order, dtype=cp.int64)
        inverse = self._rayleigh_device_array("inverse_order", plan.inverse_order, dtype=cp.int64)
        z = self._rayleigh_device_array("sorted_z", plan.sorted_z, dtype=cp.float64)
        gamma_all = self._rayleigh_device_array("gamma", plan.gamma, dtype=cp.complex128)
        phase_all = self._rayleigh_device_array(
            "sorted_xy_phase", plan.sorted_xy_phase, dtype=self.dtype
        )
        source_tables = self._rayleigh_device_array(
            "source_tables", plan.source_tables, dtype=self.dtype
        )
        destination_tables = self._rayleigh_device_array(
            "destination_tables", plan.destination_tables, dtype=self.dtype
        )
        weights = self._rayleigh_device_array("weights", plan.weights, dtype=self.dtype)
        arr_sorted = arr[order]
        y_sorted = cp.zeros_like(arr_sorted)
        chunk = resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=plan.n_modes_reciprocal,
            n_particles=int(arr.shape[0]),
            n_rhs=int(arr.shape[2]),
            dtype=self.dtype,
        )
        for start in range(0, plan.n_modes_reciprocal, chunk):
            stop = min(plan.n_modes_reciprocal, start + chunk)
            phase = phase_all[:, start:stop]
            gamma = gamma_all[start:stop]
            for direction, upward in ((0, True), (1, False)):
                source = cp.einsum(
                    "qpm,amr->aqpr",
                    source_tables[direction, start:stop],
                    arr_sorted,
                    optimize=True,
                )
                source *= cp.conjugate(phase)[:, :, None, None]
                incoming = scan_far_cupy(
                    source_amplitudes=source,
                    z=z,
                    gamma=gamma,
                    z_cut=float(plan.z_cut),
                    upward=upward,
                    cupy=cp,
                )
                del source
                y_sorted += cp.einsum(
                    "qpm,aqpr,aq,q->amr",
                    destination_tables[direction, start:stop],
                    incoming,
                    phase,
                    weights[start:stop],
                    optimize=True,
                )
                del incoming
        y = y_sorted[inverse]
        y += cp.einsum("ij,ajr->air", self._self_block_device(), arr, optimize=True)

        self._populate_near_structural_sums_device(show_progress=False)
        sums = self._near_structural_sums_gpu
        if sums is None:
            raise RuntimeError("Periodic near Ewald cache population failed.")
        _indptr, destinations, sources = self._near_structure()
        total = int(destinations.size)
        if total:
            src_gpu = self._rayleigh_device_array("near_sources", sources, dtype=cp.int32)
            dst_gpu = self._rayleigh_device_array("near_destinations", destinations, dtype=cp.int32)
            tensor = self._near_contraction_tensor_device()
            pair_batch = self._near_apply_batch_size(
                total=total,
                n_rhs=int(arr.shape[2]),
            )
            for start in range(0, total, pair_batch):
                stop = min(total, start + pair_batch)
                structural = sums[start:stop].astype(self.dtype, copy=False)
                blocks = cp.einsum(
                    "av,ijv->aij",
                    structural,
                    tensor,
                    optimize=True,
                )
                contribution = cp.matmul(
                    blocks,
                    arr[src_gpu[start:stop]],
                )
                scatter_add_complex(
                    y,
                    dst_gpu[start:stop],
                    contribution,
                    cupy=cp,
                )
                del structural, blocks, contribution

        flat = y.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))
        return flat[:, 0] if squeezed else flat

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
        """Eagerly populate reusable periodic coupling data.

        For ``method='rayleigh'`` this prepares the reciprocal projection tables
        and sparse exact-near structural cache. For cache-on Ewald it builds one
        contiguous device-side dense ``W`` matrix.

        Cache-on periodic solves already accept the full ``O(N^2 n_mode^2)``
        storage cost.  Keeping that payload as source-major 4-D chunks made
        every Krylov matvec launch one general ``einsum`` per chunk.  Flattening
        the same blocks once into ``W[destination_mode, source_mode]`` lets the
        hot path use a single cuBLAS matrix-vector or matrix-matrix product.
        """
        if self.periodic.options.method == "rayleigh":
            self._rayleigh_plan()
            self._self_block_device()
            self._populate_near_structural_sums_device(show_progress=show_progress)
            return
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
        """Return whether exact Ewald source blocks match the configured apply."""
        return self.periodic.options.method == "ewald"

    def _apply_gpu(self, x: Array | object) -> Any:
        if self.periodic.options.method == "rayleigh":
            return self._apply_rayleigh_gpu(x)
        cp = self._cupy()
        arr, squeezed = self._reshape_input_device(x)

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
