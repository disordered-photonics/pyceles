"""CuPy periodic coupling operator for rectangular two-dimensional lattices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from pyceles._cupy_memory import CuPyAllocatorSnapshot, cupy_allocator_snapshot
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
    RAYLEIGH_TRANSIENT_WORKSPACE_BYTES,
    RayleighNearCacheEstimate,
    RayleighPlan,
    _long_lived_scan_mode_indices,
    near_pair_csr,
    prepare_rayleigh_plan,
    rayleigh_near_cache_estimate,
    resolve_rayleigh_mode_chunk_size,
    valid_structural_indices,
)
from pyceles.core.periodic.rayleigh_cupy import (
    _scan_far_indexed_cupy,
    apply_sparse_near_adjoint_cupy,
    apply_sparse_near_coupling_cupy,
    scan_far_cupy,
)
from pyceles.core.periodic.scalar import structural_sum_m_normalization
from pyceles.core.periodic.structural import (
    sparse_translation_contraction,
    translation_contraction_tensor,
    transpose_sparse_translation_contraction,
)

from .base import SourceBlockBatch

Array = np.ndarray

RayleighNearCacheResidency = Literal["device", "host"]


@dataclass(frozen=True)
class RayleighNearCacheMemoryPlan:
    """Resolved persistent-cache residency under one guarded GPU budget."""

    estimate: RayleighNearCacheEstimate
    residency: RayleighNearCacheResidency
    device_limit_bytes: int
    pool_used_bytes: int
    available_device_bytes: int
    remaining_rayleigh_device_bytes: int
    transient_reserve_bytes: int
    required_device_bytes: int


def _resolve_rayleigh_near_cache_memory_plan(
    *,
    estimate: RayleighNearCacheEstimate,
    snapshot: CuPyAllocatorSnapshot,
    remaining_rayleigh_device_bytes: int,
    transient_reserve_bytes: int = RAYLEIGH_TRANSIENT_WORKSPACE_BYTES,
) -> RayleighNearCacheMemoryPlan:
    """Choose device or host cache residency without relying on WDDM spill."""
    remaining = max(0, int(remaining_rayleigh_device_bytes))
    reserve = max(0, int(transient_reserve_bytes))
    available = min(
        int(snapshot.active_headroom_bytes),
        int(snapshot.raw_free_bytes) + int(snapshot.pool_free_bytes),
    )
    base_required = remaining + reserve
    if base_required > available:
        raise MemoryError(
            "The Rayleigh reciprocal plan and bounded apply workspace do not fit "
            "inside the guarded CuPy device-memory budget. Reduce the reciprocal "
            "window, free other device allocations, or use a larger GPU."
        )
    cache_bytes = int(estimate.structural_bytes)
    required_device = base_required + cache_bytes
    cache_allocation_fits = cache_bytes <= int(snapshot.guaranteed_fresh_allocation_bytes)
    residency: RayleighNearCacheResidency = (
        "device" if required_device <= available and cache_allocation_fits else "host"
    )
    return RayleighNearCacheMemoryPlan(
        estimate=estimate,
        residency=residency,
        device_limit_bytes=int(snapshot.effective_device_limit_bytes),
        pool_used_bytes=int(snapshot.pool_used_bytes),
        available_device_bytes=int(available),
        remaining_rayleigh_device_bytes=remaining,
        transient_reserve_bytes=reserve,
        required_device_bytes=required_device,
    )


@dataclass
class CuPyPeriodicCouplingOperator:
    """GPU-backed Bloch-reduced periodic coupling descriptor.

    Exact Ewald uses fixed shell counts resolved once from the periodic options
    and remains source-batched when dense block caching is disabled. Hybrid
    Rayleigh coupling keeps the reciprocal scan on device and places its compact
    exact-near cache on device or host according to the guarded memory plan.
    ``dtype`` controls device storage and output, while ``accum_dtype`` is the
    requested precision budget for the Rayleigh scan recurrence. The c64/c128
    policy selectively widens only physically long-lived reciprocal modes.
    """

    lmax: int
    k: float
    positions: Array
    ab5: Array
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype
    accum_dtype: np.dtype | None = None
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
    _rayleigh_plan_cache: RayleighPlan | None = field(default=None, init=False, repr=False)
    _rayleigh_arrays_gpu: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _near_destinations: Array | None = field(default=None, init=False, repr=False)
    _near_sources: Array | None = field(default=None, init=False, repr=False)
    _near_structural_sums_gpu: Any | None = field(default=None, init=False, repr=False)
    _near_structural_sums_host: Array | None = field(default=None, init=False, repr=False)
    _near_structural_staging_gpu: Any | None = field(default=None, init=False, repr=False)
    _near_cache_memory_plan: RayleighNearCacheMemoryPlan | None = field(
        default=None, init=False, repr=False
    )
    _self_block_gpu: Any | None = field(default=None, init=False, repr=False)
    _near_sparse_contraction_gpu: tuple[Any, Any, Any, Any] | None = field(
        default=None, init=False, repr=False
    )
    _near_sparse_adjoint_contraction_gpu: tuple[Any, Any, Any, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        self.accum_dtype = self.dtype if self.accum_dtype is None else np.dtype(self.accum_dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy periodic coupling supports only complex64 and complex128. "
                f"Got {self.dtype!r}."
            )
        if self.accum_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy periodic coupling accumulation requires complex64 or complex128. "
                f"Got {self.accum_dtype!r}."
            )
        if self.accum_dtype.itemsize < self.dtype.itemsize:
            raise ValueError(
                "CuPy periodic coupling accumulation dtype cannot be narrower than the "
                f"operator dtype ({self.dtype.name}); got {self.accum_dtype.name}."
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
            rayleigh_z_cut = (
                self._rayleigh_plan().z_cut if self.periodic.options.method == "rayleigh" else None
            )
            self._resolved_ewald_eta = resolve_ewald_eta(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=float).reshape(2),
                positions=np.asarray(self.positions, dtype=float).reshape(-1, 3),
                lmax=int(self.lmax),
                max_vertical_offset=rayleigh_z_cut,
            )
        return float(self._resolved_ewald_eta)

    def _shell_counts(self) -> tuple[int, int]:
        if self._resolved_shell_counts is None:
            rayleigh_z_cut = (
                self._rayleigh_plan().z_cut if self.periodic.options.method == "rayleigh" else None
            )
            counts = resolve_ewald_shell_counts(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=float).reshape(2),
                positions=np.asarray(self.positions, dtype=float).reshape(-1, 3),
                lmax=int(self.lmax),
                eta=self._ewald_eta(),
                max_vertical_offset=rayleigh_z_cut,
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

    def _flatten_rayleigh_output_device(self, values: Any, *, squeezed: bool) -> Any:
        flat = values.reshape(self.n_particles * self.n_modes, int(values.shape[2]))
        return flat[:, 0] if squeezed else flat

    @staticmethod
    def _memory_bounded_batch_size(
        *,
        total: int,
        bytes_per_item: int,
        workspace_bytes: int = RAYLEIGH_TRANSIENT_WORKSPACE_BYTES,
    ) -> int:
        """Return a nonzero batch size within one temporary-memory budget."""
        return max(
            1,
            min(
                int(total),
                int(workspace_bytes) // max(1, int(bytes_per_item)),
            ),
        )

    def _near_apply_batch_size(self, *, total: int) -> int:
        """Bound host-cache staging used by sparse exact-near contractions."""
        n_structural = (2 * int(self.lmax) + 1) ** 2
        # The fused kernel reads source coefficients and sparse contraction
        # metadata in place, so the only pair-sized temporary is one structural
        # cache row when host-resident cache slices are staged to the device.
        bytes_per_pair = int(self.dtype.itemsize) * n_structural
        return self._memory_bounded_batch_size(
            total=int(total),
            bytes_per_item=int(bytes_per_pair),
        )

    def _near_host_staging_device(self, *, rows: int, width: int) -> Any:
        """Return one reusable device buffer for host-resident near-cache slices."""
        cp = self._cupy()
        required_rows = max(1, int(rows))
        required_width = int(width)
        staging = self._near_structural_staging_gpu
        if (
            staging is None
            or int(staging.shape[0]) < required_rows
            or int(staging.shape[1]) != required_width
            or np.dtype(staging.dtype) != self.dtype
        ):
            staging = cp.empty((required_rows, required_width), dtype=self.dtype)
            self._near_structural_staging_gpu = staging
        return staging[:required_rows]

    def _near_pairs(self) -> tuple[Array, Array]:
        """Return flat source-major exact-near destination/source indices."""
        if self._near_destinations is None or self._near_sources is None:
            _indptr, destinations, sources = near_pair_csr(
                self.positions, self._rayleigh_plan().z_cut
            )
            self._near_destinations = destinations
            self._near_sources = sources
        return self._near_destinations, self._near_sources

    def _near_sparse_contraction_device(self) -> tuple[Any, Any, Any, Any]:
        """Return output-mode CSR data for the sparse translation contraction."""
        cached = self._near_sparse_contraction_gpu
        if cached is not None:
            return cached
        cp = self._cupy()
        row_ptr, input_modes, channels, values = sparse_translation_contraction(
            lmax=int(self.lmax),
            ab5=np.asarray(self.ab5),
            dtype=self.dtype,
        )
        cached = (
            cp.asarray(row_ptr),
            cp.asarray(input_modes),
            cp.asarray(channels),
            cp.asarray(values),
        )
        self._near_sparse_contraction_gpu = cached
        return cached

    def _near_sparse_adjoint_contraction_device(self) -> tuple[Any, Any, Any, Any]:
        """Return input-mode CSR data for the exact-near adjoint."""
        cached = self._near_sparse_adjoint_contraction_gpu
        if cached is not None:
            return cached
        cp = self._cupy()
        row_ptr, input_modes, channels, values = sparse_translation_contraction(
            lmax=int(self.lmax), ab5=np.asarray(self.ab5), dtype=self.dtype
        )
        cached = tuple(
            cp.asarray(value)
            for value in transpose_sparse_translation_contraction(
                row_ptr, input_modes, channels, values
            )
        )
        self._near_sparse_adjoint_contraction_gpu = cached
        return cached

    def _release_rayleigh_preparation_state(self) -> None:
        """Release Ewald-only device tables after Rayleigh cache preparation."""
        if self._self_block_gpu is None:
            return
        if self._near_structural_sums_gpu is None and self._near_structural_sums_host is None:
            return
        # Preserve the sparse contraction used by repeated near applies, then
        # release the full structural tensor and reciprocal/real Ewald tables.
        self._near_sparse_contraction_device()
        self._workspace = None
        self._contraction_tensor_gpu = None
        self._self_correction_gpu = None

    def _self_block_device(self) -> Any:
        """Return the exact periodic self block shared by every particle."""
        if self._self_block_gpu is not None:
            return self._self_block_gpu
        cp = self._cupy()
        order = 2 * int(self.lmax)
        relative = cp.zeros((1, 3), dtype=cp.float64)
        real_count, reciprocal_count = self._shell_counts()
        structural = ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=relative,
            lmax_struct=int(self.lmax),
            workspace=self._workspace_device(),
            real_shell_count=int(real_count),
            reciprocal_shell_count=int(reciprocal_count),
            coordinate_scale=self._coordinate_scale_value(),
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

    @staticmethod
    def _format_gib(nbytes: int) -> str:
        return f"{float(nbytes) / float(1024**3):.2f} GiB"

    def _remaining_rayleigh_device_bytes(self) -> int:
        """Estimate persistent Rayleigh arrays not yet uploaded to the device."""
        plan = self._rayleigh_plan()
        total = 0

        def pending(name: str, value: Any, dtype: Any) -> int:
            if name in self._rayleigh_arrays_gpu:
                return 0
            return int(np.asarray(value).size) * int(np.dtype(dtype).itemsize)

        total += pending("sort_order", plan.sort_order, np.int32)
        total += pending("inverse_order", plan.inverse_order, np.int32)
        total += pending("sorted_z", plan.sorted_z, np.float64)
        total += pending("gamma", plan.gamma, np.complex128)
        total += pending("sorted_xy_phase", plan.sorted_xy_phase, self.dtype)
        total += pending("source_tables", plan.source_tables, self.dtype)
        total += pending("destination_tables", plan.destination_tables, self.dtype)
        total += pending("weights", plan.weights, self.dtype)

        destinations, sources = self._near_pairs()
        degrees, orders = valid_structural_indices(int(self.lmax))
        total += pending("near_sources", sources, np.int32)
        total += pending("near_destinations", destinations, np.int32)
        total += pending("near_structural_degrees", degrees, np.int32)
        total += pending("near_structural_orders", orders, np.int32)
        if self._positions_gpu is None:
            total += self.n_particles * 3 * int(np.dtype(np.float64).itemsize)
        return int(total)

    def _near_cache_plan(self) -> RayleighNearCacheMemoryPlan:
        plan = self._near_cache_memory_plan
        if plan is not None:
            return plan
        cp = self._cupy()
        destinations, _sources = self._near_pairs()
        # Materialize the compact contraction metadata before taking the pool
        # snapshot, so its actual allocation is included without estimating a
        # dense tensor that the repeated-apply path no longer stores.
        self._near_sparse_contraction_device()
        estimate = rayleigh_near_cache_estimate(
            pair_count=int(destinations.size),
            lmax=int(self.lmax),
            dtype=self.dtype,
        )
        remaining = self._remaining_rayleigh_device_bytes()
        snapshot = cupy_allocator_snapshot(
            cp,
            apply_pool_limit=True,
        )
        plan = _resolve_rayleigh_near_cache_memory_plan(
            estimate=estimate,
            snapshot=snapshot,
            remaining_rayleigh_device_bytes=remaining,
        )
        if (
            plan.residency == "host"
            and plan.required_device_bytes <= plan.available_device_bytes
            and int(estimate.structural_bytes) > int(snapshot.guaranteed_fresh_allocation_bytes)
            and int(snapshot.pool_free_bytes) > 0
        ):
            snapshot = cupy_allocator_snapshot(
                cp,
                apply_pool_limit=True,
                required_fresh_allocation_bytes=int(estimate.structural_bytes),
            )
            plan = _resolve_rayleigh_near_cache_memory_plan(
                estimate=estimate,
                snapshot=snapshot,
                remaining_rayleigh_device_bytes=remaining,
            )
        self._near_cache_memory_plan = plan
        return plan

    def _populate_near_structural_sums_device(self, *, show_progress: bool = False) -> None:
        if (
            self._near_structural_sums_gpu is not None
            or self._near_structural_sums_host is not None
        ):
            return
        cp = self._cupy()
        destinations, sources = self._near_pairs()
        order = 2 * int(self.lmax)
        degrees, orders = valid_structural_indices(int(self.lmax))
        total = int(destinations.size)
        memory_plan = self._near_cache_plan()
        sums_gpu: Any | None = None
        sums_host: Array | None = None
        if memory_plan.residency == "device":
            try:
                sums_gpu = cp.empty((total, degrees.size), dtype=self.dtype)
            except cp.cuda.memory.OutOfMemoryError:
                memory_plan = replace(memory_plan, residency="host")
                self._near_cache_memory_plan = memory_plan
                sums_host = np.empty((total, degrees.size), dtype=self.dtype)
        else:
            sums_host = np.empty((total, degrees.size), dtype=self.dtype)
        if total == 0:
            self._near_structural_sums_gpu = sums_gpu
            self._near_structural_sums_host = sums_host
            return

        src_gpu = self._rayleigh_device_array("near_sources", sources, dtype=cp.int32)
        dst_gpu = self._rayleigh_device_array("near_destinations", destinations, dtype=cp.int32)
        degree_gpu = self._rayleigh_device_array("near_structural_degrees", degrees, dtype=cp.int32)
        order_gpu = self._rayleigh_device_array("near_structural_orders", orders, dtype=cp.int32)
        pos = self._positions_device()
        real_count, reciprocal_count = self._shell_counts()
        structural_width = (order + 1) * (2 * order + 1)
        bytes_per_pair = structural_width * np.dtype(np.complex128).itemsize + int(
            degrees.size
        ) * int(self.dtype.itemsize)
        pair_batch = self._memory_bounded_batch_size(
            total=total,
            bytes_per_item=bytes_per_pair,
        )
        batches: Iterable[tuple[int, int]] = (
            (start, min(total, start + pair_batch)) for start in range(0, total, pair_batch)
        )
        if show_progress:
            location = "GPU" if memory_plan.residency == "device" else "host"
            batches = tqdm(
                batches,
                total=(total + pair_batch - 1) // pair_batch,
                desc=(
                    "Build periodic near Ewald cache "
                    f"({location}, {self._format_gib(memory_plan.estimate.structural_bytes)})"
                ),
            )
        for start, stop in batches:
            src = src_gpu[start:stop]
            dst = dst_gpu[start:stop]
            rel = pos[src] - pos[dst]
            local = ewald_structural_sums_2d_fixed_cupy(
                relative_source_minus_destination=rel,
                lmax_struct=int(self.lmax),
                workspace=self._workspace_device(),
                real_shell_count=int(real_count),
                reciprocal_shell_count=int(reciprocal_count),
                coordinate_scale=self._coordinate_scale_value(),
            )
            local = cp.asarray(local, dtype=cp.complex128).reshape(
                stop - start, order + 1, 2 * order + 1
            )
            compact = cp.ascontiguousarray(
                local[:, degree_gpu, order_gpu],
                dtype=self.dtype,
            )
            if sums_gpu is not None:
                sums_gpu[start:stop] = compact
            elif sums_host is not None:
                compact.get(out=sums_host[start:stop])
            else:
                raise RuntimeError("Periodic near Ewald cache allocation failed.")
            del rel, local, compact
        self._near_structural_sums_gpu = sums_gpu
        self._near_structural_sums_host = sums_host

    def _apply_rayleigh_far_adjoint_reshaped_gpu(self, arr: Any) -> Any:
        """Apply the Hermitian adjoint of the stored reciprocal Rayleigh field."""
        cp = self._cupy()
        plan = self._rayleigh_plan()
        order = self._rayleigh_device_array("sort_order", plan.sort_order, dtype=cp.int32)
        inverse = self._rayleigh_device_array("inverse_order", plan.inverse_order, dtype=cp.int32)
        z = self._rayleigh_device_array("sorted_z", plan.sorted_z, dtype=cp.float64)
        gamma_all = self._rayleigh_device_array("gamma", plan.gamma, dtype=cp.complex128)
        gamma_adjoint_all = self._rayleigh_arrays_gpu.get("gamma_adjoint")
        if gamma_adjoint_all is None:
            gamma_adjoint_all = -cp.conjugate(gamma_all)
            self._rayleigh_arrays_gpu["gamma_adjoint"] = gamma_adjoint_all
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
        accum_dtype = np.dtype(self.accum_dtype)
        z_span = float(plan.sorted_z[-1] - plan.sorted_z[0]) if plan.sorted_z.size else 0.0
        chunk = resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=plan.n_modes_reciprocal,
            n_particles=int(arr.shape[0]),
            n_rhs=int(arr.shape[2]),
            dtype=self.dtype,
        )
        for start in range(0, plan.n_modes_reciprocal, chunk):
            stop = min(plan.n_modes_reciprocal, start + chunk)
            phase = phase_all[:, start:stop]
            gamma_adjoint = gamma_adjoint_all[start:stop]
            wide_indices = (
                _long_lived_scan_mode_indices(
                    plan.gamma[start:stop],
                    z_span=z_span,
                    z_cut=float(plan.z_cut),
                    storage_dtype=self.dtype,
                )
                if accum_dtype.itemsize > self.dtype.itemsize
                else np.empty((0,), dtype=np.int64)
            )
            wide_indices_gpu = (
                self._rayleigh_device_array(
                    f"scan_adjoint_wide_indices_{start}_{stop}", wide_indices, dtype=cp.int32
                )
                if wide_indices.size
                else None
            )
            for direction, upward in ((0, True), (1, False)):
                source = cp.einsum(
                    "qpm,amr->aqpr",
                    cp.conjugate(destination_tables[direction, start:stop]),
                    arr_sorted,
                    optimize=True,
                )
                source *= cp.conjugate(phase)[:, :, None, None]
                source *= cp.conjugate(weights[start:stop])[None, :, None, None]
                if wide_indices.size == int(stop - start) and wide_indices.size:
                    incoming = scan_far_cupy(
                        source_amplitudes=source,
                        z=z,
                        gamma=gamma_adjoint,
                        z_cut=float(plan.z_cut),
                        upward=not upward,
                        cupy=cp,
                        accumulation_dtype=accum_dtype,
                    )
                else:
                    incoming = scan_far_cupy(
                        source_amplitudes=source,
                        z=z,
                        gamma=gamma_adjoint,
                        z_cut=float(plan.z_cut),
                        upward=not upward,
                        cupy=cp,
                    )
                    if wide_indices_gpu is not None:
                        _scan_far_indexed_cupy(
                            source_amplitudes=source,
                            z=z,
                            gamma=gamma_adjoint,
                            z_cut=float(plan.z_cut),
                            upward=not upward,
                            q_indices=wide_indices_gpu,
                            output=incoming,
                            cupy=cp,
                            accumulation_dtype=accum_dtype,
                        )
                del source
                y_sorted += cp.einsum(
                    "qpm,aqpr,aq->amr",
                    cp.conjugate(source_tables[direction, start:stop]),
                    incoming,
                    phase,
                    optimize=True,
                )
                del incoming
        return y_sorted[inverse]

    def _apply_rayleigh_gpu(self, x: Array | object) -> Any:
        cp = self._cupy()
        arr, squeezed = self._reshape_input_device(x)

        plan = self._rayleigh_plan()
        order = self._rayleigh_device_array("sort_order", plan.sort_order, dtype=cp.int32)
        inverse = self._rayleigh_device_array("inverse_order", plan.inverse_order, dtype=cp.int32)
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
        accum_dtype = np.dtype(self.accum_dtype)
        z_span = float(plan.sorted_z[-1] - plan.sorted_z[0]) if plan.sorted_z.size else 0.0
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
            wide_indices = (
                _long_lived_scan_mode_indices(
                    plan.gamma[start:stop],
                    z_span=z_span,
                    z_cut=float(plan.z_cut),
                    storage_dtype=self.dtype,
                )
                if accum_dtype.itemsize > self.dtype.itemsize
                else np.empty((0,), dtype=np.int64)
            )
            wide_indices_gpu = (
                self._rayleigh_device_array(
                    f"scan_wide_indices_{start}_{stop}",
                    wide_indices,
                    dtype=cp.int32,
                )
                if wide_indices.size
                else None
            )
            for direction, upward in ((0, True), (1, False)):
                source = cp.einsum(
                    "qpm,amr->aqpr",
                    source_tables[direction, start:stop],
                    arr_sorted,
                    optimize=True,
                )
                source *= cp.conjugate(phase)[:, :, None, None]
                if wide_indices.size == int(stop - start) and wide_indices.size:
                    incoming = scan_far_cupy(
                        source_amplitudes=source,
                        z=z,
                        gamma=gamma,
                        z_cut=float(plan.z_cut),
                        upward=upward,
                        cupy=cp,
                        accumulation_dtype=accum_dtype,
                    )
                else:
                    incoming = scan_far_cupy(
                        source_amplitudes=source,
                        z=z,
                        gamma=gamma,
                        z_cut=float(plan.z_cut),
                        upward=upward,
                        cupy=cp,
                    )
                    if wide_indices_gpu is not None:
                        _scan_far_indexed_cupy(
                            source_amplitudes=source,
                            z=z,
                            gamma=gamma,
                            z_cut=float(plan.z_cut),
                            upward=upward,
                            q_indices=wide_indices_gpu,
                            output=incoming,
                            cupy=cp,
                            accumulation_dtype=accum_dtype,
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
        self._release_rayleigh_preparation_state()
        sums_gpu = self._near_structural_sums_gpu
        sums_host = self._near_structural_sums_host
        if sums_gpu is None and sums_host is None:
            raise RuntimeError("Periodic near Ewald cache population failed.")
        destinations, sources = self._near_pairs()
        total = int(destinations.size)
        if total:
            src_gpu = self._rayleigh_device_array("near_sources", sources, dtype=cp.int32)
            dst_gpu = self._rayleigh_device_array("near_destinations", destinations, dtype=cp.int32)
            row_ptr, input_modes, structural_channels, contraction_values = (
                self._near_sparse_contraction_device()
            )
            pair_batch = self._near_apply_batch_size(total=total)
            structural_staging = (
                None
                if sums_gpu is not None
                else self._near_host_staging_device(
                    rows=pair_batch,
                    width=(2 * int(self.lmax) + 1) ** 2,
                )
            )
            for start in range(0, total, pair_batch):
                stop = min(total, start + pair_batch)
                if sums_gpu is not None:
                    structural = sums_gpu[start:stop]
                else:
                    if sums_host is None or structural_staging is None:
                        raise RuntimeError("Periodic host near cache is unavailable.")
                    structural = structural_staging[: stop - start]
                    structural.set(sums_host[start:stop])
                apply_sparse_near_coupling_cupy(
                    target=y,
                    structural=structural,
                    coefficients=arr,
                    sources=src_gpu[start:stop],
                    destinations=dst_gpu[start:stop],
                    row_ptr=row_ptr,
                    input_modes=input_modes,
                    structural_channels=structural_channels,
                    values=contraction_values,
                    cupy=cp,
                )
                del structural

        flat = y.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))
        return flat[:, 0] if squeezed else flat

    def _apply_rayleigh_near_adjoint_reshaped_gpu(
        self, arr: Any, *, target: Any | None = None
    ) -> Any:
        """Apply the exact adjoint of periodic self plus exact-near blocks."""
        cp = self._cupy()
        self_adjoint = self._rayleigh_arrays_gpu.get("self_block_adjoint")
        if self_adjoint is None:
            self_adjoint = cp.ascontiguousarray(cp.conjugate(self._self_block_device().T))
            self._rayleigh_arrays_gpu["self_block_adjoint"] = self_adjoint
        self_contribution = cp.einsum("ij,ajr->air", self_adjoint, arr, optimize=True)
        y = cp.ascontiguousarray(self_contribution) if target is None else target
        if target is not None:
            y += self_contribution

        self._populate_near_structural_sums_device(show_progress=False)
        self._release_rayleigh_preparation_state()
        sums_gpu = self._near_structural_sums_gpu
        sums_host = self._near_structural_sums_host
        if sums_gpu is None and sums_host is None:
            raise RuntimeError("Periodic near Ewald cache population failed.")
        destinations, sources = self._near_pairs()
        total = int(destinations.size)
        if total:
            src_gpu = self._rayleigh_device_array("near_sources", sources, dtype=cp.int32)
            dst_gpu = self._rayleigh_device_array("near_destinations", destinations, dtype=cp.int32)
            row_ptr, input_modes, structural_channels, contraction_values = (
                self._near_sparse_adjoint_contraction_device()
            )
            pair_batch = self._near_apply_batch_size(total=total)
            structural_staging = (
                None
                if sums_gpu is not None
                else self._near_host_staging_device(
                    rows=pair_batch, width=(2 * int(self.lmax) + 1) ** 2
                )
            )
            for start in range(0, total, pair_batch):
                stop = min(total, start + pair_batch)
                if sums_gpu is not None:
                    structural = sums_gpu[start:stop]
                else:
                    if sums_host is None or structural_staging is None:
                        raise RuntimeError("Periodic host near cache is unavailable.")
                    structural = structural_staging[: stop - start]
                    structural.set(sums_host[start:stop])
                apply_sparse_near_adjoint_cupy(
                    target=y,
                    structural=structural,
                    coefficients=arr,
                    input_particles=dst_gpu[start:stop],
                    output_particles=src_gpu[start:stop],
                    row_ptr=row_ptr,
                    input_modes=input_modes,
                    structural_channels=structural_channels,
                    values=contraction_values,
                    cupy=cp,
                )
                del structural
        return y

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
            plan = self._rayleigh_plan()
            if show_progress:
                tqdm.write(
                    "[Rayleigh] plan "
                    f"reciprocal_orders={plan.n_modes_reciprocal} "
                    f"z_cut={plan.z_cut:.6g}"
                )
            self._self_block_device()
            self._populate_near_structural_sums_device(show_progress=show_progress)
            self._release_rayleigh_preparation_state()
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

    def _apply_rayleigh_adjoint_gpu(self, x: Array | object) -> Any:
        arr, squeezed = self._reshape_input_device(x)
        y = self._apply_rayleigh_far_adjoint_reshaped_gpu(arr)
        self._apply_rayleigh_near_adjoint_reshaped_gpu(arr, target=y)
        return self._flatten_rayleigh_output_device(y, squeezed=squeezed)

    def apply_adjoint(self, x: Array | object) -> Array | object:
        """Apply ``W^H`` for the exact stored Rayleigh discretization."""
        if self.periodic.options.method != "rayleigh":
            raise NotImplementedError(
                "Periodic coupling adjoints are currently implemented only for method='rayleigh'."
            )
        out = self._apply_rayleigh_adjoint_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPeriodicCouplingOperator"]
