"""CuPy-backed single-particle scattering operators.

The CuPy many-body backend already evaluates the inter-particle coupling `W`
with a fused raw kernel. This module keeps the particle-local scattering
operator `T` on device as well, including mixed clusters with diagonal and
dense spherical-basis blocks, so the iterative `A = I - T W` path does not
fall back to CPU for non-spherical particle families.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes

from .groups import AxisymmetricTGroup, DenseTGroup, DiagonalTGroup

Array = np.ndarray
DEFAULT_COMPLEX_DTYPE = np.dtype(np.complex128)


_SHARED_DIAGONAL_APPLY_KERNEL: object | None = None


def _shared_diagonal_apply_kernel(cupy):
    """Return a fused gather/multiply kernel when real CuPy is available."""
    global _SHARED_DIAGONAL_APPLY_KERNEL
    if _SHARED_DIAGONAL_APPLY_KERNEL is None and hasattr(cupy, "ElementwiseKernel"):
        _SHARED_DIAGONAL_APPLY_KERNEL = cupy.ElementwiseKernel(
            "raw T diag, raw int64 operator_indices, T x, int64 nmodes, int64 nrhs",
            "T y",
            """
            const long long particle = i / (nmodes * nrhs);
            const long long mode = (i / nrhs) % nmodes;
            y = diag[operator_indices[particle] * nmodes + mode] * x;
            """,
            "pyceles_shared_diagonal_t_apply",
        )
    return _SHARED_DIAGONAL_APPLY_KERNEL


@dataclass
class CuPyDiagonalTGroup:
    particle_indices: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    operator_indices: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    dtype: np.dtype = DEFAULT_COMPLEX_DTYPE
    _T_diag_gpu: object | None = field(default=None, init=False, repr=False)
    _T_diag_adjoint_gpu: object | None = field(default=None, init=False, repr=False)
    _operator_indices_gpu: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        self.T_M = np.asarray(self.T_M, dtype=self.dtype)
        self.T_N = np.asarray(self.T_N, dtype=self.dtype)
        self.T_diag = np.asarray(self.T_diag, dtype=self.dtype)
        if self.operator_indices.size == 0:
            if self.T_diag.shape[0] == 1:
                self.operator_indices = np.zeros(self.particle_indices.size, dtype=np.int64)
            else:
                self.operator_indices = np.arange(self.particle_indices.size, dtype=np.int64)
        else:
            self.operator_indices = np.asarray(self.operator_indices, dtype=np.int64).reshape(-1)
        if self.operator_indices.size != self.particle_indices.size:
            raise ValueError("`operator_indices` must align with `particle_indices`.")

    def _diag_gpu(self):
        cupy, _ = import_cupy()
        if self._T_diag_gpu is None:
            self._T_diag_gpu = cupy.asarray(self.T_diag, dtype=self.dtype)
        return self._T_diag_gpu

    def _operators_gpu(self):
        cupy, _ = import_cupy()
        if self._operator_indices_gpu is None:
            self._operator_indices_gpu = cupy.asarray(self.operator_indices, dtype=np.int64)
        return self._operator_indices_gpu

    def apply_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Diagonal T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        if self.T_diag.shape[0] == 1:
            diag = self._diag_gpu()[0]
            return diag * arr if int(arr.ndim) == 2 else diag[:, None] * arr
        identity = self.T_diag.shape[0] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )
        if identity:
            diag = self._diag_gpu()
            return diag * arr if int(arr.ndim) == 2 else diag[:, :, None] * arr

        kernel = _shared_diagonal_apply_kernel(cupy)
        if kernel is not None:
            nrhs = 1 if int(arr.ndim) == 2 else int(arr.shape[2])
            return kernel(
                self._diag_gpu(),
                self._operators_gpu(),
                arr,
                np.int64(arr.shape[1]),
                np.int64(nrhs),
            )
        gathered = self._diag_gpu()[self._operators_gpu()]
        return gathered * arr if int(arr.ndim) == 2 else gathered[:, :, None] * arr

    def apply_adjoint_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Diagonal T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        if self._T_diag_adjoint_gpu is None:
            self._T_diag_adjoint_gpu = cupy.conjugate(self._diag_gpu())
        diag = cast(Any, self._T_diag_adjoint_gpu)
        if self.T_diag.shape[0] == 1:
            local = diag[0]
            return local * arr if int(arr.ndim) == 2 else local[:, None] * arr
        identity = self.T_diag.shape[0] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )
        if identity:
            return diag * arr if int(arr.ndim) == 2 else diag[:, :, None] * arr
        kernel = _shared_diagonal_apply_kernel(cupy)
        if kernel is not None:
            nrhs = 1 if int(arr.ndim) == 2 else int(arr.shape[2])
            return kernel(
                diag,
                self._operators_gpu(),
                arr,
                np.int64(arr.shape[1]),
                np.int64(nrhs),
            )
        gathered = diag[self._operators_gpu()]
        return gathered * arr if int(arr.ndim) == 2 else gathered[:, :, None] * arr

    def rhs_subset(self, b_subset: Array | object) -> object:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array | object) -> object:
        cupy, _ = import_cupy()
        operator_index = int(self.operator_indices[int(local_particle_index)])
        local_diag = self._diag_gpu()[operator_index]
        block_arr = cupy.asarray(block, dtype=self.dtype)
        return local_diag[:, None] * block_arr

    def mode_diagonal(self) -> Array | None:
        return np.asarray(self.T_diag[self.operator_indices], dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return (
            np.asarray(self.T_M[self.operator_indices], dtype=self.dtype),
            np.asarray(self.T_N[self.operator_indices], dtype=self.dtype),
        )


@dataclass
class CuPyDenseTGroup:
    particle_indices: Array
    T_blocks: Array
    operator_indices: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    dtype: np.dtype = DEFAULT_COMPLEX_DTYPE
    body_metadata: object | None = None
    _T_blocks_gpu: object | None = field(default=None, init=False, repr=False)
    _T_blocks_adjoint_gpu: object | None = field(default=None, init=False, repr=False)
    _local_indices_gpu: dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        self.T_blocks = np.asarray(self.T_blocks, dtype=self.dtype)
        if self.operator_indices.size == 0:
            if self.T_blocks.shape[0] == 1:
                self.operator_indices = np.zeros(self.particle_indices.size, dtype=np.int64)
            else:
                self.operator_indices = np.arange(self.particle_indices.size, dtype=np.int64)
        else:
            self.operator_indices = np.asarray(self.operator_indices, dtype=np.int64).reshape(-1)
        if self.operator_indices.size != self.particle_indices.size:
            raise ValueError("`operator_indices` must align with `particle_indices`.")

    def _blocks_gpu(self):
        cupy, _ = import_cupy()
        if self._T_blocks_gpu is None:
            self._T_blocks_gpu = cupy.asarray(self.T_blocks, dtype=self.dtype)
        return self._T_blocks_gpu

    def _local_ids_gpu(self, operator_index: int):
        cupy, _ = import_cupy()
        cached = self._local_indices_gpu.get(operator_index)
        if cached is None:
            local = np.flatnonzero(self.operator_indices == operator_index).astype(np.int64)
            cached = cupy.asarray(local, dtype=np.int64)
            self._local_indices_gpu[operator_index] = cached
        return cached

    def apply_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Dense T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        blocks = self._blocks_gpu()
        if self.T_blocks.shape[0] == 1:
            if int(arr.ndim) == 2:
                return cupy.einsum("ij,gj->gi", blocks[0], arr, optimize=True)
            return cupy.einsum("ij,gjr->gir", blocks[0], arr, optimize=True)
        identity = self.T_blocks.shape[0] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )
        if identity:
            if int(arr.ndim) == 2:
                return cupy.einsum("gij,gj->gi", blocks, arr, optimize=True)
            return cupy.einsum("gij,gjr->gir", blocks, arr, optimize=True)

        out = cupy.empty_like(arr)
        for operator_index in range(self.T_blocks.shape[0]):
            local_ids = self._local_ids_gpu(operator_index)
            subset = arr[local_ids]
            if int(arr.ndim) == 2:
                out[local_ids] = cupy.einsum(
                    "ij,gj->gi", blocks[operator_index], subset, optimize=True
                )
            else:
                out[local_ids] = cupy.einsum(
                    "ij,gjr->gir", blocks[operator_index], subset, optimize=True
                )
        return out

    def apply_adjoint_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Dense T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        if self._T_blocks_adjoint_gpu is None:
            self._T_blocks_adjoint_gpu = cupy.ascontiguousarray(
                cupy.conjugate(self._blocks_gpu()).swapaxes(-1, -2)
            )
        blocks = cast(Any, self._T_blocks_adjoint_gpu)
        if self.T_blocks.shape[0] == 1:
            if int(arr.ndim) == 2:
                return cupy.einsum("ij,gj->gi", blocks[0], arr, optimize=True)
            return cupy.einsum("ij,gjr->gir", blocks[0], arr, optimize=True)
        identity = self.T_blocks.shape[0] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )
        if identity:
            if int(arr.ndim) == 2:
                return cupy.einsum("gij,gj->gi", blocks, arr, optimize=True)
            return cupy.einsum("gij,gjr->gir", blocks, arr, optimize=True)
        out = cupy.empty_like(arr)
        for operator_index in range(self.T_blocks.shape[0]):
            local_ids = self._local_ids_gpu(operator_index)
            subset = arr[local_ids]
            if int(arr.ndim) == 2:
                out[local_ids] = cupy.einsum(
                    "ij,gj->gi", blocks[operator_index], subset, optimize=True
                )
            else:
                out[local_ids] = cupy.einsum(
                    "ij,gjr->gir", blocks[operator_index], subset, optimize=True
                )
        return out

    def rhs_subset(self, b_subset: Array | object) -> object:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array | object) -> object:
        cupy, _ = import_cupy()
        operator_index = int(self.operator_indices[int(local_particle_index)])
        block_arr = cupy.asarray(block, dtype=self.dtype)
        return self._blocks_gpu()[operator_index] @ block_arr

    def mode_diagonal(self) -> Array | None:
        return None

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return None


@dataclass
class CuPyCompositeParticleTOperator:
    """GPU-backed composite particle-local `T` operator for mixed groups."""

    lmax: int
    n_particles: int
    groups: Sequence[CuPyDiagonalTGroup | CuPyDenseTGroup]
    dtype: np.dtype = DEFAULT_COMPLEX_DTYPE
    _particle_to_group: np.ndarray = field(init=False, repr=False)
    _particle_to_local: np.ndarray = field(init=False, repr=False)
    _group_indices_gpu: dict[int, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        ns = int(self.n_particles)
        group_of = np.full(ns, -1, dtype=np.int64)
        local_of = np.full(ns, -1, dtype=np.int64)
        for gidx, group in enumerate(self.groups):
            ids = np.asarray(group.particle_indices, dtype=np.int64).reshape(-1)
            if ids.size == 0:
                raise ValueError("Particle-T operator groups must be non-empty.")
            if np.any(ids < 0) or np.any(ids >= ns):
                raise ValueError("Particle-T operator group indices out of bounds.")
            if np.any(group_of[ids] != -1):
                raise ValueError("Particle-T operator groups must not overlap.")
            group_of[ids] = gidx
            local_of[ids] = np.arange(ids.size, dtype=np.int64)
        if np.any(group_of < 0):
            raise ValueError("Particle-T operator groups must cover all particles.")
        self._particle_to_group = group_of
        self._particle_to_local = local_of

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    def _indices_gpu(self, group_index: int, ids: np.ndarray, *, cupy: Any) -> Any:
        """Return a cached device gather/scatter index array for one group."""

        cached = self._group_indices_gpu.get(int(group_index))
        if cached is None:
            cached = cupy.asarray(ids, dtype=np.int64)
            self._group_indices_gpu[int(group_index)] = cached
        return cached

    def _apply_impl(self, x: Array | object, *, group_method: str = "apply_subset") -> object:
        cupy, _ = import_cupy()
        arr_raw = coerce_array(x, dtype=self.dtype, prefer_cupy=True)
        if int(arr_raw.ndim) == 1:
            if int(arr_raw.size) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input length must match n_particles * n_modes. "
                    f"Got {int(arr_raw.size)} for {self.n_particles * self.n_modes}."
                )
            arr = arr_raw.reshape(self.n_particles, self.n_modes)
            # Groups are validated to be disjoint and exhaustive in
            # ``__post_init__``; every output row is assigned exactly once.
            out = cupy.empty((self.n_particles, self.n_modes), dtype=self.dtype)
            for group_index, group in enumerate(self.groups):
                ids = np.asarray(group.particle_indices, dtype=np.int64)
                if ids.size == 0:
                    continue
                if np.all(np.diff(ids) == 1):
                    start = int(ids[0])
                    stop = int(ids[-1]) + 1
                    subset = arr[start:stop]
                    subset_out = cupy.asarray(
                        getattr(group, group_method)(subset), dtype=self.dtype
                    )
                    out[start:stop] = subset_out
                    continue
                ids_gpu = self._indices_gpu(group_index, ids, cupy=cupy)
                subset = arr[ids_gpu]
                subset_out = cupy.asarray(getattr(group, group_method)(subset), dtype=self.dtype)
                out[ids_gpu] = subset_out
            return out.reshape(self.n_particles * self.n_modes)
        elif int(arr_raw.ndim) == 2:
            if int(arr_raw.shape[0]) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input first dimension must match n_particles * n_modes. "
                    f"Got {int(arr_raw.shape[0])} for {self.n_particles * self.n_modes}."
                )
            arr = arr_raw.reshape(self.n_particles, self.n_modes, int(arr_raw.shape[1]))
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(arr_raw.shape)}.")

        out = cupy.empty((self.n_particles, self.n_modes, int(arr.shape[2])), dtype=self.dtype)
        for group_index, group in enumerate(self.groups):
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            # Large sphere-only CuPy runs usually prepare one diagonal group
            # covering a contiguous particle range. Keep that case slice-based.
            #
            # Why this matters:
            # - the older implementation gathered one particle at a time with
            #   `stack([arr[i] for i in ids])`,
            # - then scattered results back one particle at a time,
            # - Nsight Systems profiling showed that pattern generated a storm
            #   of tiny device-to-device copies, one `nmodes`-sized block per
            #   particle per matvec.
            #
            # For `lmax=3` and `complex64`, that meant ~240-byte D2D copies.
            # The profiler count matched `n_particles * n_matvecs` almost
            # exactly, which made the origin of the issue unambiguous.
            #
            # A contiguous group lets us express the same logic as one slice
            # read and one slice write. That preserves the group abstraction
            # while removing the per-particle gather/scatter churn from the
            # hot iterative path.
            #
            # Non-contiguous groups use one vectorized device gather and
            # scatter below. That keeps mixed/interleaved clusters free of the
            # former Python-level per-particle copy storm.
            contiguous = ids.size > 0 and np.all(ids[1:] == ids[:-1] + 1)
            if contiguous:
                start = int(ids[0])
                stop = int(ids[-1]) + 1
                subset = arr[start:stop]
                subset_out = cupy.asarray(getattr(group, group_method)(subset), dtype=self.dtype)
                out[start:stop] = subset_out
                continue
            ids_gpu = self._indices_gpu(group_index, ids, cupy=cupy)
            subset = arr[ids_gpu]
            subset_out = cupy.asarray(getattr(group, group_method)(subset), dtype=self.dtype)
            out[ids_gpu] = subset_out
        out2 = out.reshape(self.n_particles * self.n_modes, int(arr.shape[2]))
        return out2

    def apply(self, x: Array | object) -> Array | object:
        out = self._apply_impl(x)
        return out if is_cupy_array(x) else asnumpy(out)

    def apply_adjoint(self, x: Array | object) -> Array | object:
        out = self._apply_impl(x, group_method="apply_adjoint_subset")
        return out if is_cupy_array(x) else asnumpy(out)

    def rhs(self, b: Array | object) -> Array | object:
        out = self._apply_impl(b)
        return out if is_cupy_array(b) else asnumpy(out)

    def apply_particle_block(self, particle_index: int, block: Array | object) -> Array | object:
        i = int(particle_index)
        gidx = int(self._particle_to_group[i])
        local = int(self._particle_to_local[i])
        out = self.groups[gidx].apply_local_block(local, block)
        return out if is_cupy_array(block) else asnumpy(out)

    def mode_diagonal(self) -> Array | None:
        out = np.empty((self.n_particles, self.n_modes), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diag = group.mode_diagonal()
            if diag is None:
                return None
            out[ids] = np.asarray(diag, dtype=self.dtype)
        return out

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        out_m = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        out_n = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diags = group.degree_diagonals()
            if diags is None:
                return None
            out_m[ids] = np.asarray(diags[0], dtype=self.dtype)
            out_n[ids] = np.asarray(diags[1], dtype=self.dtype)
        return out_m, out_n


def wrap_particle_t_groups_cupy(
    groups: Sequence[DiagonalTGroup | DenseTGroup | AxisymmetricTGroup],
    *,
    lmax: int,
    n_particles: int,
    dtype: np.dtype,
) -> CuPyCompositeParticleTOperator:
    """Wrap prepared NumPy particle-T groups as GPU-backed CuPy groups."""

    gpu_groups: list[CuPyDiagonalTGroup | CuPyDenseTGroup] = []
    for group in groups:
        if isinstance(group, DiagonalTGroup):
            gpu_groups.append(
                CuPyDiagonalTGroup(
                    particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
                    T_M=np.asarray(group.T_M, dtype=dtype),
                    T_N=np.asarray(group.T_N, dtype=dtype),
                    T_diag=np.asarray(group.T_diag, dtype=dtype),
                    operator_indices=np.asarray(group.operator_indices, dtype=np.int64),
                    dtype=dtype,
                )
            )
            continue
        if isinstance(group, DenseTGroup):
            gpu_groups.append(
                CuPyDenseTGroup(
                    particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
                    T_blocks=np.asarray(group.T_blocks, dtype=dtype),
                    operator_indices=np.asarray(group.operator_indices, dtype=np.int64),
                    dtype=dtype,
                )
            )
            continue
        if isinstance(group, AxisymmetricTGroup) and group.T_blocks is not None:
            gpu_groups.append(
                CuPyDenseTGroup(
                    particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
                    T_blocks=np.asarray(group.T_blocks, dtype=dtype),
                    operator_indices=np.asarray(group.operator_indices, dtype=np.int64),
                    body_metadata=group.body_metadata,
                    dtype=dtype,
                )
            )
            continue
        raise NotImplementedError(
            "The CuPy particle-T wrapper requires prepared groups backed by explicit "
            "diagonal data or spherical-basis dense blocks."
        )

    return CuPyCompositeParticleTOperator(
        lmax=int(lmax),
        n_particles=int(n_particles),
        groups=tuple(gpu_groups),
        dtype=np.dtype(dtype),
    )


class CuPyDiagonalParticleTOperator(CuPyCompositeParticleTOperator):
    """Compatibility alias for diagonal-only CuPy `T` preparation."""

    def __init__(
        self,
        lmax: int,
        n_particles: int,
        T_diag: Array,
        T_M: Array,
        T_N: Array,
        dtype: np.dtype = DEFAULT_COMPLEX_DTYPE,
    ) -> None:
        super().__init__(
            lmax=lmax,
            n_particles=n_particles,
            groups=(
                CuPyDiagonalTGroup(
                    particle_indices=np.arange(int(n_particles), dtype=np.int64),
                    T_M=T_M,
                    T_N=T_N,
                    T_diag=T_diag,
                    dtype=dtype,
                ),
            ),
            dtype=dtype,
        )


__all__ = [
    "CuPyCompositeParticleTOperator",
    "CuPyDenseTGroup",
    "CuPyDiagonalParticleTOperator",
    "CuPyDiagonalTGroup",
    "wrap_particle_t_groups_cupy",
]
