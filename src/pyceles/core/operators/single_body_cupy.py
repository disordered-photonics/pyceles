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

from .groups import (
    DenseTGroup,
    DiagonalTGroup,
    PreparedParticleTGroup,
    _normalized_operator_indices,
    _operator_local_indices,
)

Array = np.ndarray
DEFAULT_COMPLEX_DTYPE = np.dtype(np.complex128)


_SHARED_DIAGONAL_APPLY_KERNEL: object | None = None


def _shared_diagonal_apply_kernel(cupy):
    """Return the fused gather/multiply kernel for shared diagonal rows."""
    global _SHARED_DIAGONAL_APPLY_KERNEL
    if _SHARED_DIAGONAL_APPLY_KERNEL is None:
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
    _identity_operator_map: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        self.T_M = np.asarray(self.T_M, dtype=self.dtype)
        self.T_N = np.asarray(self.T_N, dtype=self.dtype)
        self.T_diag = np.asarray(self.T_diag, dtype=self.dtype)
        if self.T_M.shape[0] != self.T_N.shape[0] or self.T_M.shape[0] != self.T_diag.shape[0]:
            raise ValueError("Diagonal T data must have one row per prepared archetype.")
        self.particle_indices, self.operator_indices = _normalized_operator_indices(
            self.particle_indices,
            None if self.operator_indices.size == 0 else self.operator_indices,
            n_operators=self.T_diag.shape[0],
        )
        self._identity_operator_map = self.T_diag.shape[
            0
        ] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )

    @property
    def supports_adjoint(self) -> bool:
        return True

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

    def _apply_diagonal(self, x_subset: Array | object, diag: Any) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Diagonal T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        if self.T_diag.shape[0] == 1:
            local = diag[0]
            return local * arr if int(arr.ndim) == 2 else local[:, None] * arr
        if self._identity_operator_map:
            return diag * arr if int(arr.ndim) == 2 else diag[:, :, None] * arr
        nrhs = 1 if int(arr.ndim) == 2 else int(arr.shape[2])
        return _shared_diagonal_apply_kernel(cupy)(
            diag,
            self._operators_gpu(),
            arr,
            np.int64(arr.shape[1]),
            np.int64(nrhs),
        )

    def apply_subset(self, x_subset: Array | object) -> object:
        return self._apply_diagonal(x_subset, self._diag_gpu())

    def apply_adjoint_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        if self._T_diag_adjoint_gpu is None:
            self._T_diag_adjoint_gpu = cupy.conjugate(self._diag_gpu())
        return self._apply_diagonal(x_subset, self._T_diag_adjoint_gpu)

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
    _T_blocks_gpu: object | None = field(default=None, init=False, repr=False)
    _local_indices_host: tuple[Array, ...] = field(default=(), init=False, repr=False)
    _local_indices_gpu: dict[int, object] = field(default_factory=dict, init=False, repr=False)
    _identity_operator_map: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.particle_indices = np.asarray(self.particle_indices, dtype=np.int64).reshape(-1)
        self.T_blocks = np.asarray(self.T_blocks, dtype=self.dtype)
        if self.T_blocks.ndim != 3 or self.T_blocks.shape[1] != self.T_blocks.shape[2]:
            raise ValueError(f"`T_blocks` must have shape (Nu, Nm, Nm). Got {self.T_blocks.shape}.")
        self.particle_indices, self.operator_indices = _normalized_operator_indices(
            self.particle_indices,
            None if self.operator_indices.size == 0 else self.operator_indices,
            n_operators=self.T_blocks.shape[0],
        )
        self._identity_operator_map = self.T_blocks.shape[
            0
        ] == self.particle_indices.size and np.array_equal(
            self.operator_indices, np.arange(self.particle_indices.size)
        )
        self._local_indices_host = _operator_local_indices(
            self.operator_indices, self.T_blocks.shape[0]
        )

    @property
    def supports_adjoint(self) -> bool:
        return True

    def _blocks_gpu(self):
        cupy, _ = import_cupy()
        if self._T_blocks_gpu is None:
            self._T_blocks_gpu = cupy.asarray(self.T_blocks, dtype=self.dtype)
        return self._T_blocks_gpu

    def _local_ids_gpu(self, operator_index: int):
        cupy, _ = import_cupy()
        cached = self._local_indices_gpu.get(operator_index)
        if cached is None:
            local = self._local_indices_host[int(operator_index)]
            cached = cupy.asarray(local, dtype=np.int64)
            self._local_indices_gpu[operator_index] = cached
        return cached

    def _apply_blocks(self, arr: object, blocks: object, *, transposed: bool) -> object:
        cupy, _ = import_cupy()
        ndim = int(cast(Any, arr).ndim)
        shared = (
            ("ji,gj->gi" if ndim == 2 else "ji,gjr->gir")
            if transposed
            else ("ij,gj->gi" if ndim == 2 else "ij,gjr->gir")
        )
        mapped = (
            ("gji,gj->gi" if ndim == 2 else "gji,gjr->gir")
            if transposed
            else ("gij,gj->gi" if ndim == 2 else "gij,gjr->gir")
        )
        if self.T_blocks.shape[0] == 1:
            return cupy.einsum(shared, cast(Any, blocks)[0], arr, optimize=True)
        if self._identity_operator_map:
            return cupy.einsum(mapped, blocks, arr, optimize=True)

        out = cupy.empty_like(arr)
        for operator_index in range(self.T_blocks.shape[0]):
            local_ids = self._local_ids_gpu(operator_index)
            out[local_ids] = cupy.einsum(
                shared,
                cast(Any, blocks)[operator_index],
                cast(Any, arr)[local_ids],
                optimize=True,
            )
        return out

    def apply_subset(self, x_subset: Array | object) -> object:
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Dense T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        return self._apply_blocks(arr, self._blocks_gpu(), transposed=False)

    def apply_adjoint_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        if int(arr.ndim) not in {2, 3}:
            raise ValueError(f"Dense T-group subset must be 2D or 3D. Got ndim={int(arr.ndim)}.")
        # T^H x = conj(T^T conj(x)); the transpose is a view of the already
        # resident forward blocks, so adjoint use does not duplicate T storage.
        mapped = self._apply_blocks(
            cupy.conjugate(arr),
            self._blocks_gpu(),
            transposed=True,
        )
        cupy.conjugate(mapped, out=mapped)
        return mapped

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
    _full_group: CuPyDiagonalTGroup | CuPyDenseTGroup | None = field(
        default=None, init=False, repr=False
    )
    _group_slices: tuple[slice | None, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ns = int(self.n_particles)
        group_of = np.full(ns, -1, dtype=np.int64)
        local_of = np.full(ns, -1, dtype=np.int64)
        group_slices: list[slice | None] = []
        for gidx, group in enumerate(self.groups):
            ids = np.asarray(group.particle_indices, dtype=np.int64).reshape(-1)
            if ids.size == 0:
                raise ValueError("Particle-T operator groups must be non-empty.")
            if np.any(ids < 0) or np.any(ids >= ns):
                raise ValueError("Particle-T operator group indices out of bounds.")
            if np.unique(ids).size != ids.size:
                raise ValueError("Particle-T operator group indices must be unique.")
            if np.any(group_of[ids] != -1):
                raise ValueError("Particle-T operator groups must not overlap.")
            group_of[ids] = gidx
            local_of[ids] = np.arange(ids.size, dtype=np.int64)
            group_slices.append(
                slice(int(ids[0]), int(ids[-1]) + 1) if np.all(ids[1:] == ids[:-1] + 1) else None
            )
        self._group_slices = tuple(group_slices)
        if np.any(group_of < 0):
            raise ValueError("Particle-T operator groups must cover all particles.")
        self._particle_to_group = group_of
        self._particle_to_local = local_of
        if len(self.groups) == 1:
            ids = np.asarray(self.groups[0].particle_indices, dtype=np.int64).reshape(-1)
            if np.array_equal(ids, np.arange(ns, dtype=np.int64)):
                self._full_group = self.groups[0]

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    @property
    def supports_adjoint(self) -> bool:
        return all(bool(group.supports_adjoint) for group in self.groups)

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
        expected = self.n_particles * self.n_modes
        if int(arr_raw.ndim) == 1:
            if int(arr_raw.size) != expected:
                raise ValueError(
                    f"Input length must match n_particles * n_modes. Got {int(arr_raw.size)} for {expected}."
                )
            arr = arr_raw.reshape(self.n_particles, self.n_modes)
        elif int(arr_raw.ndim) == 2:
            if int(arr_raw.shape[0]) != expected:
                raise ValueError(
                    "Input first dimension must match n_particles * n_modes. "
                    f"Got {int(arr_raw.shape[0])} for {expected}."
                )
            arr = arr_raw.reshape(self.n_particles, self.n_modes, int(arr_raw.shape[1]))
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(arr_raw.shape)}.")

        if self._full_group is not None:
            result = cupy.asarray(getattr(self._full_group, group_method)(arr), dtype=self.dtype)
            return result.reshape(arr_raw.shape)

        # The immutable prepared layout was classified once in __post_init__.
        # Contiguous groups use views; interleaved groups retain the cached
        # device gather/scatter indices. Every output row is assigned once.
        out = cupy.empty_like(arr)
        for group_index, (group, selection) in enumerate(
            zip(self.groups, self._group_slices, strict=True)
        ):
            if selection is None:
                selection = self._indices_gpu(group_index, group.particle_indices, cupy=cupy)
            subset = arr[selection]
            subset_out = cupy.asarray(getattr(group, group_method)(subset), dtype=self.dtype)
            out[selection] = subset_out
            # Do not carry a previous group's gather/output into the next
            # group's allocation or dense contraction.
            del subset, subset_out
        return out.reshape(arr_raw.shape)

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
    groups: Sequence[PreparedParticleTGroup],
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


__all__ = [
    "CuPyCompositeParticleTOperator",
    "CuPyDenseTGroup",
    "CuPyDiagonalTGroup",
    "wrap_particle_t_groups_cupy",
]
