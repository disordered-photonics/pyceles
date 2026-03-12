from __future__ import annotations

"""CuPy-backed single-particle scattering operators.

The CuPy many-body backend already evaluates the inter-particle coupling `W`
with a fused raw kernel. This module keeps the particle-local scattering
operator `T` on device as well, including mixed clusters with diagonal and
dense spherical-basis blocks, so the iterative `A = I - T W` path does not
fall back to CPU for non-spherical particle families.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes

from .groups import AxisymmetricTGroup, DenseTGroup, DiagonalTGroup

Array = np.ndarray
DEFAULT_COMPLEX_DTYPE = np.dtype(np.complex128)


@dataclass
class CuPyDiagonalTGroup:
    particle_indices: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    dtype: np.dtype = DEFAULT_COMPLEX_DTYPE
    _T_diag_gpu: object | None = field(default=None, init=False, repr=False)

    def _diag_gpu(self):
        cupy, _ = import_cupy()
        if self._T_diag_gpu is None:
            self._T_diag_gpu = cupy.asarray(self.T_diag, dtype=self.dtype)
        return self._T_diag_gpu

    def apply_subset(self, x_subset: Array | object) -> object:
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        return self._diag_gpu() * arr

    def rhs_subset(self, b_subset: Array | object) -> object:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array | object) -> object:
        cupy, _ = import_cupy()
        local_diag = self._diag_gpu()[int(local_particle_index)]
        block_arr = cupy.asarray(block, dtype=self.dtype)
        return local_diag[:, None] * block_arr

    def mode_diagonal(self) -> Array | None:
        return np.asarray(self.T_diag, dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return (
            np.asarray(self.T_M, dtype=self.dtype),
            np.asarray(self.T_N, dtype=self.dtype),
        )


@dataclass
class CuPyDenseTGroup:
    particle_indices: Array
    T_blocks: Array
    dtype: np.dtype = DEFAULT_COMPLEX_DTYPE
    body_metadata: object | None = None
    _T_blocks_gpu: object | None = field(default=None, init=False, repr=False)

    def _blocks_gpu(self):
        cupy, _ = import_cupy()
        if self._T_blocks_gpu is None:
            self._T_blocks_gpu = cupy.asarray(self.T_blocks, dtype=self.dtype)
        return self._T_blocks_gpu

    def apply_subset(self, x_subset: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x_subset, dtype=self.dtype, prefer_cupy=True)
        return cupy.einsum("gij,gj->gi", self._blocks_gpu(), arr, optimize=True)

    def rhs_subset(self, b_subset: Array | object) -> object:
        return self.apply_subset(b_subset)

    def apply_local_block(self, local_particle_index: int, block: Array | object) -> object:
        cupy, _ = import_cupy()
        block_arr = cupy.asarray(block, dtype=self.dtype)
        return self._blocks_gpu()[int(local_particle_index)] @ block_arr

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

    def _apply_impl(self, x: Array | object) -> object:
        cupy, _ = import_cupy()
        arr = coerce_array(x, dtype=self.dtype, prefer_cupy=True).reshape(
            self.n_particles, self.n_modes
        )
        out = cupy.zeros((self.n_particles, self.n_modes), dtype=self.dtype)
        for group in self.groups:
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
            # Non-contiguous groups still use the older safe path below. That
            # matters for mixed clusters where representation groups can be
            # interleaved in particle order. If those cases become performance
            # critical later, they likely need a dedicated grouped-index kernel
            # rather than Python-level per-particle assembly.
            contiguous = ids.size > 0 and np.all(ids[1:] == ids[:-1] + 1)
            if contiguous:
                start = int(ids[0])
                stop = int(ids[-1]) + 1
                subset = arr[start:stop]
                subset_out = cupy.asarray(group.apply_subset(subset), dtype=self.dtype)
                out[start:stop] = subset_out
                continue
            subset = cupy.stack([arr[int(i)] for i in ids], axis=0)
            subset_out = cupy.asarray(group.apply_subset(subset), dtype=self.dtype)
            for local, particle_index in enumerate(ids):
                out[int(particle_index)] = subset_out[int(local)]
        return out.reshape(self.n_particles * self.n_modes)

    def apply(self, x: Array | object) -> Array | object:
        out = self._apply_impl(x)
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
                    dtype=dtype,
                )
            )
            continue
        if isinstance(group, DenseTGroup):
            gpu_groups.append(
                CuPyDenseTGroup(
                    particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
                    T_blocks=np.asarray(group.T_blocks, dtype=dtype),
                    dtype=dtype,
                )
            )
            continue
        if isinstance(group, AxisymmetricTGroup) and group.T_blocks is not None:
            gpu_groups.append(
                CuPyDenseTGroup(
                    particle_indices=np.asarray(group.particle_indices, dtype=np.int64),
                    T_blocks=np.asarray(group.T_blocks, dtype=dtype),
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
