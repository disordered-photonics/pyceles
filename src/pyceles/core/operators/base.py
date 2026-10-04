"""Core operator protocols and prepared-operator boundary."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from pyceles._optional import coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes

from .single_body import ParticleTOperator

Array = np.ndarray
COMPLEX128_DTYPE = np.dtype(np.complex128)


@runtime_checkable
class CouplingOperator(Protocol):
    """Prepared many-body coupling operator `W`.

    Full-vector actions return writable results independent of the input and
    of reusable operator workspace. A result must remain valid after later
    forward or adjoint calls. Views of a fresh result allocation are fine.
    """

    def apply(self, x: Array) -> Array: ...


@runtime_checkable
class AdjointCouplingOperator(CouplingOperator, Protocol):
    """Optional coupling protocol for exact Hermitian-adjoint actions."""

    def apply_adjoint(self, x: Array) -> Array: ...


@runtime_checkable
class PrecomputableCouplingOperator(Protocol):
    """Optional coupling protocol for backends that support eager precomputation."""

    def populate(self, *, show_progress: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceBlockBatch:
    """One source-major batch of dense coupling blocks.

    ``blocks`` has shape ``(n_source, n_destination, n_mode, n_mode)``.
    Couplings may produce these batches transiently or serve them from a
    persistent cache; dense assembly does not need to know which policy owns
    the data.
    """

    source_indices: tuple[int, ...]
    blocks: Any


@runtime_checkable
class SourceBlockCouplingOperator(Protocol):
    """Optional coupling protocol for source-streamed dense assembly."""

    def supports_source_block_dense_assembly(self) -> bool: ...

    def iter_source_block_batches(
        self, *, show_progress: bool = False, for_dense_assembly: bool = False
    ) -> Iterable[SourceBlockBatch]: ...


@dataclass(frozen=True, slots=True)
class SourceBlockDenseAssembly:
    """Dense matrix plus separately measured block-generation/assembly time."""

    matrix: Any
    block_generation_seconds: float
    assembly_seconds: float


def _synchronize_if_cupy(value: Any) -> None:
    if not is_cupy_array(value):
        return
    cupy, _ = import_cupy()
    cupy.cuda.Stream.null.synchronize()


@dataclass
class PreparedOperator:
    """Prepared linear operator `A = I - T W` with explicit `T`/`W` boundaries."""

    lmax: int
    k: float
    positions: Array
    particle_t: ParticleTOperator
    coupling: CouplingOperator
    dtype: np.dtype = COMPLEX128_DTYPE

    def apply_W(self, x: Array) -> Array:
        return self.coupling.apply(x)

    @staticmethod
    def _subtract_into(minuend: Any, subtrahend: Any) -> Any:
        """Compute ``minuend - subtrahend`` into the owned second operand."""
        if is_cupy_array(subtrahend):
            cupy, _ = import_cupy()
            cupy.subtract(minuend, subtrahend, out=subtrahend)
        else:
            np.subtract(minuend, subtrahend, out=subtrahend)
        return subtrahend

    def apply_A(self, x: Array) -> Array:
        wx = self.apply_W(x)
        x_arr = coerce_array(x, dtype=self.dtype, prefer_cupy=False)
        # Particle-T applications own their output. Reuse it as the final A(x)
        # vector instead of allocating a third full-width temporary.
        tx = self.particle_t.apply(wx)
        return cast(Array, self._subtract_into(x_arr, tx))

    @property
    def supports_adjoint(self) -> bool:
        """Whether this prepared operator exposes an exact Hermitian adjoint."""

        return (
            callable(getattr(self.coupling, "apply_adjoint", None))
            and callable(getattr(self.particle_t, "apply_adjoint", None))
            and bool(getattr(self.particle_t, "supports_adjoint", False))
        )

    def make_adjoint(self) -> Callable[[Any], Any]:
        """Return the exact prepared adjoint action when available."""

        if not self.supports_adjoint:
            raise NotImplementedError(
                "This prepared operator does not expose exact T/W adjoint actions."
            )
        return self.apply_adjoint

    def apply_adjoint(self, x: Array) -> Array:
        """Apply ``A^H = I - W^H T^H`` while preserving RHS shape and ownership."""
        if not self.supports_adjoint:
            raise NotImplementedError(
                "This prepared operator does not expose exact T/W adjoint actions."
            )
        coupling_adjoint = cast(AdjointCouplingOperator, self.coupling).apply_adjoint
        particle_adjoint = self.particle_t.apply_adjoint
        values = coerce_array(x, dtype=self.dtype, prefer_cupy=False)
        weighted = particle_adjoint(values)
        adjoint_coupling = coupling_adjoint(weighted)
        return cast(Array, self._subtract_into(values, adjoint_coupling))

    def rhs(self, b: Array) -> Array:
        return self.particle_t.rhs(b)

    def rhs_Tb(self, b: Array) -> Array:
        return self.rhs(b)

    def apply_particle_block(self, particle_index: int, block: Array) -> Array:
        return self.particle_t.apply_particle_block(particle_index, block)

    def populate_coupling(self, *, show_progress: bool = False) -> None:
        if isinstance(self.coupling, PrecomputableCouplingOperator):
            self.coupling.populate(show_progress=show_progress)

    def assemble_dense_from_source_blocks(
        self, *, show_progress: bool = False
    ) -> SourceBlockDenseAssembly | None:
        """Stream optional source-major W blocks directly into dense ``A``.

        This keeps temporary block ownership inside the coupling operator and
        applies particle-local T blocks through the prepared boundary. The
        diagonal path remains vectorized, while non-diagonal groups use the
        canonical local-block operation.
        """
        coupling = self.coupling
        if not isinstance(coupling, SourceBlockCouplingOperator):
            return None
        if not coupling.supports_source_block_dense_assembly():
            return None

        ns = int(np.asarray(self.positions).reshape(-1, 3).shape[0])
        nm = int(n_modes(self.lmax))
        n = ns * nm
        batches = iter(
            coupling.iter_source_block_batches(show_progress=show_progress, for_dense_assembly=True)
        )
        generation_seconds = 0.0
        assembly_seconds = 0.0
        matrix: Any | None = None
        diagonal = self.particle_t.mode_diagonal()
        assembled_sources = np.zeros((ns,), dtype=bool)

        while True:
            generation_t0 = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            _synchronize_if_cupy(batch.blocks)
            generation_seconds += time.perf_counter() - generation_t0

            blocks = batch.blocks
            if matrix is None:
                if is_cupy_array(blocks):
                    cupy, _ = import_cupy()
                    matrix = cupy.empty((n, n), dtype=self.dtype, order="F")
                    diag_backend = (
                        None
                        if diagonal is None
                        else cupy.asarray(diagonal, dtype=self.dtype).reshape(ns, nm)
                    )
                else:
                    matrix = np.empty((n, n), dtype=self.dtype)
                    diag_backend = (
                        None
                        if diagonal is None
                        else np.asarray(diagonal, dtype=self.dtype).reshape(ns, nm)
                    )

            assembly_t0 = time.perf_counter()
            source_indices = tuple(int(i) for i in batch.source_indices)
            if len(source_indices) == 0:
                continue
            if min(source_indices) < 0 or max(source_indices) >= ns:
                raise IndexError("Source-block batch contains an out-of-range particle index.")
            source_array = np.asarray(source_indices, dtype=np.int64)
            if np.any(assembled_sources[source_array]):
                raise ValueError("Source-block batches must not repeat source indices.")
            assembled_sources[source_array] = True
            if diag_backend is not None:
                weighted = -(blocks.astype(self.dtype, copy=False) * diag_backend[None, :, :, None])
            else:
                weighted = blocks.copy()
                for local_source in range(len(source_indices)):
                    for destination in range(ns):
                        weighted[local_source, destination] = -self.apply_particle_block(
                            destination, blocks[local_source, destination]
                        )

            columns = weighted.transpose(1, 2, 0, 3).reshape(n, len(source_indices) * nm)
            start = source_indices[0]
            stop = source_indices[-1] + 1
            if stop - start == len(source_indices):
                matrix[:, start * nm : stop * nm] = columns
            else:
                for local_source, source_index in enumerate(source_indices):
                    column_slice = slice(source_index * nm, (source_index + 1) * nm)
                    local_slice = slice(local_source * nm, (local_source + 1) * nm)
                    matrix[:, column_slice] = columns[:, local_slice]
            _synchronize_if_cupy(matrix)
            assembly_seconds += time.perf_counter() - assembly_t0

        if matrix is None:
            matrix = np.eye(n, dtype=self.dtype)
        else:
            if not bool(np.all(assembled_sources)):
                missing = np.flatnonzero(~assembled_sources)
                raise ValueError(
                    "Source-block batches did not cover every source particle; "
                    f"missing {missing[:8].tolist()}."
                )
            identity_t0 = time.perf_counter()
            if is_cupy_array(matrix):
                cupy, _ = import_cupy()
                diagonal_indices = cupy.arange(n)
            else:
                diagonal_indices = np.arange(n)
            matrix[diagonal_indices, diagonal_indices] += 1
            _synchronize_if_cupy(matrix)
            assembly_seconds += time.perf_counter() - identity_t0
        return SourceBlockDenseAssembly(
            matrix=matrix,
            block_generation_seconds=float(generation_seconds),
            assembly_seconds=float(assembly_seconds),
        )

    @property
    def T_diag(self) -> Array:
        diag = self.particle_t.mode_diagonal()
        if diag is None:
            raise NotImplementedError("Prepared operator does not expose diagonal per-mode T data.")
        return diag

    @property
    def T_M(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[0]

    @property
    def T_N(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[1]


__all__ = [
    "Array",
    "CouplingOperator",
    "PrecomputableCouplingOperator",
    "PreparedOperator",
    "SourceBlockBatch",
    "SourceBlockCouplingOperator",
    "SourceBlockDenseAssembly",
]
