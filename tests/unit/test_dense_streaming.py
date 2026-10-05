"""Small layout and lifetime oracles for source-streamed dense assembly."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.base import CouplingOperator, PreparedOperator, SourceBlockBatch
from pyceles.core.operators.single_body import ParticleTOperator


class _Blocks:
    def __init__(self, values: Any, batches: tuple[tuple[int, ...], ...], xp: Any) -> None:
        self.values = values
        self.batches = batches
        self.xp = xp

    def supports_source_block_dense_assembly(self) -> bool:
        return True

    def iter_source_block_batches(
        self, *, show_progress: bool = False, for_dense_assembly: bool = False
    ) -> Iterator[SourceBlockBatch]:
        assert for_dense_assembly
        previous: weakref.ReferenceType[Any] | None = None
        for indices in self.batches:
            assert previous is None or previous() is None, "previous batch is still retained"
            # Modulo indexing permits deliberate invalid-id fixtures to reach
            # the assembler's validation, rather than failing in this producer.
            ids = self.xp.asarray([i % 4 for i in indices], dtype=self.xp.int64)
            blocks = self.values[ids].copy()
            previous = weakref.ref(blocks)
            yield SourceBlockBatch(indices, blocks)
            del blocks
        assert previous is None or previous() is None


class _ParticleT:
    def __init__(self, values: Any, diagonal: bool, xp: Any) -> None:
        self.values = values
        self.diagonal = diagonal
        self.xp = xp

    def mode_diagonal(self) -> Any:
        if self.diagonal:
            return self.xp.diagonal(self.values, axis1=1, axis2=2)
        return None

    def apply_particle_block(self, particle_index: int, block: Any) -> Any:
        return self.values[particle_index] @ block


def _case(xp: Any, dtype: Any, diagonal: bool, batches: tuple[tuple[int, ...], ...]):
    rng = np.random.default_rng(42)
    ns, nm = 4, n_modes(1)
    w = (rng.standard_normal((ns, ns, nm, nm)) + 1j * rng.standard_normal((ns, ns, nm, nm))).astype(
        dtype
    )
    t = (rng.standard_normal((ns, nm, nm)) + 1j * rng.standard_normal((ns, nm, nm))).astype(dtype)
    if diagonal:
        t *= np.eye(nm, dtype=dtype)[None, :, :]
    w_device, t_device = xp.asarray(w), xp.asarray(t)
    operator = PreparedOperator(
        lmax=1,
        k=1.0,
        positions=np.zeros((ns, 3)),
        particle_t=cast(ParticleTOperator, _ParticleT(t_device, diagonal, xp)),
        coupling=cast(CouplingOperator, _Blocks(w_device, batches, xp)),
        dtype=np.dtype(dtype),
    )
    expected = np.eye(ns * nm, dtype=dtype)
    for destination in range(ns):
        for source in range(ns):
            expected[
                destination * nm : (destination + 1) * nm, source * nm : (source + 1) * nm
            ] -= t[destination] @ w[source, destination]
    return operator, expected, w_device, w, t_device, t


@pytest.mark.parametrize("diagonal", (False, True))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("batches", (((0, 1), (2, 3)), ((0, 2, 1, 3),), ((), (2, 0), (3, 1))))
def test_source_streaming_layout_and_lifetimes(diagonal, dtype, batches) -> None:
    operator, expected, w_live, w, t_live, t = _case(np, dtype, diagonal, batches)
    result = operator.assemble_dense_from_source_blocks()
    assert result is not None
    tolerance = 3e-6 if dtype == np.complex64 else 2e-14
    np.testing.assert_allclose(result.matrix, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(w_live, w)
    np.testing.assert_array_equal(t_live, t)


@pytest.mark.parametrize(
    ("batches", "error", "message"),
    [
        ((), ValueError, "cover every source"),
        (((),), ValueError, "cover every source"),
        (((0, 1, 2),), ValueError, "cover every source"),
        (((0, 0, 1, 2, 3),), ValueError, "repeat source"),
        (((0, 1), (1, 2, 3)), ValueError, "repeat source"),
        (((0, 1, 2, 4),), IndexError, "out-of-range"),
    ],
)
def test_source_streaming_rejects_incomplete_or_invalid_partition(batches, error, message) -> None:
    operator, *_ = _case(np, np.complex128, True, batches)
    with pytest.raises(error, match=message):
        operator.assemble_dense_from_source_blocks()


@pytest.mark.gpu
@pytest.mark.parametrize("diagonal", (False, True))
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("batches", (((0, 1), (2, 3)), ((0, 2, 1, 3),)))
def test_source_streaming_fortran_device_destination(
    cupy_runtime, diagonal, dtype, batches
) -> None:
    cp, _ = cupy_runtime
    # Also cover non-default-stream timing/synchronization in the assembler.
    with cp.cuda.Stream(non_blocking=True):
        operator, expected, w_live, w, t_live, t = _case(cp, dtype, diagonal, batches)
        result = operator.assemble_dense_from_source_blocks()
        assert result is not None
        assert isinstance(result.matrix, cp.ndarray)
        assert result.matrix.flags.f_contiguous
        tolerance = 4e-6 if dtype == np.complex64 else 3e-14
        np.testing.assert_allclose(
            cp.asnumpy(result.matrix), expected, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(cp.asnumpy(w_live), w)
        np.testing.assert_array_equal(cp.asnumpy(t_live), t)
