from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import CuPyPeriodicCouplingOperator, PeriodicCouplingOperator
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.translation import translation_ab5_table

pytestmark = pytest.mark.gpu


def _small_periodic_case(
    *, cache_blocks: bool = False, dtype: Any = np.complex128
) -> tuple[PeriodicCouplingOperator, CuPyPeriodicCouplingOperator]:
    lmax = 1
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [
            [0.0, 0.0, 40.0],
            [210.0, -120.0, 40.0],
        ],
        dtype=float,
    )
    periodic = PeriodicSpec(
        lattice=RectangularLattice2D(ax=900.0, ay=850.0),
        options=PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
            max_shells=4,
            shell_tolerance=1e-9,
        ),
    )
    k_parallel = np.asarray([0.0, 0.0], dtype=float)
    ab5 = translation_ab5_table(lmax, dtype=dtype)
    cpu = PeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=ab5,
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(dtype),
        cache_blocks=cache_blocks,
    )
    gpu = CuPyPeriodicCouplingOperator(
        lmax=lmax,
        k=k,
        positions=positions,
        ab5=ab5,
        periodic=periodic,
        k_parallel=k_parallel,
        dtype=np.dtype(dtype),
        cache_blocks=cache_blocks,
    )
    return cpu, gpu


def test_periodic_cupy_coupling_apply_matches_numpy_same_plane_two_particle_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False)
    rng = np.random.default_rng(12)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)

    got = gpu.apply(x.astype(np.complex128))
    want = cpu.apply(x.astype(np.complex128))
    assert not hasattr(got, "get")
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-8, atol=1e-9)

    got_device = gpu.apply(cp.asarray(x.astype(np.complex128)))
    assert hasattr(got_device, "get")
    np.testing.assert_allclose(cp.asnumpy(got_device), want, rtol=1e-8, atol=1e-9)


def test_periodic_cupy_coupling_cache_apply_parity(cupy_runtime: tuple[Any, Any]) -> None:
    cp, _ = cupy_runtime
    uncached_cpu, uncached_gpu = _small_periodic_case(cache_blocks=False)
    _cpu_cached, cached_gpu = _small_periodic_case(cache_blocks=True)
    rng = np.random.default_rng(25)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)
    x_device = cp.asarray(x.astype(np.complex128))

    uncached = uncached_gpu.apply(x_device)
    cached_first = cached_gpu.apply(x_device)
    assert len(cached_gpu._source_block_cache) == 2
    assert (0, 1) in cached_gpu._source_block_chunk_cache
    cached_second = cached_gpu.apply(x_device)

    want = uncached_cpu.apply(x.astype(np.complex128))
    np.testing.assert_allclose(cp.asnumpy(uncached), want, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(cached_first), want, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(cached_second), want, rtol=1e-8, atol=1e-9)

    # Explicit cache population should be idempotent and keep the same device-owned blocks.
    cached_gpu.populate(show_progress=False)
    assert len(cached_gpu._source_block_cache) == 2
    assert (0, 1) in cached_gpu._source_block_chunk_cache
    np.testing.assert_allclose(cp.asnumpy(cached_gpu.apply(x_device)), want, rtol=1e-8, atol=1e-9)
