from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import (
    CuPyPeriodicCouplingOperator,
    PeriodicCouplingOperator,
    prepare_matvec,
)
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.translation import translation_ab5_table
from pyceles.simulation.solve import _assemble_dense_operator_for_prepared

pytestmark = pytest.mark.gpu


def _small_periodic_case(
    *,
    cache_blocks: bool = False,
    dtype: Any = np.complex128,
    off_plane: bool = False,
    lmax: int = 1,
) -> tuple[PeriodicCouplingOperator, CuPyPeriodicCouplingOperator]:
    k = 2.0 * np.pi / 550.0
    positions = np.asarray(
        [
            [0.0, 0.0, 40.0],
            [210.0, -120.0, 185.0 if off_plane else 40.0],
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


def test_periodic_cupy_coupling_apply_matches_numpy_off_plane_two_particle_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False, off_plane=True)
    rng = np.random.default_rng(18)
    x = rng.normal(size=12) + 1j * rng.normal(size=12)

    got = gpu.apply(cp.asarray(x.astype(np.complex128)))
    want = cpu.apply(x.astype(np.complex128))

    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (np.complex128, 1e-8, 1e-9),
        (np.complex64, 3e-5, 3e-6),
    ],
)
def test_periodic_cupy_shifted_reciprocal_tables_match_numpy_at_lmax3(
    cupy_runtime: tuple[Any, Any], dtype: Any, rtol: float, atol: float
) -> None:
    cp, _ = cupy_runtime
    cpu, gpu = _small_periodic_case(cache_blocks=False, dtype=dtype, off_plane=True, lmax=3)
    rng = np.random.default_rng(20260714)
    size = 2 * gpu.n_modes
    x = rng.normal(size=size) + 1j * rng.normal(size=size)

    got = gpu.apply(cp.asarray(x, dtype=dtype))
    want = cpu.apply(x.astype(dtype))

    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=rtol, atol=atol)


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


def test_periodic_cupy_dense_assembly_from_cached_blocks_matches_matvec(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    spec = PeriodicSpec(
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
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
    )
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[
            pcl.Sphere(position=(0.0, 0.0, 40.0), radius=55.0, refractive_index=1.5 + 0j),
            pcl.Sphere(position=(210.0, -120.0, 185.0), radius=45.0, refractive_index=1.4 + 0j),
        ],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        backend="cupy",
        show_progress=False,
    )
    assert isinstance(prepared.coupling, CuPyPeriodicCouplingOperator)

    n = 2 * 6
    dense = _assemble_dense_operator_for_prepared(
        prepared=prepared,
        A_mv=prepared.apply_A,
        n=n,
        dtype=np.dtype(np.complex128),
        show_progress=False,
    )
    eye = cp.eye(n, dtype=cp.complex128)
    expected = cp.stack([prepared.apply_A(eye[:, j]) for j in range(n)], axis=1)

    np.testing.assert_allclose(cp.asnumpy(dense), cp.asnumpy(expected), rtol=1e-8, atol=1e-9)
    assert prepared.coupling.cache_blocks is False
    assert prepared.coupling._source_block_cache == {}
    assert prepared.coupling._source_block_chunk_cache == {}
