from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles._cupy_memory import cupy_pool_limit_bytes
from pyceles.core.indexing import n_modes
from pyceles.core.particles import LayeredSphere, ParticleCollection, Sphere, spheres_from_arrays
from pyceles.postprocessing.nearfield.classification import classify_internal_points
from pyceles.postprocessing.nearfield.internal import (
    _cupy_internal_pair_batch_size,
    compute_internal_field,
)
from pyceles.postprocessing.nearfield.scattered import (
    compute_scattered_electric_field,
    compute_scattered_field,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("lmax", "compute_dtype", "accum_dtype", "rtol", "atol"),
    [
        (1, np.complex64, np.complex64, 5e-5, 5e-6),
        (1, np.complex64, np.complex128, 5e-5, 5e-6),
        (1, np.complex128, np.complex128, 2e-11, 2e-12),
        (3, np.complex64, np.complex64, 5e-5, 5e-6),
        (3, np.complex64, np.complex128, 5e-5, 5e-6),
        (3, np.complex128, np.complex128, 2e-11, 2e-12),
        (6, np.complex64, np.complex64, 5e-5, 5e-6),
        (6, np.complex64, np.complex128, 5e-5, 5e-6),
        (6, np.complex128, np.complex128, 2e-11, 2e-12),
        # A representative case beyond the former lmax cutoff is enough to
        # cover the high-order fused kernel; lmax=12 LUT guarding is covered
        # separately without paying for another GPU JIT compilation.
        (8, np.complex64, np.complex128, 5e-5, 5e-6),
    ],
)
def test_fused_scattered_near_field_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    lmax: int,
    compute_dtype: Any,
    accum_dtype: Any,
    rtol: float,
    atol: float,
) -> None:
    del cupy_runtime
    rng = np.random.default_rng(1024 + lmax)
    positions = rng.uniform(-300.0, 300.0, size=(5, 3))
    points = rng.uniform(-500.0, 500.0, size=(9, 3)) + np.asarray([0.0, 0.0, 900.0])
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(compute_dtype)
    kwargs: dict[str, Any] = dict(
        k=2.0 * np.pi / 700.0,
        lmax=lmax,
        n_medium=1.2 + 0.03j,
        particle_distance_resolution=0.5,
        show_progress=False,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    e_ref, h_ref = compute_scattered_field(
        points,
        positions,
        coeffs,
        backend="numpy",
        **kwargs,
    )
    e_gpu, h_gpu = compute_scattered_field(
        points,
        positions,
        coeffs,
        backend="cupy",
        **kwargs,
    )
    e_only = compute_scattered_electric_field(
        points,
        positions,
        coeffs,
        backend="cupy",
        **kwargs,
    )

    np.testing.assert_allclose(e_gpu, e_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(e_only, e_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("lmax", [1, 2])
@pytest.mark.parametrize(
    ("compute_dtype", "rtol", "atol"),
    [
        (np.complex64, 5e-5, 5e-6),
        (np.complex128, 2e-11, 2e-12),
    ],
)
def test_batched_sphere_internal_field_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    lmax: int,
    compute_dtype: Any,
    rtol: float,
    atol: float,
) -> None:
    """Exercise the multi-pair CuPy internal-field contraction path."""
    del cupy_runtime
    centers = np.array(
        [[0.0, 0.0, 0.0], [320.0, 0.0, 0.0], [0.0, 320.0, 0.0]],
        dtype=float,
    )
    radii = np.full(centers.shape[0], 100.0)
    refractive_indices = np.array([1.5 + 0.0j, 1.45 + 0.01j, 1.6 + 0.0j])
    local_points = np.array(
        [[10.0, 5.0, 7.0], [-20.0, 12.0, -4.0], [35.0, -15.0, 8.0]],
        dtype=float,
    )
    points = (centers[:, None, :] + local_points[None, :, :]).reshape(-1, 3)
    rng = np.random.default_rng(7300 + lmax)
    coeffs = (
        rng.normal(size=(centers.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(centers.shape[0], n_modes(lmax)))
    ).astype(compute_dtype)
    particles = spheres_from_arrays(
        positions=centers,
        radii=radii,
        refractive_indices=refractive_indices,
    )
    e_ref, h_ref, inside_ref = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        particles=particles,
        backend="numpy",
        n_medium=1.0 + 0.0j,
        show_progress=False,
        compute_dtype=compute_dtype,
        accum_dtype=np.complex128,
    )
    e_gpu, h_gpu, inside_gpu = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        particles=particles,
        backend="cupy",
        n_medium=1.0 + 0.0j,
        show_progress=False,
        compute_dtype=compute_dtype,
        accum_dtype=np.complex128,
    )

    np.testing.assert_array_equal(inside_gpu, inside_ref)
    np.testing.assert_allclose(e_gpu, e_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=rtol, atol=atol)


def test_cupy_internal_field_parity_crosses_real_batch_boundaries(
    cupy_runtime: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check real-device parity when one field spans several pair batches."""
    del cupy_runtime
    import pyceles.postprocessing.nearfield.internal as internal_module

    def tiny_batch(**kwargs: Any) -> int:
        del kwargs
        return 2

    monkeypatch.setattr(internal_module, "_cupy_internal_pair_batch_size", tiny_batch)
    centers = np.array(
        [[0.0, 0.0, 0.0], [320.0, 0.0, 0.0], [0.0, 320.0, 0.0]],
        dtype=float,
    )
    particles = spheres_from_arrays(
        positions=centers,
        radii=np.full(centers.shape[0], 100.0),
        refractive_indices=np.array([1.5 + 0.0j, 1.45 + 0.01j, 1.6 + 0.0j]),
    )
    points = (centers[:, None, :] + np.array([[10.0, 5.0, 7.0]])).reshape(-1, 3)
    rng = np.random.default_rng(7432)
    coeffs = (
        rng.normal(size=(len(particles), n_modes(2)))
        + 1j * rng.normal(size=(len(particles), n_modes(2)))
    ).astype(np.complex64)
    kwargs: dict[str, Any] = dict(
        k=2.0 * np.pi / 550.0,
        lmax=2,
        particles=particles,
        n_medium=1.0 + 0.0j,
        show_progress=False,
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )
    e_ref, h_ref, inside_ref = compute_internal_field(points, coeffs, backend="numpy", **kwargs)
    e_gpu, h_gpu, inside_gpu = compute_internal_field(points, coeffs, backend="cupy", **kwargs)

    np.testing.assert_array_equal(inside_gpu, inside_ref)
    np.testing.assert_allclose(e_gpu, e_ref, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=5e-5, atol=5e-6)


def test_cupy_internal_workspace_planner_does_not_change_pool_limit(
    cupy_runtime: tuple[Any, Any],
) -> None:
    """The internal-field planner queries allocator state without mutating it."""
    cupy, _ = cupy_runtime
    pool = cupy.get_default_memory_pool()
    before = cupy_pool_limit_bytes(pool)

    batch = _cupy_internal_pair_batch_size(
        cupy=cupy,
        total_pairs=100_000,
        lmax=20,
        compute_dtype=np.dtype(np.complex128),
    )

    after = cupy_pool_limit_bytes(pool)
    assert 1 <= batch <= 65_536
    assert after == before


@pytest.mark.parametrize("lmax", [1, 2])
@pytest.mark.parametrize(
    ("compute_dtype", "rtol", "atol"),
    [
        (np.complex64, 5e-5, 5e-6),
        (np.complex128, 2e-11, 2e-12),
    ],
)
def test_batched_layered_internal_field_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    lmax: int,
    compute_dtype: Any,
    rtol: float,
    atol: float,
) -> None:
    """Exercise batched core/shell radial combinations on a real GPU."""
    del cupy_runtime
    centers = np.array(
        [[0.0, 0.0, 0.0], [320.0, 0.0, 0.0], [0.0, 320.0, 0.0]],
        dtype=float,
    )
    particles = ParticleCollection.from_particles(
        [
            LayeredSphere(
                position=tuple(center),
                layer_radii=(40.0, 100.0),
                layer_refractive_indices=(1.8 + 0.0j, 1.35 + 0.02j),
            )
            for center in centers
        ]
    )
    local_points = np.array(
        [[0.0, 0.0, 0.0], [20.0, 5.0, 7.0], [60.0, 12.0, -4.0]],
        dtype=float,
    )
    points = (centers[:, None, :] + local_points[None, :, :]).reshape(-1, 3)
    classification = classify_internal_points(points, particles, n_medium=1.0)
    rng = np.random.default_rng(9100 + lmax)
    coeffs = (
        rng.normal(size=(len(particles), n_modes(lmax)))
        + 1j * rng.normal(size=(len(particles), n_modes(lmax)))
    ).astype(compute_dtype)

    e_ref, h_ref, inside_ref = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        particles=particles,
        _point_classification=classification,
        backend="numpy",
        n_medium=1.0 + 0.0j,
        compute_dtype=compute_dtype,
        accum_dtype=np.complex128,
    )
    e_gpu, h_gpu, inside_gpu = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=lmax,
        particles=particles,
        _point_classification=classification,
        backend="cupy",
        n_medium=1.0 + 0.0j,
        compute_dtype=compute_dtype,
        accum_dtype=np.complex128,
    )

    np.testing.assert_array_equal(inside_gpu, inside_ref)
    np.testing.assert_allclose(e_gpu, e_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=rtol, atol=atol)


def test_mixed_sphere_layered_internal_field_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    """Ensure sphere and layered archetypes compose under CuPy dispatch."""
    del cupy_runtime
    particles = [
        Sphere(position=(0.0, 0.0, 0.0), radius=90.0, refractive_index=1.5 + 0.0j),
        LayeredSphere(
            position=(250.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.8 + 0.0j, 1.35 + 0.02j),
        ),
    ]
    points = np.array(
        [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [250.0, 0.0, 0.0], [300.0, 0.0, 0.0]],
        dtype=float,
    )
    classification = classify_internal_points(points, particles, n_medium=1.0)
    rng = np.random.default_rng(9300)
    coeffs = (rng.normal(size=(2, n_modes(2))) + 1j * rng.normal(size=(2, n_modes(2)))).astype(
        np.complex64
    )

    e_ref, h_ref, inside_ref = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=2,
        particles=particles,
        _point_classification=classification,
        backend="numpy",
        n_medium=1.0 + 0.0j,
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )
    e_gpu, h_gpu, inside_gpu = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=2,
        particles=particles,
        _point_classification=classification,
        backend="cupy",
        n_medium=1.0 + 0.0j,
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )

    np.testing.assert_array_equal(inside_gpu, inside_ref)
    np.testing.assert_allclose(e_gpu, e_ref, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=5e-5, atol=5e-6)
