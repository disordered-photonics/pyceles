from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.particles import (
    LayeredSphere,
    ParticleCollection,
    Sphere,
    Spheroid,
    spheres_from_arrays,
)
from pyceles.postprocessing.nearfield.classification import InternalPointClassification
from pyceles.postprocessing.nearfield.internal import (
    _cupy_internal_pair_batch_size,
    _internal_cupy_workspace_bytes_per_pair,
    _internal_pair_batch_size_for_workspace,
    _radial_internal_point_pairs,
    compute_internal_field,
)


def test_internal_cupy_pair_workspace_planner_scales_with_order_and_dtype() -> None:
    low_order = _internal_cupy_workspace_bytes_per_pair(
        lmax=2, compute_dtype=np.dtype(np.complex128)
    )
    high_order_64 = _internal_cupy_workspace_bytes_per_pair(
        lmax=20, compute_dtype=np.dtype(np.complex64)
    )
    high_order_128 = _internal_cupy_workspace_bytes_per_pair(
        lmax=20, compute_dtype=np.dtype(np.complex128)
    )

    assert high_order_128 > high_order_64 > low_order
    assert high_order_128 == 2 * high_order_64
    assert high_order_128 * 65_536 > 10 * 1024**3

    workspace = 3 * 1024**3
    assert (
        _internal_pair_batch_size_for_workspace(
            total_pairs=100_000,
            lmax=2,
            compute_dtype=np.dtype(np.complex128),
            workspace_bytes=workspace,
        )
        == 65_536
    )
    high_order_batch = _internal_pair_batch_size_for_workspace(
        total_pairs=100_000,
        lmax=20,
        compute_dtype=np.dtype(np.complex128),
        workspace_bytes=workspace,
    )
    assert 8_192 < high_order_batch < 16_384


def test_internal_cupy_pair_batch_uses_current_guarded_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyceles.postprocessing.nearfield.internal as internal_module

    snapshot = SimpleNamespace(
        raw_free_bytes=6 * 1024**3,
        pool_free_bytes=1 * 1024**3,
        active_headroom_bytes=5 * 1024**3,
    )
    apply_pool_limit_calls: list[bool] = []

    def fake_snapshot(_cupy: object, *, apply_pool_limit: bool) -> SimpleNamespace:
        apply_pool_limit_calls.append(bool(apply_pool_limit))
        return snapshot

    monkeypatch.setattr(internal_module, "cupy_allocator_snapshot", fake_snapshot)
    expected_workspace = int(0.5 * snapshot.active_headroom_bytes)
    expected = _internal_pair_batch_size_for_workspace(
        total_pairs=100_000,
        lmax=20,
        compute_dtype=np.dtype(np.complex128),
        workspace_bytes=expected_workspace,
    )

    actual = _cupy_internal_pair_batch_size(
        cupy=object(),
        total_pairs=100_000,
        lmax=20,
        compute_dtype=np.dtype(np.complex128),
    )

    assert actual == expected
    assert actual < 65_536
    assert apply_pool_limit_calls == [False]


def test_numpy_complex64_internal_modes_stay_in_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent float64 geometry from silently promoting NumPy mode algebra."""
    import pyceles.postprocessing.nearfield.internal as internal_module

    original = internal_module.build_internal_mode_tensors
    observed: list[tuple[np.dtype, np.dtype]] = []

    def checked(*args: Any, **kwargs: Any) -> Any:
        m_modes, n_modes_batch = original(*args, **kwargs)
        observed.append((m_modes.dtype, n_modes_batch.dtype))
        return m_modes, n_modes_batch

    monkeypatch.setattr(internal_module, "build_internal_mode_tensors", checked)
    particles = ParticleCollection.from_particles(
        [
            Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0.02j),
            LayeredSphere(
                position=(300.0, 0.0, 0.0),
                layer_radii=(40.0, 100.0),
                layer_refractive_indices=(1.8 + 0.0j, 1.3 + 0.02j),
            ),
            Spheroid(
                position=(600.0, 0.0, 0.0),
                equatorial_radius=70.0,
                polar_radius=100.0,
                refractive_index=1.5 + 0.02j,
            ),
        ]
    )
    compute_internal_field(
        np.asarray([[25.0, 7.0, 3.0], [360.0, 4.0, 2.0], [610.0, 0.0, 5.0]]),
        np.ones((len(particles), n_modes(1)), dtype=np.complex64),
        k=2.0 * np.pi / 550.0,
        lmax=1,
        particles=particles,
        backend="numpy",
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )

    assert observed
    assert all(m_dtype == np.dtype(np.complex64) for m_dtype, _ in observed)
    assert all(n_dtype == np.dtype(np.complex64) for _, n_dtype in observed)


def test_radial_internal_pair_mapping_handles_unsorted_subset() -> None:
    classification = InternalPointClassification.from_active_points(
        n_particles=5,
        inside_any=np.ones(4, dtype=bool),
        entries=(
            (0, np.array([1], dtype=np.intp)),
            (2, np.array([0, 3], dtype=np.intp)),
            (4, np.array([2], dtype=np.intp)),
        ),
    )

    pair_points, pair_spheres = _radial_internal_point_pairs(
        np.zeros((4, 3), dtype=float),
        np.zeros((2, 3), dtype=float),
        np.ones(2, dtype=float),
        classification=classification,
        particle_indices=np.array([4, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(pair_points, np.array([0, 3, 2]))
    np.testing.assert_array_equal(pair_spheres, np.array([1, 1, 0]))


def test_sphere_internal_ratios_are_reused_for_identical_materials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyceles.postprocessing.nearfield.internal as internal_module

    original = internal_module.sphere_internal_ratios
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(internal_module, "sphere_internal_ratios", counted)
    positions = np.column_stack((np.arange(4, dtype=float) * 250.0, np.zeros(4), np.zeros(4)))
    particles = spheres_from_arrays(
        positions=positions,
        radii=np.full(4, 100.0),
        refractive_indices=np.full(4, 1.5 + 0.0j),
    )
    points = positions + np.array([70.0, 0.0, 0.0])
    coeffs = np.ones((len(particles), n_modes(1)), dtype=np.complex128)

    _, _, inside = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        particles=particles,
        n_medium=1.0 + 0.0j,
        backend="numpy",
    )

    np.testing.assert_array_equal(inside, np.ones(points.shape[0], dtype=bool))
    assert calls == 1


def test_layered_internal_ratios_are_reused_per_archetype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyceles.postprocessing.nearfield.internal as internal_module

    original = internal_module.layered_internal_ab_ratios
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(internal_module, "layered_internal_ab_ratios", counted)
    profiles = (
        ((40.0, 100.0), (1.8 + 0.0j, 1.3 + 0.02j)),
        ((45.0, 100.0), (1.7 + 0.0j, 1.4 + 0.01j)),
    )
    particles = ParticleCollection.from_particles(
        [
            LayeredSphere(
                position=(250.0 * index, 0.0, 0.0),
                layer_radii=profiles[index // 2][0],
                layer_refractive_indices=profiles[index // 2][1],
            )
            for index in range(4)
        ]
    )
    points = particles.positions + np.array([70.0, 0.0, 0.0])
    coeffs = np.ones((len(particles), n_modes(1)), dtype=np.complex128)

    _, _, inside = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        particles=particles,
        n_medium=1.0 + 0.0j,
    )

    np.testing.assert_array_equal(inside, np.ones(points.shape[0], dtype=bool))
    assert particles.n_archetypes == 2
    assert calls == particles.n_archetypes


def test_spheroid_archetype_cache_preserves_instance_position() -> None:
    spheroid = Spheroid(
        position=(300.0, 0.0, 0.0),
        equatorial_radius=50.0,
        polar_radius=100.0,
        refractive_index=1.5 + 0.0j,
    )
    points = np.array([[300.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    coeffs = np.zeros((1, n_modes(1)), dtype=np.complex128)

    e_field, h_field, inside = compute_internal_field(
        points,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        particles=(spheroid,),
        n_medium=1.0 + 0.0j,
    )

    np.testing.assert_array_equal(inside, np.array([True, False]))
    np.testing.assert_allclose(e_field, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(h_field, 0.0, rtol=0.0, atol=0.0)
