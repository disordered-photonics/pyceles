from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyceles.core.indexing import index_vswf, n_modes
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
    _layered_radial_factors,
    _radial_internal_point_pairs,
    _regular_radial_factors,
    compute_internal_field,
)


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.5j])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "particle",
    [
        Sphere((0, 0, 0), 72, 2 + 0.1j),
        LayeredSphere((0, 0, 0), (36, 72), (1.8 + 0.05j, 2 + 0.1j)),
        Spheroid((0, 0, 0), 60, 72, 2 + 0.1j),
    ],
    ids=["sphere", "layered", "spheroid"],
)
def test_internal_centre_matches_limits_from_each_cartesian_direction(host, dtype, particle):
    directions = np.vstack((np.eye(3), -np.eye(3)))
    points = np.vstack((np.zeros((1, 3)), -np.zeros((1, 3)), 1e-7 * directions, 1e-14 * directions))
    rng = np.random.default_rng(191)
    coefficients = (
        rng.normal(size=(1, n_modes(2))) + 1j * rng.normal(size=(1, n_modes(2)))
    ).astype(dtype)
    e, h, inside = compute_internal_field(
        points,
        coefficients,
        k=2 * np.pi / 600 * host,
        lmax=2,
        particles=[particle],
        n_medium=host,
        compute_dtype=dtype,
    )
    assert np.all(inside)
    tolerance = 2e-5 if dtype == np.complex64 else 2e-7
    for field in (e, h):
        np.testing.assert_array_equal(field[0], field[1])
        np.testing.assert_allclose(
            field[1:],
            np.broadcast_to(field[0], field[1:].shape),
            rtol=tolerance,
            atol=tolerance * np.linalg.norm(field[0]),
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("tau", [1, 2])
@pytest.mark.parametrize(
    "particle",
    [Sphere((0, 0, 0), 72, 2 + 0.1j), LayeredSphere((0, 0, 0), (36, 72), (1.8 + 0.05j, 2 + 0.1j))],
    ids=["sphere", "layered"],
)
def test_pure_quadrupole_retains_its_nonzero_radius_scaling(dtype, tau, particle):
    """M_l scales as r**l and N_l as r**(l-1) near a regular origin."""
    direction = np.array([1, 2, 3]) / np.sqrt(14)
    points = np.array([1e-13, 1e-14, 0])[:, None] * direction
    coefficients = np.zeros((1, n_modes(2)), dtype=dtype)
    coefficients[0, index_vswf(2, 1, tau, 2)] = 1
    e, h, inside = compute_internal_field(
        points,
        coefficients,
        k=2 * np.pi / 600 * (1.33 + 0.5j),
        lmax=2,
        particles=[particle],
        n_medium=1.33 + 0.5j,
        compute_dtype=dtype,
    )
    assert inside.all()
    tolerance = 2e-5 if dtype == np.complex64 else 2e-13
    for field, order in ((e, 2 if tau == 1 else 1), (h, 1 if tau == 1 else 2)):
        assert np.linalg.norm(field[0]) > 0
        np.testing.assert_allclose(field[1], field[0] * 0.1**order, rtol=tolerance, atol=0)
        np.testing.assert_array_equal(field[2], 0)


@pytest.mark.reference
def test_layered_outgoing_factors_retain_strongly_attenuated_hankel_values():
    z = np.array([2 + 40j])
    zero, one = np.zeros(1), np.ones(1)
    m, n = _layered_radial_factors(
        1,
        z,
        a_m=zero,
        b_m=one,
        a_n=zero,
        b_n=one,
        dtype=np.complex128,
    )
    # Closed forms for h_1 and (z*h_1)' avoid j_1 + i*y_1 cancellation.
    h1 = -np.exp(1j * z) / z * (1 + 1j / z)
    derivative = np.exp(1j * z) * (-1j + 1 / z + 1j / z**2)
    for factors in (m, n):
        np.testing.assert_allclose(factors, (h1, h1 / z, derivative / z), rtol=2e-14, atol=0)


@pytest.mark.parametrize(("l", "z"), [(1, 1e-46 + 2e-47j), (2, 1e-24 + 2e-25j)])
def test_regular_factors_are_normalized_before_single_precision_underflow(l, z):
    argument = np.array([z])
    jl, jl_over_z, derivative_over_z = _regular_radial_factors(l, argument, dtype=np.complex64)
    # j_l ~ z**l/(2*l+1)!! can underflow while its N-wave factors remain finite.
    leading = argument ** (l - 1) / (3 if l == 1 else 15)
    np.testing.assert_array_equal(jl, 0)
    np.testing.assert_allclose(jl_over_z, leading, rtol=2e-7, atol=0)
    np.testing.assert_allclose(derivative_over_z, (l + 1) * leading, rtol=2e-7, atol=0)


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
