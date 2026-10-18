from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.operators import PairwiseCouplingOperator, prepare_matvec
from pyceles.core.operators.mlfmm import (
    MLFMMOptions,
    resolve_mlfmm_plan,
    select_mlfmm_stage,
)
from pyceles.core.operators.mlfmm_partition import (
    build_uniform_mlfmm_partition,
    classify_leaf_pairs,
    validate_leaf_size_floor,
)
from pyceles.core.particles import Sphere


def test_resolve_mlfmm_plan_depth_respects_occupancy_cap() -> None:
    positions = np.asarray(
        [
            [-2.0, -2.0, -2.0],
            [-2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, 2.0, 2.0],
            [2.0, -2.0, -2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 0.1, dtype=float)

    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=4, leaf_size_radius_factor=4.0),
    )

    assert plan.depth_from_occupancy == 1
    assert plan.selected_depth == 1
    assert plan.max_particles_per_leaf == 1


def test_resolve_mlfmm_plan_depth_respects_global_size_floor() -> None:
    positions = np.asarray(
        [
            [-4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 1.1, dtype=float)

    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=4, leaf_size_radius_factor=4.0),
    )

    assert plan.depth_from_occupancy >= 1
    assert plan.depth_from_size_floor == 0
    assert plan.selected_depth == 0
    assert plan.leaf_side_length >= plan.leaf_size_radius_factor * plan.max_global_radius


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (0, "direct"),
        (1, "direct"),
        (2, "single_level"),
        (3, "multilevel"),
        (5, "multilevel"),
    ],
)
def test_select_mlfmm_stage(depth: int, expected: str) -> None:
    assert select_mlfmm_stage(depth) == expected


def test_validate_leaf_size_floor_rejects_too_small_occupied_leaf() -> None:
    positions = np.asarray(
        [
            [-4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 1.1, dtype=float)
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=1,
    )

    with pytest.raises(ValueError, match="leaf size floor violated"):
        validate_leaf_size_floor(partition, leaf_size_radius_factor=4.0)


def test_classify_leaf_pairs_omits_global_far_storage() -> None:
    positions = np.asarray(
        [
            [-4.0, -4.0, -4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 0.1, dtype=float)
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=2,
    )

    near_pairs, far_pairs = classify_leaf_pairs(partition.leaves)
    assert far_pairs == tuple()
    assert (0, 0) in near_pairs
    assert (1, 1) in near_pairs


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_partition_depth_and_ownership_are_independent_of_length_units(scale: float) -> None:
    positions = np.indices((4, 4, 4)).reshape(3, -1).T.astype(float) - 1.5
    radii = np.full(len(positions), 0.1)
    options = MLFMMOptions(max_leaf_particles=2, max_depth=6)
    expected = resolve_mlfmm_plan(positions, particle_circumscribing_radii=radii, options=options)
    actual = resolve_mlfmm_plan(
        positions * scale, particle_circumscribing_radii=radii * scale, options=options
    )
    assert (actual.selected_depth, actual.stage) == (expected.selected_depth, expected.stage)
    np.testing.assert_allclose(actual.root_side_length / scale, expected.root_side_length)
    np.testing.assert_allclose(actual.leaf_side_length / scale, expected.leaf_side_length)
    for a, b in zip(actual.partition.leaves, expected.partition.leaves, strict=True):
        assert a.grid_index == b.grid_index
        np.testing.assert_array_equal(a.particle_indices, b.particle_indices)


@pytest.mark.parametrize("scale", [1.0, 1e-9])
def test_direct_dimer_does_not_need_an_expansion_box_floor(scale: float) -> None:
    positions = scale * np.array([[-1.01, 0.0, 0.0], [1.01, 0.0, 0.0]])
    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=scale * np.ones(2),
        options=MLFMMOptions(max_leaf_particles=1),
    )
    assert plan.stage == "direct"
    assert plan.selected_depth == 0
    particles = [Sphere(tuple(position), scale, 1.5) for position in positions]
    prepared = prepare_matvec(
        lmax=1,
        k=1 / scale,
        particles=particles,
        radial_lut_dr=0.001 * scale,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=1),
    )
    assert isinstance(prepared.coupling, PairwiseCouplingOperator)


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_leaf_size_factor_is_validated_even_for_exact_direct_stages(factor):
    positions = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
    radii = np.ones(2)
    with pytest.raises(ValueError, match="leaf_size_radius_factor"):
        resolve_mlfmm_plan(
            positions,
            particle_circumscribing_radii=radii,
            options=MLFMMOptions(max_depth=0, leaf_size_radius_factor=factor),
        )
    partition = build_uniform_mlfmm_partition(
        positions, particle_circumscribing_radii=radii, depth=0
    )
    with pytest.raises(ValueError, match="leaf_size_radius_factor"):
        validate_leaf_size_floor(partition, leaf_size_radius_factor=factor)


@pytest.mark.parametrize("scale", [1.0, 1e-13])
def test_leaf_size_rejection_does_not_depend_on_physical_units(scale: float) -> None:
    positions = scale * np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    partition = build_uniform_mlfmm_partition(
        positions, particle_circumscribing_radii=np.full(2, 0.4 * scale), depth=1
    )
    with pytest.raises(ValueError, match="leaf size floor violated"):
        validate_leaf_size_floor(partition, leaf_size_radius_factor=4.0)
