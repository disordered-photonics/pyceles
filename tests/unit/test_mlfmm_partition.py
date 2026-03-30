from __future__ import annotations

import numpy as np
import pytest

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
