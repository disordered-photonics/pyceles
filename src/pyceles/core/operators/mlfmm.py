from __future__ import annotations

"""Plan-resolution helpers for the pyceles-native MLFMM coupling backend."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .mlfmm_partition import (
    MLFMMPartition,
    build_uniform_mlfmm_partition,
    root_cube,
    uniform_box_coords,
    validate_leaf_size_floor,
)

MLFMMStage = Literal["direct", "single_level", "multilevel"]


@dataclass(frozen=True)
class MLFMMOptions:
    """Expert tuning knobs for pyceles MLFMM depth and leaf-size policy."""

    max_leaf_particles: int = 8
    max_depth: int = 12
    leaf_size_radius_factor: float = 4.0


@dataclass(frozen=True)
class MLFMMResolvedPlan:
    """Structured MLFMM stage/depth decision metadata for diagnostics."""

    stage: MLFMMStage
    selected_depth: int
    depth_from_occupancy: int
    depth_from_size_floor: int
    max_depth: int
    max_leaf_particles: int
    leaf_size_radius_factor: float
    root_side_length: float
    leaf_side_length: float
    max_global_radius: float
    occupied_leaf_count: int
    max_particles_per_leaf: int
    partition: MLFMMPartition


def _validate_positions_and_radii(
    positions: np.ndarray,
    particle_circumscribing_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3). Got {pts.shape}.")
    radii = np.asarray(particle_circumscribing_radii, dtype=float).reshape(-1)
    if radii.size != pts.shape[0]:
        raise ValueError(
            "particle_circumscribing_radii must match positions.shape[0]. "
            f"Got {radii.size} vs {pts.shape[0]}."
        )
    if np.any(radii < 0.0):
        raise ValueError("particle_circumscribing_radii must be non-negative.")
    return pts, radii


def select_mlfmm_stage(depth: int) -> MLFMMStage:
    """Map uniform hierarchy depth to direct, single-level, or multilevel MLFMM."""

    depth_i = int(depth)
    if depth_i <= 1:
        return "direct"
    if depth_i == 2:
        return "single_level"
    return "multilevel"


def _depth_from_occupancy(
    positions: np.ndarray,
    *,
    root_center: np.ndarray,
    root_half_size: float,
    max_leaf_particles: int,
    max_depth: int,
) -> int:
    target = max(1, int(max_leaf_particles))
    max_depth_i = int(max_depth)
    for candidate in range(max_depth_i + 1):
        coords = uniform_box_coords(
            positions,
            root_center=root_center,
            root_half_size=root_half_size,
            depth=candidate,
        )
        if coords.shape[0] == 0:
            return candidate
        _, counts = np.unique(coords, axis=0, return_counts=True)
        if counts.size == 0 or int(np.max(counts)) <= target:
            return candidate
    return max_depth_i


def _depth_from_global_size_floor(
    *,
    root_side_length: float,
    max_global_radius: float,
    leaf_size_radius_factor: float,
    max_depth: int,
) -> int:
    max_depth_i = int(max_depth)
    min_leaf_side = float(leaf_size_radius_factor) * float(max_global_radius)
    depth = max_depth_i
    while depth > 0 and (float(root_side_length) / float(1 << depth)) < min_leaf_side:
        depth -= 1
    return depth


def resolve_mlfmm_plan(
    positions: np.ndarray,
    *,
    particle_circumscribing_radii: np.ndarray,
    options: MLFMMOptions | None = None,
) -> MLFMMResolvedPlan:
    """Resolve the uniform-depth MLFMM plan and build the occupied-box hierarchy."""

    pts, radii = _validate_positions_and_radii(positions, particle_circumscribing_radii)
    resolved_options = MLFMMOptions() if options is None else options

    if int(resolved_options.max_leaf_particles) < 1:
        raise ValueError("MLFMMOptions.max_leaf_particles must be >= 1.")
    if int(resolved_options.max_depth) < 0:
        raise ValueError("MLFMMOptions.max_depth must be >= 0.")
    if float(resolved_options.leaf_size_radius_factor) <= 0.0:
        raise ValueError("MLFMMOptions.leaf_size_radius_factor must be > 0.")

    root_center, root_half_size = root_cube(pts)
    root_side_length = 2.0 * float(root_half_size)
    max_global_radius = float(np.max(radii)) if radii.size else 0.0

    depth_from_occupancy = _depth_from_occupancy(
        pts,
        root_center=root_center,
        root_half_size=root_half_size,
        max_leaf_particles=int(resolved_options.max_leaf_particles),
        max_depth=int(resolved_options.max_depth),
    )
    depth_from_size_floor = _depth_from_global_size_floor(
        root_side_length=root_side_length,
        max_global_radius=max_global_radius,
        leaf_size_radius_factor=float(resolved_options.leaf_size_radius_factor),
        max_depth=int(resolved_options.max_depth),
    )
    selected_depth = min(
        int(depth_from_occupancy),
        int(depth_from_size_floor),
        int(resolved_options.max_depth),
    )
    partition = build_uniform_mlfmm_partition(
        pts,
        particle_circumscribing_radii=radii,
        depth=selected_depth,
    )
    validate_leaf_size_floor(
        partition,
        leaf_size_radius_factor=float(resolved_options.leaf_size_radius_factor),
    )

    occupancies = np.asarray(
        [leaf.particle_indices.size for leaf in partition.leaves],
        dtype=np.int64,
    )
    leaf_side_length = partition.leaves[0].side_length if partition.leaves else root_side_length
    return MLFMMResolvedPlan(
        stage=select_mlfmm_stage(selected_depth),
        selected_depth=int(selected_depth),
        depth_from_occupancy=int(depth_from_occupancy),
        depth_from_size_floor=int(depth_from_size_floor),
        max_depth=int(resolved_options.max_depth),
        max_leaf_particles=int(resolved_options.max_leaf_particles),
        leaf_size_radius_factor=float(resolved_options.leaf_size_radius_factor),
        root_side_length=float(root_side_length),
        leaf_side_length=float(leaf_side_length),
        max_global_radius=float(max_global_radius),
        occupied_leaf_count=int(len(partition.leaves)),
        max_particles_per_leaf=int(np.max(occupancies)) if occupancies.size else 0,
        partition=partition,
    )


__all__ = [
    "MLFMMOptions",
    "MLFMMResolvedPlan",
    "MLFMMStage",
    "resolve_mlfmm_plan",
    "select_mlfmm_stage",
]
