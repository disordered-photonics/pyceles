"""Uniform occupied-box hierarchy helpers for pyceles MLFMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MLFMMBox:
    """One occupied box in the uniform MLFMM hierarchy."""

    id: int
    level: int
    grid_index: tuple[int, int, int]
    center: Array
    half_size: float
    particle_indices: Array
    max_radius: float

    @property
    def side_length(self) -> float:
        """Return the Cartesian side length of this occupied cube."""
        return 2.0 * float(self.half_size)


@dataclass(frozen=True)
class MLFMMPartition:
    """Uniform occupied-box hierarchy with explicit occupied leaves and leaf pairs."""

    root_center: Array
    root_half_size: float
    depth: int
    occupied_levels: tuple[tuple[MLFMMBox, ...], ...]
    leaves: tuple[MLFMMBox, ...]
    leaf_near_pairs: tuple[tuple[int, int], ...]
    leaf_far_pairs: tuple[tuple[int, int], ...]

    @property
    def root_side_length(self) -> float:
        """Return the root cube side length."""
        return 2.0 * float(self.root_half_size)


def _validate_positions(positions: Array) -> Array:
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3). Got {pts.shape}.")
    return pts


def _validate_radii(radii: Array, *, n_particles: int) -> Array:
    arr = np.asarray(radii, dtype=float).reshape(-1)
    if arr.size != n_particles:
        raise ValueError(
            "particle_circumscribing_radii must match positions.shape[0]. "
            f"Got {arr.size} vs {n_particles}."
        )
    if np.any(arr < 0.0):
        raise ValueError("particle_circumscribing_radii must be non-negative.")
    return arr


def root_cube(positions: Array) -> tuple[Array, float]:
    """Return the cubic root bounding box that encloses all particle centers."""

    pts = _validate_positions(positions)
    if pts.shape[0] == 0:
        return np.zeros((3,), dtype=float), 0.5

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    half_size = 0.5 * float(np.max(maxs - mins))
    # Keep a positive cube size so depth arithmetic remains well-defined for
    # degenerate or single-particle fixtures.
    return np.asarray(center, dtype=float), max(float(half_size) * (1.0 + 1e-12), 0.5)


def uniform_box_coords(
    positions: Array,
    *,
    root_center: Array,
    root_half_size: float,
    depth: int,
) -> Array:
    """Map particle centers to one uniform-grid box coordinate per level."""

    pts = _validate_positions(positions)
    depth_i = int(depth)
    if depth_i < 0:
        raise ValueError(f"depth must be >= 0. Got {depth}.")
    cells_per_axis = 1 << depth_i
    mins = np.asarray(root_center, dtype=float) - float(root_half_size)
    cell_side = (2.0 * float(root_half_size)) / float(cells_per_axis)
    coords = np.floor((pts - mins[None, :]) / cell_side).astype(np.int64)
    return np.clip(coords, 0, cells_per_axis - 1)


def build_uniform_mlfmm_partition(
    positions: Array,
    *,
    particle_circumscribing_radii: Array,
    depth: int,
) -> MLFMMPartition:
    """Build the uniform occupied-box hierarchy for one selected depth."""

    pts = _validate_positions(positions)
    radii = _validate_radii(particle_circumscribing_radii, n_particles=pts.shape[0])
    depth_i = int(depth)
    if depth_i < 0:
        raise ValueError(f"depth must be >= 0. Got {depth}.")

    root_center, root_half_size = root_cube(pts)
    occupied_levels: list[tuple[MLFMMBox, ...]] = []

    for level in range(depth_i + 1):
        coords = uniform_box_coords(
            pts,
            root_center=root_center,
            root_half_size=root_half_size,
            depth=level,
        )
        cells_per_axis = 1 << level
        cell_side = (2.0 * float(root_half_size)) / float(cells_per_axis)
        half_size = 0.5 * cell_side
        mins = np.asarray(root_center, dtype=float) - float(root_half_size)

        coord_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for particle_index, coord in enumerate(coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            coord_to_indices.setdefault(key, []).append(int(particle_index))

        level_boxes: list[MLFMMBox] = []
        for box_id, key in enumerate(sorted(coord_to_indices)):
            particle_indices = np.asarray(coord_to_indices[key], dtype=np.int64)
            center = mins + (np.asarray(key, dtype=float) + 0.5) * cell_side
            level_boxes.append(
                MLFMMBox(
                    id=box_id,
                    level=level,
                    grid_index=key,
                    center=np.asarray(center, dtype=float),
                    half_size=float(half_size),
                    particle_indices=particle_indices,
                    max_radius=float(np.max(radii[particle_indices]))
                    if particle_indices.size
                    else 0.0,
                )
            )
        occupied_levels.append(tuple(level_boxes))

    leaves = occupied_levels[-1] if occupied_levels else tuple()
    leaf_near_pairs, leaf_far_pairs = classify_leaf_pairs(leaves)
    return MLFMMPartition(
        root_center=np.asarray(root_center, dtype=float),
        root_half_size=float(root_half_size),
        depth=depth_i,
        occupied_levels=tuple(occupied_levels),
        leaves=tuple(leaves),
        leaf_near_pairs=leaf_near_pairs,
        leaf_far_pairs=leaf_far_pairs,
    )


def classify_leaf_pairs(
    leaves: tuple[MLFMMBox, ...] | list[MLFMMBox],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return sparse leaf-near pairs and omit explicit far-pair materialization.

    For large occupied trees, storing all separated leaf pairs is quadratic in
    occupied-leaf count and can dominate host memory before preparation starts.
    We therefore enumerate only Chebyshev-near pairs via a 27-neighbor stencil
    over occupied grid keys and keep `leaf_far_pairs` empty by design.
    """

    leaves_seq = tuple(leaves)
    if not leaves_seq:
        return tuple(), tuple()

    coord_to_leaf: dict[tuple[int, int, int], int] = {}
    for i, leaf in enumerate(leaves_seq):
        coord_to_leaf[leaf.grid_index] = int(i)

    near_pairs: list[tuple[int, int]] = []
    for i, leaf in enumerate(leaves_seq):
        xi, yi, zi = leaf.grid_index
        near_pairs.append((i, i))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    j = coord_to_leaf.get((xi + dx, yi + dy, zi + dz))
                    if j is None or int(j) <= i:
                        continue
                    near_pairs.append((i, int(j)))

    # Far pairs are grouped later where needed (single-level only), avoiding
    # global quadratic pair storage in the partition object.
    return tuple(near_pairs), tuple()


def validate_leaf_size_floor(
    partition: MLFMMPartition,
    *,
    leaf_size_radius_factor: float = 4.0,
) -> None:
    """Reject occupied leaves that are too small for local circumscribing radii."""

    factor = float(leaf_size_radius_factor)
    if factor <= 0.0:
        raise ValueError(f"leaf_size_radius_factor must be > 0. Got {leaf_size_radius_factor!r}.")

    for leaf in partition.leaves:
        min_side = factor * float(leaf.max_radius)
        if leaf.side_length + 1e-12 < min_side:
            raise ValueError(
                "Uniform MLFMM leaf size floor violated for occupied leaf "
                f"{leaf.grid_index} at depth {leaf.level}: "
                f"leaf_side={leaf.side_length:.6g} < {factor:.6g} * "
                f"max_radius_in_leaf={leaf.max_radius:.6g}."
            )


__all__ = [
    "MLFMMBox",
    "MLFMMPartition",
    "build_uniform_mlfmm_partition",
    "classify_leaf_pairs",
    "root_cube",
    "uniform_box_coords",
    "validate_leaf_size_floor",
]
