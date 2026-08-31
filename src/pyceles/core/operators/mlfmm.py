"""Plan-resolution helpers for the pyceles-native MLFMM coupling backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import scipy.sparse
from scipy.special import spherical_jn, spherical_yn
from tqdm.auto import tqdm

from pyceles.core.indexing import n_modes
from pyceles.core.translation import (
    RadialLUT,
    translation_ab5_table,
    translation_block,
    translation_block_rect,
)

from .base import CouplingOperator
from .coupling_pairwise import PairwiseCouplingOperator
from .mlfmm_directional import (
    MLFMMDirectionalInterpolation,
    MLFMMDirectionalStructuredTransforms,
    MLFMMDirectionalTransformData,
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_interpolation,
    directional_to_box_regular,
    structured_directional_transforms,
)
from .mlfmm_partition import (
    MLFMMBox,
    MLFMMPartition,
    build_uniform_mlfmm_partition,
    root_cube,
    uniform_box_coords,
    validate_leaf_size_floor,
)

MLFMMStage = Literal["direct", "single_level", "multilevel"]
MLFMMLeafApplyMode = Literal["dense", "on_the_fly"]
COMPLEX128_DTYPE = np.dtype(np.complex128)
_ROKHLIN_MINIMUM_ORDERS = (3, 7, 11, 17, 24, 30)
"""Conservative minimum truncation orders for discrete accuracy levels,
inspired from the heuristic values used in FasTMM.

These floors stabilize the small-ka regime where the asymptotic Rokhlin-style
estimate underpredicts the required order. The values are inherited from the
validated reference implementation used during pyceles MLFMM development.
"""


@dataclass(frozen=True)
class MLFMMOptions:
    """Expert tuning knobs for pyceles MLFMM partition resolution.

    These options cover the current public MLFMM policy surface: how many
    particles are allowed in one leaf, how deep the occupied tree may grow,
    how large a leaf box must remain relative to the largest circumscribing
    radius it contains, how aggressively the Rokhlin-style box-order estimate
    is padded, and where sampled high-frequency multilevel staging starts.
    """

    max_leaf_particles: int = 8
    max_depth: int = 12
    leaf_size_radius_factor: float = 4.0
    accuracy_level: int = 3
    order_additive: int = 2
    hf_start_level: int | None = None
    hf_wavelength_divisor: float = 5.0
    collect_stream_stats: bool = False


@dataclass(frozen=True)
class MLFMMResolvedPlan:
    """Structured runtime metadata for the resolved MLFMM stage and partition."""

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

    def summary(self) -> dict[str, int | float | str]:
        """Return a compact stage/partition summary in the canonical MLFMM vocabulary."""

        return {
            "stage": str(self.stage),
            "selected_depth": int(self.selected_depth),
            "depth_from_occupancy": int(self.depth_from_occupancy),
            "depth_from_size_floor": int(self.depth_from_size_floor),
            "occupied_leaf_count": int(self.occupied_leaf_count),
            "max_particles_per_leaf": int(self.max_particles_per_leaf),
            "root_side_length": float(self.root_side_length),
            "leaf_side_length": float(self.leaf_side_length),
            "max_global_radius": float(self.max_global_radius),
        }


@dataclass(frozen=True)
class MLFMMLeafApplyGroup:
    """Grouped leaf apply payload with uniform occupancy.

    NumPy uses these grouped work units for both dense reusable leaf operators
    and the reference on-the-fly path. Exactly one of `aggregation` or
    `pair_deltas` is populated.
    """

    occupancy: int
    nmodes: int
    leaf_ids: np.ndarray
    particle_indices: np.ndarray
    aggregation: np.ndarray | None = None
    pair_deltas: np.ndarray | None = None


@dataclass(frozen=True)
class MLFMMSingleLevelOperators:
    """Prepared single-level HF far operators over one occupied leaf level."""

    partition: MLFMMPartition
    box_order: int
    translator_order: int
    grid_order: int
    directional: MLFMMDirectionalTransformData
    leaf_cell_coords: dict[int, tuple[int, int, int]]
    far_offset_batches: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[tuple[int, int, int], np.ndarray]
    aggregation: tuple[np.ndarray, ...] = tuple()
    receive: tuple[np.ndarray, ...] = tuple()
    leaf_groups: tuple[MLFMMLeafApplyGroup, ...] = tuple()
    leaf_apply_mode: MLFMMLeafApplyMode = "dense"


@dataclass(frozen=True)
class MLFMMLevelOperators:
    """Prepared sampled HF data for one occupied multilevel hierarchy level."""

    level: int
    coords: np.ndarray
    centers: np.ndarray
    parent_indices: np.ndarray
    children: tuple[np.ndarray, ...]
    box_order: int
    translator_order: int
    grid_order: int
    directional: MLFMMDirectionalTransformData
    far_offset_batches: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[tuple[int, int, int], np.ndarray]


@dataclass(frozen=True)
class MLFMMTransferOperators:
    """Sampled parent/child transfer operators between adjacent occupied levels."""

    child_level: int
    parent_level: int
    interpolation: MLFMMDirectionalInterpolation
    anterpolation: MLFMMDirectionalInterpolation
    batches_by_shift: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]
    phase_up_by_shift: dict[tuple[int, int, int], np.ndarray]
    phase_down_by_shift: dict[tuple[int, int, int], np.ndarray]


@dataclass(frozen=True)
class MLFMMMultilevelOperators:
    """Prepared sampled multilevel HF operators over occupied boxes only."""

    partition: MLFMMPartition
    levels: tuple[MLFMMLevelOperators, ...]
    transfers: tuple[MLFMMTransferOperators, ...]
    leaf_level: int
    hf_start_level: int
    hf_end_level: int
    aggregation: tuple[np.ndarray, ...] = tuple()
    receive: tuple[np.ndarray, ...] = tuple()
    leaf_groups: tuple[MLFMMLeafApplyGroup, ...] = tuple()
    leaf_apply_mode: MLFMMLeafApplyMode = "dense"


@dataclass
class MLFMMCouplingOperator:
    """Prepared NumPy MLFMM coupling operator with structured stage metadata.

    The near part stays exact on the resolved leaf partition. The far part is
    applied through either a single occupied-leaf sampled level or a multilevel
    occupied-box hierarchy, depending on the resolved stage.

    Precision policy:
    - `near_dtype` follows the requested operator compute precision.
    - `far_dtype` is fixed to complex128 for the sampled-far hierarchy.
    - `dtype` controls the public coupling output dtype.
    """

    lmax: int
    k: float
    positions: np.ndarray
    radial_lut: RadialLUT
    resolved_plan: MLFMMResolvedPlan
    dtype: np.dtype = COMPLEX128_DTYPE
    near_dtype: np.dtype = COMPLEX128_DTYPE
    far_dtype: np.dtype = COMPLEX128_DTYPE
    cache_translation_blocks: bool = False
    single_level: MLFMMSingleLevelOperators | None = None
    multilevel: MLFMMMultilevelOperators | None = None
    _exact_block_cache: dict[tuple[int, int], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.resolved_plan.stage == "single_level" and self.single_level is None:
            raise ValueError("single_level operators are required for single-level MLFMM coupling.")
        if self.resolved_plan.stage == "multilevel" and self.multilevel is None:
            raise ValueError("multilevel operators are required for multilevel MLFMM coupling.")
        if self.resolved_plan.stage == "direct":
            raise ValueError("MLFMMCouplingOperator is not used for the direct stage.")
        if np.dtype(self.far_dtype) != np.dtype(np.complex128):
            raise ValueError(
                "MLFMM sampled-far path requires `far_dtype=complex128` for stability."
            )
        if self.cache_translation_blocks and self._exact_block_cache is None:
            self._exact_block_cache = {}

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the prepared MLFMM coupling operator `W`."""

        if self.resolved_plan.stage == "single_level":
            if self.single_level is None:
                raise RuntimeError("Internal error: single-level operators are missing.")
            near_dtype = np.dtype(self.near_dtype)
            far_dtype = np.dtype(self.far_dtype)
            y_near, y_far = apply_single_level_mlfmm(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                operators=self.single_level,
                radial_lut=self.radial_lut,
                near_dtype=near_dtype,
                far_dtype=far_dtype,
                block_cache=self._exact_block_cache,
            )
            y_total = np.asarray(y_near, dtype=far_dtype) + np.asarray(y_far, dtype=far_dtype)
            return np.asarray(y_total, dtype=self.dtype)
        if self.resolved_plan.stage == "multilevel":
            if self.multilevel is None:
                raise RuntimeError("Internal error: multilevel operators are missing.")
            near_dtype = np.dtype(self.near_dtype)
            far_dtype = np.dtype(self.far_dtype)
            y_near, y_far = apply_multilevel_mlfmm(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                operators=self.multilevel,
                radial_lut=self.radial_lut,
                near_dtype=near_dtype,
                far_dtype=far_dtype,
                block_cache=self._exact_block_cache,
            )
            y_total = np.asarray(y_near, dtype=far_dtype) + np.asarray(y_far, dtype=far_dtype)
            return np.asarray(y_total, dtype=self.dtype)
        raise RuntimeError(f"Unsupported MLFMM stage {self.resolved_plan.stage!r}.")

    def populate(self, *, show_progress: bool = False) -> None:
        """Optionally precompute exact near blocks used by the current partition."""

        del show_progress
        if not self.cache_translation_blocks:
            return
        if self._exact_block_cache is None:
            self._exact_block_cache = {}
        _populate_exact_leaf_near_cache(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            partition=self.resolved_plan.partition,
            radial_lut=self.radial_lut,
            block_cache=self._exact_block_cache,
        )

    def plan_summary(self) -> dict[str, int | float | str]:
        """Return the canonical resolved-plan summary for this prepared operator."""

        return self.resolved_plan.summary()

    def hierarchy_diagnostics(self) -> dict[str, object]:
        """Return stage-aware hierarchy diagnostics using the CuPy-aligned vocabulary.

        The NumPy path remains free to execute with different storage choices,
        but it should describe resolved MLFMM structure in the same conceptual
        terms as the CuPy path so backend comparisons stay interpretable.
        """

        stage = str(self.resolved_plan.stage)
        if stage == "single_level":
            if self.single_level is None:
                raise RuntimeError("Internal error: single-level operators are missing.")
            directional = self.single_level.directional
            return {
                "stage": stage,
                "leaf_level": int(self.resolved_plan.selected_depth),
                "hf_start_level": int(self.resolved_plan.selected_depth),
                "hf_end_level": int(self.resolved_plan.selected_depth),
                "transfer_edges": [],
                "levels": {
                    "n_levels": 1,
                    "translator_orders": [int(self.single_level.translator_order)],
                    "grid_orders": [int(self.single_level.grid_order)],
                    "direction_counts": [int(directional.grid.directions.shape[0])],
                    "levels": [
                        {
                            "level": int(self.resolved_plan.selected_depth),
                            "n_boxes": len(self.single_level.leaf_cell_coords),
                            "box_order": int(self.single_level.box_order),
                            "translator_order": int(self.single_level.translator_order),
                            "grid_order": int(self.single_level.grid_order),
                            "n_directions": int(directional.grid.directions.shape[0]),
                            "parity_from_hf_start": 0,
                        }
                    ],
                },
            }
        if stage == "multilevel":
            if self.multilevel is None:
                raise RuntimeError("Internal error: multilevel operators are missing.")
            levels = self.multilevel.levels
            hf_start_level = int(self.multilevel.hf_start_level)
            return {
                "stage": stage,
                "leaf_level": int(self.multilevel.leaf_level),
                "hf_start_level": int(self.multilevel.hf_start_level),
                "hf_end_level": int(self.multilevel.hf_end_level),
                "transfer_edges": [
                    {
                        "child_level": int(transfer.child_level),
                        "parent_level": int(transfer.parent_level),
                    }
                    for transfer in self.multilevel.transfers
                ],
                "levels": {
                    "n_levels": len(levels),
                    "translator_orders": [int(level.translator_order) for level in levels],
                    "grid_orders": [int(level.grid_order) for level in levels],
                    "direction_counts": sorted(
                        {int(level.directional.grid.directions.shape[0]) for level in levels}
                    ),
                    "levels": [
                        {
                            "level": int(level.level),
                            "n_boxes": int(level.coords.shape[0]),
                            "box_order": int(level.box_order),
                            "translator_order": int(level.translator_order),
                            "grid_order": int(level.grid_order),
                            "n_directions": int(level.directional.grid.directions.shape[0]),
                            "parity_from_hf_start": int((int(level.level) - hf_start_level) % 2),
                        }
                        for level in levels
                    ],
                },
            }
        raise RuntimeError(f"Unsupported MLFMM stage {self.resolved_plan.stage!r}.")

    def memory_diagnostics(self) -> dict[str, object]:
        """Return stage-aware NumPy memory and state diagnostics."""

        stage = str(self.resolved_plan.stage)
        if stage == "single_level":
            if self.single_level is None:
                raise RuntimeError("Internal error: single-level operators are missing.")
            return _single_level_memory_diagnostics(
                lmax=int(self.lmax),
                n_particles=int(self.positions.shape[0]),
                near_dtype=np.dtype(self.near_dtype),
                far_dtype=np.dtype(self.far_dtype),
                plan_summary=self.plan_summary(),
                operators=self.single_level,
            )
        if stage == "multilevel":
            if self.multilevel is None:
                raise RuntimeError("Internal error: multilevel operators are missing.")
            return _multilevel_memory_diagnostics(
                lmax=int(self.lmax),
                n_particles=int(self.positions.shape[0]),
                near_dtype=np.dtype(self.near_dtype),
                far_dtype=np.dtype(self.far_dtype),
                plan_summary=self.plan_summary(),
                operators=self.multilevel,
            )
        raise RuntimeError(f"Unsupported MLFMM stage {self.resolved_plan.stage!r}.")


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
    """Map resolved depth to direct, single-level, or multilevel MLFMM.

    Depth 0-1 stays on the exact pairwise path, depth 2 activates one occupied
    leaf level, and deeper trees activate the multilevel high-frequency
    hierarchy.
    """

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
    """Resolve the MLFMM stage/depth policy and build the occupied-box hierarchy.

    The selected depth is the shallower of the occupancy-driven depth and the
    circumscribing-radius leaf-size floor, so exact-near and sampled-far work
    on the same valid partition.
    """

    pts, radii = _validate_positions_and_radii(positions, particle_circumscribing_radii)
    resolved_options = MLFMMOptions() if options is None else options

    if int(resolved_options.max_leaf_particles) < 1:
        raise ValueError("MLFMMOptions.max_leaf_particles must be >= 1.")
    if int(resolved_options.max_depth) < 0:
        raise ValueError("MLFMMOptions.max_depth must be >= 0.")
    if float(resolved_options.leaf_size_radius_factor) <= 0.0:
        raise ValueError("MLFMMOptions.leaf_size_radius_factor must be > 0.")
    if int(resolved_options.accuracy_level) < 1:
        raise ValueError("MLFMMOptions.accuracy_level must be >= 1.")

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
        occupied_leaf_count=len(partition.leaves),
        max_particles_per_leaf=int(np.max(occupancies)) if occupancies.size else 0,
        partition=partition,
    )


def estimate_rokhlin_order(ka: float, *, accuracy: int = 3) -> int:
    """Estimate the HF box order from size parameter `ka`.

    The estimate uses a Rokhlin-style asymptotic growth law together with
    conservative small-ka floor values. The `accuracy` argument is an ordinal
    quality level, not a guaranteed number of correct decimal digits.
    """

    acc = int(accuracy)
    if acc < 1 or acc > len(_ROKHLIN_MINIMUM_ORDERS):
        raise ValueError(
            f"accuracy must be in [1, {len(_ROKHLIN_MINIMUM_ORDERS)}]. Got {accuracy!r}."
        )
    ka_abs = abs(float(ka))
    if ka_abs <= 0.0:
        return int(_ROKHLIN_MINIMUM_ORDERS[acc - 1])
    order = math.floor(ka_abs + 1.8 * (acc ** (2.0 / 3.0)) * (ka_abs ** (1.0 / 3.0)))
    return int(max(order, _ROKHLIN_MINIMUM_ORDERS[acc - 1]))


def box_order_rokhlin_like(
    *,
    particle_lmax: int,
    k: float,
    box_half_size: float,
    accuracy: int = 3,
    additive: int = 2,
) -> int:
    """Return a grouped box order from the level side length.

    This keeps one shared order per occupied level and starts from the neutral
    Rokhlin-style estimate used by the NumPy MLFMM path, while never dropping
    below the particle multipole order already present in that box.
    """

    side_length = 2.0 * float(box_half_size)
    base = estimate_rokhlin_order(abs(float(k)) * side_length, accuracy=int(accuracy)) + int(
        additive
    )
    return int(max(int(particle_lmax), base))


def _leaf_cell_coords(partition: MLFMMPartition) -> dict[int, tuple[int, int, int]]:
    """Return integer leaf cell coordinates for one uniform occupied leaf level."""

    if not partition.leaves:
        return {}
    cell_size = 2.0 * float(partition.leaves[0].half_size)
    mins = np.asarray(partition.root_center, dtype=float) - float(partition.root_half_size)
    out: dict[int, tuple[int, int, int]] = {}
    for leaf in partition.leaves:
        scaled = np.rint((np.asarray(leaf.center) - mins) / cell_size - 0.5).astype(np.int64)
        out[int(leaf.id)] = (int(scaled[0]), int(scaled[1]), int(scaled[2]))
    return out


def _leaf_offset_batches(
    partition: MLFMMPartition,
    cell_coords: dict[int, tuple[int, int, int]],
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Group directed far leaf interactions by relative offset.

    Partition construction intentionally omits explicit global far-pair storage
    to avoid quadratic host memory. Single-level plans have at most 64 leaves,
    so we rebuild far offsets locally from leaf coordinates here.
    """

    if not partition.leaves:
        return {}

    leaf_ids = np.asarray([int(leaf.id) for leaf in partition.leaves], dtype=np.int64)
    coords = np.asarray([cell_coords[int(leaf_id)] for leaf_id in leaf_ids], dtype=np.int64)
    local_batches = _coords_far_offset_batches(coords)
    return {
        offset: (
            np.ascontiguousarray(leaf_ids[np.asarray(src_local, dtype=np.int64)], dtype=np.int64),
            np.ascontiguousarray(leaf_ids[np.asarray(dst_local, dtype=np.int64)], dtype=np.int64),
        )
        for offset, (src_local, dst_local) in local_batches.items()
    }


def _coords_to_centers(
    coords: np.ndarray,
    *,
    root_center: np.ndarray,
    root_half_size: float,
    level: int,
) -> np.ndarray:
    """Map integer box coordinates at one level to physical centers."""

    mins = np.asarray(root_center, dtype=float) - float(root_half_size)
    cell_size = (2.0 * float(root_half_size)) / float(1 << int(level))
    return mins[None, :] + (np.asarray(coords, dtype=float) + 0.5) * cell_size


def _coords_far_offset_batches(
    coords: np.ndarray,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Group directed far same-level occupied-box interactions by relative offset."""

    coords_arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for i in range(coords_arr.shape[0]):
        for j in range(i + 1, coords_arr.shape[0]):
            delta = coords_arr[j] - coords_arr[i]
            if int(np.max(np.abs(delta))) <= 1:
                continue
            offset = (int(delta[0]), int(delta[1]), int(delta[2]))
            grouped.setdefault(offset, []).append((i, j))
            reverse_offset = (-offset[0], -offset[1], -offset[2])
            grouped.setdefault(reverse_offset, []).append((j, i))
    return {
        offset: (
            np.asarray([pair[0] for pair in pairs], dtype=np.int64),
            np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        )
        for offset, pairs in grouped.items()
    }


def _coords_near_neighbors(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return packed same-level occupied-box neighbors within one Chebyshev cell.

    The output is `(offsets, flat_indices)` where neighbors for destination box
    `i` are stored in `flat_indices[offsets[i]:offsets[i+1]]`.
    """

    coords_arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    n_boxes = int(coords_arr.shape[0])
    offsets = np.zeros((n_boxes + 1,), dtype=np.int32)
    if n_boxes == 0:
        return offsets, np.zeros((0,), dtype=np.int32)

    lookup = {
        (int(coord[0]), int(coord[1]), int(coord[2])): int(idx)
        for idx, coord in enumerate(coords_arr)
    }
    flat_rows: list[np.ndarray] = []
    cursor = 0
    for dst, coord in enumerate(coords_arr):
        cx, cy, cz = int(coord[0]), int(coord[1]), int(coord[2])
        row: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    src = lookup.get((cx + dx, cy + dy, cz + dz))
                    if src is not None:
                        row.append(int(src))
        row_arr = np.asarray(sorted(row), dtype=np.int32)
        flat_rows.append(row_arr)
        cursor += int(row_arr.size)
        offsets[dst + 1] = int(cursor)
    flat = (
        np.concatenate(flat_rows, dtype=np.int32) if flat_rows else np.zeros((0,), dtype=np.int32)
    )
    return offsets, flat


def _build_multilevel_far_offset_batches(
    *,
    coords: np.ndarray,
    parent_indices: np.ndarray,
    near_neighbors: tuple[np.ndarray, np.ndarray],
    parent_near_neighbors: tuple[np.ndarray, np.ndarray],
    children_by_parent: tuple[np.ndarray, ...],
    progress_desc: str | None = None,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    """Group multilevel far pairs owned by this level only.

    Ownership follows the validated high-frequency hierarchy policy: a same-level
    interaction is handled at this level only when the source and target boxes
    are not near neighbors themselves, but their parents are near neighbors.
    This avoids double-counting the far field across multiple hierarchy levels.
    """

    coords_arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords_arr.shape[0] == 0:
        return {}

    near_offsets, near_flat = near_neighbors
    parent_near_offsets, parent_near_flat = parent_near_neighbors
    grouped: dict[tuple[int, int, int], list[list[int]]] = {}
    dst_iter = (
        tqdm(range(coords_arr.shape[0]), desc=progress_desc, unit="box")
        if progress_desc is not None
        else range(coords_arr.shape[0])
    )
    for dst in dst_iter:
        parent = int(parent_indices[dst])
        if parent < 0:
            continue
        near_set = set(
            int(v) for v in near_flat[int(near_offsets[dst]) : int(near_offsets[dst + 1])].tolist()
        )
        candidates: set[int] = set()
        for near_parent in parent_near_flat[
            int(parent_near_offsets[parent]) : int(parent_near_offsets[parent + 1])
        ]:
            for src in children_by_parent[int(near_parent)]:
                src_i = int(src)
                if src_i == dst or src_i in near_set:
                    continue
                candidates.add(src_i)
        for src in sorted(candidates):
            offset = (
                int(coords_arr[dst, 0] - coords_arr[src, 0]),
                int(coords_arr[dst, 1] - coords_arr[src, 1]),
                int(coords_arr[dst, 2] - coords_arr[src, 2]),
            )
            batch = grouped.setdefault(offset, [[], []])
            batch[0].append(int(src))
            batch[1].append(int(dst))
    return {
        offset: (
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
        )
        for offset, (src, dst) in grouped.items()
    }


def _resolve_multilevel_hf_start_level(
    *,
    partition: MLFMMPartition,
    k: float,
    leaf_level: int,
    hf_start_level: int | None,
    hf_wavelength_divisor: float,
) -> int:
    """Resolve the first sampled HF ownership level.

    Policy:
    - explicit `hf_start_level` wins when provided;
    - otherwise include level 2, the first level that can own same-level far
      interactions in the current hierarchy.

    Starting later than level 2 without remapping ownership omits far
    interactions whose parents are already far at the skipped levels. Such
    coarse-level skipping is therefore an explicit expert/diagnostic choice, not
    the automatic production policy.
    """

    leaf = int(leaf_level)
    if leaf <= 1:
        return leaf
    min_level = 2
    if hf_start_level is not None:
        explicit = int(hf_start_level)
        if explicit < min_level or explicit > leaf:
            raise ValueError(
                "mlfmm_options.hf_start_level must satisfy "
                f"{min_level} <= hf_start_level <= {leaf} for this partition depth. "
                f"Got {explicit}."
            )
        return explicit

    divisor = float(hf_wavelength_divisor)
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(
            "mlfmm_options.hf_wavelength_divisor must be finite and > 0. "
            f"Got {hf_wavelength_divisor!r}."
        )
    return int(min_level)


def _offset_delta_from_half_size(half_size: float, offset: tuple[int, int, int]) -> np.ndarray:
    """Return the physical center shift for one relative box offset."""

    return 2.0 * float(half_size) * np.asarray(offset, dtype=float)


def _spherical_hankel_all(nmax: int, z: complex) -> np.ndarray:
    """Return spherical Hankel values `h_n^(1)(z)` for `n=0..nmax`."""

    orders = np.arange(int(nmax) + 1, dtype=int)
    return np.asarray(spherical_jn(orders, z) + 1j * spherical_yn(orders, z), dtype=np.complex128)


def _rokhlin_transfer_values(
    cosines: np.ndarray,
    *,
    k: complex,
    radius: float,
    truncation_order: int,
) -> np.ndarray:
    """Return the unweighted sampled Rokhlin transfer values for one offset."""

    hankel = _spherical_hankel_all(int(truncation_order), complex(k) * float(radius))
    legendre = np.polynomial.legendre.legvander(
        np.asarray(cosines, dtype=float), int(truncation_order)
    )
    orders = np.arange(int(truncation_order) + 1, dtype=int)
    coeffs = (2.0 * orders + 1.0) * (1j**orders) * hankel
    return np.asarray(legendre @ coeffs, dtype=np.complex128)


def _sampled_rokhlin_translator(
    delta: np.ndarray,
    *,
    k: complex,
    truncation_order: int,
    directions: np.ndarray,
    weights: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Return the weighted sampled Rokhlin translator for one box offset."""

    tr = np.asarray(delta, dtype=float).reshape(3)
    radius = float(np.linalg.norm(tr))
    if radius == 0.0:
        return np.asarray(weights, dtype=dtype)
    cosines = np.asarray(directions, dtype=float) @ (tr / radius)
    transfer = _rokhlin_transfer_values(
        cosines,
        k=complex(k),
        radius=radius,
        truncation_order=int(truncation_order),
    )
    return np.asarray(transfer * np.asarray(weights, dtype=float), dtype=dtype)


def _sparse_matrix_nbytes(matrix: scipy.sparse.csr_matrix) -> int:
    """Return the byte footprint of one CSR interpolation matrix."""

    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _leaf_apply_groups_nbytes(
    leaf_groups: tuple[MLFMMLeafApplyGroup, ...],
) -> dict[str, int]:
    """Return grouped leaf-apply bytes split by payload type."""

    aggregation_bytes = 0
    pair_delta_bytes = 0
    index_bytes = 0
    for group in leaf_groups:
        index_bytes += int(np.asarray(group.leaf_ids, dtype=np.int64).nbytes)
        index_bytes += int(np.asarray(group.particle_indices, dtype=np.int64).nbytes)
        if group.aggregation is not None:
            aggregation_bytes += int(np.asarray(group.aggregation).nbytes)
        if group.pair_deltas is not None:
            pair_delta_bytes += int(np.asarray(group.pair_deltas, dtype=np.float64).nbytes)
    return {
        "aggregation_bytes": int(aggregation_bytes),
        "pair_delta_bytes": int(pair_delta_bytes),
        "index_bytes": int(index_bytes),
        "persistent_total_bytes": int(aggregation_bytes + pair_delta_bytes + index_bytes),
    }


def _single_level_offset_index_bytes(
    offset_batches: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> int:
    """Return grouped offset index bytes for one same-level schedule."""

    total = 0
    for src_idx, dst_idx in offset_batches.values():
        total += int(np.asarray(src_idx, dtype=np.int64).nbytes)
        total += int(np.asarray(dst_idx, dtype=np.int64).nbytes)
    return int(total)


def _transfer_schedule_nbytes(transfers: tuple[MLFMMTransferOperators, ...]) -> dict[str, int]:
    """Return packed transfer bytes split by maps, phases, and grouped indices."""

    map_bytes = 0
    phase_bytes = 0
    batch_index_bytes = 0
    for transfer in transfers:
        map_bytes += _sparse_matrix_nbytes(transfer.interpolation.matrix)
        map_bytes += _sparse_matrix_nbytes(transfer.anterpolation.matrix)
        for child_idx, parent_idx in transfer.batches_by_shift.values():
            batch_index_bytes += int(np.asarray(child_idx, dtype=np.int64).nbytes)
            batch_index_bytes += int(np.asarray(parent_idx, dtype=np.int64).nbytes)
        for phase in transfer.phase_up_by_shift.values():
            phase_bytes += int(np.asarray(phase, dtype=np.complex128).nbytes)
        for phase in transfer.phase_down_by_shift.values():
            phase_bytes += int(np.asarray(phase, dtype=np.complex128).nbytes)
    return {
        "map_bytes": int(map_bytes),
        "phase_bytes": int(phase_bytes),
        "batch_index_bytes": int(batch_index_bytes),
        "persistent_total_bytes": int(map_bytes + phase_bytes + batch_index_bytes),
    }


def _directional_transform_nbytes(transforms: MLFMMDirectionalTransformData) -> int:
    """Return persistent host bytes for one dense or structured directional payload."""

    grid = transforms.grid
    total = int(
        np.asarray(grid.alpha, dtype=np.float64).nbytes
        + np.asarray(grid.beta, dtype=np.float64).nbytes
        + np.asarray(grid.beta_weights, dtype=np.float64).nbytes
        + np.asarray(grid.directions, dtype=np.float64).nbytes
        + np.asarray(grid.weights, dtype=np.float64).nbytes
        + np.asarray(grid.reflection_permutation, dtype=np.int64).nbytes
    )
    if isinstance(transforms, MLFMMDirectionalStructuredTransforms):
        return int(
            total
            + np.asarray(transforms.fth_beta, dtype=np.complex128).nbytes
            + np.asarray(transforms.fph_beta, dtype=np.complex128).nbytes
            + np.asarray(transforms.m_of_scalar, dtype=np.int32).nbytes
        )
    return int(
        total
        + np.asarray(transforms.Fth, dtype=np.complex128).nbytes
        + np.asarray(transforms.Fph, dtype=np.complex128).nbytes
    )


def _single_level_memory_diagnostics(
    *,
    lmax: int,
    n_particles: int,
    near_dtype: np.dtype,
    far_dtype: np.dtype,
    plan_summary: dict[str, int | float | str],
    operators: MLFMMSingleLevelOperators,
) -> dict[str, object]:
    """Return stage-aware memory diagnostics for one single-level NumPy plan."""

    nm = int(n_modes(int(lmax)))
    box_nm = int(n_modes(int(operators.box_order)))
    n_leaves = len(operators.partition.leaves)
    ndir = int(operators.directional.grid.directions.shape[0])
    far_itemsize = int(np.dtype(far_dtype).itemsize)
    near_itemsize = int(np.dtype(near_dtype).itemsize)
    leaf_apply = _leaf_apply_groups_nbytes(operators.leaf_groups)
    offset_index_bytes = _single_level_offset_index_bytes(operators.far_offset_batches)
    offset_diagonal_bytes = int(
        sum(
            np.asarray(diagonal, dtype=far_dtype).nbytes
            for diagonal in operators.offset_diagonals.values()
        )
    )
    directional_transform_bytes = _directional_transform_nbytes(operators.directional)
    return {
        "stage": "single_level",
        "plan_summary": dict(plan_summary),
        "leaf_apply": {
            "mode": str(operators.leaf_apply_mode),
            "group_count": len(operators.leaf_groups),
            "occupancies": [int(group.occupancy) for group in operators.leaf_groups],
            **leaf_apply,
        },
        "workspace_bytes": {
            "nrhs_reference": 1,
            "near_total_bytes_estimate": int(n_particles * nm * near_itemsize),
            "leaf_box_states_bytes": int(n_leaves * box_nm * far_itemsize),
            "outgoing_hierarchy_bytes": int(n_leaves * 4 * ndir * far_itemsize),
            "incoming_hierarchy_bytes": int(n_leaves * 4 * ndir * far_itemsize),
            "incoming_box_bytes": int(n_leaves * box_nm * far_itemsize),
            "full_far_hierarchy_bytes": int(n_leaves * (8 * ndir + 2 * box_nm) * far_itemsize),
        },
        "prepared_bytes": {
            "directional_transform_bytes": int(directional_transform_bytes),
            "same_level_offset_index_bytes": int(offset_index_bytes),
            "same_level_offset_diagonal_bytes": int(offset_diagonal_bytes),
        },
    }


def _multilevel_memory_diagnostics(
    *,
    lmax: int,
    n_particles: int,
    near_dtype: np.dtype,
    far_dtype: np.dtype,
    plan_summary: dict[str, int | float | str],
    operators: MLFMMMultilevelOperators,
) -> dict[str, object]:
    """Return stage-aware memory diagnostics for one multilevel NumPy plan."""

    nm = int(n_modes(int(lmax)))
    leaf_box_nm = int(n_modes(int(operators.levels[int(operators.leaf_level)].box_order)))
    n_leaves = len(operators.partition.leaves)
    far_itemsize = int(np.dtype(far_dtype).itemsize)
    near_itemsize = int(np.dtype(near_dtype).itemsize)
    outgoing_bytes = int(
        sum(
            level.coords.shape[0] * 4 * level.directional.grid.directions.shape[0] * far_itemsize
            for level in operators.levels
        )
    )
    incoming_bytes = int(
        sum(
            level.coords.shape[0] * 4 * level.directional.grid.directions.shape[0] * far_itemsize
            for level in operators.levels
        )
    )
    level_transform_bytes = int(
        sum(_directional_transform_nbytes(level.directional) for level in operators.levels)
    )
    level_offset_index_bytes = int(
        sum(
            _single_level_offset_index_bytes(level.far_offset_batches) for level in operators.levels
        )
    )
    level_offset_diagonal_bytes = int(
        sum(
            np.asarray(diagonal, dtype=far_dtype).nbytes
            for level in operators.levels
            for diagonal in level.offset_diagonals.values()
        )
    )
    transfer_bytes = _transfer_schedule_nbytes(operators.transfers)
    leaf_apply = _leaf_apply_groups_nbytes(operators.leaf_groups)
    return {
        "stage": "multilevel",
        "plan_summary": dict(plan_summary),
        "leaf_apply": {
            "mode": str(operators.leaf_apply_mode),
            "group_count": len(operators.leaf_groups),
            "occupancies": [int(group.occupancy) for group in operators.leaf_groups],
            **leaf_apply,
        },
        "workspace_bytes": {
            "nrhs_reference": 1,
            "near_total_bytes_estimate": int(n_particles * nm * near_itemsize),
            "leaf_box_states_bytes": int(n_leaves * leaf_box_nm * far_itemsize),
            "outgoing_hierarchy_bytes": int(outgoing_bytes),
            "incoming_hierarchy_bytes": int(incoming_bytes),
            "incoming_box_bytes": int(n_leaves * leaf_box_nm * far_itemsize),
            "full_far_hierarchy_bytes": int(
                outgoing_bytes + incoming_bytes + 2 * n_leaves * leaf_box_nm * far_itemsize
            ),
        },
        "prepared_bytes": {
            "directional_transform_bytes": int(level_transform_bytes),
            "same_level_offset_index_bytes": int(level_offset_index_bytes),
            "same_level_offset_diagonal_bytes": int(level_offset_diagonal_bytes),
            "transfer_map_bytes": int(transfer_bytes["map_bytes"]),
            "transfer_phase_bytes": int(transfer_bytes["phase_bytes"]),
            "transfer_batch_index_bytes": int(transfer_bytes["batch_index_bytes"]),
        },
    }


def _build_leaf_box_maps(
    *,
    lmax: int,
    box_order: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
    leaf_map_backend: Literal["numpy", "cupy"] = "numpy",
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build particle-to-box aggregation maps and adjoint box-to-particle receive maps."""

    backend = str(leaf_map_backend).strip().lower()
    if backend == "cupy":
        from .mlfmm_cupy import build_leaf_box_maps_cupy

        return build_leaf_box_maps_cupy(
            lmax=int(lmax),
            box_order=int(box_order),
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            partition=partition,
            radial_lut=radial_lut,
            dtype=np.dtype(dtype),
        )
    if backend != "numpy":
        raise ValueError(
            f"Unsupported leaf-map backend {leaf_map_backend!r}. Use 'numpy' or 'cupy'."
        )

    full_order = max(int(lmax), int(box_order))
    ab5 = translation_ab5_table(full_order, dtype=np.complex128)
    leaves = tuple(sorted(partition.leaves, key=lambda leaf: int(leaf.id)))
    n_leaves = len(leaves)
    aggregation_by_id: list[np.ndarray | None] = [None] * n_leaves
    for leaf in leaves:
        leaf_id = int(leaf.id)
        leaf_center = np.asarray(leaf.center, dtype=float)
        particle_indices = np.asarray(leaf.particle_indices, dtype=np.int64)
        agg_blocks = [
            translation_block_rect(
                int(box_order),
                int(lmax),
                float(k),
                np.asarray(leaf_center - positions[int(pidx)], dtype=float),
                ab5=ab5,
                radial_lut=radial_lut,
                family="interior",
            )
            for pidx in particle_indices
        ]
        aggregation_by_id[leaf_id] = np.hstack(agg_blocks).astype(dtype, copy=False)
    if any(agg is None for agg in aggregation_by_id):
        raise RuntimeError("Internal error: incomplete leaf aggregation build.")
    aggregation = tuple(
        np.asarray(aggregation_by_id[leaf_id], dtype=dtype) for leaf_id in range(n_leaves)
    )
    receive = [np.asarray(np.conjugate(agg).T, dtype=dtype) for agg in aggregation]
    return tuple(aggregation), tuple(receive)


def _group_leaves_by_occupancy(partition: MLFMMPartition) -> tuple[tuple[MLFMMBox, ...], ...]:
    """Return occupied leaves grouped by uniform particle occupancy."""

    leaves = tuple(sorted(partition.leaves, key=lambda leaf: int(leaf.id)))
    grouped: dict[int, list[MLFMMBox]] = {}
    for expected_id, leaf in enumerate(leaves):
        if int(leaf.id) != expected_id:
            raise ValueError(
                "Leaf ids must be contiguous in [0, n_leaves) for grouped NumPy leaf apply. "
                f"Missing leaf id {expected_id}."
            )
        occupancy = int(np.asarray(leaf.particle_indices, dtype=np.int64).reshape(-1).size)
        if occupancy <= 0:
            raise ValueError(f"leaf {expected_id} has non-positive occupancy {occupancy}.")
        grouped.setdefault(occupancy, []).append(leaf)
    return tuple(tuple(grouped[occupancy]) for occupancy in sorted(grouped))


def _build_numpy_leaf_apply_groups_dense(
    *,
    lmax: int,
    box_order: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
) -> tuple[MLFMMLeafApplyGroup, ...]:
    """Build grouped dense leaf operators for the NumPy repeated-apply path."""

    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    nmodes_in = int(n_modes(int(lmax)))
    full_order = max(int(lmax), int(box_order))
    ab5 = translation_ab5_table(full_order, dtype=np.complex128)
    groups: list[MLFMMLeafApplyGroup] = []
    for leaves in _group_leaves_by_occupancy(partition):
        occupancy = int(leaves[0].particle_indices.size)
        leaf_ids = np.asarray([int(leaf.id) for leaf in leaves], dtype=np.int64)
        particle_indices = np.empty((len(leaves), occupancy), dtype=np.int64)
        aggregation = np.empty(
            (len(leaves), int(n_modes(int(box_order))), occupancy * nmodes_in),
            dtype=np.dtype(dtype),
        )
        for local_idx, leaf in enumerate(leaves):
            particle_idx = np.asarray(leaf.particle_indices, dtype=np.int64).reshape(-1)
            particle_indices[local_idx] = particle_idx
            agg_blocks = [
                translation_block_rect(
                    int(box_order),
                    int(lmax),
                    float(k),
                    np.asarray(
                        np.asarray(leaf.center, dtype=float) - positions_arr[int(pidx)], dtype=float
                    ),
                    ab5=ab5,
                    radial_lut=radial_lut,
                    family="interior",
                )
                for pidx in particle_idx
            ]
            aggregation[local_idx] = np.hstack(agg_blocks).astype(dtype, copy=False)
        groups.append(
            MLFMMLeafApplyGroup(
                occupancy=occupancy,
                nmodes=nmodes_in,
                leaf_ids=leaf_ids,
                particle_indices=particle_indices,
                aggregation=np.ascontiguousarray(aggregation, dtype=dtype),
                pair_deltas=None,
            )
        )
    return tuple(groups)


def _build_numpy_leaf_apply_groups_otf(
    *,
    lmax: int,
    positions: np.ndarray,
    partition: MLFMMPartition,
) -> tuple[MLFMMLeafApplyGroup, ...]:
    """Build compact grouped leaf metadata for NumPy on-the-fly apply."""

    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    nmodes_in = int(n_modes(int(lmax)))
    groups: list[MLFMMLeafApplyGroup] = []
    for leaves in _group_leaves_by_occupancy(partition):
        occupancy = int(leaves[0].particle_indices.size)
        leaf_ids = np.asarray([int(leaf.id) for leaf in leaves], dtype=np.int64)
        particle_indices = np.empty((len(leaves), occupancy), dtype=np.int64)
        pair_deltas = np.empty((len(leaves), occupancy, 3), dtype=np.float64)
        for local_idx, leaf in enumerate(leaves):
            particle_idx = np.asarray(leaf.particle_indices, dtype=np.int64).reshape(-1)
            particle_indices[local_idx] = particle_idx
            pair_deltas[local_idx] = (
                np.asarray(leaf.center, dtype=np.float64).reshape(1, 3)
                - positions_arr[particle_idx]
            )
        groups.append(
            MLFMMLeafApplyGroup(
                occupancy=occupancy,
                nmodes=nmodes_in,
                leaf_ids=leaf_ids,
                particle_indices=particle_indices,
                aggregation=None,
                pair_deltas=np.ascontiguousarray(pair_deltas, dtype=np.float64),
            )
        )
    return tuple(groups)


def _leaf_translation_blocks_from_pair_deltas(
    *,
    lmax_out: int,
    lmax_in: int,
    k: float,
    pair_deltas: np.ndarray,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
) -> np.ndarray:
    """Build rectangular interior translation blocks for grouped leaf pair deltas."""

    pair_rows = np.asarray(pair_deltas, dtype=float).reshape(-1, 3)
    full_order = max(int(lmax_out), int(lmax_in))
    ab5 = translation_ab5_table(full_order, dtype=np.complex128)
    blocks = [
        translation_block_rect(
            int(lmax_out),
            int(lmax_in),
            float(k),
            np.asarray(delta, dtype=float),
            ab5=ab5,
            radial_lut=radial_lut,
            family="interior",
        )
        for delta in pair_rows
    ]
    return np.asarray(blocks, dtype=dtype)


def _apply_reflection_to_channel_batches(
    channel_batches: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Apply the sampled physical reflection permutation to batched directional channels."""

    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    arr = np.asarray(channel_batches)
    if arr.shape[-1] != perm.size:
        raise ValueError(
            f"directional channel size {arr.shape[-1]} does not match permutation {perm.size}."
        )
    return np.asarray(np.take(arr, perm, axis=-1), dtype=arr.dtype)


def _leaf_box_states_from_dense_maps(
    *,
    lmax: int,
    positions: np.ndarray,
    x: np.ndarray,
    partition: MLFMMPartition,
    aggregation: tuple[np.ndarray, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Aggregate particle outgoing coefficients into one box state per occupied leaf."""

    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=dtype).reshape(np.asarray(positions).shape[0], nm)
    box_nm = aggregation[0].shape[0]
    states = np.zeros((len(partition.leaves), box_nm), dtype=dtype)
    for leaf in partition.leaves:
        coeffs = arr[leaf.particle_indices].reshape(-1)
        states[int(leaf.id)] = aggregation[int(leaf.id)] @ coeffs
    return states


def _aggregate_leaf_box_states(
    *,
    lmax: int,
    box_order: int,
    k: float,
    x: np.ndarray,
    leaf_groups: tuple[MLFMMLeafApplyGroup, ...],
    n_leaves: int,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
) -> np.ndarray:
    """Aggregate particle coefficients into grouped leaf box states."""

    nm = int(n_modes(int(lmax)))
    box_nm = int(n_modes(int(box_order)))
    arr = np.asarray(x, dtype=dtype).reshape(-1, nm)
    states = np.zeros((int(n_leaves), box_nm), dtype=dtype)
    for group in leaf_groups:
        coeffs = arr[np.asarray(group.particle_indices, dtype=np.int64)]
        if group.aggregation is not None:
            coeffs_flat = coeffs.reshape(coeffs.shape[0], -1)
            states[np.asarray(group.leaf_ids, dtype=np.int64)] = np.einsum(
                "gmn,gn->gm",
                np.asarray(group.aggregation, dtype=dtype),
                coeffs_flat,
                optimize=True,
            )
            continue
        if group.pair_deltas is None:
            raise RuntimeError(
                "Internal error: grouped NumPy leaf apply is missing translation data."
            )
        pair_blocks = _leaf_translation_blocks_from_pair_deltas(
            lmax_out=int(box_order),
            lmax_in=int(lmax),
            k=float(k),
            pair_deltas=group.pair_deltas,
            radial_lut=radial_lut,
            dtype=np.dtype(dtype),
        ).reshape(coeffs.shape[0], int(group.occupancy), box_nm, nm)
        states[np.asarray(group.leaf_ids, dtype=np.int64)] = np.einsum(
            "gqmn,gqn->gm",
            pair_blocks,
            coeffs,
            optimize=True,
        )
    return states


def _receive_leaf_boxes_to_particles(
    *,
    lmax: int,
    box_order: int,
    k: float,
    n_particles: int,
    incoming_box: np.ndarray,
    leaf_groups: tuple[MLFMMLeafApplyGroup, ...],
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
) -> np.ndarray:
    """Apply grouped leaf receive maps or on-the-fly adjoints back to particles."""

    nm = int(n_modes(int(lmax)))
    box_nm = int(n_modes(int(box_order)))
    y_far = np.zeros((int(n_particles), nm), dtype=dtype)
    incoming = np.asarray(incoming_box, dtype=dtype).reshape(-1, box_nm)
    for group in leaf_groups:
        leaf_ids = np.asarray(group.leaf_ids, dtype=np.int64)
        particle_indices = np.asarray(group.particle_indices, dtype=np.int64)
        if group.aggregation is not None:
            # Keep only the grouped forward aggregation blocks in persistent
            # NumPy state and recover the adjoint receive action on demand.
            receive_adj = np.swapaxes(np.asarray(group.aggregation, dtype=dtype).conj(), 1, 2)
            contribution = np.einsum(
                "gmb,gb->gm",
                receive_adj,
                incoming[leaf_ids],
                optimize=True,
            ).reshape(particle_indices.shape[0], int(group.occupancy), nm)
            y_far[particle_indices] += contribution
            continue
        if group.pair_deltas is None:
            raise RuntimeError(
                "Internal error: grouped NumPy leaf receive is missing translation data."
            )
        pair_blocks = _leaf_translation_blocks_from_pair_deltas(
            lmax_out=int(box_order),
            lmax_in=int(lmax),
            k=float(k),
            pair_deltas=group.pair_deltas,
            radial_lut=radial_lut,
            dtype=np.dtype(dtype),
        ).reshape(particle_indices.shape[0], int(group.occupancy), box_nm, nm)
        contribution = np.einsum(
            "gqmn,gm->gqn",
            np.conjugate(pair_blocks),
            incoming[leaf_ids],
            optimize=True,
        )
        y_far[particle_indices] += contribution
    return y_far


def _exact_leaf_near_apply(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Apply exact interactions for leaf-near particle pairs only."""

    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=dtype).reshape(np.asarray(positions).shape[0], nm)
    y = np.zeros_like(arr, dtype=dtype)
    ab5 = translation_ab5_table(int(lmax), dtype=np.complex128)
    for a, b in partition.leaf_near_pairs:
        leaf_a = partition.leaves[a]
        leaf_b = partition.leaves[b]
        if a == b:
            for i in leaf_a.particle_indices:
                for j in leaf_a.particle_indices:
                    if int(i) == int(j):
                        continue
                    key = (int(i), int(j))
                    wij = block_cache.get(key) if block_cache is not None else None
                    if wij is None:
                        wij = translation_block(
                            int(lmax),
                            float(k),
                            np.asarray(positions[int(i)] - positions[int(j)], dtype=float),
                            ab5=ab5,
                            radial_lut=radial_lut,
                        )
                        if block_cache is not None:
                            block_cache[key] = wij
                    y[int(i)] += np.asarray(wij, dtype=dtype) @ arr[int(j)]
            continue
        for i in leaf_a.particle_indices:
            for j in leaf_b.particle_indices:
                key_ij = (int(i), int(j))
                wij = block_cache.get(key_ij) if block_cache is not None else None
                if wij is None:
                    wij = translation_block(
                        int(lmax),
                        float(k),
                        np.asarray(positions[int(i)] - positions[int(j)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )
                    if block_cache is not None:
                        block_cache[key_ij] = wij
                key_ji = (int(j), int(i))
                wji = block_cache.get(key_ji) if block_cache is not None else None
                if wji is None:
                    wji = translation_block(
                        int(lmax),
                        float(k),
                        np.asarray(positions[int(j)] - positions[int(i)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )
                    if block_cache is not None:
                        block_cache[key_ji] = wji
                y[int(i)] += np.asarray(wij, dtype=dtype) @ arr[int(j)]
                y[int(j)] += np.asarray(wji, dtype=dtype) @ arr[int(i)]
    return y.reshape(-1)


def _populate_exact_leaf_near_cache(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    block_cache: dict[tuple[int, int], np.ndarray] | None,
) -> None:
    """Populate cached exact leaf-near translation blocks without a matvec pass."""

    if block_cache is None:
        return
    ab5 = translation_ab5_table(int(lmax), dtype=np.complex128)
    positions_arr = np.asarray(positions, dtype=float)
    for a, b in partition.leaf_near_pairs:
        leaf_a = partition.leaves[a]
        leaf_b = partition.leaves[b]
        if a == b:
            for i in leaf_a.particle_indices:
                for j in leaf_a.particle_indices:
                    if int(i) == int(j):
                        continue
                    key_ij = (int(i), int(j))
                    if key_ij not in block_cache:
                        block_cache[key_ij] = translation_block(
                            int(lmax),
                            float(k),
                            np.asarray(positions_arr[int(i)] - positions_arr[int(j)], dtype=float),
                            ab5=ab5,
                            radial_lut=radial_lut,
                        )
            continue
        for i in leaf_a.particle_indices:
            for j in leaf_b.particle_indices:
                key_ij = (int(i), int(j))
                if key_ij not in block_cache:
                    block_cache[key_ij] = translation_block(
                        int(lmax),
                        float(k),
                        np.asarray(positions_arr[int(i)] - positions_arr[int(j)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )
                key_ji = (int(j), int(i))
                if key_ji not in block_cache:
                    block_cache[key_ji] = translation_block(
                        int(lmax),
                        float(k),
                        np.asarray(positions_arr[int(j)] - positions_arr[int(i)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )


def build_single_level_mlfmm_operators(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None = None,
    box_order: int | None = None,
    translator_order: int | None = None,
    accuracy_level: int = 3,
    order_additive: int = 2,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    leaf_map_backend: Literal["numpy", "cupy"] = "numpy",
    build_leaf_maps: bool = True,
) -> MLFMMSingleLevelOperators:
    """Build the sampled single-level HF far operator over one occupied leaf level.

    NumPy keeps grouped reusable leaf work units by default and can switch to
    a compact on-the-fly leaf path when `build_leaf_maps=False`. Relative-
    offset keyed sampled translators remain shared across all occupied leaves
    at the resolved depth.
    """

    if not partition.leaves:
        raise ValueError("single-level MLFMM requires at least one occupied leaf.")
    leaf_half_size = float(partition.leaves[0].half_size)
    shared_box_order = (
        box_order_rokhlin_like(
            particle_lmax=int(lmax),
            k=float(k),
            box_half_size=leaf_half_size,
            accuracy=int(accuracy_level),
            additive=int(order_additive),
        )
        if box_order is None
        else int(box_order)
    )
    shared_translator_order = (
        int(shared_box_order) if translator_order is None else int(translator_order)
    )
    shared_grid_order = max(int(shared_box_order), int(shared_translator_order))
    out_dtype = np.dtype(dtype)
    aggregation: tuple[np.ndarray, ...]
    receive: tuple[np.ndarray, ...]
    backend = str(leaf_map_backend).strip().lower()
    directional = structured_directional_transforms(
        int(shared_box_order), grid_order=int(shared_grid_order)
    )
    if backend == "numpy":
        if bool(build_leaf_maps):
            leaf_groups = _build_numpy_leaf_apply_groups_dense(
                lmax=int(lmax),
                box_order=int(shared_box_order),
                k=float(k),
                positions=np.asarray(positions, dtype=float),
                partition=partition,
                radial_lut=radial_lut,
                dtype=out_dtype,
            )
            leaf_apply_mode: MLFMMLeafApplyMode = "dense"
        else:
            leaf_groups = _build_numpy_leaf_apply_groups_otf(
                lmax=int(lmax),
                positions=np.asarray(positions, dtype=float),
                partition=partition,
            )
            leaf_apply_mode = "on_the_fly"
        aggregation = tuple()
        receive = tuple()
    elif bool(build_leaf_maps):
        aggregation, receive = _build_leaf_box_maps(
            lmax=int(lmax),
            box_order=int(shared_box_order),
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            partition=partition,
            radial_lut=radial_lut,
            dtype=out_dtype,
            leaf_map_backend=leaf_map_backend,
        )
        leaf_groups = tuple()
        leaf_apply_mode = "dense"
    else:
        aggregation = tuple()
        receive = tuple()
        leaf_groups = tuple()
        leaf_apply_mode = "on_the_fly"
    leaf_cell_coords = _leaf_cell_coords(partition)
    far_offset_batches = _leaf_offset_batches(partition, leaf_cell_coords)
    offset_diagonals: dict[tuple[int, int, int], np.ndarray] = {}
    for offset in far_offset_batches:
        delta = _offset_delta_from_half_size(leaf_half_size, offset)
        offset_diagonals[offset] = _sampled_rokhlin_translator(
            delta,
            k=complex(k),
            truncation_order=int(shared_translator_order),
            directions=directional.grid.directions,
            weights=directional.grid.weights,
            dtype=out_dtype,
        )
    return MLFMMSingleLevelOperators(
        partition=partition,
        box_order=int(shared_box_order),
        translator_order=int(shared_translator_order),
        grid_order=int(shared_grid_order),
        directional=directional,
        leaf_cell_coords=leaf_cell_coords,
        far_offset_batches=far_offset_batches,
        offset_diagonals=offset_diagonals,
        aggregation=aggregation,
        receive=receive,
        leaf_groups=leaf_groups,
        leaf_apply_mode=leaf_apply_mode,
    )


def apply_single_level_mlfmm(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    operators: MLFMMSingleLevelOperators,
    radial_lut: RadialLUT | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    near_dtype: np.dtype | type[np.complexfloating] | type[np.complex128] | None = None,
    far_dtype: np.dtype | type[np.complexfloating] | type[np.complex128] | None = None,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact leaf-near interactions and sampled single-level far interactions.

    The return value keeps the exact near contribution separate from the
    sampled far contribution so diagnostics can compare each piece directly
    against pairwise references.
    """

    out_dtype = np.dtype(dtype)
    if not operators.leaf_groups and (
        len(operators.aggregation) == 0 or len(operators.receive) == 0
    ):
        raise RuntimeError(
            "single-level MLFMM operators are missing reusable leaf apply data. "
            "This operator was likely prepared for CuPy host-cache conversion only."
        )
    near_out_dtype = np.dtype(out_dtype if near_dtype is None else near_dtype)
    far_out_dtype = np.dtype(out_dtype if far_dtype is None else far_dtype)
    y_near = _exact_leaf_near_apply(
        lmax=int(lmax),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x, dtype=near_out_dtype),
        partition=operators.partition,
        radial_lut=radial_lut,
        dtype=near_out_dtype,
        block_cache=block_cache,
    )
    if operators.leaf_groups:
        leaf_states = _aggregate_leaf_box_states(
            lmax=int(lmax),
            box_order=int(operators.box_order),
            k=float(k),
            x=np.asarray(x, dtype=far_out_dtype),
            leaf_groups=operators.leaf_groups,
            n_leaves=len(operators.partition.leaves),
            radial_lut=radial_lut,
            dtype=far_out_dtype,
        )
    else:
        leaf_states = _leaf_box_states_from_dense_maps(
            lmax=int(lmax),
            positions=np.asarray(positions, dtype=float),
            x=np.asarray(x, dtype=far_out_dtype),
            partition=operators.partition,
            aggregation=operators.aggregation,
            dtype=far_out_dtype,
        )
    ndir = int(operators.directional.grid.directions.shape[0])
    outgoing = np.zeros((len(operators.partition.leaves), 4, ndir), dtype=far_out_dtype)
    for leaf_id, box_state in enumerate(leaf_states):
        channels = box_outgoing_to_directional(operators.directional, box_state)
        for chan_idx, channel in enumerate(channels):
            outgoing[leaf_id, chan_idx] = np.asarray(channel, dtype=far_out_dtype)

    incoming = np.zeros_like(outgoing, dtype=far_out_dtype)
    for offset, (src_idx, dst_idx) in operators.far_offset_batches.items():
        translated = outgoing[src_idx] * operators.offset_diagonals[offset][None, None, :]
        np.add.at(incoming, dst_idx, translated)

    box_nm = n_modes(int(operators.box_order))
    incoming_box = np.zeros((len(operators.partition.leaves), box_nm), dtype=far_out_dtype)
    for leaf_id in range(len(operators.partition.leaves)):
        incoming_box[leaf_id] = directional_to_box_regular(
            operators.directional,
            incoming[leaf_id, 0],
            incoming[leaf_id, 1],
            incoming[leaf_id, 2],
            incoming[leaf_id, 3],
        )

    if operators.leaf_groups:
        y_far = _receive_leaf_boxes_to_particles(
            lmax=int(lmax),
            box_order=int(operators.box_order),
            k=float(k),
            n_particles=np.asarray(positions).shape[0],
            incoming_box=incoming_box,
            leaf_groups=operators.leaf_groups,
            radial_lut=radial_lut,
            dtype=far_out_dtype,
        )
    else:
        nm = n_modes(int(lmax))
        ns = np.asarray(positions).shape[0]
        y_far = np.zeros((ns, nm), dtype=far_out_dtype)
        for leaf in operators.partition.leaves:
            contribution = operators.receive[int(leaf.id)] @ incoming_box[int(leaf.id)]
            y_far[leaf.particle_indices] += contribution.reshape(leaf.particle_indices.size, nm)
    return y_near.reshape(-1), y_far.reshape(-1)


def build_multilevel_mlfmm_operators(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None = None,
    box_order: int | None = None,
    accuracy_level: int = 3,
    order_additive: int = 2,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    leaf_map_backend: Literal["numpy", "cupy"] = "numpy",
    build_leaf_maps: bool = True,
    show_progress: bool = False,
    hf_start_level: int | None = None,
    hf_wavelength_divisor: float = 5.0,
) -> MLFMMMultilevelOperators:
    """Build multilevel sampled HF operators over occupied boxes only.

    The hierarchy stores only occupied boxes, together with same-level far
    batches and parent/child transfer operators for the upward and downward
    sampled passes. Sampled start level is either explicitly forced via
    `hf_start_level` or chosen as the first same-level far ownership level. The
    automatic path does not skip coarse owned levels, because doing so would
    drop interactions instead of approximating them elsewhere.
    """

    if not partition.leaves:
        raise ValueError("multilevel MLFMM requires at least one occupied leaf.")
    out_dtype = np.dtype(dtype)
    leaf_level = int(partition.depth)
    hf_end_level = int(leaf_level)
    hf_start_level = _resolve_multilevel_hf_start_level(
        partition=partition,
        k=float(k),
        leaf_level=leaf_level,
        hf_start_level=hf_start_level,
        hf_wavelength_divisor=float(hf_wavelength_divisor),
    )
    leaf_coords = _leaf_cell_coords(partition)
    coords_by_level: list[np.ndarray] = [
        np.zeros((0, 3), dtype=np.int64) for _ in range(leaf_level + 1)
    ]
    coords_by_level[leaf_level] = np.asarray(
        sorted({leaf_coords[int(leaf.id)] for leaf in partition.leaves}),
        dtype=np.int64,
    ).reshape(-1, 3)
    for level in range(leaf_level - 1, -1, -1):
        coords_by_level[level] = np.asarray(
            sorted({tuple((coord // 2).tolist()) for coord in coords_by_level[level + 1]}),
            dtype=np.int64,
        ).reshape(-1, 3)

    near_neighbors_by_level: list[tuple[np.ndarray, np.ndarray]] = []
    for level in range(len(coords_by_level)):
        near_neighbors_by_level.append(_coords_near_neighbors(coords_by_level[level]))

    levels: list[MLFMMLevelOperators] = []
    dummy_directional = structured_directional_transforms(1, grid_order=1)
    level_progress = (
        tqdm(range(leaf_level + 1), desc="[MLFMM] build levels", unit="level")
        if show_progress
        else None
    )
    level_iter = level_progress if level_progress is not None else range(leaf_level + 1)
    for level in level_iter:
        if level_progress is not None:
            level_progress.set_postfix_str(f"L{level}: topology", refresh=True)
        coords = coords_by_level[level]
        if level == 0:
            parent_indices = np.full((coords.shape[0],), -1, dtype=np.int64)
        else:
            parent_lookup = {
                tuple(int(v) for v in coords_by_level[level - 1][idx]): idx
                for idx in range(coords_by_level[level - 1].shape[0])
            }
            parent_indices = np.asarray(
                [
                    parent_lookup[(int(coord[0] // 2), int(coord[1] // 2), int(coord[2] // 2))]
                    for coord in coords
                ],
                dtype=np.int64,
            )
        if level < leaf_level:
            children_lists: list[list[int]] = [[] for _ in range(coords.shape[0])]
            next_parent_lookup = {
                tuple(int(v) for v in coords[idx]): idx for idx in range(coords.shape[0])
            }
            for child_idx, child_coord in enumerate(coords_by_level[level + 1]):
                parent_idx = next_parent_lookup[
                    (int(child_coord[0] // 2), int(child_coord[1] // 2), int(child_coord[2] // 2))
                ]
                children_lists[int(parent_idx)].append(int(child_idx))
            children = tuple(np.asarray(items, dtype=np.int64) for items in children_lists)
        else:
            children = tuple(np.zeros((0,), dtype=np.int64) for _ in range(coords.shape[0]))

        if level == 0:
            children_by_parent_current = tuple(np.zeros((0,), dtype=np.int64) for _ in range(1))
        else:
            parent_count = int(coords_by_level[level - 1].shape[0])
            current_children_lists: list[list[int]] = [[] for _ in range(parent_count)]
            for child_idx, parent_idx_raw in enumerate(parent_indices):
                parent_idx = int(parent_idx_raw)
                current_children_lists[parent_idx].append(int(child_idx))
            children_by_parent_current = tuple(
                np.asarray(items, dtype=np.int64) for items in current_children_lists
            )

        half_size = float(partition.root_half_size) / float(1 << level)
        level_box_order = (
            box_order_rokhlin_like(
                particle_lmax=int(lmax),
                k=float(k),
                box_half_size=half_size,
                accuracy=int(accuracy_level),
                additive=int(order_additive),
            )
            if box_order is None
            else int(box_order)
        )
        if level < hf_start_level or level > hf_end_level:
            directional = dummy_directional
        else:
            if level_progress is not None:
                level_progress.set_postfix_str(
                    f"L{level}: directional basis order={level_box_order}",
                    refresh=True,
                )
            directional = structured_directional_transforms(
                int(level_box_order), grid_order=int(level_box_order)
            )
        if level == 0:
            far_offset_batches = {}
        else:
            if level_progress is not None:
                level_progress.set_postfix_str(f"L{level}: far schedule", refresh=True)
            far_offset_batches = _build_multilevel_far_offset_batches(
                coords=coords,
                parent_indices=parent_indices,
                near_neighbors=near_neighbors_by_level[level],
                parent_near_neighbors=near_neighbors_by_level[level - 1],
                children_by_parent=children_by_parent_current,
            )
        offset_diagonals: dict[tuple[int, int, int], np.ndarray] = {}
        if hf_start_level <= level <= hf_end_level:
            if level_progress is not None:
                level_progress.set_postfix_str(
                    f"L{level}: translators order={level_box_order} offsets={len(far_offset_batches)}",
                    refresh=True,
                )
            offset_diagonals = {
                offset: _sampled_rokhlin_translator(
                    _offset_delta_from_half_size(half_size, offset),
                    k=complex(k),
                    truncation_order=int(level_box_order),
                    directions=directional.grid.directions,
                    weights=directional.grid.weights,
                    dtype=out_dtype,
                )
                for offset in far_offset_batches
            }
        if level_progress is not None:
            level_progress.set_postfix_str(f"L{level}: done", refresh=False)
        levels.append(
            MLFMMLevelOperators(
                level=level,
                coords=coords,
                centers=_coords_to_centers(
                    coords,
                    root_center=np.asarray(partition.root_center, dtype=float),
                    root_half_size=float(partition.root_half_size),
                    level=level,
                ),
                parent_indices=parent_indices,
                children=children,
                box_order=int(level_box_order),
                translator_order=int(level_box_order),
                grid_order=int(level_box_order),
                directional=directional,
                far_offset_batches=far_offset_batches
                if hf_start_level <= level <= hf_end_level
                else {},
                offset_diagonals=offset_diagonals,
            )
        )

    transfers: list[MLFMMTransferOperators] = []
    for child_level in range(max(1, hf_start_level + 1), hf_end_level + 1):
        parent_level = child_level - 1
        child = levels[child_level]
        parent = levels[parent_level]
        interpolation = directional_interpolation(child.grid_order, parent.grid_order)
        anterpolation = directional_anterpolation(child.grid_order, parent.grid_order)
        grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
        for child_idx, parent_idx in enumerate(child.parent_indices):
            shift = (
                int(child.coords[child_idx, 0] - 2 * parent.coords[int(parent_idx), 0]),
                int(child.coords[child_idx, 1] - 2 * parent.coords[int(parent_idx), 1]),
                int(child.coords[child_idx, 2] - 2 * parent.coords[int(parent_idx), 2]),
            )
            grouped.setdefault(shift, []).append((int(child_idx), int(parent_idx)))
        batches = {
            shift: (
                np.asarray([pair[0] for pair in pairs], dtype=np.int64),
                np.asarray([pair[1] for pair in pairs], dtype=np.int64),
            )
            for shift, pairs in grouped.items()
        }
        phase_up: dict[tuple[int, int, int], np.ndarray] = {}
        phase_down: dict[tuple[int, int, int], np.ndarray] = {}
        for shift, (child_indices, parent_indices) in batches.items():
            up_delta = np.asarray(
                parent.centers[int(parent_indices[0])] - child.centers[int(child_indices[0])],
                dtype=float,
            )
            phase_up[shift] = np.asarray(
                np.exp(1j * complex(k) * (parent.directional.grid.directions @ up_delta)),
                dtype=out_dtype,
            )
            phase_down[shift] = np.asarray(
                np.exp(1j * complex(k) * (parent.directional.grid.directions @ (-up_delta))),
                dtype=out_dtype,
            )
        transfers.append(
            MLFMMTransferOperators(
                child_level=child_level,
                parent_level=parent_level,
                interpolation=interpolation,
                anterpolation=anterpolation,
                batches_by_shift=batches,
                phase_up_by_shift=phase_up,
                phase_down_by_shift=phase_down,
            )
        )

    backend = str(leaf_map_backend).strip().lower()
    aggregation: tuple[np.ndarray, ...]
    receive: tuple[np.ndarray, ...]
    if backend == "numpy":
        if bool(build_leaf_maps):
            leaf_groups = _build_numpy_leaf_apply_groups_dense(
                lmax=int(lmax),
                box_order=int(levels[leaf_level].box_order),
                k=float(k),
                positions=np.asarray(positions, dtype=float),
                partition=partition,
                radial_lut=radial_lut,
                dtype=out_dtype,
            )
            leaf_apply_mode: MLFMMLeafApplyMode = "dense"
        else:
            leaf_groups = _build_numpy_leaf_apply_groups_otf(
                lmax=int(lmax),
                positions=np.asarray(positions, dtype=float),
                partition=partition,
            )
            leaf_apply_mode = "on_the_fly"
        aggregation = tuple()
        receive = tuple()
    elif bool(build_leaf_maps):
        aggregation, receive = _build_leaf_box_maps(
            lmax=int(lmax),
            box_order=int(levels[leaf_level].box_order),
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            partition=partition,
            radial_lut=radial_lut,
            dtype=out_dtype,
            leaf_map_backend=leaf_map_backend,
        )
        leaf_groups = tuple()
        leaf_apply_mode = "dense"
    else:
        aggregation = tuple()
        receive = tuple()
        leaf_groups = tuple()
        leaf_apply_mode = "on_the_fly"
    return MLFMMMultilevelOperators(
        partition=partition,
        levels=tuple(levels),
        transfers=tuple(transfers),
        leaf_level=leaf_level,
        hf_start_level=hf_start_level,
        hf_end_level=hf_end_level,
        aggregation=aggregation,
        receive=receive,
        leaf_groups=leaf_groups,
        leaf_apply_mode=leaf_apply_mode,
    )


def _apply_linear_map_to_channel_batches(
    channel_batches: np.ndarray,
    interpolation: MLFMMDirectionalInterpolation,
) -> np.ndarray:
    """Apply one sparse directional interpolation operator to batched channel samples."""

    source_size = int(interpolation.source_order)
    target_size = int(interpolation.target_order)
    del source_size, target_size
    arr = np.asarray(channel_batches)
    flat = arr.reshape(-1, arr.shape[-1])
    mapped = flat @ interpolation.matrix.T
    return np.asarray(mapped, dtype=arr.dtype).reshape(
        *arr.shape[:-1], interpolation.matrix.shape[0]
    )


def apply_multilevel_mlfmm(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    operators: MLFMMMultilevelOperators,
    radial_lut: RadialLUT | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    near_dtype: np.dtype | type[np.complexfloating] | type[np.complex128] | None = None,
    far_dtype: np.dtype | type[np.complexfloating] | type[np.complex128] | None = None,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact leaf-near interactions and multilevel sampled far interactions.

    The far field is assembled by upward transfer, same-level directional
    translation, downward transfer, and leaf-local receive maps, while the
    near field remains an exact leaf-partition correction.
    """

    out_dtype = np.dtype(dtype)
    if not operators.leaf_groups and (
        len(operators.aggregation) == 0 or len(operators.receive) == 0
    ):
        raise RuntimeError(
            "multilevel MLFMM operators are missing reusable leaf apply data. "
            "This operator was likely prepared for CuPy host-cache conversion only."
        )
    near_out_dtype = np.dtype(out_dtype if near_dtype is None else near_dtype)
    far_out_dtype = np.dtype(out_dtype if far_dtype is None else far_dtype)
    y_near = _exact_leaf_near_apply(
        lmax=int(lmax),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x, dtype=near_out_dtype),
        partition=operators.partition,
        radial_lut=radial_lut,
        dtype=near_out_dtype,
        block_cache=block_cache,
    )
    outgoing = [
        np.zeros(
            (level.coords.shape[0], 4, level.directional.grid.directions.shape[0]),
            dtype=far_out_dtype,
        )
        for level in operators.levels
    ]
    incoming = [np.zeros_like(values, dtype=far_out_dtype) for values in outgoing]

    leaf_level = int(operators.leaf_level)
    if operators.leaf_groups:
        leaf_states = _aggregate_leaf_box_states(
            lmax=int(lmax),
            box_order=int(operators.levels[leaf_level].box_order),
            k=float(k),
            x=np.asarray(x, dtype=far_out_dtype),
            leaf_groups=operators.leaf_groups,
            n_leaves=len(operators.partition.leaves),
            radial_lut=radial_lut,
            dtype=far_out_dtype,
        )
    else:
        leaf_states = _leaf_box_states_from_dense_maps(
            lmax=int(lmax),
            positions=np.asarray(positions, dtype=float),
            x=np.asarray(x, dtype=far_out_dtype),
            partition=operators.partition,
            aggregation=operators.aggregation,
            dtype=far_out_dtype,
        )
    for leaf_id, box_state in enumerate(leaf_states):
        channels = box_outgoing_to_directional(operators.levels[leaf_level].directional, box_state)
        for chan_idx, channel in enumerate(channels):
            outgoing[leaf_level][leaf_id, chan_idx] = np.asarray(channel, dtype=far_out_dtype)

    # Upward transfer uses the reflection-indexed directional ordering expected
    # by the sparse interpolation tables. We therefore:
    # 1. reindex child samples from the physical directional ordering into the
    #    interpolation ordering,
    # 2. interpolate onto the parent directional grid in that reflected basis,
    # 3. reindex back to the parent physical ordering,
    # 4. attach the parent-center translation phase, because the interpolated
    #    samples still represent the child-centered outgoing field.
    for transfer in reversed(operators.transfers):
        child_values = outgoing[int(transfer.child_level)]
        parent_values = outgoing[int(transfer.parent_level)]
        child_level = operators.levels[int(transfer.child_level)]
        parent_level = operators.levels[int(transfer.parent_level)]
        for shift, (child_idx, parent_idx) in transfer.batches_by_shift.items():
            child_reindexed = _apply_reflection_to_channel_batches(
                child_values[child_idx],
                child_level.directional.grid.reflection_permutation,
            )
            mapped_reindexed = _apply_linear_map_to_channel_batches(
                child_reindexed,
                transfer.interpolation,
            )
            mapped = _apply_reflection_to_channel_batches(
                mapped_reindexed,
                parent_level.directional.grid.reflection_permutation,
            )
            mapped *= np.asarray(transfer.phase_up_by_shift[shift], dtype=far_out_dtype)[
                None, None, :
            ]
            np.add.at(parent_values, parent_idx, mapped)

    for level_idx in range(int(operators.hf_start_level), int(operators.hf_end_level) + 1):
        level = operators.levels[level_idx]
        for offset, (src_idx, dst_idx) in level.far_offset_batches.items():
            translated = (
                outgoing[level_idx][src_idx] * level.offset_diagonals[offset][None, None, :]
            )
            np.add.at(incoming[level_idx], dst_idx, translated)

    # Downward transfer mirrors the same convention change in the opposite
    # direction. The parent local samples first receive the child-center phase
    # shift, then move into the reflection-indexed interpolation basis, then
    # anterpolate onto the child grid, and finally return to the child physical
    # directional ordering before accumulating into the child local field.
    for transfer in operators.transfers:
        parent_values = incoming[int(transfer.parent_level)]
        child_values = incoming[int(transfer.child_level)]
        child_level = operators.levels[int(transfer.child_level)]
        parent_level = operators.levels[int(transfer.parent_level)]
        for shift, (child_idx, parent_idx) in transfer.batches_by_shift.items():
            shifted = (
                parent_values[parent_idx]
                * np.asarray(
                    transfer.phase_down_by_shift[shift],
                    dtype=far_out_dtype,
                )[None, None, :]
            )
            shifted_reindexed = _apply_reflection_to_channel_batches(
                shifted,
                parent_level.directional.grid.reflection_permutation,
            )
            mapped_reindexed = _apply_linear_map_to_channel_batches(
                shifted_reindexed,
                transfer.anterpolation,
            )
            mapped = _apply_reflection_to_channel_batches(
                mapped_reindexed,
                child_level.directional.grid.reflection_permutation,
            )
            np.add.at(child_values, child_idx, mapped)

    box_nm = n_modes(int(operators.levels[leaf_level].box_order))
    incoming_box = np.zeros((len(operators.partition.leaves), box_nm), dtype=far_out_dtype)
    for leaf_id in range(len(operators.partition.leaves)):
        incoming_box[leaf_id] = directional_to_box_regular(
            operators.levels[leaf_level].directional,
            incoming[leaf_level][leaf_id, 0],
            incoming[leaf_level][leaf_id, 1],
            incoming[leaf_level][leaf_id, 2],
            incoming[leaf_level][leaf_id, 3],
        )

    if operators.leaf_groups:
        y_far = _receive_leaf_boxes_to_particles(
            lmax=int(lmax),
            box_order=int(operators.levels[leaf_level].box_order),
            k=float(k),
            n_particles=np.asarray(positions).shape[0],
            incoming_box=incoming_box,
            leaf_groups=operators.leaf_groups,
            radial_lut=radial_lut,
            dtype=far_out_dtype,
        )
    else:
        nm = n_modes(int(lmax))
        ns = np.asarray(positions).shape[0]
        y_far = np.zeros((ns, nm), dtype=far_out_dtype)
        for leaf in operators.partition.leaves:
            contribution = operators.receive[int(leaf.id)] @ incoming_box[int(leaf.id)]
            y_far[leaf.particle_indices] += contribution.reshape(leaf.particle_indices.size, nm)
    return y_near.reshape(-1), y_far.reshape(-1)


def prepare_mlfmm_coupling(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    particle_circumscribing_radii: np.ndarray,
    radial_lut: RadialLUT,
    ab5: np.ndarray | None = None,
    options: MLFMMOptions | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    cache_translation_blocks: bool = False,
    show_progress: bool = False,
    leaf_map_backend: Literal["numpy", "cupy"] = "numpy",
    build_leaf_maps: bool = True,
) -> CouplingOperator:
    """Prepare the native NumPy MLFMM coupling operator or direct fallback.

    The resolved stage is determined from the uniform-depth occupied-box plan.
    If the geometry resolves to the direct stage, this helper returns the
    canonical pairwise coupling backend directly instead of wrapping it in an
    MLFMM object.

    For non-direct MLFMM stages, sampled-far operators are always prepared in
    complex128 while exact-near interactions keep the requested compute dtype.
    """

    t_prepare_start = perf_counter()
    out_dtype = np.dtype(dtype)
    leaf_backend = str(leaf_map_backend).strip().lower()
    if leaf_backend not in {"numpy", "cupy"}:
        raise ValueError(
            f"Unsupported leaf-map backend {leaf_map_backend!r}. Use 'numpy' or 'cupy'."
        )
    leaf_backend_lit: Literal["numpy", "cupy"]
    leaf_backend_lit = "cupy" if leaf_backend == "cupy" else "numpy"
    near_out_dtype = np.dtype(dtype)
    far_out_dtype = np.dtype(np.complex128)
    pts = np.asarray(positions, dtype=float)
    radii = np.asarray(particle_circumscribing_radii, dtype=float)
    resolved_options = MLFMMOptions() if options is None else options
    resolved = resolve_mlfmm_plan(
        pts,
        particle_circumscribing_radii=radii,
        options=resolved_options,
    )
    radial_lut_hf = radial_lut
    if resolved.stage != "direct" and np.dtype(radial_lut.dtype) != np.dtype(np.complex128):
        radial_lut_hf = RadialLUT(
            lmax=int(lmax),
            k=float(k),
            r_max=float(radial_lut.r_grid[-1]),
            dr=float(radial_lut.dr),
            dtype=np.complex128,
        )
    if show_progress:
        occupancies = np.asarray(
            [leaf.particle_indices.size for leaf in resolved.partition.leaves],
            dtype=np.int64,
        )
        occ_min = int(np.min(occupancies)) if occupancies.size else 0
        occ_med = int(np.median(occupancies)) if occupancies.size else 0
        occ_max = int(np.max(occupancies)) if occupancies.size else 0
        tqdm.write(
            "[MLFMM] plan "
            f"stage={resolved.stage} depth={resolved.selected_depth} "
            f"leaves={len(resolved.partition.leaves)} "
            f"leaf_occ_min/med/max={occ_min}/{occ_med}/{occ_max} "
            f"elapsed_s={perf_counter() - t_prepare_start:.2f}"
        )
    out: CouplingOperator
    if resolved.stage == "direct":
        t_stage_start = perf_counter()
        ab5_table = (
            np.asarray(ab5, dtype=out_dtype)
            if ab5 is not None
            else translation_ab5_table(int(lmax), dtype=out_dtype)
        )
        out = PairwiseCouplingOperator(
            lmax=int(lmax),
            k=float(k),
            positions=pts,
            ab5=ab5_table,
            radial_lut=radial_lut,
            dtype=out_dtype,
            cache_translation_blocks=bool(cache_translation_blocks),
        )
        if show_progress:
            tqdm.write(
                f"[MLFMM] prepare stage=direct elapsed_s={perf_counter() - t_stage_start:.2f}"
            )
            tqdm.write(f"[MLFMM] prepare total elapsed_s={perf_counter() - t_prepare_start:.2f}")
        return out
    if resolved.stage == "single_level":
        t_stage_start = perf_counter()
        single_level = build_single_level_mlfmm_operators(
            lmax=int(lmax),
            k=float(k),
            positions=pts,
            partition=resolved.partition,
            radial_lut=radial_lut_hf,
            accuracy_level=int(resolved_options.accuracy_level),
            order_additive=int(resolved_options.order_additive),
            dtype=far_out_dtype,
            leaf_map_backend=leaf_backend_lit,
            build_leaf_maps=bool(build_leaf_maps),
        )
        out = MLFMMCouplingOperator(
            lmax=int(lmax),
            k=float(k),
            positions=pts,
            radial_lut=radial_lut_hf,
            resolved_plan=resolved,
            dtype=out_dtype,
            near_dtype=near_out_dtype,
            far_dtype=far_out_dtype,
            cache_translation_blocks=bool(cache_translation_blocks),
            single_level=single_level,
        )
        if show_progress:
            tqdm.write(
                f"[MLFMM] prepare stage=single_level elapsed_s={perf_counter() - t_stage_start:.2f}"
            )
            tqdm.write(f"[MLFMM] prepare total elapsed_s={perf_counter() - t_prepare_start:.2f}")
        return out
    t_stage_start = perf_counter()
    multilevel = build_multilevel_mlfmm_operators(
        lmax=int(lmax),
        k=float(k),
        positions=pts,
        partition=resolved.partition,
        radial_lut=radial_lut_hf,
        accuracy_level=int(resolved_options.accuracy_level),
        order_additive=int(resolved_options.order_additive),
        dtype=far_out_dtype,
        leaf_map_backend=leaf_backend_lit,
        build_leaf_maps=bool(build_leaf_maps),
        show_progress=bool(show_progress),
        hf_start_level=resolved_options.hf_start_level,
        hf_wavelength_divisor=float(resolved_options.hf_wavelength_divisor),
    )
    out = MLFMMCouplingOperator(
        lmax=int(lmax),
        k=float(k),
        positions=pts,
        radial_lut=radial_lut_hf,
        resolved_plan=resolved,
        dtype=out_dtype,
        near_dtype=near_out_dtype,
        far_dtype=far_out_dtype,
        cache_translation_blocks=bool(cache_translation_blocks),
        multilevel=multilevel,
    )
    if show_progress:
        tqdm.write(
            f"[MLFMM] prepare stage=multilevel elapsed_s={perf_counter() - t_stage_start:.2f}"
        )
        tqdm.write(f"[MLFMM] prepare total elapsed_s={perf_counter() - t_prepare_start:.2f}")
    return out


__all__ = [
    "MLFMMCouplingOperator",
    "MLFMMLevelOperators",
    "MLFMMMultilevelOperators",
    "MLFMMOptions",
    "MLFMMResolvedPlan",
    "MLFMMSingleLevelOperators",
    "MLFMMStage",
    "MLFMMTransferOperators",
    "apply_multilevel_mlfmm",
    "apply_single_level_mlfmm",
    "box_order_rokhlin_like",
    "build_multilevel_mlfmm_operators",
    "build_single_level_mlfmm_operators",
    "estimate_rokhlin_order",
    "prepare_mlfmm_coupling",
    "resolve_mlfmm_plan",
    "select_mlfmm_stage",
]
