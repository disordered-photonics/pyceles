from __future__ import annotations

"""Plan-resolution helpers for the pyceles-native MLFMM coupling backend."""

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
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
    MLFMMDirectionalTransforms,
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_interpolation,
    directional_to_box_regular,
    directional_transforms,
)
from .mlfmm_partition import (
    MLFMMPartition,
    build_uniform_mlfmm_partition,
    root_cube,
    uniform_box_coords,
    validate_leaf_size_floor,
)

MLFMMStage = Literal["direct", "single_level", "multilevel"]
_ROKHLIN_MINIMUM_ORDERS = (3, 7, 11, 17, 24, 30)
"""Conservative minimum truncation orders for discrete accuracy levels,
inspired from the heuristic values used in FasTMM.

These floors stabilize the small-ka regime where the asymptotic Rokhlin-style
estimate underpredicts the required order. The values are inherited from the
validated reference implementation used during pyceles MLFMM development.
"""


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


@dataclass(frozen=True)
class MLFMMSingleLevelOperators:
    """Prepared single-level HF operators over one uniform occupied leaf level."""

    partition: MLFMMPartition
    box_order: int
    translator_order: int
    grid_order: int
    directional: MLFMMDirectionalTransforms
    aggregation: tuple[np.ndarray, ...]
    receive: tuple[np.ndarray, ...]
    leaf_cell_coords: dict[int, tuple[int, int, int]]
    far_offset_batches: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[tuple[int, int, int], np.ndarray]
    exact_box_blocks: dict[tuple[int, int, int], np.ndarray] | None


@dataclass(frozen=True)
class MLFMMLevelOperators:
    """Prepared sampled HF data for one occupied hierarchy level."""

    level: int
    coords: np.ndarray
    centers: np.ndarray
    parent_indices: np.ndarray
    children: tuple[np.ndarray, ...]
    box_order: int
    translator_order: int
    grid_order: int
    directional: MLFMMDirectionalTransforms
    far_offset_batches: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[tuple[int, int, int], np.ndarray]


@dataclass(frozen=True)
class MLFMMTransferOperators:
    """Parent/child sampled transfer operators between adjacent occupied levels."""

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
    aggregation: tuple[np.ndarray, ...]
    receive: tuple[np.ndarray, ...]


@dataclass
class MLFMMCouplingOperator:
    """Prepared NumPy MLFMM coupling operator with structured stage metadata.

    The near part stays exact on the resolved leaf partition. The far part is
    applied through either a single occupied-leaf sampled level or a multilevel
    occupied-box hierarchy, depending on the resolved stage.
    """

    lmax: int
    k: float
    positions: np.ndarray
    radial_lut: RadialLUT
    resolved_plan: MLFMMResolvedPlan
    dtype: np.dtype = np.dtype(np.complex128)
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
        if self.cache_translation_blocks and self._exact_block_cache is None:
            self._exact_block_cache = {}

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the prepared MLFMM coupling operator `W`."""

        if self.resolved_plan.stage == "single_level":
            if self.single_level is None:
                raise RuntimeError("Internal error: single-level operators are missing.")
            y_near, y_far = apply_single_level_mlfmm(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                operators=self.single_level,
                radial_lut=self.radial_lut,
                dtype=self.dtype,
                block_cache=self._exact_block_cache,
            )
            return np.asarray(y_near + y_far, dtype=self.dtype)
        if self.resolved_plan.stage == "multilevel":
            if self.multilevel is None:
                raise RuntimeError("Internal error: multilevel operators are missing.")
            y_near, y_far = apply_multilevel_mlfmm(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                operators=self.multilevel,
                radial_lut=self.radial_lut,
                dtype=self.dtype,
                block_cache=self._exact_block_cache,
            )
            return np.asarray(y_near + y_far, dtype=self.dtype)
        raise RuntimeError(f"Unsupported MLFMM stage {self.resolved_plan.stage!r}.")

    def populate(self, *, show_progress: bool = False) -> None:
        """Optionally precompute exact near blocks used by the current partition."""

        del show_progress
        if not self.cache_translation_blocks:
            return
        if self._exact_block_cache is None:
            self._exact_block_cache = {}
        _exact_leaf_near_apply(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            x=np.zeros((self.positions.shape[0] * n_modes(int(self.lmax)),), dtype=self.dtype),
            partition=self.resolved_plan.partition,
            radial_lut=self.radial_lut,
            dtype=np.dtype(self.dtype),
            block_cache=self._exact_block_cache,
        )


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
    Rokhlin-style estimate used by the NumPy MLFMM path.
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
    """Group directed far leaf interactions by relative offset."""

    grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for a, b in partition.leaf_far_pairs:
        offset = (
            int(cell_coords[b][0] - cell_coords[a][0]),
            int(cell_coords[b][1] - cell_coords[a][1]),
            int(cell_coords[b][2] - cell_coords[a][2]),
        )
        grouped.setdefault(offset, []).append((a, b))
        reverse_offset = (-offset[0], -offset[1], -offset[2])
        grouped.setdefault(reverse_offset, []).append((b, a))
    return {
        offset: (
            np.asarray([pair[0] for pair in pairs], dtype=np.int64),
            np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        )
        for offset, pairs in grouped.items()
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


def _build_leaf_box_maps(
    *,
    lmax: int,
    box_order: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    dtype: np.dtype,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build particle-to-box aggregation maps and box-to-particle receive maps."""

    full_order = max(int(lmax), int(box_order))
    ab5 = translation_ab5_table(full_order, dtype=np.complex128)
    aggregation: list[np.ndarray] = []
    receive: list[np.ndarray] = []
    for leaf in partition.leaves:
        agg_blocks = [
            translation_block_rect(
                int(box_order),
                int(lmax),
                float(k),
                np.asarray(leaf.center - positions[int(pidx)], dtype=float),
                ab5=ab5,
                radial_lut=radial_lut,
                family="interior",
            )
            for pidx in leaf.particle_indices
        ]
        aggregation.append(np.hstack(agg_blocks).astype(dtype, copy=False))
        recv_blocks = [
            translation_block_rect(
                int(lmax),
                int(box_order),
                float(k),
                np.asarray(positions[int(pidx)] - leaf.center, dtype=float),
                ab5=ab5,
                radial_lut=radial_lut,
                family="interior",
            )
            for pidx in leaf.particle_indices
        ]
        receive.append(np.vstack(recv_blocks).astype(dtype, copy=False))
    return tuple(aggregation), tuple(receive)


def _leaf_box_states(
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


def build_single_level_mlfmm_operators(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None = None,
    box_order: int | None = None,
    translator_order: int | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    build_exact_box_blocks: bool = False,
) -> MLFMMSingleLevelOperators:
    """Build the sampled single-level HF far operator over one occupied leaf level."""

    if not partition.leaves:
        raise ValueError("single-level MLFMM requires at least one occupied leaf.")
    leaf_half_size = float(partition.leaves[0].half_size)
    shared_box_order = (
        box_order_rokhlin_like(
            particle_lmax=int(lmax),
            k=float(k),
            box_half_size=leaf_half_size,
        )
        if box_order is None
        else int(box_order)
    )
    shared_translator_order = (
        int(shared_box_order) if translator_order is None else int(translator_order)
    )
    shared_grid_order = max(int(shared_box_order), int(shared_translator_order))
    out_dtype = np.dtype(dtype)
    directional = directional_transforms(int(shared_box_order), grid_order=int(shared_grid_order))
    aggregation, receive = _build_leaf_box_maps(
        lmax=int(lmax),
        box_order=int(shared_box_order),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        partition=partition,
        radial_lut=radial_lut,
        dtype=out_dtype,
    )
    leaf_cell_coords = _leaf_cell_coords(partition)
    far_offset_batches = _leaf_offset_batches(partition, leaf_cell_coords)
    offset_diagonals: dict[tuple[int, int, int], np.ndarray] = {}
    exact_box_blocks: dict[tuple[int, int, int], np.ndarray] | None = (
        {} if build_exact_box_blocks else None
    )
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
        if exact_box_blocks is not None:
            exact_box_blocks[offset] = np.asarray(
                translation_block_rect(
                    int(shared_box_order),
                    int(shared_box_order),
                    float(k),
                    np.asarray(delta, dtype=float),
                    ab5=translation_ab5_table(int(shared_box_order), dtype=np.complex128),
                    radial_lut=radial_lut,
                ),
                dtype=out_dtype,
            )
    return MLFMMSingleLevelOperators(
        partition=partition,
        box_order=int(shared_box_order),
        translator_order=int(shared_translator_order),
        grid_order=int(shared_grid_order),
        directional=directional,
        aggregation=aggregation,
        receive=receive,
        leaf_cell_coords=leaf_cell_coords,
        far_offset_batches=far_offset_batches,
        offset_diagonals=offset_diagonals,
        exact_box_blocks=exact_box_blocks,
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
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact leaf-near interactions and sampled single-level far interactions."""

    out_dtype = np.dtype(dtype)
    y_near = _exact_leaf_near_apply(
        lmax=int(lmax),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        radial_lut=radial_lut,
        dtype=out_dtype,
        block_cache=block_cache,
    )
    leaf_states = _leaf_box_states(
        lmax=int(lmax),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        aggregation=operators.aggregation,
        dtype=out_dtype,
    )
    ndir = int(operators.directional.grid.directions.shape[0])
    outgoing = np.zeros((len(operators.partition.leaves), 4, ndir), dtype=out_dtype)
    for leaf_id, box_state in enumerate(leaf_states):
        channels = box_outgoing_to_directional(operators.directional, box_state)
        for chan_idx, channel in enumerate(channels):
            outgoing[leaf_id, chan_idx] = np.asarray(channel, dtype=out_dtype)

    incoming = np.zeros_like(outgoing, dtype=out_dtype)
    for offset, (src_idx, dst_idx) in operators.far_offset_batches.items():
        translated = outgoing[src_idx] * operators.offset_diagonals[offset][None, None, :]
        np.add.at(incoming, dst_idx, translated)

    box_nm = n_modes(int(operators.box_order))
    incoming_box = np.zeros((len(operators.partition.leaves), box_nm), dtype=out_dtype)
    for leaf_id in range(len(operators.partition.leaves)):
        incoming_box[leaf_id] = directional_to_box_regular(
            operators.directional,
            incoming[leaf_id, 0],
            incoming[leaf_id, 1],
            incoming[leaf_id, 2],
            incoming[leaf_id, 3],
        )

    nm = n_modes(int(lmax))
    ns = np.asarray(positions).shape[0]
    y_far = np.zeros((ns, nm), dtype=out_dtype)
    for leaf in operators.partition.leaves:
        contribution = operators.receive[int(leaf.id)] @ incoming_box[int(leaf.id)]
        y_far[leaf.particle_indices] += contribution.reshape(leaf.particle_indices.size, nm)
    return y_near.reshape(-1), y_far.reshape(-1)


def apply_single_level_mlfmm_exact_box_reference(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    operators: MLFMMSingleLevelOperators,
    radial_lut: RadialLUT | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact leaf-near interactions and exact grouped-box far interactions."""

    if operators.exact_box_blocks is None:
        raise ValueError("single-level exact box blocks were not built.")
    out_dtype = np.dtype(dtype)
    y_near = _exact_leaf_near_apply(
        lmax=int(lmax),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        radial_lut=radial_lut,
        dtype=out_dtype,
        block_cache=block_cache,
    )
    leaf_states = _leaf_box_states(
        lmax=int(lmax),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        aggregation=operators.aggregation,
        dtype=out_dtype,
    )
    incoming_box = np.zeros_like(leaf_states, dtype=out_dtype)
    for offset, (src_idx, dst_idx) in operators.far_offset_batches.items():
        block = np.asarray(operators.exact_box_blocks[offset], dtype=out_dtype)
        translated = (block[None, :, :] @ leaf_states[src_idx, :, None]).reshape(
            src_idx.size, block.shape[0]
        )
        np.add.at(incoming_box, dst_idx, translated)

    nm = n_modes(int(lmax))
    ns = np.asarray(positions).shape[0]
    y_far = np.zeros((ns, nm), dtype=out_dtype)
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
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
) -> MLFMMMultilevelOperators:
    """Build multilevel sampled HF operators over occupied boxes only."""

    if not partition.leaves:
        raise ValueError("multilevel MLFMM requires at least one occupied leaf.")
    out_dtype = np.dtype(dtype)
    leaf_level = int(partition.depth)
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

    levels: list[MLFMMLevelOperators] = []
    for level in range(leaf_level + 1):
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

        half_size = float(partition.root_half_size) / float(1 << level)
        level_box_order = (
            box_order_rokhlin_like(
                particle_lmax=int(lmax),
                k=float(k),
                box_half_size=half_size,
            )
            if box_order is None
            else int(box_order)
        )
        directional = directional_transforms(int(level_box_order), grid_order=int(level_box_order))
        far_offset_batches = {} if level == 0 else _coords_far_offset_batches(coords)
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
                far_offset_batches=far_offset_batches,
                offset_diagonals=offset_diagonals,
            )
        )

    transfers: list[MLFMMTransferOperators] = []
    for child_level in range(1, leaf_level + 1):
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

    aggregation, receive = _build_leaf_box_maps(
        lmax=int(lmax),
        box_order=int(levels[leaf_level].box_order),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        partition=partition,
        radial_lut=radial_lut,
        dtype=out_dtype,
    )
    return MLFMMMultilevelOperators(
        partition=partition,
        levels=tuple(levels),
        transfers=tuple(transfers),
        leaf_level=leaf_level,
        aggregation=aggregation,
        receive=receive,
    )


def _apply_linear_map_to_channel_batches(
    channel_batches: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply one directional interpolation matrix to a batch of 4-channel samples."""

    return np.einsum(
        "bcn,tn->bct",
        np.asarray(channel_batches),
        np.asarray(matrix),
        optimize=True,
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
    block_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exact leaf-near interactions and multilevel sampled far interactions."""

    out_dtype = np.dtype(dtype)
    y_near = _exact_leaf_near_apply(
        lmax=int(lmax),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        radial_lut=radial_lut,
        dtype=out_dtype,
        block_cache=block_cache,
    )
    outgoing = [
        np.zeros(
            (level.coords.shape[0], 4, level.directional.grid.directions.shape[0]),
            dtype=out_dtype,
        )
        for level in operators.levels
    ]
    incoming = [np.zeros_like(values, dtype=out_dtype) for values in outgoing]

    leaf_level = int(operators.leaf_level)
    leaf_states = _leaf_box_states(
        lmax=int(lmax),
        positions=np.asarray(positions, dtype=float),
        x=np.asarray(x),
        partition=operators.partition,
        aggregation=operators.aggregation,
        dtype=out_dtype,
    )
    for leaf_id, box_state in enumerate(leaf_states):
        channels = box_outgoing_to_directional(operators.levels[leaf_level].directional, box_state)
        for chan_idx, channel in enumerate(channels):
            outgoing[leaf_level][leaf_id, chan_idx] = np.asarray(channel, dtype=out_dtype)

    for transfer in reversed(operators.transfers):
        child_values = outgoing[int(transfer.child_level)]
        parent_values = outgoing[int(transfer.parent_level)]
        for shift, (child_idx, parent_idx) in transfer.batches_by_shift.items():
            mapped = _apply_linear_map_to_channel_batches(
                child_values[child_idx],
                np.asarray(transfer.interpolation.matrix, dtype=out_dtype),
            )
            mapped *= np.asarray(transfer.phase_up_by_shift[shift], dtype=out_dtype)[None, None, :]
            np.add.at(parent_values, parent_idx, mapped)

    for level in operators.levels[1:]:
        for offset, (src_idx, dst_idx) in level.far_offset_batches.items():
            translated = (
                outgoing[int(level.level)][src_idx] * level.offset_diagonals[offset][None, None, :]
            )
            np.add.at(incoming[int(level.level)], dst_idx, translated)

    for transfer in operators.transfers:
        parent_values = incoming[int(transfer.parent_level)]
        child_values = incoming[int(transfer.child_level)]
        for shift, (child_idx, parent_idx) in transfer.batches_by_shift.items():
            shifted = (
                parent_values[parent_idx]
                * np.asarray(
                    transfer.phase_down_by_shift[shift],
                    dtype=out_dtype,
                )[None, None, :]
            )
            mapped = _apply_linear_map_to_channel_batches(
                shifted,
                np.asarray(transfer.anterpolation.matrix, dtype=out_dtype),
            )
            np.add.at(child_values, child_idx, mapped)

    box_nm = n_modes(int(operators.levels[leaf_level].box_order))
    incoming_box = np.zeros((len(operators.partition.leaves), box_nm), dtype=out_dtype)
    for leaf_id in range(len(operators.partition.leaves)):
        incoming_box[leaf_id] = directional_to_box_regular(
            operators.levels[leaf_level].directional,
            incoming[leaf_level][leaf_id, 0],
            incoming[leaf_level][leaf_id, 1],
            incoming[leaf_level][leaf_id, 2],
            incoming[leaf_level][leaf_id, 3],
        )

    nm = n_modes(int(lmax))
    ns = np.asarray(positions).shape[0]
    y_far = np.zeros((ns, nm), dtype=out_dtype)
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
    ab5: np.ndarray,
    options: MLFMMOptions | None = None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128] = np.complex128,
    cache_translation_blocks: bool = False,
    show_progress: bool = False,
) -> CouplingOperator:
    """Prepare the native NumPy MLFMM coupling operator or direct fallback.

    The resolved stage is determined from the uniform-depth occupied-box plan.
    If the geometry resolves to the direct stage, this helper returns the
    canonical pairwise coupling backend directly instead of wrapping it in an
    MLFMM object.
    """

    t_prepare_start = perf_counter()
    out_dtype = np.dtype(dtype)
    pts = np.asarray(positions, dtype=float)
    radii = np.asarray(particle_circumscribing_radii, dtype=float)
    resolved = resolve_mlfmm_plan(
        pts,
        particle_circumscribing_radii=radii,
        options=options,
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
        out = PairwiseCouplingOperator(
            lmax=int(lmax),
            k=float(k),
            positions=pts,
            ab5=np.asarray(ab5, dtype=out_dtype),
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
            radial_lut=radial_lut,
            dtype=out_dtype,
        )
        out = MLFMMCouplingOperator(
            lmax=int(lmax),
            k=float(k),
            positions=pts,
            radial_lut=radial_lut,
            resolved_plan=resolved,
            dtype=out_dtype,
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
        radial_lut=radial_lut,
        dtype=out_dtype,
    )
    out = MLFMMCouplingOperator(
        lmax=int(lmax),
        k=float(k),
        positions=pts,
        radial_lut=radial_lut,
        resolved_plan=resolved,
        dtype=out_dtype,
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
    "apply_single_level_mlfmm_exact_box_reference",
    "box_order_rokhlin_like",
    "build_multilevel_mlfmm_operators",
    "build_single_level_mlfmm_operators",
    "estimate_rokhlin_order",
    "prepare_mlfmm_coupling",
    "resolve_mlfmm_plan",
    "select_mlfmm_stage",
]
