from __future__ import annotations

"""CuPy-side prepared-data wrappers for the NumPy MLFMM reference plan.

CPU build/planning remains the single source of truth in `mlfmm.py`.
This module validates and uploads repeated-apply structures to device memory.
"""

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from pyceles._optional import coerce_array, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.translation import translation_ab5_table, translation_block

from .mlfmm import (
    MLFMMCouplingOperator,
    MLFMMLevelOperators,
    MLFMMMultilevelOperators,
    MLFMMResolvedPlan,
    MLFMMSingleLevelOperators,
    MLFMMTransferOperators,
)
from .mlfmm_directional import MLFMMDirectionalInterpolation, MLFMMDirectionalTransforms
from .mlfmm_partition import MLFMMPartition

Offset3 = tuple[int, int, int]


@dataclass(frozen=True)
class CuPyDirectionalGridData:
    """Device copy of one directional sampling grid."""

    order: int
    directions: Any
    weights: Any
    reflection_permutation: Any


@dataclass(frozen=True)
class CuPyDirectionalTransformsData:
    """Device copy of SVWF<->directional transforms for one box/grid order."""

    box_order: int
    grid_order: int
    grid: CuPyDirectionalGridData
    Fth: Any
    Fph: Any
    Gth: Any
    Gph: Any
    Fth_adj: Any
    Fph_adj: Any
    Gth_adj: Any
    Gph_adj: Any


@dataclass(frozen=True)
class CuPyDirectionalInterpolationData:
    """Device copy of one sparse interpolation/anterpolation operator."""

    source_order: int
    target_order: int
    matrix: Any
    nnz: int


@dataclass(frozen=True)
class CuPyMLFMMPartitionData:
    """Device-ready encoding of occupied leaves and pair schedules."""

    root_center: Any
    root_half_size: float
    depth: int
    leaf_particle_offsets: Any
    leaf_particle_indices: Any
    leaf_near_pairs: Any
    leaf_far_pairs: Any


@dataclass(frozen=True)
class CuPyOffsetBatchData:
    """Device copy of grouped source/target index batches for one relative offset."""

    src_indices: Any
    dst_indices: Any


@dataclass(frozen=True)
class CuPyMLFMMLevelData:
    """Device-ready per-level data used in multilevel repeated applies."""

    level: int
    coords: Any
    centers: Any
    parent_indices: Any
    children_offsets: Any
    children_indices: Any
    box_order: int
    translator_order: int
    grid_order: int
    directional: CuPyDirectionalTransformsData
    far_offset_batches: dict[Offset3, CuPyOffsetBatchData]
    offset_diagonals: dict[Offset3, Any]


@dataclass(frozen=True)
class CuPyMLFMMTransferData:
    """Device-ready parent/child transfer data for one adjacent level pair."""

    child_level: int
    parent_level: int
    interpolation: CuPyDirectionalInterpolationData
    anterpolation: CuPyDirectionalInterpolationData
    batches_by_shift: dict[Offset3, CuPyOffsetBatchData]
    phase_up_by_shift: dict[Offset3, Any]
    phase_down_by_shift: dict[Offset3, Any]


@dataclass(frozen=True)
class CuPyMLFMMSingleLevelData:
    """Device-ready container for single-level repeated-apply structures."""

    box_order: int
    translator_order: int
    grid_order: int
    directional: CuPyDirectionalTransformsData
    aggregation: tuple[Any, ...]
    receive: tuple[Any, ...]
    far_offset_batches: dict[Offset3, CuPyOffsetBatchData]
    offset_diagonals: dict[Offset3, Any]


@dataclass(frozen=True)
class CuPyMLFMMNearPairData:
    """Device-ready directed exact-near particle-pair blocks."""

    dst_particle_indices: Any
    src_particle_indices: Any
    blocks: Any


@dataclass(frozen=True)
class CuPyMLFMMMultilevelData:
    """Device-ready container for multilevel repeated-apply structures."""

    levels: tuple[CuPyMLFMMLevelData, ...]
    transfers: tuple[CuPyMLFMMTransferData, ...]
    leaf_level: int
    hf_start_level: int
    hf_end_level: int
    aggregation: tuple[Any, ...]
    receive: tuple[Any, ...]


@dataclass(frozen=True)
class CuPyMLFMMPreparedData:
    """Top-level CuPy representation of a CPU-built MLFMM plan."""

    lmax: int
    k: float
    stage: str
    positions: Any
    partition: CuPyMLFMMPartitionData
    resolved_plan: MLFMMResolvedPlan
    near_pairs: CuPyMLFMMNearPairData
    single_level: CuPyMLFMMSingleLevelData | None = None
    multilevel: CuPyMLFMMMultilevelData | None = None


def _as_numpy_1d(arr: np.ndarray, *, dtype: npt.DTypeLike, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return np.ascontiguousarray(out)


def _as_numpy_2d(arr: np.ndarray, *, dtype: npt.DTypeLike, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={out.ndim}.")
    return np.ascontiguousarray(out)


def _as_numpy_3cols(arr: np.ndarray, *, dtype: npt.DTypeLike, name: str) -> np.ndarray:
    out = _as_numpy_2d(arr, dtype=dtype, name=name)
    if out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3). Got {out.shape}.")
    return out


def _pack_index_lists(
    index_lists: tuple[np.ndarray, ...] | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(index_lists) + 1,), dtype=np.int64)
    flat_parts: list[np.ndarray] = []
    cursor = 0
    for i, values in enumerate(index_lists):
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        flat_parts.append(np.ascontiguousarray(arr))
        cursor += int(arr.size)
        offsets[i + 1] = cursor
    flat = (
        np.concatenate(flat_parts, dtype=np.int64) if flat_parts else np.zeros((0,), dtype=np.int64)
    )
    return offsets, flat


def _pack_pairs(pairs: tuple[tuple[int, int], ...], *, name: str) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    out = np.asarray(pairs, dtype=np.int64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"{name} must be a list/tuple of (a, b) pairs.")
    return np.ascontiguousarray(out)


def _upload_dense_matrix(matrix: np.ndarray, *, name: str, cupy: Any) -> Any:
    arr = _as_numpy_2d(matrix, dtype=np.complex128, name=name)
    return cupy.asarray(arr, dtype=cupy.complex128)


def _upload_directional_transforms(
    transforms: MLFMMDirectionalTransforms, *, cupy: Any
) -> CuPyDirectionalTransformsData:
    grid = transforms.grid
    directions = _as_numpy_3cols(
        grid.directions, dtype=np.float64, name="directional.grid.directions"
    )
    weights = _as_numpy_1d(grid.weights, dtype=np.float64, name="directional.grid.weights")
    reflection = _as_numpy_1d(
        grid.reflection_permutation,
        dtype=np.int64,
        name="directional.grid.reflection_permutation",
    )
    if directions.shape[0] != weights.size:
        raise ValueError(
            "Directional grid directions/weights mismatch: "
            f"{directions.shape[0]} vs {weights.size}."
        )
    if reflection.size != weights.size:
        raise ValueError(
            "Directional grid reflection-permutation size mismatch: "
            f"{reflection.size} vs {weights.size}."
        )

    # Directional transforms are complex128 reference operators; keep them exact.
    fth = _as_numpy_2d(transforms.Fth, dtype=np.complex128, name="directional.Fth")
    fph = _as_numpy_2d(transforms.Fph, dtype=np.complex128, name="directional.Fph")
    gth = _as_numpy_2d(transforms.Gth, dtype=np.complex128, name="directional.Gth")
    gph = _as_numpy_2d(transforms.Gph, dtype=np.complex128, name="directional.Gph")
    fth_adj = _as_numpy_2d(transforms.Fth_adj, dtype=np.complex128, name="directional.Fth_adj")
    fph_adj = _as_numpy_2d(transforms.Fph_adj, dtype=np.complex128, name="directional.Fph_adj")
    gth_adj = _as_numpy_2d(transforms.Gth_adj, dtype=np.complex128, name="directional.Gth_adj")
    gph_adj = _as_numpy_2d(transforms.Gph_adj, dtype=np.complex128, name="directional.Gph_adj")
    n_dir = directions.shape[0]
    if (
        fth.shape[0] != n_dir
        or fph.shape[0] != n_dir
        or gth.shape[0] != n_dir
        or gph.shape[0] != n_dir
    ):
        raise ValueError(
            "Directional forward transform row count must match sampled direction count."
        )
    if (
        fth_adj.shape[1] != n_dir
        or fph_adj.shape[1] != n_dir
        or gth_adj.shape[1] != n_dir
        or gph_adj.shape[1] != n_dir
    ):
        raise ValueError(
            "Directional adjoint transform column count must match sampled direction count."
        )

    return CuPyDirectionalTransformsData(
        box_order=int(transforms.box_order),
        grid_order=int(transforms.grid.order),
        grid=CuPyDirectionalGridData(
            order=int(grid.order),
            directions=cupy.asarray(directions, dtype=cupy.float64),
            weights=cupy.asarray(weights, dtype=cupy.float64),
            reflection_permutation=cupy.asarray(reflection, dtype=cupy.int64),
        ),
        Fth=cupy.asarray(fth, dtype=cupy.complex128),
        Fph=cupy.asarray(fph, dtype=cupy.complex128),
        Gth=cupy.asarray(gth, dtype=cupy.complex128),
        Gph=cupy.asarray(gph, dtype=cupy.complex128),
        Fth_adj=cupy.asarray(fth_adj, dtype=cupy.complex128),
        Fph_adj=cupy.asarray(fph_adj, dtype=cupy.complex128),
        Gth_adj=cupy.asarray(gth_adj, dtype=cupy.complex128),
        Gph_adj=cupy.asarray(gph_adj, dtype=cupy.complex128),
    )


def _upload_sparse_interpolation(
    interpolation: MLFMMDirectionalInterpolation, *, cupy: Any, cupyx_sparse: Any
) -> CuPyDirectionalInterpolationData:
    csr = interpolation.matrix.tocsr()
    data = np.ascontiguousarray(np.asarray(csr.data, dtype=np.complex128))
    indices = np.ascontiguousarray(np.asarray(csr.indices, dtype=np.int32))
    indptr = np.ascontiguousarray(np.asarray(csr.indptr, dtype=np.int32))
    if indptr.size != csr.shape[0] + 1:
        raise ValueError("Interpolation CSR indptr size is inconsistent with shape.")
    if indices.size != data.size:
        raise ValueError("Interpolation CSR indices/data length mismatch.")
    dev = cupyx_sparse.csr_matrix(
        (
            cupy.asarray(data, dtype=cupy.complex128),
            cupy.asarray(indices, dtype=cupy.int32),
            cupy.asarray(indptr, dtype=cupy.int32),
        ),
        shape=csr.shape,
    )
    return CuPyDirectionalInterpolationData(
        source_order=int(interpolation.source_order),
        target_order=int(interpolation.target_order),
        matrix=dev,
        nnz=int(csr.nnz),
    )


def _upload_offset_batches(
    batches: dict[Offset3, tuple[np.ndarray, np.ndarray]], *, cupy: Any, name: str
) -> dict[Offset3, CuPyOffsetBatchData]:
    out: dict[Offset3, CuPyOffsetBatchData] = {}
    for offset, (src_idx, dst_idx) in batches.items():
        src = _as_numpy_1d(src_idx, dtype=np.int64, name=f"{name}[{offset}].src")
        dst = _as_numpy_1d(dst_idx, dtype=np.int64, name=f"{name}[{offset}].dst")
        if src.size != dst.size:
            raise ValueError(
                f"{name}[{offset}] source/target batch size mismatch: {src.size} vs {dst.size}."
            )
        out[offset] = CuPyOffsetBatchData(
            src_indices=cupy.asarray(src, dtype=cupy.int64),
            dst_indices=cupy.asarray(dst, dtype=cupy.int64),
        )
    return out


def _upload_partition(partition: MLFMMPartition, *, cupy: Any) -> CuPyMLFMMPartitionData:
    leaf_particle_offsets, leaf_particle_indices = _pack_index_lists(
        [leaf.particle_indices for leaf in partition.leaves]
    )
    return CuPyMLFMMPartitionData(
        root_center=cupy.asarray(
            _as_numpy_1d(partition.root_center, dtype=np.float64, name="partition.root_center"),
            dtype=cupy.float64,
        ),
        root_half_size=float(partition.root_half_size),
        depth=int(partition.depth),
        leaf_particle_offsets=cupy.asarray(leaf_particle_offsets, dtype=cupy.int64),
        leaf_particle_indices=cupy.asarray(leaf_particle_indices, dtype=cupy.int64),
        leaf_near_pairs=cupy.asarray(
            _pack_pairs(partition.leaf_near_pairs, name="partition.leaf_near_pairs"),
            dtype=cupy.int64,
        ),
        leaf_far_pairs=cupy.asarray(
            _pack_pairs(partition.leaf_far_pairs, name="partition.leaf_far_pairs"),
            dtype=cupy.int64,
        ),
    )


def _upload_single_level(
    single: MLFMMSingleLevelOperators, *, cupy: Any
) -> CuPyMLFMMSingleLevelData:
    directional = _upload_directional_transforms(single.directional, cupy=cupy)
    ndir = int(np.asarray(single.directional.grid.directions).shape[0])
    offset_diagonals: dict[Offset3, Any] = {}
    for offset, diag in single.offset_diagonals.items():
        diag_arr = _as_numpy_1d(
            diag,
            dtype=np.complex128,
            name=f"single_level.offset_diagonals[{offset}]",
        )
        if diag_arr.size != ndir:
            raise ValueError(
                "single-level offset diagonal length mismatch with directional grid: "
                f"{diag_arr.size} vs {ndir}."
            )
        offset_diagonals[offset] = cupy.asarray(diag_arr, dtype=cupy.complex128)

    return CuPyMLFMMSingleLevelData(
        box_order=int(single.box_order),
        translator_order=int(single.translator_order),
        grid_order=int(single.grid_order),
        directional=directional,
        aggregation=tuple(
            _upload_dense_matrix(matrix, name=f"single_level.aggregation[{i}]", cupy=cupy)
            for i, matrix in enumerate(single.aggregation)
        ),
        receive=tuple(
            _upload_dense_matrix(matrix, name=f"single_level.receive[{i}]", cupy=cupy)
            for i, matrix in enumerate(single.receive)
        ),
        far_offset_batches=_upload_offset_batches(
            single.far_offset_batches,
            cupy=cupy,
            name="single_level.far_offset_batches",
        ),
        offset_diagonals=offset_diagonals,
    )


def _upload_level(level: MLFMMLevelOperators, *, cupy: Any) -> CuPyMLFMMLevelData:
    directional = _upload_directional_transforms(level.directional, cupy=cupy)
    children_offsets, children_indices = _pack_index_lists(level.children)
    ndir = int(np.asarray(level.directional.grid.directions).shape[0])
    offset_diagonals: dict[Offset3, Any] = {}
    for offset, diag in level.offset_diagonals.items():
        diag_arr = _as_numpy_1d(
            diag, dtype=np.complex128, name=f"level[{level.level}].diag[{offset}]"
        )
        if diag_arr.size != ndir:
            raise ValueError(
                f"level {level.level} offset diagonal length mismatch: {diag_arr.size} vs {ndir}."
            )
        offset_diagonals[offset] = cupy.asarray(diag_arr, dtype=cupy.complex128)

    coords = _as_numpy_3cols(level.coords, dtype=np.int64, name=f"level[{level.level}].coords")
    centers = _as_numpy_3cols(level.centers, dtype=np.float64, name=f"level[{level.level}].centers")
    parent_indices = _as_numpy_1d(
        level.parent_indices,
        dtype=np.int64,
        name=f"level[{level.level}].parent_indices",
    )
    if coords.shape[0] != centers.shape[0] or coords.shape[0] != parent_indices.size:
        raise ValueError(
            f"level {level.level} inconsistent box counts among coords/centers/parent_indices."
        )

    return CuPyMLFMMLevelData(
        level=int(level.level),
        coords=cupy.asarray(coords, dtype=cupy.int64),
        centers=cupy.asarray(centers, dtype=cupy.float64),
        parent_indices=cupy.asarray(parent_indices, dtype=cupy.int64),
        children_offsets=cupy.asarray(children_offsets, dtype=cupy.int64),
        children_indices=cupy.asarray(children_indices, dtype=cupy.int64),
        box_order=int(level.box_order),
        translator_order=int(level.translator_order),
        grid_order=int(level.grid_order),
        directional=directional,
        far_offset_batches=_upload_offset_batches(
            level.far_offset_batches,
            cupy=cupy,
            name=f"level[{level.level}].far_offset_batches",
        ),
        offset_diagonals=offset_diagonals,
    )


def _upload_transfer(
    transfer: MLFMMTransferOperators, *, cupy: Any, cupyx_sparse: Any
) -> CuPyMLFMMTransferData:
    phase_up: dict[Offset3, Any] = {}
    phase_down: dict[Offset3, Any] = {}
    for shift, phase in transfer.phase_up_by_shift.items():
        phase_arr = _as_numpy_1d(
            phase,
            dtype=np.complex128,
            name=f"transfer[{transfer.child_level}->{transfer.parent_level}].phase_up[{shift}]",
        )
        phase_up[shift] = cupy.asarray(phase_arr, dtype=cupy.complex128)
    for shift, phase in transfer.phase_down_by_shift.items():
        phase_arr = _as_numpy_1d(
            phase,
            dtype=np.complex128,
            name=f"transfer[{transfer.parent_level}->{transfer.child_level}].phase_down[{shift}]",
        )
        phase_down[shift] = cupy.asarray(phase_arr, dtype=cupy.complex128)

    return CuPyMLFMMTransferData(
        child_level=int(transfer.child_level),
        parent_level=int(transfer.parent_level),
        interpolation=_upload_sparse_interpolation(
            transfer.interpolation,
            cupy=cupy,
            cupyx_sparse=cupyx_sparse,
        ),
        anterpolation=_upload_sparse_interpolation(
            transfer.anterpolation,
            cupy=cupy,
            cupyx_sparse=cupyx_sparse,
        ),
        batches_by_shift=_upload_offset_batches(
            transfer.batches_by_shift,
            cupy=cupy,
            name=f"transfer[{transfer.child_level}->{transfer.parent_level}].batches_by_shift",
        ),
        phase_up_by_shift=phase_up,
        phase_down_by_shift=phase_down,
    )


def _upload_multilevel(
    multilevel: MLFMMMultilevelOperators, *, cupy: Any, cupyx_sparse: Any
) -> CuPyMLFMMMultilevelData:
    return CuPyMLFMMMultilevelData(
        levels=tuple(_upload_level(level, cupy=cupy) for level in multilevel.levels),
        transfers=tuple(
            _upload_transfer(transfer, cupy=cupy, cupyx_sparse=cupyx_sparse)
            for transfer in multilevel.transfers
        ),
        leaf_level=int(multilevel.leaf_level),
        hf_start_level=int(multilevel.hf_start_level),
        hf_end_level=int(multilevel.hf_end_level),
        aggregation=tuple(
            _upload_dense_matrix(matrix, name=f"multilevel.aggregation[{i}]", cupy=cupy)
            for i, matrix in enumerate(multilevel.aggregation)
        ),
        receive=tuple(
            _upload_dense_matrix(matrix, name=f"multilevel.receive[{i}]", cupy=cupy)
            for i, matrix in enumerate(multilevel.receive)
        ),
    )


def _build_exact_near_pair_blocks(coupling: MLFMMCouplingOperator) -> CuPyMLFMMNearPairData:
    """Build directed exact-near pair blocks on CPU and return NumPy arrays."""

    partition = coupling.resolved_plan.partition
    positions = np.asarray(coupling.positions, dtype=float)
    nm = n_modes(int(coupling.lmax))
    ab5 = translation_ab5_table(int(coupling.lmax), dtype=np.complex128)
    dst_indices: list[int] = []
    src_indices: list[int] = []
    blocks: list[np.ndarray] = []
    cache = coupling._exact_block_cache

    def _block_for_pair(i: int, j: int) -> np.ndarray:
        key = (int(i), int(j))
        if cache is not None and key in cache:
            return np.asarray(cache[key], dtype=np.complex128)
        wij = translation_block(
            int(coupling.lmax),
            float(coupling.k),
            np.asarray(positions[int(i)] - positions[int(j)], dtype=float),
            ab5=ab5,
            radial_lut=coupling.radial_lut,
        )
        wij_arr = np.asarray(wij, dtype=np.complex128)
        if cache is not None:
            cache[key] = wij_arr
        return wij_arr

    for a, b in partition.leaf_near_pairs:
        leaf_a = partition.leaves[int(a)]
        leaf_b = partition.leaves[int(b)]
        if int(a) == int(b):
            for i in np.asarray(leaf_a.particle_indices, dtype=np.int64):
                for j in np.asarray(leaf_a.particle_indices, dtype=np.int64):
                    if int(i) == int(j):
                        continue
                    dst_indices.append(int(i))
                    src_indices.append(int(j))
                    blocks.append(_block_for_pair(int(i), int(j)))
            continue
        for i in np.asarray(leaf_a.particle_indices, dtype=np.int64):
            for j in np.asarray(leaf_b.particle_indices, dtype=np.int64):
                dst_indices.append(int(i))
                src_indices.append(int(j))
                blocks.append(_block_for_pair(int(i), int(j)))
                dst_indices.append(int(j))
                src_indices.append(int(i))
                blocks.append(_block_for_pair(int(j), int(i)))

    if blocks:
        blocks_arr = np.ascontiguousarray(np.stack(blocks, axis=0), dtype=np.complex128)
    else:
        blocks_arr = np.zeros((0, nm, nm), dtype=np.complex128)
    if blocks_arr.ndim != 3 or blocks_arr.shape[1:] != (nm, nm):
        raise ValueError(f"Exact-near pair block tensor must have shape (P, {nm}, {nm}).")
    return CuPyMLFMMNearPairData(
        dst_particle_indices=np.asarray(dst_indices, dtype=np.int64),
        src_particle_indices=np.asarray(src_indices, dtype=np.int64),
        blocks=blocks_arr,
    )


def _upload_exact_near_pair_blocks(
    coupling: MLFMMCouplingOperator, *, cupy: Any
) -> CuPyMLFMMNearPairData:
    """Upload directed exact-near pair blocks to device memory.

    The near-pair tensor can be large for dense clusters; use blocking host->device
    copies to avoid large transient pinned-memory allocations in CuPy.
    """

    built = _build_exact_near_pair_blocks(coupling)
    return CuPyMLFMMNearPairData(
        dst_particle_indices=cupy.asarray(
            _as_numpy_1d(
                np.asarray(built.dst_particle_indices),
                dtype=np.int64,
                name="exact_near.dst_particle_indices",
            ),
            dtype=cupy.int64,
            blocking=True,
        ),
        src_particle_indices=cupy.asarray(
            _as_numpy_1d(
                np.asarray(built.src_particle_indices),
                dtype=np.int64,
                name="exact_near.src_particle_indices",
            ),
            dtype=cupy.int64,
            blocking=True,
        ),
        blocks=cupy.asarray(
            np.asarray(built.blocks, dtype=np.complex128),
            dtype=cupy.complex128,
            blocking=True,
        ),
    )


def _reshape_unknowns_to_particle_modes(
    x: Any, *, n_particles: int, nm: int, cupy: Any
) -> tuple[Any, bool]:
    """Reshape `(n*nm,)` or `(n*nm, nrhs)` unknowns to `(n, nm, nrhs)`."""

    arr = coerce_array(x, dtype=np.dtype(np.complex128), prefer_cupy=True)
    if arr.ndim == 1:
        if int(arr.size) != int(n_particles * nm):
            raise ValueError(
                f"MLFMM CuPy apply expected vector size {n_particles * nm}, got {int(arr.size)}."
            )
        return cupy.asarray(arr, dtype=cupy.complex128).reshape(n_particles, nm, 1), True
    if arr.ndim == 2:
        if int(arr.shape[0]) != int(n_particles * nm):
            raise ValueError(
                "MLFMM CuPy apply expected 2D unknowns with first dimension "
                f"{n_particles * nm}, got {tuple(int(v) for v in arr.shape)}."
            )
        return cupy.asarray(arr, dtype=cupy.complex128).reshape(
            n_particles, nm, int(arr.shape[1])
        ), False
    raise ValueError(f"MLFMM CuPy apply expects 1D or 2D unknowns, got ndim={arr.ndim}.")


def _restore_unknown_shape(y: Any, *, squeezed: bool) -> Any:
    """Restore `(n, nm, nrhs)` output to original linear-system shape."""

    if squeezed:
        return y.reshape(-1)
    n_particles, nm, nrhs = (int(v) for v in y.shape)
    return y.reshape(n_particles * nm, nrhs)


def _apply_reflection_to_direction_axis(values: Any, permutation: Any, *, cupy: Any) -> Any:
    """Apply one directional reflection permutation on axis 2."""

    return cupy.take(values, permutation, axis=2)


@cache
def _add_at_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void add_at_complex128(
        const long long n_pairs,
        const long long width,
        const long long* dst,
        const complex<double>* values,
        complex<double>* out
    ) {
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * width;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / width;
            const long long lane = i - pair_idx * width;
            const long long out_row = dst[pair_idx];
            const long long out_idx = out_row * width + lane;
            const complex<double> v = values[i];
            double* out_ptr = reinterpret_cast<double*>(&out[out_idx]);
            atomicAdd(out_ptr + 0, v.real());
            atomicAdd(out_ptr + 1, v.imag());
        }
    }
    """
    return cupy.RawKernel(source, "add_at_complex128")


def _add_at_complex128(target: Any, indices: Any, values: Any, *, cupy: Any) -> None:
    """Apply `add.at`-style indexed accumulation for complex128 batches on device.

    We use a specialized kernel because, as of CuPy 14.0.1, `cupy.add.at`
    does not accept complex dtypes. The custom kernel also avoids splitting
    real/imag updates in the hot loop.
    """

    idx = cupy.asarray(indices, dtype=cupy.int64).reshape(-1)
    if int(idx.size) == 0:
        return
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    values_arr = cupy.asarray(values, dtype=cupy.complex128)
    if int(values_arr.shape[0]) != int(idx.size):
        raise ValueError(
            "Scatter-add value batch count must match index count. "
            f"Got values={int(values_arr.shape[0])}, indices={int(idx.size)}."
        )
    width = int(np.prod(values_arr.shape[1:], dtype=np.int64))
    if width <= 0:
        raise ValueError("Scatter-add trailing width must be positive.")
    values_flat = cupy.ascontiguousarray(values_arr.reshape(int(idx.size), width))
    out_flat = tgt.reshape(int(tgt.shape[0]), width)
    if not bool(out_flat.flags.c_contiguous):
        raise ValueError("Scatter-add target view must be C-contiguous.")
    if int(cupy.max(idx)) >= int(tgt.shape[0]) or int(cupy.min(idx)) < 0:
        raise ValueError("Scatter-add index out of bounds for target tensor.")

    kernel = _add_at_complex128_raw_kernel()
    threads = 256
    total = int(idx.size) * width
    blocks = max(1, (total + threads - 1) // threads)
    kernel(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(idx.size)),
            np.int64(width),
            idx,
            values_flat,
            out_flat,
        ),
    )


def _apply_sparse_directional_map(values: Any, matrix: Any, *, cupy: Any) -> Any:
    """Apply one sparse directional map on channel batches.

    Parameters
    ----------
    values:
        Shape `(nbatch, 4, n_source, nrhs)`.
    matrix:
        CSR-like sparse matrix with shape `(n_target, n_source)`.
    """

    arr = cupy.asarray(values, dtype=cupy.complex128)
    n_batch = int(arr.shape[0])
    n_chan = int(arr.shape[1])
    n_rhs = int(arr.shape[3])
    flat = arr.transpose(0, 1, 3, 2).reshape(-1, int(arr.shape[2]))
    mapped_flat = (matrix @ flat.T).T
    return mapped_flat.reshape(n_batch, n_chan, n_rhs, int(matrix.shape[0])).transpose(0, 1, 3, 2)


def _box_outgoing_to_directional_cupy(
    directional: CuPyDirectionalTransformsData, box_states: Any, *, cupy: Any
) -> Any:
    """Map batched outgoing box SVWF states to directional channels on device."""

    states = cupy.asarray(box_states, dtype=cupy.complex128)
    nscl = int(directional.Fth.shape[1])
    if int(states.shape[1]) != 2 * nscl:
        raise ValueError(
            f"box_states second dimension must be {2 * nscl}, got {int(states.shape[1])}."
        )
    a_box = states[:, :nscl, :]
    b_box = states[:, nscl:, :]
    a_theta = cupy.einsum("dn,bnr->bdr", directional.Fth, a_box)
    a_phi = cupy.einsum("dn,bnr->bdr", directional.Fph, a_box)
    b_theta = cupy.einsum("dn,bnr->bdr", directional.Fth, b_box)
    b_phi = cupy.einsum("dn,bnr->bdr", directional.Fph, b_box)
    perm = directional.grid.reflection_permutation
    return cupy.stack(
        (
            cupy.take(a_theta, perm, axis=1),
            cupy.take(a_phi, perm, axis=1),
            cupy.take(b_theta, perm, axis=1),
            cupy.take(b_phi, perm, axis=1),
        ),
        axis=1,
    )


def _directional_to_box_regular_cupy(
    directional: CuPyDirectionalTransformsData, directional_channels: Any, *, cupy: Any
) -> Any:
    """Map batched directional channels to regular box SVWF states on device."""

    channels = cupy.asarray(directional_channels, dtype=cupy.complex128)
    if int(channels.shape[1]) != 4:
        raise ValueError(
            f"directional channel batch must have 4 channels, got {int(channels.shape[1])}."
        )
    perm = directional.grid.reflection_permutation
    a_theta = cupy.take(channels[:, 0], perm, axis=1)
    a_phi = cupy.take(channels[:, 1], perm, axis=1)
    b_theta = cupy.take(channels[:, 2], perm, axis=1)
    b_phi = cupy.take(channels[:, 3], perm, axis=1)

    top = (
        cupy.einsum("sn,bnr->bsr", directional.Fth_adj, a_theta)
        + cupy.einsum("sn,bnr->bsr", directional.Fph_adj, a_phi)
        + cupy.einsum("sn,bnr->bsr", directional.Gth_adj, b_theta)
        + cupy.einsum("sn,bnr->bsr", directional.Gph_adj, b_phi)
    )
    bottom = (
        cupy.einsum("sn,bnr->bsr", directional.Fth_adj, b_theta)
        + cupy.einsum("sn,bnr->bsr", directional.Fph_adj, b_phi)
        + cupy.einsum("sn,bnr->bsr", directional.Gth_adj, a_theta)
        + cupy.einsum("sn,bnr->bsr", directional.Gph_adj, a_phi)
    )
    return cupy.concatenate((top, bottom), axis=1)


def _aggregate_leaf_box_states(
    x_states: Any,
    *,
    leaves: tuple[Any, ...],
    aggregation: tuple[Any, ...],
    box_nm: int,
    nrhs: int,
    cupy: Any,
) -> Any:
    """Aggregate particle coefficients into one outgoing box state per leaf."""

    box_states = cupy.zeros((len(leaves), int(box_nm), int(nrhs)), dtype=cupy.complex128)
    for leaf in leaves:
        particle_indices = np.asarray(leaf.particle_indices, dtype=np.int64)
        idx = cupy.asarray(particle_indices, dtype=cupy.int64)
        coeffs = x_states[idx].reshape(-1, nrhs)
        box_states[int(leaf.id)] = aggregation[int(leaf.id)] @ coeffs
    return box_states


def _receive_leaf_boxes_to_particles(
    incoming_box: Any,
    *,
    leaves: tuple[Any, ...],
    receive: tuple[Any, ...],
    nm: int,
    n_particles: int,
    nrhs: int,
    cupy: Any,
) -> Any:
    """Scatter leaf-local incoming box states back to particle coefficients."""

    y = cupy.zeros((int(n_particles), int(nm), int(nrhs)), dtype=cupy.complex128)
    for leaf in leaves:
        pid = np.asarray(leaf.particle_indices, dtype=np.int64)
        idx = cupy.asarray(pid, dtype=cupy.int64)
        contribution = receive[int(leaf.id)] @ incoming_box[int(leaf.id)]
        y[idx] += contribution.reshape(pid.size, nm, nrhs)
    return y


def _apply_exact_near_pairs(prepared: CuPyMLFMMPreparedData, x_states: Any, *, cupy: Any) -> Any:
    """Apply directed exact-near pair blocks on device."""

    near = prepared.near_pairs
    n_pairs = int(near.dst_particle_indices.size)
    y = cupy.zeros_like(x_states, dtype=cupy.complex128)
    if n_pairs == 0:
        return y
    src_values = x_states[near.src_particle_indices]  # (P, nm, nrhs)
    contrib = cupy.einsum("pij,pjr->pir", near.blocks, src_values)
    _add_at_complex128(y, near.dst_particle_indices, contrib, cupy=cupy)
    return y


def _apply_single_level_far(prepared: CuPyMLFMMPreparedData, x_states: Any, *, cupy: Any) -> Any:
    """Apply sampled single-level far interactions on device."""

    single = prepared.single_level
    if single is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing single-level prepared data.")
    leaves = prepared.resolved_plan.partition.leaves
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    box_nm = int(single.aggregation[0].shape[0])
    box_states = _aggregate_leaf_box_states(
        x_states,
        leaves=leaves,
        aggregation=single.aggregation,
        box_nm=box_nm,
        nrhs=nrhs,
        cupy=cupy,
    )
    outgoing = _box_outgoing_to_directional_cupy(single.directional, box_states, cupy=cupy)
    incoming = cupy.zeros_like(outgoing, dtype=cupy.complex128)
    for offset, batch in single.far_offset_batches.items():
        translated = (
            outgoing[batch.src_indices] * single.offset_diagonals[offset][None, None, :, None]
        )
        _add_at_complex128(incoming, batch.dst_indices, translated, cupy=cupy)
    incoming_box = _directional_to_box_regular_cupy(single.directional, incoming, cupy=cupy)
    return _receive_leaf_boxes_to_particles(
        incoming_box,
        leaves=leaves,
        receive=single.receive,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        cupy=cupy,
    )


def _apply_multilevel_far(prepared: CuPyMLFMMPreparedData, x_states: Any, *, cupy: Any) -> Any:
    """Apply sampled multilevel far interactions on device."""

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    leaves = prepared.resolved_plan.partition.leaves
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    levels = multilevel.levels
    outgoing = [
        cupy.zeros(
            (
                int(level.coords.shape[0]),
                4,
                int(level.directional.grid.directions.shape[0]),
                nrhs,
            ),
            dtype=cupy.complex128,
        )
        for level in levels
    ]
    incoming = [cupy.zeros_like(values, dtype=cupy.complex128) for values in outgoing]
    leaf_level = int(multilevel.leaf_level)
    box_nm = int(multilevel.aggregation[0].shape[0])
    leaf_box_states = _aggregate_leaf_box_states(
        x_states,
        leaves=leaves,
        aggregation=multilevel.aggregation,
        box_nm=box_nm,
        nrhs=nrhs,
        cupy=cupy,
    )
    outgoing[leaf_level] = _box_outgoing_to_directional_cupy(
        levels[leaf_level].directional,
        leaf_box_states,
        cupy=cupy,
    )

    for transfer in reversed(multilevel.transfers):
        child_level = int(transfer.child_level)
        parent_level = int(transfer.parent_level)
        child_values = outgoing[child_level]
        parent_values = outgoing[parent_level]
        child_perm = levels[child_level].directional.grid.reflection_permutation
        parent_perm = levels[parent_level].directional.grid.reflection_permutation
        for shift, batch in transfer.batches_by_shift.items():
            child_reindexed = _apply_reflection_to_direction_axis(
                child_values[batch.src_indices], child_perm, cupy=cupy
            )
            mapped_reindexed = _apply_sparse_directional_map(
                child_reindexed, transfer.interpolation.matrix, cupy=cupy
            )
            mapped = _apply_reflection_to_direction_axis(mapped_reindexed, parent_perm, cupy=cupy)
            mapped *= transfer.phase_up_by_shift[shift][None, None, :, None]
            _add_at_complex128(parent_values, batch.dst_indices, mapped, cupy=cupy)

    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = levels[level_idx]
        for offset, batch in level.far_offset_batches.items():
            translated = (
                outgoing[level_idx][batch.src_indices]
                * level.offset_diagonals[offset][None, None, :, None]
            )
            _add_at_complex128(incoming[level_idx], batch.dst_indices, translated, cupy=cupy)

    for transfer in multilevel.transfers:
        child_level = int(transfer.child_level)
        parent_level = int(transfer.parent_level)
        parent_values = incoming[parent_level]
        child_values = incoming[child_level]
        parent_perm = levels[parent_level].directional.grid.reflection_permutation
        child_perm = levels[child_level].directional.grid.reflection_permutation
        for shift, batch in transfer.batches_by_shift.items():
            shifted = (
                parent_values[batch.dst_indices]
                * transfer.phase_down_by_shift[shift][None, None, :, None]
            )
            shifted_reindexed = _apply_reflection_to_direction_axis(shifted, parent_perm, cupy=cupy)
            mapped_reindexed = _apply_sparse_directional_map(
                shifted_reindexed, transfer.anterpolation.matrix, cupy=cupy
            )
            mapped = _apply_reflection_to_direction_axis(mapped_reindexed, child_perm, cupy=cupy)
            _add_at_complex128(child_values, batch.src_indices, mapped, cupy=cupy)

    incoming_box = _directional_to_box_regular_cupy(
        levels[leaf_level].directional,
        incoming[leaf_level],
        cupy=cupy,
    )
    return _receive_leaf_boxes_to_particles(
        incoming_box,
        leaves=leaves,
        receive=multilevel.receive,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        cupy=cupy,
    )


@dataclass
class CuPyMLFMMCouplingOperator:
    """CuPy-backed repeated-apply MLFMM coupling operator.

    The MLFMM plan and one-time operators are built on CPU (NumPy reference
    path). This class executes repeated exact-near and sampled-far applies on
    CuPy device arrays.
    """

    cpu_coupling: MLFMMCouplingOperator
    prepared_data: CuPyMLFMMPreparedData
    dtype: np.dtype = np.dtype(np.complex128)

    def apply(self, x: Any) -> Any:
        cupy, _ = import_cupy()
        if np.dtype(self.dtype) != np.dtype(np.complex128):
            raise ValueError(
                "CuPy MLFMM apply currently supports only complex128 "
                f"(got {np.dtype(self.dtype)!r})."
            )
        nm = n_modes(int(self.cpu_coupling.lmax))
        n_particles = int(self.cpu_coupling.positions.shape[0])
        x_states, squeezed = _reshape_unknowns_to_particle_modes(
            x,
            n_particles=n_particles,
            nm=nm,
            cupy=cupy,
        )
        y_near = _apply_exact_near_pairs(self.prepared_data, x_states, cupy=cupy)
        stage = str(self.prepared_data.stage)
        if stage == "single_level":
            y_far = _apply_single_level_far(self.prepared_data, x_states, cupy=cupy)
        elif stage == "multilevel":
            y_far = _apply_multilevel_far(self.prepared_data, x_states, cupy=cupy)
        else:
            raise RuntimeError(f"Unsupported CuPy MLFMM stage {stage!r}.")
        return _restore_unknown_shape(y_near + y_far, squeezed=squeezed)


def prepare_mlfmm_cupy_coupling(coupling: MLFMMCouplingOperator) -> CuPyMLFMMCouplingOperator:
    """Wrap a CPU-built MLFMM coupling plan in a CuPy repeated-apply operator."""

    prepared = prepare_mlfmm_cupy_data(coupling)
    return CuPyMLFMMCouplingOperator(
        cpu_coupling=coupling,
        prepared_data=prepared,
        dtype=np.dtype(coupling.dtype),
    )


def prepare_mlfmm_cupy_data(coupling: MLFMMCouplingOperator) -> CuPyMLFMMPreparedData:
    """Upload repeated-apply MLFMM structures from a CPU-built coupling plan.

    Parameters
    ----------
    coupling:
        CPU-built MLFMM coupling object from `prepare_mlfmm_coupling(...)`.

    Returns
    -------
    CuPyMLFMMPreparedData
        Device-resident representation of all repeated-apply data needed by
        the CuPy MLFMM apply path.
    """

    cupy, _ = import_cupy()
    cupyx_sparse = import_module("cupyx.scipy.sparse")
    if np.dtype(coupling.dtype) != np.dtype(np.complex128):
        raise ValueError(
            "CuPy MLFMM preparation currently supports only complex128 "
            f"(got {np.dtype(coupling.dtype)!r})."
        )
    stage = str(coupling.resolved_plan.stage)
    if stage not in {"single_level", "multilevel"}:
        raise ValueError(
            "CuPy MLFMM preparation requires a non-direct MLFMM stage. "
            f"Resolved stage is {stage!r}."
        )

    partition_data = _upload_partition(coupling.resolved_plan.partition, cupy=cupy)
    near_pairs = _upload_exact_near_pair_blocks(coupling, cupy=cupy)
    positions = _as_numpy_3cols(coupling.positions, dtype=np.float64, name="coupling.positions")
    single_level_data: CuPyMLFMMSingleLevelData | None = None
    multilevel_data: CuPyMLFMMMultilevelData | None = None
    if stage == "single_level":
        if coupling.single_level is None:
            raise ValueError("single_level stage was selected but no single-level operators exist.")
        single_level_data = _upload_single_level(coupling.single_level, cupy=cupy)
    else:
        if coupling.multilevel is None:
            raise ValueError("multilevel stage was selected but no multilevel operators exist.")
        multilevel_data = _upload_multilevel(
            coupling.multilevel, cupy=cupy, cupyx_sparse=cupyx_sparse
        )

    return CuPyMLFMMPreparedData(
        lmax=int(coupling.lmax),
        k=float(coupling.k),
        stage=stage,
        positions=cupy.asarray(positions, dtype=cupy.float64),
        partition=partition_data,
        resolved_plan=coupling.resolved_plan,
        near_pairs=near_pairs,
        single_level=single_level_data,
        multilevel=multilevel_data,
    )


__all__ = [
    "CuPyMLFMMCouplingOperator",
    "CuPyDirectionalGridData",
    "CuPyDirectionalInterpolationData",
    "CuPyDirectionalTransformsData",
    "CuPyMLFMMLevelData",
    "CuPyMLFMMMultilevelData",
    "CuPyMLFMMNearPairData",
    "CuPyMLFMMPartitionData",
    "CuPyMLFMMPreparedData",
    "CuPyMLFMMSingleLevelData",
    "CuPyMLFMMTransferData",
    "prepare_mlfmm_cupy_coupling",
    "prepare_mlfmm_cupy_data",
]
