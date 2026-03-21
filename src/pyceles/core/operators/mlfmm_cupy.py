from __future__ import annotations

"""CuPy-side prepared-data wrappers for the NumPy MLFMM reference plan.

CPU build/planning remains the single source of truth in `mlfmm.py`.
This module validates and uploads repeated-apply structures to device memory.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from pyceles._optional import import_cupy

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
        the future CuPy MLFMM apply path.
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
        single_level=single_level_data,
        multilevel=multilevel_data,
    )


__all__ = [
    "CuPyDirectionalGridData",
    "CuPyDirectionalInterpolationData",
    "CuPyDirectionalTransformsData",
    "CuPyMLFMMLevelData",
    "CuPyMLFMMMultilevelData",
    "CuPyMLFMMPartitionData",
    "CuPyMLFMMPreparedData",
    "CuPyMLFMMSingleLevelData",
    "CuPyMLFMMTransferData",
    "prepare_mlfmm_cupy_data",
]
