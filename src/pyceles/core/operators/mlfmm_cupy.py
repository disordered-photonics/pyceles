"""CuPy repeated-apply data and kernels for CPU-built MLFMM plans.

CPU build/planning remains the single source of truth in `mlfmm.py`.
This module validates and uploads finite repeated-apply structures to device
memory.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cache
from importlib import import_module
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from pyceles._cupy_memory import (
    CuPyAllocatorSnapshot,
    cupy_allocator_memory_info,
    cupy_allocator_snapshot,
)
from pyceles._optional import coerce_array, import_cupy
from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
    translation_ab5_table,
    translation_block,
)
from pyceles.core.wigner import wigner_3j

from .mlfmm import (
    MLFMMCouplingOperator,
    MLFMMLevelOperators,
    MLFMMMultilevelOperators,
    MLFMMSingleLevelOperators,
    MLFMMTransferOperators,
)
from .mlfmm_directional import (
    MLFMMDirectionalStructuredTransforms,
    MLFMMDirectionalTransformData,
    structured_directional_transforms,
)
from .mlfmm_partition import MLFMMPartition
from .mode_metadata import (
    mode_m_table,
    mode_pair_p_range_tables,
    mode_tau_l_tables,
)

COMPLEX128_DTYPE = np.dtype(np.complex128)
_MIB = 1024**2

# Streamed CuPy MLFMM memory-policy constants.
#
# Automatic multilevel streaming first applies a hard CuPy pool ceiling below
# physical device memory. Source/outgoing chunks and frontier incoming buffers
# then share one transient budget inside the remaining active headroom. The
# transient fraction is intentionally below 1.0 because the model does not
# include short-lived traversal temporaries such as search/filter/concatenate
# arrays or allocator fragmentation.
_STREAMED_FAR_TRANSIENT_FRACTION_OF_ACTIVE_HEADROOM = 0.86
_STREAMED_FAR_FRONTIER_FRACTION_OF_TRANSIENT = 0.67
_STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES = 16 * _MIB
_STREAMED_FAR_MIN_FRONTIER_BYTES = 64 * _MIB
_SINGLE_LEVEL_LEAF_OTF_FRACTION_OF_EFFECTIVE_FREE = 0.05
_SINGLE_LEVEL_LEAF_OTF_MIN_BYTES = 8 * _MIB
_SINGLE_LEVEL_LEAF_OTF_MAX_BYTES = 256 * _MIB

Offset3 = tuple[int, int, int]
# CuPy production runs use on-the-fly leaf apply; dense leaf payloads remain
# available only for validation/debug parity against the CPU reference plan.
CuPyMLFMMLeafApplyMode = Literal["dense", "on_the_fly"]
CuPyMLFMMHostCacheRetention = Literal["full", "summary", "none"]


@dataclass(frozen=True)
class CuPyMLFMMHostCachePolicy:
    """Policy for compact host payload retention and repeated-apply shape.

    The intended CuPy repeated-apply path is:
    - `leaf_apply_mode="on_the_fly"`
    - streamed sampled-far traversal
    - `host_cache_retention="summary"`
    - opt-in streamed diagnostics

    `leaf_apply_mode` controls how leaf aggregation/disaggregation is
    represented in the reusable host payload:
    - `dense`: preserve grouped dense aggregation maps only for validation/debug
      comparisons against the CPU reference path,
    - `on_the_fly`: keep compact schedules + translation ingredients and
      regenerate leaf translation blocks during repeated apply.

    `host_cache_retention` controls what stays attached to the runtime coupling
    object after upload:
    - `full`: retain the full compact host payload plus summary,
    - `summary`: retain only the lightweight diagnostics summary (default),
    - `none`: retain no host-side diagnostics payload.

    `leaf_otf_chunk_leaves` caps how many leaves of one occupancy-grouped batch
    are processed per on-the-fly translation substep. This is a simple
    leaf-count limiter; it does not directly bound the largest temporary bytes
    when one leaf has very high occupancy.

    `leaf_otf_bytes_budget` bounds single-level on-the-fly leaf receive
    substeps by estimated dense pair-block bytes. Multilevel streamed leaf
    aggregation/receive uses fused kernels and does not materialize that
    pair-block scratch.
    `streamed_far_chunk_bytes_budget` bounds one sampled-far directional chunk
    by estimated directional-state bytes. When omitted, source and frontier
    budgets are derived together from one guarded device-residency plan during
    apply, while runtime diagnostics report the resolved chunking in native
    leaf/box units.

    `collect_stream_stats` enables profiling instrumentation for streamed
    multilevel chunk/build counters. Leave it off for production runs so
    repeated applies do not spend time updating per-apply bookkeeping
    dictionaries.

    `near_adjoint_cache_bytes_budget` bounds the resident exact-near blocks
    used by the CuPy reverse action.  Blocks beyond this budget are rebuilt
    one leaf pair at a time during an adjoint apply, keeping memory bounded
    for tall/high-occupancy MLFMM plans.  Set it to zero to disable block
    retention, or to ``None`` only for an explicitly unbounded diagnostic run.
    """

    leaf_apply_mode: CuPyMLFMMLeafApplyMode = "on_the_fly"
    host_cache_retention: CuPyMLFMMHostCacheRetention = "summary"
    leaf_otf_chunk_leaves: int | None = None
    leaf_otf_bytes_budget: int | None = None
    streamed_far_chunk_bytes_budget: int | None = None
    collect_stream_stats: bool = False
    near_adjoint_cache_bytes_budget: int | None = 128 * _MIB


@dataclass(frozen=True)
class CuPyHostLeafOnTheFlyGroupData:
    """Compact host grouped leaf schedule for on-the-fly translations."""

    occupancy: int
    nmodes: int
    leaf_ids: np.ndarray
    particle_indices: np.ndarray
    pair_deltas: np.ndarray


@dataclass(frozen=True)
class CuPyHostLeafTranslationTablesData:
    """Compact host translation ingredients shared by on-the-fly leaf groups."""

    full_order: int
    nmodes_in: int
    nmodes_out: int
    nmodes_full: int
    out_mode_indices: np.ndarray
    in_mode_indices: np.ndarray
    mode_m_out: np.ndarray
    mode_m_in: np.ndarray
    pair_offset: np.ndarray
    pair_pmin: np.ndarray
    pair_pcount: np.ndarray
    plm_coeffs: np.ndarray
    compact_re_ab: np.ndarray
    compact_im_ab: np.ndarray
    re_j: np.ndarray
    im_j: np.ndarray
    inv_dr: float
    last_index: int


@dataclass(frozen=True)
class CuPyDirectionalGridData:
    """Device copy of one directional sampling grid."""

    order: int
    n_alpha: int
    n_beta: int
    n_directions: int
    reflection_permutation: Any
    beta_reflection_permutation: Any


@dataclass(frozen=True)
class CuPyHostDirectionalGridData:
    """Compact host directional-grid payload used for cache serialization."""

    order: int
    n_alpha: int
    n_beta: int
    n_directions: int
    alpha: np.ndarray
    reflection_permutation: np.ndarray


@dataclass(frozen=True)
class CuPyHostDirectionalTransformsData:
    """Compact host directional transform payload with separable factors only."""

    box_order: int
    grid_order: int
    grid: CuPyHostDirectionalGridData
    fth_beta: np.ndarray
    fph_beta: np.ndarray
    m_of_scalar: np.ndarray


@dataclass(frozen=True)
class CuPyHostLevelData:
    """Compact host per-level payload for multilevel upload."""

    level: int
    n_boxes: int
    box_order: int
    translator_order: int
    grid_order: int
    directional: CuPyHostDirectionalTransformsData
    far_offset_batches: dict[Offset3, tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[Offset3, np.ndarray]


@dataclass(frozen=True)
class CuPyHostTransferData:
    """Compact host transfer payload with canonical interpolation map only."""

    child_level: int
    parent_level: int
    interpolation: Any
    batches_by_shift: dict[Offset3, tuple[np.ndarray, np.ndarray]]
    phase_up_by_shift: dict[Offset3, np.ndarray]
    phase_down_by_shift: dict[Offset3, np.ndarray]


@dataclass(frozen=True)
class CuPyHostSingleLevelData:
    """Compact host single-level payload for CuPy upload.

    Dense `aggregation` blocks are populated only for validation/debug uploads;
    production CuPy runs use the on-the-fly leaf schedule instead.
    """

    box_order: int
    translator_order: int
    grid_order: int
    directional: CuPyHostDirectionalTransformsData
    aggregation: tuple[np.ndarray, ...] | None
    far_offset_batches: dict[Offset3, tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[Offset3, np.ndarray]
    leaf_groups_otf: tuple[CuPyHostLeafOnTheFlyGroupData, ...] | None = None
    leaf_translation_tables: CuPyHostLeafTranslationTablesData | None = None
    leaf_apply_mode: CuPyMLFMMLeafApplyMode = "dense"


@dataclass(frozen=True)
class CuPyHostMultilevelData:
    """Compact host multilevel payload for CuPy upload.

    Dense `aggregation` blocks are populated only for validation/debug uploads;
    production CuPy runs use the on-the-fly leaf schedule instead.
    """

    levels: tuple[CuPyHostLevelData, ...]
    transfers: tuple[CuPyHostTransferData, ...]
    leaf_level: int
    hf_start_level: int
    hf_end_level: int
    aggregation: tuple[np.ndarray, ...] | None
    leaf_groups_otf: tuple[CuPyHostLeafOnTheFlyGroupData, ...] | None = None
    leaf_translation_tables: CuPyHostLeafTranslationTablesData | None = None
    leaf_apply_mode: CuPyMLFMMLeafApplyMode = "dense"


@dataclass(frozen=True)
class CuPyMLFMMHostCacheData:
    """Compact host-only payload for CuPy MLFMM preparation.

    Dense leaf aggregation payloads are retained only for explicit
    validation/debug uploads; the default CuPy path keeps compact on-the-fly
    schedules and translation ingredients instead.
    """

    lmax: int
    k: float
    stage: str
    dtype: np.dtype
    near_dtype: np.dtype
    far_dtype: np.dtype
    n_particles: int
    leaf_particle_offsets: np.ndarray
    leaf_particle_indices: np.ndarray
    near_positions_flat: np.ndarray
    near_dst_leaf_indices: np.ndarray
    near_src_leaf_indices: np.ndarray
    near_lut_re: np.ndarray
    near_lut_im: np.ndarray
    near_inv_dr: float
    near_last_index: int
    near_plm_coeffs: np.ndarray | None
    near_compact_re_ab: np.ndarray | None
    near_compact_im_ab: np.ndarray | None
    near_mode_m: np.ndarray | None
    near_pair_offset: np.ndarray | None
    near_pair_pmin: np.ndarray | None
    near_pair_pcount: np.ndarray | None
    single_level: CuPyHostSingleLevelData | None = None
    multilevel: CuPyHostMultilevelData | None = None
    plan_summary: dict[str, int | float | str] | None = None


@dataclass(frozen=True)
class CuPyDirectionalTransformsData:
    """Device directional transform payload for one box/grid order.

    The CuPy path stores beta factors and azimuthal phases instead of dense
    `(n_alpha*n_beta, n_scalar)` F matrices. Repeated apply expands the
    directional channels on demand, which trades explicit `m` sums for a much
    lower resident memory footprint on coarse high-frequency levels.
    """

    box_order: int
    grid_order: int
    grid: CuPyDirectionalGridData
    nscl: int
    fth_beta: Any
    fph_beta: Any
    m_of_scalar: Any
    phase_by_m: Any
    mode_indices_by_m: tuple[Any, ...]


@dataclass(frozen=True)
class CuPyDirectionalInterpolationData:
    """Device copy of one directional transfer map.

    The map is stored either as a CuPy dense matrix (`storage="dense"`) or
    as a CuPy CSR sparse matrix (`storage="sparse"`).
    """

    source_order: int
    target_order: int
    matrix: Any
    nnz: int
    storage: str


@dataclass(frozen=True)
class CuPyMLFMMPartitionData:
    """Host-side leaf-to-particle lookup used only during upload grouping."""

    leaf_particle_offsets_host: np.ndarray
    leaf_particle_indices_host: np.ndarray


@dataclass(frozen=True)
class CuPyOffsetBatchData:
    """Device copy of grouped source/target index batches for one relative offset.

    Grouped schedules are validated as unique during upload, so repeated apply
    can run non-atomic unique-index accumulation kernels unconditionally.
    """

    src_indices: Any
    dst_indices: Any


@dataclass(frozen=True)
class CuPyLeafApplyGroupData:
    """Grouped leaf data with uniform occupancy for dense or on-the-fly apply.

    Exactly one of `aggregation` (dense mode) or `pair_deltas`
    (on-the-fly mode) must be populated.
    """

    occupancy: int
    nmodes: int
    leaf_ids: Any
    particle_indices: Any
    aggregation: Any | None = None
    pair_deltas: Any | None = None


@dataclass(frozen=True)
class CuPyLeafTranslationTablesData:
    """Device translation ingredients shared by on-the-fly leaf groups."""

    full_order: int
    nmodes_in: int
    nmodes_out: int
    nmodes_full: int
    out_mode_indices: Any
    in_mode_indices: Any
    mode_m_out: Any
    mode_m_in: Any
    pair_offset: Any
    pair_pmin: Any
    pair_pcount: Any
    plm_coeffs: Any
    compact_re_ab: Any
    compact_im_ab: Any
    re_j: Any
    im_j: Any
    inv_dr: float
    last_index: int


@dataclass(frozen=True)
class CuPyMLFMMLevelData:
    """Device-ready per-level data used in multilevel repeated applies."""

    level: int
    n_boxes: int
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
    map_up: CuPyDirectionalInterpolationData
    map_down: CuPyDirectionalInterpolationData
    batches_by_shift: dict[Offset3, CuPyOffsetBatchData]
    phase_up_by_shift: dict[Offset3, Any]
    phase_down_by_shift: dict[Offset3, Any]


@dataclass(frozen=True)
class CuPyMLFMMSingleLevelData:
    """Device-ready container for single-level repeated-apply structures."""

    box_order: int
    translator_order: int
    grid_order: int
    box_nm: int
    n_leaves: int
    directional: CuPyDirectionalTransformsData
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...]
    far_offset_batches: dict[Offset3, CuPyOffsetBatchData]
    offset_diagonals: dict[Offset3, Any]
    leaf_apply_mode: CuPyMLFMMLeafApplyMode = "dense"
    leaf_translation_tables: CuPyLeafTranslationTablesData | None = None


@dataclass(frozen=True)
class CuPyMLFMMNearPairData:
    """Device-ready exact-near schedule and translation tables."""

    near_dtype: np.dtype
    positions: Any
    leaf_particle_offsets: Any
    leaf_particle_indices: Any
    dst_leaf_indices: Any
    src_leaf_indices: Any
    lut_re: Any
    lut_im: Any
    inv_dr: float
    last_index: int
    plm_coeffs: Any
    compact_re_ab: Any
    compact_im_ab: Any
    mode_m: Any
    pair_offset: Any
    pair_pmin: Any
    pair_pcount: Any


@dataclass(frozen=True)
class _CuPyMLFMMNearAdjointContext:
    """Host-side ingredients for bounded exact-near adjoint assembly.

    The forward near kernel already keeps these compact lookup tables on the
    device.  The reverse fallback reconstructs one leaf-pair block at a time
    from this context instead of retaining every dense block simultaneously.
    """

    positions: np.ndarray
    leaf_particle_offsets: np.ndarray
    leaf_particle_indices: np.ndarray
    dst_leaf_indices: np.ndarray
    src_leaf_indices: np.ndarray
    radial_lut: RadialLUT
    ab5: np.ndarray
    lmax: int
    k: float
    near_dtype: np.dtype
    nm: int


@dataclass(frozen=True)
class CuPyMLFMMMultilevelData:
    """Device-ready container for multilevel repeated-apply structures."""

    levels: tuple[CuPyMLFMMLevelData, ...]
    transfers: tuple[CuPyMLFMMTransferData, ...]
    leaf_level: int
    hf_start_level: int
    hf_end_level: int
    box_nm: int
    n_leaves: int
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...]
    leaf_apply_mode: CuPyMLFMMLeafApplyMode = "dense"
    leaf_translation_tables: CuPyLeafTranslationTablesData | None = None


@dataclass(frozen=True)
class CuPyMLFMMPreparedData:
    """Top-level CuPy representation of a CPU-built MLFMM plan."""

    lmax: int
    stage: str
    near_pairs: CuPyMLFMMNearPairData
    single_level: CuPyMLFMMSingleLevelData | None = None
    multilevel: CuPyMLFMMMultilevelData | None = None


@dataclass
class CuPyMLFMMSingleLevelWorkspace:
    """Reusable single-level far-path work buffers for one RHS width."""

    nrhs: int
    box_states: Any
    outgoing: Any
    incoming: Any
    incoming_box: Any
    y_states: Any


@dataclass
class CuPyMLFMMMultilevelWorkspace:
    """Reusable multilevel far-path buffers for one RHS width.

    The sampled-far multilevel path now streams selected box chunks instead of
    keeping full per-level directional hierarchies resident, so the reusable
    workspace retains only the final particle-sized accumulator.
    """

    nrhs: int
    y_states: Any
    outgoing_roll_even: Any | None = None
    outgoing_roll_odd: Any | None = None
    incoming_roll_even: Any | None = None
    incoming_roll_odd: Any | None = None
    leaf_box_states: Any | None = None
    incoming_box: Any | None = None


@dataclass
class CuPyMLFMMNearWorkspace:
    """Reusable exact-near output buffer for one RHS width."""

    nrhs: int
    y_states: Any


@dataclass(frozen=True)
class _ExactLeafLaunchContext:
    """Device-specific launch invariants shared by central and image batches."""

    kernel: Any
    real_scalar_type: type[np.floating[Any]]
    threads: int
    max_grid_y: int
    max_grid_z: int


@dataclass(frozen=True)
class CuPyMLFMMSingleLevelWorkspaceKey:
    """Cache key for reusable single-level far workspaces."""

    nrhs: int
    n_particles: int
    nm: int
    n_leaves: int
    box_nm: int
    n_directions: int


@dataclass(frozen=True)
class CuPyMLFMMMultilevelWorkspaceKey:
    """Cache key for reusable multilevel far workspaces."""

    nrhs: int
    n_particles: int
    nm: int


@dataclass(frozen=True)
class CuPyMLFMMNearWorkspaceKey:
    """Cache key for reusable exact-near workspaces."""

    nrhs: int
    n_particles: int
    nm: int
    n_leaf_pairs: int
    near_dtype: str


def _cupy_complex_dtype(dtype: np.dtype, *, cupy: Any) -> Any:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.complex64):
        return cupy.complex64
    if dt == np.dtype(np.complex128):
        return cupy.complex128
    raise ValueError(f"Unsupported complex dtype {dt!r}.")


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


def _split_complex_table_transposed(
    table: np.ndarray,
    *,
    real_dtype: npt.DTypeLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous real/imaginary row-major arrays without a complex copy."""

    values_t = np.asarray(table).T
    return (
        np.ascontiguousarray(values_t.real.reshape(-1), dtype=real_dtype),
        np.ascontiguousarray(values_t.imag.reshape(-1), dtype=real_dtype),
    )


def _pack_index_lists(
    index_lists: tuple[np.ndarray, ...] | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros((len(index_lists) + 1,), dtype=np.int32)
    flat_parts: list[np.ndarray] = []
    cursor = 0
    for i, values in enumerate(index_lists):
        arr = np.asarray(values, dtype=np.int32).reshape(-1)
        flat_parts.append(np.ascontiguousarray(arr))
        cursor += int(arr.size)
        offsets[i + 1] = cursor
    flat = (
        np.concatenate(flat_parts, dtype=np.int32) if flat_parts else np.zeros((0,), dtype=np.int32)
    )
    return offsets, flat


def _leaf_rect_pair_tables_and_ab(
    *,
    full_order: int,
    out_mode_indices: np.ndarray,
    in_mode_indices: np.ndarray,
    out_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact rectangular `(out_mode, in_mode)` translation metadata.

    Unlike the full square CELES tables, this helper builds only the mode-pair
    subset needed by leaf on-the-fly aggregation/disaggregation:
    `n_modes(box_order) x n_modes(particle_lmax)`.
    """

    full = int(full_order)
    out_idx = np.ascontiguousarray(np.asarray(out_mode_indices, dtype=np.int32).reshape(-1))
    in_idx = np.ascontiguousarray(np.asarray(in_mode_indices, dtype=np.int32).reshape(-1))
    n_out = int(out_idx.size)
    n_in = int(in_idx.size)
    if n_out <= 0 or n_in <= 0:
        raise ValueError("Rectangular leaf translation tables require non-empty mode index sets.")

    mode_m_full = mode_m_table(full)
    mode_tau_full, mode_l_full = mode_tau_l_tables(full)
    mode_m_out = np.ascontiguousarray(mode_m_full[out_idx], dtype=np.int32)
    mode_m_in = np.ascontiguousarray(mode_m_full[in_idx], dtype=np.int32)

    pair_offset = np.zeros((n_out * n_in,), dtype=np.int32)
    pair_pmin = np.zeros_like(pair_offset)
    pair_pcount = np.zeros_like(pair_offset)
    re_entries: list[float] = []
    im_entries: list[float] = []
    offset = 0

    for out_local, n_out_mode in enumerate(out_idx.tolist()):
        tau2 = int(mode_tau_full[n_out_mode])
        l2 = int(mode_l_full[n_out_mode])
        m2 = int(mode_m_full[n_out_mode])
        for in_local, n_in_mode in enumerate(in_idx.tolist()):
            tau1 = int(mode_tau_full[n_in_mode])
            l1 = int(mode_l_full[n_in_mode])
            m1 = int(mode_m_full[n_in_mode])

            p_min = max(abs(m1 - m2), abs(l1 - l2) + abs(tau1 - tau2))
            p_max = l1 + l2
            p_count = p_max - p_min + 1
            pair_meta_idx = out_local * n_in + in_local
            pair_offset[pair_meta_idx] = offset
            pair_pmin[pair_meta_idx] = p_min
            pair_pcount[pair_meta_idx] = p_count

            phase_exp_base = abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1
            sign_dm = -1.0 if ((m1 - m2) % 2) else 1.0
            pref = np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1)))

            for p in range(p_min, p_max + 1):
                if tau1 == tau2:
                    i_phase = (1j) ** (phase_exp_base + p)
                    factor = (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1)
                    w = wigner_3j(l1, l2, p, m1, -m2, -m1 + m2) * wigner_3j(l1, l2, p, 0, 0, 0)
                    entry = i_phase * sign_dm * pref * factor * w
                else:
                    if p == 0:
                        entry = 0.0 + 0.0j
                    else:
                        inside = (
                            (l1 + l2 + 1 + p)
                            * (l1 + l2 + 1 - p)
                            * (p + l1 - l2)
                            * (p - l1 + l2)
                            * (2 * p + 1)
                        )
                        if inside < 0:
                            entry = 0.0 + 0.0j
                        else:
                            i_phase = (1j) ** (phase_exp_base + p)
                            factor = np.sqrt(inside)
                            w = wigner_3j(l1, l2, p, m1, -m2, -m1 + m2) * wigner_3j(
                                l1, l2, p - 1, 0, 0, 0
                            )
                            entry = i_phase * sign_dm * pref * factor * w
                re_entries.append(float(np.real(entry)))
                im_entries.append(float(np.imag(entry)))
                offset += 1

    real_dtype = np.float32 if out_dtype == np.dtype(np.complex64) else np.float64
    return (
        np.ascontiguousarray(np.asarray(re_entries, dtype=real_dtype)),
        np.ascontiguousarray(np.asarray(im_entries, dtype=real_dtype)),
        np.ascontiguousarray(pair_offset, dtype=np.int32),
        np.ascontiguousarray(pair_pmin, dtype=np.int32),
        np.ascontiguousarray(pair_pcount, dtype=np.int32),
        mode_m_out,
        mode_m_in,
    )


@cache
def _leaf_translation_blocks_rect_raw_kernel(full_order: int, dtype_name: str) -> Any:
    """Return a cached RawKernel for batched interior rectangular translation blocks.

    Launch policy is one CUDA block per particle-to-leaf pair. This keeps the
    expensive spherical precompute (`j_p`, `P_p^m`, `exp(i m phi)`) shared once
    per pair instead of duplicating it across `(out_mode, in_mode)` tiles.
    """

    cupy, _ = import_cupy()
    order = int(full_order)
    out_dtype = np.dtype(dtype_name)
    if out_dtype == np.dtype(np.complex64):
        real_t = "float"
        complex_t = "complex<float>"
    elif out_dtype == np.dtype(np.complex128):
        real_t = "double"
        complex_t = "complex<double>"
    else:
        raise ValueError(
            "Leaf-map GPU kernel supports only complex64/complex128 output dtypes. "
            f"Got {out_dtype!r}."
        )
    n_orders = 2 * order + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    sincos_fn = "sincosf" if real_t == "float" else "sincos"
    source = f"""
    #include <cupy/complex.cuh>
    __device__ {real_t} assoc_legendre_function(
        const int l,
        const int m,
        const {real_t}* ct_powers,
        const {real_t}* st_powers,
        const {real_t}* plm_coeffs
    ) {{
        {real_t} plm = ({real_t})0.0;
        const {real_t} st_pow = st_powers[m];
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    __device__ {complex_t} bessel_lookup_linear(
        const int p,
        const {real_t} r,
        const {real_t}* re_table,
        const {real_t}* im_table,
        const {real_t} inv_dr,
        const int last_index
    ) {{
        if (r <= ({real_t})0.0) {{
            return {complex_t}(re_table[p], im_table[p]);
        }}
        {real_t} t = r * inv_dr;
        int i0 = (int)floor(t);
        {real_t} frac = t - ({real_t})i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = ({real_t})0.0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = ({real_t})1.0;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        const {real_t} re_val = (({real_t})1.0 - frac) * re_table[base0] + frac * re_table[base1];
        const {real_t} im_val = (({real_t})1.0 - frac) * im_table[base0] + frac * im_table[base1];
        return {complex_t}(re_val, im_val);
    }}

    extern "C" __global__ void mlfmm_leaf_translation_blocks_rect(
        const int n_pairs,
        const int n_out_modes,
        const int n_in_modes,
        const {real_t}* pair_deltas,
        const int* out_mode_indices,
        const int* in_mode_indices,
        const int* mode_m_out,
        const int* mode_m_in,
        const {real_t}* re_j,
        const {real_t}* im_j,
        const {real_t} inv_dr,
        const int last_index,
        const {real_t}* plm_coeffs,
        const {real_t}* re_ab,
        const {real_t}* im_ab,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        {complex_t}* out_blocks
    ) {{
        const int pair_idx = blockIdx.x;
        if (pair_idx >= n_pairs) {{
            return;
        }}
        const int tid = threadIdx.x;
        const int n_threads = blockDim.x;
        const int n_mode_pairs = n_out_modes * n_in_modes;

        __shared__ {real_t} r_shared;
        __shared__ {real_t} ct_shared;
        __shared__ {real_t} st_shared;
        __shared__ {real_t} phi_shared;
        __shared__ {real_t} re_j_shared[{n_orders}];
        __shared__ {real_t} im_j_shared[{n_orders}];
        __shared__ {real_t} p_pdm_shared[{n_p_pdm}];
        __shared__ {real_t} cos_mphi_shared[{n_phase}];
        __shared__ {real_t} sin_mphi_shared[{n_phase}];
        __shared__ {real_t} ct_pow_shared[{n_orders}];
        __shared__ {real_t} st_pow_shared[{n_orders}];

        if (tid == 0) {{
            const {real_t} dx = pair_deltas[3 * pair_idx + 0];
            const {real_t} dy = pair_deltas[3 * pair_idx + 1];
            const {real_t} dz = pair_deltas[3 * pair_idx + 2];
            const {real_t} rr = sqrt(dx * dx + dy * dy + dz * dz);
            r_shared = rr;
            if (rr > ({real_t})0.0) {{
                ct_shared = dz / rr;
                st_shared = sqrt(fmax(({real_t})0.0, ({real_t})1.0 - ct_shared * ct_shared));
                phi_shared = atan2(dy, dx);
            }} else {{
                ct_shared = ({real_t})1.0;
                st_shared = ({real_t})0.0;
                phi_shared = ({real_t})0.0;
            }}
        }}
        __syncthreads();

        if (r_shared <= ({real_t})0.0) {{
            for (int flat = tid; flat < n_mode_pairs; flat += n_threads) {{
                const int out_idx = flat / n_in_modes;
                const int in_idx = flat - out_idx * n_in_modes;
                const int n1_zero = out_mode_indices[out_idx];
                const int n2_zero = in_mode_indices[in_idx];
                const long long flat_idx = (long long)pair_idx * (long long)n_mode_pairs + (long long)flat;
                if (n1_zero == n2_zero) {{
                    out_blocks[flat_idx] = {complex_t}(({real_t})1.0, ({real_t})0.0);
                }} else {{
                    out_blocks[flat_idx] = {complex_t}(({real_t})0.0, ({real_t})0.0);
                }}
            }}
            return;
        }}

        if (tid == 0) {{
            ct_pow_shared[0] = ({real_t})1.0;
            st_pow_shared[0] = ({real_t})1.0;
            for (int p = 1; p < {n_orders}; ++p) {{
                ct_pow_shared[p] = ct_pow_shared[p - 1] * ct_shared;
                st_pow_shared[p] = st_pow_shared[p - 1] * st_shared;
            }}
        }}
        __syncthreads();

        for (int p = tid; p < {n_orders}; p += n_threads) {{
            const {complex_t} radial = bessel_lookup_linear(p, r_shared, re_j, im_j, inv_dr, last_index);
            re_j_shared[p] = radial.real();
            im_j_shared[p] = radial.imag();
            for (int absdm = 0; absdm <= p; ++absdm) {{
                p_pdm_shared[p * (p + 1) / 2 + absdm] =
                    assoc_legendre_function(
                        p, absdm, ct_pow_shared, st_pow_shared, plm_coeffs
                    );
            }}
        }}
        if (tid == 0) {{
            for (int dm = -2 * {order}; dm <= 2 * {order}; ++dm) {{
                const int phase_idx = dm + 2 * {order};
                {real_t} s_val;
                {real_t} c_val;
                {sincos_fn}(({real_t})dm * phi_shared, &s_val, &c_val);
                cos_mphi_shared[phase_idx] = c_val;
                sin_mphi_shared[phase_idx] = s_val;
            }}
        }}
        __syncthreads();

        for (int flat = tid; flat < n_mode_pairs; flat += n_threads) {{
            const int out_idx = flat / n_in_modes;
            const int in_idx = flat - out_idx * n_in_modes;
            const int delta_m = mode_m_in[in_idx] - mode_m_out[out_idx];
            const int phase_idx = delta_m + 2 * {order};
            const int table_idx = flat;
            const int base = pair_offset[table_idx];
            const int p_min = pair_pmin[table_idx];
            const int p_count = pair_pcount[table_idx];
            {real_t} re_acc = ({real_t})0.0;
            {real_t} im_acc = ({real_t})0.0;
            for (int ip = 0; ip < p_count; ++ip) {{
                const int p = p_min + ip;
                const int ab_idx = base + ip;
                const {real_t} plm = p_pdm_shared[p * (p + 1) / 2 + abs(delta_m)];
                const {real_t} re_abp = re_ab[ab_idx] * plm;
                const {real_t} im_abp = im_ab[ab_idx] * plm;
                const {real_t} re_abpr = re_abp * re_j_shared[p] - im_abp * im_j_shared[p];
                const {real_t} im_abpr = re_abp * im_j_shared[p] + im_abp * re_j_shared[p];
                const {real_t} re_phase =
                    re_abpr * cos_mphi_shared[phase_idx] - im_abpr * sin_mphi_shared[phase_idx];
                const {real_t} im_phase =
                    re_abpr * sin_mphi_shared[phase_idx] + im_abpr * cos_mphi_shared[phase_idx];
                re_acc += re_phase;
                im_acc += im_phase;
            }}
            const long long flat_idx = (long long)pair_idx * (long long)n_mode_pairs + (long long)flat;
            out_blocks[flat_idx] = {complex_t}(re_acc, im_acc);
        }}
    }}
    """
    return cupy.RawKernel(source, "mlfmm_leaf_translation_blocks_rect")


def build_leaf_box_maps_cupy(
    *,
    lmax: int,
    box_order: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT | None,
    dtype: np.dtype | type[np.complexfloating] | type[np.complex128],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build leaf aggregation/receive maps with a CuPy-kernelized block build stage."""

    if radial_lut is None:
        raise ValueError("CuPy leaf-map build requires a radial_lut for interior bessel sampling.")

    cupy, _ = import_cupy()
    out_dtype = np.dtype(dtype)
    if out_dtype == np.dtype(np.complex64):
        real_dtype: np.dtype[Any] = np.dtype(np.float32)
        cupy_complex_dtype = cupy.complex64
        cupy_real_dtype = cupy.float32
    elif out_dtype == np.dtype(np.complex128):
        real_dtype = np.dtype(np.float64)
        cupy_complex_dtype = cupy.complex128
        cupy_real_dtype = cupy.float64
    else:
        raise ValueError(
            "CuPy leaf-map build supports only complex64/complex128 output dtypes. "
            f"Got {out_dtype!r}."
        )

    lmax_in = int(lmax)
    lmax_out = int(box_order)
    full_order = max(lmax_in, lmax_out)
    nmodes_in = int(n_modes(lmax_in))
    nmodes_out = int(n_modes(lmax_out))
    leaves = tuple(sorted(partition.leaves, key=lambda leaf: int(leaf.id)))
    n_leaves = len(leaves)
    if n_leaves == 0:
        return tuple(), tuple()

    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    if int(radial_lut.lmax) < full_order:
        radial_lut_full = RadialLUT(
            lmax=full_order,
            k=float(k),
            r_max=float(radial_lut.r_grid[-1]),
            dr=float(radial_lut.dr),
            dtype=out_dtype,
        )
    else:
        radial_lut_full = radial_lut

    out_mode_idx = np.fromiter(
        (
            index_vswf(int(l_i), int(m_i), int(tau_i), full_order)
            for tau_i, l_i, m_i, _idx in iter_modes(lmax_out)
        ),
        dtype=np.int32,
        count=nmodes_out,
    )
    in_mode_idx = np.fromiter(
        (
            index_vswf(int(l_i), int(m_i), int(tau_i), full_order)
            for tau_i, l_i, m_i, _idx in iter_modes(lmax_in)
        ),
        dtype=np.int32,
        count=nmodes_in,
    )

    (
        compact_re_ab_raw,
        compact_im_ab_raw,
        pair_offset_raw,
        pair_pmin_raw,
        pair_pcount_raw,
        mode_m_out_raw,
        mode_m_in_raw,
    ) = _leaf_rect_pair_tables_and_ab(
        full_order=int(full_order),
        out_mode_indices=out_mode_idx,
        in_mode_indices=in_mode_idx,
        out_dtype=np.dtype(out_dtype),
    )
    plm_coeffs_raw = _translation_plm_coeff_table(full_order, dtype=real_dtype).reshape(-1)

    re_j_raw, im_j_raw = _split_complex_table_transposed(
        np.asarray(radial_lut_full.j)[: 2 * full_order + 1, :],
        real_dtype=real_dtype,
    )
    last_index = int(radial_lut_full._last_index)
    inv_dr_scalar = (
        np.float32(float(radial_lut_full._inv_dr))
        if out_dtype == np.dtype(np.complex64)
        else np.float64(float(radial_lut_full._inv_dr))
    )

    aggregation_by_id: list[np.ndarray | None] = [None] * n_leaves
    pair_leaf_ids: list[int] = []
    pair_local_ids: list[int] = []
    pair_delta_parts: list[np.ndarray] = []
    for leaf in leaves:
        leaf_id = int(leaf.id)
        if leaf_id < 0 or leaf_id >= n_leaves:
            raise ValueError(
                "Leaf ids must be contiguous in [0, n_leaves) for grouped apply indexing. "
                f"Got leaf_id={leaf_id}, n_leaves={n_leaves}."
            )
        particle_indices = np.asarray(leaf.particle_indices, dtype=np.int64).reshape(-1)
        occupancy = int(particle_indices.size)
        if occupancy <= 0:
            raise ValueError(f"leaf {leaf_id} has non-positive occupancy {occupancy}.")
        aggregation_by_id[leaf_id] = np.empty(
            (nmodes_out, occupancy * nmodes_in),
            dtype=out_dtype,
        )
        leaf_center = np.asarray(leaf.center, dtype=float).reshape(1, 3)
        deltas = np.ascontiguousarray(
            leaf_center - positions_arr[particle_indices, :],
            dtype=real_dtype,
        )
        pair_delta_parts.append(deltas)
        pair_leaf_ids.extend([leaf_id] * occupancy)
        pair_local_ids.extend(range(occupancy))

    if not pair_delta_parts:
        raise RuntimeError("Internal error: empty leaf-map pair schedule.")
    pair_deltas = np.ascontiguousarray(np.vstack(pair_delta_parts), dtype=real_dtype)
    pair_leaf_ids_arr = np.ascontiguousarray(np.asarray(pair_leaf_ids, dtype=np.int32).reshape(-1))
    pair_local_ids_arr = np.ascontiguousarray(
        np.asarray(pair_local_ids, dtype=np.int32).reshape(-1)
    )
    n_pairs = int(pair_deltas.shape[0])
    if n_pairs != int(pair_leaf_ids_arr.size):
        raise RuntimeError("Internal error: inconsistent pair schedule sizes.")

    out_mode_idx_dev = cupy.asarray(np.ascontiguousarray(out_mode_idx), dtype=cupy.int32)
    in_mode_idx_dev = cupy.asarray(np.ascontiguousarray(in_mode_idx), dtype=cupy.int32)
    mode_m_out_dev = cupy.asarray(np.ascontiguousarray(mode_m_out_raw), dtype=cupy.int32)
    mode_m_in_dev = cupy.asarray(np.ascontiguousarray(mode_m_in_raw), dtype=cupy.int32)
    pair_offset_dev = cupy.asarray(
        np.ascontiguousarray(pair_offset_raw, dtype=np.int32), dtype=cupy.int32
    )
    pair_pmin_dev = cupy.asarray(
        np.ascontiguousarray(pair_pmin_raw, dtype=np.int32), dtype=cupy.int32
    )
    pair_pcount_dev = cupy.asarray(
        np.ascontiguousarray(pair_pcount_raw, dtype=np.int32), dtype=cupy.int32
    )
    plm_coeffs_dev = cupy.asarray(np.ascontiguousarray(plm_coeffs_raw), dtype=cupy_real_dtype)
    re_ab_dev = cupy.asarray(np.ascontiguousarray(compact_re_ab_raw), dtype=cupy_real_dtype)
    im_ab_dev = cupy.asarray(np.ascontiguousarray(compact_im_ab_raw), dtype=cupy_real_dtype)
    re_j_dev = cupy.asarray(re_j_raw, dtype=cupy_real_dtype)
    im_j_dev = cupy.asarray(im_j_raw, dtype=cupy_real_dtype)

    kernel = _leaf_translation_blocks_rect_raw_kernel(full_order, out_dtype.str)
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_pairs = int(props["maxGridSize"][0])
    max_threads = int(props["maxThreadsPerBlock"])
    if max_threads >= 256:
        threads = 256
    elif max_threads >= 128:
        threads = 128
    else:
        threads = 64

    bytes_per_pair = max(1, nmodes_out * nmodes_in * out_dtype.itemsize)
    target_chunk_bytes = 256 * 1024 * 1024
    chunk_pairs = max(1, target_chunk_bytes // bytes_per_pair)
    chunk_pairs = max(1, min(int(chunk_pairs), int(max_grid_pairs)))

    for start in range(0, n_pairs, chunk_pairs):
        end = min(n_pairs, start + chunk_pairs)
        count = int(end - start)
        deltas_dev = cupy.asarray(
            np.ascontiguousarray(pair_deltas[start:end, :]),
            dtype=cupy_real_dtype,
            blocking=True,
        )
        blocks_dev = cupy.empty((count, nmodes_out, nmodes_in), dtype=cupy_complex_dtype)
        kernel(
            (int(count),),
            (int(threads),),
            (
                np.int32(count),
                np.int32(nmodes_out),
                np.int32(nmodes_in),
                deltas_dev.reshape(-1),
                out_mode_idx_dev,
                in_mode_idx_dev,
                mode_m_out_dev,
                mode_m_in_dev,
                re_j_dev,
                im_j_dev,
                inv_dr_scalar,
                np.int32(last_index),
                plm_coeffs_dev,
                re_ab_dev,
                im_ab_dev,
                pair_offset_dev,
                pair_pmin_dev,
                pair_pcount_dev,
                blocks_dev.reshape(-1),
            ),
        )
        blocks_host = np.asarray(cupy.asnumpy(blocks_dev), dtype=out_dtype)
        leaf_chunk = pair_leaf_ids_arr[start:end]
        local_chunk = pair_local_ids_arr[start:end]
        for idx_local in range(count):
            leaf_id = int(leaf_chunk[idx_local])
            local_particle = int(local_chunk[idx_local])
            col_start = local_particle * nmodes_in
            col_stop = col_start + nmodes_in
            agg_leaf = aggregation_by_id[leaf_id]
            if agg_leaf is None:
                raise RuntimeError(
                    f"Internal error: missing aggregation buffer for leaf {leaf_id}."
                )
            agg_leaf[:, col_start:col_stop] = blocks_host[idx_local]

    if any(agg is None for agg in aggregation_by_id):
        raise RuntimeError("Internal error: incomplete CuPy leaf-map aggregation build.")
    aggregation = tuple(
        np.asarray(aggregation_by_id[leaf_id], dtype=out_dtype) for leaf_id in range(n_leaves)
    )
    receive = tuple(np.asarray(np.conjugate(agg).T, dtype=out_dtype) for agg in aggregation)
    return aggregation, receive


def _host_directional_grid_size(grid: Any) -> tuple[int, int, int]:
    """Return `(n_alpha, n_beta, n_directions)` for one host directional grid."""

    if hasattr(grid, "n_alpha") and hasattr(grid, "n_beta"):
        n_alpha = int(grid.n_alpha)
        n_beta = int(grid.n_beta)
        n_dir = int(grid.n_directions)
    else:
        alpha = _as_numpy_1d(grid.alpha, dtype=np.float64, name="directional.grid.alpha")
        beta = _as_numpy_1d(grid.beta, dtype=np.float64, name="directional.grid.beta")
        directions = _as_numpy_3cols(
            grid.directions, dtype=np.float64, name="directional.grid.directions"
        )
        weights = _as_numpy_1d(grid.weights, dtype=np.float64, name="directional.grid.weights")
        if directions.shape[0] != weights.size:
            raise ValueError(
                "Directional grid directions/weights mismatch: "
                f"{directions.shape[0]} vs {weights.size}."
            )
        n_alpha = int(alpha.size)
        n_beta = int(beta.size)
        n_dir = int(directions.shape[0])
    if n_alpha <= 0 or n_beta <= 0 or n_dir != n_alpha * n_beta:
        raise ValueError(
            "Directional grid dimensions are inconsistent: "
            f"n_alpha={n_alpha}, n_beta={n_beta}, n_directions={n_dir}."
        )
    return n_alpha, n_beta, n_dir


def _directional_structured_host_view(
    transforms: MLFMMDirectionalTransformData | CuPyHostDirectionalTransformsData,
) -> tuple[CuPyHostDirectionalGridData, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact structured factors for one host directional transform."""

    if isinstance(transforms, CuPyHostDirectionalTransformsData):
        grid = transforms.grid
        fth_beta = _as_numpy_2d(
            transforms.fth_beta, dtype=np.complex128, name="directional.fth_beta"
        )
        fph_beta = _as_numpy_2d(
            transforms.fph_beta, dtype=np.complex128, name="directional.fph_beta"
        )
        m_of_scalar = _as_numpy_1d(
            transforms.m_of_scalar, dtype=np.int32, name="directional.m_of_scalar"
        )
        return grid, fth_beta, fph_beta, m_of_scalar

    structured = (
        transforms
        if isinstance(transforms, MLFMMDirectionalStructuredTransforms)
        else structured_directional_transforms(
            int(transforms.box_order), grid_order=int(transforms.grid.order)
        )
    )
    grid_raw = structured.grid
    n_alpha = int(grid_raw.alpha.size)
    n_beta = int(grid_raw.beta.size)
    grid = CuPyHostDirectionalGridData(
        order=int(grid_raw.order),
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_directions=n_alpha * n_beta,
        alpha=np.ascontiguousarray(np.asarray(grid_raw.alpha, dtype=np.float64).reshape(-1)),
        reflection_permutation=np.ascontiguousarray(
            np.asarray(grid_raw.reflection_permutation, dtype=np.int32).reshape(-1)
        ),
    )
    return (
        grid,
        np.ascontiguousarray(np.asarray(structured.fth_beta, dtype=np.complex128)),
        np.ascontiguousarray(np.asarray(structured.fph_beta, dtype=np.complex128)),
        np.ascontiguousarray(np.asarray(structured.m_of_scalar, dtype=np.int32).reshape(-1)),
    )


def _beta_reflection_permutation(
    reflection: np.ndarray,
    *,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Extract the beta-only reflection permutation from one directional grid."""

    reflection_grid = np.asarray(reflection, dtype=np.int32).reshape(int(n_alpha), int(n_beta))
    beta_perm = np.ascontiguousarray(reflection_grid[0], dtype=np.int32)
    expected = np.arange(int(n_alpha), dtype=np.int32)[:, None] * int(n_beta) + beta_perm[None, :]
    if not np.array_equal(reflection_grid, expected):
        raise ValueError("Directional reflection must preserve alpha and permute beta only.")
    return beta_perm


def _upload_directional_transforms(
    transforms: MLFMMDirectionalTransformData | CuPyHostDirectionalTransformsData, *, cupy: Any
) -> CuPyDirectionalTransformsData:
    grid, fth_beta, fph_beta, m_of_scalar = _directional_structured_host_view(transforms)
    n_alpha, n_beta, n_dir = _host_directional_grid_size(grid)
    reflection = _as_numpy_1d(
        grid.reflection_permutation,
        dtype=np.int32,
        name="directional.grid.reflection_permutation",
    )
    if reflection.size != n_dir:
        raise ValueError(
            f"Directional grid reflection-permutation size mismatch: {reflection.size} vs {n_dir}."
        )
    beta_reflection = _beta_reflection_permutation(
        reflection,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
    if fth_beta.shape != fph_beta.shape:
        raise ValueError("Directional theta/phi beta-factor shapes must match.")
    if fth_beta.shape[0] != n_beta:
        raise ValueError(
            "Directional beta-factor row count must match beta grid size: "
            f"{fth_beta.shape[0]} vs {n_beta}."
        )
    nscl = int(fth_beta.shape[1])
    if m_of_scalar.size != nscl:
        raise ValueError(
            f"Directional m table length must match scalar modes: {m_of_scalar.size} vs {nscl}."
        )

    box_order = int(transforms.box_order)
    m_values = np.arange(-box_order, box_order + 1, dtype=np.int32)
    alpha = _as_numpy_1d(grid.alpha, dtype=np.float64, name="directional.grid.alpha")
    if alpha.size != n_alpha:
        raise ValueError(f"Directional alpha size mismatch: {alpha.size} vs {n_alpha}.")
    phase_by_m = np.asarray(
        np.exp(1j * alpha[:, None] * m_values[None, :]),
        dtype=np.complex128,
    )
    mode_indices_by_m = tuple(
        cupy.asarray(
            np.ascontiguousarray(np.flatnonzero(m_of_scalar == int(m)).astype(np.int32)),
            dtype=cupy.int32,
        )
        for m in m_values.tolist()
    )

    return CuPyDirectionalTransformsData(
        box_order=box_order,
        grid_order=int(grid.order),
        grid=CuPyDirectionalGridData(
            order=int(grid.order),
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_directions=int(n_dir),
            reflection_permutation=cupy.asarray(np.asarray(reflection, dtype=np.int32)),
            beta_reflection_permutation=cupy.asarray(beta_reflection, dtype=cupy.int32),
        ),
        nscl=nscl,
        fth_beta=cupy.asarray(np.ascontiguousarray(fth_beta, dtype=np.complex128)),
        fph_beta=cupy.asarray(np.ascontiguousarray(fph_beta, dtype=np.complex128)),
        m_of_scalar=cupy.asarray(np.ascontiguousarray(m_of_scalar, dtype=np.int32)),
        phase_by_m=cupy.asarray(np.ascontiguousarray(phase_by_m, dtype=np.complex128)),
        mode_indices_by_m=mode_indices_by_m,
    )


def _upload_directional_map(
    matrix_csr: Any,
    *,
    source_order: int,
    target_order: int,
    cupy: Any,
    cupyx_sparse: Any,
) -> CuPyDirectionalInterpolationData:
    """Upload one directional transfer map with adaptive dense/sparse storage.

    Dense storage is only enabled for substantially dense operators. For
    low-width stencil-like maps, a packed row-stencil kernel is preferred.
    """

    csr = matrix_csr.tocsr().astype(np.complex128)
    rows, cols = int(csr.shape[0]), int(csr.shape[1])
    nnz = int(csr.nnz)
    total = rows * cols
    density = (float(nnz) / float(total)) if total > 0 else 0.0
    dense_bytes = total * np.dtype(np.complex128).itemsize
    use_dense = bool(total > 0 and dense_bytes <= (128 * 1024 * 1024) and density >= 2.5e-1)
    if use_dense:
        dense = np.ascontiguousarray(csr.toarray(), dtype=np.complex128)
        return CuPyDirectionalInterpolationData(
            source_order=int(source_order),
            target_order=int(target_order),
            matrix=cupy.asarray(dense, dtype=cupy.complex128),
            nnz=nnz,
            storage="dense",
        )

    row_ptr = np.ascontiguousarray(np.asarray(csr.indptr, dtype=np.int32))
    row_nnz = np.diff(row_ptr)
    max_row_nnz = int(row_nnz.max(initial=0))
    use_packed_stencil = bool(nnz > 0 and max_row_nnz > 0 and max_row_nnz <= 32)
    if use_packed_stencil:
        idx_pack = np.full((rows, max_row_nnz), -1, dtype=np.int32)
        val_pack = np.zeros((rows, max_row_nnz), dtype=np.complex128)
        cols_arr = np.asarray(csr.indices, dtype=np.int32)
        vals_arr = np.asarray(csr.data, dtype=np.complex128)
        for row in range(rows):
            start = int(row_ptr[row])
            end = int(row_ptr[row + 1])
            width = end - start
            if width <= 0:
                continue
            idx_pack[row, :width] = cols_arr[start:end]
            val_pack[row, :width] = vals_arr[start:end]
        packed = (
            cupy.asarray(np.ascontiguousarray(idx_pack.reshape(-1)), dtype=cupy.int32),
            cupy.asarray(np.ascontiguousarray(val_pack.reshape(-1)), dtype=cupy.complex128),
            np.int32(max_row_nnz),
        )
        return CuPyDirectionalInterpolationData(
            source_order=int(source_order),
            target_order=int(target_order),
            matrix=packed,
            nnz=nnz,
            storage="packed_stencil",
        )

    data = np.ascontiguousarray(np.asarray(csr.data, dtype=np.complex128))
    indices = np.ascontiguousarray(np.asarray(csr.indices, dtype=np.int32))
    indptr = row_ptr
    sparse = cupyx_sparse.csr_matrix(
        (
            cupy.asarray(data, dtype=cupy.complex128),
            cupy.asarray(indices, dtype=cupy.int32),
            cupy.asarray(indptr, dtype=cupy.int32),
        ),
        shape=(rows, cols),
    )
    return CuPyDirectionalInterpolationData(
        source_order=int(source_order),
        target_order=int(target_order),
        matrix=sparse,
        nnz=nnz,
        storage="sparse",
    )


def _upload_unique_index_batch(
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    *,
    cupy: Any,
    name: str,
) -> CuPyOffsetBatchData:
    """Upload one unique grouped source/destination schedule."""

    src = _as_numpy_1d(source_indices, dtype=np.int32, name=f"{name}.src")
    dst = _as_numpy_1d(destination_indices, dtype=np.int32, name=f"{name}.dst")
    if src.size != dst.size:
        raise ValueError(f"{name} source/target batch size mismatch: {src.size} vs {dst.size}.")
    src_unique = bool(np.unique(src).size == src.size)
    dst_unique = bool(np.unique(dst).size == dst.size)
    if not src_unique or not dst_unique:
        raise ValueError(
            f"{name} violates grouped uniqueness contract "
            f"(src_unique={src_unique}, dst_unique={dst_unique})."
        )
    return CuPyOffsetBatchData(
        src_indices=cupy.asarray(src, dtype=cupy.int32),
        dst_indices=cupy.asarray(dst, dtype=cupy.int32),
    )


def _upload_offset_batches(
    batches: dict[Offset3, tuple[np.ndarray, np.ndarray]], *, cupy: Any, name: str
) -> dict[Offset3, CuPyOffsetBatchData]:
    """Upload grouped source/target batches and enforce uniqueness contract."""

    return {
        offset: _upload_unique_index_batch(
            src_idx,
            dst_idx,
            cupy=cupy,
            name=f"{name}[{offset}]",
        )
        for offset, (src_idx, dst_idx) in batches.items()
    }


def _partition_from_host_cache(cache: CuPyMLFMMHostCacheData) -> CuPyMLFMMPartitionData:
    """Build upload-only partition lookup from compact host cache data."""

    return CuPyMLFMMPartitionData(
        leaf_particle_offsets_host=np.asarray(cache.leaf_particle_offsets, dtype=np.int32),
        leaf_particle_indices_host=np.asarray(cache.leaf_particle_indices, dtype=np.int32),
    )


def _copy_batches_host(
    batches: dict[Offset3, tuple[np.ndarray, np.ndarray]],
) -> dict[Offset3, tuple[np.ndarray, np.ndarray]]:
    """Return contiguous host copies of index batches for cache payloads."""

    out: dict[Offset3, tuple[np.ndarray, np.ndarray]] = {}
    for key, (src, dst) in batches.items():
        out[key] = (
            np.ascontiguousarray(np.asarray(src, dtype=np.int32).reshape(-1)),
            np.ascontiguousarray(np.asarray(dst, dtype=np.int32).reshape(-1)),
        )
    return out


def _copy_directional_host(
    transforms: MLFMMDirectionalTransformData,
) -> CuPyHostDirectionalTransformsData:
    """Extract separable directional factors into a compact host payload."""

    structured = (
        transforms
        if isinstance(transforms, MLFMMDirectionalStructuredTransforms)
        else structured_directional_transforms(
            int(transforms.box_order), grid_order=int(transforms.grid.order)
        )
    )
    n_alpha = int(structured.grid.alpha.size)
    n_beta = int(structured.grid.beta.size)
    return CuPyHostDirectionalTransformsData(
        box_order=int(structured.box_order),
        grid_order=int(structured.grid.order),
        grid=CuPyHostDirectionalGridData(
            order=int(structured.grid.order),
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_directions=n_alpha * n_beta,
            alpha=np.ascontiguousarray(np.asarray(structured.grid.alpha, dtype=np.float64)),
            reflection_permutation=np.ascontiguousarray(
                np.asarray(structured.grid.reflection_permutation, dtype=np.int32).reshape(-1)
            ),
        ),
        fth_beta=np.ascontiguousarray(np.asarray(structured.fth_beta, dtype=np.complex128)),
        fph_beta=np.ascontiguousarray(np.asarray(structured.fph_beta, dtype=np.complex128)),
        m_of_scalar=np.ascontiguousarray(
            np.asarray(structured.m_of_scalar, dtype=np.int32).reshape(-1)
        ),
    )


def _build_host_leaf_otf_payload(
    *,
    lmax: int,
    box_order: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT,
    out_dtype: np.dtype,
) -> tuple[tuple[CuPyHostLeafOnTheFlyGroupData, ...], CuPyHostLeafTranslationTablesData]:
    """Build compact grouped leaf schedules and translation tables for on-the-fly apply."""

    lmax_in = int(lmax)
    lmax_out = int(box_order)
    full_order = max(lmax_in, lmax_out)
    nmodes_in = int(n_modes(lmax_in))
    nmodes_out = int(n_modes(lmax_out))
    nmodes_full = int(n_modes(full_order))

    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    leaves = tuple(sorted(partition.leaves, key=lambda leaf: int(leaf.id)))
    n_leaves = len(leaves)
    leaf_by_id = {int(leaf.id): leaf for leaf in leaves}
    for leaf_id in range(n_leaves):
        if leaf_id not in leaf_by_id:
            raise ValueError(
                "Leaf ids must be contiguous in [0, n_leaves) for on-the-fly CuPy leaf schedules. "
                f"Missing leaf id {leaf_id}."
            )

    grouped_ids: dict[int, list[int]] = {}
    for leaf_id in range(n_leaves):
        leaf = leaf_by_id[leaf_id]
        occupancy = int(np.asarray(leaf.particle_indices, dtype=np.int64).reshape(-1).size)
        if occupancy <= 0:
            raise ValueError(f"leaf {leaf_id} has non-positive occupancy {occupancy}.")
        grouped_ids.setdefault(occupancy, []).append(leaf_id)

    groups: list[CuPyHostLeafOnTheFlyGroupData] = []
    max_pair_radius = 0.0
    for occupancy in sorted(grouped_ids):
        leaf_ids_np = np.asarray(grouped_ids[occupancy], dtype=np.int32)
        n_group = int(leaf_ids_np.size)
        particle_indices_np = np.empty((n_group, occupancy), dtype=np.int32)
        pair_deltas_np = np.empty((n_group * occupancy, 3), dtype=np.float64)
        for local_idx, leaf_id in enumerate(leaf_ids_np.tolist()):
            leaf = leaf_by_id[int(leaf_id)]
            particle_idx = np.asarray(leaf.particle_indices, dtype=np.int32).reshape(-1)
            if int(particle_idx.size) != occupancy:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: occupancy grouping mismatch while building "
                    f"on-the-fly leaf schedule for leaf {leaf_id}."
                )
            particle_indices_np[local_idx] = particle_idx
            leaf_center = np.asarray(leaf.center, dtype=np.float64).reshape(1, 3)
            start = local_idx * occupancy
            stop = start + occupancy
            pair_deltas_np[start:stop, :] = leaf_center - positions_arr[particle_idx, :]
        if pair_deltas_np.size:
            max_pair_radius = max(
                max_pair_radius,
                float(np.max(np.linalg.norm(pair_deltas_np, axis=1), initial=0.0)),
            )
        groups.append(
            CuPyHostLeafOnTheFlyGroupData(
                occupancy=int(occupancy),
                nmodes=int(nmodes_in),
                leaf_ids=np.ascontiguousarray(leaf_ids_np, dtype=np.int32),
                particle_indices=np.ascontiguousarray(particle_indices_np, dtype=np.int32),
                pair_deltas=np.ascontiguousarray(pair_deltas_np, dtype=np.float64),
            )
        )

    out_mode_indices = np.fromiter(
        (
            index_vswf(int(l_i), int(m_i), int(tau_i), full_order)
            for tau_i, l_i, m_i, _idx in iter_modes(lmax_out)
        ),
        dtype=np.int32,
        count=nmodes_out,
    )
    in_mode_indices = np.fromiter(
        (
            index_vswf(int(l_i), int(m_i), int(tau_i), full_order)
            for tau_i, l_i, m_i, _idx in iter_modes(lmax_in)
        ),
        dtype=np.int32,
        count=nmodes_in,
    )
    (
        compact_re_ab_raw,
        compact_im_ab_raw,
        pair_offset_raw,
        pair_pmin_raw,
        pair_pcount_raw,
        mode_m_out_raw,
        mode_m_in_raw,
    ) = _leaf_rect_pair_tables_and_ab(
        full_order=int(full_order),
        out_mode_indices=out_mode_indices,
        in_mode_indices=in_mode_indices,
        out_dtype=np.dtype(np.complex128),
    )
    plm_coeffs_raw = _translation_plm_coeff_table(full_order, dtype=np.float64).reshape(-1)

    # Leaf aggregation/receive only needs regular-wave Bessel values between a
    # particle and its own leaf center.  Do not inherit a much longer exact-near
    # Hankel range, because doing so would duplicate a large, unused J table on
    # host and device.  Keep one guard interval beyond the largest stored delta,
    # matching RadialLUT's linear-interpolation contract.
    required_grid_size = max(
        2,
        int(np.ceil(max_pair_radius / float(radial_lut.dr))) + 2,
    )
    if int(radial_lut.lmax) < full_order or int(radial_lut.j.shape[1]) < required_grid_size:
        radial_lut_full = RadialLUT(
            lmax=full_order,
            k=float(k),
            r_max=float(max_pair_radius),
            dr=float(radial_lut.dr),
            dtype=out_dtype,
        )
    else:
        radial_lut_full = radial_lut
    n_grid = min(int(radial_lut_full.j.shape[1]), int(required_grid_size))
    # Leaf aggregation and receive feed the sampled hierarchy, whose precision
    # remains complex128 even when the public/exact-near dtype is complex64.
    re_j, im_j = _split_complex_table_transposed(
        np.asarray(radial_lut_full.j)[: 2 * full_order + 1, :n_grid],
        real_dtype=np.float64,
    )
    tables = CuPyHostLeafTranslationTablesData(
        full_order=int(full_order),
        nmodes_in=int(nmodes_in),
        nmodes_out=int(nmodes_out),
        nmodes_full=int(nmodes_full),
        out_mode_indices=np.ascontiguousarray(out_mode_indices, dtype=np.int32),
        in_mode_indices=np.ascontiguousarray(in_mode_indices, dtype=np.int32),
        mode_m_out=np.ascontiguousarray(mode_m_out_raw, dtype=np.int32),
        mode_m_in=np.ascontiguousarray(mode_m_in_raw, dtype=np.int32),
        pair_offset=np.ascontiguousarray(pair_offset_raw, dtype=np.int32),
        pair_pmin=np.ascontiguousarray(pair_pmin_raw, dtype=np.int32),
        pair_pcount=np.ascontiguousarray(pair_pcount_raw, dtype=np.int32),
        plm_coeffs=np.ascontiguousarray(plm_coeffs_raw, dtype=np.float64),
        compact_re_ab=np.ascontiguousarray(compact_re_ab_raw, dtype=np.float64),
        compact_im_ab=np.ascontiguousarray(compact_im_ab_raw, dtype=np.float64),
        re_j=re_j,
        im_j=im_j,
        inv_dr=float(radial_lut_full._inv_dr),
        last_index=int(n_grid - 1),
    )
    return tuple(groups), tables


def _build_host_single_level(
    single: MLFMMSingleLevelOperators,
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT,
    out_dtype: np.dtype,
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
) -> CuPyHostSingleLevelData:
    """Build compact host payload for one single-level sampled-far stage.

    Dense leaf aggregation is preserved only for explicit validation/debug
    uploads; the production path stores the compact on-the-fly leaf payload.
    """

    if str(leaf_apply_mode) == "on_the_fly":
        leaf_groups_otf, leaf_tables = _build_host_leaf_otf_payload(
            lmax=int(lmax),
            box_order=int(single.box_order),
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            partition=partition,
            radial_lut=radial_lut,
            out_dtype=np.dtype(out_dtype),
        )
        aggregation_payload: tuple[np.ndarray, ...] | None = None
    else:
        leaf_groups_otf = None
        leaf_tables = None
        aggregation_payload = tuple(
            np.ascontiguousarray(np.asarray(block, dtype=np.complex128))
            for block in single.aggregation
        )
    return CuPyHostSingleLevelData(
        box_order=int(single.box_order),
        translator_order=int(single.translator_order),
        grid_order=int(single.grid_order),
        directional=_copy_directional_host(single.directional),
        aggregation=aggregation_payload,
        far_offset_batches=_copy_batches_host(single.far_offset_batches),
        offset_diagonals={
            key: np.ascontiguousarray(np.asarray(values, dtype=np.complex128).reshape(-1))
            for key, values in single.offset_diagonals.items()
        },
        leaf_groups_otf=leaf_groups_otf,
        leaf_translation_tables=leaf_tables,
        leaf_apply_mode=leaf_apply_mode,
    )


def _build_host_multilevel(
    multilevel: MLFMMMultilevelOperators,
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    partition: MLFMMPartition,
    radial_lut: RadialLUT,
    out_dtype: np.dtype,
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
) -> CuPyHostMultilevelData:
    """Build compact host payload for one multilevel sampled-far stage.

    Dense leaf aggregation is preserved only for explicit validation/debug
    uploads; the production path stores the compact on-the-fly leaf payload.
    """

    levels = tuple(
        CuPyHostLevelData(
            level=int(level.level),
            n_boxes=int(np.asarray(level.coords).shape[0]),
            box_order=int(level.box_order),
            translator_order=int(level.translator_order),
            grid_order=int(level.grid_order),
            directional=_copy_directional_host(level.directional),
            far_offset_batches=_copy_batches_host(level.far_offset_batches),
            offset_diagonals={
                key: np.ascontiguousarray(np.asarray(values, dtype=np.complex128).reshape(-1))
                for key, values in level.offset_diagonals.items()
            },
        )
        for level in multilevel.levels
    )
    transfers = tuple(
        CuPyHostTransferData(
            child_level=int(transfer.child_level),
            parent_level=int(transfer.parent_level),
            interpolation=transfer.interpolation,
            batches_by_shift=_copy_batches_host(transfer.batches_by_shift),
            phase_up_by_shift={
                key: np.ascontiguousarray(np.asarray(values, dtype=np.complex128).reshape(-1))
                for key, values in transfer.phase_up_by_shift.items()
            },
            phase_down_by_shift={
                key: np.ascontiguousarray(np.asarray(values, dtype=np.complex128).reshape(-1))
                for key, values in transfer.phase_down_by_shift.items()
            },
        )
        for transfer in multilevel.transfers
    )
    if str(leaf_apply_mode) == "on_the_fly":
        leaf_groups_otf, leaf_tables = _build_host_leaf_otf_payload(
            lmax=int(lmax),
            box_order=int(levels[int(multilevel.leaf_level)].box_order),
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            partition=partition,
            radial_lut=radial_lut,
            out_dtype=np.dtype(out_dtype),
        )
        aggregation_payload: tuple[np.ndarray, ...] | None = None
    else:
        leaf_groups_otf = None
        leaf_tables = None
        aggregation_payload = tuple(
            np.ascontiguousarray(np.asarray(block, dtype=np.complex128))
            for block in multilevel.aggregation
        )
    return CuPyHostMultilevelData(
        levels=levels,
        transfers=transfers,
        leaf_level=int(multilevel.leaf_level),
        hf_start_level=int(multilevel.hf_start_level),
        hf_end_level=int(multilevel.hf_end_level),
        aggregation=aggregation_payload,
        leaf_groups_otf=leaf_groups_otf,
        leaf_translation_tables=leaf_tables,
        leaf_apply_mode=leaf_apply_mode,
    )


def _resolve_host_cache_policy(
    policy: CuPyMLFMMHostCachePolicy | None,
) -> CuPyMLFMMHostCachePolicy:
    resolved = CuPyMLFMMHostCachePolicy() if policy is None else policy
    mode = str(resolved.leaf_apply_mode)
    if mode not in {"dense", "on_the_fly"}:
        raise ValueError(
            f"Unsupported CuPy MLFMM leaf_apply_mode={resolved.leaf_apply_mode!r}. "
            "Use 'dense' or 'on_the_fly'."
        )
    retention = str(resolved.host_cache_retention)
    if retention not in {"full", "summary", "none"}:
        raise ValueError(
            "Unsupported CuPy MLFMM host_cache_retention="
            f"{resolved.host_cache_retention!r}. Use 'full', 'summary', or 'none'."
        )
    chunk_leaves = resolved.leaf_otf_chunk_leaves
    if chunk_leaves is not None and int(chunk_leaves) <= 0:
        raise ValueError(
            "CuPy MLFMM leaf_otf_chunk_leaves must be positive when set. "
            f"Got {resolved.leaf_otf_chunk_leaves!r}."
        )
    leaf_otf_bytes_budget = resolved.leaf_otf_bytes_budget
    if leaf_otf_bytes_budget is not None and int(leaf_otf_bytes_budget) <= 0:
        raise ValueError(
            "CuPy MLFMM leaf_otf_bytes_budget must be positive when set. "
            f"Got {resolved.leaf_otf_bytes_budget!r}."
        )
    streamed_far_chunk_bytes_budget = resolved.streamed_far_chunk_bytes_budget
    if streamed_far_chunk_bytes_budget is not None and int(streamed_far_chunk_bytes_budget) <= 0:
        raise ValueError(
            "CuPy MLFMM streamed_far_chunk_bytes_budget must be positive when set. "
            f"Got {resolved.streamed_far_chunk_bytes_budget!r}."
        )
    near_adjoint_cache_bytes_budget = resolved.near_adjoint_cache_bytes_budget
    if near_adjoint_cache_bytes_budget is not None and int(near_adjoint_cache_bytes_budget) < 0:
        raise ValueError(
            "CuPy MLFMM near_adjoint_cache_bytes_budget must be non-negative when set. "
            f"Got {resolved.near_adjoint_cache_bytes_budget!r}."
        )
    return replace(
        resolved,
        host_cache_retention=cast(CuPyMLFMMHostCacheRetention, retention),
        leaf_otf_chunk_leaves=(None if chunk_leaves is None else int(chunk_leaves)),
        leaf_otf_bytes_budget=(
            None if leaf_otf_bytes_budget is None else int(leaf_otf_bytes_budget)
        ),
        streamed_far_chunk_bytes_budget=(
            None
            if streamed_far_chunk_bytes_budget is None
            else int(streamed_far_chunk_bytes_budget)
        ),
        near_adjoint_cache_bytes_budget=(
            None
            if near_adjoint_cache_bytes_budget is None
            else int(near_adjoint_cache_bytes_budget)
        ),
    )


def _estimate_numpy_payload_bytes(obj: object) -> int:
    """Estimate bytes from NumPy payloads stored inside nested dataclasses/containers."""

    seen: set[int] = set()

    def _walk(value: object) -> int:
        if isinstance(value, np.ndarray):
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            return int(value.nbytes)
        if is_dataclass(value):
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            return int(sum(_walk(getattr(value, f.name)) for f in fields(value)))
        if isinstance(value, dict):
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            total = 0
            for key, item in value.items():
                total += _walk(key)
                total += _walk(item)
            return int(total)
        if isinstance(value, (tuple, list)):
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            return int(sum(_walk(item) for item in value))
        return 0

    return int(_walk(obj))


def _runtime_host_cache_summary(
    cache: CuPyMLFMMHostCacheData,
    *,
    policy: CuPyMLFMMHostCachePolicy,
) -> dict[str, object]:
    """Build a compact diagnostic summary before optional host-cache detachment."""

    return {
        "retention": str(policy.host_cache_retention),
        "retained": str(policy.host_cache_retention) == "full",
        "leaf_apply_mode": str(policy.leaf_apply_mode),
        "leaf_otf_chunk_leaves": (
            None if policy.leaf_otf_chunk_leaves is None else int(policy.leaf_otf_chunk_leaves)
        ),
        "leaf_otf_bytes_budget": (
            None if policy.leaf_otf_bytes_budget is None else int(policy.leaf_otf_bytes_budget)
        ),
        "streamed_far_chunk_bytes_budget": (
            None
            if policy.streamed_far_chunk_bytes_budget is None
            else int(policy.streamed_far_chunk_bytes_budget)
        ),
        "near_adjoint_cache_bytes_budget": (
            None
            if policy.near_adjoint_cache_bytes_budget is None
            else int(policy.near_adjoint_cache_bytes_budget)
        ),
        "collect_stream_stats": bool(policy.collect_stream_stats),
        "stage": str(cache.stage),
        "near_static_tables_cached": bool(cache.near_plm_coeffs is not None),
        "numpy_payload_bytes_estimate": int(_estimate_numpy_payload_bytes(cache)),
        "plan_summary": None if cache.plan_summary is None else dict(cache.plan_summary),
    }


def _build_mlfmm_cupy_host_cache(
    coupling: MLFMMCouplingOperator,
    *,
    host_cache_policy: CuPyMLFMMHostCachePolicy | None = None,
) -> CuPyMLFMMHostCacheData:
    """Build a compact host-only MLFMM payload from CPU reference operators."""

    policy = _resolve_host_cache_policy(host_cache_policy)
    near_dtype = np.dtype(coupling.near_dtype)
    real_dtype: type[np.floating[Any]] = (
        np.float32 if near_dtype == np.dtype(np.complex64) else np.float64
    )
    partition = coupling.resolved_plan.partition
    leaf_offsets, leaf_indices = _pack_index_lists(
        [leaf.particle_indices for leaf in partition.leaves]
    )
    dst_leaf_indices, src_leaf_indices = _build_exact_near_leaf_pair_schedule(partition)
    near_lut_re, near_lut_im = _split_complex_table_transposed(
        np.asarray(coupling.radial_lut.h),
        real_dtype=real_dtype,
    )
    compact_re_ab = None
    compact_im_ab = None
    plm_coeffs = None
    mode_m = None
    pair_offset = None
    pair_pmin = None
    pair_pcount = None
    plan = coupling.resolved_plan
    plan_summary: dict[str, int | float | str] = {
        "stage": str(plan.stage),
        "selected_depth": int(plan.selected_depth),
        "depth_from_occupancy": int(plan.depth_from_occupancy),
        "depth_from_size_floor": int(plan.depth_from_size_floor),
        "occupied_leaf_count": int(plan.occupied_leaf_count),
        "max_particles_per_leaf": int(plan.max_particles_per_leaf),
        "root_side_length": float(plan.root_side_length),
        "leaf_side_length": float(plan.leaf_side_length),
        "max_global_radius": float(plan.max_global_radius),
    }
    return CuPyMLFMMHostCacheData(
        lmax=int(coupling.lmax),
        k=float(coupling.k),
        stage=str(coupling.resolved_plan.stage),
        dtype=np.dtype(coupling.dtype),
        near_dtype=np.dtype(coupling.near_dtype),
        far_dtype=np.dtype(coupling.far_dtype),
        n_particles=int(np.asarray(coupling.positions).shape[0]),
        leaf_particle_offsets=np.ascontiguousarray(leaf_offsets, dtype=np.int32),
        leaf_particle_indices=np.ascontiguousarray(leaf_indices, dtype=np.int32),
        near_positions_flat=np.ascontiguousarray(
            np.asarray(coupling.positions, dtype=real_dtype).reshape(-1)
        ),
        near_dst_leaf_indices=np.ascontiguousarray(dst_leaf_indices, dtype=np.int32),
        near_src_leaf_indices=np.ascontiguousarray(src_leaf_indices, dtype=np.int32),
        near_lut_re=near_lut_re,
        near_lut_im=near_lut_im,
        near_inv_dr=float(coupling.radial_lut._inv_dr),
        near_last_index=int(coupling.radial_lut._last_index),
        near_plm_coeffs=(
            np.ascontiguousarray(plm_coeffs, dtype=real_dtype) if plm_coeffs is not None else None
        ),
        near_compact_re_ab=(
            np.ascontiguousarray(compact_re_ab, dtype=real_dtype)
            if compact_re_ab is not None
            else None
        ),
        near_compact_im_ab=(
            np.ascontiguousarray(compact_im_ab, dtype=real_dtype)
            if compact_im_ab is not None
            else None
        ),
        near_mode_m=(np.ascontiguousarray(mode_m, dtype=np.int32) if mode_m is not None else None),
        near_pair_offset=(
            np.ascontiguousarray(pair_offset.reshape(-1), dtype=np.int32)
            if pair_offset is not None
            else None
        ),
        near_pair_pmin=(
            np.ascontiguousarray(pair_pmin.reshape(-1), dtype=np.int32)
            if pair_pmin is not None
            else None
        ),
        near_pair_pcount=(
            np.ascontiguousarray(pair_pcount.reshape(-1), dtype=np.int32)
            if pair_pcount is not None
            else None
        ),
        single_level=(
            _build_host_single_level(
                coupling.single_level,
                lmax=int(coupling.lmax),
                k=float(coupling.k),
                positions=np.asarray(coupling.positions, dtype=float),
                partition=partition,
                radial_lut=coupling.radial_lut,
                out_dtype=np.dtype(coupling.far_dtype),
                leaf_apply_mode=policy.leaf_apply_mode,
            )
            if coupling.single_level is not None
            else None
        ),
        multilevel=(
            _build_host_multilevel(
                coupling.multilevel,
                lmax=int(coupling.lmax),
                k=float(coupling.k),
                positions=np.asarray(coupling.positions, dtype=float),
                partition=partition,
                radial_lut=coupling.radial_lut,
                out_dtype=np.dtype(coupling.far_dtype),
                leaf_apply_mode=policy.leaf_apply_mode,
            )
            if coupling.multilevel is not None
            else None
        ),
        plan_summary=plan_summary,
    )


def _upload_leaf_apply_groups_dense(
    *,
    aggregation: tuple[np.ndarray, ...],
    partition: CuPyMLFMMPartitionData,
    cupy: Any,
    name: str,
) -> tuple[int, tuple[CuPyLeafApplyGroupData, ...]]:
    """Upload grouped leaf operators with uniform occupancy for batched GEMM."""

    n_leaves = len(aggregation)
    offsets = partition.leaf_particle_offsets_host
    flat_indices = partition.leaf_particle_indices_host
    if int(offsets.size) != n_leaves + 1:
        raise ValueError(
            f"{name} partition offsets size mismatch: {int(offsets.size)} vs {n_leaves + 1}."
        )
    if n_leaves == 0:
        return 0, tuple()

    box_nm = int(np.asarray(aggregation[0]).shape[0])
    leaf_ids_by_occupancy: dict[int, list[int]] = {}
    nmodes_ref: int | None = None
    for leaf_id in range(n_leaves):
        q = int(offsets[leaf_id + 1] - offsets[leaf_id])
        if q <= 0:
            raise ValueError(f"{name}[{leaf_id}] has non-positive occupancy {q}.")
        agg_leaf = np.asarray(aggregation[leaf_id], dtype=np.complex128)
        if agg_leaf.ndim != 2:
            raise ValueError(f"{name}[{leaf_id}] aggregation must be a 2D matrix.")
        if int(agg_leaf.shape[0]) != box_nm:
            raise ValueError(
                f"{name}[{leaf_id}] box-row mismatch: {int(agg_leaf.shape[0])} vs {box_nm}."
            )
        if int(agg_leaf.shape[1]) % q != 0:
            raise ValueError(
                f"{name}[{leaf_id}] aggregation columns {int(agg_leaf.shape[1])} not divisible by occupancy {q}."
            )
        nmodes_leaf = int(agg_leaf.shape[1] // q)
        if nmodes_ref is None:
            nmodes_ref = nmodes_leaf
        elif nmodes_leaf != nmodes_ref:
            raise ValueError(
                f"{name} inconsistent nmodes across leaves: {nmodes_leaf} vs {nmodes_ref}."
            )
        leaf_ids_by_occupancy.setdefault(q, []).append(leaf_id)

    if nmodes_ref is None:
        return box_nm, tuple()

    grouped: list[CuPyLeafApplyGroupData] = []
    for occupancy in sorted(leaf_ids_by_occupancy):
        leaf_ids_np = np.asarray(leaf_ids_by_occupancy[occupancy], dtype=np.int32)
        n_group = int(leaf_ids_np.size)
        part_idx_np = np.empty((n_group, occupancy), dtype=np.int32)
        agg_group = np.empty((n_group, box_nm, occupancy * nmodes_ref), dtype=np.complex128)
        for local_idx, leaf_id in enumerate(leaf_ids_np.tolist()):
            start = int(offsets[leaf_id])
            end = int(offsets[leaf_id + 1])
            part_idx_np[local_idx] = flat_indices[start:end]
            agg_group[local_idx] = np.asarray(aggregation[leaf_id], dtype=np.complex128)
        grouped.append(
            CuPyLeafApplyGroupData(
                occupancy=occupancy,
                nmodes=nmodes_ref,
                leaf_ids=cupy.asarray(leaf_ids_np, dtype=cupy.int32),
                particle_indices=cupy.asarray(part_idx_np, dtype=cupy.int32),
                aggregation=cupy.asarray(
                    np.ascontiguousarray(agg_group),
                    dtype=cupy.complex128,
                    blocking=True,
                ),
                pair_deltas=None,
            )
        )
    return box_nm, tuple(grouped)


def _upload_leaf_translation_tables(
    tables: CuPyHostLeafTranslationTablesData,
    *,
    cupy: Any,
) -> CuPyLeafTranslationTablesData:
    """Upload compact on-the-fly leaf translation ingredients to device memory."""

    return CuPyLeafTranslationTablesData(
        full_order=int(tables.full_order),
        nmodes_in=int(tables.nmodes_in),
        nmodes_out=int(tables.nmodes_out),
        nmodes_full=int(tables.nmodes_full),
        out_mode_indices=cupy.asarray(
            np.ascontiguousarray(tables.out_mode_indices, dtype=np.int32), dtype=cupy.int32
        ),
        in_mode_indices=cupy.asarray(
            np.ascontiguousarray(tables.in_mode_indices, dtype=np.int32), dtype=cupy.int32
        ),
        mode_m_out=cupy.asarray(
            np.ascontiguousarray(tables.mode_m_out, dtype=np.int32), dtype=cupy.int32
        ),
        mode_m_in=cupy.asarray(
            np.ascontiguousarray(tables.mode_m_in, dtype=np.int32), dtype=cupy.int32
        ),
        pair_offset=cupy.asarray(
            np.ascontiguousarray(tables.pair_offset, dtype=np.int32), dtype=cupy.int32
        ),
        pair_pmin=cupy.asarray(
            np.ascontiguousarray(tables.pair_pmin, dtype=np.int32), dtype=cupy.int32
        ),
        pair_pcount=cupy.asarray(
            np.ascontiguousarray(tables.pair_pcount, dtype=np.int32), dtype=cupy.int32
        ),
        plm_coeffs=cupy.asarray(
            np.ascontiguousarray(tables.plm_coeffs, dtype=np.float64), dtype=cupy.float64
        ),
        compact_re_ab=cupy.asarray(
            np.ascontiguousarray(tables.compact_re_ab, dtype=np.float64), dtype=cupy.float64
        ),
        compact_im_ab=cupy.asarray(
            np.ascontiguousarray(tables.compact_im_ab, dtype=np.float64), dtype=cupy.float64
        ),
        re_j=cupy.asarray(np.ascontiguousarray(tables.re_j, dtype=np.float64), dtype=cupy.float64),
        im_j=cupy.asarray(np.ascontiguousarray(tables.im_j, dtype=np.float64), dtype=cupy.float64),
        inv_dr=float(tables.inv_dr),
        last_index=int(tables.last_index),
    )


def _upload_leaf_apply_groups_otf(
    *,
    groups: tuple[CuPyHostLeafOnTheFlyGroupData, ...],
    n_leaves: int,
    box_nm: int,
    cupy: Any,
    name: str,
) -> tuple[int, tuple[CuPyLeafApplyGroupData, ...]]:
    """Upload grouped on-the-fly leaf schedules with compact pair-delta tables."""

    if n_leaves < 0:
        raise ValueError(f"{name} n_leaves must be non-negative, got {n_leaves}.")
    if box_nm <= 0 and n_leaves > 0:
        raise ValueError(f"{name} box_nm must be positive when leaves are present.")
    uploaded: list[CuPyLeafApplyGroupData] = []
    seen_leaf_ids: set[int] = set()
    for group in groups:
        occupancy = int(group.occupancy)
        nmodes = int(group.nmodes)
        if occupancy <= 0 or nmodes <= 0:
            raise ValueError(
                f"{name} invalid on-the-fly group metadata occupancy={occupancy}, nmodes={nmodes}."
            )
        leaf_ids_np = np.ascontiguousarray(np.asarray(group.leaf_ids, dtype=np.int32).reshape(-1))
        part_idx_np = np.ascontiguousarray(
            np.asarray(group.particle_indices, dtype=np.int32).reshape(-1, occupancy)
        )
        n_group = int(leaf_ids_np.size)
        if int(part_idx_np.shape[0]) != n_group:
            raise ValueError(
                f"{name} on-the-fly particle index row count mismatch: {int(part_idx_np.shape[0])} vs {n_group}."
            )
        deltas_np = np.ascontiguousarray(
            np.asarray(group.pair_deltas, dtype=np.float64).reshape(-1, 3), dtype=np.float64
        )
        expected_pairs = n_group * occupancy
        if int(deltas_np.shape[0]) != expected_pairs:
            raise ValueError(
                f"{name} on-the-fly pair-delta size mismatch: {int(deltas_np.shape[0])} vs {expected_pairs}."
            )
        for leaf_id in leaf_ids_np.tolist():
            leaf_id_i = int(leaf_id)
            if leaf_id_i < 0 or leaf_id_i >= int(n_leaves):
                raise ValueError(
                    f"{name} on-the-fly leaf id {leaf_id_i} out of range [0, {int(n_leaves)})."
                )
            if leaf_id_i in seen_leaf_ids:
                raise ValueError(
                    f"{name} on-the-fly leaf id {leaf_id_i} appears in multiple groups."
                )
            seen_leaf_ids.add(leaf_id_i)
        uploaded.append(
            CuPyLeafApplyGroupData(
                occupancy=occupancy,
                nmodes=nmodes,
                leaf_ids=cupy.asarray(leaf_ids_np, dtype=cupy.int32),
                particle_indices=cupy.asarray(part_idx_np, dtype=cupy.int32),
                aggregation=None,
                pair_deltas=cupy.asarray(deltas_np, dtype=cupy.float64),
            )
        )
    if len(seen_leaf_ids) != int(n_leaves):
        raise ValueError(
            f"{name} on-the-fly groups do not cover all leaves exactly once "
            f"(covered={len(seen_leaf_ids)}, expected={int(n_leaves)})."
        )
    return int(box_nm), tuple(uploaded)


def _upload_single_level(
    single: MLFMMSingleLevelOperators | CuPyHostSingleLevelData,
    partition: CuPyMLFMMPartitionData,
    *,
    cupy: Any,
) -> CuPyMLFMMSingleLevelData:
    directional = _upload_directional_transforms(single.directional, cupy=cupy)
    ndir = int(directional.grid.n_directions)
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

    leaf_apply_mode = str(single.leaf_apply_mode) if hasattr(single, "leaf_apply_mode") else "dense"
    leaf_tables_dev: CuPyLeafTranslationTablesData | None = None
    if leaf_apply_mode == "on_the_fly":
        groups_host = getattr(single, "leaf_groups_otf", None)
        tables_host = getattr(single, "leaf_translation_tables", None)
        if groups_host is None or tables_host is None:
            raise ValueError(
                "single-level on-the-fly leaf apply mode requires host leaf groups and translation tables."
            )
        n_leaves = int(partition.leaf_particle_offsets_host.size - 1)
        box_nm = int(n_modes(int(single.box_order)))
        box_nm, leaf_groups = _upload_leaf_apply_groups_otf(
            groups=groups_host,
            n_leaves=n_leaves,
            box_nm=box_nm,
            cupy=cupy,
            name="single_level",
        )
        leaf_tables_dev = _upload_leaf_translation_tables(tables_host, cupy=cupy)
    else:
        if single.aggregation is None:
            raise ValueError("single-level dense leaf apply mode requires aggregation payload.")
        box_nm, leaf_groups = _upload_leaf_apply_groups_dense(
            aggregation=single.aggregation,
            partition=partition,
            cupy=cupy,
            name="single_level",
        )
    return CuPyMLFMMSingleLevelData(
        box_order=int(single.box_order),
        translator_order=int(single.translator_order),
        grid_order=int(single.grid_order),
        box_nm=box_nm,
        n_leaves=int(partition.leaf_particle_offsets_host.size - 1),
        directional=directional,
        leaf_groups=leaf_groups,
        far_offset_batches=_upload_offset_batches(
            single.far_offset_batches,
            cupy=cupy,
            name="single_level.far_offset_batches",
        ),
        offset_diagonals=offset_diagonals,
        leaf_apply_mode=cast(CuPyMLFMMLeafApplyMode, leaf_apply_mode),
        leaf_translation_tables=leaf_tables_dev,
    )


def _upload_level(
    level: MLFMMLevelOperators | CuPyHostLevelData, *, cupy: Any
) -> CuPyMLFMMLevelData:
    directional = _upload_directional_transforms(level.directional, cupy=cupy)
    ndir = int(directional.grid.n_directions)
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

    if hasattr(level, "n_boxes"):
        n_boxes = int(level.n_boxes)
    else:
        n_boxes = int(np.asarray(level.coords).shape[0])

    return CuPyMLFMMLevelData(
        level=int(level.level),
        n_boxes=n_boxes,
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
    transfer: MLFMMTransferOperators | CuPyHostTransferData,
    *,
    child_reflection_permutation: np.ndarray,
    parent_reflection_permutation: np.ndarray,
    cupy: Any,
    cupyx_sparse: Any,
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

    child_perm = _as_numpy_1d(
        child_reflection_permutation,
        dtype=np.int32,
        name=f"transfer[{transfer.child_level}->{transfer.parent_level}].child_reflection_permutation",
    )
    parent_perm = _as_numpy_1d(
        parent_reflection_permutation,
        dtype=np.int32,
        name=f"transfer[{transfer.child_level}->{transfer.parent_level}].parent_reflection_permutation",
    )
    child_inv = np.ascontiguousarray(np.argsort(child_perm), dtype=np.int32)
    parent_inv = np.ascontiguousarray(np.argsort(parent_perm), dtype=np.int32)

    interp_csr = transfer.interpolation.matrix.tocsr().astype(np.complex128)
    map_up_csr = interp_csr[parent_perm, :][:, child_inv].tocsr().astype(np.complex128)
    map_down_csr = interp_csr.T.tocsr()[child_perm, :][:, parent_inv].tocsr().astype(np.complex128)

    map_up = _upload_directional_map(
        map_up_csr,
        source_order=int(interp_csr.shape[1]),
        target_order=int(interp_csr.shape[0]),
        cupy=cupy,
        cupyx_sparse=cupyx_sparse,
    )
    map_down = _upload_directional_map(
        map_down_csr,
        source_order=int(map_down_csr.shape[1]),
        target_order=int(map_down_csr.shape[0]),
        cupy=cupy,
        cupyx_sparse=cupyx_sparse,
    )
    return CuPyMLFMMTransferData(
        child_level=int(transfer.child_level),
        parent_level=int(transfer.parent_level),
        map_up=map_up,
        map_down=map_down,
        batches_by_shift=_upload_offset_batches(
            transfer.batches_by_shift,
            cupy=cupy,
            name=f"transfer[{transfer.child_level}->{transfer.parent_level}].batches_by_shift",
        ),
        phase_up_by_shift=phase_up,
        phase_down_by_shift=phase_down,
    )


def _upload_multilevel(
    multilevel: MLFMMMultilevelOperators | CuPyHostMultilevelData,
    partition: CuPyMLFMMPartitionData,
    *,
    cupy: Any,
    cupyx_sparse: Any,
) -> CuPyMLFMMMultilevelData:
    leaf_apply_mode = (
        str(multilevel.leaf_apply_mode) if hasattr(multilevel, "leaf_apply_mode") else "dense"
    )
    leaf_tables_dev: CuPyLeafTranslationTablesData | None = None
    if leaf_apply_mode == "on_the_fly":
        groups_host = getattr(multilevel, "leaf_groups_otf", None)
        tables_host = getattr(multilevel, "leaf_translation_tables", None)
        if groups_host is None or tables_host is None:
            raise ValueError(
                "multilevel on-the-fly leaf apply mode requires host leaf groups and translation tables."
            )
        n_leaves = int(partition.leaf_particle_offsets_host.size - 1)
        leaf_level_idx = int(multilevel.leaf_level)
        if leaf_level_idx < 0 or leaf_level_idx >= len(multilevel.levels):
            raise ValueError(
                f"multilevel leaf_level index {leaf_level_idx} out of bounds for levels payload."
            )
        box_nm = int(n_modes(int(multilevel.levels[leaf_level_idx].box_order)))
        box_nm, leaf_groups = _upload_leaf_apply_groups_otf(
            groups=groups_host,
            n_leaves=n_leaves,
            box_nm=box_nm,
            cupy=cupy,
            name="multilevel",
        )
        leaf_tables_dev = _upload_leaf_translation_tables(tables_host, cupy=cupy)
    else:
        if multilevel.aggregation is None:
            raise ValueError("multilevel dense leaf apply mode requires aggregation payload.")
        box_nm, leaf_groups = _upload_leaf_apply_groups_dense(
            aggregation=multilevel.aggregation,
            partition=partition,
            cupy=cupy,
            name="multilevel",
        )
    cpu_level_by_index = {int(level.level): level for level in multilevel.levels}
    levels = tuple(_upload_level(level, cupy=cupy) for level in multilevel.levels)
    return CuPyMLFMMMultilevelData(
        levels=levels,
        transfers=tuple(
            _upload_transfer(
                transfer,
                child_reflection_permutation=np.asarray(
                    cpu_level_by_index[
                        int(transfer.child_level)
                    ].directional.grid.reflection_permutation,
                    dtype=np.int32,
                ),
                parent_reflection_permutation=np.asarray(
                    cpu_level_by_index[
                        int(transfer.parent_level)
                    ].directional.grid.reflection_permutation,
                    dtype=np.int32,
                ),
                cupy=cupy,
                cupyx_sparse=cupyx_sparse,
            )
            for transfer in multilevel.transfers
        ),
        leaf_level=int(multilevel.leaf_level),
        hf_start_level=int(multilevel.hf_start_level),
        hf_end_level=int(multilevel.hf_end_level),
        box_nm=box_nm,
        n_leaves=int(partition.leaf_particle_offsets_host.size - 1),
        leaf_groups=leaf_groups,
        leaf_apply_mode=cast(CuPyMLFMMLeafApplyMode, leaf_apply_mode),
        leaf_translation_tables=leaf_tables_dev,
    )


def _build_exact_near_leaf_pair_schedule(
    partition: MLFMMPartition,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed exact-near leaf-pair schedule from the resolved partition."""

    dst_leaf_indices: list[int] = []
    src_leaf_indices: list[int] = []
    for a, b in partition.leaf_near_pairs:
        ia = int(a)
        ib = int(b)
        dst_leaf_indices.append(ia)
        src_leaf_indices.append(ib)
        if ia != ib:
            dst_leaf_indices.append(ib)
            src_leaf_indices.append(ia)
    return np.asarray(dst_leaf_indices, dtype=np.int32), np.asarray(
        src_leaf_indices, dtype=np.int32
    )


def _resolve_exact_near_static_tables_from_host_cache(
    cache: CuPyMLFMMHostCacheData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return static exact-near tables, recomputing when cache policy omits them."""

    near_dtype = np.dtype(cache.near_dtype)
    if near_dtype == np.dtype(np.complex64):
        real_dtype: np.dtype[Any] = np.dtype(np.float32)
    elif near_dtype == np.dtype(np.complex128):
        real_dtype = np.dtype(np.float64)
    else:
        raise ValueError(
            "CuPy MLFMM near path supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    if (
        cache.near_plm_coeffs is not None
        and cache.near_compact_re_ab is not None
        and cache.near_compact_im_ab is not None
        and cache.near_mode_m is not None
        and cache.near_pair_offset is not None
        and cache.near_pair_pmin is not None
        and cache.near_pair_pcount is not None
    ):
        plm_coeffs = np.asarray(cache.near_plm_coeffs, dtype=real_dtype).reshape(-1)
        compact_re_ab = np.asarray(cache.near_compact_re_ab, dtype=real_dtype).reshape(-1)
        compact_im_ab = np.asarray(cache.near_compact_im_ab, dtype=real_dtype).reshape(-1)
        mode_m = np.asarray(cache.near_mode_m, dtype=np.int32).reshape(-1)
        pair_offset = np.asarray(cache.near_pair_offset, dtype=np.int32).reshape(-1)
        pair_pmin = np.asarray(cache.near_pair_pmin, dtype=np.int32).reshape(-1)
        pair_pcount = np.asarray(cache.near_pair_pcount, dtype=np.int32).reshape(-1)
        return (
            np.ascontiguousarray(plm_coeffs, dtype=real_dtype),
            np.ascontiguousarray(compact_re_ab, dtype=real_dtype),
            np.ascontiguousarray(compact_im_ab, dtype=real_dtype),
            np.ascontiguousarray(mode_m, dtype=np.int32),
            np.ascontiguousarray(pair_offset, dtype=np.int32),
            np.ascontiguousarray(pair_pmin, dtype=np.int32),
            np.ascontiguousarray(pair_pcount, dtype=np.int32),
        )

    # Static exact-near tables are intentionally omitted from host cache and
    # rebuilt on demand during upload.
    lmax = int(cache.lmax)
    compact_re_ab_raw, compact_im_ab_raw = _translation_ab5_compact_tables(
        lmax, dtype=np.complex128
    )
    plm_coeffs_raw = _translation_plm_coeff_table(lmax, dtype=np.float64).reshape(-1)
    mode_m = mode_m_table(lmax)
    pair_offset_raw, pair_pmin_raw, pair_pcount_raw = mode_pair_p_range_tables(lmax)
    return (
        np.ascontiguousarray(plm_coeffs_raw, dtype=real_dtype),
        np.ascontiguousarray(compact_re_ab_raw, dtype=real_dtype),
        np.ascontiguousarray(compact_im_ab_raw, dtype=real_dtype),
        np.ascontiguousarray(mode_m, dtype=np.int32),
        np.ascontiguousarray(pair_offset_raw.reshape(-1), dtype=np.int32),
        np.ascontiguousarray(pair_pmin_raw.reshape(-1), dtype=np.int32),
        np.ascontiguousarray(pair_pcount_raw.reshape(-1), dtype=np.int32),
    )


def _upload_exact_near_pair_data_from_host_cache(
    cache: CuPyMLFMMHostCacheData, *, cupy: Any
) -> CuPyMLFMMNearPairData:
    """Upload exact-near leaf schedule and translation payload from host cache."""

    near_dtype = np.dtype(cache.near_dtype)
    if near_dtype == np.dtype(np.complex64):
        cupy_real_dtype = cupy.float32
    elif near_dtype == np.dtype(np.complex128):
        cupy_real_dtype = cupy.float64
    else:
        raise ValueError(
            "CuPy MLFMM near path supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    (
        plm_coeffs,
        compact_re_ab,
        compact_im_ab,
        mode_m,
        pair_offset,
        pair_pmin,
        pair_pcount,
    ) = _resolve_exact_near_static_tables_from_host_cache(cache)
    return CuPyMLFMMNearPairData(
        near_dtype=near_dtype,
        positions=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.near_positions_flat).reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        leaf_particle_offsets=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.leaf_particle_offsets, dtype=np.int32)),
            dtype=cupy.int32,
            blocking=True,
        ),
        leaf_particle_indices=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.leaf_particle_indices, dtype=np.int32)),
            dtype=cupy.int32,
            blocking=True,
        ),
        dst_leaf_indices=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.near_dst_leaf_indices, dtype=np.int32)),
            dtype=cupy.int32,
            blocking=True,
        ),
        src_leaf_indices=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.near_src_leaf_indices, dtype=np.int32)),
            dtype=cupy.int32,
            blocking=True,
        ),
        lut_re=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.near_lut_re).reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        lut_im=cupy.asarray(
            np.ascontiguousarray(np.asarray(cache.near_lut_im).reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        inv_dr=float(cache.near_inv_dr),
        last_index=int(cache.near_last_index),
        plm_coeffs=cupy.asarray(
            np.ascontiguousarray(plm_coeffs.reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        compact_re_ab=cupy.asarray(
            np.ascontiguousarray(compact_re_ab.reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        compact_im_ab=cupy.asarray(
            np.ascontiguousarray(compact_im_ab.reshape(-1)),
            dtype=cupy_real_dtype,
            blocking=True,
        ),
        mode_m=cupy.asarray(
            np.ascontiguousarray(mode_m, dtype=np.int32),
            dtype=cupy.int32,
            blocking=True,
        ),
        pair_offset=cupy.asarray(
            np.ascontiguousarray(pair_offset, dtype=np.int32),
            dtype=cupy.int32,
            blocking=True,
        ),
        pair_pmin=cupy.asarray(
            np.ascontiguousarray(pair_pmin, dtype=np.int32),
            dtype=cupy.int32,
            blocking=True,
        ),
        pair_pcount=cupy.asarray(
            np.ascontiguousarray(pair_pcount, dtype=np.int32),
            dtype=cupy.int32,
            blocking=True,
        ),
    )


def _reshape_unknowns_to_particle_modes(
    x: Any,
    *,
    n_particles: int,
    nm: int,
    dtype: np.dtype,
    cupy: Any,
) -> tuple[Any, bool]:
    """Reshape `(n*nm,)` or `(n*nm, nrhs)` unknowns to `(n, nm, nrhs)`."""

    arr = coerce_array(x, dtype=np.dtype(dtype), prefer_cupy=True)
    cupy_dtype = _cupy_complex_dtype(np.dtype(dtype), cupy=cupy)
    if arr.ndim == 1:
        if int(arr.size) != int(n_particles * nm):
            raise ValueError(
                f"MLFMM CuPy apply expected vector size {n_particles * nm}, got {int(arr.size)}."
            )
        return cupy.asarray(arr, dtype=cupy_dtype).reshape(n_particles, nm, 1), True
    if arr.ndim == 2:
        if int(arr.shape[0]) != int(n_particles * nm):
            raise ValueError(
                "MLFMM CuPy apply expected 2D unknowns with first dimension "
                f"{n_particles * nm}, got {tuple(int(v) for v in arr.shape)}."
            )
        return cupy.asarray(arr, dtype=cupy_dtype).reshape(
            n_particles, nm, int(arr.shape[1])
        ), False
    raise ValueError(f"MLFMM CuPy apply expects 1D or 2D unknowns, got ndim={arr.ndim}.")


def _restore_unknown_shape(y: Any, *, squeezed: bool) -> Any:
    """Restore `(n, nm, nrhs)` output to original linear-system shape."""

    if squeezed:
        return y.reshape(-1)
    n_particles, nm, nrhs = (int(v) for v in y.shape)
    return y.reshape(n_particles * nm, nrhs)


@cache
def _exact_leaf_pairs_raw_kernel(lmax: int, near_dtype_name: str) -> Any:
    cupy, _ = import_cupy()
    lmax = int(lmax)
    near_dtype = np.dtype(near_dtype_name)
    if near_dtype == np.dtype(np.complex64):
        real_t = "float"
        complex_t = "complex<float>"
    elif near_dtype == np.dtype(np.complex128):
        real_t = "double"
        complex_t = "complex<double>"
    else:
        raise ValueError(
            "CuPy MLFMM near kernel supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    math = {
        "atan2": "atan2f" if real_t == "float" else "atan2",
        "floor": "floorf" if real_t == "float" else "floor",
        "max": "fmaxf" if real_t == "float" else "fmax",
        "sincos": "sincosf" if real_t == "float" else "sincos",
        "sqrt": "sqrtf" if real_t == "float" else "sqrt",
    }
    n_orders = 2 * lmax + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    #include <cupy/complex.cuh>
    __device__ {real_t} assoc_legendre_function(
        const int l,
        const int m,
        const {real_t}* ct_powers,
        const {real_t}* st_powers,
        const {real_t}* plm_coeffs
    ) {{
        {real_t} plm = ({real_t})0.0;
        const {real_t} st_pow = st_powers[m];
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    __device__ {real_t} hankel_lookup_linear(
        const int p,
        const {real_t} r,
        const {real_t}* table,
        const {real_t} inv_dr,
        const int last_index
    ) {{
        if (r <= ({real_t})0.0) {{
            return table[p];
        }}
        {real_t} t = r * inv_dr;
        int i0 = (int){math["floor"]}(t);
        {real_t} frac = t - ({real_t})i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = ({real_t})0.0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = ({real_t})1.0;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        return (({real_t})1.0 - frac) * table[base0] + frac * table[base1];
    }}

    extern "C" __global__ void mlfmm_exact_leaf_pairs(
        const int n_leaf_pairs,
        const int nmodes,
        const int nrhs,
        const {real_t}* positions,
        const int skip_identity,
        const int* dst_leaf_indices,
        const int* src_leaf_indices,
        const int* leaf_particle_offsets,
        const int* leaf_particle_indices,
        const {real_t}* re_h,
        const {real_t}* im_h,
        const {real_t} inv_dr,
        const int last_index,
        const {real_t}* plm_coeffs,
        const {real_t}* re_ab,
        const {real_t}* im_ab,
        const int* mode_m,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        const {complex_t}* x,
        {complex_t}* y
    ) {{
        const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
        const bool active = (n1 < nmodes);
        __shared__ {real_t} re_h_shared[{n_orders}];
        __shared__ {real_t} im_h_shared[{n_orders}];
        __shared__ {real_t} p_pdm_shared[{n_p_pdm}];
        __shared__ {real_t} cos_mphi_shared[{n_phase}];
        __shared__ {real_t} sin_mphi_shared[{n_phase}];
        __shared__ {real_t} r_shared;
        __shared__ {real_t} ct_shared;
        __shared__ {real_t} st_shared;
        __shared__ {real_t} phi_shared;
        __shared__ int dst_leaf_shared;
        __shared__ int src_leaf_shared;
        __shared__ int dst_particle_shared;
        __shared__ int src_particle_shared;
        __shared__ {real_t} ct_pow_shared[{n_orders}];
        __shared__ {real_t} st_pow_shared[{n_orders}];

        const int m1 = active ? mode_m[n1] : 0;
        for (int rhs = blockIdx.z; rhs < nrhs; rhs += gridDim.z) {{
            for (int leaf_pair_idx = blockIdx.y; leaf_pair_idx < n_leaf_pairs; leaf_pair_idx += gridDim.y) {{
                if (threadIdx.x == 0) {{
                    dst_leaf_shared = dst_leaf_indices[leaf_pair_idx];
                    src_leaf_shared = src_leaf_indices[leaf_pair_idx];
                }}
                __syncthreads();

                const int dst_start = leaf_particle_offsets[dst_leaf_shared];
                const int dst_end = leaf_particle_offsets[dst_leaf_shared + 1];
                const int src_start = leaf_particle_offsets[src_leaf_shared];
                const int src_end = leaf_particle_offsets[src_leaf_shared + 1];

                for (int dst_ptr = dst_start; dst_ptr < dst_end; ++dst_ptr) {{
                    const int dst_particle = leaf_particle_indices[dst_ptr];
                    for (int src_ptr = src_start; src_ptr < src_end; ++src_ptr) {{
                        const int src_particle = leaf_particle_indices[src_ptr];
                        if (
                            skip_identity != 0
                            && dst_leaf_shared == src_leaf_shared
                            && dst_particle == src_particle
                        ) {{
                            continue;
                        }}

                        if (threadIdx.x == 0) {{
                            dst_particle_shared = dst_particle;
                            src_particle_shared = src_particle;
                            const {real_t} x21 =
                                positions[3 * dst_particle_shared] - positions[3 * src_particle_shared];
                            const {real_t} y21 =
                                positions[3 * dst_particle_shared + 1] - positions[3 * src_particle_shared + 1];
                            const {real_t} z21 =
                                positions[3 * dst_particle_shared + 2] - positions[3 * src_particle_shared + 2];
                            r_shared = {math["sqrt"]}(x21 * x21 + y21 * y21 + z21 * z21);
                            ct_shared = z21 / r_shared;
                            st_shared = {math["sqrt"]}(
                                {math["max"]}(({real_t})0.0, ({real_t})1.0 - ct_shared * ct_shared)
                            );
                            phi_shared = {math["atan2"]}(y21, x21);
                            ct_pow_shared[0] = ({real_t})1.0;
                            st_pow_shared[0] = ({real_t})1.0;
                            for (int p = 1; p < {n_orders}; ++p) {{
                                ct_pow_shared[p] = ct_pow_shared[p - 1] * ct_shared;
                                st_pow_shared[p] = st_pow_shared[p - 1] * st_shared;
                            }}
                        }}
                        __syncthreads();

                        for (int p = threadIdx.x; p < {n_orders}; p += blockDim.x) {{
                            re_h_shared[p] = hankel_lookup_linear(p, r_shared, re_h, inv_dr, last_index);
                            im_h_shared[p] = hankel_lookup_linear(p, r_shared, im_h, inv_dr, last_index);
                        }}
                        for (int table_idx = threadIdx.x; table_idx < {n_p_pdm};
                             table_idx += blockDim.x) {{
                            int p = 0;
                            while (table_idx >= (p + 1) * (p + 2) / 2) {{
                                ++p;
                            }}
                            const int absdm = table_idx - p * (p + 1) / 2;
                            p_pdm_shared[table_idx] = assoc_legendre_function(
                                p, absdm, ct_pow_shared, st_pow_shared, plm_coeffs
                            );
                        }}
                        for (int idx = threadIdx.x; idx < {n_phase}; idx += blockDim.x) {{
                            const int dm = idx - 2 * {lmax};
                            {math["sincos"]}(
                                ({real_t})dm * phi_shared,
                                &sin_mphi_shared[idx],
                                &cos_mphi_shared[idx]
                            );
                        }}
                        __syncthreads();

                        if (active) {{
                            {real_t} re_incr = ({real_t})0.0;
                            {real_t} im_incr = ({real_t})0.0;
                            for (int n2 = 0; n2 < nmodes; ++n2) {{
                                const long long x_idx =
                                    (((long long)src_particle_shared * nmodes + n2) * nrhs) + rhs;
                                const {complex_t} x_tmp = x[x_idx];
                                const {real_t} re_x_tmp = x_tmp.real();
                                const {real_t} im_x_tmp = x_tmp.imag();
                                const int delta_m = mode_m[n2] - m1;
                                const int phase_idx = delta_m + 2 * {lmax};
                                const int pair_table_idx = n1 * nmodes + n2;
                                const int base = pair_offset[pair_table_idx];
                                const int p_min = pair_pmin[pair_table_idx];
                                const int p_count = pair_pcount[pair_table_idx];
                                for (int ip = 0; ip < p_count; ++ip) {{
                                    const int p = p_min + ip;
                                    const int ab_idx = base + ip;
                                    const {real_t} plm =
                                        p_pdm_shared[p * (p + 1) / 2 + abs(delta_m)];
                                    const {real_t} re_abp = re_ab[ab_idx] * plm;
                                    const {real_t} im_abp = im_ab[ab_idx] * plm;
                                    const {real_t} re_abph =
                                        re_abp * re_h_shared[p] - im_abp * im_h_shared[p];
                                    const {real_t} im_abph =
                                        re_abp * im_h_shared[p] + im_abp * re_h_shared[p];
                                    const {real_t} re_phase =
                                        re_abph * cos_mphi_shared[phase_idx] - im_abph * sin_mphi_shared[phase_idx];
                                    const {real_t} im_phase =
                                        re_abph * sin_mphi_shared[phase_idx] + im_abph * cos_mphi_shared[phase_idx];
                                    re_incr += re_phase * re_x_tmp - im_phase * im_x_tmp;
                                    im_incr += re_phase * im_x_tmp + im_phase * re_x_tmp;
                                }}
                            }}

                            const long long y_idx =
                                (((long long)dst_particle_shared * nmodes + n1) * nrhs) + rhs;
                            {real_t}* y_ptr = reinterpret_cast<{real_t}*>(&y[y_idx]);
                            atomicAdd(y_ptr + 0, re_incr);
                            atomicAdd(y_ptr + 1, im_incr);
                        }}
                        __syncthreads();
                    }}
                }}
            }}
        }}
    }}
    """
    return cupy.RawKernel(source, "mlfmm_exact_leaf_pairs")


@cache
def _add_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void add_unique_complex128(
        const long long n_pairs,
        const long long width,
        const int* dst,
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
            out[out_idx] += values[i];
        }
    }
    """
    return cupy.RawKernel(source, "add_unique_complex128")


@cache
def _weighted_add_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void weighted_add_unique_complex128(
        const long long n_pairs,
        const long long n_dirs,
        const long long nrhs,
        const int* dst,
        const complex<double>* values,
        const complex<double>* weights,
        complex<double>* out
    ) {
        const long long width = 4LL * n_dirs * nrhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * width;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / width;
            const long long lane = i - pair_idx * width;
            const long long chan_stride = n_dirs * nrhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dir = rem / nrhs;
            const long long rhs = rem - dir * nrhs;
            const long long out_row = dst[pair_idx];
            const long long out_idx = ((out_row * 4LL + chan) * n_dirs + dir) * nrhs + rhs;
            out[out_idx] += values[i] * weights[dir];
        }
    }
    """
    return cupy.RawKernel(source, "weighted_add_unique_complex128")


@cache
def _weighted_gather_add_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void weighted_gather_add_unique_complex128(
        const long long n_pairs,
        const long long n_dirs,
        const long long nrhs,
        const int* src,
        const int* dst,
        const complex<double>* source_values,
        const complex<double>* weights,
        complex<double>* out
    ) {
        const long long width = 4LL * n_dirs * nrhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * width;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / width;
            const long long lane = i - pair_idx * width;
            const long long chan_stride = n_dirs * nrhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dir = rem / nrhs;
            const long long rhs = rem - dir * nrhs;
            const long long src_row = src[pair_idx];
            const long long dst_row = dst[pair_idx];
            const long long src_idx = ((src_row * 4LL + chan) * n_dirs + dir) * nrhs + rhs;
            const long long out_idx = ((dst_row * 4LL + chan) * n_dirs + dir) * nrhs + rhs;
            out[out_idx] += source_values[src_idx] * weights[dir];
        }
    }
    """
    return cupy.RawKernel(source, "weighted_gather_add_unique_complex128")


def _add_at_complex128(
    target: Any,
    indices: Any,
    values: Any,
    *,
    cupy: Any,
) -> None:
    """Apply unique-index indexed accumulation for complex128 batches on device."""

    idx = cupy.asarray(indices, dtype=cupy.int32).reshape(-1)
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

    kernel = _add_unique_complex128_raw_kernel()
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


def _weighted_add_at_complex128(
    target: Any,
    indices: Any,
    values: Any,
    weights: Any,
    *,
    cupy: Any,
) -> None:
    """Apply weighted `add.at` accumulation for `(pair, 4, ndir, nrhs)` batches."""

    idx = cupy.asarray(indices, dtype=cupy.int32).reshape(-1)
    if int(idx.size) == 0:
        return
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    vals = cupy.asarray(values, dtype=cupy.complex128)
    w = cupy.asarray(weights, dtype=cupy.complex128).reshape(-1)
    if vals.ndim != 4 or int(vals.shape[0]) != int(idx.size):
        raise ValueError(
            "Weighted add-at expects values with shape (n_pairs, 4, ndir, nrhs). "
            f"Got {tuple(int(v) for v in vals.shape)} for n_pairs={int(idx.size)}."
        )
    if int(vals.shape[1]) != 4:
        raise ValueError(
            f"Weighted add-at expects 4 directional channels, got {int(vals.shape[1])}."
        )
    if int(vals.shape[2]) != int(w.size):
        raise ValueError(
            "Weighted add-at direction count mismatch between values and weights: "
            f"{int(vals.shape[2])} vs {int(w.size)}."
        )
    if tuple(int(v) for v in tgt.shape[1:]) != (
        int(vals.shape[1]),
        int(vals.shape[2]),
        int(vals.shape[3]),
    ):
        raise ValueError(
            "Weighted add-at target/value trailing-shape mismatch: "
            f"target={tuple(int(v) for v in tgt.shape[1:])}, "
            f"values={tuple(int(v) for v in vals.shape[1:])}."
        )
    if int(cupy.max(idx)) >= int(tgt.shape[0]) or int(cupy.min(idx)) < 0:
        raise ValueError("Weighted add-at index out of bounds for target tensor.")
    vals_flat = cupy.ascontiguousarray(vals.reshape(int(idx.size), -1))
    out_flat = tgt.reshape(-1)
    threads = 256
    total = int(vals_flat.size)
    blocks = max(1, (total + threads - 1) // threads)
    kernel = _weighted_add_unique_complex128_raw_kernel()
    kernel(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(idx.size)),
            np.int64(int(w.size)),
            np.int64(int(vals.shape[3])),
            idx,
            vals_flat,
            w,
            out_flat,
        ),
    )


def _weighted_gather_add_complex128(
    target: Any,
    dst_indices: Any,
    source_values: Any,
    src_indices: Any,
    weights: Any,
    *,
    cupy: Any,
) -> None:
    """Gather directional rows from `source_values`, apply directional weights, and add into `target`."""

    src = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    dst = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(src.size) == 0:
        return
    if int(src.size) != int(dst.size):
        raise ValueError(
            "Weighted gather-add source/destination index count mismatch: "
            f"{int(src.size)} vs {int(dst.size)}."
        )
    src_arr = cupy.asarray(source_values, dtype=cupy.complex128)
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    w = cupy.asarray(weights, dtype=cupy.complex128).reshape(-1)
    if src_arr.ndim != 4:
        raise ValueError(
            f"Weighted gather-add expects source shape (nbox, 4, ndir, nrhs), got ndim={src_arr.ndim}."
        )
    if int(src_arr.shape[1]) != 4 or int(tgt.shape[1]) != 4:
        raise ValueError("Weighted gather-add expects 4 directional channels.")
    if int(src_arr.shape[2]) != int(w.size):
        raise ValueError(
            "Weighted gather-add direction count mismatch between source and weights: "
            f"{int(src_arr.shape[2])} vs {int(w.size)}."
        )
    if tuple(int(v) for v in tgt.shape[1:]) != tuple(int(v) for v in src_arr.shape[1:]):
        raise ValueError(
            "Weighted gather-add source/target trailing-shape mismatch: "
            f"source={tuple(int(v) for v in src_arr.shape[1:])}, "
            f"target={tuple(int(v) for v in tgt.shape[1:])}."
        )
    if (
        int(cupy.max(src)) >= int(src_arr.shape[0])
        or int(cupy.min(src)) < 0
        or int(cupy.max(dst)) >= int(tgt.shape[0])
        or int(cupy.min(dst)) < 0
    ):
        raise ValueError("Weighted gather-add index out of bounds.")

    threads = 256
    total = int(src.size) * 4 * int(w.size) * int(src_arr.shape[3])
    blocks = max(1, (total + threads - 1) // threads)
    kernel = _weighted_gather_add_unique_complex128_raw_kernel()
    kernel(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(src.size)),
            np.int64(int(w.size)),
            np.int64(int(src_arr.shape[3])),
            src,
            dst,
            src_arr.reshape(-1),
            w,
            tgt.reshape(-1),
        ),
    )


@cache
def _transfer_up_packed_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void transfer_up_packed_unique_complex128(
        const long long n_pairs,
        const long long n_source,
        const long long n_target,
        const long long n_rhs,
        const int width,
        const int* src_rows,
        const int* dst_rows,
        const int* packed_cols,
        const complex<double>* packed_vals,
        const complex<double>* phase,
        const complex<double>* source_values,
        complex<double>* out
    ) {
        const long long span = 4LL * n_target * n_rhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * span;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / span;
            const long long lane = i - pair_idx * span;
            const long long chan_stride = n_target * n_rhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dst_dir = rem / n_rhs;
            const long long rhs = rem - dst_dir * n_rhs;

            const long long src_box = src_rows[pair_idx];
            const long long dst_box = dst_rows[pair_idx];
            const int row_base = (int)(dst_dir * (long long)width);

            const long long src_base = ((src_box * 4LL + chan) * n_source) * n_rhs + rhs;
            complex<double> acc = complex<double>(0.0, 0.0);
            for (int k = 0; k < width; ++k) {
                const int col = packed_cols[row_base + k];
                if (col < 0) {
                    break;
                }
                const complex<double> w = packed_vals[row_base + k];
                const complex<double> x = source_values[src_base + (long long)col * n_rhs];
                acc += w * x;
            }
            const long long out_idx = ((dst_box * 4LL + chan) * n_target + dst_dir) * n_rhs + rhs;
            out[out_idx] += acc * phase[dst_dir];
        }
    }
    """
    return cupy.RawKernel(source, "transfer_up_packed_unique_complex128")


@cache
def _transfer_down_packed_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void transfer_down_packed_unique_complex128(
        const long long n_pairs,
        const long long n_source,
        const long long n_target,
        const long long n_rhs,
        const int width,
        const int* src_rows,
        const int* dst_rows,
        const int* packed_cols,
        const complex<double>* packed_vals,
        const complex<double>* phase,
        const complex<double>* source_values,
        complex<double>* out
    ) {
        const long long span = 4LL * n_target * n_rhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * span;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / span;
            const long long lane = i - pair_idx * span;
            const long long chan_stride = n_target * n_rhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dst_dir = rem / n_rhs;
            const long long rhs = rem - dst_dir * n_rhs;

            const long long src_box = src_rows[pair_idx];
            const long long dst_box = dst_rows[pair_idx];
            const int row_base = (int)(dst_dir * (long long)width);

            const long long src_base = ((src_box * 4LL + chan) * n_source) * n_rhs + rhs;
            complex<double> acc = complex<double>(0.0, 0.0);
            for (int k = 0; k < width; ++k) {
                const int col = packed_cols[row_base + k];
                if (col < 0) {
                    break;
                }
                const complex<double> w = packed_vals[row_base + k];
                const complex<double> x = source_values[src_base + (long long)col * n_rhs];
                acc += w * (phase[col] * x);
            }
            const long long out_idx = ((dst_box * 4LL + chan) * n_target + dst_dir) * n_rhs + rhs;
            out[out_idx] += acc;
        }
    }
    """
    return cupy.RawKernel(source, "transfer_down_packed_unique_complex128")


def _transfer_up_packed_unique_complex128(
    target: Any,
    dst_indices: Any,
    source_values: Any,
    src_indices: Any,
    map_data: CuPyDirectionalInterpolationData,
    phase: Any,
    *,
    cupy: Any,
) -> None:
    """Fused packed-stencil upward transfer: map + phase + destination accumulate."""

    if str(map_data.storage) != "packed_stencil":
        raise ValueError("Packed transfer kernel requires packed_stencil map storage.")
    src = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    dst = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(src.size) == 0:
        return
    if int(src.size) != int(dst.size):
        raise ValueError("Packed transfer source/destination index count mismatch.")
    src_arr = cupy.asarray(source_values, dtype=cupy.complex128)
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    if src_arr.ndim != 4 or tgt.ndim != 4:
        raise ValueError("Packed transfer expects source/target shape (nbox,4,ndir,nrhs).")
    n_source = int(map_data.source_order)
    n_target = int(map_data.target_order)
    if int(src_arr.shape[2]) != n_source or int(tgt.shape[2]) != n_target:
        raise ValueError(
            f"Packed transfer directional size mismatch source={int(src_arr.shape[2])}/{n_source} "
            f"target={int(tgt.shape[2])}/{n_target}."
        )
    if int(src_arr.shape[3]) != int(tgt.shape[3]):
        raise ValueError("Packed transfer RHS mismatch between source and target.")
    if int(src_arr.shape[1]) != 4 or int(tgt.shape[1]) != 4:
        raise ValueError("Packed transfer expects 4 directional channels.")
    packed_cols, packed_vals, width_i32 = map_data.matrix
    phase_arr = cupy.asarray(phase, dtype=cupy.complex128).reshape(-1)
    if int(phase_arr.size) != n_target:
        raise ValueError(
            f"Packed transfer-up phase length mismatch: {int(phase_arr.size)} vs {n_target}."
        )
    width = int(width_i32)
    threads = 256
    total = int(src.size) * 4 * n_target * int(src_arr.shape[3])
    blocks = max(1, (total + threads - 1) // threads)
    _transfer_up_packed_unique_complex128_raw_kernel()(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(src.size)),
            np.int64(n_source),
            np.int64(n_target),
            np.int64(int(src_arr.shape[3])),
            np.int32(width),
            src,
            dst,
            cupy.asarray(packed_cols, dtype=cupy.int32),
            cupy.asarray(packed_vals, dtype=cupy.complex128),
            phase_arr,
            src_arr.reshape(-1),
            tgt.reshape(-1),
        ),
    )


def _transfer_down_packed_unique_complex128(
    target: Any,
    dst_indices: Any,
    source_values: Any,
    src_indices: Any,
    map_data: CuPyDirectionalInterpolationData,
    phase: Any,
    *,
    cupy: Any,
) -> None:
    """Fused packed-stencil downward transfer: phase + map + destination accumulate."""

    if str(map_data.storage) != "packed_stencil":
        raise ValueError("Packed transfer kernel requires packed_stencil map storage.")
    src = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    dst = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(src.size) == 0:
        return
    if int(src.size) != int(dst.size):
        raise ValueError("Packed transfer source/destination index count mismatch.")
    src_arr = cupy.asarray(source_values, dtype=cupy.complex128)
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    if src_arr.ndim != 4 or tgt.ndim != 4:
        raise ValueError("Packed transfer expects source/target shape (nbox,4,ndir,nrhs).")
    n_source = int(map_data.source_order)
    n_target = int(map_data.target_order)
    if int(src_arr.shape[2]) != n_source or int(tgt.shape[2]) != n_target:
        raise ValueError(
            f"Packed transfer directional size mismatch source={int(src_arr.shape[2])}/{n_source} "
            f"target={int(tgt.shape[2])}/{n_target}."
        )
    if int(src_arr.shape[3]) != int(tgt.shape[3]):
        raise ValueError("Packed transfer RHS mismatch between source and target.")
    if int(src_arr.shape[1]) != 4 or int(tgt.shape[1]) != 4:
        raise ValueError("Packed transfer expects 4 directional channels.")
    packed_cols, packed_vals, width_i32 = map_data.matrix
    phase_arr = cupy.asarray(phase, dtype=cupy.complex128).reshape(-1)
    if int(phase_arr.size) != n_source:
        raise ValueError(
            f"Packed transfer-down phase length mismatch: {int(phase_arr.size)} vs {n_source}."
        )
    width = int(width_i32)
    threads = 256
    total = int(src.size) * 4 * n_target * int(src_arr.shape[3])
    blocks = max(1, (total + threads - 1) // threads)
    _transfer_down_packed_unique_complex128_raw_kernel()(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(src.size)),
            np.int64(n_source),
            np.int64(n_target),
            np.int64(int(src_arr.shape[3])),
            np.int32(width),
            src,
            dst,
            cupy.asarray(packed_cols, dtype=cupy.int32),
            cupy.asarray(packed_vals, dtype=cupy.complex128),
            phase_arr,
            src_arr.reshape(-1),
            tgt.reshape(-1),
        ),
    )


@cache
def _transfer_up_csr_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void transfer_up_csr_unique_complex128(
        const long long n_pairs,
        const long long n_source,
        const long long n_target,
        const long long n_rhs,
        const int* src_rows,
        const int* dst_rows,
        const int* indptr,
        const int* indices,
        const complex<double>* values,
        const complex<double>* phase,
        const complex<double>* source_values,
        complex<double>* out
    ) {
        const long long span = 4LL * n_target * n_rhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * span;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / span;
            const long long lane = i - pair_idx * span;
            const long long chan_stride = n_target * n_rhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dst_dir = rem / n_rhs;
            const long long rhs = rem - dst_dir * n_rhs;

            const long long src_box = src_rows[pair_idx];
            const long long dst_box = dst_rows[pair_idx];
            const int start = indptr[dst_dir];
            const int end = indptr[dst_dir + 1];
            const long long src_base = ((src_box * 4LL + chan) * n_source) * n_rhs + rhs;

            complex<double> acc = complex<double>(0.0, 0.0);
            for (int p = start; p < end; ++p) {
                const int col = indices[p];
                const complex<double> w = values[p];
                const complex<double> x = source_values[src_base + (long long)col * n_rhs];
                acc += w * x;
            }
            const long long out_idx = ((dst_box * 4LL + chan) * n_target + dst_dir) * n_rhs + rhs;
            out[out_idx] += acc * phase[dst_dir];
        }
    }
    """
    return cupy.RawKernel(source, "transfer_up_csr_unique_complex128")


@cache
def _transfer_down_csr_unique_complex128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void transfer_down_csr_unique_complex128(
        const long long n_pairs,
        const long long n_source,
        const long long n_target,
        const long long n_rhs,
        const int* src_rows,
        const int* dst_rows,
        const int* indptr,
        const int* indices,
        const complex<double>* values,
        const complex<double>* phase,
        const complex<double>* source_values,
        complex<double>* out
    ) {
        const long long span = 4LL * n_target * n_rhs;
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_pairs * span;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            const long long pair_idx = i / span;
            const long long lane = i - pair_idx * span;
            const long long chan_stride = n_target * n_rhs;
            const long long chan = lane / chan_stride;
            const long long rem = lane - chan * chan_stride;
            const long long dst_dir = rem / n_rhs;
            const long long rhs = rem - dst_dir * n_rhs;

            const long long src_box = src_rows[pair_idx];
            const long long dst_box = dst_rows[pair_idx];
            const int start = indptr[dst_dir];
            const int end = indptr[dst_dir + 1];
            const long long src_base = ((src_box * 4LL + chan) * n_source) * n_rhs + rhs;

            complex<double> acc = complex<double>(0.0, 0.0);
            for (int p = start; p < end; ++p) {
                const int col = indices[p];
                const complex<double> w = values[p];
                const complex<double> x = source_values[src_base + (long long)col * n_rhs];
                acc += w * (phase[col] * x);
            }
            const long long out_idx = ((dst_box * 4LL + chan) * n_target + dst_dir) * n_rhs + rhs;
            out[out_idx] += acc;
        }
    }
    """
    return cupy.RawKernel(source, "transfer_down_csr_unique_complex128")


def _transfer_up_sparse_unique_complex128(
    target: Any,
    dst_indices: Any,
    source_values: Any,
    src_indices: Any,
    map_data: CuPyDirectionalInterpolationData,
    phase: Any,
    *,
    cupy: Any,
) -> None:
    """Fused sparse upward transfer: CSR map + phase + destination accumulate."""

    if str(map_data.storage) != "sparse":
        raise ValueError("Sparse transfer kernel requires sparse map storage.")
    src = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    dst = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(src.size) == 0:
        return
    if int(src.size) != int(dst.size):
        raise ValueError("Sparse transfer source/destination index count mismatch.")
    src_arr = cupy.asarray(source_values, dtype=cupy.complex128)
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    if src_arr.ndim != 4 or tgt.ndim != 4:
        raise ValueError("Sparse transfer expects source/target shape (nbox,4,ndir,nrhs).")
    n_source = int(map_data.source_order)
    n_target = int(map_data.target_order)
    if int(src_arr.shape[2]) != n_source or int(tgt.shape[2]) != n_target:
        raise ValueError(
            f"Sparse transfer directional size mismatch source={int(src_arr.shape[2])}/{n_source} "
            f"target={int(tgt.shape[2])}/{n_target}."
        )
    if int(src_arr.shape[3]) != int(tgt.shape[3]):
        raise ValueError("Sparse transfer RHS mismatch between source and target.")
    if int(src_arr.shape[1]) != 4 or int(tgt.shape[1]) != 4:
        raise ValueError("Sparse transfer expects 4 directional channels.")
    sparse = map_data.matrix
    phase_arr = cupy.asarray(phase, dtype=cupy.complex128).reshape(-1)
    if int(phase_arr.size) != n_target:
        raise ValueError(
            f"Sparse transfer-up phase length mismatch: {int(phase_arr.size)} vs {n_target}."
        )
    threads = 256
    total = int(src.size) * 4 * n_target * int(src_arr.shape[3])
    blocks = max(1, (total + threads - 1) // threads)
    _transfer_up_csr_unique_complex128_raw_kernel()(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(src.size)),
            np.int64(n_source),
            np.int64(n_target),
            np.int64(int(src_arr.shape[3])),
            src,
            dst,
            cupy.asarray(sparse.indptr, dtype=cupy.int32),
            cupy.asarray(sparse.indices, dtype=cupy.int32),
            cupy.asarray(sparse.data, dtype=cupy.complex128),
            phase_arr,
            src_arr.reshape(-1),
            tgt.reshape(-1),
        ),
    )


def _transfer_down_sparse_unique_complex128(
    target: Any,
    dst_indices: Any,
    source_values: Any,
    src_indices: Any,
    map_data: CuPyDirectionalInterpolationData,
    phase: Any,
    *,
    cupy: Any,
) -> None:
    """Fused sparse downward transfer: phase + CSR map + destination accumulate."""

    if str(map_data.storage) != "sparse":
        raise ValueError("Sparse transfer kernel requires sparse map storage.")
    src = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    dst = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(src.size) == 0:
        return
    if int(src.size) != int(dst.size):
        raise ValueError("Sparse transfer source/destination index count mismatch.")
    src_arr = cupy.asarray(source_values, dtype=cupy.complex128)
    tgt = cupy.asarray(target, dtype=cupy.complex128)
    if src_arr.ndim != 4 or tgt.ndim != 4:
        raise ValueError("Sparse transfer expects source/target shape (nbox,4,ndir,nrhs).")
    n_source = int(map_data.source_order)
    n_target = int(map_data.target_order)
    if int(src_arr.shape[2]) != n_source or int(tgt.shape[2]) != n_target:
        raise ValueError(
            f"Sparse transfer directional size mismatch source={int(src_arr.shape[2])}/{n_source} "
            f"target={int(tgt.shape[2])}/{n_target}."
        )
    if int(src_arr.shape[3]) != int(tgt.shape[3]):
        raise ValueError("Sparse transfer RHS mismatch between source and target.")
    if int(src_arr.shape[1]) != 4 or int(tgt.shape[1]) != 4:
        raise ValueError("Sparse transfer expects 4 directional channels.")
    sparse = map_data.matrix
    phase_arr = cupy.asarray(phase, dtype=cupy.complex128).reshape(-1)
    if int(phase_arr.size) != n_source:
        raise ValueError(
            f"Sparse transfer-down phase length mismatch: {int(phase_arr.size)} vs {n_source}."
        )
    threads = 256
    total = int(src.size) * 4 * n_target * int(src_arr.shape[3])
    blocks = max(1, (total + threads - 1) // threads)
    _transfer_down_csr_unique_complex128_raw_kernel()(
        (int(blocks),),
        (threads,),
        (
            np.int64(int(src.size)),
            np.int64(n_source),
            np.int64(n_target),
            np.int64(int(src_arr.shape[3])),
            src,
            dst,
            cupy.asarray(sparse.indptr, dtype=cupy.int32),
            cupy.asarray(sparse.indices, dtype=cupy.int32),
            cupy.asarray(sparse.data, dtype=cupy.complex128),
            phase_arr,
            src_arr.reshape(-1),
            tgt.reshape(-1),
        ),
    )


@cache
def _directional_packed_stencil_map_c128_raw_kernel() -> Any:
    cupy, _ = import_cupy()
    source = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void directional_packed_stencil_map_c128(
        const long long n_batch,
        const long long n_chan,
        const long long n_source,
        const long long n_target,
        const long long n_rhs,
        const int width,
        const int* packed_cols,
        const complex<double>* packed_vals,
        const complex<double>* src,
        complex<double>* out
    ) {
        const long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
        const long long total = n_batch * n_chan * n_target * n_rhs;
        for (long long i = tid; i < total; i += (long long)blockDim.x * (long long)gridDim.x) {
            long long t = i;
            const long long rhs = t % n_rhs;
            t /= n_rhs;
            const long long dst = t % n_target;
            t /= n_target;
            const long long chan = t % n_chan;
            const long long batch = t / n_chan;

            const long long src_base = ((batch * n_chan + chan) * n_source) * n_rhs + rhs;
            const int row_base = (int)(dst * (long long)width);
            complex<double> acc = complex<double>(0.0, 0.0);
            for (int k = 0; k < width; ++k) {
                const int col = packed_cols[row_base + k];
                if (col < 0) {
                    break;
                }
                const complex<double> w = packed_vals[row_base + k];
                const complex<double> x = src[src_base + (long long)col * n_rhs];
                acc += w * x;
            }
            out[i] = acc;
        }
    }
    """
    return cupy.RawKernel(source, "directional_packed_stencil_map_c128")


def _apply_directional_map(
    values: Any, map_data: CuPyDirectionalInterpolationData, *, cupy: Any
) -> Any:
    """Apply one directional transfer map on channel batches.

    Parameters
    ----------
    values:
        Shape `(nbatch, 4, n_source, nrhs)`.
    map_data:
        Device transfer map metadata.
    """

    arr = cupy.asarray(values, dtype=cupy.complex128)
    if arr.ndim != 4 or int(arr.shape[1]) != 4:
        raise ValueError("Directional transfer input must have shape (nbatch, 4, n_source, nrhs).")
    if int(arr.shape[2]) != int(map_data.source_order):
        raise ValueError(
            "Directional transfer source-order mismatch: "
            f"{int(arr.shape[2])} vs {int(map_data.source_order)}."
        )
    n_batch = int(arr.shape[0])
    n_chan = int(arr.shape[1])
    n_rhs = int(arr.shape[3])
    source_order = int(arr.shape[2])
    flat = arr.transpose(0, 1, 3, 2).reshape(-1, source_order)
    if str(map_data.storage) == "dense":
        dense = cupy.asarray(map_data.matrix, dtype=cupy.complex128)
        mapped_flat = flat @ dense.T
        target_order = int(dense.shape[0])
    elif str(map_data.storage) == "packed_stencil":
        packed_cols, packed_vals, width_i32 = map_data.matrix
        width = int(width_i32)
        target_order = int(map_data.target_order)
        src = cupy.ascontiguousarray(arr)
        out = cupy.empty((n_batch, n_chan, target_order, n_rhs), dtype=cupy.complex128)
        total = n_batch * n_chan * target_order * n_rhs
        threads = 256
        blocks = max(1, (total + threads - 1) // threads)
        _directional_packed_stencil_map_c128_raw_kernel()(
            (int(blocks),),
            (threads,),
            (
                np.int64(n_batch),
                np.int64(n_chan),
                np.int64(source_order),
                np.int64(target_order),
                np.int64(n_rhs),
                np.int32(width),
                cupy.asarray(packed_cols, dtype=cupy.int32),
                cupy.asarray(packed_vals, dtype=cupy.complex128),
                src.reshape(-1),
                out.reshape(-1),
            ),
        )
        return out
    else:
        sparse = map_data.matrix
        mapped_flat = (sparse @ flat.T).T
        target_order = int(sparse.shape[0])
    return mapped_flat.reshape(n_batch, n_chan, n_rhs, target_order).transpose(0, 1, 3, 2)


def _box_outgoing_to_directional_cupy(
    directional: CuPyDirectionalTransformsData,
    box_states: Any,
    *,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Map batched outgoing box SVWF states to directional channels on device."""

    states = cupy.asarray(box_states, dtype=cupy.complex128)
    nscl = int(directional.nscl)
    if int(states.shape[1]) != 2 * nscl:
        raise ValueError(
            f"box_states second dimension must be {2 * nscl}, got {int(states.shape[1])}."
        )
    ndir = int(directional.grid.n_directions)
    n_batch = int(states.shape[0])
    n_rhs = int(states.shape[2])
    n_alpha = int(directional.grid.n_alpha)
    n_beta = int(directional.grid.n_beta)
    a_box = states[:, :nscl, :]
    b_box = states[:, nscl:, :]
    out_arr = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.empty((n_batch, 4, ndir, n_rhs), dtype=cupy.complex128)
    )
    out_arr.fill(0)
    work = out_arr.reshape(n_batch, 4, n_alpha, n_beta, n_rhs)
    beta_perm = directional.grid.beta_reflection_permutation
    fth_reflected = directional.fth_beta[beta_perm]
    fph_reflected = directional.fph_beta[beta_perm]

    # An FFT over alpha was tested on large-scale CuPy MLFMM smokes. It gave
    # effectively identical runtime while increasing peak memory, so keep the
    # simpler explicit phase contraction until profiling shows a different
    # bottleneck.
    for im, mode_idx in enumerate(directional.mode_indices_by_m):
        if int(mode_idx.size) == 0:
            continue
        fth_m = fth_reflected[:, mode_idx]
        fph_m = fph_reflected[:, mode_idx]
        a_m = a_box[:, mode_idx, :]
        b_m = b_box[:, mode_idx, :]
        a_theta_beta = cupy.matmul(fth_m[None, :, :], a_m)
        a_phi_beta = cupy.matmul(fph_m[None, :, :], a_m)
        b_theta_beta = cupy.matmul(fth_m[None, :, :], b_m)
        b_phi_beta = cupy.matmul(fph_m[None, :, :], b_m)
        phase = directional.phase_by_m[:, im].reshape(1, n_alpha, 1, 1)
        work[:, 0] += phase * a_theta_beta[:, None, :, :]
        work[:, 1] += phase * a_phi_beta[:, None, :, :]
        work[:, 2] += phase * b_theta_beta[:, None, :, :]
        work[:, 3] += phase * b_phi_beta[:, None, :, :]
    return out_arr


def _directional_to_box_regular_cupy(
    directional: CuPyDirectionalTransformsData,
    directional_channels: Any,
    *,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Map batched directional channels to regular box SVWF states on device."""

    channels = cupy.asarray(directional_channels, dtype=cupy.complex128)
    if int(channels.shape[1]) != 4:
        raise ValueError(
            f"directional channel batch must have 4 channels, got {int(channels.shape[1])}."
        )
    n_batch = int(channels.shape[0])
    n_rhs = int(channels.shape[3])
    n_alpha = int(directional.grid.n_alpha)
    n_beta = int(directional.grid.n_beta)
    nscl = int(directional.nscl)
    reflected = channels.reshape(n_batch, 4, n_alpha, n_beta, n_rhs)[
        :, :, :, directional.grid.beta_reflection_permutation, :
    ]
    out_arr = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.empty((n_batch, 2 * nscl, n_rhs), dtype=cupy.complex128)
    )
    top = out_arr[:, :nscl, :]
    bottom = out_arr[:, nscl:, :]
    top.fill(0)
    bottom.fill(0)
    phase_adj = cupy.conjugate(directional.phase_by_m)
    fth_h_all = cupy.conjugate(cupy.swapaxes(directional.fth_beta, 0, 1))
    fph_h_all = cupy.conjugate(cupy.swapaxes(directional.fph_beta, 0, 1))

    for im, mode_idx in enumerate(directional.mode_indices_by_m):
        if int(mode_idx.size) == 0:
            continue
        phase_m = phase_adj[:, im]
        a_theta_m = cupy.einsum("a,bakr->bkr", phase_m, reflected[:, 0], optimize=True)
        a_phi_m = cupy.einsum("a,bakr->bkr", phase_m, reflected[:, 1], optimize=True)
        b_theta_m = cupy.einsum("a,bakr->bkr", phase_m, reflected[:, 2], optimize=True)
        b_phi_m = cupy.einsum("a,bakr->bkr", phase_m, reflected[:, 3], optimize=True)
        fth_h = fth_h_all[mode_idx, :]
        fph_h = fph_h_all[mode_idx, :]
        fth_a_theta = cupy.matmul(fth_h[None, :, :], a_theta_m)
        fph_a_phi = cupy.matmul(fph_h[None, :, :], a_phi_m)
        fph_b_theta = cupy.matmul(fph_h[None, :, :], b_theta_m)
        fth_b_phi = cupy.matmul(fth_h[None, :, :], b_phi_m)
        top[:, mode_idx, :] = fth_a_theta + fph_a_phi - 1j * fph_b_theta + 1j * fth_b_phi
        fth_b_theta = cupy.matmul(fth_h[None, :, :], b_theta_m)
        fph_b_phi = cupy.matmul(fph_h[None, :, :], b_phi_m)
        fph_a_theta = cupy.matmul(fph_h[None, :, :], a_theta_m)
        fth_a_phi = cupy.matmul(fth_h[None, :, :], a_phi_m)
        bottom[:, mode_idx, :] = fth_b_theta + fph_b_phi - 1j * fph_a_theta + 1j * fth_a_phi
    return out_arr


def _box_outgoing_to_directional_adjoint_cupy(
    directional: CuPyDirectionalTransformsData,
    directional_channels: Any,
    *,
    cupy: Any,
) -> Any:
    """Apply the adjoint of the outgoing directional transform on device."""

    channels = cupy.asarray(directional_channels, dtype=cupy.complex128)
    if channels.ndim != 4 or int(channels.shape[1]) != 4:
        raise ValueError("Directional adjoint input must have shape (batch, 4, ndir, nrhs).")
    n_batch = int(channels.shape[0])
    n_rhs = int(channels.shape[3])
    n_alpha = int(directional.grid.n_alpha)
    n_beta = int(directional.grid.n_beta)
    nscl = int(directional.nscl)
    reflected = channels.reshape(n_batch, 4, n_alpha, n_beta, n_rhs)[
        :, :, :, directional.grid.beta_reflection_permutation, :
    ]
    out = cupy.zeros((n_batch, 2 * nscl, n_rhs), dtype=cupy.complex128)
    top = out[:, :nscl, :]
    bottom = out[:, nscl:, :]
    fth_h_all = cupy.conjugate(cupy.swapaxes(directional.fth_beta, 0, 1))
    fph_h_all = cupy.conjugate(cupy.swapaxes(directional.fph_beta, 0, 1))
    phase_adj = cupy.conjugate(directional.phase_by_m)
    for im, mode_idx in enumerate(directional.mode_indices_by_m):
        if int(mode_idx.size) == 0:
            continue
        a_theta = cupy.einsum("a,bakr->bkr", phase_adj[:, im], reflected[:, 0], optimize=True)
        a_phi = cupy.einsum("a,bakr->bkr", phase_adj[:, im], reflected[:, 1], optimize=True)
        b_theta = cupy.einsum("a,bakr->bkr", phase_adj[:, im], reflected[:, 2], optimize=True)
        b_phi = cupy.einsum("a,bakr->bkr", phase_adj[:, im], reflected[:, 3], optimize=True)
        fth_h = fth_h_all[mode_idx, :]
        fph_h = fph_h_all[mode_idx, :]
        top[:, mode_idx, :] = cupy.matmul(fth_h[None, :, :], a_theta) + cupy.matmul(
            fph_h[None, :, :], a_phi
        )
        bottom[:, mode_idx, :] = cupy.matmul(fth_h[None, :, :], b_theta) + cupy.matmul(
            fph_h[None, :, :], b_phi
        )
    return out


def _directional_to_box_regular_adjoint_cupy(
    directional: CuPyDirectionalTransformsData,
    box_states: Any,
    *,
    cupy: Any,
) -> Any:
    """Apply the adjoint of the regular directional receive transform."""

    states = cupy.asarray(box_states, dtype=cupy.complex128)
    nscl = int(directional.nscl)
    if states.ndim != 3 or int(states.shape[1]) != 2 * nscl:
        raise ValueError("Regular-transform adjoint input must have shape (batch, 2*nscl, nrhs).")
    zeros = cupy.zeros_like(states)
    a_state = cupy.concatenate((states[:, :nscl, :], zeros[:, nscl:, :]), axis=1)
    b_state = cupy.concatenate((zeros[:, :nscl, :], states[:, nscl:, :]), axis=1)
    a_channels = _box_outgoing_to_directional_cupy(directional, a_state, cupy=cupy)
    b_channels = _box_outgoing_to_directional_cupy(directional, b_state, cupy=cupy)
    out = cupy.empty(
        (int(states.shape[0]), 4, int(directional.grid.n_directions), int(states.shape[2])),
        dtype=cupy.complex128,
    )
    out[:, 0] = a_channels[:, 0] + 1j * b_channels[:, 3]
    out[:, 1] = a_channels[:, 1] - 1j * b_channels[:, 2]
    out[:, 2] = 1j * a_channels[:, 1] + b_channels[:, 2]
    out[:, 3] = -1j * a_channels[:, 0] + b_channels[:, 3]
    return out


def _apply_directional_map_adjoint(
    values: Any,
    map_data: CuPyDirectionalInterpolationData,
    *,
    cupy: Any,
    cache: dict[int, Any] | None = None,
) -> Any:
    """Apply a conjugate-transposed directional interpolation map on device."""

    arr = cupy.asarray(values, dtype=cupy.complex128)
    if arr.ndim != 4 or int(arr.shape[1]) != 4:
        raise ValueError(
            "Directional adjoint map input must have shape (batch, 4, n_target, nrhs)."
        )
    target_order = int(map_data.target_order)
    if int(arr.shape[2]) != target_order:
        raise ValueError(
            f"Directional adjoint map target mismatch: {int(arr.shape[2])} vs {target_order}."
        )
    source_order = int(map_data.source_order)
    flat = arr.transpose(0, 1, 3, 2).reshape(-1, target_order)
    if str(map_data.storage) == "dense":
        matrix = cupy.asarray(map_data.matrix, dtype=cupy.complex128)
        mapped = flat @ cupy.conjugate(matrix)
    elif str(map_data.storage) == "sparse":
        sparse = map_data.matrix
        mapped = (sparse.conjugate().T @ flat.T).T
    elif str(map_data.storage) == "packed_stencil":
        key = id(map_data)
        sparse = None if cache is None else cache.get(key)
        if sparse is None:
            cupyx_sparse = import_module("cupyx.scipy.sparse")
            packed_cols, packed_vals, width_i32 = map_data.matrix
            width = int(width_i32)
            rows = cupy.repeat(cupy.arange(target_order, dtype=cupy.int32), width)
            cols = cupy.asarray(packed_cols, dtype=cupy.int32).reshape(-1)
            vals = cupy.asarray(packed_vals, dtype=cupy.complex128).reshape(-1)
            valid = cols >= 0
            sparse = cupyx_sparse.coo_matrix(
                (vals[valid], (rows[valid], cols[valid])),
                shape=(target_order, source_order),
            ).tocsr()
            if cache is not None:
                cache[key] = sparse
        mapped = (sparse.conjugate().T @ flat.T).T
    else:
        raise ValueError(f"Unsupported directional map storage {map_data.storage!r}.")
    return mapped.reshape(int(arr.shape[0]), 4, int(arr.shape[3]), source_order).transpose(
        0, 1, 3, 2
    )


def _leaf_translation_blocks_from_pair_deltas(
    pair_deltas: Any,
    *,
    tables: CuPyLeafTranslationTablesData,
    pair_blocks_scratch: dict[str, Any] | None,
    cupy: Any,
) -> Any:
    """Build on-the-fly leaf translation blocks for one grouped pair schedule.

    This helper currently materializes a dense `(n_pairs, nmodes_out, nmodes_in)`
    tensor, so peak temporary memory scales with `n_pairs * nmodes_out * nmodes_in`.
    """

    deltas = cupy.asarray(pair_deltas, dtype=cupy.float64).reshape(-1, 3)
    n_pairs = int(deltas.shape[0])
    if n_pairs == 0:
        return cupy.empty((0, int(tables.nmodes_out), int(tables.nmodes_in)), dtype=cupy.complex128)
    nmodes_out = int(tables.nmodes_out)
    nmodes_in = int(tables.nmodes_in)
    required = int(n_pairs * nmodes_out * nmodes_in)
    if pair_blocks_scratch is not None:
        backing = pair_blocks_scratch.get("buffer")
        if backing is None or int(backing.size) < required:
            backing = cupy.empty((required,), dtype=cupy.complex128)
            pair_blocks_scratch["buffer"] = backing
        blocks = backing[:required].reshape(n_pairs, nmodes_out, nmodes_in)
    else:
        blocks = cupy.empty((n_pairs, nmodes_out, nmodes_in), dtype=cupy.complex128)

    kernel = _leaf_translation_blocks_rect_raw_kernel(
        int(tables.full_order), np.dtype(np.complex128).str
    )
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_pairs = int(props["maxGridSize"][0])
    max_threads = int(props["maxThreadsPerBlock"])
    if max_threads >= 256:
        threads = 256
    elif max_threads >= 128:
        threads = 128
    else:
        threads = 64
    chunk_pairs = max(1, min(int(max_grid_pairs), n_pairs))
    for start in range(0, n_pairs, chunk_pairs):
        end = min(n_pairs, start + chunk_pairs)
        count = int(end - start)
        kernel(
            (int(count),),
            (int(threads),),
            (
                np.int32(count),
                np.int32(nmodes_out),
                np.int32(nmodes_in),
                deltas[start:end].reshape(-1),
                tables.out_mode_indices,
                tables.in_mode_indices,
                tables.mode_m_out,
                tables.mode_m_in,
                tables.re_j,
                tables.im_j,
                np.float64(float(tables.inv_dr)),
                np.int32(int(tables.last_index)),
                tables.plm_coeffs,
                tables.compact_re_ab,
                tables.compact_im_ab,
                tables.pair_offset,
                tables.pair_pmin,
                tables.pair_pcount,
                blocks[start:end].reshape(-1),
            ),
        )
    return blocks


@cache
def _leaf_otf_aggregate_fused_raw_kernel(full_order: int, coeff_dtype_name: str) -> Any:
    """Return a RawKernel that builds leaf translations and aggregates directly."""

    cupy, _ = import_cupy()
    order = int(full_order)
    coeff_dtype = np.dtype(coeff_dtype_name)
    if coeff_dtype == np.dtype(np.complex64):
        coeff_t = "complex<float>"
    elif coeff_dtype == np.dtype(np.complex128):
        coeff_t = "complex<double>"
    else:
        raise ValueError(
            "Fused leaf aggregation supports only complex64/complex128 coefficients. "
            f"Got {coeff_dtype!r}."
        )
    n_orders = 2 * order + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    #include <cupy/atomics.cuh>
    #include <cupy/complex.cuh>

    __device__ double assoc_legendre_function_fused(
        const int l,
        const int m,
        const double* ct_powers,
        const double* st_powers,
        const double* plm_coeffs
    ) {{
        double plm = 0.0;
        const double st_pow = st_powers[m];
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    __device__ complex<double> bessel_lookup_linear_fused(
        const int p,
        const double r,
        const double* re_table,
        const double* im_table,
        const double inv_dr,
        const int last_index
    ) {{
        if (r <= 0.0) {{
            return complex<double>(re_table[p], im_table[p]);
        }}
        double t = r * inv_dr;
        int i0 = (int)floor(t);
        double frac = t - (double)i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = 0.0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = 1.0;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        const double re_val = (1.0 - frac) * re_table[base0] + frac * re_table[base1];
        const double im_val = (1.0 - frac) * im_table[base0] + frac * im_table[base1];
        return complex<double>(re_val, im_val);
    }}

    __device__ complex<double> coeff_to_c128_fused(const {coeff_t} value) {{
        return complex<double>((double)value.real(), (double)value.imag());
    }}

    __device__ void atomic_add_c128(complex<double>* address, const complex<double> value) {{
        double* raw = reinterpret_cast<double*>(address);
        atomicAdd(raw, value.real());
        atomicAdd(raw + 1, value.imag());
    }}

    extern "C" __global__ void mlfmm_leaf_otf_aggregate_fused(
        const int n_pairs,
        const int occupancy,
        const int n_out_modes,
        const int n_in_modes,
        const int n_rhs,
        const double* pair_deltas,
        const int* particle_indices,
        const int* row_indices,
        const int* target_rows,
        const {coeff_t}* x_states,
        const int* out_mode_indices,
        const int* in_mode_indices,
        const int* mode_m_out,
        const int* mode_m_in,
        const double* re_j,
        const double* im_j,
        const double inv_dr,
        const int last_index,
        const double* plm_coeffs,
        const double* re_ab,
        const double* im_ab,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        complex<double>* out
    ) {{
        for (int pair_idx = blockIdx.x; pair_idx < n_pairs; pair_idx += gridDim.x) {{
            const int tid = threadIdx.x;
            const int n_threads = blockDim.x;
            const int leaf_row = pair_idx / occupancy;
            const int occ = pair_idx - leaf_row * occupancy;
            const int group_row = row_indices[leaf_row];
            const int group_pair_idx = group_row * occupancy + occ;
            const int particle = particle_indices[group_pair_idx];
            const int out_row = target_rows[leaf_row];

            __shared__ double r_shared;
            __shared__ double ct_shared;
            __shared__ double st_shared;
            __shared__ double phi_shared;
            __shared__ double re_j_shared[{n_orders}];
            __shared__ double im_j_shared[{n_orders}];
            __shared__ double p_pdm_shared[{n_p_pdm}];
            __shared__ double cos_mphi_shared[{n_phase}];
            __shared__ double sin_mphi_shared[{n_phase}];
            __shared__ double ct_pow_shared[{n_orders}];
            __shared__ double st_pow_shared[{n_orders}];

            if (tid == 0) {{
                const double dx = pair_deltas[3 * group_pair_idx + 0];
                const double dy = pair_deltas[3 * group_pair_idx + 1];
                const double dz = pair_deltas[3 * group_pair_idx + 2];
                const double rr = sqrt(dx * dx + dy * dy + dz * dz);
                r_shared = rr;
                if (rr > 0.0) {{
                    ct_shared = dz / rr;
                    st_shared = sqrt(fmax(0.0, 1.0 - ct_shared * ct_shared));
                    phi_shared = atan2(dy, dx);
                }} else {{
                    ct_shared = 1.0;
                    st_shared = 0.0;
                    phi_shared = 0.0;
                }}
            }}
            __syncthreads();

            if (r_shared > 0.0) {{
                if (tid == 0) {{
                    ct_pow_shared[0] = 1.0;
                    st_pow_shared[0] = 1.0;
                    for (int p = 1; p < {n_orders}; ++p) {{
                        ct_pow_shared[p] = ct_pow_shared[p - 1] * ct_shared;
                        st_pow_shared[p] = st_pow_shared[p - 1] * st_shared;
                    }}
                }}
                __syncthreads();

                for (int p = tid; p < {n_orders}; p += n_threads) {{
                    const complex<double> radial =
                        bessel_lookup_linear_fused(p, r_shared, re_j, im_j, inv_dr, last_index);
                    re_j_shared[p] = radial.real();
                    im_j_shared[p] = radial.imag();
                    for (int absdm = 0; absdm <= p; ++absdm) {{
                        p_pdm_shared[p * (p + 1) / 2 + absdm] =
                            assoc_legendre_function_fused(
                                p, absdm, ct_pow_shared, st_pow_shared, plm_coeffs
                            );
                    }}
                }}
                if (tid == 0) {{
                    for (int dm = -2 * {order}; dm <= 2 * {order}; ++dm) {{
                        const int phase_idx = dm + 2 * {order};
                        sincos((double)dm * phi_shared,
                               &sin_mphi_shared[phase_idx],
                               &cos_mphi_shared[phase_idx]);
                    }}
                }}
            }}
            __syncthreads();

            const int n_outputs = n_out_modes * n_rhs;
            for (int flat = tid; flat < n_outputs; flat += n_threads) {{
                const int rhs = flat % n_rhs;
                const int out_idx = flat / n_rhs;
                complex<double> acc(0.0, 0.0);
                for (int in_idx = 0; in_idx < n_in_modes; ++in_idx) {{
                    complex<double> coeff(0.0, 0.0);
                    if (r_shared <= 0.0) {{
                        if (out_mode_indices[out_idx] == in_mode_indices[in_idx]) {{
                            coeff = complex<double>(1.0, 0.0);
                        }}
                    }} else {{
                        const int delta_m = mode_m_in[in_idx] - mode_m_out[out_idx];
                        const int phase_idx = delta_m + 2 * {order};
                        const int table_idx = out_idx * n_in_modes + in_idx;
                        const int base = pair_offset[table_idx];
                        const int p_min = pair_pmin[table_idx];
                        const int p_count = pair_pcount[table_idx];
                        double re_acc = 0.0;
                        double im_acc = 0.0;
                        for (int ip = 0; ip < p_count; ++ip) {{
                            const int p = p_min + ip;
                            const int ab_idx = base + ip;
                            const double plm =
                                p_pdm_shared[p * (p + 1) / 2 + abs(delta_m)];
                            const double re_abp = re_ab[ab_idx] * plm;
                            const double im_abp = im_ab[ab_idx] * plm;
                            const double re_abpr =
                                re_abp * re_j_shared[p] - im_abp * im_j_shared[p];
                            const double im_abpr =
                                re_abp * im_j_shared[p] + im_abp * re_j_shared[p];
                            const double re_phase =
                                re_abpr * cos_mphi_shared[phase_idx]
                                - im_abpr * sin_mphi_shared[phase_idx];
                            const double im_phase =
                                re_abpr * sin_mphi_shared[phase_idx]
                                + im_abpr * cos_mphi_shared[phase_idx];
                            re_acc += re_phase;
                            im_acc += im_phase;
                        }}
                        coeff = complex<double>(re_acc, im_acc);
                    }}
                    const complex<double> x = coeff_to_c128_fused(
                        x_states[((long long)particle * n_in_modes + in_idx) * n_rhs + rhs]
                    );
                    acc += coeff * x;
                }}
                atomic_add_c128(
                    out + ((long long)out_row * n_out_modes + out_idx) * n_rhs + rhs,
                    acc
                );
            }}
            __syncthreads();
        }}
    }}
    """
    return cupy.RawKernel(source, "mlfmm_leaf_otf_aggregate_fused")


def _launch_leaf_otf_aggregate_fused(
    *,
    pair_deltas: Any,
    particle_indices: Any,
    row_indices: Any,
    target_rows: Any,
    x_states: Any,
    out: Any,
    tables: CuPyLeafTranslationTablesData,
    occupancy: int,
    box_nm: int,
    nmodes: int,
    nrhs: int,
    cupy: Any,
) -> None:
    """Build on-the-fly leaf translations and aggregate without pair-block tensors."""

    count = int(row_indices.size)
    if count <= 0:
        return
    n_pairs = int(count * int(occupancy))
    kernel = _leaf_otf_aggregate_fused_raw_kernel(
        int(tables.full_order), np.dtype(x_states.dtype).str
    )
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_pairs = int(props["maxGridSize"][0])
    max_threads = int(props["maxThreadsPerBlock"])
    threads = 256 if max_threads >= 256 else 128 if max_threads >= 128 else 64
    blocks = max(1, min(max_grid_pairs, n_pairs))
    kernel(
        (blocks,),
        (threads,),
        (
            np.int32(n_pairs),
            np.int32(occupancy),
            np.int32(box_nm),
            np.int32(nmodes),
            np.int32(nrhs),
            pair_deltas.reshape(-1),
            particle_indices.reshape(-1),
            cupy.asarray(row_indices, dtype=cupy.int32).reshape(-1),
            cupy.asarray(target_rows, dtype=cupy.int32).reshape(-1),
            x_states.reshape(-1),
            tables.out_mode_indices,
            tables.in_mode_indices,
            tables.mode_m_out,
            tables.mode_m_in,
            tables.re_j,
            tables.im_j,
            np.float64(float(tables.inv_dr)),
            np.int32(int(tables.last_index)),
            tables.plm_coeffs,
            tables.compact_re_ab,
            tables.compact_im_ab,
            tables.pair_offset,
            tables.pair_pmin,
            tables.pair_pcount,
            out.reshape(-1),
        ),
    )


@cache
def _leaf_otf_receive_contract_raw_kernel() -> Any:
    """Return a RawKernel for leaf on-the-fly receive contractions."""

    cupy, _ = import_cupy()
    source = """
    #include <cupy/complex.cuh>

    extern "C" __global__ void mlfmm_leaf_otf_receive_contract(
        const long long n_items,
        const int occupancy,
        const int box_nm,
        const int nmodes,
        const int n_rhs,
        const complex<double>* pair_blocks,
        const complex<double>* incoming,
        const int* incoming_rows,
        const int* particle_indices,
        complex<double>* out
    ) {
        for (long long item = blockIdx.x * blockDim.x + threadIdx.x;
             item < n_items;
             item += (long long)blockDim.x * gridDim.x) {
            const int rhs = (int)(item % n_rhs);
            const long long tmp0 = item / n_rhs;
            const int mode = (int)(tmp0 % nmodes);
            const long long tmp1 = tmp0 / nmodes;
            const int occ = (int)(tmp1 % occupancy);
            const int leaf_row = (int)(tmp1 / occupancy);

            complex<double> acc(0.0, 0.0);
            const int incoming_row = incoming_rows[leaf_row];
            for (int box_mode = 0; box_mode < box_nm; ++box_mode) {
                const complex<double> w =
                    pair_blocks[(((long long)leaf_row * occupancy + occ) * box_nm + box_mode)
                                * nmodes + mode];
                const complex<double> w_conj(w.real(), -w.imag());
                const complex<double> x =
                    incoming[((long long)incoming_row * box_nm + box_mode) * n_rhs + rhs];
                acc += w_conj * x;
            }
            const int particle = particle_indices[(long long)leaf_row * occupancy + occ];
            out[((long long)particle * nmodes + mode) * n_rhs + rhs] += acc;
        }
    }
    """
    return cupy.RawKernel(source, "mlfmm_leaf_otf_receive_contract")


def _launch_leaf_otf_receive_contract(
    pair_blocks: Any,
    incoming: Any,
    incoming_rows: Any,
    particle_indices: Any,
    out: Any,
    *,
    occupancy: int,
    box_nm: int,
    nmodes: int,
    nrhs: int,
    cupy: Any,
) -> None:
    """Contract on-the-fly incoming box states back to particle coefficients."""

    count = int(incoming_rows.size)
    if count <= 0:
        return
    n_items = int(count * int(occupancy) * int(nmodes) * int(nrhs))
    kernel = _leaf_otf_receive_contract_raw_kernel()
    threads = 256
    blocks = max(1, min(65535, (n_items + threads - 1) // threads))
    kernel(
        (blocks,),
        (threads,),
        (
            np.int64(n_items),
            np.int32(occupancy),
            np.int32(box_nm),
            np.int32(nmodes),
            np.int32(nrhs),
            pair_blocks,
            incoming,
            cupy.asarray(incoming_rows, dtype=cupy.int32),
            cupy.asarray(particle_indices, dtype=cupy.int32).reshape(-1),
            out,
        ),
    )


def _leaf_otf_receive_shared_bytes(full_order: int, threads: int) -> int:
    """Return static shared-memory bytes for the fused leaf receive kernel."""

    order = int(full_order)
    thread_count = int(threads)
    if order < 0:
        raise ValueError(f"full_order must be >= 0. Got {full_order!r}.")
    if thread_count < 32 or thread_count % 32:
        raise ValueError(
            f"threads must be a positive whole-warp count of at least 32. Got {threads!r}."
        )
    n_orders = 2 * order + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    # Four scalar geometry values, radial tables, triangular P_l^m table,
    # azimuthal phase tables, ct/st powers, and one complex reduction slot per
    # thread. All entries are doubles.
    doubles = 4 + 2 * n_orders + n_p_pdm + 2 * n_phase + 2 * n_orders + 2 * thread_count
    return int(doubles * np.dtype(np.float64).itemsize)


def _leaf_otf_receive_threads(
    *,
    full_order: int,
    max_threads_per_block: int,
    shared_mem_per_block: int,
) -> int:
    """Choose the widest whole-warp leaf-receive block that fits static shared memory."""

    max_threads = int(max_threads_per_block)
    shared_limit = int(shared_mem_per_block)
    for threads in (256, 128, 64, 32):
        if threads > max_threads:
            continue
        if _leaf_otf_receive_shared_bytes(int(full_order), threads) <= shared_limit:
            return threads
    minimum = _leaf_otf_receive_shared_bytes(int(full_order), 32)
    raise ValueError(
        "Fused CuPy MLFMM leaf receive exceeds the device static shared-memory limit "
        f"even with one warp: order={int(full_order)}, required={minimum} bytes, "
        f"available={shared_limit} bytes."
    )


@cache
def _leaf_otf_receive_fused_raw_kernel(full_order: int, threads: int) -> Any:
    """Return a RawKernel for receive-side leaf translations without pair-block scratch."""

    cupy, _ = import_cupy()
    order = int(full_order)
    reduction_threads = int(threads)
    if reduction_threads < 32 or reduction_threads % 32:
        raise ValueError("leaf receive kernel threads must be a whole-warp count >= 32")
    n_orders = 2 * order + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    #include <cupy/complex.cuh>

    __device__ double assoc_legendre_function_receive(
        const int l,
        const int m,
        const double* ct_powers,
        const double* st_powers,
        const double* plm_coeffs
    ) {{
        double plm = 0.0;
        const double st_pow = st_powers[m];
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    __device__ complex<double> bessel_lookup_linear_receive(
        const int p,
        const double r,
        const double* re_table,
        const double* im_table,
        const double inv_dr,
        const int last_index
    ) {{
        if (r <= 0.0) {{
            return complex<double>(re_table[p], im_table[p]);
        }}
        double t = r * inv_dr;
        int i0 = (int)floor(t);
        double frac = t - (double)i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = 0.0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = 1.0;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        const double re_val = (1.0 - frac) * re_table[base0] + frac * re_table[base1];
        const double im_val = (1.0 - frac) * im_table[base0] + frac * im_table[base1];
        return complex<double>(re_val, im_val);
    }}

    extern "C" __global__ void mlfmm_leaf_otf_receive_fused(
        const int n_pairs,
        const int occupancy,
        const int n_out_modes,
        const int n_in_modes,
        const int n_rhs,
        const double* pair_deltas,
        const int* particle_indices,
        const int* row_indices,
        const int* incoming_rows,
        const complex<double>* incoming,
        const int* out_mode_indices,
        const int* in_mode_indices,
        const int* mode_m_out,
        const int* mode_m_in,
        const double* re_j,
        const double* im_j,
        const double inv_dr,
        const int last_index,
        const double* plm_coeffs,
        const double* re_ab,
        const double* im_ab,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        complex<double>* out
    ) {{
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int n_warps = blockDim.x >> 5;
        const int tile_base = blockIdx.y * 32;
        const int n_outputs = n_in_modes * n_rhs;

        __shared__ double r_shared;
        __shared__ double ct_shared;
        __shared__ double st_shared;
        __shared__ double phi_shared;
        __shared__ double re_j_shared[{n_orders}];
        __shared__ double im_j_shared[{n_orders}];
        __shared__ double p_pdm_shared[{n_p_pdm}];
        __shared__ double cos_mphi_shared[{n_phase}];
        __shared__ double sin_mphi_shared[{n_phase}];
        __shared__ double ct_pow_shared[{n_orders}];
        __shared__ double st_pow_shared[{n_orders}];
        __shared__ double partial_re[{reduction_threads}];
        __shared__ double partial_im[{reduction_threads}];

        for (int pair_idx = blockIdx.x; pair_idx < n_pairs; pair_idx += gridDim.x) {{
            const int leaf_row = pair_idx / occupancy;
            const int occ = pair_idx - leaf_row * occupancy;
            const int group_row = row_indices[leaf_row];
            const int group_pair_idx = group_row * occupancy + occ;

            if (threadIdx.x == 0) {{
                const double dx = pair_deltas[3 * group_pair_idx + 0];
                const double dy = pair_deltas[3 * group_pair_idx + 1];
                const double dz = pair_deltas[3 * group_pair_idx + 2];
                const double rr = sqrt(dx * dx + dy * dy + dz * dz);
                r_shared = rr;
                if (rr > 0.0) {{
                    ct_shared = dz / rr;
                    st_shared = sqrt(fmax(0.0, 1.0 - ct_shared * ct_shared));
                    phi_shared = atan2(dy, dx);
                    ct_pow_shared[0] = 1.0;
                    st_pow_shared[0] = 1.0;
                    for (int p = 1; p < {n_orders}; ++p) {{
                        ct_pow_shared[p] = ct_pow_shared[p - 1] * ct_shared;
                        st_pow_shared[p] = st_pow_shared[p - 1] * st_shared;
                    }}
                }}
            }}
            __syncthreads();

            if (r_shared > 0.0) {{
                for (int p = threadIdx.x; p < {n_orders}; p += blockDim.x) {{
                    const complex<double> radial =
                        bessel_lookup_linear_receive(p, r_shared, re_j, im_j, inv_dr, last_index);
                    re_j_shared[p] = radial.real();
                    im_j_shared[p] = radial.imag();
                    for (int absdm = 0; absdm <= p; ++absdm) {{
                        p_pdm_shared[p * (p + 1) / 2 + absdm] =
                            assoc_legendre_function_receive(
                                p, absdm, ct_pow_shared, st_pow_shared, plm_coeffs
                            );
                    }}
                }}
                if (threadIdx.x == 0) {{
                    for (int dm = -2 * {order}; dm <= 2 * {order}; ++dm) {{
                        const int phase_idx = dm + 2 * {order};
                        sincos((double)dm * phi_shared,
                               &sin_mphi_shared[phase_idx],
                               &cos_mphi_shared[phase_idx]);
                    }}
                }}
            }}
            __syncthreads();

            const int flat = tile_base + lane;
            double acc_re = 0.0;
            double acc_im = 0.0;
            int rhs = 0;
            int in_idx = 0;
            if (flat < n_outputs) {{
                rhs = flat % n_rhs;
                in_idx = flat / n_rhs;
                const int incoming_row = incoming_rows[leaf_row];
                for (int out_idx = warp; out_idx < n_out_modes; out_idx += n_warps) {{
                    double coeff_re = 0.0;
                    double coeff_im = 0.0;
                    if (r_shared <= 0.0) {{
                        if (out_mode_indices[out_idx] == in_mode_indices[in_idx]) {{
                            coeff_re = 1.0;
                        }}
                    }} else {{
                        const int delta_m = mode_m_in[in_idx] - mode_m_out[out_idx];
                        const int phase_idx = delta_m + 2 * {order};
                        const int table_idx = out_idx * n_in_modes + in_idx;
                        const int base = pair_offset[table_idx];
                        const int p_min = pair_pmin[table_idx];
                        const int p_count = pair_pcount[table_idx];
                        double re_sum = 0.0;
                        double im_sum = 0.0;
                        for (int ip = 0; ip < p_count; ++ip) {{
                            const int p = p_min + ip;
                            const int ab_idx = base + ip;
                            const double plm = p_pdm_shared[p * (p + 1) / 2 + abs(delta_m)];
                            const double re_abp = re_ab[ab_idx] * plm;
                            const double im_abp = im_ab[ab_idx] * plm;
                            const double re_abpr =
                                re_abp * re_j_shared[p] - im_abp * im_j_shared[p];
                            const double im_abpr =
                                re_abp * im_j_shared[p] + im_abp * re_j_shared[p];
                            const double re_phase =
                                re_abpr * cos_mphi_shared[phase_idx]
                                - im_abpr * sin_mphi_shared[phase_idx];
                            const double im_phase =
                                re_abpr * sin_mphi_shared[phase_idx]
                                + im_abpr * cos_mphi_shared[phase_idx];
                            re_sum += re_phase;
                            im_sum += im_phase;
                        }}
                        coeff_re = re_sum;
                        coeff_im = im_sum;
                    }}
                    const complex<double> x =
                        incoming[((long long)incoming_row * n_out_modes + out_idx) * n_rhs + rhs];
                    const double xr = x.real();
                    const double xi = x.imag();
                    acc_re += coeff_re * xr + coeff_im * xi;
                    acc_im += coeff_re * xi - coeff_im * xr;
                }}
            }}

            const int partial_idx = warp * 32 + lane;
            partial_re[partial_idx] = acc_re;
            partial_im[partial_idx] = acc_im;
            __syncthreads();

            if (warp == 0 && flat < n_outputs) {{
                double sum_re = 0.0;
                double sum_im = 0.0;
                for (int w = 0; w < n_warps; ++w) {{
                    const int idx = w * 32 + lane;
                    sum_re += partial_re[idx];
                    sum_im += partial_im[idx];
                }}
                const int particle = particle_indices[group_pair_idx];
                out[((long long)particle * n_in_modes + in_idx) * n_rhs + rhs] +=
                    complex<double>(sum_re, sum_im);
            }}
            __syncthreads();
        }}
    }}
    """
    return cupy.RawKernel(source, "mlfmm_leaf_otf_receive_fused")


def _launch_leaf_otf_receive_fused(
    *,
    pair_deltas: Any,
    particle_indices: Any,
    row_indices: Any,
    incoming_rows: Any,
    incoming: Any,
    out: Any,
    tables: CuPyLeafTranslationTablesData,
    occupancy: int,
    box_nm: int,
    nmodes: int,
    nrhs: int,
    cupy: Any,
) -> None:
    """Receive leaf box states without materializing the pair-block tensor."""

    count = int(row_indices.size)
    if count <= 0:
        return
    n_pairs = int(count * int(occupancy))
    n_outputs = int(nmodes) * int(nrhs)
    n_tiles = max(1, (n_outputs + 31) // 32)
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_pairs = int(props["maxGridSize"][0])
    max_threads = int(props["maxThreadsPerBlock"])
    shared_limit = int(props["sharedMemPerBlock"])
    threads = _leaf_otf_receive_threads(
        full_order=int(tables.full_order),
        max_threads_per_block=max_threads,
        shared_mem_per_block=shared_limit,
    )
    kernel = _leaf_otf_receive_fused_raw_kernel(int(tables.full_order), int(threads))
    blocks_x = max(1, min(max_grid_pairs, n_pairs))
    kernel(
        (blocks_x, n_tiles),
        (threads,),
        (
            np.int32(n_pairs),
            np.int32(occupancy),
            np.int32(box_nm),
            np.int32(nmodes),
            np.int32(nrhs),
            pair_deltas.reshape(-1),
            particle_indices.reshape(-1),
            cupy.asarray(row_indices, dtype=cupy.int32).reshape(-1),
            cupy.asarray(incoming_rows, dtype=cupy.int32).reshape(-1),
            incoming.reshape(-1),
            tables.out_mode_indices,
            tables.in_mode_indices,
            tables.mode_m_out,
            tables.mode_m_in,
            tables.re_j,
            tables.im_j,
            np.float64(float(tables.inv_dr)),
            np.int32(int(tables.last_index)),
            tables.plm_coeffs,
            tables.compact_re_ab,
            tables.compact_im_ab,
            tables.pair_offset,
            tables.pair_pmin,
            tables.pair_pcount,
            out.reshape(-1),
        ),
    )


def _leaf_otf_group_chunk_leaves(
    *,
    n_group: int,
    chunk_leaves: int | None,
    occupancy: int,
    box_nm: int,
    nmodes: int,
    nrhs: int,
    bytes_budget: int | None,
) -> int:
    """Choose grouped on-the-fly leaf chunk size from leaf units and live bytes."""

    if int(n_group) <= 0:
        return 1
    by_leaves = (
        int(n_group) if chunk_leaves is None else max(1, min(int(n_group), int(chunk_leaves)))
    )
    if bytes_budget is None:
        return int(by_leaves)
    pair_block_bytes_per_leaf = (
        int(occupancy) * int(box_nm) * int(nmodes) * np.dtype(np.complex128).itemsize
    )
    coeff_bytes_per_leaf = (
        int(occupancy) * int(nmodes) * int(nrhs) * np.dtype(np.complex128).itemsize
    )
    out_bytes_per_leaf = int(box_nm) * int(nrhs) * np.dtype(np.complex128).itemsize
    live_bytes_per_leaf = int(pair_block_bytes_per_leaf + coeff_bytes_per_leaf + out_bytes_per_leaf)
    by_bytes = max(1, int(bytes_budget) // max(1, live_bytes_per_leaf))
    return max(1, min(int(n_group), int(by_leaves), int(by_bytes)))


def _leaf_otf_fused_aggregate_chunk_leaves(
    *,
    n_group: int,
    chunk_leaves: int | None,
) -> int:
    """Choose leaf chunks for fused aggregation, which has no pair-block tensor."""

    if int(n_group) <= 0:
        return 1
    if chunk_leaves is None:
        return int(n_group)
    return max(1, min(int(n_group), int(chunk_leaves)))


def _leaf_otf_resolved_chunk_summary(
    *,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_otf_chunk_leaves: int | None,
) -> dict[str, object] | None:
    """Summarize fused multilevel leaf OTF chunk sizes in native leaf units."""

    if not leaf_groups:
        return None
    resolved_by_occupancy: dict[str, int] = {}
    chunk_values: list[int] = []
    for group in leaf_groups:
        n_group = int(group.leaf_ids.shape[0])
        if n_group <= 0:
            continue
        resolved_chunk = _leaf_otf_fused_aggregate_chunk_leaves(
            n_group=n_group,
            chunk_leaves=leaf_otf_chunk_leaves,
        )
        resolved_by_occupancy[str(int(group.occupancy))] = int(resolved_chunk)
        chunk_values.append(int(resolved_chunk))
    if not chunk_values:
        return None
    return {
        "by_occupancy": resolved_by_occupancy,
        "min": int(min(chunk_values)),
        "max": int(max(chunk_values)),
    }


def _matched_sorted_rows(
    *,
    container_ids: Any,
    selected_ids: Any,
    cupy: Any,
) -> tuple[Any, Any] | None:
    """Return `(selected_rows, container_rows)` for sorted integer ids."""

    selected = cupy.asarray(selected_ids, dtype=cupy.int32).reshape(-1)
    container = cupy.asarray(container_ids, dtype=cupy.int32).reshape(-1)
    if int(selected.size) == 0 or int(container.size) == 0:
        return None
    positions = cupy.searchsorted(container, selected)
    max_index = int(container.size) - 1
    clamped = cupy.minimum(positions, max_index)
    valid = (positions < int(container.size)) & (container[clamped] == selected)
    selected_rows = cupy.nonzero(valid)[0].astype(cupy.int32, copy=False)
    if int(selected_rows.size) == 0:
        return None
    return selected_rows, positions[selected_rows].astype(cupy.int32, copy=False)


def _aggregate_selected_leaf_box_states(
    x_states: Any,
    *,
    selected_leaf_ids: Any,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    leaf_otf_chunk_leaves: int | None,
    box_nm: int,
    nrhs: int,
    cupy: Any,
) -> Any:
    """Aggregate one selected leaf subset into regular box states."""

    selected_ids = cupy.asarray(selected_leaf_ids, dtype=cupy.int32).reshape(-1)
    n_selected = int(selected_ids.size)
    out = cupy.zeros((n_selected, int(box_nm), int(nrhs)), dtype=cupy.complex128)
    if n_selected == 0:
        return out
    for group in leaf_groups:
        matched = _matched_sorted_rows(
            container_ids=group.leaf_ids,
            selected_ids=selected_ids,
            cupy=cupy,
        )
        if matched is None:
            continue
        selected_rows, group_rows = matched
        occupancy = int(group.occupancy)
        nmodes = int(group.nmodes)
        n_group_rows = int(group_rows.size)
        if str(leaf_apply_mode) == "on_the_fly":
            if leaf_translation_tables is None or group.pair_deltas is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: on-the-fly leaf aggregation requires translation "
                    "tables and pair-delta schedules."
                )
            chunk_leaves = _leaf_otf_fused_aggregate_chunk_leaves(
                n_group=n_group_rows,
                chunk_leaves=leaf_otf_chunk_leaves,
            )
            for start in range(0, n_group_rows, chunk_leaves):
                end = min(n_group_rows, start + chunk_leaves)
                _launch_leaf_otf_aggregate_fused(
                    pair_deltas=group.pair_deltas,
                    particle_indices=group.particle_indices,
                    row_indices=group_rows[start:end],
                    target_rows=selected_rows[start:end],
                    x_states=x_states,
                    out=out,
                    tables=leaf_translation_tables,
                    occupancy=occupancy,
                    box_nm=int(box_nm),
                    nmodes=nmodes,
                    nrhs=int(nrhs),
                    cupy=cupy,
                )
        else:
            if group.aggregation is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: dense leaf aggregation mode requires aggregation tensors."
                )
            coeffs = x_states[group.particle_indices[group_rows]].reshape(
                n_group_rows,
                occupancy * nmodes,
                int(nrhs),
            )
            out[selected_rows] = cupy.matmul(group.aggregation[group_rows], coeffs)
    return out


def _receive_selected_leaf_boxes_to_particles(
    incoming_box: Any,
    *,
    selected_leaf_ids: Any,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    leaf_otf_chunk_leaves: int | None,
    receive_adjoint_cache: dict[int, Any] | None,
    nm: int,
    out: Any,
    cupy: Any,
    level_idx: int | None = None,
    stream_stats: dict[str, object] | None = None,
) -> None:
    """Accumulate one selected leaf subset from regular box states to particles."""

    receive_started = time.perf_counter() if stream_stats is not None else 0.0
    selected_ids = cupy.asarray(selected_leaf_ids, dtype=cupy.int32).reshape(-1)
    incoming = cupy.asarray(incoming_box, dtype=cupy.complex128)
    if int(selected_ids.size) != int(incoming.shape[0]):
        raise ValueError(
            "Selected leaf receive row count mismatch: "
            f"{int(selected_ids.size)} vs {int(incoming.shape[0])}."
        )
    y = cupy.asarray(out, dtype=cupy.complex128)
    timing_level = int(level_idx) if level_idx is not None else 0
    for group in leaf_groups:
        matched = _matched_sorted_rows(
            container_ids=group.leaf_ids,
            selected_ids=selected_ids,
            cupy=cupy,
        )
        if matched is None:
            continue
        selected_rows, group_rows = matched
        occupancy = int(group.occupancy)
        n_group_rows = int(group_rows.size)
        idx_rows = group.particle_indices[group_rows]
        if str(leaf_apply_mode) == "on_the_fly":
            if leaf_translation_tables is None or group.pair_deltas is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: on-the-fly leaf receive requires translation "
                    "tables and pair-delta schedules."
                )
            chunk_leaves = _leaf_otf_fused_aggregate_chunk_leaves(
                n_group=n_group_rows,
                chunk_leaves=leaf_otf_chunk_leaves,
            )
            for start in range(0, n_group_rows, chunk_leaves):
                end = min(n_group_rows, start + chunk_leaves)
                count = int(end - start)
                if level_idx is not None:
                    _record_leaf_receive_chunk_stats(
                        stream_stats,
                        level_idx=int(level_idx),
                        leaves=count,
                        pair_block_bytes=0,
                    )
                fused_started = time.perf_counter() if stream_stats is not None else 0.0
                _launch_leaf_otf_receive_fused(
                    pair_deltas=group.pair_deltas,
                    particle_indices=group.particle_indices,
                    row_indices=group_rows[start:end],
                    incoming_rows=selected_rows[start:end],
                    incoming=incoming,
                    out=y,
                    tables=leaf_translation_tables,
                    occupancy=occupancy,
                    box_nm=int(leaf_translation_tables.nmodes_out),
                    nmodes=int(nm),
                    nrhs=int(incoming.shape[2]),
                    cupy=cupy,
                )
                _accumulate_stream_seconds(
                    stream_stats,
                    level_idx=timing_level,
                    key="leaf_receive_fused",
                    seconds=time.perf_counter() - fused_started,
                )
        else:
            if group.aggregation is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: dense leaf receive mode requires aggregation tensors."
                )
            agg_group = group.aggregation[group_rows]
            if receive_adjoint_cache is not None:
                cache_key = int(agg_group.data.ptr)
                receive_adj = receive_adjoint_cache.get(cache_key)
                if receive_adj is None:
                    receive_adj = cupy.swapaxes(agg_group, 1, 2).conj()
                    receive_adjoint_cache[cache_key] = receive_adj
            else:
                receive_adj = cupy.swapaxes(agg_group, 1, 2).conj()
            contribution = cupy.matmul(
                receive_adj,
                incoming[selected_rows],
            ).reshape(n_group_rows, occupancy, int(nm), int(incoming.shape[2]))
            y[idx_rows] += contribution
    _accumulate_stream_seconds(
        stream_stats,
        level_idx=timing_level,
        key="leaf_receive_total",
        seconds=time.perf_counter() - receive_started,
    )


def _aggregate_leaf_box_states(
    x_states: Any,
    *,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    n_leaves: int,
    box_nm: int,
    nrhs: int,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Aggregate particle coefficients into one outgoing box state per leaf."""

    box_states = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.zeros((n_leaves, int(box_nm), int(nrhs)), dtype=cupy.complex128)
    )
    box_states.fill(0)
    for group in leaf_groups:
        idx = group.particle_indices
        n_group = int(idx.shape[0])
        occupancy = int(group.occupancy)
        nmodes = int(group.nmodes)
        if str(leaf_apply_mode) == "on_the_fly":
            if leaf_translation_tables is None or group.pair_deltas is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: on-the-fly leaf aggregation requires translation "
                    "tables and pair-delta schedules."
                )
            chunk_leaves = _leaf_otf_fused_aggregate_chunk_leaves(
                n_group=n_group,
                chunk_leaves=leaf_otf_chunk_leaves,
            )
            row_indices = cupy.arange(n_group, dtype=cupy.int32)
            for start in range(0, n_group, chunk_leaves):
                end = min(n_group, start + chunk_leaves)
                _launch_leaf_otf_aggregate_fused(
                    pair_deltas=group.pair_deltas,
                    particle_indices=group.particle_indices,
                    row_indices=row_indices[start:end],
                    target_rows=group.leaf_ids[start:end],
                    x_states=x_states,
                    out=box_states,
                    tables=leaf_translation_tables,
                    occupancy=occupancy,
                    box_nm=int(box_nm),
                    nmodes=nmodes,
                    nrhs=int(nrhs),
                    cupy=cupy,
                )
        else:
            if group.aggregation is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: dense leaf aggregation mode requires aggregation tensors."
                )
            coeffs = x_states[idx].reshape(n_group, occupancy * nmodes, int(nrhs))
            box_states[group.leaf_ids] = cupy.matmul(group.aggregation, coeffs)
    return box_states


def _receive_leaf_boxes_to_particles(
    incoming_box: Any,
    *,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    receive_adjoint_cache: dict[int, Any] | None,
    nm: int,
    n_particles: int,
    nrhs: int,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Scatter leaf-local incoming box states back to particle coefficients.

    When provided, `receive_adjoint_cache` stores one conjugate-transposed
    grouped aggregation tensor per leaf group so repeated applies do not
    recompute `swapaxes(...).conj()` in the hot loop.
    """

    y = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.zeros((int(n_particles), int(nm), int(nrhs)), dtype=cupy.complex128)
    )
    y.fill(0)
    for group in leaf_groups:
        leaf_ids = group.leaf_ids
        idx = group.particle_indices
        n_group = int(idx.shape[0])
        occupancy = int(group.occupancy)
        if str(leaf_apply_mode) == "on_the_fly":
            if leaf_translation_tables is None or group.pair_deltas is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: on-the-fly leaf receive requires translation "
                    "tables and pair-delta schedules."
                )
            chunk_leaves = _leaf_otf_group_chunk_leaves(
                n_group=n_group,
                chunk_leaves=leaf_otf_chunk_leaves,
                occupancy=occupancy,
                box_nm=int(leaf_translation_tables.nmodes_out),
                nmodes=int(nm),
                nrhs=int(nrhs),
                bytes_budget=leaf_otf_bytes_budget,
            )
            for start in range(0, n_group, chunk_leaves):
                end = min(n_group, start + chunk_leaves)
                count = int(end - start)
                pair_start = int(start * occupancy)
                pair_end = int(end * occupancy)
                pair_blocks = _leaf_translation_blocks_from_pair_deltas(
                    group.pair_deltas[pair_start:pair_end],
                    tables=leaf_translation_tables,
                    pair_blocks_scratch=pair_blocks_scratch,
                    cupy=cupy,
                ).reshape(count, occupancy, int(leaf_translation_tables.nmodes_out), int(nm))
                _launch_leaf_otf_receive_contract(
                    pair_blocks,
                    incoming_box,
                    leaf_ids[start:end],
                    idx[start:end],
                    y,
                    occupancy=occupancy,
                    box_nm=int(leaf_translation_tables.nmodes_out),
                    nmodes=int(nm),
                    nrhs=int(nrhs),
                    cupy=cupy,
                )
        else:
            if group.aggregation is None:
                raise RuntimeError(
                    "Internal CuPy MLFMM error: dense leaf receive mode requires aggregation tensors."
                )
            if receive_adjoint_cache is not None:
                cache_key = int(group.aggregation.data.ptr)
                receive_adj = receive_adjoint_cache.get(cache_key)
                if receive_adj is None:
                    receive_adj = cupy.swapaxes(group.aggregation, 1, 2).conj()
                    receive_adjoint_cache[cache_key] = receive_adj
            else:
                receive_adj = cupy.swapaxes(group.aggregation, 1, 2).conj()
            contribution = cupy.matmul(receive_adj, incoming_box[leaf_ids]).reshape(
                n_group, occupancy, int(nm), int(nrhs)
            )
            y[idx] += contribution
    return y


def _ensure_single_level_workspace(
    prepared: CuPyMLFMMPreparedData,
    *,
    n_particles: int,
    nm: int,
    nrhs: int,
    cache: dict[CuPyMLFMMSingleLevelWorkspaceKey, CuPyMLFMMSingleLevelWorkspace],
    cupy: Any,
) -> CuPyMLFMMSingleLevelWorkspace:
    """Return reusable single-level far workspace keyed by runtime shape metadata."""

    single = prepared.single_level
    if single is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing single-level prepared data.")
    key = CuPyMLFMMSingleLevelWorkspaceKey(
        nrhs=int(nrhs),
        n_particles=int(n_particles),
        nm=int(nm),
        n_leaves=int(single.n_leaves),
        box_nm=int(single.box_nm),
        n_directions=int(single.directional.grid.n_directions),
    )
    ws = cache.get(key)
    if ws is not None:
        return ws
    ws = CuPyMLFMMSingleLevelWorkspace(
        nrhs=int(key.nrhs),
        box_states=cupy.empty(
            (int(key.n_leaves), int(key.box_nm), int(key.nrhs)), dtype=cupy.complex128
        ),
        outgoing=cupy.empty(
            (int(key.n_leaves), 4, int(key.n_directions), int(key.nrhs)),
            dtype=cupy.complex128,
        ),
        incoming=cupy.empty(
            (int(key.n_leaves), 4, int(key.n_directions), int(key.nrhs)),
            dtype=cupy.complex128,
        ),
        incoming_box=cupy.empty(
            (int(key.n_leaves), int(key.box_nm), int(key.nrhs)), dtype=cupy.complex128
        ),
        y_states=cupy.empty(
            (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy.complex128
        ),
    )
    cache[key] = ws
    return ws


def _ensure_multilevel_workspace(
    prepared: CuPyMLFMMPreparedData,
    *,
    n_particles: int,
    nm: int,
    nrhs: int,
    cache: dict[CuPyMLFMMMultilevelWorkspaceKey, CuPyMLFMMMultilevelWorkspace],
    cupy: Any,
) -> CuPyMLFMMMultilevelWorkspace:
    """Return reusable multilevel far workspace keyed by runtime shape metadata."""

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    key = CuPyMLFMMMultilevelWorkspaceKey(
        nrhs=int(nrhs),
        n_particles=int(n_particles),
        nm=int(nm),
    )
    ws = cache.get(key)
    if ws is not None:
        return ws
    if str(multilevel.leaf_apply_mode) == "on_the_fly":
        ws = CuPyMLFMMMultilevelWorkspace(
            nrhs=int(key.nrhs),
            y_states=cupy.empty(
                (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy.complex128
            ),
        )
    else:
        levels = multilevel.levels
        hf_start = int(multilevel.hf_start_level)
        hf_end = int(multilevel.hf_end_level)
        max_elements_by_parity = [1, 1]
        for level_idx in range(hf_start, hf_end + 1):
            level = levels[level_idx]
            parity = int((level_idx - hf_start) & 1)
            level_elements = (
                int(level.n_boxes) * 4 * int(level.directional.grid.n_directions) * int(key.nrhs)
            )
            max_elements_by_parity[parity] = max(
                max_elements_by_parity[parity], int(level_elements)
            )
        ws = CuPyMLFMMMultilevelWorkspace(
            nrhs=int(key.nrhs),
            y_states=cupy.empty(
                (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy.complex128
            ),
            outgoing_roll_even=cupy.empty((int(max_elements_by_parity[0]),), dtype=cupy.complex128),
            outgoing_roll_odd=cupy.empty((int(max_elements_by_parity[1]),), dtype=cupy.complex128),
            incoming_roll_even=cupy.empty((int(max_elements_by_parity[0]),), dtype=cupy.complex128),
            incoming_roll_odd=cupy.empty((int(max_elements_by_parity[1]),), dtype=cupy.complex128),
            leaf_box_states=cupy.empty(
                (int(multilevel.n_leaves), int(multilevel.box_nm), int(key.nrhs)),
                dtype=cupy.complex128,
            ),
            incoming_box=cupy.empty(
                (int(multilevel.n_leaves), int(multilevel.box_nm), int(key.nrhs)),
                dtype=cupy.complex128,
            ),
        )
    cache[key] = ws
    return ws


def _ensure_exact_near_workspace(
    prepared: CuPyMLFMMPreparedData,
    *,
    n_particles: int,
    nm: int,
    nrhs: int,
    near_dtype: np.dtype,
    cache: dict[CuPyMLFMMNearWorkspaceKey, CuPyMLFMMNearWorkspace],
    cupy: Any,
) -> CuPyMLFMMNearWorkspace:
    """Return reusable exact-near workspace keyed by runtime shape and near dtype."""

    near = prepared.near_pairs
    key = CuPyMLFMMNearWorkspaceKey(
        nrhs=int(nrhs),
        n_particles=int(n_particles),
        nm=int(nm),
        n_leaf_pairs=int(near.dst_leaf_indices.size),
        near_dtype=str(np.dtype(near_dtype).str),
    )
    ws = cache.get(key)
    if ws is not None:
        return ws
    if np.dtype(near_dtype) == np.dtype(np.complex64):
        cupy_near_dtype = cupy.complex64
    elif np.dtype(near_dtype) == np.dtype(np.complex128):
        cupy_near_dtype = cupy.complex128
    else:
        raise ValueError(
            "CuPy MLFMM near workspace supports only complex64/complex128 near dtypes. "
            f"Got {np.dtype(near_dtype)!r}."
        )
    ws = CuPyMLFMMNearWorkspace(
        nrhs=int(key.nrhs),
        y_states=cupy.empty(
            (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy_near_dtype
        ),
    )
    cache[key] = ws
    return ws


def _exact_leaf_launch_context(
    prepared: CuPyMLFMMPreparedData,
    *,
    cupy: Any,
) -> _ExactLeafLaunchContext:
    """Resolve one exact-leaf kernel and launch geometry per matvec."""

    near_dtype = np.dtype(prepared.near_pairs.near_dtype)
    if near_dtype == np.dtype(np.complex64):
        real_scalar_type: type[np.floating[Any]] = np.float32
    elif near_dtype == np.dtype(np.complex128):
        real_scalar_type = np.float64
    else:
        raise ValueError(
            "CuPy MLFMM exact leaf path supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    warp_size = int(props["warpSize"])
    max_threads = int(props["maxThreadsPerBlock"])
    nm = int(n_modes(int(prepared.lmax)))
    return _ExactLeafLaunchContext(
        kernel=_exact_leaf_pairs_raw_kernel(int(prepared.lmax), near_dtype.str),
        real_scalar_type=real_scalar_type,
        threads=max(warp_size, min(max_threads, nm)),
        max_grid_y=int(props["maxGridSize"][1]),
        max_grid_z=int(props["maxGridSize"][2]),
    )


def _launch_exact_leaf_pairs(
    prepared: CuPyMLFMMPreparedData,
    x_arr: Any,
    y_arr: Any,
    *,
    destination_leaf_indices: Any,
    source_leaf_indices: Any,
    skip_identity: bool,
    launch: _ExactLeafLaunchContext,
) -> None:
    """Launch one exact central leaf-pair batch."""

    nm = int(x_arr.shape[1])
    nrhs = int(x_arr.shape[2])
    n_leaf_pairs = int(destination_leaf_indices.size)
    if n_leaf_pairs == 0:
        return

    grid_y = min(n_leaf_pairs, int(launch.max_grid_y))
    blocks_x = max(1, (nm + int(launch.threads) - 1) // int(launch.threads))
    grid_z = min(max(1, nrhs), int(launch.max_grid_z))
    scalar = launch.real_scalar_type
    near = prepared.near_pairs

    args = (
        np.int32(n_leaf_pairs),
        np.int32(nm),
        np.int32(nrhs),
        near.positions,
        np.int32(1 if skip_identity else 0),
        destination_leaf_indices,
        source_leaf_indices,
        near.leaf_particle_offsets,
        near.leaf_particle_indices,
        near.lut_re,
        near.lut_im,
        scalar(float(near.inv_dr)),
        np.int32(int(near.last_index)),
        near.plm_coeffs,
        near.compact_re_ab,
        near.compact_im_ab,
        near.mode_m,
        near.pair_offset,
        near.pair_pmin,
        near.pair_pcount,
        x_arr.reshape(-1),
        y_arr.reshape(-1),
    )
    launch.kernel(
        (int(blocks_x), int(grid_y), int(grid_z)),
        (int(launch.threads),),
        args,
    )


def _apply_exact_near_pairs(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    workspace: CuPyMLFMMNearWorkspace | None,
    cupy: Any,
) -> Any:
    """Apply exact central leaf interactions on device."""

    near = prepared.near_pairs
    near_dtype = np.dtype(near.near_dtype)
    if near_dtype == np.dtype(np.complex64):
        cupy_out_dtype = cupy.complex64
    elif near_dtype == np.dtype(np.complex128):
        cupy_out_dtype = cupy.complex128
    else:
        raise ValueError(
            "CuPy MLFMM near path supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )

    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    x_arr = cupy.ascontiguousarray(cupy.asarray(x_states, dtype=cupy_out_dtype))
    if workspace is None:
        y_arr = cupy.zeros((n_particles, nm, nrhs), dtype=cupy_out_dtype)
    else:
        y_arr = workspace.y_states
        y_arr.fill(0)
    launch = _exact_leaf_launch_context(prepared, cupy=cupy)
    _launch_exact_leaf_pairs(
        prepared,
        x_arr,
        y_arr,
        destination_leaf_indices=near.dst_leaf_indices,
        source_leaf_indices=near.src_leaf_indices,
        skip_identity=True,
        launch=launch,
    )
    return y_arr


def _build_exact_near_adjoint_context(
    prepared: CuPyMLFMMPreparedData,
    *,
    lmax: int,
    k: float,
    near_dtype: np.dtype,
    cupy: Any,
) -> _CuPyMLFMMNearAdjointContext:
    """Extract compact host ingredients for the exact-near reverse action."""

    near = prepared.near_pairs
    positions = np.asarray(cupy.asnumpy(near.positions), dtype=float).reshape(-1, 3)
    offsets = np.asarray(cupy.asnumpy(near.leaf_particle_offsets), dtype=np.int32)
    indices = np.asarray(cupy.asnumpy(near.leaf_particle_indices), dtype=np.int32)
    dst_leaves = np.asarray(cupy.asnumpy(near.dst_leaf_indices), dtype=np.int32)
    src_leaves = np.asarray(cupy.asnumpy(near.src_leaf_indices), dtype=np.int32)
    # Reconstruct the exact radial lookup object from its uploaded row-major
    # real/imaginary table. The CuPy forward kernel uses linear interpolation.
    dr = 1.0 / float(near.inv_dr)
    radial_lut = RadialLUT(
        lmax=int(lmax),
        k=float(k),
        r_max=float(near.last_index) * dr,
        dr=dr,
        dtype=np.dtype(near_dtype),
    )
    n_orders = 2 * int(lmax) + 1
    radial_lut.h = np.asarray(
        np.asarray(cupy.asnumpy(near.lut_re)).reshape(-1, n_orders)
        + 1j * np.asarray(cupy.asnumpy(near.lut_im)).reshape(-1, n_orders),
        dtype=np.dtype(near_dtype),
    ).T
    # Match the uploaded table exactly; the constructor's guard row is not
    # part of the device lookup payload and must not be addressed here.
    radial_lut.r_grid = dr * np.arange(int(near.last_index) + 1, dtype=float)
    radial_lut._last_index = int(near.last_index)
    radial_lut._inv_dr = float(near.inv_dr)
    ab5 = translation_ab5_table(int(lmax), dtype=np.complex128)
    nm = int(n_modes(int(lmax)))
    return _CuPyMLFMMNearAdjointContext(
        positions=positions,
        leaf_particle_offsets=offsets,
        leaf_particle_indices=indices,
        dst_leaf_indices=dst_leaves,
        src_leaf_indices=src_leaves,
        radial_lut=radial_lut,
        ab5=ab5,
        lmax=int(lmax),
        k=float(k),
        near_dtype=np.dtype(near_dtype),
        nm=nm,
    )


def _build_exact_near_adjoint_block(
    context: _CuPyMLFMMNearAdjointContext,
    pair_index: int,
    *,
    cupy: Any,
) -> tuple[Any, Any, Any]:
    """Build and upload one directed exact-near leaf-pair block."""

    dst_leaf = int(context.dst_leaf_indices[int(pair_index)])
    src_leaf = int(context.src_leaf_indices[int(pair_index)])
    offsets = context.leaf_particle_offsets
    indices = context.leaf_particle_indices
    dst_ids = indices[int(offsets[dst_leaf]) : int(offsets[dst_leaf + 1])]
    src_ids = indices[int(offsets[src_leaf]) : int(offsets[src_leaf + 1])]
    blocks = np.empty(
        (dst_ids.size, src_ids.size, context.nm, context.nm), dtype=context.near_dtype
    )
    for dst_row, dst_particle in enumerate(dst_ids.tolist()):
        for src_col, src_particle in enumerate(src_ids.tolist()):
            if int(dst_particle) == int(src_particle):
                blocks[dst_row, src_col].fill(0)
                continue
            blocks[dst_row, src_col] = np.asarray(
                translation_block(
                    int(context.lmax),
                    float(context.k),
                    context.positions[int(dst_particle)] - context.positions[int(src_particle)],
                    ab5=context.ab5,
                    radial_lut=context.radial_lut,
                ),
                dtype=context.near_dtype,
            )
    # The reverse contraction consumes conjugated W blocks.  Conjugate once
    # on the compact host block before upload instead of allocating a dense
    # device-sized conjugation temporary on every adjoint action.
    np.conjugate(blocks, out=blocks)
    return (
        cupy.asarray(dst_ids, dtype=cupy.int32),
        cupy.asarray(src_ids, dtype=cupy.int32),
        cupy.asarray(blocks, dtype=_cupy_complex_dtype(context.near_dtype, cupy=cupy)),
    )


def _exact_near_adjoint_block_bytes(block: tuple[Any, Any, Any]) -> int:
    """Return resident bytes for one uploaded reverse near block."""

    return int(sum(_device_array_nbytes(value) for value in block))


def _apply_exact_near_adjoint_streaming(
    context: _CuPyMLFMMNearAdjointContext,
    cached_blocks: dict[int, tuple[Any, Any, Any]],
    x_states: Any,
    *,
    out: Any,
    cache_bytes: int,
    cache_budget: int | None,
    cupy: Any,
) -> tuple[Any, int]:
    """Apply exact-near reverse blocks with a bounded resident cache."""

    out.fill(0)
    n_pairs = int(context.dst_leaf_indices.size)
    for pair_index in range(n_pairs):
        block = cached_blocks.get(pair_index)
        if block is None:
            block = _build_exact_near_adjoint_block(context, pair_index, cupy=cupy)
            block_bytes = _exact_near_adjoint_block_bytes(block)
            if cache_budget is None or cache_bytes + block_bytes <= int(cache_budget):
                cached_blocks[pair_index] = block
                cache_bytes += block_bytes
        dst_ids, src_ids, pair_blocks = block
        if int(dst_ids.size) == 0 or int(src_ids.size) == 0:
            continue
        # The cache stores conjugated ``W[dst_mode, src_mode]``.  Contracting
        # with the destination state produces the source state.
        contribution = cupy.einsum("abij,air->bjr", pair_blocks, x_states[dst_ids], optimize=True)
        out[src_ids] += contribution
    return out, int(cache_bytes)


def _apply_single_level_far(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    workspace: CuPyMLFMMSingleLevelWorkspace | None,
    cupy: Any,
) -> Any:
    """Apply sampled single-level far interactions on device.

    Grouped far-offset accumulation uses weighted fused kernels to avoid
    intermediate `translated` tensors in the hot loop.
    """

    single = prepared.single_level
    if single is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing single-level prepared data.")
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    ws = workspace
    box_nm = int(single.box_nm)
    box_states = _aggregate_leaf_box_states(
        x_states,
        leaf_groups=single.leaf_groups,
        leaf_apply_mode=single.leaf_apply_mode,
        leaf_translation_tables=single.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        n_leaves=int(single.n_leaves),
        box_nm=box_nm,
        nrhs=nrhs,
        out=(ws.box_states if ws is not None else None),
        cupy=cupy,
    )
    outgoing = _box_outgoing_to_directional_cupy(
        single.directional,
        box_states,
        out=(ws.outgoing if ws is not None else None),
        cupy=cupy,
    )
    incoming = ws.incoming if ws is not None else cupy.zeros_like(outgoing, dtype=cupy.complex128)
    incoming.fill(0)
    for offset, batch in single.far_offset_batches.items():
        _weighted_gather_add_complex128(
            incoming,
            batch.dst_indices,
            outgoing,
            batch.src_indices,
            single.offset_diagonals[offset],
            cupy=cupy,
        )
    incoming_box = _directional_to_box_regular_cupy(
        single.directional,
        incoming,
        out=(ws.incoming_box if ws is not None else None),
        cupy=cupy,
    )
    return _receive_leaf_boxes_to_particles(
        incoming_box,
        leaf_groups=single.leaf_groups,
        leaf_apply_mode=single.leaf_apply_mode,
        leaf_translation_tables=single.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        out=(ws.y_states if ws is not None else None),
        cupy=cupy,
    )


def _apply_single_level_far_adjoint(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    map_adjoint_cache: dict[int, Any] | None,
    cupy: Any,
) -> Any:
    """Apply the Hermitian adjoint of the single-level sampled far map."""

    single = prepared.single_level
    if single is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing single-level prepared data.")
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    box_states = _aggregate_leaf_box_states(
        x_states,
        leaf_groups=single.leaf_groups,
        leaf_apply_mode=single.leaf_apply_mode,
        leaf_translation_tables=single.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        n_leaves=int(single.n_leaves),
        box_nm=int(single.box_nm),
        nrhs=nrhs,
        cupy=cupy,
    )
    incoming = _directional_to_box_regular_adjoint_cupy(single.directional, box_states, cupy=cupy)
    outgoing = cupy.zeros_like(incoming, dtype=cupy.complex128)
    for offset, batch in single.far_offset_batches.items():
        _weighted_gather_add_complex128(
            outgoing,
            batch.src_indices,
            incoming,
            batch.dst_indices,
            cupy.conjugate(single.offset_diagonals[offset]),
            cupy=cupy,
        )
    outgoing_box = _box_outgoing_to_directional_adjoint_cupy(
        single.directional, outgoing, cupy=cupy
    )
    return _receive_leaf_boxes_to_particles(
        outgoing_box,
        leaf_groups=single.leaf_groups,
        leaf_apply_mode=single.leaf_apply_mode,
        leaf_translation_tables=single.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        cupy=cupy,
    )


def _apply_multilevel_far_adjoint(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    map_adjoint_cache: dict[int, Any] | None,
    cupy: Any,
) -> Any:
    """Apply the Hermitian adjoint of the resident multilevel sampled far map."""

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    leaf_level = int(multilevel.leaf_level)
    leaf_box_states = _aggregate_leaf_box_states(
        x_states,
        leaf_groups=multilevel.leaf_groups,
        leaf_apply_mode=multilevel.leaf_apply_mode,
        leaf_translation_tables=multilevel.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        n_leaves=int(multilevel.n_leaves),
        box_nm=int(multilevel.box_nm),
        nrhs=nrhs,
        cupy=cupy,
    )
    levels = multilevel.levels
    incoming = [
        cupy.zeros(
            (int(level.n_boxes), 4, int(level.directional.grid.n_directions), nrhs),
            dtype=cupy.complex128,
        )
        for level in levels
    ]
    incoming[leaf_level] = _directional_to_box_regular_adjoint_cupy(
        levels[leaf_level].directional, leaf_box_states, cupy=cupy
    )
    # Reverse the forward downward transfer, propagating leaf adjoints toward
    # the coarser sampled levels.
    for transfer in reversed(multilevel.transfers):
        parent_adj = incoming[int(transfer.parent_level)]
        child_adj = incoming[int(transfer.child_level)]
        for shift, batch in transfer.batches_by_shift.items():
            mapped = _apply_directional_map_adjoint(
                child_adj[batch.src_indices],
                transfer.map_down,
                cupy=cupy,
                cache=map_adjoint_cache,
            )
            values = (
                mapped * cupy.conjugate(transfer.phase_down_by_shift[shift])[None, None, :, None]
            )
            _add_at_complex128(parent_adj, batch.dst_indices, values, cupy=cupy)

    outgoing = [cupy.zeros_like(values, dtype=cupy.complex128) for values in incoming]
    for level_idx in range(int(multilevel.hf_end_level), int(multilevel.hf_start_level) - 1, -1):
        level = levels[level_idx]
        for offset, batch in level.far_offset_batches.items():
            _weighted_gather_add_complex128(
                outgoing[level_idx],
                batch.src_indices,
                incoming[level_idx],
                batch.dst_indices,
                cupy.conjugate(level.offset_diagonals[offset]),
                cupy=cupy,
            )

    # Reverse the forward upward transfer, now propagating parent adjoints to
    # child outgoing states. The map adjoint handles dense, sparse, and packed
    # interpolation storage uniformly.
    for transfer in multilevel.transfers:
        parent_adj = outgoing[int(transfer.parent_level)]
        child_adj = outgoing[int(transfer.child_level)]
        for shift, batch in transfer.batches_by_shift.items():
            shifted = (
                parent_adj[batch.dst_indices]
                * cupy.conjugate(transfer.phase_up_by_shift[shift])[None, None, :, None]
            )
            mapped = _apply_directional_map_adjoint(
                shifted,
                transfer.map_up,
                cupy=cupy,
                cache=map_adjoint_cache,
            )
            _add_at_complex128(child_adj, batch.src_indices, mapped, cupy=cupy)

    outgoing_box = _box_outgoing_to_directional_adjoint_cupy(
        levels[leaf_level].directional,
        outgoing[leaf_level],
        cupy=cupy,
    )
    return _receive_leaf_boxes_to_particles(
        outgoing_box,
        leaf_groups=multilevel.leaf_groups,
        leaf_apply_mode=multilevel.leaf_apply_mode,
        leaf_translation_tables=multilevel.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        cupy=cupy,
    )


def _multilevel_incoming_roll_view(
    *,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    hf_start: int,
    level_idx: int,
    level: CuPyMLFMMLevelData,
    nrhs: int,
    cupy: Any,
) -> Any:
    """Return one reusable rolling incoming buffer view for one hierarchy level."""

    n_boxes = int(level.n_boxes)
    n_dirs = int(level.directional.grid.n_directions)
    n_rhs = int(nrhs)
    required = int(n_boxes * 4 * n_dirs * n_rhs)
    if workspace is None:
        return cupy.zeros((n_boxes, 4, n_dirs, n_rhs), dtype=cupy.complex128)
    parity = int((int(level_idx) - int(hf_start)) & 1)
    backing = workspace.incoming_roll_even if parity == 0 else workspace.incoming_roll_odd
    if backing is None:
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling incoming backing buffer is missing for "
            f"level {int(level_idx)}."
        )
    if int(backing.size) < required:
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling incoming buffer is undersized for "
            f"level {int(level_idx)} (need elements={required}; "
            f"have elements={int(backing.size)})."
        )
    # Fused transfer/gather kernels flatten directional tensors; reshaping from a
    # 1D contiguous arena guarantees each per-level view is contiguous.
    view = backing[:required].reshape(n_boxes, 4, n_dirs, n_rhs)
    if not bool(view.flags.c_contiguous):
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling incoming view must be contiguous. "
            f"Got shape={tuple(int(v) for v in view.shape)}."
        )
    view.fill(0)
    return view


def _multilevel_outgoing_roll_view(
    *,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    hf_start: int,
    level_idx: int,
    level: CuPyMLFMMLevelData,
    nrhs: int,
    cupy: Any,
    zero: bool,
) -> Any:
    """Return one reusable rolling outgoing buffer view for one hierarchy level."""

    shape = (int(level.n_boxes), 4, int(level.directional.grid.n_directions), int(nrhs))
    if workspace is None:
        arr = cupy.empty(shape, dtype=cupy.complex128)
        if zero:
            arr.fill(0)
        return arr
    parity = int((int(level_idx) - int(hf_start)) & 1)
    backing = workspace.outgoing_roll_even if parity == 0 else workspace.outgoing_roll_odd
    if backing is None:
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling outgoing backing buffer is missing for "
            f"level {int(level_idx)}."
        )
    needed = int(np.prod(shape, dtype=np.int64))
    if int(backing.size) < needed:
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling outgoing buffer is undersized for "
            f"level {level_idx} (need {needed}, have {int(backing.size)})."
        )
    view = backing[:needed].reshape(shape)
    if not bool(getattr(view.flags, "c_contiguous", False)):
        raise RuntimeError(
            "Internal CuPy MLFMM error: rolling outgoing view must be contiguous. "
            f"Got shape={tuple(int(v) for v in view.shape)}."
        )
    if zero:
        view.fill(0)
    return view


def _build_multilevel_outgoing_level_rolling(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_by_parent: dict[int, CuPyMLFMMTransferData],
    leaf_box_states: Any,
    hf_start: int,
    leaf_level: int,
    target_level: int,
    nrhs: int,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    cupy: Any,
) -> Any:
    """Build one outgoing hierarchy level into rolling buffers.

    This function intentionally recomputes outgoing channels from leaf box
    states for the requested level so apply can avoid keeping a full outgoing
    hierarchy resident in device workspace.
    """

    if not (int(hf_start) <= int(target_level) <= int(leaf_level)):
        raise RuntimeError(
            "Internal CuPy MLFMM error: outgoing target level outside sampled hierarchy "
            f"(target={target_level}, hf_start={hf_start}, leaf={leaf_level})."
        )
    leaf_values = _multilevel_outgoing_roll_view(
        workspace=workspace,
        hf_start=hf_start,
        level_idx=leaf_level,
        level=levels[leaf_level],
        nrhs=nrhs,
        cupy=cupy,
        zero=True,
    )
    _box_outgoing_to_directional_cupy(
        levels[leaf_level].directional,
        leaf_box_states,
        out=leaf_values,
        cupy=cupy,
    )
    if int(target_level) == int(leaf_level):
        return leaf_values

    for parent_level in range(int(leaf_level) - 1, int(target_level) - 1, -1):
        transfer = transfer_by_parent.get(int(parent_level))
        if transfer is None:
            raise RuntimeError(
                "Internal CuPy MLFMM error: missing transfer while building outgoing level "
                f"{target_level} (parent level {parent_level})."
            )
        child_level = int(transfer.child_level)
        child_values = _multilevel_outgoing_roll_view(
            workspace=workspace,
            hf_start=hf_start,
            level_idx=child_level,
            level=levels[child_level],
            nrhs=nrhs,
            cupy=cupy,
            zero=False,
        )
        parent_values = _multilevel_outgoing_roll_view(
            workspace=workspace,
            hf_start=hf_start,
            level_idx=parent_level,
            level=levels[parent_level],
            nrhs=nrhs,
            cupy=cupy,
            zero=True,
        )
        for shift, batch in transfer.batches_by_shift.items():
            if str(transfer.map_up.storage) == "packed_stencil":
                _transfer_up_packed_unique_complex128(
                    parent_values,
                    batch.dst_indices,
                    child_values,
                    batch.src_indices,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            elif str(transfer.map_up.storage) == "sparse":
                _transfer_up_sparse_unique_complex128(
                    parent_values,
                    batch.dst_indices,
                    child_values,
                    batch.src_indices,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            else:
                # Internal fallback for unsupported map storage variants.
                mapped = _apply_directional_map(
                    child_values[batch.src_indices], transfer.map_up, cupy=cupy
                )
                _weighted_add_at_complex128(
                    parent_values,
                    batch.dst_indices,
                    mapped,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )

    return _multilevel_outgoing_roll_view(
        workspace=workspace,
        hf_start=hf_start,
        level_idx=target_level,
        level=levels[target_level],
        nrhs=nrhs,
        cupy=cupy,
        zero=False,
    )


def _resolve_stream_bytes_budget(
    *,
    explicit_budget: int | None,
    free_bytes: int,
    fraction: float,
    minimum_bytes: int,
    maximum_bytes: int,
) -> int:
    """Resolve one conservative streamed runtime budget in bytes."""

    if explicit_budget is not None:
        return max(1, int(explicit_budget))
    derived = int(max(float(minimum_bytes), float(free_bytes) * float(fraction)))
    return max(int(minimum_bytes), min(int(maximum_bytes), int(derived)))


def _level_chunk_bytes_per_box(*, level: CuPyMLFMMLevelData, nrhs: int) -> int:
    """Return live directional bytes per source box for one streamed chunk."""

    return (
        3
        * 4
        * int(level.directional.grid.n_directions)
        * int(nrhs)
        * np.dtype(np.complex128).itemsize
    )


def _level_group_bytes_per_box(*, level: CuPyMLFMMLevelData, nrhs: int) -> int:
    """Return incoming-buffer bytes per destination box in one frontier group."""

    return (
        4 * int(level.directional.grid.n_directions) * int(nrhs) * np.dtype(np.complex128).itemsize
    )


def _level_chunk_box_cap(
    *,
    level: CuPyMLFMMLevelData,
    nrhs: int,
    bytes_budget: int,
) -> int:
    """Return a conservative chunk-box cap for one directional level."""

    per_box_bytes = _level_chunk_bytes_per_box(level=level, nrhs=nrhs)
    return max(1, int(bytes_budget) // max(1, int(per_box_bytes)))


def _child_outgoing_bytes_budget(
    *,
    level: CuPyMLFMMLevelData,
    nrhs: int,
    total_budget: int,
    bytes_in_flight: int,
) -> int:
    """Return child outgoing budget after reserving live ancestor outgoing arrays."""

    per_box_bytes = _level_chunk_bytes_per_box(level=level, nrhs=nrhs)
    remaining = int(total_budget) - int(bytes_in_flight)
    return max(int(per_box_bytes), int(remaining))


def _iter_id_chunks(ids: Any, *, chunk_size: int) -> Iterator[Any]:
    """Yield chunk-sized views of one sorted id vector."""

    n_total = int(ids.shape[0])
    if n_total == 0:
        return
    for start in range(0, n_total, int(chunk_size)):
        yield ids[start : min(n_total, start + int(chunk_size))]


def _filter_batch_for_sorted_dst_ids(
    *,
    src_indices: Any,
    dst_indices: Any,
    dst_ids_sorted: Any,
    cupy: Any,
) -> tuple[Any, Any] | None:
    """Filter one `(src,dst)` batch to a sorted destination-id subset.

    Returns `(src_filtered_global, dst_filtered_local)` where destination rows
    are mapped into the local chunk order given by `dst_ids_sorted`.
    """

    dst_ids = cupy.asarray(dst_ids_sorted, dtype=cupy.int32).reshape(-1)
    if int(dst_ids.size) == 0:
        return None
    dst_all = cupy.asarray(dst_indices, dtype=cupy.int32).reshape(-1)
    if int(dst_all.size) == 0:
        return None
    src_all = cupy.asarray(src_indices, dtype=cupy.int32).reshape(-1)
    positions = cupy.searchsorted(dst_ids, dst_all)
    max_index = int(dst_ids.size) - 1
    clamped = cupy.minimum(positions, max_index)
    valid = (positions < int(dst_ids.size)) & (dst_ids[clamped] == dst_all)
    matched_rows = cupy.nonzero(valid)[0].astype(cupy.int32, copy=False)
    if int(matched_rows.size) == 0:
        return None
    return src_all[matched_rows], positions[matched_rows].astype(cupy.int32, copy=False)


def _filter_query_ids_to_sorted_chunk(
    *,
    chunk_ids_sorted: Any,
    query_ids: Any,
    cupy: Any,
) -> tuple[Any, Any] | None:
    """Return `(query_rows, local_rows)` for arbitrary queries against sorted ids."""

    chunk_ids = cupy.asarray(chunk_ids_sorted, dtype=cupy.int32).reshape(-1)
    queries = cupy.asarray(query_ids, dtype=cupy.int32).reshape(-1)
    if int(chunk_ids.size) == 0 or int(queries.size) == 0:
        return None
    positions = cupy.searchsorted(chunk_ids, queries)
    max_index = int(chunk_ids.size) - 1
    clamped = cupy.minimum(positions, max_index)
    valid = (positions < int(chunk_ids.size)) & (chunk_ids[clamped] == queries)
    query_rows = cupy.nonzero(valid)[0].astype(cupy.int32, copy=False)
    if int(query_rows.size) == 0:
        return None
    return query_rows, positions[query_rows].astype(cupy.int32, copy=False)


def _partition_query_rows_by_compact_unique_chunks(
    *,
    unique_ids_sorted: Any,
    query_ids: Any,
    chunk_size: int,
    cupy: Any,
) -> dict[int, tuple[Any, Any]]:
    """Bucket arbitrary query ids by contiguous chunks of one compact unique-id superset."""

    matched = _filter_query_ids_to_sorted_chunk(
        chunk_ids_sorted=unique_ids_sorted,
        query_ids=query_ids,
        cupy=cupy,
    )
    if matched is None:
        return {}
    query_rows, unique_local_rows = matched
    chunk_size = int(chunk_size)
    chunk_indices = unique_local_rows // chunk_size
    order = cupy.argsort(chunk_indices)
    chunk_indices_sorted = chunk_indices[order]
    query_rows_sorted = query_rows[order]
    unique_local_rows_sorted = unique_local_rows[order]
    chunk_ids, start_rows, counts = cupy.unique(
        chunk_indices_sorted,
        return_index=True,
        return_counts=True,
    )
    partitions: dict[int, tuple[Any, Any]] = {}
    partition_triplets = np.asarray(
        cupy.asnumpy(
            cupy.stack(
                (
                    chunk_ids.astype(cupy.int32, copy=False),
                    start_rows.astype(cupy.int32, copy=False),
                    counts.astype(cupy.int32, copy=False),
                ),
                axis=1,
            )
        ),
        dtype=np.int32,
    )
    for chunk_idx, start, count in partition_triplets.tolist():
        row_slice = slice(int(start), int(start) + int(count))
        chunk_row_offset = int(chunk_idx) * chunk_size
        partitions[int(chunk_idx)] = (
            query_rows_sorted[row_slice].astype(cupy.int32, copy=False),
            (unique_local_rows_sorted[row_slice] - int(chunk_row_offset)).astype(
                cupy.int32, copy=False
            ),
        )
    return partitions


def _sorted_unique_id_overlap_count(
    *,
    lhs_ids_sorted: Any,
    rhs_ids_sorted: Any,
    cupy: Any,
) -> int:
    """Count overlap between two sorted unique id vectors."""

    lhs_ids = cupy.asarray(lhs_ids_sorted, dtype=cupy.int32).reshape(-1)
    rhs_ids = cupy.asarray(rhs_ids_sorted, dtype=cupy.int32).reshape(-1)
    if int(lhs_ids.size) == 0 or int(rhs_ids.size) == 0:
        return 0
    positions = cupy.searchsorted(rhs_ids, lhs_ids)
    max_index = int(rhs_ids.size) - 1
    clamped = cupy.minimum(positions, max_index)
    valid = (positions < int(rhs_ids.size)) & (rhs_ids[clamped] == lhs_ids)
    return int(cupy.count_nonzero(valid))


def _accumulate_stream_seconds(
    stream_stats: dict[str, object] | None,
    *,
    level_idx: int,
    key: str,
    seconds: float,
) -> None:
    """Accumulate optional streamed timing counters by sampled level."""

    if stream_stats is None:
        return
    timings_by_level = cast(
        dict[str, dict[str, float]],
        stream_stats.setdefault("timings_seconds_by_level", {}),
    )
    level_timings = timings_by_level.setdefault(str(level_idx), {})
    level_timings[str(key)] = float(level_timings.get(str(key), 0.0) + float(seconds))


def _record_leaf_receive_chunk_stats(
    stream_stats: dict[str, object] | None,
    *,
    level_idx: int,
    leaves: int,
    pair_block_bytes: int,
) -> None:
    """Record optional leaf receive chunk counters for streamed diagnostics."""

    if stream_stats is None:
        return
    level_key = str(level_idx)
    counts = cast(dict[str, int], stream_stats.setdefault("leaf_receive_chunk_count", {}))
    counts[level_key] = int(counts.get(level_key, 0) + 1)
    leaf_sums = cast(dict[str, int], stream_stats.setdefault("leaf_receive_leaf_sum", {}))
    leaf_sums[level_key] = int(leaf_sums.get(level_key, 0) + int(leaves))
    peak_leaves = cast(dict[str, int], stream_stats.setdefault("leaf_receive_peak_leaves", {}))
    peak_leaves[level_key] = max(int(peak_leaves.get(level_key, 0)), int(leaves))
    peak_pair_bytes = cast(
        dict[str, int],
        stream_stats.setdefault("leaf_receive_pair_blocks_peak_bytes", {}),
    )
    peak_pair_bytes[level_key] = max(
        int(peak_pair_bytes.get(level_key, 0)),
        int(pair_block_bytes),
    )


def _record_same_level_source_union_stats(
    stream_stats: dict[str, object] | None,
    *,
    level_idx: int,
    source_ids_unique: Any,
    cupy: Any,
) -> None:
    """Record optional overlap diagnostics for adjacent same-level source unions."""

    if stream_stats is None:
        return
    source_ids = cupy.asarray(source_ids_unique, dtype=cupy.int32).reshape(-1)
    union_stats_by_level = cast(
        dict[str, dict[str, float | int]],
        stream_stats.setdefault("same_level_source_union_stats", {}),
    )
    level_stats = union_stats_by_level.setdefault(str(level_idx), {})
    n_source_ids = int(source_ids.size)
    level_stats["count"] = int(level_stats.get("count", 0)) + 1
    level_stats["box_sum"] = int(level_stats.get("box_sum", 0)) + int(n_source_ids)
    level_stats["box_peak"] = max(int(level_stats.get("box_peak", 0)), int(n_source_ids))
    history_by_level = cast(
        dict[str, list[Any]],
        stream_stats.setdefault("_internal_same_level_source_union_history", {}),
    )
    history = history_by_level.setdefault(str(level_idx), [])
    for lag_idx, prev_ids in enumerate(reversed(history[-3:]), start=1):
        if n_source_ids <= 0:
            overlap_ratio = 0.0
        else:
            overlap_count = _sorted_unique_id_overlap_count(
                lhs_ids_sorted=source_ids,
                rhs_ids_sorted=prev_ids,
                cupy=cupy,
            )
            overlap_ratio = float(overlap_count) / float(n_source_ids)
        ratio_key = f"overlap_prev{lag_idx}_ratio_sum"
        count_key = f"overlap_prev{lag_idx}_count"
        level_stats[ratio_key] = float(level_stats.get(ratio_key, 0.0)) + float(overlap_ratio)
        level_stats[count_key] = int(level_stats.get(count_key, 0)) + 1
    history.append(source_ids)
    if len(history) > 3:
        del history[:-3]


def _level_group_box_cap(
    *,
    level: CuPyMLFMMLevelData,
    nrhs: int,
    bytes_budget: int,
) -> int:
    """Cap one resident destination-chunk group from incoming-buffer live bytes."""

    bytes_per_box = _level_group_bytes_per_box(level=level, nrhs=nrhs)
    if bytes_per_box <= 0:
        return 1
    return max(1, int(bytes_budget) // int(bytes_per_box))


def _iter_chunk_groups_by_total_boxes(
    chunks: Iterable[tuple[Any, Any]],
    *,
    box_cap: int,
) -> Iterator[list[tuple[Any, Any]]]:
    """Yield contiguous chunk groups whose total box count stays within `box_cap`."""

    grouped: list[tuple[Any, Any]] = []
    grouped_boxes = 0
    for box_ids, incoming in chunks:
        n_boxes = int(box_ids.shape[0])
        if grouped and grouped_boxes + n_boxes > int(box_cap):
            yield grouped
            grouped = []
            grouped_boxes = 0
        grouped.append((box_ids, incoming))
        grouped_boxes += int(n_boxes)
    if grouped:
        yield grouped


def _iter_zero_incoming_chunks_for_level(
    *,
    level: CuPyMLFMMLevelData,
    chunk_box_cap: int,
    nrhs: int,
    cupy: Any,
) -> Iterator[tuple[Any, Any]]:
    """Yield zero-initialized destination chunks for one sampled level lazily."""

    level_box_ids = cupy.arange(int(level.n_boxes), dtype=cupy.int32)
    ndirs = int(level.directional.grid.n_directions)
    for box_ids in _iter_id_chunks(level_box_ids, chunk_size=int(chunk_box_cap)):
        incoming = cupy.zeros(
            (int(box_ids.shape[0]), 4, ndirs, int(nrhs)),
            dtype=cupy.complex128,
        )
        yield box_ids, incoming


def _iter_level_far_interactions(
    level: CuPyMLFMMLevelData,
) -> Iterator[tuple[CuPyOffsetBatchData, Any]]:
    """Yield ordinary same-level M2L batches."""

    for offset, batch in level.far_offset_batches.items():
        yield batch, level.offset_diagonals[offset]


def _apply_same_level_far_streamed_chunk_group(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_by_parent: dict[int, CuPyMLFMMTransferData],
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    leaf_otf_chunk_leaves: int | None,
    streamed_far_chunk_bytes_budget: int,
    level_idx: int,
    leaf_level: int,
    chunks: list[tuple[Any, Any]],
    box_nm_leaf: int,
    x_states: Any,
    nrhs: int,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> None:
    """Accumulate same-level far contributions for one destination-chunk group."""

    if not chunks:
        return
    level = levels[int(level_idx)]
    source_box_cap = _level_chunk_box_cap(
        level=level,
        nrhs=int(nrhs),
        bytes_budget=int(streamed_far_chunk_bytes_budget),
    )
    if stream_stats is not None:
        caps = cast(dict[str, int], stream_stats.setdefault("level_chunk_box_cap", {}))
        caps[str(level_idx)] = int(source_box_cap)
    full_source_ids: Any | None = None
    full_source_outgoing: Any | None = None
    if int(source_box_cap) >= int(level.n_boxes):
        full_source_started = time.perf_counter() if stream_stats is not None else 0.0
        full_source_ids = cupy.arange(int(level.n_boxes), dtype=cupy.int32)
        full_source_outgoing = _build_outgoing_subset_streamed(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_groups=leaf_groups,
            leaf_apply_mode=leaf_apply_mode,
            leaf_translation_tables=leaf_translation_tables,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            outgoing_bytes_in_flight=0,
            level_idx=int(level_idx),
            leaf_level=int(leaf_level),
            box_ids_sorted=full_source_ids,
            box_nm_leaf=int(box_nm_leaf),
            x_states=x_states,
            nrhs=int(nrhs),
            build_reason="same_level_far",
            cupy=cupy,
            stream_stats=stream_stats,
        )
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="same_level_full_source_outgoing_build",
            seconds=time.perf_counter() - full_source_started,
        )
        if stream_stats is not None:
            fast_counts = cast(
                dict[str, int], stream_stats.setdefault("full_level_outgoing_reuse_count", {})
            )
            fast_counts[str(level_idx)] = int(fast_counts.get(str(level_idx), 0) + len(chunks))
    subset_filter_started = time.perf_counter() if stream_stats is not None else 0.0
    filtered_by_chunk: list[list[tuple[Any, Any, Any]]] = []
    source_batches: list[Any] = []
    for box_ids, _incoming in chunks:
        filtered_offsets: list[tuple[Any, Any, Any]] = []
        for batch, diagonal in _iter_level_far_interactions(level):
            filtered = _filter_batch_for_sorted_dst_ids(
                src_indices=batch.src_indices,
                dst_indices=batch.dst_indices,
                dst_ids_sorted=box_ids,
                cupy=cupy,
            )
            if filtered is None:
                continue
            source_ids_all, dst_local_all = filtered
            filtered_offsets.append((diagonal, source_ids_all, dst_local_all))
            source_batches.append(source_ids_all)
        filtered_by_chunk.append(filtered_offsets)
    if not source_batches:
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="same_level_subset_filter",
            seconds=time.perf_counter() - subset_filter_started,
        )
        return
    if full_source_outgoing is not None and full_source_ids is not None:
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="same_level_subset_filter",
            seconds=time.perf_counter() - subset_filter_started,
        )
        for (_box_ids, incoming), filtered_offsets in zip(chunks, filtered_by_chunk, strict=True):
            current_incoming = cupy.asarray(incoming, dtype=cupy.complex128)
            for diagonal, source_ids_all, dst_local_all in filtered_offsets:
                matched = _filter_query_ids_to_sorted_chunk(
                    chunk_ids_sorted=full_source_ids,
                    query_ids=source_ids_all,
                    cupy=cupy,
                )
                if matched is None:
                    continue
                source_query_rows, source_local = matched
                _weighted_gather_add_complex128(
                    current_incoming,
                    dst_local_all[source_query_rows],
                    full_source_outgoing,
                    source_local,
                    diagonal,
                    cupy=cupy,
                )
        return
    source_ids_unique = cupy.unique(cupy.concatenate(source_batches, axis=0)).astype(
        cupy.int32, copy=False
    )
    _record_same_level_source_union_stats(
        stream_stats,
        level_idx=int(level_idx),
        source_ids_unique=source_ids_unique,
        cupy=cupy,
    )
    source_matches_by_chunk: list[list[tuple[Any, Any, dict[int, tuple[Any, Any]]]]] = []
    for filtered_offsets in filtered_by_chunk:
        offset_matches: list[tuple[Any, Any, dict[int, tuple[Any, Any]]]] = []
        for diagonal, source_ids_all, dst_local_all in filtered_offsets:
            offset_matches.append(
                (
                    diagonal,
                    dst_local_all,
                    _partition_query_rows_by_compact_unique_chunks(
                        unique_ids_sorted=source_ids_unique,
                        query_ids=source_ids_all,
                        chunk_size=int(source_box_cap),
                        cupy=cupy,
                    ),
                )
            )
        source_matches_by_chunk.append(offset_matches)
    _accumulate_stream_seconds(
        stream_stats,
        level_idx=int(level_idx),
        key="same_level_subset_filter",
        seconds=time.perf_counter() - subset_filter_started,
    )
    for source_chunk_idx, source_chunk_ids in enumerate(
        _iter_id_chunks(source_ids_unique, chunk_size=int(source_box_cap))
    ):
        source_outgoing = _build_outgoing_subset_streamed(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_groups=leaf_groups,
            leaf_apply_mode=leaf_apply_mode,
            leaf_translation_tables=leaf_translation_tables,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            outgoing_bytes_in_flight=0,
            level_idx=int(level_idx),
            leaf_level=int(leaf_level),
            box_ids_sorted=source_chunk_ids,
            box_nm_leaf=int(box_nm_leaf),
            x_states=x_states,
            nrhs=int(nrhs),
            build_reason="same_level_far",
            cupy=cupy,
            stream_stats=stream_stats,
        )
        for (_box_ids, incoming), offset_matches in zip(
            chunks, source_matches_by_chunk, strict=True
        ):
            current_incoming = cupy.asarray(incoming, dtype=cupy.complex128)
            for diagonal, dst_local_all, source_matches in offset_matches:
                matched = source_matches.get(int(source_chunk_idx))
                if matched is None:
                    continue
                source_query_rows, source_local = matched
                _weighted_gather_add_complex128(
                    current_incoming,
                    dst_local_all[source_query_rows],
                    source_outgoing,
                    source_local,
                    diagonal,
                    cupy=cupy,
                )
        source_outgoing = None


def _iter_child_chunk_specs_streamed(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_down: CuPyMLFMMTransferData,
    level_idx: int,
    box_ids_sorted: Any,
    streamed_far_chunk_bytes_budget: int,
    nrhs: int,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> Iterator[tuple[int, Any, list[tuple[Offset3, Any, Any]]]]:
    """Yield child incoming chunk specs before allocating their payloads."""

    box_ids = cupy.asarray(box_ids_sorted, dtype=cupy.int32).reshape(-1)
    child_level = int(transfer_down.child_level)
    filtered_child_by_shift: list[tuple[Offset3, Any, Any]] = []
    child_batches: list[Any] = []
    for shift, batch in transfer_down.batches_by_shift.items():
        filtered = _filter_batch_for_sorted_dst_ids(
            src_indices=batch.src_indices,
            dst_indices=batch.dst_indices,
            dst_ids_sorted=box_ids,
            cupy=cupy,
        )
        if filtered is None:
            continue
        child_ids_all, parent_local_all = filtered
        filtered_child_by_shift.append((shift, child_ids_all, parent_local_all))
        child_batches.append(child_ids_all)
    if not filtered_child_by_shift:
        return
    child_ids_unique = cupy.unique(cupy.concatenate(child_batches, axis=0)).astype(cupy.int32)
    child_chunk_box_cap = _level_chunk_box_cap(
        level=levels[child_level],
        nrhs=int(nrhs),
        bytes_budget=int(streamed_far_chunk_bytes_budget),
    )
    if stream_stats is not None:
        caps = cast(dict[str, int], stream_stats.setdefault("level_chunk_box_cap", {}))
        caps[str(child_level)] = int(child_chunk_box_cap)
    child_matches_by_shift: list[tuple[Offset3, Any, dict[int, tuple[Any, Any]]]] = []
    for shift, child_ids_all, parent_local_all in filtered_child_by_shift:
        child_matches_by_shift.append(
            (
                shift,
                parent_local_all,
                _partition_query_rows_by_compact_unique_chunks(
                    unique_ids_sorted=child_ids_unique,
                    query_ids=child_ids_all,
                    chunk_size=int(child_chunk_box_cap),
                    cupy=cupy,
                ),
            )
        )
    for child_chunk_idx, child_chunk_ids in enumerate(
        _iter_id_chunks(child_ids_unique, chunk_size=int(child_chunk_box_cap))
    ):
        chunk_matches: list[tuple[Offset3, Any, Any]] = []
        for shift, parent_local_all, child_matches in child_matches_by_shift:
            matched = child_matches.get(int(child_chunk_idx))
            if matched is None:
                continue
            child_query_rows, child_local = matched
            parent_rows = parent_local_all[child_query_rows]
            chunk_matches.append((shift, child_local, parent_rows))
        if chunk_matches:
            yield child_level, child_chunk_ids, chunk_matches


def _build_child_incoming_chunk_streamed(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_down: CuPyMLFMMTransferData,
    child_level: int,
    child_chunk_ids: Any,
    current_incoming: Any,
    chunk_matches: list[tuple[Offset3, Any, Any]],
    nrhs: int,
    cupy: Any,
) -> Any:
    """Allocate and fill one child incoming chunk after frontier flushing."""

    child_incoming = cupy.zeros(
        (
            int(child_chunk_ids.shape[0]),
            4,
            int(levels[int(child_level)].directional.grid.n_directions),
            int(nrhs),
        ),
        dtype=cupy.complex128,
    )
    for shift, child_local, parent_rows in chunk_matches:
        if str(transfer_down.map_down.storage) == "packed_stencil":
            _transfer_down_packed_unique_complex128(
                child_incoming,
                child_local,
                current_incoming,
                parent_rows,
                transfer_down.map_down,
                transfer_down.phase_down_by_shift[shift],
                cupy=cupy,
            )
        elif str(transfer_down.map_down.storage) == "sparse":
            _transfer_down_sparse_unique_complex128(
                child_incoming,
                child_local,
                current_incoming,
                parent_rows,
                transfer_down.map_down,
                transfer_down.phase_down_by_shift[shift],
                cupy=cupy,
            )
        else:
            shifted = (
                current_incoming[parent_rows]
                * transfer_down.phase_down_by_shift[shift][None, None, :, None]
            )
            mapped = _apply_directional_map(shifted, transfer_down.map_down, cupy=cupy)
            _add_at_complex128(
                child_incoming,
                child_local,
                mapped,
                cupy=cupy,
            )
    return child_incoming


def _build_outgoing_subset_streamed(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_by_parent: dict[int, CuPyMLFMMTransferData],
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    leaf_otf_chunk_leaves: int | None,
    streamed_far_chunk_bytes_budget: int,
    outgoing_bytes_in_flight: int,
    level_idx: int,
    leaf_level: int,
    box_ids_sorted: Any,
    box_nm_leaf: int,
    x_states: Any,
    nrhs: int,
    build_reason: str,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> Any:
    """Build outgoing directional channels for one selected box subset."""

    build_started = time.perf_counter() if stream_stats is not None else 0.0
    box_ids = cupy.asarray(box_ids_sorted, dtype=cupy.int32).reshape(-1)
    level = levels[int(level_idx)]
    n_boxes_sel = int(box_ids.size)
    if stream_stats is not None:
        build_counts = cast(dict[str, int], stream_stats.setdefault("outgoing_build_count", {}))
        build_counts[str(level_idx)] = int(build_counts.get(str(level_idx), 0) + 1)
        build_reason_counts = cast(
            dict[str, dict[str, int]],
            stream_stats.setdefault("outgoing_build_count_by_reason", {}),
        )
        level_reason_counts = build_reason_counts.setdefault(str(level_idx), {})
        level_reason_counts[str(build_reason)] = int(
            level_reason_counts.get(str(build_reason), 0) + 1
        )
        peak_boxes = cast(dict[str, int], stream_stats.setdefault("outgoing_build_peak_boxes", {}))
        peak_boxes[str(level_idx)] = max(
            int(peak_boxes.get(str(level_idx), 0)),
            int(n_boxes_sel),
        )
    outgoing = cupy.empty(
        (n_boxes_sel, 4, int(level.directional.grid.n_directions), int(nrhs)),
        dtype=cupy.complex128,
    )
    _record_stream_pool_peak(cupy, stream_stats)
    current_outgoing_bytes = _device_array_nbytes(outgoing)
    child_outgoing_bytes_in_flight = int(outgoing_bytes_in_flight) + int(current_outgoing_bytes)
    if stream_stats is not None:
        peak_bytes = cast(
            dict[str, int],
            stream_stats.setdefault("outgoing_build_stack_peak_bytes", {}),
        )
        peak_bytes[str(level_idx)] = max(
            int(peak_bytes.get(str(level_idx), 0)),
            int(child_outgoing_bytes_in_flight),
        )
    if n_boxes_sel == 0:
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="outgoing_build_total_inclusive",
            seconds=time.perf_counter() - build_started,
        )
        return outgoing
    if int(level_idx) == int(leaf_level):
        leaf_started = time.perf_counter() if stream_stats is not None else 0.0
        leaf_box_states = _aggregate_selected_leaf_box_states(
            x_states,
            selected_leaf_ids=box_ids,
            leaf_groups=leaf_groups,
            leaf_apply_mode=leaf_apply_mode,
            leaf_translation_tables=leaf_translation_tables,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            box_nm=int(box_nm_leaf),
            nrhs=int(nrhs),
            cupy=cupy,
        )
        outgoing_leaf = _box_outgoing_to_directional_cupy(
            levels[int(leaf_level)].directional,
            leaf_box_states,
            out=outgoing,
            cupy=cupy,
        )
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="leaf_outgoing_build",
            seconds=time.perf_counter() - leaf_started,
        )
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="outgoing_build_total_inclusive",
            seconds=time.perf_counter() - build_started,
        )
        return outgoing_leaf

    outgoing.fill(0)
    transfer = transfer_by_parent.get(int(level_idx))
    if transfer is None:
        raise RuntimeError(
            "Internal CuPy MLFMM error: missing transfer while streaming outgoing level "
            f"{int(level_idx)}."
        )
    child_level = int(transfer.child_level)
    child_bytes_budget = _child_outgoing_bytes_budget(
        level=levels[child_level],
        nrhs=int(nrhs),
        total_budget=int(streamed_far_chunk_bytes_budget),
        bytes_in_flight=int(child_outgoing_bytes_in_flight),
    )
    child_box_cap = _level_chunk_box_cap(
        level=levels[child_level],
        nrhs=int(nrhs),
        bytes_budget=int(child_bytes_budget),
    )
    if stream_stats is not None:
        caps = cast(dict[str, int], stream_stats.setdefault("level_chunk_box_cap", {}))
        caps[str(child_level)] = int(child_box_cap)
    full_child_ids: Any | None = None
    full_child_outgoing: Any | None = None
    if int(child_box_cap) >= int(levels[child_level].n_boxes):
        full_child_ids = cupy.arange(int(levels[child_level].n_boxes), dtype=cupy.int32)
        full_child_outgoing = _build_outgoing_subset_streamed(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_groups=leaf_groups,
            leaf_apply_mode=leaf_apply_mode,
            leaf_translation_tables=leaf_translation_tables,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            outgoing_bytes_in_flight=int(child_outgoing_bytes_in_flight),
            level_idx=child_level,
            leaf_level=int(leaf_level),
            box_ids_sorted=full_child_ids,
            box_nm_leaf=int(box_nm_leaf),
            x_states=x_states,
            nrhs=int(nrhs),
            build_reason="parent_transfer",
            cupy=cupy,
            stream_stats=stream_stats,
        )
        if stream_stats is not None:
            fast_counts = cast(
                dict[str, int], stream_stats.setdefault("full_level_outgoing_reuse_count", {})
            )
            fast_counts[str(child_level)] = int(fast_counts.get(str(child_level), 0) + 1)
    filtered_by_shift: list[tuple[Offset3, Any, Any]] = []
    child_batches: list[Any] = []
    for shift, batch in transfer.batches_by_shift.items():
        filtered = _filter_batch_for_sorted_dst_ids(
            src_indices=batch.src_indices,
            dst_indices=batch.dst_indices,
            dst_ids_sorted=box_ids,
            cupy=cupy,
        )
        if filtered is None:
            continue
        child_ids_all, parent_local_all = filtered
        filtered_by_shift.append((shift, child_ids_all, parent_local_all))
        child_batches.append(child_ids_all)
    if not filtered_by_shift:
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="outgoing_build_total_inclusive",
            seconds=time.perf_counter() - build_started,
        )
        return outgoing
    if full_child_outgoing is not None and full_child_ids is not None:
        for shift, child_ids_all, parent_local_all in filtered_by_shift:
            matched = _filter_query_ids_to_sorted_chunk(
                chunk_ids_sorted=full_child_ids,
                query_ids=child_ids_all,
                cupy=cupy,
            )
            if matched is None:
                continue
            child_query_rows, child_local = matched
            parent_rows = parent_local_all[child_query_rows]
            if str(transfer.map_up.storage) == "packed_stencil":
                _transfer_up_packed_unique_complex128(
                    outgoing,
                    parent_rows,
                    full_child_outgoing,
                    child_local,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            elif str(transfer.map_up.storage) == "sparse":
                _transfer_up_sparse_unique_complex128(
                    outgoing,
                    parent_rows,
                    full_child_outgoing,
                    child_local,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            else:
                mapped = _apply_directional_map(
                    full_child_outgoing[child_local],
                    transfer.map_up,
                    cupy=cupy,
                )
                _weighted_add_at_complex128(
                    outgoing,
                    parent_rows,
                    mapped,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
        _accumulate_stream_seconds(
            stream_stats,
            level_idx=int(level_idx),
            key="outgoing_build_total_inclusive",
            seconds=time.perf_counter() - build_started,
        )
        return outgoing
    child_ids_unique = cupy.unique(cupy.concatenate(child_batches, axis=0)).astype(
        cupy.int32, copy=False
    )
    child_matches_by_shift: list[tuple[Offset3, Any, dict[int, tuple[Any, Any]]]] = []
    for shift, child_ids_all, parent_local_all in filtered_by_shift:
        child_matches_by_shift.append(
            (
                shift,
                parent_local_all,
                _partition_query_rows_by_compact_unique_chunks(
                    unique_ids_sorted=child_ids_unique,
                    query_ids=child_ids_all,
                    chunk_size=int(child_box_cap),
                    cupy=cupy,
                ),
            )
        )
    # Keep chunks aligned to this call's compact unique-id order. Global chunk
    # ranges fight the frontier-local source subsets used by streamed far apply,
    # so only reintroduce them if the traversal itself becomes source-centric.
    for child_chunk_idx, child_chunk_ids in enumerate(
        _iter_id_chunks(child_ids_unique, chunk_size=int(child_box_cap))
    ):
        child_outgoing = _build_outgoing_subset_streamed(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_groups=leaf_groups,
            leaf_apply_mode=leaf_apply_mode,
            leaf_translation_tables=leaf_translation_tables,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            outgoing_bytes_in_flight=int(child_outgoing_bytes_in_flight),
            level_idx=child_level,
            leaf_level=int(leaf_level),
            box_ids_sorted=child_chunk_ids,
            box_nm_leaf=int(box_nm_leaf),
            x_states=x_states,
            nrhs=int(nrhs),
            build_reason="parent_transfer",
            cupy=cupy,
            stream_stats=stream_stats,
        )
        for shift, parent_local_all, child_matches in child_matches_by_shift:
            matched = child_matches.get(int(child_chunk_idx))
            if matched is None:
                continue
            child_query_rows, child_local = matched
            parent_rows = parent_local_all[child_query_rows]
            if str(transfer.map_up.storage) == "packed_stencil":
                _transfer_up_packed_unique_complex128(
                    outgoing,
                    parent_rows,
                    child_outgoing,
                    child_local,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            elif str(transfer.map_up.storage) == "sparse":
                _transfer_up_sparse_unique_complex128(
                    outgoing,
                    parent_rows,
                    child_outgoing,
                    child_local,
                    transfer.map_up,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
            else:
                mapped = _apply_directional_map(
                    child_outgoing[child_local],
                    transfer.map_up,
                    cupy=cupy,
                )
                _weighted_add_at_complex128(
                    outgoing,
                    parent_rows,
                    mapped,
                    transfer.phase_up_by_shift[shift],
                    cupy=cupy,
                )
        child_outgoing = None
    _accumulate_stream_seconds(
        stream_stats,
        level_idx=int(level_idx),
        key="outgoing_build_total_inclusive",
        seconds=time.perf_counter() - build_started,
    )
    return outgoing


def _apply_multilevel_frontier_streamed(
    *,
    levels: tuple[CuPyMLFMMLevelData, ...],
    transfer_by_parent: dict[int, CuPyMLFMMTransferData],
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
    leaf_apply_mode: CuPyMLFMMLeafApplyMode,
    leaf_translation_tables: CuPyLeafTranslationTablesData | None,
    receive_adjoint_cache: dict[int, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    streamed_far_chunk_bytes_budget: int,
    streamed_far_frontier_bytes_budget: int,
    frontier_bytes_in_flight: int,
    level_idx: int,
    leaf_level: int,
    frontier_chunks: list[tuple[Any, Any]],
    box_nm_leaf: int,
    x_states: Any,
    y_out: Any,
    nm: int,
    nrhs: int,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> None:
    """Apply one streamed multilevel frontier window recursively."""

    if not frontier_chunks:
        return
    current_frontier_bytes = sum(int(box_ids.shape[0]) for box_ids, _incoming in frontier_chunks)
    current_frontier_bytes *= _level_group_bytes_per_box(
        level=levels[int(level_idx)],
        nrhs=int(nrhs),
    )
    current_frontier_bytes_in_flight = int(frontier_bytes_in_flight) + int(current_frontier_bytes)
    if stream_stats is not None:
        stream_stats["frontier_in_flight_peak_bytes"] = max(
            int(cast(int, stream_stats.get("frontier_in_flight_peak_bytes", 0))),
            int(current_frontier_bytes_in_flight),
        )
        level_counts = cast(dict[str, int], stream_stats.setdefault("processed_chunk_count", {}))
        level_counts[str(level_idx)] = int(
            level_counts.get(str(level_idx), 0) + len(frontier_chunks)
        )
        peak_boxes = cast(dict[str, int], stream_stats.setdefault("processed_chunk_peak_boxes", {}))
        peak_boxes[str(level_idx)] = max(
            int(peak_boxes.get(str(level_idx), 0)),
            max(int(box_ids.shape[0]) for box_ids, _incoming in frontier_chunks),
        )
    _apply_same_level_far_streamed_chunk_group(
        levels=levels,
        transfer_by_parent=transfer_by_parent,
        leaf_groups=leaf_groups,
        leaf_apply_mode=leaf_apply_mode,
        leaf_translation_tables=leaf_translation_tables,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
        level_idx=int(level_idx),
        leaf_level=int(leaf_level),
        chunks=frontier_chunks,
        box_nm_leaf=int(box_nm_leaf),
        x_states=x_states,
        nrhs=int(nrhs),
        cupy=cupy,
        stream_stats=stream_stats,
    )

    if int(level_idx) >= int(leaf_level):
        for box_ids, incoming_chunk in frontier_chunks:
            receive_project_started = time.perf_counter() if stream_stats is not None else 0.0
            incoming_box = _directional_to_box_regular_cupy(
                levels[int(leaf_level)].directional,
                cupy.asarray(incoming_chunk, dtype=cupy.complex128),
                cupy=cupy,
            )
            _accumulate_stream_seconds(
                stream_stats,
                level_idx=int(leaf_level),
                key="leaf_receive_directional_to_box",
                seconds=time.perf_counter() - receive_project_started,
            )
            _receive_selected_leaf_boxes_to_particles(
                incoming_box,
                selected_leaf_ids=box_ids,
                leaf_groups=leaf_groups,
                leaf_apply_mode=leaf_apply_mode,
                leaf_translation_tables=leaf_translation_tables,
                leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
                receive_adjoint_cache=receive_adjoint_cache,
                nm=int(nm),
                out=y_out,
                cupy=cupy,
                level_idx=int(leaf_level),
                stream_stats=stream_stats,
            )
        return

    child_level: int | None = None
    child_frontier_box_cap: int | None = None
    child_frontier: list[tuple[Any, Any]] = []
    child_frontier_boxes = 0
    for box_ids, incoming_chunk in frontier_chunks:
        current_incoming = cupy.asarray(incoming_chunk, dtype=cupy.complex128)
        transfer_down = transfer_by_parent.get(int(level_idx))
        if transfer_down is None:
            raise RuntimeError(
                "Internal CuPy MLFMM error: missing multilevel transfer for streamed parent level "
                f"{int(level_idx)}."
            )
        for next_child_level, child_chunk_ids, chunk_matches in _iter_child_chunk_specs_streamed(
            levels=levels,
            transfer_down=transfer_down,
            level_idx=int(level_idx),
            box_ids_sorted=box_ids,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            nrhs=int(nrhs),
            cupy=cupy,
            stream_stats=stream_stats,
        ):
            if child_level is None:
                child_level = int(next_child_level)
                child_frontier_bytes_budget = max(
                    _level_group_bytes_per_box(level=levels[child_level], nrhs=int(nrhs)),
                    int(streamed_far_frontier_bytes_budget) - int(current_frontier_bytes_in_flight),
                )
                child_frontier_box_cap = _level_group_box_cap(
                    level=levels[child_level],
                    nrhs=int(nrhs),
                    bytes_budget=int(child_frontier_bytes_budget),
                )
                if stream_stats is not None:
                    caps = cast(
                        dict[str, int], stream_stats.setdefault("level_frontier_box_cap", {})
                    )
                    caps[str(child_level)] = int(child_frontier_box_cap)
            elif int(next_child_level) != int(child_level):
                raise RuntimeError(
                    "Internal CuPy MLFMM error: inconsistent streamed child frontier levels "
                    f"{int(child_level)} vs {int(next_child_level)}."
                )
            child_boxes = int(child_chunk_ids.shape[0])
            if (
                child_frontier
                and child_frontier_box_cap is not None
                and child_frontier_boxes + child_boxes > int(child_frontier_box_cap)
            ):
                _apply_multilevel_frontier_streamed(
                    levels=levels,
                    transfer_by_parent=transfer_by_parent,
                    leaf_groups=leaf_groups,
                    leaf_apply_mode=leaf_apply_mode,
                    leaf_translation_tables=leaf_translation_tables,
                    receive_adjoint_cache=receive_adjoint_cache,
                    leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
                    streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
                    streamed_far_frontier_bytes_budget=int(streamed_far_frontier_bytes_budget),
                    frontier_bytes_in_flight=int(current_frontier_bytes_in_flight),
                    level_idx=int(child_level),
                    leaf_level=int(leaf_level),
                    frontier_chunks=child_frontier,
                    box_nm_leaf=int(box_nm_leaf),
                    x_states=x_states,
                    y_out=y_out,
                    nm=int(nm),
                    nrhs=int(nrhs),
                    cupy=cupy,
                    stream_stats=stream_stats,
                )
                child_frontier = []
                child_frontier_boxes = 0
            child_incoming = _build_child_incoming_chunk_streamed(
                levels=levels,
                transfer_down=transfer_down,
                child_level=int(next_child_level),
                child_chunk_ids=child_chunk_ids,
                current_incoming=current_incoming,
                chunk_matches=chunk_matches,
                nrhs=int(nrhs),
                cupy=cupy,
            )
            _record_stream_pool_peak(cupy, stream_stats)
            child_chunk = (child_chunk_ids, child_incoming)
            child_frontier.append(child_chunk)
            child_frontier_boxes += int(child_boxes)

    if child_level is None or not child_frontier:
        return
    _apply_multilevel_frontier_streamed(
        levels=levels,
        transfer_by_parent=transfer_by_parent,
        leaf_groups=leaf_groups,
        leaf_apply_mode=leaf_apply_mode,
        leaf_translation_tables=leaf_translation_tables,
        receive_adjoint_cache=receive_adjoint_cache,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
        streamed_far_frontier_bytes_budget=int(streamed_far_frontier_bytes_budget),
        frontier_bytes_in_flight=int(current_frontier_bytes_in_flight),
        level_idx=int(child_level),
        leaf_level=int(leaf_level),
        frontier_chunks=child_frontier,
        box_nm_leaf=int(box_nm_leaf),
        x_states=x_states,
        y_out=y_out,
        nm=int(nm),
        nrhs=int(nrhs),
        cupy=cupy,
        stream_stats=stream_stats,
    )


def _apply_multilevel_far_streamed(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    streamed_far_chunk_bytes_budget: int,
    streamed_far_frontier_bytes_budget: int,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> Any:
    """Apply multilevel far interactions by streaming selected destination frontiers."""

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    y_far = (
        workspace.y_states
        if workspace is not None
        else cupy.zeros((n_particles, int(nm), int(nrhs)), dtype=cupy.complex128)
    )
    y_far.fill(0)
    _record_stream_pool_peak(cupy, stream_stats)
    levels = multilevel.levels
    transfer_by_parent: dict[int, CuPyMLFMMTransferData] = {}
    for transfer in multilevel.transfers:
        parent_level = int(transfer.parent_level)
        if parent_level in transfer_by_parent:
            raise RuntimeError(
                "Internal CuPy MLFMM error: duplicate multilevel transfer parent level "
                f"{parent_level}."
            )
        transfer_by_parent[parent_level] = transfer
    hf_start = int(multilevel.hf_start_level)
    leaf_level = int(multilevel.leaf_level)
    start_level = levels[hf_start]
    top_chunk_box_cap = _level_chunk_box_cap(
        level=start_level,
        nrhs=int(nrhs),
        bytes_budget=int(streamed_far_chunk_bytes_budget),
    )
    if stream_stats is not None:
        caps = cast(dict[str, int], stream_stats.setdefault("level_chunk_box_cap", {}))
        caps[str(hf_start)] = int(top_chunk_box_cap)
    top_frontier_box_cap = _level_group_box_cap(
        level=start_level,
        nrhs=int(nrhs),
        bytes_budget=int(streamed_far_frontier_bytes_budget),
    )
    if stream_stats is not None:
        caps = cast(dict[str, int], stream_stats.setdefault("level_frontier_box_cap", {}))
        caps[str(hf_start)] = int(top_frontier_box_cap)
    # Process the sampled hierarchy as frontiers so one source chunk can feed
    # more than one destination chunk before the traversal descends.
    top_chunks = _iter_zero_incoming_chunks_for_level(
        level=start_level,
        chunk_box_cap=int(top_chunk_box_cap),
        nrhs=int(nrhs),
        cupy=cupy,
    )
    for top_frontier in _iter_chunk_groups_by_total_boxes(
        top_chunks, box_cap=int(top_frontier_box_cap)
    ):
        _apply_multilevel_frontier_streamed(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_groups=multilevel.leaf_groups,
            leaf_apply_mode=multilevel.leaf_apply_mode,
            leaf_translation_tables=multilevel.leaf_translation_tables,
            receive_adjoint_cache=receive_adjoint_cache,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            streamed_far_frontier_bytes_budget=int(streamed_far_frontier_bytes_budget),
            frontier_bytes_in_flight=0,
            level_idx=int(hf_start),
            leaf_level=int(leaf_level),
            frontier_chunks=top_frontier,
            box_nm_leaf=int(multilevel.box_nm),
            x_states=x_states,
            y_out=y_far,
            nm=int(nm),
            nrhs=int(nrhs),
            cupy=cupy,
            stream_stats=stream_stats,
        )
    _record_stream_pool_peak(cupy, stream_stats)
    return y_far


def _apply_multilevel_far(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    pair_blocks_scratch: dict[str, Any] | None,
    leaf_otf_chunk_leaves: int | None,
    leaf_otf_bytes_budget: int | None,
    streamed_far_chunk_bytes_budget: int,
    streamed_far_frontier_bytes_budget: int,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    cupy: Any,
    stream_stats: dict[str, object] | None,
) -> Any:
    """Apply sampled multilevel far interactions on device.

    Grouped source/target schedules are validated as unique during upload.
    Transfer loops run unique-index kernels and keep a high-level fallback only
    for map-storage variants that do not have a fused kernel yet.
    """

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    if str(multilevel.leaf_apply_mode) == "on_the_fly":
        return _apply_multilevel_far_streamed(
            prepared,
            x_states,
            receive_adjoint_cache=receive_adjoint_cache,
            leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
            streamed_far_chunk_bytes_budget=int(streamed_far_chunk_bytes_budget),
            streamed_far_frontier_bytes_budget=int(streamed_far_frontier_bytes_budget),
            workspace=workspace,
            cupy=cupy,
            stream_stats=stream_stats,
        )
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    levels = multilevel.levels
    ws = workspace
    leaf_level = int(multilevel.leaf_level)
    box_nm = int(multilevel.box_nm)
    leaf_box_states = _aggregate_leaf_box_states(
        x_states,
        leaf_groups=multilevel.leaf_groups,
        leaf_apply_mode=multilevel.leaf_apply_mode,
        leaf_translation_tables=multilevel.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        n_leaves=int(multilevel.n_leaves),
        box_nm=box_nm,
        nrhs=nrhs,
        out=(ws.leaf_box_states if ws is not None else None),
        cupy=cupy,
    )

    hf_start = int(multilevel.hf_start_level)
    hf_end = int(multilevel.hf_end_level)
    transfer_by_parent: dict[int, CuPyMLFMMTransferData] = {}
    for transfer in multilevel.transfers:
        parent_level = int(transfer.parent_level)
        if parent_level in transfer_by_parent:
            raise RuntimeError(
                "Internal CuPy MLFMM error: duplicate multilevel transfer parent level "
                f"{parent_level}."
            )
        transfer_by_parent[parent_level] = transfer

    current_level = hf_start
    current_incoming = _multilevel_incoming_roll_view(
        workspace=ws,
        hf_start=hf_start,
        level_idx=current_level,
        level=levels[current_level],
        nrhs=nrhs,
        cupy=cupy,
    )
    for level_idx in range(hf_start, hf_end + 1):
        if level_idx != current_level:
            raise RuntimeError(
                "Internal CuPy MLFMM error: inconsistent downward traversal state "
                f"(expected level {current_level}, got {level_idx})."
            )
        level = levels[level_idx]
        outgoing_level = _build_multilevel_outgoing_level_rolling(
            levels=levels,
            transfer_by_parent=transfer_by_parent,
            leaf_box_states=leaf_box_states,
            hf_start=hf_start,
            leaf_level=leaf_level,
            target_level=level_idx,
            nrhs=nrhs,
            workspace=ws,
            cupy=cupy,
        )
        for batch, diagonal in _iter_level_far_interactions(level):
            _weighted_gather_add_complex128(
                current_incoming,
                batch.dst_indices,
                outgoing_level,
                batch.src_indices,
                diagonal,
                cupy=cupy,
            )
        if level_idx >= hf_end:
            continue
        transfer_down = transfer_by_parent.get(level_idx)
        if transfer_down is None:
            raise RuntimeError(
                "Internal CuPy MLFMM error: missing multilevel transfer for parent level "
                f"{level_idx}."
            )
        child_level = int(transfer_down.child_level)
        child_values = _multilevel_incoming_roll_view(
            workspace=ws,
            hf_start=hf_start,
            level_idx=child_level,
            level=levels[child_level],
            nrhs=nrhs,
            cupy=cupy,
        )
        for shift, batch in transfer_down.batches_by_shift.items():
            if str(transfer_down.map_down.storage) == "packed_stencil":
                _transfer_down_packed_unique_complex128(
                    child_values,
                    batch.src_indices,
                    current_incoming,
                    batch.dst_indices,
                    transfer_down.map_down,
                    transfer_down.phase_down_by_shift[shift],
                    cupy=cupy,
                )
            elif str(transfer_down.map_down.storage) == "sparse":
                _transfer_down_sparse_unique_complex128(
                    child_values,
                    batch.src_indices,
                    current_incoming,
                    batch.dst_indices,
                    transfer_down.map_down,
                    transfer_down.phase_down_by_shift[shift],
                    cupy=cupy,
                )
            else:
                # Internal fallback for unsupported map storage variants.
                shifted = (
                    current_incoming[batch.dst_indices]
                    * transfer_down.phase_down_by_shift[shift][None, None, :, None]
                )
                mapped = _apply_directional_map(shifted, transfer_down.map_down, cupy=cupy)
                _add_at_complex128(
                    child_values,
                    batch.src_indices,
                    mapped,
                    cupy=cupy,
                )
        current_level = child_level
        current_incoming = child_values

    if current_level != leaf_level:
        raise RuntimeError(
            "Internal CuPy MLFMM error: downward traversal did not end at leaf level "
            f"(current={current_level}, leaf={leaf_level})."
        )

    incoming_box = _directional_to_box_regular_cupy(
        levels[leaf_level].directional,
        current_incoming,
        out=(ws.incoming_box if ws is not None else None),
        cupy=cupy,
    )
    return _receive_leaf_boxes_to_particles(
        incoming_box,
        leaf_groups=multilevel.leaf_groups,
        leaf_apply_mode=multilevel.leaf_apply_mode,
        leaf_translation_tables=multilevel.leaf_translation_tables,
        pair_blocks_scratch=pair_blocks_scratch,
        leaf_otf_chunk_leaves=leaf_otf_chunk_leaves,
        leaf_otf_bytes_budget=leaf_otf_bytes_budget,
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        out=(ws.y_states if ws is not None else None),
        cupy=cupy,
    )


def _device_array_nbytes(arr: Any) -> int:
    """Return device-array byte size when available, otherwise zero."""

    if arr is None:
        return 0
    nbytes = getattr(arr, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    # cupyx sparse matrices expose their resident CSR arrays rather than a
    # scalar ``nbytes`` attribute.
    return int(
        sum(
            _device_array_nbytes(getattr(arr, name, None)) for name in ("data", "indices", "indptr")
        )
    )


@dataclass(frozen=True)
class _StreamedFarMemoryPlan:
    """Resolved transient-memory plan for one streamed multilevel apply."""

    device_limit_bytes: int
    active_used_bytes_before_far: int
    stream_transient_budget_bytes: int
    frontier_bytes_budget: int
    source_outgoing_bytes_budget: int
    safety_margin_bytes: int
    unmodeled_temp_reserve_bytes: int
    pool_limit_bytes: int
    pool_limit_applied: bool
    source_outgoing_budget_clipped: bool
    source_outgoing_budget_fragmentation_clipped: bool = False
    largest_single_allocation_bytes: int = 0
    guaranteed_fresh_allocation_bytes: int = 0
    fragmentation_guard_bytes: int = 0
    retry_count: int = 0


def _record_stream_pool_peak(cupy: Any, stream_stats: dict[str, object] | None) -> None:
    """Record per-apply CuPy pool peaks when stream diagnostics are enabled."""

    if stream_stats is None:
        return
    pool = cupy.get_default_memory_pool()
    stream_stats["pool_peak_used_bytes"] = max(
        int(cast(int, stream_stats.get("pool_peak_used_bytes", 0))),
        int(pool.used_bytes()),
    )
    stream_stats["pool_peak_total_bytes"] = max(
        int(cast(int, stream_stats.get("pool_peak_total_bytes", 0))),
        int(pool.total_bytes()),
    )


def _streamed_far_memory_plan(
    *,
    snapshot: CuPyAllocatorSnapshot,
    explicit_source_budget: int | None,
    hf_start_incoming_bytes: int,
    full_level_live_bytes: int,
    full_level_incoming_bytes: int,
) -> _StreamedFarMemoryPlan:
    """Resolve source/frontier budgets from one active-memory plan."""

    safety_margin = 0
    active_headroom = max(
        0,
        int(snapshot.effective_device_limit_bytes) - int(snapshot.pool_used_bytes),
    )
    # Only part of the active headroom is donated to streamed far transients.
    # The rest covers short-lived CuPy temporaries, cached-block fragmentation,
    # and traversal-local arrays that are not represented by the source/frontier
    # directional byte models.
    stream_transient_budget = max(
        _STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES,
        int(float(active_headroom) * _STREAMED_FAR_TRANSIENT_FRACTION_OF_ACTIVE_HEADROOM),
    )
    unmodeled_temp_reserve = max(0, int(active_headroom) - int(stream_transient_budget))
    min_frontier = max(
        _STREAMED_FAR_MIN_FRONTIER_BYTES,
        int(hf_start_incoming_bytes),
    )
    desired_frontier = max(
        min_frontier,
        int(float(stream_transient_budget) * _STREAMED_FAR_FRONTIER_FRACTION_OF_TRANSIENT),
    )
    frontier_budget = min(
        int(full_level_incoming_bytes),
        int(desired_frontier),
        max(
            _STREAMED_FAR_MIN_FRONTIER_BYTES,
            int(stream_transient_budget) - _STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES,
        ),
    )
    frontier_budget = max(_STREAMED_FAR_MIN_FRONTIER_BYTES, int(frontier_budget))
    source_room = max(
        _STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES,
        int(stream_transient_budget) - int(frontier_budget),
    )
    if int(full_level_live_bytes) <= max(
        _STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES,
        int(source_room) // 2,
    ):
        minimum_source = int(full_level_live_bytes)
    else:
        minimum_source = _STREAMED_FAR_MIN_SOURCE_CHUNK_BYTES
    source_budget = max(int(minimum_source), int(source_room))
    source_clipped = False
    if explicit_source_budget is not None:
        requested = int(explicit_source_budget)
        source_clipped = requested > int(source_room)
        source_budget = min(requested, max(int(minimum_source), int(source_room)))
    return _StreamedFarMemoryPlan(
        device_limit_bytes=int(snapshot.effective_device_limit_bytes),
        active_used_bytes_before_far=int(snapshot.pool_used_bytes),
        stream_transient_budget_bytes=int(stream_transient_budget),
        frontier_bytes_budget=int(frontier_budget),
        source_outgoing_bytes_budget=int(source_budget),
        safety_margin_bytes=int(safety_margin),
        unmodeled_temp_reserve_bytes=int(unmodeled_temp_reserve),
        pool_limit_bytes=int(snapshot.pool_limit_bytes),
        pool_limit_applied=bool(snapshot.pool_limit_applied),
        source_outgoing_budget_clipped=bool(source_clipped),
    )


def _largest_streamed_directional_allocation_bytes(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
    source_outgoing_bytes_budget: int,
) -> int:
    """Return the largest single directional array requested by streaming.

    The source budget models three live directional arrays per box. Individual
    outgoing and child-incoming allocations contain one such array, so their
    actual allocation size is one third of the corresponding chunk model.
    """

    largest = 0
    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = multilevel.levels[level_idx]
        box_cap = min(
            int(level.n_boxes),
            _level_chunk_box_cap(
                level=level,
                nrhs=int(nrhs),
                bytes_budget=int(source_outgoing_bytes_budget),
            ),
        )
        largest = max(
            int(largest),
            int(box_cap) * _level_group_bytes_per_box(level=level, nrhs=int(nrhs)),
        )
    return int(largest)


def _clip_stream_plan_to_fresh_allocation_headroom(
    plan: _StreamedFarMemoryPlan,
    *,
    multilevel: CuPyMLFMMMultilevelData,
    nrhs: int,
    guaranteed_fresh_allocation_bytes: int,
) -> _StreamedFarMemoryPlan:
    """Keep a source chunk allocatable when cached blocks remain fragmented."""

    fresh_bytes = max(0, int(guaranteed_fresh_allocation_bytes))
    largest = _largest_streamed_directional_allocation_bytes(
        multilevel,
        nrhs=int(nrhs),
        source_outgoing_bytes_budget=int(plan.source_outgoing_bytes_budget),
    )
    source_budget = int(plan.source_outgoing_bytes_budget)
    fragmentation_clipped = False
    if largest > fresh_bytes:
        minimum_fresh_bytes = max(
            _level_group_bytes_per_box(
                level=multilevel.levels[level_idx],
                nrhs=int(nrhs),
            )
            for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1)
        )
        if fresh_bytes < int(minimum_fresh_bytes):
            raise MemoryError(
                "CuPy MLFMM cannot reserve one streamed directional box within the "
                "guarded device-memory limit. Free other device allocations or set a "
                "smaller problem/box order."
            )
        # `_level_chunk_bytes_per_box()` models three live directional arrays,
        # while the allocator request contains one. Limiting the modeled source
        # budget to three fresh blocks therefore bounds each individual request.
        source_budget = min(int(source_budget), 3 * int(fresh_bytes))
        fragmentation_clipped = source_budget < int(plan.source_outgoing_bytes_budget)
        largest = _largest_streamed_directional_allocation_bytes(
            multilevel,
            nrhs=int(nrhs),
            source_outgoing_bytes_budget=int(source_budget),
        )
        if largest > fresh_bytes:
            raise MemoryError(
                "CuPy MLFMM cannot fit its smallest streamed directional allocation "
                "within guaranteed fresh device-memory headroom."
            )
    return replace(
        plan,
        source_outgoing_bytes_budget=int(source_budget),
        source_outgoing_budget_clipped=(
            bool(plan.source_outgoing_budget_clipped) or bool(fragmentation_clipped)
        ),
        source_outgoing_budget_fragmentation_clipped=bool(fragmentation_clipped),
        largest_single_allocation_bytes=int(largest),
        guaranteed_fresh_allocation_bytes=int(fresh_bytes),
        fragmentation_guard_bytes=max(
            0,
            int(plan.source_outgoing_bytes_budget) - int(source_budget),
        ),
    )


def _resolve_streamed_far_memory_plan(
    cupy: Any,
    *,
    multilevel: CuPyMLFMMMultilevelData,
    nrhs: int,
    explicit_source_budget: int | None,
    hf_start_incoming_bytes: int,
    full_level_live_bytes: int,
    full_level_incoming_bytes: int,
) -> tuple[CuPyAllocatorSnapshot, _StreamedFarMemoryPlan]:
    """Resolve one allocation-shape-aware streamed-far memory plan."""

    snapshot = cupy_allocator_snapshot(cupy, apply_pool_limit=True)
    plan = _streamed_far_memory_plan(
        snapshot=snapshot,
        explicit_source_budget=explicit_source_budget,
        hf_start_incoming_bytes=int(hf_start_incoming_bytes),
        full_level_live_bytes=int(full_level_live_bytes),
        full_level_incoming_bytes=int(full_level_incoming_bytes),
    )
    largest_allocation = _largest_streamed_directional_allocation_bytes(
        multilevel,
        nrhs=int(nrhs),
        source_outgoing_bytes_budget=int(plan.source_outgoing_bytes_budget),
    )
    if (
        int(largest_allocation) > int(snapshot.guaranteed_fresh_allocation_bytes)
        and int(snapshot.pool_free_bytes) > 0
    ):
        first_snapshot = snapshot
        snapshot = cupy_allocator_snapshot(
            cupy,
            apply_pool_limit=True,
            required_fresh_allocation_bytes=int(largest_allocation),
        )
        snapshot = replace(
            snapshot,
            pool_limit_applied=(
                bool(first_snapshot.pool_limit_applied) or bool(snapshot.pool_limit_applied)
            ),
            pool_trimmed_to_limit=(
                bool(first_snapshot.pool_trimmed_to_limit) or bool(snapshot.pool_trimmed_to_limit)
            ),
            pool_trimmed_for_fragmentation=(
                bool(first_snapshot.pool_trimmed_for_fragmentation)
                or bool(snapshot.pool_trimmed_for_fragmentation)
            ),
        )
        plan = _streamed_far_memory_plan(
            snapshot=snapshot,
            explicit_source_budget=explicit_source_budget,
            hf_start_incoming_bytes=int(hf_start_incoming_bytes),
            full_level_live_bytes=int(full_level_live_bytes),
            full_level_incoming_bytes=int(full_level_incoming_bytes),
        )
    plan = _clip_stream_plan_to_fresh_allocation_headroom(
        plan,
        multilevel=multilevel,
        nrhs=int(nrhs),
        guaranteed_fresh_allocation_bytes=int(snapshot.guaranteed_fresh_allocation_bytes),
    )
    return snapshot, plan


def _multilevel_full_incoming_bytes(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes for a fully materialized incoming hierarchy."""

    total = 0
    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = multilevel.levels[level_idx]
        total += (
            int(level.n_boxes)
            * 4
            * int(level.directional.grid.n_directions)
            * int(nrhs)
            * np.dtype(np.complex128).itemsize
        )
    return int(total)


def _multilevel_rolling_incoming_bytes_theoretical(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes for the two-parity rolling incoming hierarchy arenas."""

    hf_start = int(multilevel.hf_start_level)
    hf_end = int(multilevel.hf_end_level)
    max_elements_by_parity = [1, 1]
    for level_idx in range(hf_start, hf_end + 1):
        level = multilevel.levels[level_idx]
        parity = int((level_idx - hf_start) & 1)
        elements = int(level.n_boxes) * 4 * int(level.directional.grid.n_directions) * int(nrhs)
        max_elements_by_parity[parity] = max(max_elements_by_parity[parity], int(elements))
    bytes_per_complex = np.dtype(np.complex128).itemsize
    return int((max_elements_by_parity[0] + max_elements_by_parity[1]) * bytes_per_complex)


def _multilevel_outgoing_hierarchy_bytes(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes for the fully resident outgoing hierarchy buffers."""

    total = 0
    for level in multilevel.levels:
        total += (
            int(level.n_boxes)
            * 4
            * int(level.directional.grid.n_directions)
            * int(nrhs)
            * np.dtype(np.complex128).itemsize
        )
    return int(total)


def _multilevel_stream_full_level_incoming_bytes_theoretical(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes needed to keep one full sampled incoming level resident."""

    peak = 0
    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = multilevel.levels[level_idx]
        incoming_bytes = (
            int(level.n_boxes)
            * 4
            * int(level.directional.grid.n_directions)
            * int(nrhs)
            * np.dtype(np.complex128).itemsize
        )
        peak = max(int(peak), int(incoming_bytes))
    return int(peak)


def _multilevel_stream_full_level_live_bytes_theoretical(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes needed to stream one full sampled level as a single chunk."""

    peak = 0
    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = multilevel.levels[level_idx]
        live_bytes = (
            3
            * int(level.n_boxes)
            * 4
            * int(level.directional.grid.n_directions)
            * int(nrhs)
            * np.dtype(np.complex128).itemsize
        )
        peak = max(int(peak), int(live_bytes))
    return int(peak)


def _multilevel_rolling_outgoing_bytes_theoretical(
    multilevel: CuPyMLFMMMultilevelData,
    *,
    nrhs: int,
) -> int:
    """Return bytes for two-parity rolling outgoing hierarchy arenas."""

    hf_start = int(multilevel.hf_start_level)
    hf_end = int(multilevel.hf_end_level)
    max_elements_by_parity = [1, 1]
    for level_idx in range(hf_start, hf_end + 1):
        level = multilevel.levels[level_idx]
        parity = int((level_idx - hf_start) & 1)
        elements = int(level.n_boxes) * 4 * int(level.directional.grid.n_directions) * int(nrhs)
        max_elements_by_parity[parity] = max(max_elements_by_parity[parity], int(elements))
    bytes_per_complex = np.dtype(np.complex128).itemsize
    return int((max_elements_by_parity[0] + max_elements_by_parity[1]) * bytes_per_complex)


@dataclass
class CuPyMLFMMCouplingOperator:
    """CuPy-backed repeated-apply MLFMM coupling operator.

    The MLFMM plan and one-time operators are built on CPU (NumPy reference
    path). This class executes repeated exact-near and sampled-far applies on
    CuPy device arrays. The intended production configuration is on-the-fly
    leaf apply with streamed sampled-far traversal and compact host-summary
    retention; dense leaf payloads remain available only for validation/debug
    comparisons.

    Precision policy mirrors the NumPy MLFMM reference path:
    - exact-near runs at `near_dtype` (`complex64` or `complex128`);
    - sampled-far runs at `far_dtype` (currently fixed to `complex128`);
    - public output is cast to `dtype`.
    """

    lmax: int
    n_particles: int
    k: float
    prepared_data: CuPyMLFMMPreparedData
    host_cache: CuPyMLFMMHostCacheData | None = field(default=None, repr=False)
    host_cache_summary: dict[str, object] | None = field(default=None, repr=False)
    host_cache_policy: CuPyMLFMMHostCachePolicy = field(default_factory=CuPyMLFMMHostCachePolicy)
    dtype: np.dtype = COMPLEX128_DTYPE
    near_dtype: np.dtype = COMPLEX128_DTYPE
    far_dtype: np.dtype = COMPLEX128_DTYPE
    _receive_adjoint_cache: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _map_adjoint_cache: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _near_adjoint_context: _CuPyMLFMMNearAdjointContext | None = field(
        default=None, init=False, repr=False
    )
    _near_adjoint_blocks: dict[int, tuple[Any, Any, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _near_adjoint_cached_bytes: int = field(default=0, init=False, repr=False)
    _leaf_otf_pair_blocks_scratch: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_resolved_leaf_otf_bytes_budget: int | None = field(default=None, init=False, repr=False)
    _last_resolved_streamed_far_chunk_box_cap: int | None = field(
        default=None, init=False, repr=False
    )
    _last_resolved_streamed_far_frontier_box_cap: int | None = field(
        default=None, init=False, repr=False
    )
    _last_resolved_streamed_far_chunk_bytes_budget: int | None = field(
        default=None, init=False, repr=False
    )
    _last_resolved_streamed_far_frontier_bytes_budget: int | None = field(
        default=None, init=False, repr=False
    )
    _last_stream_memory_plan: _StreamedFarMemoryPlan | None = field(
        default=None, init=False, repr=False
    )
    _last_allocator_snapshot_before_stream_plan: CuPyAllocatorSnapshot | None = field(
        default=None, init=False, repr=False
    )
    _last_stream_stats: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_apply_timing_seconds: dict[str, float] | None = field(
        default=None, init=False, repr=False
    )
    _device_pool_peak_used_bytes: int = field(default=0, init=False, repr=False)
    _device_pool_peak_total_bytes: int = field(default=0, init=False, repr=False)
    _near_workspace_cache: dict[CuPyMLFMMNearWorkspaceKey, CuPyMLFMMNearWorkspace] = field(
        default_factory=dict, init=False, repr=False
    )
    _single_level_workspace_cache: dict[
        CuPyMLFMMSingleLevelWorkspaceKey, CuPyMLFMMSingleLevelWorkspace
    ] = field(default_factory=dict, init=False, repr=False)
    _multilevel_workspace_cache: dict[
        CuPyMLFMMMultilevelWorkspaceKey, CuPyMLFMMMultilevelWorkspace
    ] = field(default_factory=dict, init=False, repr=False)

    def _update_device_pool_peak(self, *, cupy: Any) -> None:
        """Track high-watermark device-pool bytes observed by this runtime object."""

        pool = cupy.get_default_memory_pool()
        pool_used = int(pool.used_bytes())
        pool_total = int(pool.total_bytes())
        if pool_used > int(self._device_pool_peak_used_bytes):
            self._device_pool_peak_used_bytes = int(pool_used)
        if pool_total > int(self._device_pool_peak_total_bytes):
            self._device_pool_peak_total_bytes = int(pool_total)

    def hierarchy_diagnostics(self) -> dict[str, object]:
        """Return sampled-level order and grid diagnostics for the CuPy MLFMM plan."""

        data = self.prepared_data
        if data.single_level is not None:
            single = data.single_level
            plan_summary = (self.host_cache_summary or {}).get("plan_summary")
            level_id = (
                int(plan_summary.get("selected_depth", 0)) if isinstance(plan_summary, dict) else 0
            )
            return {
                "stage": str(data.stage),
                "levels": {
                    "n_levels": 1,
                    "translator_orders": [int(single.translator_order)],
                    "grid_orders": [int(single.grid_order)],
                    "direction_counts": [int(single.directional.grid.n_directions)],
                    "levels": [
                        {
                            "level": level_id,
                            "n_boxes": int(single.n_leaves),
                            "box_order": int(single.box_order),
                            "translator_order": int(single.translator_order),
                            "grid_order": int(single.grid_order),
                            "n_directions": int(single.directional.grid.n_directions),
                            "far_offset_count": len(single.far_offset_batches),
                        }
                    ],
                },
            }
        multilevel = data.multilevel
        if multilevel is None:
            raise RuntimeError("Internal error: CuPy MLFMM prepared data has no far plan.")
        hf_start_level = int(multilevel.hf_start_level)
        return {
            "stage": str(data.stage),
            "leaf_level": int(multilevel.leaf_level),
            "hf_start_level": int(multilevel.hf_start_level),
            "hf_end_level": int(multilevel.hf_end_level),
            "transfer_edges": [
                {
                    "child_level": int(transfer.child_level),
                    "parent_level": int(transfer.parent_level),
                }
                for transfer in multilevel.transfers
            ],
            "levels": {
                "n_levels": len(multilevel.levels),
                "translator_orders": [int(level.translator_order) for level in multilevel.levels],
                "grid_orders": [int(level.grid_order) for level in multilevel.levels],
                "direction_counts": sorted(
                    {int(level.directional.grid.n_directions) for level in multilevel.levels}
                ),
                "levels": [
                    {
                        "level": int(level.level),
                        "n_boxes": int(level.n_boxes),
                        "box_order": int(level.box_order),
                        "translator_order": int(level.translator_order),
                        "grid_order": int(level.grid_order),
                        "n_directions": int(level.directional.grid.n_directions),
                        "far_offset_count": len(level.far_offset_batches),
                        "parity_from_hf_start": int((int(level.level) - hf_start_level) % 2),
                    }
                    for level_index, level in enumerate(multilevel.levels)
                ],
            },
        }

    def memory_diagnostics(self) -> dict[str, object]:
        """Return runtime memory diagnostics for the CuPy MLFMM operator."""

        cupy, _ = import_cupy()
        pinned_pool = cupy.get_default_pinned_memory_pool()
        allocator_info = cupy_allocator_memory_info(cupy)

        near_ws_total = int(
            sum(_device_array_nbytes(ws.y_states) for ws in self._near_workspace_cache.values())
        )
        single_ws_total = int(
            sum(
                _device_array_nbytes(ws.box_states)
                + _device_array_nbytes(ws.outgoing)
                + _device_array_nbytes(ws.incoming)
                + _device_array_nbytes(ws.incoming_box)
                + _device_array_nbytes(ws.y_states)
                for ws in self._single_level_workspace_cache.values()
            )
        )
        multilevel_ws_total = int(
            sum(
                _device_array_nbytes(ws.outgoing_roll_even)
                + _device_array_nbytes(ws.outgoing_roll_odd)
                + _device_array_nbytes(ws.incoming_roll_even)
                + _device_array_nbytes(ws.incoming_roll_odd)
                + _device_array_nbytes(ws.leaf_box_states)
                + _device_array_nbytes(ws.incoming_box)
                + _device_array_nbytes(ws.y_states)
                for ws in self._multilevel_workspace_cache.values()
            )
        )
        leaf_otf_scratch_bytes = _device_array_nbytes(
            self._leaf_otf_pair_blocks_scratch.get("buffer")
        )
        near_adjoint_blocks_bytes = int(
            sum(
                _device_array_nbytes(dst_ids)
                + _device_array_nbytes(src_ids)
                + _device_array_nbytes(pair_blocks)
                for dst_ids, src_ids, pair_blocks in self._near_adjoint_blocks.values()
            )
        )
        map_adjoint_cache_bytes = int(
            sum(_device_array_nbytes(value) for value in self._map_adjoint_cache.values())
        )

        multilevel = self.prepared_data.multilevel
        rolling_diag: dict[str, object] | None = None
        streaming_diag: dict[str, object] | None = None
        if multilevel is not None:
            if self._multilevel_workspace_cache:
                nrhs_ref = int(next(iter(self._multilevel_workspace_cache.values())).nrhs)
                rolling_actual = int(
                    max(
                        _device_array_nbytes(ws.incoming_roll_even)
                        + _device_array_nbytes(ws.incoming_roll_odd)
                        for ws in self._multilevel_workspace_cache.values()
                    )
                )
                rolling_outgoing_actual = int(
                    max(
                        _device_array_nbytes(ws.outgoing_roll_even)
                        + _device_array_nbytes(ws.outgoing_roll_odd)
                        for ws in self._multilevel_workspace_cache.values()
                    )
                )
            else:
                nrhs_ref = 1
                rolling_actual = None
                rolling_outgoing_actual = None
            full_incoming = _multilevel_full_incoming_bytes(multilevel, nrhs=nrhs_ref)
            rolling_theoretical = _multilevel_rolling_incoming_bytes_theoretical(
                multilevel, nrhs=nrhs_ref
            )
            outgoing_hierarchy = _multilevel_outgoing_hierarchy_bytes(multilevel, nrhs=nrhs_ref)
            rolling_outgoing_theoretical = _multilevel_rolling_outgoing_bytes_theoretical(
                multilevel, nrhs=nrhs_ref
            )
            streamed_chunk_local = str(multilevel.leaf_apply_mode) == "on_the_fly"
            leaf_otf_chunk_summary = (
                None
                if not streamed_chunk_local
                else _leaf_otf_resolved_chunk_summary(
                    leaf_groups=multilevel.leaf_groups,
                    leaf_otf_chunk_leaves=self.host_cache_policy.leaf_otf_chunk_leaves,
                )
            )
            if streamed_chunk_local:
                rolling_actual = None
                rolling_outgoing_actual = None
                rolling_used = None
                rolling_outgoing_used = None
            else:
                rolling_used = (
                    rolling_theoretical if rolling_actual is None else int(rolling_actual)
                )
                rolling_outgoing_used = (
                    rolling_outgoing_theoretical
                    if rolling_outgoing_actual is None
                    else int(rolling_outgoing_actual)
                )
            rolling_diag = {
                "execution_mode": "streamed_chunk_local"
                if streamed_chunk_local
                else "rolling_resident",
                "nrhs_reference": int(nrhs_ref),
                "full_incoming_hierarchy_bytes": int(full_incoming),
                "rolling_incoming_bytes_theoretical": int(rolling_theoretical),
                "rolling_incoming_bytes_actual_peak": (
                    None if rolling_actual is None else int(rolling_actual)
                ),
                "incoming_reduction_ratio": (
                    None
                    if rolling_used is None or full_incoming <= 0
                    else float(rolling_used) / float(full_incoming)
                ),
                "outgoing_hierarchy_bytes": int(outgoing_hierarchy),
                "rolling_outgoing_bytes_theoretical": int(rolling_outgoing_theoretical),
                "rolling_outgoing_bytes_actual_peak": (
                    None if rolling_outgoing_actual is None else int(rolling_outgoing_actual)
                ),
                "outgoing_reduction_ratio": (
                    None
                    if rolling_outgoing_used is None or outgoing_hierarchy <= 0
                    else float(rolling_outgoing_used) / float(outgoing_hierarchy)
                ),
                "full_far_hierarchy_bytes": int(full_incoming + outgoing_hierarchy),
                "rolling_far_hierarchy_bytes": (
                    None
                    if rolling_used is None or rolling_outgoing_used is None
                    else int(rolling_used + rolling_outgoing_used)
                ),
            }
            streaming_diag = {
                "leaf_apply_mode": str(multilevel.leaf_apply_mode),
                "collect_stream_stats": bool(self.host_cache_policy.collect_stream_stats),
                "device_limit_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.device_limit_bytes)
                ),
                "pool_limit_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.pool_limit_bytes)
                ),
                "pool_limit_applied_by_mlfmm": (
                    None
                    if self._last_stream_memory_plan is None
                    else bool(self._last_stream_memory_plan.pool_limit_applied)
                ),
                "pool_trimmed_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else bool(
                        self._last_allocator_snapshot_before_stream_plan.pool_trimmed_to_limit
                        or self._last_allocator_snapshot_before_stream_plan.pool_trimmed_for_fragmentation
                    )
                ),
                "pool_trimmed_for_fragmentation": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else bool(
                        self._last_allocator_snapshot_before_stream_plan.pool_trimmed_for_fragmentation
                    )
                ),
                "pool_used_bytes_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else int(self._last_allocator_snapshot_before_stream_plan.pool_used_bytes)
                ),
                "pool_total_bytes_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else int(self._last_allocator_snapshot_before_stream_plan.pool_total_bytes)
                ),
                "pool_free_bytes_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else int(self._last_allocator_snapshot_before_stream_plan.pool_free_bytes)
                ),
                "raw_free_bytes_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else int(self._last_allocator_snapshot_before_stream_plan.raw_free_bytes)
                ),
                "raw_total_bytes_before_stream_plan": (
                    None
                    if self._last_allocator_snapshot_before_stream_plan is None
                    else int(self._last_allocator_snapshot_before_stream_plan.raw_total_bytes)
                ),
                "active_used_bytes_before_far": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.active_used_bytes_before_far)
                ),
                "stream_transient_budget_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.stream_transient_budget_bytes)
                ),
                "stream_safety_margin_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.safety_margin_bytes)
                ),
                "stream_unmodeled_temp_reserve_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.unmodeled_temp_reserve_bytes)
                ),
                "stream_memory_retry_count": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.retry_count)
                ),
                "source_outgoing_budget_clipped": (
                    None
                    if self._last_stream_memory_plan is None
                    else bool(self._last_stream_memory_plan.source_outgoing_budget_clipped)
                ),
                "source_outgoing_budget_fragmentation_clipped": (
                    None
                    if self._last_stream_memory_plan is None
                    else bool(
                        self._last_stream_memory_plan.source_outgoing_budget_fragmentation_clipped
                    )
                ),
                "largest_single_stream_allocation_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.largest_single_allocation_bytes)
                ),
                "guaranteed_fresh_allocation_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.guaranteed_fresh_allocation_bytes)
                ),
                "stream_fragmentation_guard_bytes": (
                    None
                    if self._last_stream_memory_plan is None
                    else int(self._last_stream_memory_plan.fragmentation_guard_bytes)
                ),
                "resolved_streamed_far_chunk_box_cap": (
                    None
                    if self._last_resolved_streamed_far_chunk_box_cap is None
                    else int(self._last_resolved_streamed_far_chunk_box_cap)
                ),
                "resolved_streamed_far_frontier_box_cap": (
                    None
                    if self._last_resolved_streamed_far_frontier_box_cap is None
                    else int(self._last_resolved_streamed_far_frontier_box_cap)
                ),
                "resolved_leaf_otf_chunk_leaves_by_occupancy": (
                    None
                    if leaf_otf_chunk_summary is None
                    else dict(cast(dict[str, int], leaf_otf_chunk_summary["by_occupancy"]))
                ),
                "resolved_leaf_otf_chunk_leaves_min": (
                    None
                    if leaf_otf_chunk_summary is None
                    else int(cast(int, leaf_otf_chunk_summary["min"]))
                ),
                "resolved_leaf_otf_chunk_leaves_max": (
                    None
                    if leaf_otf_chunk_summary is None
                    else int(cast(int, leaf_otf_chunk_summary["max"]))
                ),
                "resolved_streamed_far_chunk_bytes_budget": (
                    None
                    if self._last_resolved_streamed_far_chunk_bytes_budget is None
                    else int(self._last_resolved_streamed_far_chunk_bytes_budget)
                ),
                "resolved_streamed_far_frontier_bytes_budget": (
                    None
                    if self._last_resolved_streamed_far_frontier_bytes_budget is None
                    else int(self._last_resolved_streamed_far_frontier_bytes_budget)
                ),
                "last_apply_stats": (
                    None if self._last_stream_stats is None else dict(self._last_stream_stats)
                ),
                "last_apply_timing_seconds": (
                    None
                    if self._last_apply_timing_seconds is None
                    else dict(self._last_apply_timing_seconds)
                ),
            }

        return {
            "host_cache_leaf_apply_mode": str(self.host_cache_policy.leaf_apply_mode),
            "host_cache_retention": str(self.host_cache_policy.host_cache_retention),
            "host_cache_retained": bool(self.host_cache is not None),
            "host_cache_summary": (
                None if self.host_cache_summary is None else dict(self.host_cache_summary)
            ),
            "leaf_otf_chunk_leaves": (
                None
                if self.host_cache_policy.leaf_otf_chunk_leaves is None
                else int(self.host_cache_policy.leaf_otf_chunk_leaves)
            ),
            "leaf_otf_bytes_budget": (
                None
                if self.host_cache_policy.leaf_otf_bytes_budget is None
                else int(self.host_cache_policy.leaf_otf_bytes_budget)
            ),
            "streamed_far_chunk_bytes_budget": (
                None
                if self.host_cache_policy.streamed_far_chunk_bytes_budget is None
                else int(self.host_cache_policy.streamed_far_chunk_bytes_budget)
            ),
            "near_adjoint_cache_bytes_budget": (
                None
                if self.host_cache_policy.near_adjoint_cache_bytes_budget is None
                else int(self.host_cache_policy.near_adjoint_cache_bytes_budget)
            ),
            "streamed_far_frontier_bytes_budget": (
                None
                if self._last_resolved_streamed_far_frontier_bytes_budget is None
                else int(self._last_resolved_streamed_far_frontier_bytes_budget)
            ),
            "collect_stream_stats": bool(self.host_cache_policy.collect_stream_stats),
            "device_pool": {
                "used_bytes": int(allocator_info["pool_used_bytes"]),
                "total_bytes": int(allocator_info["pool_total_bytes"]),
                "cached_bytes": int(allocator_info["pool_cached_bytes"]),
                "peak_used_bytes_seen_by_operator": int(self._device_pool_peak_used_bytes),
                "peak_total_bytes_seen_by_operator": int(self._device_pool_peak_total_bytes),
                "pinned_free_blocks": int(pinned_pool.n_free_blocks()),
            },
            "device_mem_info": {
                "free_bytes": int(allocator_info["free_bytes"]),
                "effective_free_bytes": int(allocator_info["effective_free_bytes"]),
                "total_bytes": int(allocator_info["total_bytes"]),
            },
            "workspace_cache_entries": {
                "near": len(self._near_workspace_cache),
                "single_level": len(self._single_level_workspace_cache),
                "multilevel": len(self._multilevel_workspace_cache),
            },
            "workspace_bytes": {
                "near_total_bytes": int(near_ws_total),
                "single_level_total_bytes": int(single_ws_total),
                "multilevel_total_bytes": int(multilevel_ws_total),
                "leaf_otf_pair_blocks_scratch_bytes": int(leaf_otf_scratch_bytes),
                "near_adjoint_blocks_bytes": int(near_adjoint_blocks_bytes),
                "map_adjoint_cache_bytes": int(map_adjoint_cache_bytes),
                "total_bytes": int(
                    near_ws_total
                    + single_ws_total
                    + multilevel_ws_total
                    + leaf_otf_scratch_bytes
                    + near_adjoint_blocks_bytes
                    + map_adjoint_cache_bytes
                ),
            },
            "multilevel_rolling": rolling_diag,
            "multilevel_streaming": streaming_diag,
        }

    def apply(self, x: Any) -> Any:
        cupy, _ = import_cupy()
        collect_apply_timing = bool(self.host_cache_policy.collect_stream_stats)

        def start_timer() -> float:
            return time.perf_counter() if collect_apply_timing else 0.0

        def elapsed_after_device_work(started: float) -> float:
            if not collect_apply_timing:
                return 0.0
            cupy.cuda.Stream.null.synchronize()
            return time.perf_counter() - float(started)

        apply_started = start_timer()
        self._update_device_pool_peak(cupy=cupy)
        out_dtype = np.dtype(self.dtype)
        near_dtype = np.dtype(self.near_dtype)
        far_dtype = np.dtype(self.far_dtype)
        if out_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise ValueError(
                "CuPy MLFMM apply supports only complex64/complex128 output dtypes "
                f"(got {out_dtype!r})."
            )
        if near_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise ValueError(
                "CuPy MLFMM apply supports only complex64/complex128 near dtypes "
                f"(got {near_dtype!r})."
            )
        if far_dtype != np.dtype(np.complex128):
            raise ValueError(
                "CuPy MLFMM sampled-far apply requires complex128 far precision "
                f"(got {far_dtype!r})."
            )
        nm = n_modes(int(self.lmax))
        n_particles = int(self.n_particles)
        x_states, squeezed = _reshape_unknowns_to_particle_modes(
            x,
            n_particles=n_particles,
            nm=nm,
            dtype=out_dtype,
            cupy=cupy,
        )
        setup_started = start_timer()
        near_ws = _ensure_exact_near_workspace(
            self.prepared_data,
            n_particles=n_particles,
            nm=nm,
            nrhs=int(x_states.shape[2]),
            near_dtype=near_dtype,
            cache=self._near_workspace_cache,
            cupy=cupy,
        )
        setup_elapsed = elapsed_after_device_work(setup_started)
        near_started = start_timer()
        y_near = _apply_exact_near_pairs(
            self.prepared_data,
            x_states,
            workspace=near_ws,
            cupy=cupy,
        )
        near_elapsed = elapsed_after_device_work(near_started)
        stage = str(self.prepared_data.stage)
        resolved_leaf_otf_bytes_budget: int | None = None
        resolved_streamed_far_chunk_box_cap: int | None = None
        resolved_streamed_far_frontier_box_cap: int | None = None
        resolved_streamed_far_chunk_bytes_budget: int | None = None
        resolved_streamed_far_frontier_bytes_budget: int | None = None
        stream_memory_plan: _StreamedFarMemoryPlan | None = None
        allocator_snapshot_before_stream_plan: CuPyAllocatorSnapshot | None = None
        stream_stats: dict[str, object] | None = None
        far_setup_elapsed = 0.0
        if stage == "single_level":
            far_setup_started = start_timer()
            single_ws = _ensure_single_level_workspace(
                self.prepared_data,
                n_particles=n_particles,
                nm=nm,
                nrhs=int(x_states.shape[2]),
                cache=self._single_level_workspace_cache,
                cupy=cupy,
            )
            far_setup_elapsed += elapsed_after_device_work(far_setup_started)
            single = self.prepared_data.single_level
            if single is None:
                raise RuntimeError("Internal CuPy MLFMM error: missing single-level prepared data.")
            if str(single.leaf_apply_mode) == "on_the_fly":
                free_bytes = cupy_allocator_memory_info(cupy)["effective_free_bytes"]
                resolved_leaf_otf_bytes_budget = _resolve_stream_bytes_budget(
                    explicit_budget=self.host_cache_policy.leaf_otf_bytes_budget,
                    free_bytes=int(free_bytes),
                    fraction=_SINGLE_LEVEL_LEAF_OTF_FRACTION_OF_EFFECTIVE_FREE,
                    minimum_bytes=_SINGLE_LEVEL_LEAF_OTF_MIN_BYTES,
                    maximum_bytes=_SINGLE_LEVEL_LEAF_OTF_MAX_BYTES,
                )
            far_started = start_timer()
            y_far = _apply_single_level_far(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                pair_blocks_scratch=self._leaf_otf_pair_blocks_scratch,
                leaf_otf_chunk_leaves=self.host_cache_policy.leaf_otf_chunk_leaves,
                leaf_otf_bytes_budget=resolved_leaf_otf_bytes_budget,
                workspace=single_ws,
                cupy=cupy,
            )
            far_elapsed = elapsed_after_device_work(far_started)
        elif stage == "multilevel":
            far_setup_started = start_timer()
            multi_ws = _ensure_multilevel_workspace(
                self.prepared_data,
                n_particles=n_particles,
                nm=nm,
                nrhs=int(x_states.shape[2]),
                cache=self._multilevel_workspace_cache,
                cupy=cupy,
            )
            far_setup_elapsed += elapsed_after_device_work(far_setup_started)
            multilevel = self.prepared_data.multilevel
            if multilevel is None:
                raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
            if str(multilevel.leaf_apply_mode) == "on_the_fly":
                nrhs = int(x_states.shape[2])
                full_level_live_bytes = _multilevel_stream_full_level_live_bytes_theoretical(
                    multilevel,
                    nrhs=nrhs,
                )
                full_level_incoming_bytes = (
                    _multilevel_stream_full_level_incoming_bytes_theoretical(
                        multilevel,
                        nrhs=nrhs,
                    )
                )
                hf_start_level = multilevel.levels[int(multilevel.hf_start_level)]
                hf_end_level = multilevel.levels[int(multilevel.hf_end_level)]
                hf_start_incoming_bytes = (
                    int(hf_start_level.n_boxes)
                    * 4
                    * int(hf_start_level.directional.grid.n_directions)
                    * nrhs
                    * np.dtype(np.complex128).itemsize
                )
                # Source/outgoing chunks and frontier incoming buffers share
                # one guarded device-residency plan. This keeps automatic
                # chunking below the CuPy pool limit instead of letting one
                # component consume all freed headroom and spill on WDDM.
                (
                    allocator_snapshot_before_stream_plan,
                    stream_memory_plan,
                ) = _resolve_streamed_far_memory_plan(
                    cupy,
                    multilevel=multilevel,
                    nrhs=int(nrhs),
                    explicit_source_budget=self.host_cache_policy.streamed_far_chunk_bytes_budget,
                    hf_start_incoming_bytes=int(hf_start_incoming_bytes),
                    full_level_live_bytes=int(full_level_live_bytes),
                    full_level_incoming_bytes=int(full_level_incoming_bytes),
                )
                resolved_streamed_far_frontier_bytes_budget = (
                    stream_memory_plan.frontier_bytes_budget
                )
                resolved_streamed_far_frontier_box_cap = min(
                    int(hf_start_level.n_boxes),
                    _level_group_box_cap(
                        level=hf_start_level,
                        nrhs=nrhs,
                        bytes_budget=int(resolved_streamed_far_frontier_bytes_budget),
                    ),
                )
                resolved_streamed_far_chunk_bytes_budget = (
                    stream_memory_plan.source_outgoing_bytes_budget
                )
                resolved_streamed_far_chunk_box_cap = min(
                    int(hf_end_level.n_boxes),
                    _level_chunk_box_cap(
                        level=hf_end_level,
                        nrhs=nrhs,
                        bytes_budget=int(resolved_streamed_far_chunk_bytes_budget),
                    ),
                )
                if bool(self.host_cache_policy.collect_stream_stats):
                    stream_stats = {}
            far_started = start_timer()
            y_far = _apply_multilevel_far(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                pair_blocks_scratch=self._leaf_otf_pair_blocks_scratch,
                leaf_otf_chunk_leaves=self.host_cache_policy.leaf_otf_chunk_leaves,
                leaf_otf_bytes_budget=resolved_leaf_otf_bytes_budget,
                streamed_far_chunk_bytes_budget=(
                    0
                    if resolved_streamed_far_chunk_bytes_budget is None
                    else int(resolved_streamed_far_chunk_bytes_budget)
                ),
                streamed_far_frontier_bytes_budget=(
                    0
                    if resolved_streamed_far_frontier_bytes_budget is None
                    else int(resolved_streamed_far_frontier_bytes_budget)
                ),
                workspace=multi_ws,
                cupy=cupy,
                stream_stats=stream_stats,
            )
            far_elapsed = elapsed_after_device_work(far_started)
        else:
            raise RuntimeError(f"Unsupported CuPy MLFMM stage {stage!r}.")
        combine_started = start_timer()
        self._last_resolved_leaf_otf_bytes_budget = resolved_leaf_otf_bytes_budget
        self._last_stream_memory_plan = stream_memory_plan
        self._last_allocator_snapshot_before_stream_plan = allocator_snapshot_before_stream_plan
        self._last_resolved_streamed_far_chunk_box_cap = resolved_streamed_far_chunk_box_cap
        self._last_resolved_streamed_far_frontier_box_cap = resolved_streamed_far_frontier_box_cap
        self._last_resolved_streamed_far_chunk_bytes_budget = (
            resolved_streamed_far_chunk_bytes_budget
        )
        self._last_resolved_streamed_far_frontier_bytes_budget = (
            resolved_streamed_far_frontier_bytes_budget
        )
        if stream_stats is not None:
            stream_stats.pop("_internal_same_level_source_union_history", None)
        self._last_stream_stats = None if stream_stats is None else dict(stream_stats)
        y_total = cupy.asarray(y_far, dtype=cupy.complex128)
        y_total += cupy.asarray(y_near, dtype=cupy.complex128)
        combine_elapsed = elapsed_after_device_work(combine_started)
        self._update_device_pool_peak(cupy=cupy)
        if collect_apply_timing:
            total_elapsed = time.perf_counter() - apply_started
            self._last_apply_timing_seconds = {
                "total": float(total_elapsed),
                "setup": float(setup_elapsed + far_setup_elapsed),
                "exact_near": float(near_elapsed),
                "sampled_far": float(far_elapsed),
                "combine": float(combine_elapsed),
            }
        else:
            self._last_apply_timing_seconds = None
        return _restore_unknown_shape(
            y_total.astype(_cupy_complex_dtype(out_dtype, cupy=cupy), copy=False),
            squeezed=squeezed,
        )

    def apply_adjoint(self, x: Any) -> Any:
        """Apply the Hermitian adjoint of the finite MLFMM coupling on device.

        Sampled-far reverse traversals currently retain all hierarchy levels for
        the duration of this action. This is intentionally the first correct
        implementation; the forward path remains streamed and memory-bounded,
        while a later optimization can introduce the analogous reverse
        frontier traversal once validation establishes the algebra.
        """

        cupy, _ = import_cupy()
        out_dtype = np.dtype(self.dtype)
        near_dtype = np.dtype(self.near_dtype)
        far_dtype = np.dtype(self.far_dtype)
        if out_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise ValueError(f"Unsupported CuPy MLFMM output dtype {out_dtype!r}.")
        if near_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise ValueError(f"Unsupported CuPy MLFMM near dtype {near_dtype!r}.")
        if far_dtype != np.dtype(np.complex128):
            raise ValueError("CuPy MLFMM adjoint requires complex128 sampled-far precision.")
        nm = n_modes(int(self.lmax))
        x_states, squeezed = _reshape_unknowns_to_particle_modes(
            x,
            n_particles=int(self.n_particles),
            nm=nm,
            dtype=out_dtype,
            cupy=cupy,
        )
        near_context = self._near_adjoint_context
        if near_context is None:
            near_context = _build_exact_near_adjoint_context(
                self.prepared_data,
                lmax=int(self.lmax),
                k=float(self.k),
                near_dtype=near_dtype,
                cupy=cupy,
            )
            self._near_adjoint_context = near_context
        near_ws = _ensure_exact_near_workspace(
            self.prepared_data,
            n_particles=int(self.n_particles),
            nm=nm,
            nrhs=int(x_states.shape[2]),
            near_dtype=near_dtype,
            cache=self._near_workspace_cache,
            cupy=cupy,
        )
        _, self._near_adjoint_cached_bytes = _apply_exact_near_adjoint_streaming(
            near_context,
            self._near_adjoint_blocks,
            x_states,
            out=near_ws.y_states,
            cache_bytes=int(self._near_adjoint_cached_bytes),
            cache_budget=self.host_cache_policy.near_adjoint_cache_bytes_budget,
            cupy=cupy,
        )
        y_near = near_ws.y_states
        stage = str(self.prepared_data.stage)
        if stage == "single_level":
            y_far = _apply_single_level_far_adjoint(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                pair_blocks_scratch=self._leaf_otf_pair_blocks_scratch,
                leaf_otf_chunk_leaves=self.host_cache_policy.leaf_otf_chunk_leaves,
                leaf_otf_bytes_budget=self.host_cache_policy.leaf_otf_bytes_budget,
                map_adjoint_cache=self._map_adjoint_cache,
                cupy=cupy,
            )
        elif stage == "multilevel":
            y_far = _apply_multilevel_far_adjoint(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                pair_blocks_scratch=self._leaf_otf_pair_blocks_scratch,
                leaf_otf_chunk_leaves=self.host_cache_policy.leaf_otf_chunk_leaves,
                leaf_otf_bytes_budget=self.host_cache_policy.leaf_otf_bytes_budget,
                map_adjoint_cache=self._map_adjoint_cache,
                cupy=cupy,
            )
        else:
            raise RuntimeError(f"Unsupported CuPy MLFMM stage {stage!r}.")
        y_total = cupy.asarray(y_far, dtype=cupy.complex128)
        y_total += cupy.asarray(y_near, dtype=cupy.complex128)
        return _restore_unknown_shape(
            y_total.astype(_cupy_complex_dtype(out_dtype, cupy=cupy), copy=False),
            squeezed=squeezed,
        )

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError(
            "CuPyMLFMMCouplingOperator is runtime-only and intentionally non-picklable. "
            "Persist the compact host cache explicitly if debug serialization is required."
        )

    def __setstate__(self, _state: dict[str, Any]) -> None:
        raise TypeError(
            "CuPyMLFMMCouplingOperator cannot be unpickled. Rebuild it from a CPU MLFMM "
            "coupling plan or from an explicit compact host cache payload."
        )


def prepare_mlfmm_cupy_coupling(
    coupling: MLFMMCouplingOperator,
    *,
    host_cache_policy: CuPyMLFMMHostCachePolicy | None = None,
) -> CuPyMLFMMCouplingOperator:
    """Wrap a CPU-built MLFMM coupling plan in a CuPy repeated-apply operator."""

    policy = _resolve_host_cache_policy(host_cache_policy)
    host_cache = _build_mlfmm_cupy_host_cache(coupling, host_cache_policy=policy)
    prepared = prepare_mlfmm_cupy_data(host_cache, host_cache_policy=policy)
    retention = str(policy.host_cache_retention)
    host_cache_summary = (
        None if retention == "none" else _runtime_host_cache_summary(host_cache, policy=policy)
    )
    return CuPyMLFMMCouplingOperator(
        lmax=int(coupling.lmax),
        n_particles=int(np.asarray(coupling.positions).shape[0]),
        prepared_data=prepared,
        k=float(coupling.k),
        host_cache=(host_cache if retention == "full" else None),
        host_cache_summary=host_cache_summary,
        host_cache_policy=policy,
        dtype=np.dtype(coupling.dtype),
        near_dtype=np.dtype(coupling.near_dtype),
        far_dtype=np.dtype(coupling.far_dtype),
    )


def build_mlfmm_cupy_host_cache(
    coupling: MLFMMCouplingOperator,
    *,
    host_cache_policy: CuPyMLFMMHostCachePolicy | None = None,
) -> CuPyMLFMMHostCacheData:
    """Build compact host-only reusable MLFMM state for explicit debug workflows."""

    return _build_mlfmm_cupy_host_cache(coupling, host_cache_policy=host_cache_policy)


def prepare_mlfmm_cupy_data(
    source: MLFMMCouplingOperator | CuPyMLFMMHostCacheData,
    *,
    host_cache_policy: CuPyMLFMMHostCachePolicy | None = None,
) -> CuPyMLFMMPreparedData:
    """Upload repeated-apply MLFMM structures from a CPU-built coupling plan.

    Parameters
    ----------
    source:
        Either the CPU-built MLFMM coupling object from
        `prepare_mlfmm_coupling(...)`, or a compact host cache payload.

    Returns
    -------
    CuPyMLFMMPreparedData
        Device-resident representation of all repeated-apply data needed by
        the CuPy MLFMM apply path. Exact-near payload dtype follows
        `coupling.near_dtype`; sampled-far payloads stay on complex128.
    """

    cupy, _ = import_cupy()
    cupyx_sparse = import_module("cupyx.scipy.sparse")
    policy = _resolve_host_cache_policy(host_cache_policy)
    host_cache = (
        source
        if isinstance(source, CuPyMLFMMHostCacheData)
        else _build_mlfmm_cupy_host_cache(source, host_cache_policy=policy)
    )
    near_dtype = np.dtype(host_cache.near_dtype)
    if near_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError(
            "CuPy MLFMM preparation supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    if np.dtype(host_cache.far_dtype) != np.dtype(np.complex128):
        raise ValueError(
            "CuPy MLFMM preparation requires complex128 sampled-far precision. "
            f"Got far_dtype={np.dtype(host_cache.far_dtype)!r}."
        )
    stage = str(host_cache.stage)
    if stage not in {"single_level", "multilevel"}:
        raise ValueError(
            "CuPy MLFMM preparation requires a non-direct MLFMM stage. "
            f"Resolved stage is {stage!r}."
        )

    partition_data = _partition_from_host_cache(host_cache)
    near_pairs = _upload_exact_near_pair_data_from_host_cache(host_cache, cupy=cupy)
    single_level_data: CuPyMLFMMSingleLevelData | None = None
    multilevel_data: CuPyMLFMMMultilevelData | None = None
    if stage == "single_level":
        if host_cache.single_level is None:
            raise ValueError("single_level stage was selected but no single-level operators exist.")
        single_level_data = _upload_single_level(host_cache.single_level, partition_data, cupy=cupy)
    else:
        if host_cache.multilevel is None:
            raise ValueError("multilevel stage was selected but no multilevel operators exist.")
        multilevel_data = _upload_multilevel(
            host_cache.multilevel, partition_data, cupy=cupy, cupyx_sparse=cupyx_sparse
        )

    return CuPyMLFMMPreparedData(
        lmax=int(host_cache.lmax),
        stage=stage,
        near_pairs=near_pairs,
        single_level=single_level_data,
        multilevel=multilevel_data,
    )


__all__ = [
    "CuPyDirectionalGridData",
    "CuPyDirectionalInterpolationData",
    "CuPyDirectionalTransformsData",
    "CuPyLeafApplyGroupData",
    "CuPyMLFMMCouplingOperator",
    "CuPyMLFMMHostCacheData",
    "CuPyMLFMMHostCachePolicy",
    "CuPyMLFMMLevelData",
    "CuPyMLFMMMultilevelData",
    "CuPyMLFMMNearPairData",
    "CuPyMLFMMPartitionData",
    "CuPyMLFMMPreparedData",
    "CuPyMLFMMSingleLevelData",
    "CuPyMLFMMTransferData",
    "build_mlfmm_cupy_host_cache",
    "prepare_mlfmm_cupy_coupling",
    "prepare_mlfmm_cupy_data",
]
