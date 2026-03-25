from __future__ import annotations

"""CuPy-side prepared-data wrappers for the NumPy MLFMM reference plan.

CPU build/planning remains the single source of truth in `mlfmm.py`.
This module validates and uploads repeated-apply structures to device memory.
"""

from dataclasses import dataclass, field
from functools import cache
from importlib import import_module
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyceles._optional import coerce_array, import_cupy
from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
)

from .mlfmm import (
    MLFMMCouplingOperator,
    MLFMMLevelOperators,
    MLFMMMultilevelOperators,
    MLFMMSingleLevelOperators,
    MLFMMTransferOperators,
)
from .mlfmm_directional import MLFMMDirectionalTransforms
from .mlfmm_partition import MLFMMPartition

Offset3 = tuple[int, int, int]
CuPyMLFMMHostMemoryBudget = Literal["balanced", "low_host_memory"]


@dataclass(frozen=True)
class CuPyMLFMMHostCachePolicy:
    """Memory-tier policy for compact host cache retention vs recomputation.

    `balanced` keeps static exact-near tables in host RAM to minimize rebuild
    overhead. `low_host_memory` drops those static tables from host cache and
    recomputes them during device upload.
    """

    memory_budget: CuPyMLFMMHostMemoryBudget = "balanced"


@dataclass(frozen=True)
class CuPyDirectionalGridData:
    """Device copy of one directional sampling grid."""

    order: int
    n_directions: int
    reflection_permutation: Any


@dataclass(frozen=True)
class CuPyHostDirectionalGridData:
    """Compact host directional-grid payload used for cache serialization."""

    order: int
    n_directions: int
    reflection_permutation: np.ndarray


@dataclass(frozen=True)
class CuPyHostDirectionalTransformsData:
    """Compact host directional transform payload with canonical operators only."""

    box_order: int
    grid_order: int
    grid: CuPyHostDirectionalGridData
    Fth: np.ndarray
    Fph: np.ndarray
    Gth: np.ndarray
    Gph: np.ndarray


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
    """Compact host single-level payload for CuPy upload."""

    box_order: int
    translator_order: int
    grid_order: int
    directional: CuPyHostDirectionalTransformsData
    aggregation: tuple[np.ndarray, ...]
    far_offset_batches: dict[Offset3, tuple[np.ndarray, np.ndarray]]
    offset_diagonals: dict[Offset3, np.ndarray]


@dataclass(frozen=True)
class CuPyHostMultilevelData:
    """Compact host multilevel payload for CuPy upload."""

    levels: tuple[CuPyHostLevelData, ...]
    transfers: tuple[CuPyHostTransferData, ...]
    leaf_level: int
    hf_start_level: int
    hf_end_level: int
    aggregation: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CuPyMLFMMHostCacheData:
    """Compact host-only cache artifact for CuPy MLFMM preparation."""

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
    host_memory_budget: CuPyMLFMMHostMemoryBudget = "balanced"
    single_level: CuPyHostSingleLevelData | None = None
    multilevel: CuPyHostMultilevelData | None = None
    plan_summary: dict[str, int | float | str] | None = None


@dataclass(frozen=True)
class CuPyDirectionalTransformsData:
    """Device directional transform payload for one box/grid order.

    Matrices are prepacked for batched GEMM on repeated apply:
    - `forward_F` maps one scalar SVWF block (`nscl`) to two reflected
      directional channels (`theta`, `phi`) with shape `(2*ndir, nscl)`.
    - `inverse_A_adj` and `inverse_G_adj` are reflected adjoint blocks used by
      the inverse map with shapes `(nscl, 2*ndir)`.
    """

    box_order: int
    grid_order: int
    grid: CuPyDirectionalGridData
    nscl: int
    forward_F: Any
    inverse_A_adj: Any
    inverse_G_adj: Any


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
    """Grouped leaf data with uniform occupancy for batched leaf GEMMs."""

    occupancy: int
    nmodes: int
    leaf_ids: Any
    particle_indices: Any
    aggregation: Any


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


@dataclass(frozen=True)
class CuPyMLFMMNearPairData:
    """Device-ready directed exact-near leaf schedule and translation tables."""

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
    """Reusable multilevel far-path work buffers for one RHS width."""

    nrhs: int
    outgoing: list[Any]
    incoming: list[Any]
    leaf_box_states: Any
    incoming_box: Any
    y_states: Any


@dataclass
class CuPyMLFMMNearWorkspace:
    """Reusable exact-near work buffers for one RHS width."""

    nrhs: int
    x_states: Any
    y_states: Any


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
    n_leaves: int
    box_nm: int
    level_shapes: tuple[tuple[int, int], ...]


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


@cache
def _mode_metadata_tables(lmax: int) -> np.ndarray:
    """Return CELES/SMUTHI `m` mode indices indexed by flattened mode id."""

    mode_m = np.zeros((n_modes(lmax),), dtype=np.int32)
    for _tau_i, _l_i, m_i, idx in iter_modes(lmax):
        mode_m[idx] = m_i
    mode_m.setflags(write=False)
    return mode_m


@cache
def _mode_pair_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact p-range metadata for each `(n1, n2)` mode pair."""

    nmodes_total = n_modes(lmax)
    mode_m = _mode_metadata_tables(lmax)
    mode_tau = np.zeros((nmodes_total,), dtype=np.int32)
    mode_l = np.zeros((nmodes_total,), dtype=np.int32)
    for tau_i, l_i, _m_i, idx in iter_modes(lmax):
        mode_tau[idx] = tau_i
        mode_l[idx] = l_i
    pair_offset = np.zeros((nmodes_total, nmodes_total), dtype=np.int32)
    pair_pmin = np.zeros_like(pair_offset)
    pair_pcount = np.zeros_like(pair_offset)
    offset = 0
    for n1 in range(nmodes_total):
        for n2 in range(nmodes_total):
            p_min = max(
                abs(int(mode_m[n1]) - int(mode_m[n2])),
                abs(int(mode_l[n1]) - int(mode_l[n2])) + abs(int(mode_tau[n1]) - int(mode_tau[n2])),
            )
            p_count = int(mode_l[n1]) + int(mode_l[n2]) - p_min + 1
            pair_offset[n1, n2] = offset
            pair_pmin[n1, n2] = p_min
            pair_pcount[n1, n2] = p_count
            offset += p_count
    pair_offset.setflags(write=False)
    pair_pmin.setflags(write=False)
    pair_pcount.setflags(write=False)
    return pair_offset, pair_pmin, pair_pcount


@cache
def _leaf_translation_blocks_rect_raw_kernel(full_order: int, dtype_name: str) -> Any:
    """Return a cached RawKernel for batched interior rectangular translation blocks."""

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
    source = f"""
    #include <cupy/complex.cuh>
    extern "C" __device__ {real_t} assoc_legendre_function(
        const int l,
        const int m,
        const {real_t} ct,
        const {real_t} st,
        const {real_t}* plm_coeffs
    ) {{
        {real_t} plm = ({real_t})0.0;
        const {real_t} st_pow = (m == 0) ? ({real_t})1.0 : pow(st, ({real_t})m);
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * pow(ct, ({real_t})lambda) * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    extern "C" __device__ {complex_t} bessel_lookup_linear(
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
        const int nmodes_full,
        const {real_t}* pair_deltas,
        const int* out_mode_indices,
        const int* in_mode_indices,
        const {real_t}* re_j,
        const {real_t}* im_j,
        const {real_t} inv_dr,
        const int last_index,
        const {real_t}* plm_coeffs,
        const {real_t}* re_ab,
        const {real_t}* im_ab,
        const int* mode_m,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        {complex_t}* out_blocks
    ) {{
        const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int in_idx = blockIdx.y * blockDim.y + threadIdx.y;
        const int pair_idx = blockIdx.z;
        if (pair_idx >= n_pairs) {{
            return;
        }}
        const int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
        const int n_threads = blockDim.x * blockDim.y;

        __shared__ {real_t} r_shared;
        __shared__ {real_t} ct_shared;
        __shared__ {real_t} st_shared;
        __shared__ {real_t} phi_shared;
        __shared__ {real_t} re_j_shared[{n_orders}];
        __shared__ {real_t} im_j_shared[{n_orders}];
        __shared__ {real_t} p_pdm_shared[{n_p_pdm}];
        __shared__ {real_t} cos_mphi_shared[{n_phase}];
        __shared__ {real_t} sin_mphi_shared[{n_phase}];

        if (tid_flat == 0) {{
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
            if (out_idx < n_out_modes && in_idx < n_in_modes) {{
                const int n1_zero = out_mode_indices[out_idx];
                const int n2_zero = in_mode_indices[in_idx];
                const long long flat_idx = ((long long)pair_idx * n_out_modes + out_idx) * n_in_modes + in_idx;
                if (n1_zero == n2_zero) {{
                    out_blocks[flat_idx] = {complex_t}(({real_t})1.0, ({real_t})0.0);
                }} else {{
                    out_blocks[flat_idx] = {complex_t}(({real_t})0.0, ({real_t})0.0);
                }}
            }}
            return;
        }}

        for (int p = tid_flat; p < {n_orders}; p += n_threads) {{
            const {complex_t} radial = bessel_lookup_linear(p, r_shared, re_j, im_j, inv_dr, last_index);
            re_j_shared[p] = radial.real();
            im_j_shared[p] = radial.imag();
            for (int absdm = 0; absdm <= p; ++absdm) {{
                p_pdm_shared[p * (p + 1) / 2 + absdm] =
                    assoc_legendre_function(p, absdm, ct_shared, st_shared, plm_coeffs);
            }}
        }}
        if (tid_flat == 0) {{
            for (int dm = -2 * {order}; dm <= 2 * {order}; ++dm) {{
                const int phase_idx = dm + 2 * {order};
                cos_mphi_shared[phase_idx] = cos(({real_t})dm * phi_shared);
                sin_mphi_shared[phase_idx] = sin(({real_t})dm * phi_shared);
            }}
        }}
        __syncthreads();

        if (out_idx >= n_out_modes || in_idx >= n_in_modes) {{
            return;
        }}

        const int n1 = out_mode_indices[out_idx];
        const int n2 = in_mode_indices[in_idx];
        const int delta_m = mode_m[n2] - mode_m[n1];
        const int phase_idx = delta_m + 2 * {order};
        const int table_idx = n1 * nmodes_full + n2;
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

        const long long flat_idx = ((long long)pair_idx * n_out_modes + out_idx) * n_in_modes + in_idx;
        out_blocks[flat_idx] = {complex_t}(re_acc, im_acc);
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
    nmodes_full = int(n_modes(full_order))
    leaves = tuple(sorted(partition.leaves, key=lambda leaf: int(leaf.id)))
    n_leaves = int(len(leaves))
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

    compact_re_ab_raw, compact_im_ab_raw = _translation_ab5_compact_tables(
        full_order, dtype=out_dtype
    )
    plm_coeffs_raw = _translation_plm_coeff_table(full_order, dtype=real_dtype).reshape(-1)
    mode_m = _mode_metadata_tables(full_order)
    pair_offset_raw, pair_pmin_raw, pair_pcount_raw = _mode_pair_tables(full_order)

    lut_j = np.asarray(radial_lut_full.j, dtype=out_dtype)[: 2 * full_order + 1, :]
    lut_j_rows = np.ascontiguousarray(lut_j.T)
    re_j_raw = np.ascontiguousarray(lut_j_rows.real.reshape(-1), dtype=real_dtype)
    im_j_raw = np.ascontiguousarray(lut_j_rows.imag.reshape(-1), dtype=real_dtype)
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
    mode_m_dev = cupy.asarray(np.ascontiguousarray(mode_m), dtype=cupy.int32)
    pair_offset_dev = cupy.asarray(
        np.ascontiguousarray(pair_offset_raw.reshape(-1), dtype=np.int32), dtype=cupy.int32
    )
    pair_pmin_dev = cupy.asarray(
        np.ascontiguousarray(pair_pmin_raw.reshape(-1), dtype=np.int32), dtype=cupy.int32
    )
    pair_pcount_dev = cupy.asarray(
        np.ascontiguousarray(pair_pcount_raw.reshape(-1), dtype=np.int32), dtype=cupy.int32
    )
    plm_coeffs_dev = cupy.asarray(np.ascontiguousarray(plm_coeffs_raw), dtype=cupy_real_dtype)
    re_ab_dev = cupy.asarray(np.ascontiguousarray(compact_re_ab_raw), dtype=cupy_real_dtype)
    im_ab_dev = cupy.asarray(np.ascontiguousarray(compact_im_ab_raw), dtype=cupy_real_dtype)
    re_j_dev = cupy.asarray(re_j_raw, dtype=cupy_real_dtype)
    im_j_dev = cupy.asarray(im_j_raw, dtype=cupy_real_dtype)

    kernel = _leaf_translation_blocks_rect_raw_kernel(full_order, out_dtype.str)
    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_z = int(props["maxGridSize"][2])
    max_threads = int(props["maxThreadsPerBlock"])
    if max_threads >= 128:
        threads_x, threads_y = 16, 8
    elif max_threads >= 64:
        threads_x, threads_y = 8, 8
    else:
        threads_x, threads_y = 8, 4
    grid_x = max(1, (nmodes_out + threads_x - 1) // threads_x)
    grid_y = max(1, (nmodes_in + threads_y - 1) // threads_y)

    bytes_per_pair = max(1, nmodes_out * nmodes_in * out_dtype.itemsize)
    target_chunk_bytes = 256 * 1024 * 1024
    chunk_pairs = max(1, target_chunk_bytes // bytes_per_pair)
    chunk_pairs = max(1, min(int(chunk_pairs), int(max_grid_z)))

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
            (int(grid_x), int(grid_y), count),
            (threads_x, threads_y, 1),
            (
                np.int32(count),
                np.int32(nmodes_out),
                np.int32(nmodes_in),
                np.int32(nmodes_full),
                deltas_dev.reshape(-1),
                out_mode_idx_dev,
                in_mode_idx_dev,
                re_j_dev,
                im_j_dev,
                inv_dr_scalar,
                np.int32(last_index),
                plm_coeffs_dev,
                re_ab_dev,
                im_ab_dev,
                mode_m_dev,
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


def _upload_directional_transforms(
    transforms: MLFMMDirectionalTransforms | CuPyHostDirectionalTransformsData, *, cupy: Any
) -> CuPyDirectionalTransformsData:
    grid = transforms.grid
    n_dir: int
    if hasattr(grid, "n_directions"):
        n_dir = int(grid.n_directions)
    else:
        directions = _as_numpy_3cols(
            grid.directions, dtype=np.float64, name="directional.grid.directions"
        )
        weights = _as_numpy_1d(grid.weights, dtype=np.float64, name="directional.grid.weights")
        if directions.shape[0] != weights.size:
            raise ValueError(
                "Directional grid directions/weights mismatch: "
                f"{directions.shape[0]} vs {weights.size}."
            )
        n_dir = int(directions.shape[0])
    reflection = _as_numpy_1d(
        grid.reflection_permutation,
        dtype=np.int32,
        name="directional.grid.reflection_permutation",
    )
    if reflection.size != n_dir:
        raise ValueError(
            f"Directional grid reflection-permutation size mismatch: {reflection.size} vs {n_dir}."
        )

    # Directional transforms are complex128 reference operators; keep them exact.
    fth = _as_numpy_2d(transforms.Fth, dtype=np.complex128, name="directional.Fth")
    fph = _as_numpy_2d(transforms.Fph, dtype=np.complex128, name="directional.Fph")
    gth = _as_numpy_2d(transforms.Gth, dtype=np.complex128, name="directional.Gth")
    gph = _as_numpy_2d(transforms.Gph, dtype=np.complex128, name="directional.Gph")
    if (
        fth.shape[0] != n_dir
        or fph.shape[0] != n_dir
        or gth.shape[0] != n_dir
        or gph.shape[0] != n_dir
    ):
        raise ValueError(
            "Directional forward transform row count must match sampled direction count."
        )
    nscl = int(fth.shape[1])
    perm = np.asarray(reflection, dtype=np.int32)
    inv_perm = np.ascontiguousarray(np.argsort(perm), dtype=np.int32)
    perm_gpu = cupy.asarray(perm, dtype=cupy.int32)
    inv_perm_gpu = cupy.asarray(inv_perm, dtype=cupy.int32)

    # Build packed directional operators on GPU so the prepare path avoids
    # large host-side stack/hstack temporaries before upload.
    fth_gpu = cupy.asarray(fth, dtype=cupy.complex128)
    fph_gpu = cupy.asarray(fph, dtype=cupy.complex128)
    gth_gpu = cupy.asarray(gth, dtype=cupy.complex128)
    gph_gpu = cupy.asarray(gph, dtype=cupy.complex128)

    # Outgoing map: pre-fold reflection row permutation and stack theta/phi
    # into one (2*ndir, nscl) matrix for batched GEMM.
    f_stack = cupy.concatenate((fth_gpu[perm_gpu, :], fph_gpu[perm_gpu, :]), axis=0)

    # Incoming map: pre-fold reflection on columns via A @ P equivalence
    # (implemented as column reindex by inverse permutation), then stack
    # [theta,phi] blocks for compact batched GEMM.
    a_adj = cupy.concatenate(
        (
            cupy.conjugate(fth_gpu.T)[:, inv_perm_gpu],
            cupy.conjugate(fph_gpu.T)[:, inv_perm_gpu],
        ),
        axis=1,
    )
    g_adj = cupy.concatenate(
        (
            cupy.conjugate(gth_gpu.T)[:, inv_perm_gpu],
            cupy.conjugate(gph_gpu.T)[:, inv_perm_gpu],
        ),
        axis=1,
    )

    return CuPyDirectionalTransformsData(
        box_order=int(transforms.box_order),
        grid_order=int(transforms.grid.order),
        grid=CuPyDirectionalGridData(
            order=int(grid.order),
            n_directions=int(n_dir),
            reflection_permutation=perm_gpu,
        ),
        nscl=nscl,
        forward_F=f_stack,
        inverse_A_adj=a_adj,
        inverse_G_adj=g_adj,
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


def _upload_offset_batches(
    batches: dict[Offset3, tuple[np.ndarray, np.ndarray]], *, cupy: Any, name: str
) -> dict[Offset3, CuPyOffsetBatchData]:
    """Upload grouped source/target batches and enforce uniqueness contract."""

    out: dict[Offset3, CuPyOffsetBatchData] = {}
    for offset, (src_idx, dst_idx) in batches.items():
        src = _as_numpy_1d(src_idx, dtype=np.int32, name=f"{name}[{offset}].src")
        dst = _as_numpy_1d(dst_idx, dtype=np.int32, name=f"{name}[{offset}].dst")
        if src.size != dst.size:
            raise ValueError(
                f"{name}[{offset}] source/target batch size mismatch: {src.size} vs {dst.size}."
            )
        src_unique = bool(np.unique(src).size == src.size)
        dst_unique = bool(np.unique(dst).size == dst.size)
        if not src_unique or not dst_unique:
            raise ValueError(
                f"{name}[{offset}] violates grouped uniqueness contract "
                f"(src_unique={src_unique}, dst_unique={dst_unique})."
            )
        out[offset] = CuPyOffsetBatchData(
            src_indices=cupy.asarray(src, dtype=cupy.int32),
            dst_indices=cupy.asarray(dst, dtype=cupy.int32),
        )
    return out


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
    transforms: MLFMMDirectionalTransforms,
) -> CuPyHostDirectionalTransformsData:
    """Extract canonical directional operators into a compact host payload."""

    n_dir = int(np.asarray(transforms.grid.reflection_permutation).size)
    return CuPyHostDirectionalTransformsData(
        box_order=int(transforms.box_order),
        grid_order=int(transforms.grid.order),
        grid=CuPyHostDirectionalGridData(
            order=int(transforms.grid.order),
            n_directions=n_dir,
            reflection_permutation=np.ascontiguousarray(
                np.asarray(transforms.grid.reflection_permutation, dtype=np.int32).reshape(-1)
            ),
        ),
        Fth=np.ascontiguousarray(np.asarray(transforms.Fth, dtype=np.complex128)),
        Fph=np.ascontiguousarray(np.asarray(transforms.Fph, dtype=np.complex128)),
        Gth=np.ascontiguousarray(np.asarray(transforms.Gth, dtype=np.complex128)),
        Gph=np.ascontiguousarray(np.asarray(transforms.Gph, dtype=np.complex128)),
    )


def _build_host_single_level(single: MLFMMSingleLevelOperators) -> CuPyHostSingleLevelData:
    """Build compact host cache payload for one single-level sampled-far stage."""

    return CuPyHostSingleLevelData(
        box_order=int(single.box_order),
        translator_order=int(single.translator_order),
        grid_order=int(single.grid_order),
        directional=_copy_directional_host(single.directional),
        aggregation=tuple(
            np.ascontiguousarray(np.asarray(block, dtype=np.complex128))
            for block in single.aggregation
        ),
        far_offset_batches=_copy_batches_host(single.far_offset_batches),
        offset_diagonals={
            key: np.ascontiguousarray(np.asarray(values, dtype=np.complex128).reshape(-1))
            for key, values in single.offset_diagonals.items()
        },
    )


def _build_host_multilevel(multilevel: MLFMMMultilevelOperators) -> CuPyHostMultilevelData:
    """Build compact host cache payload for one multilevel sampled-far stage."""

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
    return CuPyHostMultilevelData(
        levels=levels,
        transfers=transfers,
        leaf_level=int(multilevel.leaf_level),
        hf_start_level=int(multilevel.hf_start_level),
        hf_end_level=int(multilevel.hf_end_level),
        aggregation=tuple(
            np.ascontiguousarray(np.asarray(block, dtype=np.complex128))
            for block in multilevel.aggregation
        ),
    )


def _resolve_host_cache_policy(
    policy: CuPyMLFMMHostCachePolicy | None,
) -> CuPyMLFMMHostCachePolicy:
    return CuPyMLFMMHostCachePolicy() if policy is None else policy


def _build_mlfmm_cupy_host_cache(
    coupling: MLFMMCouplingOperator,
    *,
    host_cache_policy: CuPyMLFMMHostCachePolicy | None = None,
) -> CuPyMLFMMHostCacheData:
    """Build a compact host-only MLFMM cache artifact from CPU reference operators."""

    policy = _resolve_host_cache_policy(host_cache_policy)
    near_dtype = np.dtype(coupling.near_dtype)
    real_dtype: type[np.floating[Any]]
    lut_dtype: type[np.complexfloating[Any, Any]]
    if near_dtype == np.dtype(np.complex64):
        real_dtype = np.float32
        lut_dtype = np.complex64
    else:
        real_dtype = np.float64
        lut_dtype = np.complex128
    partition = coupling.resolved_plan.partition
    leaf_offsets, leaf_indices = _pack_index_lists(
        [leaf.particle_indices for leaf in partition.leaves]
    )
    dst_leaf_indices, src_leaf_indices = _build_exact_near_leaf_pair_schedule(partition)
    lut = np.asarray(coupling.radial_lut.h, dtype=np.complex128).T.astype(lut_dtype, copy=False)
    keep_static_near_tables = policy.memory_budget == "balanced"
    if keep_static_near_tables:
        compact_re_ab, compact_im_ab = _translation_ab5_compact_tables(
            int(coupling.lmax), dtype=np.complex128
        )
        plm_coeffs = _translation_plm_coeff_table(int(coupling.lmax), dtype=np.float64).reshape(-1)
        mode_m = _mode_metadata_tables(int(coupling.lmax))
        pair_offset, pair_pmin, pair_pcount = _mode_pair_tables(int(coupling.lmax))
    else:
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
        near_lut_re=np.ascontiguousarray(lut.real.reshape(-1), dtype=real_dtype),
        near_lut_im=np.ascontiguousarray(lut.imag.reshape(-1), dtype=real_dtype),
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
        host_memory_budget=policy.memory_budget,
        single_level=(
            _build_host_single_level(coupling.single_level)
            if coupling.single_level is not None
            else None
        ),
        multilevel=(
            _build_host_multilevel(coupling.multilevel) if coupling.multilevel is not None else None
        ),
        plan_summary=plan_summary,
    )


def _upload_leaf_apply_groups(
    *,
    aggregation: tuple[np.ndarray, ...],
    partition: CuPyMLFMMPartitionData,
    cupy: Any,
    name: str,
) -> tuple[int, tuple[CuPyLeafApplyGroupData, ...]]:
    """Upload grouped leaf operators with uniform occupancy for batched GEMM."""

    n_leaves = int(len(aggregation))
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
            )
        )
    return box_nm, tuple(grouped)


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

    box_nm, leaf_groups = _upload_leaf_apply_groups(
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
        n_leaves=int(len(single.aggregation)),
        directional=directional,
        leaf_groups=leaf_groups,
        far_offset_batches=_upload_offset_batches(
            single.far_offset_batches,
            cupy=cupy,
            name="single_level.far_offset_batches",
        ),
        offset_diagonals=offset_diagonals,
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
    box_nm, leaf_groups = _upload_leaf_apply_groups(
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
        n_leaves=int(len(multilevel.aggregation)),
        leaf_groups=leaf_groups,
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

    # Under `low_host_memory`, static tables are intentionally omitted from the
    # host cache and rebuilt on demand during upload.
    lmax = int(cache.lmax)
    compact_re_ab_raw, compact_im_ab_raw = _translation_ab5_compact_tables(
        lmax, dtype=np.complex128
    )
    plm_coeffs_raw = _translation_plm_coeff_table(lmax, dtype=np.float64).reshape(-1)
    mode_m = _mode_metadata_tables(lmax)
    pair_offset_raw, pair_pmin_raw, pair_pcount_raw = _mode_pair_tables(lmax)
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
def _exact_near_pairs_raw_kernel(lmax: int, near_dtype_name: str) -> Any:
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
    n_orders = 2 * lmax + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    #include <cupy/complex.cuh>
    extern "C" __device__ {real_t} assoc_legendre_function(
        const int l,
        const int m,
        const {real_t} ct,
        const {real_t} st,
        const {real_t}* plm_coeffs
    ) {{
        {real_t} plm = ({real_t})0.0;
        const {real_t} st_pow = (m == 0) ? ({real_t})1.0 : pow(st, ({real_t})m);
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * pow(ct, ({real_t})lambda) * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    extern "C" __device__ {real_t} hankel_lookup_linear(
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
        return (({real_t})1.0 - frac) * table[base0] + frac * table[base1];
    }}

    extern "C" __global__ void mlfmm_exact_near_pairs(
        const int n_leaf_pairs,
        const int nmodes,
        const int nrhs,
        const {real_t}* positions,
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
                        if (dst_leaf_shared == src_leaf_shared && dst_particle == src_particle) {{
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
                            r_shared = sqrt(x21 * x21 + y21 * y21 + z21 * z21);
                            ct_shared = z21 / r_shared;
                            st_shared = sqrt(fmax(({real_t})0.0, ({real_t})1.0 - ct_shared * ct_shared));
                            phi_shared = atan2(y21, x21);
                        }}
                        __syncthreads();

                        for (int p = threadIdx.x; p < {n_orders}; p += blockDim.x) {{
                            re_h_shared[p] = hankel_lookup_linear(p, r_shared, re_h, inv_dr, last_index);
                            im_h_shared[p] = hankel_lookup_linear(p, r_shared, im_h, inv_dr, last_index);
                            for (int absdm = 0; absdm <= p; ++absdm) {{
                                p_pdm_shared[p * (p + 1) / 2 + absdm] =
                                    assoc_legendre_function(p, absdm, ct_shared, st_shared, plm_coeffs);
                            }}
                        }}
                        if (threadIdx.x == 0) {{
                            for (int dm = -2 * {lmax}; dm <= 2 * {lmax}; ++dm) {{
                                const int idx = dm + 2 * {lmax};
                                cos_mphi_shared[idx] = cos(({real_t})dm * phi_shared);
                                sin_mphi_shared[idx] = sin(({real_t})dm * phi_shared);
                            }}
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
    return cupy.RawKernel(source, "mlfmm_exact_near_pairs")


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
    a_box = states[:, :nscl, :]
    b_box = states[:, nscl:, :]
    pair_dir = cupy.matmul(
        directional.forward_F[None, :, :],
        cupy.concatenate((a_box, b_box), axis=0),
    )
    a_dir = pair_dir[:n_batch]
    b_dir = pair_dir[n_batch:]
    out_arr = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.empty((n_batch, 4, ndir, int(states.shape[2])), dtype=cupy.complex128)
    )
    out_arr[:, 0, :, :] = a_dir[:, :ndir, :]
    out_arr[:, 1, :, :] = a_dir[:, ndir:, :]
    out_arr[:, 2, :, :] = b_dir[:, :ndir, :]
    out_arr[:, 3, :, :] = b_dir[:, ndir:, :]
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
    ch_ab = cupy.concatenate((channels[:, 0], channels[:, 1]), axis=1)
    ch_bb = cupy.concatenate((channels[:, 2], channels[:, 3]), axis=1)
    ch_pair = cupy.concatenate((ch_ab, ch_bb), axis=0)
    a_pair = cupy.matmul(directional.inverse_A_adj[None, :, :], ch_pair)
    g_pair = cupy.matmul(directional.inverse_G_adj[None, :, :], ch_pair)
    a_ab = a_pair[:n_batch]
    a_bb = a_pair[n_batch:]
    g_ab = g_pair[:n_batch]
    g_bb = g_pair[n_batch:]
    out_arr = (
        cupy.asarray(out, dtype=cupy.complex128)
        if out is not None
        else cupy.empty(
            (n_batch, int(directional.nscl) * 2, int(channels.shape[3])),
            dtype=cupy.complex128,
        )
    )
    nscl = int(directional.nscl)
    out_arr[:, :nscl, :] = a_ab + g_bb
    out_arr[:, nscl:, :] = g_ab + a_bb
    return out_arr


def _aggregate_leaf_box_states(
    x_states: Any,
    *,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
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
        coeffs = x_states[idx].reshape(n_group, occupancy * nmodes, int(nrhs))
        box_states[group.leaf_ids] = cupy.matmul(group.aggregation, coeffs)
    return box_states


def _receive_leaf_boxes_to_particles(
    incoming_box: Any,
    *,
    leaf_groups: tuple[CuPyLeafApplyGroupData, ...],
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
        receive_adj: Any
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
    level_shapes = tuple(
        (int(level.n_boxes), int(level.directional.grid.n_directions))
        for level in multilevel.levels
    )
    key = CuPyMLFMMMultilevelWorkspaceKey(
        nrhs=int(nrhs),
        n_particles=int(n_particles),
        nm=int(nm),
        n_leaves=int(multilevel.n_leaves),
        box_nm=int(multilevel.box_nm),
        level_shapes=level_shapes,
    )
    ws = cache.get(key)
    if ws is not None:
        return ws
    levels = multilevel.levels
    outgoing = [
        cupy.empty(
            (int(level.n_boxes), 4, int(level.directional.grid.n_directions), int(key.nrhs)),
            dtype=cupy.complex128,
        )
        for level in levels
    ]
    incoming = [cupy.empty_like(values, dtype=cupy.complex128) for values in outgoing]
    ws = CuPyMLFMMMultilevelWorkspace(
        nrhs=int(key.nrhs),
        outgoing=outgoing,
        incoming=incoming,
        leaf_box_states=cupy.empty(
            (int(key.n_leaves), int(key.box_nm), int(key.nrhs)), dtype=cupy.complex128
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
        x_states=cupy.empty(
            (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy_near_dtype
        ),
        y_states=cupy.empty(
            (int(key.n_particles), int(key.nm), int(key.nrhs)), dtype=cupy_near_dtype
        ),
    )
    cache[key] = ws
    return ws


def _apply_exact_near_pairs(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    workspace: CuPyMLFMMNearWorkspace | None,
    cupy: Any,
) -> Any:
    """Apply exact near interactions from directed near-pair indices on device."""

    near = prepared.near_pairs
    near_dtype = np.dtype(near.near_dtype)
    np_real_type: type[np.floating[Any]]
    if near_dtype == np.dtype(np.complex64):
        cupy_out_dtype = cupy.complex64
        np_real_type = np.float32
    elif near_dtype == np.dtype(np.complex128):
        cupy_out_dtype = cupy.complex128
        np_real_type = np.float64
    else:
        raise ValueError(
            "CuPy MLFMM near path supports only complex64/complex128 near dtypes. "
            f"Got {near_dtype!r}."
        )
    cupy_compute_dtype = cupy_out_dtype
    kernel = _exact_near_pairs_raw_kernel(int(prepared.lmax), near_dtype.str)

    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    n_leaf_pairs = int(near.dst_leaf_indices.size)
    if n_leaf_pairs == 0:
        if workspace is not None:
            workspace.y_states.fill(0)
            return workspace.y_states
        return cupy.zeros((n_particles, nm, nrhs), dtype=cupy_out_dtype)
    if workspace is None:
        x_arr = cupy.ascontiguousarray(cupy.asarray(x_states, dtype=cupy_compute_dtype))
        y_arr = cupy.zeros((n_particles, nm, nrhs), dtype=cupy_compute_dtype)
    else:
        x_arr = workspace.x_states
        y_arr = workspace.y_states
        x_arr[...] = cupy.asarray(x_states, dtype=cupy_compute_dtype)
        y_arr.fill(0)

    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_grid_y = int(props["maxGridSize"][1])
    max_grid_z = int(props["maxGridSize"][2])
    warp_size = int(props["warpSize"])
    threads = max(warp_size, min(int(props["maxThreadsPerBlock"]), nm))
    blocks_x = max(1, (nm + threads - 1) // threads)
    grid_y = min(n_leaf_pairs, max_grid_y)
    grid_z = min(max(1, nrhs), max_grid_z)

    kernel(
        (int(blocks_x), int(grid_y), int(grid_z)),
        (int(threads),),
        (
            np.int32(n_leaf_pairs),
            np.int32(nm),
            np.int32(nrhs),
            near.positions,
            near.dst_leaf_indices,
            near.src_leaf_indices,
            near.leaf_particle_offsets,
            near.leaf_particle_indices,
            near.lut_re,
            near.lut_im,
            np_real_type(float(near.inv_dr)),
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
        ),
    )
    return y_arr.astype(cupy_out_dtype, copy=False)


def _apply_single_level_far(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
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
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        out=(ws.y_states if ws is not None else None),
        cupy=cupy,
    )


def _apply_multilevel_far(
    prepared: CuPyMLFMMPreparedData,
    x_states: Any,
    *,
    receive_adjoint_cache: dict[int, Any] | None,
    workspace: CuPyMLFMMMultilevelWorkspace | None,
    cupy: Any,
) -> Any:
    """Apply sampled multilevel far interactions on device.

    Grouped source/target schedules are validated as unique during upload.
    Transfer loops run unique-index kernels and keep a high-level fallback only
    for map-storage variants that do not have a fused kernel yet.
    """

    multilevel = prepared.multilevel
    if multilevel is None:
        raise RuntimeError("Internal CuPy MLFMM error: missing multilevel prepared data.")
    n_particles, nm, nrhs = (int(v) for v in x_states.shape)
    levels = multilevel.levels
    ws = workspace
    if ws is None:
        outgoing = [
            cupy.zeros(
                (
                    int(level.n_boxes),
                    4,
                    int(level.directional.grid.n_directions),
                    nrhs,
                ),
                dtype=cupy.complex128,
            )
            for level in levels
        ]
        incoming = [cupy.zeros_like(values, dtype=cupy.complex128) for values in outgoing]
    else:
        outgoing = ws.outgoing
        incoming = ws.incoming
        for arr in outgoing:
            arr.fill(0)
        for arr in incoming:
            arr.fill(0)
    leaf_level = int(multilevel.leaf_level)
    box_nm = int(multilevel.box_nm)
    leaf_box_states = _aggregate_leaf_box_states(
        x_states,
        leaf_groups=multilevel.leaf_groups,
        n_leaves=int(multilevel.n_leaves),
        box_nm=box_nm,
        nrhs=nrhs,
        out=(ws.leaf_box_states if ws is not None else None),
        cupy=cupy,
    )
    _box_outgoing_to_directional_cupy(
        levels[leaf_level].directional,
        leaf_box_states,
        out=outgoing[leaf_level],
        cupy=cupy,
    )

    for transfer in reversed(multilevel.transfers):
        child_level = int(transfer.child_level)
        parent_level = int(transfer.parent_level)
        child_values = outgoing[child_level]
        parent_values = outgoing[parent_level]
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

    for level_idx in range(int(multilevel.hf_start_level), int(multilevel.hf_end_level) + 1):
        level = levels[level_idx]
        for offset, batch in level.far_offset_batches.items():
            _weighted_gather_add_complex128(
                incoming[level_idx],
                batch.dst_indices,
                outgoing[level_idx],
                batch.src_indices,
                level.offset_diagonals[offset],
                cupy=cupy,
            )

    for transfer in multilevel.transfers:
        child_level = int(transfer.child_level)
        parent_level = int(transfer.parent_level)
        parent_values = incoming[parent_level]
        child_values = incoming[child_level]
        for shift, batch in transfer.batches_by_shift.items():
            if str(transfer.map_down.storage) == "packed_stencil":
                _transfer_down_packed_unique_complex128(
                    child_values,
                    batch.src_indices,
                    parent_values,
                    batch.dst_indices,
                    transfer.map_down,
                    transfer.phase_down_by_shift[shift],
                    cupy=cupy,
                )
            elif str(transfer.map_down.storage) == "sparse":
                _transfer_down_sparse_unique_complex128(
                    child_values,
                    batch.src_indices,
                    parent_values,
                    batch.dst_indices,
                    transfer.map_down,
                    transfer.phase_down_by_shift[shift],
                    cupy=cupy,
                )
            else:
                # Internal fallback for unsupported map storage variants.
                shifted = (
                    parent_values[batch.dst_indices]
                    * transfer.phase_down_by_shift[shift][None, None, :, None]
                )
                mapped = _apply_directional_map(shifted, transfer.map_down, cupy=cupy)
                _add_at_complex128(
                    child_values,
                    batch.src_indices,
                    mapped,
                    cupy=cupy,
                )

    incoming_box = _directional_to_box_regular_cupy(
        levels[leaf_level].directional,
        incoming[leaf_level],
        out=(ws.incoming_box if ws is not None else None),
        cupy=cupy,
    )
    return _receive_leaf_boxes_to_particles(
        incoming_box,
        leaf_groups=multilevel.leaf_groups,
        receive_adjoint_cache=receive_adjoint_cache,
        nm=nm,
        n_particles=n_particles,
        nrhs=nrhs,
        out=(ws.y_states if ws is not None else None),
        cupy=cupy,
    )


@dataclass
class CuPyMLFMMCouplingOperator:
    """CuPy-backed repeated-apply MLFMM coupling operator.

    The MLFMM plan and one-time operators are built on CPU (NumPy reference
    path). This class executes repeated exact-near and sampled-far applies on
    CuPy device arrays.

    Precision policy mirrors the NumPy MLFMM reference path:
    - exact-near runs at `near_dtype` (`complex64` or `complex128`);
    - sampled-far runs at `far_dtype` (currently fixed to `complex128`);
    - public output is cast to `dtype`.
    """

    lmax: int
    n_particles: int
    prepared_data: CuPyMLFMMPreparedData
    host_cache: CuPyMLFMMHostCacheData
    host_cache_policy: CuPyMLFMMHostCachePolicy = field(default_factory=CuPyMLFMMHostCachePolicy)
    dtype: np.dtype = np.dtype(np.complex128)
    near_dtype: np.dtype = np.dtype(np.complex128)
    far_dtype: np.dtype = np.dtype(np.complex128)
    _receive_adjoint_cache: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _near_workspace_cache: dict[CuPyMLFMMNearWorkspaceKey, CuPyMLFMMNearWorkspace] = field(
        default_factory=dict, init=False, repr=False
    )
    _single_level_workspace_cache: dict[
        CuPyMLFMMSingleLevelWorkspaceKey, CuPyMLFMMSingleLevelWorkspace
    ] = field(default_factory=dict, init=False, repr=False)
    _multilevel_workspace_cache: dict[
        CuPyMLFMMMultilevelWorkspaceKey, CuPyMLFMMMultilevelWorkspace
    ] = field(default_factory=dict, init=False, repr=False)

    def apply(self, x: Any) -> Any:
        cupy, _ = import_cupy()
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
        near_ws = _ensure_exact_near_workspace(
            self.prepared_data,
            n_particles=n_particles,
            nm=nm,
            nrhs=int(x_states.shape[2]),
            near_dtype=near_dtype,
            cache=self._near_workspace_cache,
            cupy=cupy,
        )
        y_near = _apply_exact_near_pairs(
            self.prepared_data,
            x_states,
            workspace=near_ws,
            cupy=cupy,
        )
        stage = str(self.prepared_data.stage)
        if stage == "single_level":
            single_ws = _ensure_single_level_workspace(
                self.prepared_data,
                n_particles=n_particles,
                nm=nm,
                nrhs=int(x_states.shape[2]),
                cache=self._single_level_workspace_cache,
                cupy=cupy,
            )
            y_far = _apply_single_level_far(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                workspace=single_ws,
                cupy=cupy,
            )
        elif stage == "multilevel":
            multi_ws = _ensure_multilevel_workspace(
                self.prepared_data,
                n_particles=n_particles,
                nm=nm,
                nrhs=int(x_states.shape[2]),
                cache=self._multilevel_workspace_cache,
                cupy=cupy,
            )
            y_far = _apply_multilevel_far(
                self.prepared_data,
                x_states,
                receive_adjoint_cache=self._receive_adjoint_cache,
                workspace=multi_ws,
                cupy=cupy,
            )
        else:
            raise RuntimeError(f"Unsupported CuPy MLFMM stage {stage!r}.")
        y_total = cupy.asarray(y_near, dtype=cupy.complex128) + cupy.asarray(
            y_far, dtype=cupy.complex128
        )
        return _restore_unknown_shape(
            y_total.astype(_cupy_complex_dtype(out_dtype, cupy=cupy), copy=False),
            squeezed=squeezed,
        )

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError(
            "CuPyMLFMMCouplingOperator is runtime-only and intentionally non-picklable. "
            "Persist the compact host cache explicitly if debug serialization is required."
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
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
    return CuPyMLFMMCouplingOperator(
        lmax=int(coupling.lmax),
        n_particles=int(np.asarray(coupling.positions).shape[0]),
        prepared_data=prepared,
        host_cache=host_cache,
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
    "CuPyMLFMMHostCacheData",
    "CuPyMLFMMHostCachePolicy",
    "CuPyMLFMMCouplingOperator",
    "CuPyDirectionalGridData",
    "CuPyDirectionalInterpolationData",
    "CuPyDirectionalTransformsData",
    "CuPyLeafApplyGroupData",
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
