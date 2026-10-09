"""Prepared CuPy sampled-direction operators for MLFMM.

``mlfmm_directional`` owns the shared grids, conventions, factor construction,
and NumPy reference. This module owns compact host directional payloads, their
device representation, map-storage selection, and repeated boundary/map
application. It has no dependency on tree topology, leaf operators, traversal,
or the runtime workspace/budget policy in ``mlfmm_cupy``.

The module is safe to import without an initialized CuPy runtime: CuPy is
loaded lazily, and device arrays and kernels are created only when a payload is
uploaded or applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import import_cupy

from .mlfmm_directional import (
    _SAMPLED_DIRECTIONAL_CHANNELS,
    MLFMMDirectionalStructuredTransforms,
    MLFMMDirectionalTransformData,
    _circular_beta_factors,
    _directional_beta_factors,
)


@dataclass(frozen=True)
class CuPyDirectionalGridData:
    """Dimensions of a prepared device directional grid.

    Reflection is absorbed into factors and transfer maps during upload;
    repeated actions do not need device permutation arrays.
    """

    order: int
    n_alpha: int
    n_beta: int
    n_directions: int


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
class CuPyDirectionalModeData:
    """Contiguous factor views for one azimuthal group, prepared once.

    The views partition shared packed allocations. No second factor cache or
    per-action factor gather is needed; box states retain canonical ordering.
    """

    indices: Any
    hplus_beta: Any
    hminus_beta: Any


@dataclass(frozen=True)
class CuPyDirectionalTransformsData:
    """Device directional transform payload for one box/grid order.

    The CuPy path stores beta factors and azimuthal phases instead of dense
    `(n_alpha*n_beta, n_scalar)` F matrices. Repeated apply expands the
    directional channels on demand, which trades explicit `m` sums for a much
    lower resident memory footprint on coarse high-frequency levels. Physical
    beta reflection is absorbed once into the uploaded factors; adjoint maps
    use their conjugate transposes, not reflected copies of channel batches.
    The stored H+/- factors are (Fth +/- i*Fph)/sqrt(2), so each circular
    sampled channel needs only one beta contraction per azimuthal group.
    """

    box_order: int
    grid_order: int
    grid: CuPyDirectionalGridData
    nscl: int
    phase_by_m: Any
    mode_groups: tuple[CuPyDirectionalModeData, ...]


@dataclass(frozen=True)
class CuPyDirectionalInterpolationData:
    """Device copy of one directional transfer map.

    The map is stored either as a CuPy dense matrix (`storage="dense"`),
    a CuPy CSR sparse matrix (`storage="sparse"`), or a low-width packed
    row stencil (`storage="packed_stencil"`).
    """

    source_order: int
    target_order: int
    matrix: Any
    nnz: int
    storage: str


def _directional_structured_host_view(
    transforms: MLFMMDirectionalTransformData | CuPyHostDirectionalTransformsData,
) -> tuple[CuPyHostDirectionalGridData, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact structured factors for one host directional transform."""

    if isinstance(transforms, CuPyHostDirectionalTransformsData):
        grid = transforms.grid
        fth_beta = np.ascontiguousarray(transforms.fth_beta, dtype=np.complex128)
        fph_beta = np.ascontiguousarray(transforms.fph_beta, dtype=np.complex128)
        if fth_beta.ndim != 2 or fph_beta.ndim != 2:
            raise ValueError("Directional theta/phi beta factors must be 2D arrays.")
        m_of_scalar = np.ascontiguousarray(
            np.asarray(transforms.m_of_scalar, dtype=np.int32).reshape(-1)
        )
        return grid, fth_beta, fph_beta, m_of_scalar

    if isinstance(transforms, MLFMMDirectionalStructuredTransforms):
        structured = transforms
    else:
        # Preserve the actual quadrature, including non-default alpha/beta
        # sampling factors, rather than rebuilding the default grid by order.
        fth_beta, fph_beta, m_of_scalar = _directional_beta_factors(
            int(transforms.box_order), transforms.grid
        )
        structured = MLFMMDirectionalStructuredTransforms(
            box_order=int(transforms.box_order),
            grid=transforms.grid,
            fth_beta=fth_beta,
            fph_beta=fph_beta,
            m_of_scalar=m_of_scalar,
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
    beta_indices = np.arange(int(n_beta), dtype=np.int32)
    if not np.array_equal(np.sort(beta_perm), beta_indices):
        raise ValueError("Directional beta reflection must be a permutation.")
    if not np.array_equal(beta_perm[beta_perm], beta_indices):
        raise ValueError("Directional beta reflection must be an involution.")
    expected = np.arange(int(n_alpha), dtype=np.int32)[:, None] * int(n_beta) + beta_perm[None, :]
    if not np.array_equal(reflection_grid, expected):
        raise ValueError("Directional reflection must preserve alpha and permute beta only.")
    return beta_perm


def _upload_directional_transforms(
    transforms: MLFMMDirectionalTransformData | CuPyHostDirectionalTransformsData, *, cupy: Any
) -> CuPyDirectionalTransformsData:
    grid, fth_beta, fph_beta, m_of_scalar = _directional_structured_host_view(transforms)
    n_alpha, n_beta, n_dir = int(grid.n_alpha), int(grid.n_beta), int(grid.n_directions)
    if n_alpha <= 0 or n_beta <= 0 or n_dir != n_alpha * n_beta:
        raise ValueError(
            "Directional grid dimensions are inconsistent: "
            f"n_alpha={n_alpha}, n_beta={n_beta}, n_directions={n_dir}."
        )
    reflection = np.ascontiguousarray(
        np.asarray(grid.reflection_permutation, dtype=np.int32).reshape(-1)
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
    if np.any(np.abs(m_of_scalar.astype(np.int64)) > box_order):
        raise ValueError("Directional m values must lie within the box order.")
    # Equality masks below partition every scalar mode exactly once. Receive
    # writes, rather than accumulates, each corresponding output row.
    m_values = np.arange(-box_order, box_order + 1, dtype=np.int32)
    alpha = np.ascontiguousarray(np.asarray(grid.alpha, dtype=np.float64).reshape(-1))
    if alpha.size != n_alpha:
        raise ValueError(f"Directional alpha size mismatch: {alpha.size} vs {n_alpha}.")
    phase_by_m = np.asarray(
        np.exp(1j * alpha[:, None] * m_values[None, :]),
        dtype=np.complex128,
    )
    indices_by_m = tuple(
        np.flatnonzero(m_of_scalar == int(m)).astype(np.int32) for m in m_values.tolist()
    )
    # Each scalar mode occurs in one group. Pack all factors and mode indices
    # in two uploads, not a persistent canonical table plus gathered copies.
    packed_indices = cupy.asarray(np.concatenate(indices_by_m), dtype=cupy.int32)
    packed_host = np.empty((2, n_beta * nscl), dtype=np.complex128)
    offset = 0
    for indices in indices_by_m:
        width = int(indices.size)
        stop = offset + n_beta * width
        selection = np.ix_(beta_reflection, indices)
        hplus, hminus = _circular_beta_factors(fth_beta[selection], fph_beta[selection])
        packed_host[0, offset:stop] = hplus.reshape(-1)
        packed_host[1, offset:stop] = hminus.reshape(-1)
        offset = stop
    packed_beta = cupy.asarray(packed_host)
    groups = []
    mode_start = 0
    for indices in indices_by_m:
        width = int(indices.size)
        mode_stop = mode_start + width
        factor_start, factor_stop = n_beta * mode_start, n_beta * mode_stop
        groups.append(
            CuPyDirectionalModeData(
                indices=packed_indices[mode_start:mode_stop],
                hplus_beta=packed_beta[0, factor_start:factor_stop].reshape(n_beta, width),
                hminus_beta=packed_beta[1, factor_start:factor_stop].reshape(n_beta, width),
            )
        )
        mode_start = mode_stop

    return CuPyDirectionalTransformsData(
        box_order=box_order,
        grid_order=int(grid.order),
        grid=CuPyDirectionalGridData(
            order=int(grid.order),
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_directions=int(n_dir),
        ),
        nscl=nscl,
        phase_by_m=cupy.asarray(np.ascontiguousarray(phase_by_m, dtype=np.complex128)),
        mode_groups=tuple(groups),
    )


_PACKED_STENCIL_WIDTH_LIMIT = 32


def _upload_packed_stencil(
    csr: Any,
    *,
    cupy: Any,
) -> tuple[Any, Any, np.int32] | None:
    """Pack a low-width CSR map for a fused device transfer kernel."""

    rows = int(csr.shape[0])
    row_ptr = np.ascontiguousarray(np.asarray(csr.indptr, dtype=np.int32))
    row_nnz = np.diff(row_ptr)
    max_row_nnz = int(row_nnz.max(initial=0))
    if max_row_nnz <= 0 or max_row_nnz > _PACKED_STENCIL_WIDTH_LIMIT:
        return None
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
    return (
        cupy.asarray(np.ascontiguousarray(idx_pack.reshape(-1)), dtype=cupy.int32),
        cupy.asarray(np.ascontiguousarray(val_pack.reshape(-1)), dtype=cupy.complex128),
        np.int32(max_row_nnz),
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
    use_packed_stencil = bool(
        nnz > 0 and max_row_nnz > 0 and max_row_nnz <= _PACKED_STENCIL_WIDTH_LIMIT
    )
    if use_packed_stencil:
        packed = _upload_packed_stencil(csr, cupy=cupy)
        if packed is None:
            raise RuntimeError("Internal CuPy MLFMM error: packed map width changed during upload.")
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


def _copy_directional_host(
    transforms: MLFMMDirectionalTransformData,
) -> CuPyHostDirectionalTransformsData:
    """Extract separable directional factors into a compact host payload."""

    grid, fth_beta, fph_beta, m_of_scalar = _directional_structured_host_view(transforms)
    return CuPyHostDirectionalTransformsData(
        box_order=int(transforms.box_order),
        grid_order=int(grid.order),
        grid=grid,
        fth_beta=fth_beta,
        fph_beta=fph_beta,
        m_of_scalar=m_of_scalar,
    )


@cache
def _directional_packed_stencil_map_c128_raw_kernel() -> Any:
    """Compile the packed directional interpolation kernel."""

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
    """Apply one packed, dense, or sparse directional transfer map."""

    arr = cupy.asarray(values, dtype=cupy.complex128)
    if arr.ndim != 4 or int(arr.shape[1]) != _SAMPLED_DIRECTIONAL_CHANNELS:
        raise ValueError(
            "Directional transfer input must have shape "
            f"(nbatch,{_SAMPLED_DIRECTIONAL_CHANNELS},n_source,nrhs)."
        )
    if int(arr.shape[2]) != int(map_data.source_order):
        raise ValueError(
            "Directional transfer source-order mismatch: "
            f"{int(arr.shape[2])} vs {int(map_data.source_order)}."
        )
    n_batch = int(arr.shape[0])
    n_chan = int(arr.shape[1])
    n_rhs = int(arr.shape[3])
    source_order = int(arr.shape[2])
    if str(map_data.storage) == "packed_stencil":
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
    flat = arr.transpose(0, 1, 3, 2).reshape(-1, source_order)
    if str(map_data.storage) == "dense":
        dense = cupy.asarray(map_data.matrix, dtype=cupy.complex128)
        mapped_flat = flat @ dense.T
        target_order = int(dense.shape[0])
    else:
        sparse = map_data.matrix
        mapped_flat = (sparse @ flat.T).T
        target_order = int(sparse.shape[0])
    return mapped_flat.reshape(n_batch, n_chan, n_rhs, target_order).transpose(0, 1, 3, 2)


@cache
def _directional_sum_difference_kernel() -> Any:
    """Mix two owned arrays; each output may alias its matching input.

    Both expressions are evaluated before either write. Callers pass disjoint,
    equally indexed storage, never shifted overlaps or borrowed solver inputs.
    The normalization already belongs to the prepared H+/- beta factors.
    """

    cupy, _ = import_cupy()
    return cupy.ElementwiseKernel(
        "complex128 first, complex128 second",
        "complex128 plus, complex128 minus",
        "const complex<double> p = first + second; "
        "const complex<double> m = first - second; plus = p; minus = m;",
        "pyceles_mlfmm_directional_sum_difference_c128",
    )


@cache
def _directional_phase_add_kernel() -> Any:
    """Compile the phase-weighted circular-channel accumulation kernel."""

    cupy, _ = import_cupy()
    return cupy.ElementwiseKernel(
        "complex128 phase, complex128 values",
        "complex128 output",
        "output += phase * values;",
        "pyceles_mlfmm_directional_phase_add_c128",
    )


def _directional_output(out: Any | None, shape: tuple[int, ...], *, cupy: Any) -> Any:
    """Return the exact destination, never a converted copy of caller storage.

    Inputs may be converted to complex128; an explicit output is different:
    hierarchy callers may ignore the returned array and rely on its mutation.
    Strided CuPy outputs are supported. Callers must keep output and input
    storage disjoint for the duration of the action.
    """

    if out is None:
        return cupy.empty(shape, dtype=cupy.complex128)
    if not isinstance(out, cupy.ndarray):
        raise TypeError("Directional out must be a CuPy array.")
    if out.dtype != cupy.complex128:
        raise ValueError("Directional out must have dtype complex128.")
    if tuple(out.shape) != shape:
        raise ValueError(f"Directional out must have shape {shape}, got {tuple(out.shape)}.")
    return out


def _box_outgoing_to_directional_cupy(
    directional: CuPyDirectionalTransformsData,
    box_states: Any,
    *,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Map outgoing box states to circular channels, optionally into owned storage.

    ``out`` must be an exact-shape complex128 CuPy array disjoint from the input.
    It is returned unchanged as an object, including when it is a strided view.
    """

    states = cupy.asarray(box_states, dtype=cupy.complex128)
    nscl = int(directional.nscl)
    if states.ndim != 3:
        raise ValueError("box_states must have shape (nbatch, 2*nscl, nrhs).")
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
    out_arr = _directional_output(
        out, (n_batch, _SAMPLED_DIRECTIONAL_CHANNELS, ndir, n_rhs), cupy=cupy
    )
    out_arr.fill(0)
    work = out_arr.reshape(n_batch, _SAMPLED_DIRECTIONAL_CHANNELS, n_alpha, n_beta, n_rhs)
    # Rotate coefficients in mode space, not the large sampled-channel grid.
    # H+/- already includes 1/sqrt(2); the two channels are an orthonormal
    # change of the previous (u, v) representation, not a rescaled operator.
    mix = _directional_sum_difference_kernel()
    phase_add = _directional_phase_add_kernel()
    for im, group in enumerate(directional.mode_groups):
        mode_idx = group.indices
        if int(mode_idx.size) == 0:
            continue
        plus_m = a_box[:, mode_idx, :]
        minus_m = b_box[:, mode_idx, :]
        mix(plus_m, minus_m, plus_m, minus_m)
        plus_beta = cupy.matmul(group.hplus_beta[None, :, :], plus_m)
        minus_beta = cupy.matmul(group.hminus_beta[None, :, :], minus_m)
        del plus_m, minus_m
        phase = directional.phase_by_m[:, im].reshape(1, n_alpha, 1, 1)
        phase_add(phase, plus_beta[:, None, :, :], work[:, 0])
        phase_add(phase, minus_beta[:, None, :, :], work[:, 1])
        del plus_beta, minus_beta
    return out_arr


def _directional_to_box_regular_cupy(
    directional: CuPyDirectionalTransformsData,
    directional_channels: Any,
    *,
    out: Any | None = None,
    cupy: Any,
) -> Any:
    """Map circular channels to regular box states, optionally into owned storage.

    ``out`` must be an exact-shape complex128 CuPy array disjoint from the input.
    It is returned unchanged as an object, including when it is a strided view.
    """

    channels = cupy.asarray(directional_channels, dtype=cupy.complex128)
    if channels.ndim != 4:
        raise ValueError("directional_channels must have shape (nbatch, 2, ndir, nrhs).")
    if int(channels.shape[1]) != _SAMPLED_DIRECTIONAL_CHANNELS:
        raise ValueError(
            "directional channel batch must have "
            f"{_SAMPLED_DIRECTIONAL_CHANNELS} channels, got {int(channels.shape[1])}."
        )
    if int(channels.shape[2]) != int(directional.grid.n_directions):
        raise ValueError("Directional channel direction count must match the prepared grid.")
    n_batch = int(channels.shape[0])
    n_rhs = int(channels.shape[3])
    n_alpha = int(directional.grid.n_alpha)
    n_beta = int(directional.grid.n_beta)
    nscl = int(directional.nscl)
    # Uploaded factors already contain the beta reflection. Keep the input as
    # a view rather than materializing a reflected directional copy.
    channel_grid = channels.reshape(n_batch, _SAMPLED_DIRECTIONAL_CHANNELS, n_alpha, n_beta, n_rhs)
    out_arr = _directional_output(out, (n_batch, 2 * nscl, n_rhs), cupy=cupy)
    top = out_arr[:, :nscl, :]
    bottom = out_arr[:, nscl:, :]
    phase_adj = cupy.conjugate(directional.phase_by_m)
    mix = _directional_sum_difference_kernel()
    for im, group in enumerate(directional.mode_groups):
        mode_idx = group.indices
        if int(mode_idx.size) == 0:
            continue
        phase_m = phase_adj[:, im]
        plus_beta = cupy.einsum("a,bakr->bkr", phase_m, channel_grid[:, 0], optimize=True)
        minus_beta = cupy.einsum("a,bakr->bkr", phase_m, channel_grid[:, 1], optimize=True)
        hplus_h = cupy.conjugate(group.hplus_beta).T
        hminus_h = cupy.conjugate(group.hminus_beta).T
        plus_box = cupy.matmul(hplus_h[None, :, :], plus_beta)
        minus_box = cupy.matmul(hminus_h[None, :, :], minus_beta)
        del plus_beta, minus_beta
        mix(plus_box, minus_box, plus_box, minus_box)
        top[:, mode_idx, :] = plus_box
        bottom[:, mode_idx, :] = minus_box
        del plus_box, minus_box, hplus_h, hminus_h
    return out_arr


def _box_outgoing_to_directional_adjoint_cupy(
    directional: CuPyDirectionalTransformsData, directional_channels: Any, *, cupy: Any
) -> Any:
    """Apply the adjoint of the outgoing directional transform."""

    return _directional_to_box_regular_cupy(directional, directional_channels, cupy=cupy)


def _directional_to_box_regular_adjoint_cupy(
    directional: CuPyDirectionalTransformsData, box_states: Any, *, cupy: Any
) -> Any:
    """Apply the adjoint of the regular directional receive transform."""

    return _box_outgoing_to_directional_cupy(directional, box_states, cupy=cupy)
