"""CuPy helpers for the hybrid periodic Rayleigh operator."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import import_cupy
from pyceles.core.indexing import index_vswf
from pyceles.core.periodic.rayleigh import (
    RayleighPlan,
    _cartesian_xy_z_layout,
    resolve_rayleigh_mode_chunk_size,
)


def _cuda_types(dtype_name: str) -> tuple[str, str, str, str, str]:
    dtype = np.dtype(dtype_name)
    if dtype == np.dtype(np.complex64):
        return "float", "complex<float>", "c64", "sincosf", "expf"
    if dtype == np.dtype(np.complex128):
        return "double", "complex<double>", "c128", "sincos", "exp"
    raise TypeError(f"Unsupported Rayleigh CUDA dtype {dtype!r}.")


@cache
def _scan_kernel_source(dtype_name: str) -> tuple[str, str]:
    scalar, complex_type, suffix, sincos, exponential = _cuda_types(dtype_name)
    kernel_name = f"pyceles_rayleigh_scan_{suffix}"
    source = f"""
#include <cupy/complex.cuh>

__device__ inline {complex_type} pyceles_rayleigh_phase_{suffix}(
    const complex<double> gamma,
    const double distance)
{{
    const {scalar} scaled_distance = ({scalar})distance;
    const {scalar} gamma_imag = ({scalar})gamma.imag();
    if (gamma_imag != ({scalar})0) {{
        return {complex_type}(
            {exponential}(-gamma_imag * scaled_distance),
            ({scalar})0
        );
    }}
    {scalar} sine;
    {scalar} cosine;
    {sincos}(({scalar})gamma.real() * scaled_distance, &sine, &cosine);
    return {complex_type}(cosine, sine);
}}

extern "C" __global__
void {kernel_name}(
    const {complex_type}* source,
    const double* z,
    const complex<double>* gamma,
    const long long n_particles,
    const long long n_q,
    const long long n_pol,
    const long long n_rhs,
    const double z_cut,
    const int upward,
    {complex_type}* output)
{{
    const long long lane = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long n_lanes = n_q * n_pol * n_rhs;
    if (lane >= n_lanes) return;
    const long long rhs = lane % n_rhs;
    const long long temp = lane / n_rhs;
    const long long pol = temp % n_pol;
    const long long q = temp / n_pol;
    {complex_type} accumulator = {complex_type}(0.0, 0.0);
    if (upward != 0) {{
        long long pointer = 0;
        for (long long i = 0; i < n_particles; ++i) {{
            if (i > 0) {{
                accumulator *= pyceles_rayleigh_phase_{suffix}(gamma[q], z[i] - z[i - 1]);
            }}
            while (pointer < i && z[i] - z[pointer] > z_cut) {{
                const long long src_idx = (((pointer * n_q + q) * n_pol + pol) * n_rhs + rhs);
                accumulator += source[src_idx]
                    * pyceles_rayleigh_phase_{suffix}(gamma[q], z[i] - z[pointer]);
                ++pointer;
            }}
            const long long out_idx = (((i * n_q + q) * n_pol + pol) * n_rhs + rhs);
            output[out_idx] = accumulator;
        }}
    }} else {{
        long long pointer = n_particles - 1;
        for (long long i = n_particles - 1; i >= 0; --i) {{
            if (i + 1 < n_particles) {{
                accumulator *= pyceles_rayleigh_phase_{suffix}(gamma[q], z[i + 1] - z[i]);
            }}
            while (pointer > i && z[pointer] - z[i] > z_cut) {{
                const long long src_idx = (((pointer * n_q + q) * n_pol + pol) * n_rhs + rhs);
                accumulator += source[src_idx]
                    * pyceles_rayleigh_phase_{suffix}(gamma[q], z[pointer] - z[i]);
                --pointer;
            }}
            const long long out_idx = (((i * n_q + q) * n_pol + pol) * n_rhs + rhs);
            output[out_idx] = accumulator;
        }}
    }}
}}
"""
    return kernel_name, source


@cache
def _scan_kernel(dtype_name: str) -> Any:
    cp, _ = import_cupy()
    name, source = _scan_kernel_source(dtype_name)
    return cp.RawKernel(source, name)


@cache
def _point_scan_kernel_source(dtype_name: str) -> tuple[str, str]:
    scalar, complex_type, suffix, sincos, exponential = _cuda_types(dtype_name)
    kernel_name = f"pyceles_rayleigh_point_scan_{suffix}"
    source = f"""
#include <cupy/complex.cuh>

__device__ inline {complex_type} pyceles_rayleigh_point_phase_{suffix}(
    const complex<double> gamma,
    const double distance)
{{
    const {scalar} scaled_distance = ({scalar})distance;
    const {scalar} gamma_imag = ({scalar})gamma.imag();
    if (gamma_imag != ({scalar})0) {{
        return {complex_type}(
            {exponential}(-gamma_imag * scaled_distance),
            ({scalar})0
        );
    }}
    {scalar} sine;
    {scalar} cosine;
    {sincos}(({scalar})gamma.real() * scaled_distance, &sine, &cosine);
    return {complex_type}(cosine, sine);
}}

extern "C" __global__
void {kernel_name}(
    const {complex_type}* source,
    const double* source_z,
    const double* destination_z,
    const complex<double>* gamma,
    const long long n_sources,
    const long long n_destinations,
    const long long n_q,
    const long long n_pol,
    const long long n_rhs,
    const double z_cut,
    const int upward,
    {complex_type}* output)
{{
    const long long lane = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long n_lanes = n_q * n_pol * n_rhs;
    if (lane >= n_lanes) return;
    const long long rhs = lane % n_rhs;
    const long long temp = lane / n_rhs;
    const long long pol = temp % n_pol;
    const long long q = temp / n_pol;
    {complex_type} accumulator = {complex_type}(0.0, 0.0);
    if (upward != 0) {{
        long long pointer = 0;
        for (long long i = 0; i < n_destinations; ++i) {{
            if (i > 0) {{
                accumulator *= pyceles_rayleigh_point_phase_{suffix}(
                    gamma[q], destination_z[i] - destination_z[i - 1]
                );
            }}
            while (pointer < n_sources && destination_z[i] - source_z[pointer] > z_cut) {{
                const long long src_idx = (((pointer * n_q + q) * n_pol + pol) * n_rhs + rhs);
                accumulator += source[src_idx]
                    * pyceles_rayleigh_point_phase_{suffix}(
                        gamma[q], destination_z[i] - source_z[pointer]
                    );
                ++pointer;
            }}
            const long long out_idx = (((i * n_q + q) * n_pol + pol) * n_rhs + rhs);
            output[out_idx] = accumulator;
        }}
    }} else {{
        long long pointer = n_sources - 1;
        for (long long i = n_destinations - 1; i >= 0; --i) {{
            if (i + 1 < n_destinations) {{
                accumulator *= pyceles_rayleigh_point_phase_{suffix}(
                    gamma[q], destination_z[i + 1] - destination_z[i]
                );
            }}
            while (pointer >= 0 && source_z[pointer] - destination_z[i] > z_cut) {{
                const long long src_idx = (((pointer * n_q + q) * n_pol + pol) * n_rhs + rhs);
                accumulator += source[src_idx]
                    * pyceles_rayleigh_point_phase_{suffix}(
                        gamma[q], source_z[pointer] - destination_z[i]
                    );
                --pointer;
            }}
            const long long out_idx = (((i * n_q + q) * n_pol + pol) * n_rhs + rhs);
            output[out_idx] = accumulator;
        }}
    }}
}}
"""
    return kernel_name, source


@cache
def _point_scan_kernel(dtype_name: str) -> Any:
    cp, _ = import_cupy()
    name, source = _point_scan_kernel_source(dtype_name)
    return cp.RawKernel(source, name)


@cache
def _scatter_kernel_source(dtype_name: str) -> tuple[str, str]:
    scalar, complex_type, suffix, _, _ = _cuda_types(dtype_name)
    kernel_name = f"pyceles_rayleigh_scatter_{suffix}"
    source = f"""
#include <cupy/complex.cuh>
extern "C" __global__
void {kernel_name}(
    const long long n_pairs,
    const long long width,
    const int* destinations,
    const {complex_type}* values,
    {scalar}* output)
{{
    const long long flat = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = n_pairs * width;
    if (flat >= total) return;
    const long long pair = flat / width;
    const long long lane = flat - pair * width;
    const long long output_index = ((long long)destinations[pair] * width + lane) * 2;
    const {complex_type} value = values[flat];
    atomicAdd(output + output_index, ({scalar})value.real());
    atomicAdd(output + output_index + 1, ({scalar})value.imag());
}}
"""
    return kernel_name, source


@cache
def _scatter_kernel(dtype_name: str) -> Any:
    cp, _ = import_cupy()
    name, source = _scatter_kernel_source(dtype_name)
    return cp.RawKernel(source, name)


def scatter_add_complex(target: Any, indices: Any, values: Any, *, cupy: Any) -> None:
    """Accumulate complex rows at possibly repeated destination indices.

    CuPy ``add.at`` does not support complex targets on all supported CUDA
    stacks. One thread handles each scalar complex value and atomically adds
    its real and imaginary components to the flattened target.
    """
    dtype = np.dtype(target.dtype)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise TypeError(f"Unsupported Rayleigh scatter dtype {dtype!r}.")
    output = cupy.asarray(target, dtype=dtype)
    if not bool(output.flags.c_contiguous):
        raise ValueError("Rayleigh scatter target must be C-contiguous.")
    destination = cupy.asarray(indices, dtype=cupy.int32).reshape(-1)
    contribution = cupy.ascontiguousarray(cupy.asarray(values, dtype=dtype))
    if int(contribution.shape[0]) != int(destination.size):
        raise ValueError("Rayleigh scatter indices and values have different pair counts.")
    width = int(np.prod(contribution.shape[1:], dtype=np.int64))
    contribution = contribution.reshape(int(destination.size), width)
    output = output.reshape(int(output.shape[0]), width)
    scalar_dtype = cupy.float32 if dtype == np.dtype(np.complex64) else cupy.float64
    output_scalar = output.view(scalar_dtype).reshape(-1)
    total = int(destination.size) * width
    if total == 0:
        return
    threads = 128
    blocks = (total + threads - 1) // threads
    kernel = _scatter_kernel(dtype.name)
    kernel(
        (blocks,),
        (threads,),
        (
            np.int64(destination.size),
            np.int64(width),
            destination,
            contribution,
            output_scalar,
        ),
    )


@cache
def _sparse_near_kernel_source(dtype_name: str) -> tuple[str, str]:
    scalar, complex_type, suffix, _, _ = _cuda_types(dtype_name)
    kernel_name = f"pyceles_rayleigh_sparse_near_{suffix}"
    source = f"""
#include <cupy/complex.cuh>
extern "C" __global__
void {kernel_name}(
    const long long n_pairs,
    const long long n_modes,
    const long long n_structural,
    const long long n_rhs,
    const int* sources,
    const int* destinations,
    const {complex_type}* structural,
    const long long* row_ptr,
    const int* input_modes,
    const int* structural_channels,
    const {complex_type}* coefficients,
    const {complex_type}* values,
    {scalar}* output)
{{
    const long long flat = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = n_pairs * n_modes * n_rhs;
    if (flat >= total) return;
    const long long rhs = flat % n_rhs;
    const long long mode_lane = flat / n_rhs;
    const long long output_mode = mode_lane % n_modes;
    const long long pair = mode_lane / n_modes;
    const long long source = (long long)sources[pair];
    {complex_type} accumulator = {complex_type}(0.0, 0.0);
    for (long long entry = row_ptr[output_mode]; entry < row_ptr[output_mode + 1]; ++entry) {{
        const long long input_mode = (long long)input_modes[entry];
        const long long structural_channel = (long long)structural_channels[entry];
        const {complex_type} incident = coefficients[
            (source * n_modes + input_mode) * n_rhs + rhs
        ];
        accumulator += structural[pair * n_structural + structural_channel]
            * values[entry] * incident;
    }}
    const long long destination = (long long)destinations[pair];
    const long long output_index = (
        (destination * n_modes + output_mode) * n_rhs + rhs
    ) * 2;
    atomicAdd(output + output_index, ({scalar})accumulator.real());
    atomicAdd(output + output_index + 1, ({scalar})accumulator.imag());
}}
"""
    return kernel_name, source


@cache
def _sparse_near_kernel(dtype_name: str) -> Any:
    cp, _ = import_cupy()
    name, source = _sparse_near_kernel_source(dtype_name)
    return cp.RawKernel(source, name)


def apply_sparse_near_coupling_cupy(
    *,
    target: Any,
    structural: Any,
    coefficients: Any,
    sources: Any,
    destinations: Any,
    row_ptr: Any,
    input_modes: Any,
    structural_channels: Any,
    values: Any,
    cupy: Any,
) -> None:
    """Accumulate sparse exact-near translation contractions on device."""
    dtype = np.dtype(target.dtype)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise TypeError(f"Unsupported Rayleigh sparse-near dtype {dtype!r}.")
    output = cupy.asarray(target, dtype=dtype)
    structure = cupy.ascontiguousarray(cupy.asarray(structural, dtype=dtype))
    coeff = cupy.ascontiguousarray(cupy.asarray(coefficients, dtype=dtype))
    if output.ndim != 3 or coeff.ndim != 3 or structure.ndim != 2:
        raise ValueError("Sparse Rayleigh near inputs must have ranks 3, 3, and 2.")
    if output.shape != coeff.shape:
        raise ValueError("Sparse Rayleigh target and coefficient blocks must have equal shapes.")
    source = cupy.asarray(sources, dtype=cupy.int32).reshape(-1)
    destination = cupy.asarray(destinations, dtype=cupy.int32).reshape(-1)
    if int(source.size) != int(destination.size) or int(source.size) != int(structure.shape[0]):
        raise ValueError("Sparse Rayleigh pair arrays must have equal lengths.")
    pointers = cupy.asarray(row_ptr, dtype=cupy.int64).reshape(-1)
    in_modes = cupy.asarray(input_modes, dtype=cupy.int32).reshape(-1)
    channels = cupy.asarray(structural_channels, dtype=cupy.int32).reshape(-1)
    factors = cupy.asarray(values, dtype=dtype).reshape(-1)
    if not (int(in_modes.size) == int(channels.size) == int(factors.size)):
        raise ValueError("Sparse Rayleigh contraction arrays must have equal lengths.")
    _n_particles, n_modes, n_rhs = (int(value) for value in output.shape)
    if int(pointers.size) != n_modes + 1:
        raise ValueError("Sparse Rayleigh row pointers must cover every output mode.")
    total = int(source.size) * n_modes * n_rhs
    if total == 0:
        return
    scalar_dtype = cupy.float32 if dtype == np.dtype(np.complex64) else cupy.float64
    output_scalar = output.view(scalar_dtype).reshape(-1)
    threads = 128
    blocks = (total + threads - 1) // threads
    kernel = _sparse_near_kernel(dtype.name)
    kernel(
        (blocks,),
        (threads,),
        (
            np.int64(source.size),
            np.int64(n_modes),
            np.int64(structure.shape[1]),
            np.int64(n_rhs),
            source,
            destination,
            structure,
            pointers,
            in_modes,
            channels,
            coeff,
            factors,
            output_scalar,
        ),
    )


def scan_far_cupy(
    *,
    source_amplitudes: Any,
    z: Any,
    gamma: Any,
    z_cut: float,
    upward: bool,
    cupy: Any,
) -> Any:
    """Apply the stable delayed-entry scan on the device."""
    src = cupy.ascontiguousarray(source_amplitudes)
    if int(src.ndim) != 4:
        raise ValueError("`source_amplitudes` must have shape (N, Nq, 2, nrhs).")
    n_particles, n_q, n_pol, n_rhs = (int(v) for v in src.shape)
    out = cupy.empty_like(src)
    kernel = _scan_kernel(np.dtype(src.dtype).name)
    lanes = n_q * n_pol * n_rhs
    threads = 128
    blocks = (lanes + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            src,
            cupy.asarray(z, dtype=cupy.float64),
            cupy.asarray(gamma, dtype=cupy.complex128),
            np.int64(n_particles),
            np.int64(n_q),
            np.int64(n_pol),
            np.int64(n_rhs),
            np.float64(z_cut),
            np.int32(1 if upward else 0),
            out,
        ),
    )
    return out


def scan_far_to_points_cupy(
    *,
    source_amplitudes: Any,
    source_z: Any,
    destination_z: Any,
    gamma: Any,
    z_cut: float,
    upward: bool,
    cupy: Any,
) -> Any:
    """Apply the stable delayed-entry scan to a distinct destination z grid."""
    src = cupy.ascontiguousarray(source_amplitudes)
    if int(src.ndim) != 4:
        raise ValueError("`source_amplitudes` must have shape (N, Nq, 2, nrhs).")
    n_sources, n_q, n_pol, n_rhs = (int(v) for v in src.shape)
    destination = cupy.ascontiguousarray(
        cupy.asarray(destination_z, dtype=cupy.float64).reshape(-1)
    )
    out = cupy.empty((int(destination.size), n_q, n_pol, n_rhs), dtype=src.dtype)
    if n_sources == 0 or int(destination.size) == 0:
        out.fill(0)
        return out
    kernel = _point_scan_kernel(np.dtype(src.dtype).name)
    source_z_cp = cupy.ascontiguousarray(cupy.asarray(source_z, dtype=cupy.float64).reshape(-1))
    gamma_cp = cupy.ascontiguousarray(cupy.asarray(gamma, dtype=cupy.complex128).reshape(-1))
    lanes = n_q * n_pol * n_rhs
    threads = 128
    blocks = (lanes + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            src,
            source_z_cp,
            destination,
            gamma_cp,
            np.int64(n_sources),
            np.int64(destination.size),
            np.int64(n_q),
            np.int64(n_pol),
            np.int64(n_rhs),
            np.float64(z_cut),
            np.int32(1 if upward else 0),
            out,
        ),
    )
    return out


def apply_rayleigh_far_to_points_cupy(
    plan: RayleighPlan,
    coeffs: Any,
    points: Any,
    *,
    cupy: Any,
) -> Any:
    """Return far-source point-local regular ``l=1`` coefficients on the GPU.

    Complete Cartesian ``XY x Z`` point sets use a compact path: the delayed
    Rayleigh scan is evaluated only on unique z coordinates, lateral phases
    only on unique xy coordinates, and the two are recombined with GEMM.
    Arbitrary point clouds retain the original point-wise implementation.
    """
    coeff_raw = cupy.asarray(coeffs)
    squeezed = int(coeff_raw.ndim) == 2
    if squeezed:
        coeff_arr = coeff_raw[:, :, None]
    elif int(coeff_raw.ndim) == 3:
        coeff_arr = coeff_raw
    else:
        raise ValueError("Rayleigh coefficients must have shape (N, Nm) or (N, Nm, nrhs).")
    if int(coeff_arr.shape[0]) != int(plan.sort_order.size):
        raise ValueError("Rayleigh coefficient particle count does not match the plan.")
    if int(coeff_arr.shape[1]) != int(plan.source_tables.shape[-1]):
        raise ValueError("Rayleigh coefficient mode count does not match the plan.")

    pts_np = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = int(pts_np.shape[0])
    n_rhs = int(coeff_arr.shape[2])
    dtype = np.dtype(plan.source_tables.dtype)
    if n_points == 0:
        empty = cupy.zeros((0, 6, n_rhs), dtype=dtype)
        return empty[:, :, 0] if squeezed else empty

    reciprocal = cupy.asarray(plan.reciprocal_xy, dtype=cupy.float64)
    source_order = cupy.asarray(plan.sort_order, dtype=cupy.int64)
    coeff_sorted = cupy.asarray(coeff_arr[source_order], dtype=dtype)
    source_phase_all = cupy.asarray(plan.sorted_xy_phase, dtype=dtype)
    source_tables = cupy.asarray(plan.source_tables, dtype=dtype)
    destination_tables = cupy.asarray(plan.destination_tables, dtype=dtype)
    gamma_all = cupy.asarray(plan.gamma, dtype=cupy.complex128)
    weights = cupy.asarray(plan.weights, dtype=dtype)
    source_z = cupy.asarray(plan.sorted_z, dtype=cupy.float64)
    l1_indices = cupy.asarray(
        [index_vswf(1, m, tau, plan.lmax) for tau in (1, 2) for m in (-1, 0, 1)],
        dtype=cupy.int64,
    )
    # The six destination l=1 rows are invariant across point batches and
    # reciprocal chunks. Gather them once instead of allocating one take()
    # result for every direction/chunk pair below.
    destination_l1_tables = cupy.ascontiguousarray(
        cupy.take(destination_tables, l1_indices, axis=-1)
    )

    layout = _cartesian_xy_z_layout(pts_np)
    if layout is not None:
        xy_unique, z_unique, xy_inverse, z_inverse = layout
        n_xy = int(xy_unique.shape[0])
        n_z = int(z_unique.size)
        xy_cp = cupy.asarray(xy_unique, dtype=cupy.float64)
        z_cp = cupy.asarray(z_unique, dtype=cupy.float64)
        y_grid = cupy.zeros((n_z, n_xy, 6, n_rhs), dtype=dtype)
        # Account for both the two-polarization incoming scan and the projected
        # six-mode table when sizing a reciprocal chunk.
        chunk = resolve_rayleigh_mode_chunk_size(
            n_modes_reciprocal=plan.n_modes_reciprocal,
            n_particles=int(coeff_arr.shape[0]),
            n_destinations=max(1, 4 * n_z),
            n_phase_rows=n_xy,
            n_rhs=n_rhs,
            dtype=dtype,
        )
        for start in range(0, plan.n_modes_reciprocal, chunk):
            stop = min(plan.n_modes_reciprocal, start + chunk)
            q_count = int(stop - start)
            source_phase = source_phase_all[:, start:stop]
            destination_phase = cupy.exp(1j * (xy_cp @ reciprocal[start:stop].T)).astype(
                dtype, copy=False
            )
            gamma = gamma_all[start:stop]
            weight = weights[start:stop]
            for direction, upward in ((0, True), (1, False)):
                source = cupy.einsum(
                    "qpm,amr->aqpr",
                    source_tables[direction, start:stop],
                    coeff_sorted,
                    optimize=True,
                )
                source *= cupy.conjugate(source_phase)[:, :, None, None]
                incoming = scan_far_to_points_cupy(
                    source_amplitudes=source,
                    source_z=source_z,
                    destination_z=z_cp,
                    gamma=gamma,
                    z_cut=float(plan.z_cut),
                    upward=upward,
                    cupy=cupy,
                )
                destination_l1 = destination_l1_tables[direction, start:stop]
                projected = cupy.einsum(
                    "qpm,zqpr->zqmr",
                    destination_l1,
                    incoming,
                    optimize=True,
                )
                projected *= weight[None, :, None, None]
                projected_rows = cupy.ascontiguousarray(projected.transpose(0, 2, 3, 1)).reshape(
                    n_z * 6 * n_rhs, q_count
                )
                contribution = projected_rows @ destination_phase.T
                y_grid += contribution.reshape(n_z, 6, n_rhs, n_xy).transpose(0, 3, 1, 2)
                del source, incoming, projected, projected_rows, contribution
        z_inverse_cp = cupy.asarray(z_inverse, dtype=cupy.int64)
        xy_inverse_cp = cupy.asarray(xy_inverse, dtype=cupy.int64)
        y = y_grid[z_inverse_cp, xy_inverse_cp]
        return y[:, :, 0] if squeezed else y

    destination_order = np.argsort(pts_np[:, 2], kind="stable").astype(np.int64, copy=False)
    destination_inverse = np.empty_like(destination_order)
    destination_inverse[destination_order] = np.arange(n_points, dtype=np.int64)
    pts_sorted = cupy.asarray(pts_np[destination_order], dtype=cupy.float64)
    y_sorted = cupy.zeros((n_points, 6, n_rhs), dtype=dtype)
    chunk = resolve_rayleigh_mode_chunk_size(
        n_modes_reciprocal=plan.n_modes_reciprocal,
        n_particles=int(coeff_arr.shape[0]),
        n_destinations=n_points,
        n_phase_rows=n_points,
        n_rhs=n_rhs,
        dtype=dtype,
    )
    for start in range(0, plan.n_modes_reciprocal, chunk):
        stop = min(plan.n_modes_reciprocal, start + chunk)
        source_phase = source_phase_all[:, start:stop]
        destination_phase = cupy.exp(1j * (pts_sorted[:, :2] @ reciprocal[start:stop].T)).astype(
            dtype, copy=False
        )
        gamma = gamma_all[start:stop]
        for direction, upward in ((0, True), (1, False)):
            source = cupy.einsum(
                "qpm,amr->aqpr",
                source_tables[direction, start:stop],
                coeff_sorted,
                optimize=True,
            )
            source *= cupy.conjugate(source_phase)[:, :, None, None]
            incoming = scan_far_to_points_cupy(
                source_amplitudes=source,
                source_z=source_z,
                destination_z=pts_sorted[:, 2],
                gamma=gamma,
                z_cut=float(plan.z_cut),
                upward=upward,
                cupy=cupy,
            )
            destination_l1 = destination_l1_tables[direction, start:stop]
            y_sorted += cupy.einsum(
                "qpm,dqpr,dq,q->dmr",
                destination_l1,
                incoming,
                destination_phase,
                weights[start:stop],
                optimize=True,
            )
            del source, incoming
    destination_inverse_cp = cupy.asarray(destination_inverse, dtype=cupy.int64)
    y = y_sorted[destination_inverse_cp]
    return y[:, :, 0] if squeezed else y


__all__ = [
    "apply_rayleigh_far_to_points_cupy",
    "apply_sparse_near_coupling_cupy",
    "scan_far_cupy",
    "scan_far_to_points_cupy",
    "scatter_add_complex",
]
