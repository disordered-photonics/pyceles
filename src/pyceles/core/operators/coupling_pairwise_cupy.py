"""Fused CuPy pairwise coupling for direct GPU matvecs.

We explicitly ship the CuPy direct backend as a fused RawKernel rather than as
a composition of higher-level CuPy operations. That higher-level decomposition
was explored separately during development, but for the direct `O(N^2)` pairwise
matvec it produced too many small GPU operations, too many temporary arrays,
too much Python-side orchestration, and too much kernel-launch overhead to stay
competitive.

The shipped CuPy path therefore keeps the translation tables on device and
accumulates the pairwise coupling inside the fused kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
)

from .mode_metadata import mode_metadata_tables, mode_pair_p_range_tables

COMPLEX128_DTYPE = np.dtype(np.complex128)


@cache
def _translation_matvec_raw_kernel(lmax: int, dtype_str: str):
    cupy, _ = import_cupy()
    lmax = int(lmax)
    dtype = np.dtype(dtype_str)
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported CuPy raw-kernel dtype {dtype!r}.")

    real_type = "float" if dtype == np.dtype(np.complex64) else "double"
    complex_type = "complex<float>" if dtype == np.dtype(np.complex64) else "complex<double>"
    math = {
        "atan2": "atan2f" if real_type == "float" else "atan2",
        "floor": "floorf" if real_type == "float" else "floor",
        "max": "fmaxf" if real_type == "float" else "fmax",
        "sincos": "sincosf" if real_type == "float" else "sincos",
        "sqrt": "sqrtf" if real_type == "float" else "sqrt",
    }
    kernel_name = (
        "translation_matrix_product_c64"
        if real_type == "float"
        else "translation_matrix_product_c128"
    )

    n_orders = 2 * lmax + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    #include <cupy/complex.cuh>
    // This kernel evaluates one direct many-body coupling matvec y = W x.
    //
    // Conventions:
    // - `s1` is the destination particle index (row block of W).
    // - `s2` is the source particle index (column block of W).
    // - `n1`/`n2` are CELES/SMUTHI SVWF mode indices within one particle block.
    // - `re_ab` / `im_ab` store the translation prefactors in the same compact
    //   loop order as the nested (n1, n2, p) traversal below.
    //
    // Launch geometry:
    // - one CUDA block handles one destination particle `s1` and one tile of
    //   output modes `n1`
    // - threads in that block reuse the same geometry-dependent quantities for
    //   each source particle through shared memory
    //
    // This structure avoids the earlier one-launch-per-source pattern, which
    // left the GPU badly under-occupied for low-lmax many-particle cases.

    __device__ {real_type} assoc_legendre_function(
        const int l,
        const int m,
        const {real_type}* ct_powers,
        const {real_type}* st_powers,
        const {real_type}* plm_coeffs
    ) {{
        {real_type} plm = 0;
        const {real_type} st_pow = st_powers[m];
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    __device__ {real_type} hankel_lookup_linear(
        const int p,
        const {real_type} r,
        const {real_type}* table,
        const {real_type} inv_dr,
        const int last_index
    ) {{
        if (r <= ({real_type})0) {{
            return table[p];
        }}
        {real_type} t = r * inv_dr;
        int i0 = (int){math["floor"]}(t);
        {real_type} frac = t - ({real_type})i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = ({real_type})0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = ({real_type})1;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        return (({real_type})1 - frac) * table[base0] + frac * table[base1];
    }}

    extern "C" __global__ void {kernel_name}(
        const int ns,
        const int nmodes,
        const int nrhs,
        const {real_type}* positions,
        const {real_type}* re_h,
        const {real_type}* im_h,
        const {real_type} inv_dr,
        const int last_index,
        const {real_type}* plm_coeffs,
        const {real_type}* re_ab,
        const {real_type}* im_ab,
        const int* mode_m,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        const {complex_type}* x,
        {complex_type}* wx
    ) {{
        const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
        if (n1 >= nmodes) {{
            return;
        }}
        __shared__ {real_type} re_h_shared[{n_orders}];
        __shared__ {real_type} im_h_shared[{n_orders}];
        __shared__ {real_type} p_pdm_shared[{n_p_pdm}];
        __shared__ {real_type} cos_mphi_shared[{n_phase}];
        __shared__ {real_type} sin_mphi_shared[{n_phase}];
        __shared__ {real_type} ct_pow_shared[{n_orders}];
        __shared__ {real_type} st_pow_shared[{n_orders}];
        __shared__ {real_type} r_shared;
        __shared__ {real_type} ct_shared;
        __shared__ {real_type} st_shared;
        __shared__ {real_type} phi_shared;

        const int m1 = mode_m[n1];

        for (int rhs = blockIdx.z; rhs < nrhs; rhs += gridDim.z) {{
            for (int s1 = blockIdx.y; s1 < ns; s1 += gridDim.y) {{
                {real_type} re_incr = ({real_type})0;
                {real_type} im_incr = ({real_type})0;

                for (int s2 = 0; s2 < ns; ++s2) {{
                    if (s2 == s1) {{
                        continue;
                    }}

                    if (threadIdx.x == 0) {{
                        // Geometry and angular factors depend only on the particle pair
                        // (s1, s2), not on the output mode. Compute them once per block
                        // and let all mode threads reuse them. The heavier table
                        // fills below are then distributed cooperatively.
                        const {real_type} x21 = positions[3 * s1] - positions[3 * s2];
                        const {real_type} y21 = positions[3 * s1 + 1] - positions[3 * s2 + 1];
                        const {real_type} z21 = positions[3 * s1 + 2] - positions[3 * s2 + 2];
                        r_shared = {math["sqrt"]}(x21 * x21 + y21 * y21 + z21 * z21);
                        ct_shared = z21 / r_shared;
                        st_shared = {math["sqrt"]}(
                            {math["max"]}(({real_type})0, ({real_type})1 - ct_shared * ct_shared)
                        );
                        phi_shared = {math["atan2"]}(y21, x21);
                        ct_pow_shared[0] = ({real_type})1;
                        st_pow_shared[0] = ({real_type})1;
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
                            ({real_type})dm * phi_shared,
                            &sin_mphi_shared[idx],
                            &cos_mphi_shared[idx]
                        );
                    }}
                    __syncthreads();

                    for (int n2 = 0; n2 < nmodes; ++n2) {{
                        // We intentionally read x[s2, n2] directly from global memory.
                        // Profiling on the 5k-particle c64 benchmark showed that
                        // staging this small mode vector in shared memory did not
                        // produce a material end-to-end speedup, while it made the
                        // kernel more verbose and stateful.
                        const int x_idx = ((s2 * nmodes + n2) * nrhs) + rhs;
                        const {complex_type} x_tmp = x[x_idx];
                        const {real_type} re_x_tmp = x_tmp.real();
                        const {real_type} im_x_tmp = x_tmp.imag();
                        const int delta_m = mode_m[n2] - m1;
                        const int phase_idx = delta_m + 2 * {lmax};
                        const int pair_idx = n1 * nmodes + n2;
                        const int base = pair_offset[pair_idx];
                        const int p_min = pair_pmin[pair_idx];
                        const int p_count = pair_pcount[pair_idx];
                        // `pair_*` turns the CELES/SMUTHI triangular p-range for each
                        // (n1, n2) pair into a simple flat interval in the compact ab5
                        // arrays. That keeps the kernel inner loop branch-light.
                        for (int ip = 0; ip < p_count; ++ip) {{
                            const int p = p_min + ip;
                            const int ab_idx = base + ip;
                            const {real_type} plm = p_pdm_shared[p * (p + 1) / 2 + abs(delta_m)];
                            const {real_type} re_abp = re_ab[ab_idx] * plm;
                            const {real_type} im_abp = im_ab[ab_idx] * plm;
                            const {real_type} re_abph =
                                re_abp * re_h_shared[p] - im_abp * im_h_shared[p];
                            const {real_type} im_abph =
                                re_abp * im_h_shared[p] + im_abp * re_h_shared[p];
                            const {real_type} re_phase =
                                re_abph * cos_mphi_shared[phase_idx] - im_abph * sin_mphi_shared[phase_idx];
                            const {real_type} im_phase =
                                re_abph * sin_mphi_shared[phase_idx] + im_abph * cos_mphi_shared[phase_idx];
                            re_incr += re_phase * re_x_tmp - im_phase * im_x_tmp;
                            im_incr += re_phase * im_x_tmp + im_phase * re_x_tmp;
                        }}
                    }}
                    __syncthreads();
                }}

                const int y_idx = ((s1 * nmodes + n1) * nrhs) + rhs;
                wx[y_idx] = {complex_type}(re_incr, im_incr);
            }}
        }}
    }}
    """
    return cupy.RawKernel(source, kernel_name)


@dataclass
class CuPyPairwiseCouplingOperator:
    """Direct GPU pairwise coupling using dtype-specific fused RawKernels.

    The free-space coupling `W` is geometry-only and does not depend on the
    particle single-body representation. The CuPy preparation path is limited
    by the available GPU `T` operator and therefore exposes only diagonal
    particle groups (spheres and layered spheres).
    """

    lmax: int
    k: float
    positions: np.ndarray
    ab5: np.ndarray
    radial_lut: RadialLUT
    dtype: np.dtype = COMPLEX128_DTYPE
    _positions_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_re_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_im_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _plm_coeff_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_re_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_im_gpu: Any | None = field(default=None, init=False, repr=False)
    _mode_m_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_offset_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_pmin_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_pcount_gpu: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy pairwise coupling supports only complex64 and complex128. "
                f"Got {self.dtype!r}."
            )

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def real_dtype(self) -> np.dtype:
        return np.dtype(np.float32 if self.dtype == np.dtype(np.complex64) else np.float64)

    def _raw_kernel_resources(self):
        cupy, _ = import_cupy()
        real_dtype = self.real_dtype
        if self._positions_gpu is None:
            self._positions_gpu = cupy.asarray(self.positions, dtype=real_dtype).reshape(-1)
        if self._compact_re_ab_gpu is None or self._compact_im_ab_gpu is None:
            re_ab, im_ab = _translation_ab5_compact_tables(self.lmax, dtype=self.dtype)
            self._compact_re_ab_gpu = cupy.asarray(re_ab, dtype=real_dtype)
            self._compact_im_ab_gpu = cupy.asarray(im_ab, dtype=real_dtype)
        if self._plm_coeff_gpu is None:
            coeff = _translation_plm_coeff_table(self.lmax, dtype=real_dtype).reshape(-1)
            self._plm_coeff_gpu = cupy.asarray(coeff, dtype=real_dtype)
        if self._lut_re_gpu is None or self._lut_im_gpu is None:
            lut = np.asarray(self.radial_lut.h, dtype=self.dtype).T.copy()
            self._lut_re_gpu = cupy.asarray(
                np.ascontiguousarray(lut.real.reshape(-1)),
                dtype=real_dtype,
            )
            self._lut_im_gpu = cupy.asarray(
                np.ascontiguousarray(lut.imag.reshape(-1)),
                dtype=real_dtype,
            )
        if self._mode_m_gpu is None:
            _, _, mode_m = mode_metadata_tables(self.lmax)
            self._mode_m_gpu = cupy.asarray(mode_m)
        if (
            self._pair_offset_gpu is None
            or self._pair_pmin_gpu is None
            or self._pair_pcount_gpu is None
        ):
            pair_offset, pair_pmin, pair_pcount = mode_pair_p_range_tables(self.lmax)
            self._pair_offset_gpu = cupy.asarray(pair_offset.reshape(-1))
            self._pair_pmin_gpu = cupy.asarray(pair_pmin.reshape(-1))
            self._pair_pcount_gpu = cupy.asarray(pair_pcount.reshape(-1))
        return (
            _translation_matvec_raw_kernel(self.lmax, self.dtype.str),
            self._positions_gpu,
            self._lut_re_gpu,
            self._lut_im_gpu,
            self._plm_coeff_gpu,
            self._compact_re_ab_gpu,
            self._compact_im_ab_gpu,
            self._mode_m_gpu,
            self._pair_offset_gpu,
            self._pair_pmin_gpu,
            self._pair_pcount_gpu,
        )

    def _launch_config(self, *, nrhs: int = 1) -> tuple[int, int, int, int]:
        cupy, _ = import_cupy()
        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
        max_threads = int(props["maxThreadsPerBlock"])
        warp_size = int(props["warpSize"])
        nmodes_total = self.n_modes
        # Keep one block responsible for one destination particle and one tile of
        # output modes. This materially improves occupancy over the earlier
        # one-launch-per-source structure while retaining shared geometry reuse.
        #
        # The heuristic is intentionally simple and hardware-aware:
        # - complex128 keeps one warp per block to limit register pressure,
        # - complex64 uses up to two warps when available,
        # - the tile is capped by both the device limit and the actual mode count.
        #
        # We also tested allowing "helper" threads above the active mode count
        # so extra lanes could participate in the cooperative pair setup.
        # On the 5k-particle c64 benchmark that was slower overall, so we keep
        # the tighter mode-count-capped launch.
        #
        # This remains valid for modestly larger lmax because `blocks_x` grows as
        # needed. For very large N we cap the grid height at the device maximum
        # and let the kernel walk destination particles with a grid-stride loop.
        #
        # Forcing `48` and `64` threads per block in the case of `lmax=4` case
        # (`nmodes=48`) by forcing on the 5k-particle c64 benchmark did not bring
        # any material advantage over the default policy, so we keep the leaner
        # mode-count-capped heuristic.
        target_threads = warp_size if self.dtype == np.dtype(np.complex128) else 2 * warp_size
        threads_per_block = max(warp_size, min(max_threads, target_threads, nmodes_total))
        blocks_x = (nmodes_total + threads_per_block - 1) // threads_per_block
        max_grid_y = int(props["maxGridSize"][1])
        max_grid_z = int(props["maxGridSize"][2])
        grid_y = min(self.n_particles, max_grid_y)
        grid_z = min(max(1, int(nrhs)), max_grid_z)
        return int(blocks_x), int(threads_per_block), int(grid_y), int(grid_z)

    def _apply_gpu(self, x: np.ndarray | object):
        cupy, _ = import_cupy()
        (
            kernel,
            positions_gpu,
            lut_re_gpu,
            lut_im_gpu,
            plm_coeff_gpu,
            compact_re_ab_gpu,
            compact_im_ab_gpu,
            mode_m_gpu,
            pair_offset_gpu,
            pair_pmin_gpu,
            pair_pcount_gpu,
        ) = self._raw_kernel_resources()

        arr_raw = coerce_array(x, dtype=self.dtype, prefer_cupy=True)
        squeezed = False
        if int(arr_raw.ndim) == 1:
            if int(arr_raw.size) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input length must match n_particles * n_modes. "
                    f"Got {int(arr_raw.size)} for {self.n_particles * self.n_modes}."
                )
            arr = cupy.asarray(arr_raw, dtype=self.dtype).reshape(self.n_particles, self.n_modes, 1)
            squeezed = True
        elif int(arr_raw.ndim) == 2:
            if int(arr_raw.shape[0]) != self.n_particles * self.n_modes:
                raise ValueError(
                    "Input first dimension must match n_particles * n_modes. "
                    f"Got {int(arr_raw.shape[0])} for {self.n_particles * self.n_modes}."
                )
            arr = cupy.asarray(arr_raw, dtype=self.dtype).reshape(
                self.n_particles, self.n_modes, int(arr_raw.shape[1])
            )
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(arr_raw.shape)}.")

        arr = cupy.ascontiguousarray(arr)
        nrhs = int(arr.shape[2])
        y = cupy.empty((self.n_particles, self.n_modes, nrhs), dtype=self.dtype)

        blocks_x, threads_per_block, grid_y, grid_z = self._launch_config(nrhs=nrhs)
        inv_dr = self.real_dtype.type(self.radial_lut._inv_dr)
        kernel(
            (blocks_x, grid_y, grid_z),
            (threads_per_block,),
            (
                np.int32(self.n_particles),
                np.int32(self.n_modes),
                np.int32(nrhs),
                positions_gpu,
                lut_re_gpu,
                lut_im_gpu,
                inv_dr,
                np.int32(self.radial_lut._last_index),
                plm_coeff_gpu,
                compact_re_ab_gpu,
                compact_im_ab_gpu,
                mode_m_gpu,
                pair_offset_gpu,
                pair_pmin_gpu,
                pair_pcount_gpu,
                arr.reshape(-1),
                y.reshape(-1),
            ),
        )
        if squeezed:
            return y.reshape(self.n_particles * self.n_modes)
        return y.reshape(self.n_particles * self.n_modes, nrhs)

    def apply(self, x: np.ndarray | object) -> np.ndarray | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPairwiseCouplingOperator"]
