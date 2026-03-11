from __future__ import annotations

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

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
)


@cache
def _translation_matvec_raw_kernel(lmax: int, dtype_str: str):
    cupy, _ = import_cupy()
    lmax = int(lmax)
    dtype = np.dtype(dtype_str)
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported CuPy raw-kernel dtype {dtype!r}.")

    real_type = "float" if dtype == np.dtype(np.complex64) else "double"
    math = {
        "atan2": "atan2f" if real_type == "float" else "atan2",
        "cos": "cosf" if real_type == "float" else "cos",
        "floor": "floorf" if real_type == "float" else "floor",
        "max": "fmaxf" if real_type == "float" else "fmax",
        "pow": "powf" if real_type == "float" else "pow",
        "sin": "sinf" if real_type == "float" else "sin",
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

    extern "C" __device__ {real_type} assoc_legendre_function(
        const int l,
        const int m,
        const {real_type} ct,
        const {real_type} st,
        const {real_type}* plm_coeffs
    ) {{
        {real_type} plm = 0;
        {real_type} st_pow = (m == 0) ? ({real_type})1 : {math["pow"]}(st, ({real_type})m);
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * {math["pow"]}(ct, ({real_type})lambda) * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    extern "C" __device__ {real_type} hankel_lookup_linear(
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
        const {real_type}* positions,
        const {real_type}* re_h,
        const {real_type}* im_h,
        const {real_type} inv_dr,
        const int last_index,
        const {real_type}* plm_coeffs,
        const {real_type}* re_ab,
        const {real_type}* im_ab,
        const int* mode_tau,
        const int* mode_l,
        const int* mode_m,
        const int* pair_offset,
        const int* pair_pmin,
        const int* pair_pcount,
        const {real_type}* re_x,
        const {real_type}* im_x,
        {real_type}* re_wx,
        {real_type}* im_wx
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

        const int tau1 = mode_tau[n1];
        const int l1 = mode_l[n1];
        const int m1 = mode_m[n1];

        for (int s1 = blockIdx.y; s1 < ns; s1 += gridDim.y) {{
            {real_type} re_incr = 0;
            {real_type} im_incr = 0;

            for (int s2 = 0; s2 < ns; ++s2) {{
                if (s2 == s1) {{
                    continue;
                }}

                if (threadIdx.x == 0) {{
                    // Geometry and angular factors depend only on the particle pair
                    // (s1, s2), not on the output mode. Compute them once per block
                    // and let all mode threads reuse them.
                    const {real_type} x21 = positions[3 * s1] - positions[3 * s2];
                    const {real_type} y21 = positions[3 * s1 + 1] - positions[3 * s2 + 1];
                    const {real_type} z21 = positions[3 * s1 + 2] - positions[3 * s2 + 2];
                    const {real_type} r = {math["sqrt"]}(x21 * x21 + y21 * y21 + z21 * z21);
                    const {real_type} ct = z21 / r;
                    const {real_type} st = {math["sqrt"]}({math["max"]}(({real_type})0, ({real_type})1 - ct * ct));
                    const {real_type} phi = {math["atan2"]}(y21, x21);

                    for (int p = 0; p < {n_orders}; ++p) {{
                        re_h_shared[p] = hankel_lookup_linear(p, r, re_h, inv_dr, last_index);
                        im_h_shared[p] = hankel_lookup_linear(p, r, im_h, inv_dr, last_index);
                        for (int absdm = 0; absdm <= p; ++absdm) {{
                            p_pdm_shared[p * (p + 1) / 2 + absdm] =
                                assoc_legendre_function(p, absdm, ct, st, plm_coeffs);
                        }}
                    }}

                    for (int dm = -2 * {lmax}; dm <= 2 * {lmax}; ++dm) {{
                        const int idx = dm + 2 * {lmax};
                        cos_mphi_shared[idx] = {math["cos"]}(({real_type})dm * phi);
                        sin_mphi_shared[idx] = {math["sin"]}(({real_type})dm * phi);
                    }}
                }}
                __syncthreads();

                for (int n2 = 0; n2 < nmodes; ++n2) {{
                    const {real_type} re_x_tmp = re_x[s2 * nmodes + n2];
                    const {real_type} im_x_tmp = im_x[s2 * nmodes + n2];
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

            re_wx[s1 * nmodes + n1] = re_incr;
            im_wx[s1 * nmodes + n1] = im_incr;
        }}
    }}
    """
    return cupy.RawKernel(source, kernel_name)


@cache
def _mode_metadata_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CELES/SMUTHI mode metadata arrays indexed by flattened mode id.

    The raw kernel works with the canonical flattened mode index `n`. These
    small lookup arrays recover `(tau, l, m)` without rebuilding the indexing
    logic on device.
    """
    tau = np.zeros((n_modes(lmax),), dtype=np.int32)
    ell = np.zeros_like(tau)
    m = np.zeros_like(tau)
    for tau_i, l_i, m_i, idx in iter_modes(lmax):
        tau[idx] = tau_i
        ell[idx] = l_i
        m[idx] = m_i
    tau.setflags(write=False)
    ell.setflags(write=False)
    m.setflags(write=False)
    return tau, ell, m


@cache
def _mode_pair_tables(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact p-range metadata for every `(n1, n2)` mode pair.

    For each destination/source mode pair we store:
    - the starting offset into the flattened `ab5` tables,
    - the first admissible translation order `p_min`,
    - the number of consecutive `p` values to visit.

    This encodes the CELES/SMUTHI triangular coupling rules on CPU once so the
    GPU kernel can walk a tight contiguous interval in its hot loop.
    """
    nmodes_total = n_modes(lmax)
    tau, ell, m = _mode_metadata_tables(lmax)
    pair_offset = np.zeros((nmodes_total, nmodes_total), dtype=np.int32)
    pair_pmin = np.zeros_like(pair_offset)
    pair_pcount = np.zeros_like(pair_offset)
    offset = 0
    for n1 in range(nmodes_total):
        for n2 in range(nmodes_total):
            p_min = max(
                abs(int(m[n1]) - int(m[n2])),
                abs(int(ell[n1]) - int(ell[n2])) + abs(int(tau[n1]) - int(tau[n2])),
            )
            p_count = int(ell[n1]) + int(ell[n2]) - p_min + 1
            pair_offset[n1, n2] = offset
            pair_pmin[n1, n2] = p_min
            pair_pcount[n1, n2] = p_count
            offset += p_count
    pair_offset.setflags(write=False)
    pair_pmin.setflags(write=False)
    pair_pcount.setflags(write=False)
    return pair_offset, pair_pmin, pair_pcount


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
    dtype: np.dtype = np.dtype(np.complex128)
    _positions_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_re_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_im_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _plm_coeff_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_re_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_im_gpu: Any | None = field(default=None, init=False, repr=False)
    _mode_tau_gpu: Any | None = field(default=None, init=False, repr=False)
    _mode_l_gpu: Any | None = field(default=None, init=False, repr=False)
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
        if self._mode_tau_gpu is None or self._mode_l_gpu is None or self._mode_m_gpu is None:
            mode_tau, mode_l, mode_m = _mode_metadata_tables(self.lmax)
            self._mode_tau_gpu = cupy.asarray(mode_tau)
            self._mode_l_gpu = cupy.asarray(mode_l)
            self._mode_m_gpu = cupy.asarray(mode_m)
        if (
            self._pair_offset_gpu is None
            or self._pair_pmin_gpu is None
            or self._pair_pcount_gpu is None
        ):
            pair_offset, pair_pmin, pair_pcount = _mode_pair_tables(self.lmax)
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
            self._mode_tau_gpu,
            self._mode_l_gpu,
            self._mode_m_gpu,
            self._pair_offset_gpu,
            self._pair_pmin_gpu,
            self._pair_pcount_gpu,
        )

    def _launch_config(self) -> tuple[int, int, int]:
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
        # This remains valid for modestly larger lmax because `blocks_x` grows as
        # needed. For very large N we cap the grid height at the device maximum
        # and let the kernel walk destination particles with a grid-stride loop.
        if self.dtype == np.dtype(np.complex128):
            target_threads = warp_size
        else:
            target_threads = 2 * warp_size
        threads_per_block = max(warp_size, min(max_threads, target_threads, nmodes_total))
        blocks_x = (nmodes_total + threads_per_block - 1) // threads_per_block
        max_grid_y = int(props["maxGridSize"][1])
        grid_y = min(self.n_particles, max_grid_y)
        return int(blocks_x), int(threads_per_block), int(grid_y)

    def _apply_gpu(self, x: np.ndarray | object):
        cupy, _ = import_cupy()
        real_dtype = cupy.float32 if self.dtype == np.dtype(np.complex64) else cupy.float64
        (
            kernel,
            positions_gpu,
            lut_re_gpu,
            lut_im_gpu,
            plm_coeff_gpu,
            compact_re_ab_gpu,
            compact_im_ab_gpu,
            mode_tau_gpu,
            mode_l_gpu,
            mode_m_gpu,
            pair_offset_gpu,
            pair_pmin_gpu,
            pair_pcount_gpu,
        ) = self._raw_kernel_resources()

        arr = coerce_array(x, dtype=self.dtype, prefer_cupy=True).reshape(
            self.n_particles, self.n_modes
        )
        x_re = cupy.ascontiguousarray(arr.real.reshape(-1).astype(real_dtype, copy=False))
        x_im = cupy.ascontiguousarray(arr.imag.reshape(-1).astype(real_dtype, copy=False))
        y_re = cupy.zeros((self.n_particles * self.n_modes,), dtype=real_dtype)
        y_im = cupy.zeros((self.n_particles * self.n_modes,), dtype=real_dtype)

        blocks_x, threads_per_block, grid_y = self._launch_config()
        inv_dr = self.real_dtype.type(self.radial_lut._inv_dr)
        kernel(
            (blocks_x, grid_y),
            (threads_per_block,),
            (
                np.int32(self.n_particles),
                np.int32(self.n_modes),
                positions_gpu,
                lut_re_gpu,
                lut_im_gpu,
                inv_dr,
                np.int32(self.radial_lut._last_index),
                plm_coeff_gpu,
                compact_re_ab_gpu,
                compact_im_ab_gpu,
                mode_tau_gpu,
                mode_l_gpu,
                mode_m_gpu,
                pair_offset_gpu,
                pair_pmin_gpu,
                pair_pcount_gpu,
                x_re,
                x_im,
                y_re,
                y_im,
            ),
        )
        out = (y_re + 1j * y_im).astype(self.dtype, copy=False)
        return out.reshape(self.n_particles * self.n_modes)

    def apply(self, x: np.ndarray | object) -> np.ndarray | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPairwiseCouplingOperator"]
