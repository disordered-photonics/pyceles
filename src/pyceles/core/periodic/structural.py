"""Structural lattice sums for periodic SVWF coupling."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.spherical import legendre_normalized_trigon_scalar
from pyceles.core.translation import spherical_bessel_jy

Array = np.ndarray


def _validated_window(window: int) -> int:
    value = int(window)
    if value < 0:
        raise ValueError(f"`directsum_window` must be >= 0. Got {window!r}.")
    return value


def direct_structural_sums_2d(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    window: int,
    exclude_zero_shift: bool = False,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Evaluate finite-window scalar lattice sums for SVWF block assembly.

    The returned table stores outgoing spherical Hankel angular sums
    `h_l^(1)(k r) P_l^|m|(cos theta) exp(i m phi)` over Bloch-shifted
    lattice images. It is the direct, validation-oriented counterpart of the
    accelerated structural constants used by a periodic Ewald coupling.
    """
    out_dtype = np.dtype(dtype)
    order = 2 * int(lmax)
    m_offset = order
    sums = np.zeros((order + 1, 2 * order + 1), dtype=np.complex128)
    m_values = np.arange(-order, order + 1, dtype=np.int32)

    k_f = float(k)
    dst = np.asarray(destination, dtype=float).reshape(3)
    src = np.asarray(source, dtype=float).reshape(3)
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    nwin = _validated_window(window)

    for p in range(-nwin, nwin + 1):
        for q in range(-nwin, nwin + 1):
            if exclude_zero_shift and p == 0 and q == 0:
                continue
            shift = lattice.lattice_vector(p, q)
            rvec = dst - src - shift
            r = float(np.linalg.norm(rvec))
            if r == 0.0:
                continue

            ct = float(rvec[2] / r)
            st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            phi = float(np.arctan2(rvec[1], rvec[0]))
            j, y = spherical_bessel_jy(order, np.asarray(k_f * r, dtype=np.complex128))
            hankel = np.asarray(j + 1j * y, dtype=np.complex128).reshape(order + 1)
            plm = legendre_normalized_trigon_scalar(ct, st, order)

            bloch_phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
            azimuthal_phase = np.exp(1j * phi * m_values)
            for degree in range(order + 1):
                m_slice = slice(m_offset - degree, m_offset + degree + 1)
                abs_m = np.abs(m_values[m_slice])
                sums[degree, m_slice] += (
                    bloch_phase * hankel[degree] * plm[degree, abs_m] * azimuthal_phase[m_slice]
                )

    return np.asarray(sums, dtype=out_dtype)


def block_from_structural_sums(
    *,
    lmax: int,
    structural_sums: Array,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Assemble one SVWF translation block from scalar structural sums.

    This isolates the CELES/SMUTHI `ab5` contraction from how the scalar
    periodic sums are evaluated, so the same block assembly can be checked
    against finite direct sums before an Ewald evaluator supplies the table.
    """
    out_dtype = np.dtype(dtype)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    order = 2 * lmax_i
    expected_sums_shape = (order + 1, 2 * order + 1)
    sums = np.asarray(structural_sums)
    if sums.shape != expected_sums_shape:
        raise ValueError(
            f"`structural_sums` must have shape {expected_sums_shape}. Got {sums.shape}."
        )
    ab5_arr = np.asarray(ab5)
    expected_ab5_shape = (nm, nm, order + 1)
    if ab5_arr.shape != expected_ab5_shape:
        raise ValueError(f"`ab5` must have shape {expected_ab5_shape}. Got {ab5_arr.shape}.")

    block_dtype = np.result_type(ab5_arr.dtype, sums.dtype, np.complex64)
    block = np.zeros((nm, nm), dtype=block_dtype)
    m_offset = order
    for _tau_dst, _l_dst, m_dst, dst_idx in iter_modes(lmax_i):
        for _tau_src, _l_src, m_src, src_idx in iter_modes(lmax_i):
            m_delta = m_src - m_dst
            block[dst_idx, src_idx] = np.sum(
                ab5_arr[dst_idx, src_idx, :] * sums[:, m_delta + m_offset]
            )
    return np.asarray(block, dtype=out_dtype)


def periodic_direct_structural_block(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    window: int,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    exclude_zero_shift: bool = False,
) -> Array:
    """Assemble a finite-window Bloch block through scalar structural sums."""
    sums = direct_structural_sums_2d(
        lmax=int(lmax),
        k=float(k),
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        window=int(window),
        exclude_zero_shift=bool(exclude_zero_shift),
        dtype=np.complex128,
    )
    return block_from_structural_sums(
        lmax=int(lmax),
        structural_sums=sums,
        ab5=ab5,
        dtype=dtype,
    )


__all__ = [
    "block_from_structural_sums",
    "direct_structural_sums_2d",
    "periodic_direct_structural_block",
]
