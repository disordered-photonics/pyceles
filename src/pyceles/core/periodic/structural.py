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


def translation_contraction_tensor(
    *,
    lmax: int,
    ab5: Array,
    dtype: npt.DTypeLike | None = None,
) -> Array:
    """Return the dense tensor that contracts scalar structural sums into W blocks.

    For a fixed ``lmax`` and CELES/SMUTHI ``ab5`` table, periodic Ewald block
    assembly repeatedly performs the same sparse selection/contraction:

    ``block[dst, src] = sum_p ab5[dst, src, p] * S[p, m_src - m_dst]``.

    Building this tensor once removes Python-level mode-pair loops from dense
    block-cache population and from the matrix-free batched ``W @ x`` path.
    The tensor is small for the low orders used by typycal pycels examples.
    """
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    order = 2 * lmax_i
    ab5_arr = np.asarray(ab5)
    expected_ab5_shape = (nm, nm, order + 1)
    if ab5_arr.shape != expected_ab5_shape:
        raise ValueError(f"`ab5` must have shape {expected_ab5_shape}. Got {ab5_arr.shape}.")
    out_dtype = (
        np.dtype(dtype) if dtype is not None else np.result_type(ab5_arr.dtype, np.complex64)
    )
    tensor = np.zeros((nm, nm, order + 1, 2 * order + 1), dtype=out_dtype)
    m_offset = order
    modes = tuple(iter_modes(lmax_i))
    for _tau_dst, _l_dst, m_dst, dst_idx in modes:
        for _tau_src, _l_src, m_src, src_idx in modes:
            tensor[dst_idx, src_idx, :, m_src - m_dst + m_offset] = ab5_arr[dst_idx, src_idx, :]
    return tensor


def blocks_from_structural_sums(
    *,
    lmax: int,
    structural_sums: Array,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    contraction_tensor: Array | None = None,
) -> Array:
    """Assemble one or more SVWF translation blocks from scalar structural sums.

    ``structural_sums`` can be either one table with shape ``(P, M)`` or a batch
    with shape ``(n_destinations, P, M)``. The returned array has shape
    ``(n_destinations, n_modes, n_modes)`` for batched input and
    ``(n_modes, n_modes)`` for single-table input.
    """
    out_dtype = np.dtype(dtype)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    order = 2 * lmax_i
    expected_sums_shape = (order + 1, 2 * order + 1)
    sums = np.asarray(structural_sums)
    single = False
    if sums.shape == expected_sums_shape:
        sums = sums[None, :, :]
        single = True
    if sums.ndim != 3 or sums.shape[1:] != expected_sums_shape:
        raise ValueError(
            "`structural_sums` must have shape "
            f"{expected_sums_shape} or (n_destinations, {expected_sums_shape[0]}, {expected_sums_shape[1]}). "
            f"Got {np.asarray(structural_sums).shape}."
        )
    tensor = (
        translation_contraction_tensor(lmax=lmax_i, ab5=ab5)
        if contraction_tensor is None
        else np.asarray(contraction_tensor)
    )
    expected_tensor_shape = (nm, nm, order + 1, 2 * order + 1)
    if tensor.shape != expected_tensor_shape:
        raise ValueError(
            f"`contraction_tensor` must have shape {expected_tensor_shape}. Got {tensor.shape}."
        )
    block_dtype = np.result_type(sums.dtype, tensor.dtype, np.complex64)
    blocks = np.einsum(
        "dpm,ijpm->dij",
        np.asarray(sums, dtype=block_dtype),
        np.asarray(tensor, dtype=block_dtype),
        optimize=True,
    )
    blocks = np.asarray(blocks, dtype=out_dtype)
    return blocks[0] if single else blocks


def block_from_structural_sums(
    *,
    lmax: int,
    structural_sums: Array,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    contraction_tensor: Array | None = None,
) -> Array:
    """Assemble one SVWF translation block from scalar structural sums."""
    return np.asarray(
        blocks_from_structural_sums(
            lmax=int(lmax),
            structural_sums=structural_sums,
            ab5=ab5,
            dtype=dtype,
            contraction_tensor=contraction_tensor,
        ),
        dtype=np.dtype(dtype),
    )


def apply_structural_sums_to_vector(
    *,
    lmax: int,
    structural_sums: Array,
    ab5: Array,
    vector: Array,
    dtype: npt.DTypeLike = np.complex128,
    contraction_tensor: Array | None = None,
) -> Array:
    """Apply batched scalar structural tables to one source coefficient vector.

    This is the matrix-free counterpart of ``blocks_from_structural_sums``. It
    uses the same precomputed contraction tensor when available, but contracts
    directly with the source vector to avoid materializing one block per
    destination.
    """
    out_dtype = np.dtype(dtype)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    order = 2 * lmax_i
    sums = np.asarray(structural_sums)
    expected_sums_shape = (order + 1, 2 * order + 1)
    if sums.ndim != 3 or sums.shape[1:] != expected_sums_shape:
        raise ValueError(
            "`structural_sums` must have shape "
            f"(n_destinations, {expected_sums_shape[0]}, {expected_sums_shape[1]}). "
            f"Got {sums.shape}."
        )
    tensor = (
        translation_contraction_tensor(lmax=lmax_i, ab5=ab5)
        if contraction_tensor is None
        else np.asarray(contraction_tensor)
    )
    expected_tensor_shape = (nm, nm, order + 1, 2 * order + 1)
    if tensor.shape != expected_tensor_shape:
        raise ValueError(
            f"`contraction_tensor` must have shape {expected_tensor_shape}. Got {tensor.shape}."
        )
    vec = np.asarray(vector, dtype=np.result_type(out_dtype, np.complex64)).reshape(nm)
    result_dtype = np.result_type(sums.dtype, tensor.dtype, vec.dtype, np.complex64)
    result = np.einsum(
        "dpm,ijpm,j->di",
        np.asarray(sums, dtype=result_dtype),
        np.asarray(tensor, dtype=result_dtype),
        vec,
        optimize=True,
    )
    return np.asarray(result, dtype=out_dtype)


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
    "apply_structural_sums_to_vector",
    "block_from_structural_sums",
    "blocks_from_structural_sums",
    "direct_structural_sums_2d",
    "periodic_direct_structural_block",
    "translation_contraction_tensor",
]
