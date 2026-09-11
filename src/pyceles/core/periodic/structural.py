"""Structural lattice sums for periodic SVWF coupling."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.spherical import legendre_normalized_trigon_scalar
from pyceles.core.translation import spherical_bessel_jy

Array = np.ndarray


def free_space_structural_sums(
    *,
    max_degree: int,
    k: float,
    displacement: Array,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Return scalar outgoing-wave structural constants for one displacement.

    This is the nonperiodic primitive shared by direct lattice validation and
    periodic Ewald preparation.  The table follows the same pyceles
    normalization and centered-order layout as the periodic Ewald routines,
    but its maximum spherical degree is explicit rather than inferred from a
    particle ``lmax``.
    """

    degree_max = int(max_degree)
    if degree_max < 0:
        raise ValueError(f"`max_degree` must be >= 0. Got {max_degree!r}.")
    out_dtype = np.dtype(dtype)
    rvec = np.asarray(displacement, dtype=float).reshape(3)
    radius = float(np.linalg.norm(rvec))
    if radius == 0.0:
        raise ValueError("A free-space structural sum is singular at zero displacement.")

    offset = degree_max
    ct = float(rvec[2] / radius)
    st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
    phi = float(np.arctan2(rvec[1], rvec[0]))
    j, y = spherical_bessel_jy(
        degree_max,
        np.asarray(float(k) * radius, dtype=np.complex128),
    )
    hankel = np.asarray(j + 1j * y, dtype=np.complex128).reshape(degree_max + 1)
    plm = legendre_normalized_trigon_scalar(ct, st, degree_max)
    orders = np.arange(-degree_max, degree_max + 1, dtype=np.int32)
    azimuthal_phase = np.exp(1j * phi * orders)
    sums = np.zeros((degree_max + 1, 2 * degree_max + 1), dtype=np.complex128)
    for degree in range(degree_max + 1):
        order_slice = slice(offset - degree, offset + degree + 1)
        abs_orders = np.abs(orders[order_slice])
        sums[degree, order_slice] = (
            hankel[degree] * plm[degree, abs_orders] * azimuthal_phase[order_slice]
        )
    return np.asarray(sums, dtype=out_dtype)


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
    sums = np.zeros((order + 1, 2 * order + 1), dtype=np.complex128)
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

            bloch_phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
            sums += bloch_phase * free_space_structural_sums(
                max_degree=order,
                k=k_f,
                displacement=rvec,
                dtype=np.complex128,
            )

    return np.asarray(sums, dtype=out_dtype)


def sparse_translation_contraction(
    *,
    lmax: int,
    ab5: Array,
    dtype: npt.DTypeLike | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return output-mode CSR data for the SVWF structural contraction.

    The compact periodic structural cache stores valid ``(p, m)`` channels in
    degree-major order.  For a mode pair only ``m = m_src - m_dst`` contributes,
    so the compact channel index is ``p**2 + p + m``.  Constructing this sparse
    map directly avoids ever materializing the dense ``(Nm, Nm, P, M)`` tensor
    merely to discard its structural zeros.
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
    modes = tuple(iter_modes(lmax_i))
    row_ptr = np.zeros((nm + 1,), dtype=np.int64)
    input_modes: list[int] = []
    structural_channels: list[int] = []
    values: list[complex] = []
    for _tau_dst, _l_dst, m_dst, dst_idx in modes:
        for _tau_src, _l_src, m_src, src_idx in modes:
            m_delta = int(m_src - m_dst)
            for degree in range(abs(m_delta), order + 1):
                value = ab5_arr[dst_idx, src_idx, degree]
                if value == 0:
                    continue
                input_modes.append(int(src_idx))
                structural_channels.append(degree * degree + degree + m_delta)
                values.append(complex(value))
        row_ptr[int(dst_idx) + 1] = len(values)
    return (
        row_ptr,
        np.asarray(input_modes, dtype=np.int32),
        np.asarray(structural_channels, dtype=np.int32),
        np.asarray(values, dtype=out_dtype),
    )


def transpose_sparse_translation_contraction(
    row_ptr: Array,
    input_modes: Array,
    structural_channels: Array,
    values: Array,
) -> tuple[Array, Array, Array, Array]:
    """Transpose output-mode CSR metadata without assembling dense blocks.

    ``sparse_translation_contraction`` groups entries by output mode.  The
    exact-near adjoint needs the same entries grouped by the original input
    mode.  Structural sums and coefficients are conjugated by the device
    kernel at application time, so the returned values preserve their stored
    dtype and are merely reordered here.
    """

    pointers = np.asarray(row_ptr, dtype=np.int64).reshape(-1)
    inputs = np.asarray(input_modes, dtype=np.int32).reshape(-1)
    channels = np.asarray(structural_channels, dtype=np.int32).reshape(-1)
    factors = np.asarray(values).reshape(-1)
    if pointers.size < 1:
        raise ValueError("Sparse contraction row pointers must be non-empty.")
    n_modes_i = int(pointers.size - 1)
    if not (inputs.size == channels.size == factors.size == int(pointers[-1])):
        raise ValueError("Sparse contraction metadata arrays have inconsistent lengths.")
    if inputs.size and (int(inputs.min()) < 0 or int(inputs.max()) >= n_modes_i):
        raise ValueError("Sparse contraction input-mode index is out of range.")

    output_modes = np.repeat(
        np.arange(n_modes_i, dtype=np.int32),
        np.diff(pointers),
    )
    order = np.argsort(inputs, kind="stable")
    transposed_rows = inputs[order]
    counts = np.bincount(transposed_rows, minlength=n_modes_i).astype(np.int64, copy=False)
    transposed_ptr = np.empty((n_modes_i + 1,), dtype=np.int64)
    transposed_ptr[0] = 0
    np.cumsum(counts, out=transposed_ptr[1:])
    return (
        transposed_ptr,
        output_modes[order],
        channels[order],
        factors[order],
    )


def translation_contraction_tensor(
    *,
    lmax: int,
    ab5: Array,
    dtype: npt.DTypeLike | None = None,
) -> Array:
    """Return the dense tensor that contracts scalar structural sums into W blocks.

    For a fixed ``lmax`` and translation ``ab5`` table, periodic Ewald block
    assembly repeatedly performs the same sparse selection/contraction:

    ``block[dst, src] = sum_p ab5[dst, src, p] * S[p, m_src - m_dst]``.

    Building this tensor once removes Python-level mode-pair loops from dense
    block-cache population and from the matrix-free batched ``W @ x`` path.
    The tensor is small for the low orders used by typical pyceles examples.
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
    ``(n_modes, n_modes)`` for single-table input. ``dtype`` controls the
    contraction and returned blocks even when the supplied structural table is
    wider, as required by the periodic c64/c128 precision policy.
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
    # Structural Ewald tables may be retained in c128 to protect their
    # cancellation-heavy lattice sum, while the requested operator dtype still
    # controls the actual translation contraction.  This mirrors the finite
    # pairwise path: c64/c128 therefore performs the dense contraction in c64
    # without weakening the scalar Ewald reference arithmetic.
    block_dtype = np.result_type(out_dtype, np.complex64)
    tensor = (
        translation_contraction_tensor(lmax=lmax_i, ab5=ab5, dtype=block_dtype)
        if contraction_tensor is None
        else np.asarray(contraction_tensor)
    )
    expected_tensor_shape = (nm, nm, order + 1, 2 * order + 1)
    if tensor.shape != expected_tensor_shape:
        raise ValueError(
            f"`contraction_tensor` must have shape {expected_tensor_shape}. Got {tensor.shape}."
        )
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
    result_dtype = np.result_type(out_dtype, np.complex64)
    tensor = (
        translation_contraction_tensor(lmax=lmax_i, ab5=ab5, dtype=result_dtype)
        if contraction_tensor is None
        else np.asarray(contraction_tensor)
    )
    expected_tensor_shape = (nm, nm, order + 1, 2 * order + 1)
    if tensor.shape != expected_tensor_shape:
        raise ValueError(
            f"`contraction_tensor` must have shape {expected_tensor_shape}. Got {tensor.shape}."
        )
    vec = np.asarray(vector, dtype=result_dtype).reshape(nm)
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
    "free_space_structural_sums",
    "periodic_direct_structural_block",
    "sparse_translation_contraction",
    "translation_contraction_tensor",
    "transpose_sparse_translation_contraction",
]
