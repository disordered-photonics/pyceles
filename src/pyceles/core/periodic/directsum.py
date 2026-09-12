"""Direct periodic coupling sums for validation-oriented CPU runs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.translation import translation_block

Array = np.ndarray


def _validated_window(window: int) -> int:
    value = int(window)
    if value < 0:
        raise ValueError(f"`directsum_window` must be >= 0. Got {window!r}.")
    return value


def periodic_direct_sum_block(
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
    """Sum free-space SVWF translation blocks over a finite image window.

    The returned block maps outgoing coefficients at `source` to regular
    coefficients at `destination` under the Bloch-reduced image sum. The
    singular reference-cell self term is excluded only when requested by the
    caller.
    """
    out_dtype = np.dtype(dtype)
    nm = n_modes(int(lmax))
    w = np.zeros((nm, nm), dtype=out_dtype)
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
            phase = np.exp(1j * float(np.dot(kp, shift[:2])))
            block = translation_block(
                int(lmax),
                float(k),
                rvec,
                ab5=ab5,
                radial_lut=None,
            )
            w += np.asarray(phase * block, dtype=out_dtype)
    return w


def apply_periodic_direct_sum(
    *,
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    window: int,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Apply the finite-window Bloch image sum to stacked SVWF coefficients."""
    out_dtype = np.dtype(dtype)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ns = pos.shape[0]
    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=out_dtype).reshape(ns, nm)
    y = np.zeros_like(arr, dtype=out_dtype)

    for i in range(ns):
        for j in range(ns):
            wij = periodic_direct_sum_block(
                lmax=int(lmax),
                k=float(k),
                destination=pos[i],
                source=pos[j],
                lattice=lattice,
                k_parallel=k_parallel,
                window=window,
                ab5=ab5,
                dtype=out_dtype,
                exclude_zero_shift=(i == j),
            )
            y[i] += wij @ arr[j]
    return y.reshape(ns * nm)


def apply_periodic_direct_sum_adjoint(
    *,
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    window: int,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Apply the exact Hermitian adjoint of the finite-window image sum."""
    out_dtype = np.dtype(dtype)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ns = pos.shape[0]
    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=out_dtype).reshape(ns, nm)
    arr_conj = np.conjugate(arr)
    y = np.zeros_like(arr, dtype=out_dtype)
    for i in range(ns):
        for j in range(ns):
            wij = periodic_direct_sum_block(
                lmax=int(lmax),
                k=float(k),
                destination=pos[i],
                source=pos[j],
                lattice=lattice,
                k_parallel=k_parallel,
                window=window,
                ab5=ab5,
                dtype=out_dtype,
                exclude_zero_shift=(i == j),
            )
            y[j] += np.conjugate(wij.T @ arr_conj[i])
    return y.reshape(ns * nm)


__all__ = [
    "apply_periodic_direct_sum",
    "apply_periodic_direct_sum_adjoint",
    "periodic_direct_sum_block",
]
