"""Spherical outgoing radial functions for the exp(-i*omega*t) convention."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import hankel1


def sph_hankel1(l: int | np.ndarray, z: npt.ArrayLike) -> np.ndarray:
    """Evaluate spherical h_l^(1)(z) in double precision.

    Half-integer cylindrical Hankel functions avoid cancellation between the
    exponentially growing j_l and i*y_l terms for attenuating arguments.
    Outgoing waves are singular at zero; callers must exclude their centres.
    """
    argument = np.asarray(z, dtype=np.complex128)
    return np.asarray(np.sqrt(np.pi / (2 * argument)) * hankel1(np.asarray(l) + 0.5, argument))


def dx_xz_hankel1(l: int | np.ndarray, z: npt.ArrayLike) -> np.ndarray:
    """Evaluate d[z*h_l^(1)(z)]/dz, including order zero, in double precision."""
    argument = np.asarray(z, dtype=np.complex128)
    order = np.asarray(l)
    scale = np.sqrt(np.pi / (2 * argument))
    return np.asarray(
        scale * (argument * hankel1(order - 0.5, argument) - order * hankel1(order + 0.5, argument))
    )


def passive_wavenumber(k: complex) -> complex:
    """Require a finite wavenumber with positive real and nonnegative imaginary parts."""
    value = complex(k)
    if not np.isfinite(value) or value.real <= 0 or value.imag < 0:
        raise ValueError("k must be finite with positive real and nonnegative imaginary parts")
    return value


def _cast_radial_table_with_finite_guard(
    values: np.ndarray, dtype: npt.DTypeLike
) -> tuple[np.ndarray, np.ndarray]:
    """Store finite outgoing rows and identify their first physical samples.

    Each order has its own representable radius. Repeat its first finite value
    across the singular prefix without flattening valid lower-order samples.
    These copies are padding; index zero is always outside the physical domain.
    Nonfinite values beyond a leading prefix are a construction failure.
    """
    source = np.asarray(values)
    target = np.dtype(dtype)
    if source.ndim == 0:
        raise ValueError("Radial samples require a radius axis")
    if target not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("Radial tables require complex64 or complex128 storage")
    real_dtype = np.empty((), dtype=target).real.dtype
    max_finite = np.finfo(real_dtype).max
    rows = source.reshape(-1, source.shape[-1])
    stored = np.empty(rows.shape, dtype=target)
    first_indices = np.empty(len(rows), dtype=np.int64)
    for index, (row, out) in enumerate(zip(rows, stored, strict=True)):
        representable = (
            np.isfinite(row.real)
            & np.isfinite(row.imag)
            & (np.abs(row.real) <= max_finite)
            & (np.abs(row.imag) <= max_finite)
        )
        finite = np.flatnonzero(representable)
        if not finite.size:
            raise FloatingPointError(
                f"Radial table for dtype {target} has no finite sample in the requested range."
            )
        first = int(finite[0])
        if not np.all(representable[first:]):
            raise FloatingPointError(
                "Radial table contains non-finite samples beyond its guard prefix."
            )
        out[first:] = row[first:].astype(target, copy=False)
        out[:first] = out[first]
        first_indices[index] = max(1, first)
    return stored.reshape(source.shape), first_indices.reshape(source.shape[:-1])
