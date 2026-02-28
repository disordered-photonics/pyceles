from __future__ import annotations

import numpy as np
import numpy.typing as npt

_ALLOWED_COMPLEX_DTYPE_NAMES = {"complex64", "complex128"}


def as_supported_complex_dtype(value: npt.DTypeLike, *, name: str) -> np.dtype:
    """Normalize one dtype-like input and enforce pyceles complex dtype policy."""
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{name}` must be one of {sorted(_ALLOWED_COMPLEX_DTYPE_NAMES)}. Got {value!r}."
        ) from exc
    if dtype.name not in _ALLOWED_COMPLEX_DTYPE_NAMES:
        raise ValueError(
            f"`{name}` must be one of {sorted(_ALLOWED_COMPLEX_DTYPE_NAMES)}. Got {value!r}."
        )
    return dtype


def resolve_compute_accum_dtypes(
    *, compute_dtype: npt.DTypeLike, accum_dtype: npt.DTypeLike
) -> tuple[np.dtype, np.dtype]:
    """Return validated `(compute_dtype, accum_dtype)` as canonical NumPy dtypes."""
    compute = as_supported_complex_dtype(compute_dtype, name="compute_dtype")
    accum = as_supported_complex_dtype(accum_dtype, name="accum_dtype")
    if accum.itemsize < compute.itemsize:
        raise ValueError(
            f"`accum_dtype` ({accum.name}) must be at least as precise as `compute_dtype` ({compute.name})."
        )
    return compute, accum
