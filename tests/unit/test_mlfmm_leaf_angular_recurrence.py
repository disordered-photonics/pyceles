"""Normalized angular recurrences used by rectangular and fused leaf kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import gammaln, lpmv

from pyceles.core.operators import mlfmm_cupy as kernels
from pyceles.core.spherical import legendre_normalized_trigon_scalar


@pytest.mark.gpu
@pytest.mark.parametrize("degree", [6, 25, 42, 78, 102])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_leaf_legendre_recurrence_on_device(
    cupy_runtime: tuple[Any, Any], degree: int, dtype: Any
) -> None:
    cupy, _ = cupy_runtime
    real = "float" if dtype == np.float32 else "double"
    source = (
        kernels._LEAF_LEGENDRE_COLUMN_SOURCE
        + f"""
    extern "C" __global__ void probe(const int degree, const {real} ct,
        const {real} st, const {real}* coefficients, {real}* values) {{
        const int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m <= degree) leaf_legendre_column<{real}>(m, degree, ct, st, coefficients, values);
    }}
    """
    )
    kernel = cupy.RawKernel(source, "probe")
    coefficients = cupy.asarray(kernels._leaf_legendre_coefficients(degree, dtype))
    l, m = np.tril_indices(degree + 1)
    for cosine in [-1.0, -0.999, -0.95, -0.3, 0.0, 0.3, 0.95, 0.999, 1.0]:
        ct = dtype(cosine)
        st = dtype(np.sqrt(max(0.0, 1.0 - float(ct) ** 2)))
        values = cupy.empty(len(l), dtype=dtype)
        kernel(((degree + 128) // 128,), (128,), (np.int32(degree), ct, st, coefficients, values))
        expected = legendre_normalized_trigon_scalar(float(ct), float(st), degree)[l, m]
        actual = cupy.asnumpy(values)
        tolerance = 2e-5 if dtype == np.float32 else 5e-12
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
        if dtype == np.float64:
            # Unnormalized lpmv can overflow at large m, even at the poles.
            # Use its representable orders and check all diagonals analytically.
            selected = m <= 42
            ls, ms = l[selected], m[selected]
            normalized = np.exp(
                0.5 * (np.log((2 * ls + 1) / 2) + gammaln(ls - ms + 1) - gammaln(ls + ms + 1))
            )
            independent = (-1.0) ** ms * normalized * lpmv(ms, ls, float(ct))
            np.testing.assert_allclose(actual[selected], independent, rtol=5e-11, atol=5e-12)
            diagonal = np.arange(degree + 1)
            scale = np.exp(
                0.5
                * (
                    np.log((2 * diagonal + 1) / 2)
                    + gammaln(2 * diagonal + 1)
                    - 2 * diagonal * np.log(2.0)
                    - 2 * gammaln(diagonal + 1)
                )
            )
            np.testing.assert_allclose(
                actual[l == m], scale * float(st) ** diagonal, rtol=5e-12, atol=5e-12
            )
