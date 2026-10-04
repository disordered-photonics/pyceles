from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.linear.krylov_cupy import _block_basis_projection, _dot, _norm, _norms_block

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("case", ("roundoff", "overflow"))
def test_dot_widens_operands_before_multiplication(
    cupy_runtime: tuple[Any, Any], case: str
) -> None:
    cupy, _ = cupy_runtime
    if case == "roundoff":
        # The first product rounds to 1 in float32, erasing the final -2**-46.
        eps = 2.0**-23
        left = np.asarray([1.0 + eps, 1.0], dtype=np.complex64)
        right = np.asarray([1.0 - eps, -1.0], dtype=np.complex64)
    else:
        # Inputs fit in float32, but their products require float64 range.
        scale = 2.0**70
        left = np.asarray([scale * (1.0 + 1.0j), -scale], dtype=np.complex64)
        right = np.asarray([scale * (1.0 - 1.0j), scale * 1.0j], dtype=np.complex64)
    expected = np.vdot(left.astype(np.complex128), right.astype(np.complex128))
    actual = _dot(
        cupy.asarray(left),
        cupy.asarray(right),
        cupy=cupy,
        accum_dtype=np.dtype(np.complex128),
    )
    assert actual.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(cupy.asnumpy(actual), expected)


@pytest.mark.parametrize(
    ("input_dtype", "accum_dtype"),
    ((np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)),
)
@pytest.mark.parametrize("strided", (False, True))
def test_block_norms_match_wide_reference_without_widening_input(
    cupy_runtime: tuple[Any, Any],
    monkeypatch,
    input_dtype,
    accum_dtype,
    strided: bool,
) -> None:
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(20261003)
    host = (rng.standard_normal((258, 6)) + 1j * rng.standard_normal((258, 6))).astype(input_dtype)
    device = cupy.asarray(host)
    if strided:
        host = host[::2, ::2]
        device = device[::2, ::2]
    expected = np.linalg.norm(host.astype(accum_dtype), axis=0)
    expected_frob = np.linalg.norm(host.astype(accum_dtype))
    original_asarray = cupy.asarray

    def no_input_widening(values, dtype=None, *args, **kwargs):
        if values is device and dtype is not None:
            assert np.dtype(dtype).itemsize <= device.dtype.itemsize
        return original_asarray(values, dtype, *args, **kwargs)

    monkeypatch.setattr(cupy, "asarray", no_input_widening)
    actual = _norms_block(device, cupy=cupy, accum_dtype=np.dtype(accum_dtype))
    tolerance = 2.0e-6 if accum_dtype == np.complex64 else 2.0e-14
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=tolerance, atol=0.0)
    actual_frob = _norm(device, cupy=cupy, accum_dtype=np.dtype(accum_dtype))
    np.testing.assert_allclose(actual_frob, expected_frob, rtol=tolerance, atol=0.0)


@pytest.mark.parametrize("shape", ((0, 3), (7, 0), (0, 0)))
def test_block_norms_empty_axes(cupy_runtime: tuple[Any, Any], shape) -> None:
    cupy, _ = cupy_runtime
    actual = _norms_block(
        cupy.empty(shape, dtype=cupy.complex64),
        cupy=cupy,
        accum_dtype=np.dtype(np.complex128),
    )
    assert actual.shape == (shape[1],)
    assert actual.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(cupy.asnumpy(actual), np.zeros(shape[1]))


@pytest.mark.parametrize(
    ("input_dtype", "accum_dtype"),
    ((np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)),
)
@pytest.mark.parametrize("n_blocks", (1, 4))
def test_block_projection_packs_into_owned_accumulation_workspace(
    cupy_runtime: tuple[Any, Any],
    input_dtype,
    accum_dtype,
    n_blocks: int,
) -> None:
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(20261004)
    n, p = 17, 3
    basis_host = (
        rng.standard_normal((n_blocks, 2 * n, p)) + 1j * rng.standard_normal((n_blocks, 2 * n, p))
    ).astype(input_dtype)
    values_host = (rng.standard_normal((2 * n, p)) + 1j * rng.standard_normal((2 * n, p))).astype(
        input_dtype
    )
    basis = cupy.asarray(basis_host)[:, ::2, :]
    values = cupy.asarray(values_host)[::2, :]
    # Force several chunks and a short final chunk without large allocations.
    budget = 5 * (n_blocks * p + p) * np.dtype(accum_dtype).itemsize
    actual = _block_basis_projection(
        basis,
        values,
        cupy=cupy,
        accum_dtype=np.dtype(accum_dtype),
        workspace_bytes=budget,
    )
    packed = basis_host[:, ::2, :].transpose(1, 0, 2).reshape(n, n_blocks * p)
    expected = packed.astype(accum_dtype).conj().T @ values_host[::2, :].astype(accum_dtype)
    tolerance = 5.0e-6 if accum_dtype == np.complex64 else 2.0e-13
    assert actual.dtype == np.dtype(accum_dtype)
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(cupy.asnumpy(basis), basis_host[:, ::2, :])
    np.testing.assert_array_equal(cupy.asnumpy(values), values_host[::2, :])
