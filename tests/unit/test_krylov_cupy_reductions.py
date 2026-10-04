from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import pytest

from pyceles.linear.krylov_cupy import (
    _block_basis_projection,
    _dot,
    _norm,
    _norms_block,
    _row_basis_combination,
    block_gmres_cupy_native,
    fgmres_cupy_native,
    gmres_cupy_native,
    lgmres_cupy_native,
)

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


@pytest.mark.parametrize("input_dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("budget", (1, 576, 2**20))
def test_row_basis_combination_preserves_precision_and_bounds_widening(
    cupy_runtime, monkeypatch, input_dtype, budget
):
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(20261004)
    storage = np.asarray(
        rng.standard_normal((10, 34)) + 1j * rng.standard_normal((10, 34)),
        dtype=input_dtype,
    )
    basis = cupy.asarray(storage)[::2, ::2]
    host_basis = storage[::2, ::2]
    weights_host = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    weights = cupy.asarray(weights_host, dtype=np.complex128)
    packed_shapes = []
    original_empty = cupy.empty

    def tracked_empty(shape, dtype=None, *args, **kwargs):
        if len(shape) == 2 and np.dtype(dtype) == np.dtype(np.complex128):
            packed_shapes.append(tuple(shape))
        return original_empty(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(cupy, "empty", tracked_empty)
    actual = _row_basis_combination(
        weights, basis, cupy=cupy, accum_dtype=np.dtype(np.complex128), workspace_bytes=budget
    )
    expected = (weights_host @ host_basis.astype(np.complex128)).astype(input_dtype)
    tolerance = 3.0e-6 if input_dtype == np.complex64 else 1.0e-12
    assert actual.dtype == np.dtype(input_dtype)
    np.testing.assert_allclose(cupy.asnumpy(actual), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(cupy.asnumpy(basis), host_basis)
    np.testing.assert_array_equal(cupy.asnumpy(weights), weights_host)
    if input_dtype == np.complex64:
        chunk = max(1, min(17, budget // (6 * 16)))
        assert packed_shapes
        assert all(rows == 5 and columns <= chunk for rows, columns in packed_shapes)
    else:
        assert not packed_shapes


def test_row_basis_combination_does_not_narrow_cgs_coefficients(cupy_runtime):
    cupy, _ = cupy_runtime
    epsilon = 2.0**-24
    weights = cupy.asarray([1.0 + epsilon, -1.0], dtype=cupy.complex128)
    basis = cupy.ones((2, 3), dtype=cupy.complex64)
    actual = _row_basis_combination(
        weights, basis, cupy=cupy, accum_dtype=np.dtype(np.complex128), workspace_bytes=1
    )
    # Casting the small coefficient vector to complex64 instead would give 0.
    np.testing.assert_array_equal(cupy.asnumpy(actual), np.full(3, epsilon, dtype=np.complex64))


@pytest.mark.parametrize("solver", (gmres_cupy_native, fgmres_cupy_native, lgmres_cupy_native))
@pytest.mark.parametrize("refinement", ("never", "ifneeded", "always"))
def test_cgs_mixed_precision_solvers_reconstruct_across_restarts(cupy_runtime, solver, refinement):
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(23)
    n = 24
    matrix = np.diag(np.linspace(1.0, 3.0, n)).astype(np.complex64)
    matrix += (0.02 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))).astype(
        np.complex64
    )
    expected = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    a = cupy.asarray(matrix)
    b = cupy.asarray(matrix @ expected)
    result = solver(
        lambda x: a @ x,
        b,
        cupy=cupy,
        operator_dtype=np.complex64,
        accum_dtype=np.complex128,
        orthogonalization="cgs",
        cgs_refinement=refinement,
        restart=3,
        maxiter=60,
        rtol=2e-6,
    )
    assert result.info == 0
    assert result.iterations > 3
    assert result.relative_residual <= 2e-6
    np.testing.assert_allclose(cupy.asnumpy(result.x), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "solver", (fgmres_cupy_native, lgmres_cupy_native, block_gmres_cupy_native)
)
def test_device_restart_releases_previous_basis(cupy_runtime, monkeypatch, solver):
    cupy, _ = cupy_runtime
    n = 64
    block = solver is block_gmres_cupy_native
    basis_shape = (2, n, 2) if block else (2, n)
    refs: list[weakref.ReferenceType[Any]] = []
    live_at_allocation = []
    original_empty = cupy.empty

    def tracked_empty(shape, *args, **kwargs):
        result = original_empty(shape, *args, **kwargs)
        if isinstance(shape, tuple) and shape == basis_shape:
            live_at_allocation.append(sum(ref() is not None for ref in refs))
            refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(cupy, "empty", tracked_empty)
    rng = np.random.default_rng(41)
    b = cupy.asarray(rng.standard_normal((n, 2) if block else (n,)), dtype=cupy.complex128)
    diagonal = cupy.linspace(1.0, 3.0, n)
    if block:
        diagonal = diagonal[:, None]
    kwargs = {"outer_k": 0} if solver is lgmres_cupy_native else {}
    result = solver(
        lambda x: diagonal * x,
        b,
        cupy=cupy,
        preconditioner=lambda x: x,
        restart=1,
        maxiter=2,
        rtol=0.0,
        atol=0.0,
        **kwargs,
    )
    assert result.iterations == 2
    assert live_at_allocation == [0, 0]
    assert all(ref() is None for ref in refs)
