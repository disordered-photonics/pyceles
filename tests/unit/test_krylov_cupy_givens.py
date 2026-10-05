"""Independent QR checks for the fused, per-Arnoldi-step Givens update."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg.lapack import get_lapack_funcs

from pyceles.linear._gcro_cupy import gcro_cupy_native
from pyceles.linear.krylov_cupy import (
    _apply_givens_rotation,
    _mgs_subtract_scaled,
    fgmres_cupy_native,
    gmres_cupy_native,
    lgmres_cupy_native,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_fused_mgs_update_establishes_ownership_before_in_place_reuse(
    cupy_runtime: tuple[Any, Any], dtype
) -> None:
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(5102026)
    n = 257
    source = np.asarray(rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n), dtype=dtype)
    w = cupy.asarray(source[::2])
    basis = cupy.asarray(np.asarray(rng.normal(size=n) + 1j * rng.normal(size=n), dtype=dtype))
    coefficient = np.asarray(0.37 - 0.19j, dtype=dtype)
    w_before = w.copy()
    expected = cupy.asnumpy(w) - coefficient * cupy.asnumpy(basis)

    updated, owns_w = _mgs_subtract_scaled(w, basis, coefficient, cupy=cupy, owns_w=False)
    assert owns_w
    np.testing.assert_array_equal(cupy.asnumpy(w), cupy.asnumpy(w_before))
    np.testing.assert_allclose(cupy.asnumpy(updated), expected, rtol=2e-6, atol=2e-6)

    second_coefficient = np.asarray(-0.11 + 0.07j, dtype=dtype)
    expected_second = expected - second_coefficient * cupy.asnumpy(basis)
    updated_again, owns_w_again = _mgs_subtract_scaled(
        updated, basis, second_coefficient, cupy=cupy, owns_w=owns_w
    )
    assert owns_w_again
    np.testing.assert_allclose(cupy.asnumpy(updated_again), expected_second, rtol=2e-6, atol=2e-6)


def _reference_step(h, cs, sn, g, col: int) -> float:
    for row in range(col):
        a, b = h[col, row], h[col, row + 1]
        c, s = cs[row].real, sn[row]
        h[col, row] = c * a + s * b
        h[col, row + 1] = -s.conjugate() * a + c * b
    a, b = h[col, col], h[col, col + 1]
    if b == 0:
        c, s, r = 1.0, 0.0j, a
    elif a == 0:
        c, s, r = 0.0, 1.0 + 0.0j, b
    else:
        c, s, r = get_lapack_funcs("lartg", (h,))(a, b)
    cs[col], sn[col] = c, s
    h[col, col], h[col, col + 1] = r, 0.0j
    rhs = g[col]
    g[col], g[col + 1] = c * rhs, -s.conjugate() * rhs
    return float(abs(g[col + 1]))


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_fused_givens_matches_sequential_lapack_and_preserves_inactive_storage(
    cupy_runtime: tuple[Any, Any], dtype
) -> None:
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(4102026)
    steps = 13
    h = np.full((steps, steps + 1), complex(np.nan, np.nan), dtype=dtype)
    for col in range(steps):
        h[col, : col + 1] = rng.normal(size=col + 1) + 1j * rng.normal(size=col + 1)
        h[col, col + 1] = abs(rng.normal())
    cs = np.full(steps, complex(np.nan, np.nan), dtype=dtype)
    sn = cs.copy()
    g = np.full(steps + 1, complex(np.nan, np.nan), dtype=dtype)
    g[0] = 2.0 - 0.3j
    h_device, cs_device, sn_device, g_device = (cupy.asarray(arr) for arr in (h, cs, sn, g))
    residual = cupy.empty((), dtype=h.real.dtype)
    tolerance = 3.0e-5 if dtype == np.complex64 else 3.0e-13
    for col in range(steps):
        expected = _reference_step(h, cs, sn, g, col)
        actual = _apply_givens_rotation(
            h_device, cs_device, sn_device, g_device, col, residual=residual
        )
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        for got, reference in zip(
            (h_device, cs_device, sn_device, g_device), (h, cs, sn, g), strict=True
        ):
            np.testing.assert_allclose(
                cupy.asnumpy(got), reference, rtol=tolerance, atol=tolerance, equal_nan=True
            )
    np.testing.assert_allclose(cs.real**2 + np.abs(sn) ** 2, 1.0, atol=tolerance)


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize(
    "case", ("zero_b", "zero_a", "zeros", "huge", "tiny", "phase", "wide_norm")
)
def test_fused_givens_scaled_generation(cupy_runtime, dtype, case: str) -> None:
    cupy, _ = cupy_runtime
    real_dtype = np.empty((), dtype=dtype).real.dtype
    finfo = np.finfo(real_dtype)
    if case == "zero_b":
        a, b = 3.0 + 4.0j, 0.0j
    elif case == "zero_a":
        a, b = 0.0j, 2.0j  # Preserve pyceles' r=b, s=1 convention.
    elif case == "zeros":
        a, b = 0.0j, 0.0j
    elif case == "huge":
        a, b = 0.6j * finfo.max, 0.6 * finfo.max
    elif case == "wide_norm":
        a, b = (0.9 + 0.9j) * finfo.max, 0.2 * finfo.max
    elif case == "tiny":
        # Keep the test in the normalized range; GPU single precision
        # may flush subnormal inputs to zero before the kernel can scale.
        a, b = 1j * finfo.tiny, finfo.tiny
    else:
        exponent = 25 if dtype == np.complex64 else 200
        a, b = 1j * 10.0 ** (-exponent), 10.0**exponent
    host = np.asarray([[a, b]], dtype=dtype)
    h = cupy.asarray(host)
    cs = cupy.empty(1, dtype=dtype)
    sn = cupy.empty_like(cs)
    g = cupy.asarray([1.0 + 0.3j, complex(np.nan, np.nan)], dtype=dtype)
    residual = cupy.empty((), dtype=real_dtype)
    got_residual = _apply_givens_rotation(h, cs, sn, g, 0, residual=residual)
    c, s = complex(cupy.asnumpy(cs)[0]), complex(cupy.asnumpy(sn)[0])
    row = cupy.asnumpy(h)[0]
    tolerance = 2.0e-6 if dtype == np.complex64 else 2.0e-14
    assert np.isfinite(row).all()
    assert np.isfinite(got_residual)
    assert row[1] == 0.0
    np.testing.assert_allclose(abs(c) ** 2 + abs(s) ** 2, 1.0, atol=tolerance)
    scale = float(max(np.max(np.abs(host.real)), np.max(np.abs(host.imag))))
    if scale > 0.0:
        scaled_a, scaled_b = complex(host[0, 0]) / scale, complex(host[0, 1]) / scale
        np.testing.assert_allclose(-s.conjugate() * scaled_a + c * scaled_b, 0.0, atol=tolerance)
        np.testing.assert_allclose(
            complex(row[0]) / scale,
            c * scaled_a + s * scaled_b,
            rtol=tolerance,
            atol=tolerance,
        )
    if case == "phase":
        np.testing.assert_allclose(s, 1j, atol=tolerance)


@pytest.mark.parametrize("method", ("gmres", "fgmres", "lgmres", "gcro"))
@pytest.mark.parametrize(
    ("compute_dtype", "accum_dtype"),
    ((np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)),
)
def test_restarted_solvers_use_fused_givens(
    cupy_runtime, method, compute_dtype, accum_dtype
) -> None:
    cupy, _ = cupy_runtime
    n = 12
    matrix = np.diag(np.linspace(1.0, 2.0, n).astype(compute_dtype))
    matrix += np.diag(np.full(n - 1, 0.03j, dtype=compute_dtype), k=1)
    solution = np.asarray(np.linspace(0.2, 1.2, n) + 0.1j, dtype=compute_dtype)
    rhs = matrix @ solution
    device_matrix = cupy.asarray(matrix)
    device_rhs = cupy.asarray(rhs)
    x0 = cupy.asarray(0.1 * solution)
    x0_before = x0.copy()
    solver: Any = {
        "gmres": gmres_cupy_native,
        "fgmres": fgmres_cupy_native,
        "lgmres": lgmres_cupy_native,
        "gcro": gcro_cupy_native,
    }[method]

    def action(values):
        return device_matrix @ values

    kwargs: dict[str, Any] = {"recycle_dim": 1} if method == "gcro" else {}
    history: list[float] = []
    tolerance = 3.0e-5 if compute_dtype == np.complex64 else 1.0e-10
    result = solver(
        action,
        device_rhs,
        cupy=cupy,
        x0=x0,
        restart=3,
        maxiter=80,
        operator_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        rtol=tolerance,
        callback=history.append,
        **kwargs,
    )
    assert result.info == 0
    assert result.iterations > 3
    assert len(history) == result.iterations
    np.testing.assert_array_equal(cupy.asnumpy(x0), cupy.asnumpy(x0_before))
    np.testing.assert_array_equal(cupy.asnumpy(device_rhs), rhs)
    actual = cupy.asnumpy(result.x)
    assert np.linalg.norm(matrix @ actual - rhs) / np.linalg.norm(rhs) <= 1.2 * tolerance


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("a, b", ((complex(np.nan, 0), 1), (1, complex(0, np.inf))))
def test_fused_givens_does_not_hide_nonfinite_inputs(cupy_runtime, dtype, a, b) -> None:
    cupy, _ = cupy_runtime
    h = cupy.asarray([[a, b]], dtype=dtype)
    cs = cupy.empty(1, dtype=dtype)
    sn = cupy.empty_like(cs)
    g = cupy.asarray([1, 0], dtype=dtype)
    residual = cupy.empty((), dtype=h.real.dtype)
    value = _apply_givens_rotation(h, cs, sn, g, 0, residual=residual)
    assert np.isnan(value)
    assert np.isnan(cupy.asnumpy(cs)[0])
    assert np.isnan(cupy.asnumpy(sn)[0])
