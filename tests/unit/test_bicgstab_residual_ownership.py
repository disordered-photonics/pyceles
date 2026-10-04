from __future__ import annotations

import numpy as np
import pytest

from pyceles.linear import krylov_cupy

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("preconditioner_kind", ("none", "identity", "diagonal"))
@pytest.mark.parametrize("warm_start", (False, True))
def test_bicgstab_reuses_owned_residual_without_mutating_inputs(
    cupy_runtime, monkeypatch, preconditioner_kind: str, warm_start: bool
) -> None:
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(20261004)
    matrix_host = np.diag(np.linspace(2.0, 5.0, 12)).astype(np.complex128)
    matrix_host += 0.03 * (rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12)))
    b_host = rng.standard_normal(12) + 1j * rng.standard_normal(12)
    matrix = cupy.asarray(matrix_host)
    b = cupy.asarray(b_host)
    x0 = cupy.full(12, 0.1 + 0.05j) if warm_start else None
    initial_x = None if x0 is None else x0.copy()
    first_residual = None
    reused = False
    original_dot = krylov_cupy._dot
    original_norm = krylov_cupy._norm

    def record_residual(u, v, **kwargs):
        nonlocal first_residual
        if first_residual is None:
            first_residual = v
        return original_dot(u, v, **kwargs)

    def inspect_norm(v, **kwargs):
        nonlocal reused
        reused |= v is first_residual
        return original_norm(v, **kwargs)

    monkeypatch.setattr(krylov_cupy, "_dot", record_residual)
    monkeypatch.setattr(krylov_cupy, "_norm", inspect_norm)
    diagonal = cupy.diag(matrix)
    preconditioner = None
    if preconditioner_kind == "identity":

        def preconditioner(values):
            return values
    elif preconditioner_kind == "diagonal":

        def preconditioner(values):
            return values / diagonal

    result = krylov_cupy.bicgstab_cupy_native(
        lambda values: matrix @ values,
        b,
        cupy=cupy,
        x0=x0,
        preconditioner=preconditioner,
        rtol=1e-11,
        maxiter=80,
        operator_dtype=np.complex128,
        accum_dtype=np.complex128,
    )
    assert reused
    assert result.info == 0
    np.testing.assert_allclose(
        cupy.asnumpy(result.x), np.linalg.solve(matrix_host, b_host), rtol=1e-9, atol=1e-11
    )
    np.testing.assert_array_equal(cupy.asnumpy(b), b_host)
    if initial_x is not None:
        np.testing.assert_array_equal(cupy.asnumpy(x0), cupy.asnumpy(initial_x))
