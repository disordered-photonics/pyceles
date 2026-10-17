"""Source amplitude must not change the solvability of a linear system."""

from typing import Literal

import numpy as np
import pytest

from pyceles.linear import solve_linear_system


@pytest.mark.parametrize(
    ("backend", "method"),
    [("numpy", method) for method in ("direct", "gmres", "bicgstab", "lgmres", "gcrotmk", "lsqr")]
    + [
        pytest.param("cupy", method, marks=pytest.mark.gpu)
        for method in (
            "direct",
            "gmres",
            "fgmres",
            "bicgstab",
            "lgmres",
            "gcro",
            "lsqr",
            "block",
            "block_mixed",
        )
    ],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_solver_source_amplitude_invariance(
    request, backend: Literal["numpy", "cupy"], method, dtype
):
    xp = np if backend == "numpy" else request.getfixturevalue("cupy_runtime")[0]
    matrix = np.diag(np.linspace(2, 4, 6)).astype(dtype)
    matrix += np.diag(np.full(5, 0.1j), 1) + np.diag(np.full(5, -0.2), -1)
    device_matrix = xp.asarray(matrix)
    solution = np.array([1 + 0.2j, -0.3j, 0.4, -0.2, 0.1j, 0.6], dtype=dtype)
    if method.startswith("block"):
        solution = np.column_stack((solution, solution[::-1]))
    tolerance = 2e-6 if dtype == np.complex64 else 1e-11
    # GPU subnormal arithmetic is not an amplitude-invariance guarantee.
    # Keep c64 corrections normal; c128 also probes below absolute 1e-30 cutoffs.
    small = 1e-24 if dtype == np.complex64 else 1e-36
    cases = [(scale, tolerance, 0.0) for scale in (1.0, 1e-12, small, 1e12)]
    if method != "block_mixed":
        unit_norm = np.min(np.linalg.norm(np.asarray(matrix @ solution, dtype=complex), axis=0))
        cases.append((1e-12, 0.0, tolerance * unit_norm * 1e-12))
    for scale, relative, absolute in cases:
        scaling = np.array([scale, 1.0]) if method == "block_mixed" else np.asarray(scale)
        expected = solution * scaling
        rhs = matrix @ expected
        result = solve_linear_system(
            lambda values: device_matrix @ xp.asarray(values),
            rhs,
            A_h_mv=lambda values: device_matrix.conj().T @ xp.asarray(values),
            A_dense=matrix,
            method="gmres" if method.startswith("block") else method,
            backend=backend,
            dtype=dtype,
            accum_dtype=np.complex128,
            x0=0.1 * expected,
            rtol=relative,
            atol=absolute,
            maxiter=100,
            restart=6,
            gcro_recycle_dim=2,
            show_progress=False,
        )
        context = (backend, method, dtype, scale)
        assert np.all(np.asarray(result.info) == 0), (*context, result)
        error = np.linalg.norm(
            np.asarray(result.x / scaling - solution, dtype=complex), axis=0
        ) / np.linalg.norm(solution, axis=0)
        assert np.all(error < 5 * tolerance), (*context, error)
        assert np.all(np.asarray(result.relative_residual) < 2 * tolerance), (*context, result)
        if method == "lsqr":
            assert result.block_metadata is not None
            np.testing.assert_allclose(
                result.block_metadata["correction_norm"],
                0.9 * np.linalg.norm(np.asarray(expected, dtype=complex)),
                rtol=5 * tolerance,
                atol=0,
            )
