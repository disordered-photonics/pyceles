"""LSQR stationarity is not physical convergence, with or without diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.linear.krylov_cupy import lsqr_cupy_native
from pyceles.linear.solvers import lsqr_scipy


@pytest.mark.parametrize("rhs", [[0.0, 1.0], [1.0, 1.0]])
@pytest.mark.parametrize("final_check", [False, True])
def test_lsqr_scipy_rejects_unsolved_singular_system(rhs: list[float], final_check: bool) -> None:
    matrix = np.diag(np.asarray([1.0, 0.0], dtype=np.complex128))
    b = np.asarray(rhs, dtype=np.complex128)
    forward_calls = 0
    adjoint_calls = 0

    def forward(values: np.ndarray) -> np.ndarray:
        nonlocal forward_calls
        forward_calls += 1
        return np.asarray(matrix @ values)

    def adjoint(values: np.ndarray) -> np.ndarray:
        nonlocal adjoint_calls
        adjoint_calls += 1
        return np.asarray(matrix.conj().T @ values)

    result = lsqr_scipy(
        forward,
        adjoint,
        b,
        rtol=1.0e-10,
        maxiter=10,
        show_progress=False,
        compute_final_residual=final_check,
    )
    assert result.info != 0
    assert result.converged_reason != "converged"
    assert np.linalg.norm(matrix @ result.x - b) >= 1.0
    assert result.block_metadata is not None
    assert result.block_metadata["operator_applications"] == forward_calls
    assert result.block_metadata["adjoint_applications"] == adjoint_calls
    if final_check:
        assert np.isfinite(result.residual_norm)
    else:
        assert np.isnan(result.residual_norm)
        assert np.isnan(result.relative_residual)
        assert result.true_residual_history is None


@pytest.mark.parametrize("istop", [0, 1, 2])
@pytest.mark.parametrize("estimate", [0.0, 0.5, float("nan"), float("inf")])
def test_lsqr_scipy_uses_returned_residual_without_an_extra_action(
    monkeypatch: pytest.MonkeyPatch, istop: int, estimate: float
) -> None:
    import scipy.sparse.linalg

    b = np.ones(2, dtype=np.complex128)

    def fake_lsqr(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (b.copy(), istop, 2, estimate, estimate, 1.0, 1.0, 0.0, 1.0, np.zeros(2))

    def unexpected_action(values: np.ndarray) -> np.ndarray:
        pytest.fail("No terminal action is permitted with diagnostics disabled.")

    monkeypatch.setattr(scipy.sparse.linalg, "lsqr", fake_lsqr)
    result = lsqr_scipy(
        unexpected_action,
        unexpected_action,
        b,
        rtol=1.0e-6,
        show_progress=False,
        compute_final_residual=False,
    )
    assert (result.info == 0) == (estimate == 0.0)
    assert np.isnan(result.residual_norm)
    assert result.block_metadata is not None
    assert result.block_metadata["operator_applications"] == 0
    assert result.block_metadata["adjoint_applications"] == 0


def test_lsqr_scipy_physical_check_overrides_an_optimistic_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.sparse.linalg

    def fake_lsqr(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return (np.zeros(2, dtype=np.complex128), 1, 1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, np.zeros(2))

    def identity(values: np.ndarray) -> np.ndarray:
        return values.copy()

    monkeypatch.setattr(scipy.sparse.linalg, "lsqr", fake_lsqr)
    result = lsqr_scipy(identity, identity, np.ones(2, dtype=np.complex128), show_progress=False)
    assert result.info != 0
    assert result.converged_reason == "tolerance_not_met"


@pytest.mark.fake_gpu
@pytest.mark.parametrize("size", [1, 3])
def test_native_lsqr_checks_warm_start_before_zero_residual_shortcut(size: int) -> None:
    def unexpected_action(values: Any) -> Any:
        pytest.fail("Warm-start shape must be checked before either operator action.")

    with pytest.raises(ValueError, match=r"x0.*size.*b.*size"):
        lsqr_cupy_native(
            unexpected_action,
            unexpected_action,
            np.ones(2, dtype=np.complex128),
            cupy=np,
            x0=np.zeros(size, dtype=np.complex128),
            initial_residual=np.zeros(2, dtype=np.complex128),
        )


@pytest.mark.parametrize("final_check", [False, True])
def test_lsqr_scipy_known_zero_residual_respects_diagnostic_policy(final_check: bool) -> None:
    b = np.ones(2, dtype=np.complex128)

    def unexpected_action(values: np.ndarray) -> np.ndarray:
        pytest.fail("A supplied converged residual must not trigger an operator probe.")

    result = lsqr_scipy(
        unexpected_action,
        unexpected_action,
        b,
        x0=b,
        initial_residual=np.zeros_like(b),
        show_progress=False,
        compute_final_residual=final_check,
    )
    assert result.info == 0
    np.testing.assert_array_equal(result.x, b)
    assert not np.shares_memory(result.x, b)
    if final_check:
        assert result.residual_norm == 0.0
        assert result.relative_residual == 0.0
    else:
        assert np.isnan(result.residual_norm)
        assert np.isnan(result.relative_residual)
