"""Setup guards must not replace or add work to the native norm reductions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyceles.linear import _gcro_cupy, krylov_cupy
from pyceles.linear.solvers import solve_linear_system
from pyceles.simulation import SimulationConfig


@pytest.mark.parametrize(
    ("norm", "rtol", "atol", "expected"),
    [(2.0, 0.1, 0.0, 0.2), (2.0, 0.1, 0.5, 0.5), (0.0, 0.0, 0.0, 0.0)],
)
def test_absolute_residual_target_preserves_finite_policy(
    norm: float, rtol: float, atol: float, expected: float
) -> None:
    assert krylov_cupy._absolute_residual_target(norm, rtol=rtol, atol=atol) == expected


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_absolute_residual_target_rejects_invalid_tolerances(name: str, value: float) -> None:
    tolerances = {"rtol": 1.0e-6, "atol": 0.0}
    tolerances[name] = value
    with pytest.raises(ValueError, match=name):
        krylov_cupy._absolute_residual_target(1.0, **tolerances)


@pytest.mark.parametrize("norm", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_absolute_residual_target_rejects_unusable_rhs_norm(norm: float) -> None:
    with pytest.raises(FloatingPointError, match="RHS norm"):
        krylov_cupy._absolute_residual_target(norm, rtol=1.0e-6, atol=0.0)


def test_absolute_residual_target_rejects_product_overflow() -> None:
    with pytest.raises(FloatingPointError, match="target overflowed"):
        krylov_cupy._absolute_residual_target(float(np.finfo(float).max), rtol=2.0, atol=0.0)


@pytest.mark.fake_gpu
@pytest.mark.parametrize(
    "method", ["gmres", "fgmres", "lgmres", "bicgstab", "lsqr", "gcro", "block"]
)
@pytest.mark.parametrize("bad_norm", [float("nan"), float("inf")])
def test_native_solvers_reject_nonfinite_rhs_norm_before_operator_actions(
    monkeypatch: pytest.MonkeyPatch, method: str, bad_norm: float
) -> None:
    # A test-only host stand-in. All cases must stop before a recurrence, custom
    # kernel, preconditioner action, or extra norm evaluation is needed.
    cupy = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        array=np.array,
        zeros=np.zeros,
        zeros_like=np.zeros_like,
    )
    norm_calls = 0

    def norm(values: Any, **kwargs: Any) -> float:
        nonlocal norm_calls
        norm_calls += 1
        return bad_norm

    def column_norms(values: Any, **kwargs: Any) -> np.ndarray:
        return np.full(values.shape[1], bad_norm)

    def unexpected_action(values: Any) -> Any:
        pytest.fail("A solve with an unusable RHS norm must not apply the operator.")

    monkeypatch.setattr(krylov_cupy, "_norm", norm)
    monkeypatch.setattr(_gcro_cupy, "_norm", norm)
    monkeypatch.setattr(krylov_cupy, "_norms_block", column_norms)
    rhs = np.ones(3, dtype=np.complex128)
    with pytest.raises(FloatingPointError, match="RHS norm"):
        if method == "lsqr":
            krylov_cupy.lsqr_cupy_native(unexpected_action, unexpected_action, rhs, cupy=cupy)
        elif method == "gcro":
            _gcro_cupy.gcro_cupy_native(unexpected_action, rhs, cupy=cupy)
        elif method == "block":
            krylov_cupy.block_gmres_cupy_native(unexpected_action, rhs[:, None], cupy=cupy)
        else:
            solver = getattr(krylov_cupy, f"{method}_cupy_native")
            solver(unexpected_action, rhs, cupy=cupy)
    assert norm_calls == 1


@pytest.mark.parametrize("rtol", [float("nan"), float("inf")])
def test_simulation_rejects_nonfinite_relative_tolerance(rtol: float) -> None:
    with pytest.raises(ValueError, match=r"solver_rtol.*finite"):
        SimulationConfig(solver_rtol=rtol, verbose=False)


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_public_iterative_dispatch_rejects_invalid_tolerances_before_actions(
    name: str, value: float
) -> None:
    def unexpected_action(values: np.ndarray) -> np.ndarray:
        pytest.fail("Invalid stopping controls must be rejected before an operator action.")

    tolerances = {"rtol": 1.0e-6, "atol": 0.0}
    tolerances[name] = value
    with pytest.raises(ValueError, match=name):
        solve_linear_system(
            unexpected_action,
            np.ones(2, dtype=np.complex128),
            method="gmres",
            show_progress=False,
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
        )
