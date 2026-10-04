import sys
import types
import weakref
from typing import Any, Literal, cast

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres

from pyceles.io.hdf5 import load_solution_h5, save_solution_h5
from pyceles.linear import solvers
from pyceles.linear.solvers import (
    direct_dense_scipy,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    gcrotmk_scipy,
    gmres_scipy,
    lgmres_scipy,
    lsqr_scipy,
    solve_linear_system,
)


def _fake_cupy_numpy_backend():
    class _FakeCuPy:
        @staticmethod
        def asarray(x, dtype=None, order=None):
            del order
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def asnumpy(x):
            return np.asarray(x)

        @staticmethod
        def array(x, dtype=None):
            return np.array(x, dtype=dtype)

        @staticmethod
        def zeros_like(x, dtype=None):
            return np.zeros_like(np.asarray(x), dtype=dtype)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None):
            return np.empty(shape, dtype=dtype)

        @staticmethod
        def concatenate(xs, axis=0):
            return np.concatenate(xs, axis=axis)

        @staticmethod
        def einsum(*args, **kwargs):
            return np.einsum(*args, **kwargs)

        @staticmethod
        def subtract(x, y, out=None):
            return np.subtract(x, y, out=out)

        @staticmethod
        def column_stack(xs):
            return np.column_stack(xs)

        @staticmethod
        def abs(x):
            return np.abs(x)

        @staticmethod
        def sqrt(x):
            return np.sqrt(x)

        @staticmethod
        def conj(x, out=None):
            return np.conj(x, out=out)

        @staticmethod
        def vdot(x, y):
            return np.vdot(x, y)

        @staticmethod
        def max(x):
            return np.max(x)

        @staticmethod
        def min(x):
            return np.min(x)

        @staticmethod
        def diag(x):
            return np.diag(x)

        @staticmethod
        def argsort(x):
            return np.argsort(x)

        @staticmethod
        def count_nonzero(x):
            return np.count_nonzero(x)

        class linalg:
            norm = staticmethod(np.linalg.norm)
            solve = staticmethod(np.linalg.solve)
            qr = staticmethod(np.linalg.qr)
            lstsq = staticmethod(np.linalg.lstsq)
            eigh = staticmethod(np.linalg.eigh)

    return _FakeCuPy()


def test_gmres_result_reports_true_residual():
    n = 16
    rng = np.random.default_rng(2)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def A_mv(x: np.ndarray) -> np.ndarray:
        return x.copy()

    out = gmres_scipy(A_mv, b, rtol=1e-12, atol=0.0, restart=10, maxiter=20, show_progress=False)
    np.testing.assert_allclose(out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.relative_residual <= 1e-12
    assert out.residual_norm <= 1e-12 * np.linalg.norm(b)
    assert out.iterations >= 1
    assert out.method == "gmres"


def test_gmres_scipy_uses_inner_iteration_limit(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_gmres(*args: Any, **kwargs: Any) -> tuple[np.ndarray, int]:
        captured.update(kwargs)
        return np.asarray(args[1]).copy(), 0

    monkeypatch.setattr("scipy.sparse.linalg.gmres", fake_gmres)
    b = np.asarray([1.0 + 0.0j, 2.0 + 0.0j])
    out = gmres_scipy(
        lambda x: np.asarray(x),
        b,
        restart=7,
        maxiter=3,
        show_progress=False,
        compute_final_residual=False,
    )

    assert captured["restart"] == 7
    assert captured["maxiter"] == 3
    assert captured["callback_type"] == "legacy"
    assert out.info == 0


def test_direct_dense_solve_identity():
    n = 8
    rng = np.random.default_rng(5)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    out = direct_dense_scipy(lambda x: x.copy(), b, show_progress=False)
    np.testing.assert_allclose(out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.method == "direct"
    assert out.relative_residual <= 1e-12


def test_direct_dense_uses_preassembled_matrix():
    A = np.array([[2.0 + 0j, 1.0 - 1.0j], [0.5 + 0.2j, 3.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, -2.0 + 0.5j], dtype=np.complex128)
    out = direct_dense_scipy(lambda x: x.copy(), b, A_dense=A, show_progress=False)
    np.testing.assert_allclose(A @ out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.method == "direct"


def test_direct_dense_uses_precomputed_lu_factorization():
    A = np.array(
        [[2.0 + 0j, 1.0 - 1.0j], [0.5 + 0.2j, 3.0 + 0j]],
        dtype=np.complex128,
    )
    b = np.array([1.0 + 0j, -2.0 + 0.5j], dtype=np.complex128)
    lu = factorize_dense_matrix(A, dtype=np.complex128)
    out = direct_dense_scipy(
        lambda x: A @ np.asarray(x),
        b,
        A_factorized=lu,
        show_progress=False,
    )
    np.testing.assert_allclose(A @ out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.method == "direct"


def test_solve_linear_system_rejects_direct_preconditioner() -> None:
    b = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="does not use a preconditioner"):
        solve_linear_system(
            lambda x: np.asarray(x),
            b,
            method="direct",
            preconditioner=lambda x: x,
            show_progress=False,
        )


def test_solve_linear_system_defaults_to_gmres():
    b = np.array([1.0 + 0j, 2.0 + 0j])
    out = solve_linear_system(lambda x: x.copy(), b, show_progress=False)
    assert out.method == "gmres"
    np.testing.assert_allclose(out.x, b)


def test_estimate_dense_matrix_bytes():
    n = 10
    assert estimate_dense_matrix_bytes(n) == n * n * np.dtype(np.complex128).itemsize


def test_lgmres_and_gcrotmk_identity():
    n = 12
    rng = np.random.default_rng(11)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    out_lgmres = lgmres_scipy(
        lambda x: x.copy(), b, rtol=1e-12, atol=0.0, maxiter=20, show_progress=False
    )
    np.testing.assert_allclose(out_lgmres.x, b, atol=1e-10, rtol=1e-10)
    assert out_lgmres.info == 0
    assert out_lgmres.relative_residual <= 1e-10
    assert out_lgmres.iterations >= 1

    out_gcrotmk = gcrotmk_scipy(
        lambda x: x.copy(), b, rtol=1e-12, atol=0.0, maxiter=20, show_progress=False
    )
    np.testing.assert_allclose(out_gcrotmk.x, b, atol=1e-10, rtol=1e-10)
    assert out_gcrotmk.info == 0
    assert out_gcrotmk.relative_residual <= 1e-10
    assert out_gcrotmk.iterations >= 1


def test_final_residual_progress_update_does_not_advance_iteration_count(monkeypatch):
    bars: list[Any] = []

    class FakeProgressBar:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.n = 0
            self.total = None
            self.postfix = ""
            bars.append(self)

        def update(self, amount):
            self.n += int(amount)

        def set_postfix_str(self, value, *, refresh):
            del refresh
            self.postfix = str(value)

        def close(self):
            return None

    monkeypatch.setattr(solvers, "tqdm", FakeProgressBar)
    update, close, history = solvers._make_progress_tracker(
        "gmres", show_progress=True, target_rel=1e-6, max_iters=10
    )
    update(1e-2)
    update(1e-4)
    update(1e-7, residual_label_override="true_final_rel_res", advance=False)
    close()

    assert bars[0].n == 2
    assert history == [1e-2, 1e-4]
    assert bars[0].postfix == "true_final_rel_res=1.000e-07"


def test_bicgstab_progress_callback_does_not_add_residual_matvec(monkeypatch):
    A = np.array(
        [
            [4.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 3.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    b = np.asarray([1.0 + 0.0j, -1.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
    progress_updates: list[float | None] = []

    def fake_progress_tracker(*args, **kwargs):
        return progress_updates.append, lambda: None, []

    monkeypatch.setattr(solvers, "_make_progress_tracker", fake_progress_tracker)

    def run(*, show_progress: bool) -> tuple[solvers.LinearSolveResult, int]:
        calls = 0

        def A_mv(x: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return cast(np.ndarray, A @ np.asarray(x))

        out = solvers.bicgstab_scipy(
            A_mv,
            b,
            rtol=1e-15,
            atol=0.0,
            maxiter=1,
            show_progress=show_progress,
            compute_final_residual=False,
        )
        return out, calls

    out_no_progress, calls_no_progress = run(show_progress=False)
    progress_updates.clear()
    out, calls_progress = run(show_progress=True)

    assert int(out_no_progress.iterations) == 1
    assert int(out.iterations) == 1
    assert progress_updates == [None]
    assert calls_progress == calls_no_progress


def test_solve_linear_system_supports_multi_rhs_direct():
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    B = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, -1.0 + 0j]], dtype=np.complex128)
    out = solve_linear_system(
        lambda x: A @ np.asarray(x),
        B,
        method="direct",
        A_dense=A,
        show_progress=False,
    )
    np.testing.assert_allclose(A @ out.x, B, atol=1e-12, rtol=1e-12)
    assert out.rhs_count == 2
    assert np.asarray(out.info).shape == (2,)


def test_solve_linear_system_supports_multi_rhs_direct_with_precomputed_lu():
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    B = np.array([[1.0 + 0j, 2.0 + 0j], [3.0 + 0j, -1.0 + 0j]], dtype=np.complex128)
    lu = factorize_dense_matrix(A, dtype=np.complex128)
    out = solve_linear_system(
        lambda x: A @ np.asarray(x),
        B,
        method="direct",
        A_factorized=lu,
        show_progress=False,
    )
    np.testing.assert_allclose(A @ out.x, B, atol=1e-12, rtol=1e-12)
    assert out.rhs_count == 2
    assert np.asarray(out.info).shape == (2,)


def test_solve_linear_system_direct_with_lu_uses_dense_residual_path():
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, 3.0 + 0j], dtype=np.complex128)
    lu = factorize_dense_matrix(A, dtype=np.complex128)
    calls = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, A @ np.asarray(x))

    out = solve_linear_system(
        A_mv,
        b,
        method="direct",
        A_dense=A,
        A_factorized=lu,
        show_progress=False,
    )
    np.testing.assert_allclose(A @ out.x, b, atol=1e-12, rtol=1e-12)
    assert calls == 0


def test_solve_linear_system_direct_can_skip_final_residual_with_lu_only():
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, 3.0 + 0j], dtype=np.complex128)
    lu = factorize_dense_matrix(A, dtype=np.complex128)
    calls = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, A @ np.asarray(x))

    out = solve_linear_system(
        A_mv,
        b,
        method="direct",
        A_factorized=lu,
        show_progress=False,
        compute_final_residual=False,
    )
    np.testing.assert_allclose(A @ out.x, b, atol=1e-12, rtol=1e-12)
    assert calls == 0
    assert np.isnan(float(out.residual_norm))
    assert np.isnan(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_solve_linear_system_forwards_accum_dtype_to_native_cupy_bicgstab(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    seen: dict[str, object] = {}
    real_native = solvers.bicgstab_cupy_native

    def _capture_native(*args: Any, **kwargs: Any):
        seen["accum_dtype"] = kwargs.get("accum_dtype")
        return real_native(*args, **kwargs)

    monkeypatch.setattr(solvers, "bicgstab_cupy_native", _capture_native)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex64)
    out = solve_linear_system(
        lambda x: np.asarray(x),
        b,
        method="bicgstab",
        backend="cupy",
        dtype=np.complex64,
        accum_dtype=np.complex64,
        rtol=1e-6,
        maxiter=4,
        show_progress=False,
    )

    assert int(out.info) == 0
    assert np.dtype(cast(Any, seen["accum_dtype"])) == np.dtype(np.complex64)


@pytest.mark.fake_gpu
def test_cupy_solver_rejects_accumulation_downcast(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match=r"accum_dtype.*at least as precise"):
        solvers.bicgstab_cupy(
            lambda x: np.asarray(x),
            b,
            accum_dtype=np.complex64,
            maxiter=4,
            show_progress=False,
        )


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_backend_supports_bicgstab(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, 2.0 + 0j])
    out = solve_linear_system(
        lambda x: np.asarray(x),
        b,
        method="bicgstab",
        backend="cupy",
        rtol=1e-12,
        atol=0.0,
        maxiter=10,
        show_progress=False,
    )
    assert int(out.info) == 0
    assert str(out.method) == "bicgstab[cupy]"
    np.testing.assert_allclose(np.asarray(out.x), b, atol=1e-10, rtol=1e-10)


@pytest.mark.fake_gpu
def test_factorize_dense_matrix_cupy_requests_inplace_overwrite(monkeypatch):
    calls: list[tuple[bool, bool]] = []

    class _FakeCuPy:
        @staticmethod
        def asarray(x, dtype=None, order=None):
            del order
            return np.asarray(x, dtype=dtype)

    fake_linalg = types.SimpleNamespace()

    def _lu_factor(a, overwrite_a=False, check_finite=True):
        calls.append((bool(overwrite_a), bool(check_finite)))
        return (np.asarray(a), np.array([0], dtype=int))

    fake_linalg.lu_factor = _lu_factor
    fake_scipy = types.ModuleType("cupyx.scipy")
    cast(Any, fake_scipy).linalg = fake_linalg
    fake_cupyx = types.ModuleType("cupyx")
    cast(Any, fake_cupyx).scipy = fake_scipy
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_linalg)
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_FakeCuPy(), None))

    A = np.eye(2, dtype=np.complex64)
    solvers.factorize_dense_matrix(
        A,
        dtype=np.complex64,
        backend="cupy",
        overwrite_input=True,
    )
    assert calls == [(True, True)]


@pytest.mark.fake_gpu
def test_gmres_cupy_reports_clear_import_failure(monkeypatch):
    def fail_import():
        raise RuntimeError("broken cuda path")

    monkeypatch.setattr(solvers, "import_cupy", fail_import)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(RuntimeError, match="broken cuda path"):
        solvers.gmres_cupy(lambda x: x, b, show_progress=False)


@pytest.mark.fake_gpu
def test_fgmres_cupy_reports_clear_import_failure(monkeypatch):
    def fail_import():
        raise RuntimeError("broken cuda path")

    monkeypatch.setattr(solvers, "import_cupy", fail_import)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(RuntimeError, match="broken cuda path"):
        solvers.fgmres_cupy(lambda x: x, b, show_progress=False)


@pytest.mark.fake_gpu
def test_lgmres_cupy_reports_clear_import_failure(monkeypatch):
    def fail_import():
        raise RuntimeError("broken cuda path")

    monkeypatch.setattr(solvers, "import_cupy", fail_import)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(RuntimeError, match="broken cuda path"):
        solvers.lgmres_cupy(lambda x: x, b, show_progress=False)


@pytest.mark.fake_gpu
def test_bicgstab_cupy_reports_clear_import_failure(monkeypatch):
    def fail_import():
        raise RuntimeError("broken cuda path")

    monkeypatch.setattr(solvers, "import_cupy", fail_import)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(RuntimeError, match="broken cuda path"):
        solvers.bicgstab_cupy(lambda x: x, b, show_progress=False)


@pytest.mark.fake_gpu
def test_gmres_cupy_native_reports_inner_iteration_progress(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    progress: list[float] = []
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    x_true = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)
    b = A @ x_true

    out = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=2,
        maxiter=8,
        callback=progress.append,
        show_progress=False,
    )

    assert out.info == 0
    assert int(out.iterations) >= 1
    assert out.residual_history is not None
    assert np.asarray(progress).shape == (int(out.iterations),)
    assert progress[0] >= progress[-1]
    np.testing.assert_allclose(np.asarray(out.x), x_true, atol=1e-9, rtol=1e-9)
    assert float(out.relative_residual) <= 1e-10


@pytest.mark.fake_gpu
def test_gmres_cupy_final_flag_does_not_skip_restart_residuals(monkeypatch):
    """The cheap-result flag must not corrupt the next restarted cycle."""
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    matrix = np.asarray(
        [[1.0 + 0.0j, 0.25 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]],
        dtype=np.complex128,
    )
    rhs = np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    matvec_calls = 0

    def _matvec(values: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        matvec_calls += 1
        return cast(np.ndarray, matrix @ np.asarray(values))

    out = solvers.gmres_cupy(
        _matvec,
        rhs,
        rtol=0.0,
        atol=0.0,
        restart=1,
        maxiter=2,
        show_progress=False,
        compute_final_residual=False,
    )

    # Two Arnoldi matvecs are required.  Only the first cycle boundary needs
    # another physical residual because a second restarted cycle follows; the
    # terminal residual is exactly what the cheap-result flag is allowed to omit.
    assert matvec_calls == 3
    assert out.true_residual_history is not None
    assert len(np.asarray(out.true_residual_history)) == 2
    assert np.isnan(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_fgmres_cupy_variable_preconditioner_state(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -2.0 + 0j, 0.5 + 0j], dtype=np.complex128)
    seen_states: list[tuple[int, int]] = []

    def _pre(v: np.ndarray, state: dict[str, int]) -> np.ndarray:
        seen_states.append((int(state["iteration"]), int(state["cycle_iteration"])))
        # Deliberately iteration-varying scaling to exercise flexible path.
        scale = 1.0 + 0.1 * float(state["iteration"] + 1)
        return np.asarray(v) / scale

    out = solvers.fgmres_cupy(
        lambda x: np.asarray(x),
        b,
        preconditioner=_pre,
        rtol=1e-12,
        atol=0.0,
        restart=5,
        maxiter=20,
        show_progress=False,
    )
    assert int(out.info) == 0
    assert str(out.converged_reason) == "converged"
    np.testing.assert_allclose(np.asarray(out.x), b, atol=1e-10, rtol=1e-10)
    assert len(seen_states) >= 1
    assert seen_states[0] == (0, 0)


@pytest.mark.fake_gpu
def test_fgmres_cupy_matches_gmres_on_toy_system(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(21)
    n = 10
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (0.5 + 0j) * np.eye(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    out_g = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=10,
        maxiter=60,
        show_progress=False,
    )
    out_f = solvers.fgmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=10,
        maxiter=60,
        show_progress=False,
    )
    assert int(out_g.info) == 0
    assert int(out_f.info) == 0
    np.testing.assert_allclose(np.asarray(out_f.x), np.asarray(out_g.x), atol=1e-8, rtol=1e-8)


@pytest.mark.gpu
def test_gcro_cupy_harmonic_recycling_solves_toy_system(
    cupy_runtime: tuple[Any, Any],
) -> None:
    """Exercise several recycled cycles through the native CuPy path."""
    cupy, _ = cupy_runtime
    rng = np.random.default_rng(24)
    n = 12
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (0.5 + 0.0j) * np.eye(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    out = solvers.solve_linear_system(
        lambda x: cupy.asarray(A, dtype=cupy.complex128) @ x,
        b,
        method="gcro",
        backend="cupy",
        restart=5,
        gcro_recycle_dim=2,
        maxiter=100,
        rtol=1e-10,
        accum_dtype=np.complex128,
        show_progress=False,
    )

    assert int(out.info) == 0
    assert int(out.iterations) > 5
    metadata = out.block_metadata
    assert metadata is not None
    assert int(metadata["restart_cycles"]) >= 2
    assert int(metadata["recycle_dimension"]) == 2
    np.testing.assert_allclose(np.asarray(out.x), np.linalg.solve(A, b), atol=1e-8, rtol=1e-8)
    assert float(out.relative_residual) <= 1e-10

    proxy_only = solvers.solve_linear_system(
        lambda x: cupy.asarray(A, dtype=cupy.complex128) @ x,
        b,
        method="gcro",
        backend="cupy",
        restart=5,
        gcro_recycle_dim=2,
        maxiter=5,
        rtol=1e-10,
        accum_dtype=np.complex128,
        show_progress=False,
        compute_final_residual=False,
    )
    assert np.isnan(float(proxy_only.relative_residual))
    assert proxy_only.true_residual_history is not None
    assert proxy_only.true_residual_history.size == 1

    multi = solvers.solve_linear_system(
        lambda x: cupy.asarray(A, dtype=cupy.complex128) @ x,
        np.column_stack((b, 2.0 * b)),
        method="gcro",
        backend="cupy",
        restart=5,
        gcro_recycle_dim=2,
        maxiter=100,
        rtol=1e-10,
        accum_dtype=np.complex128,
        show_progress=False,
    )
    assert multi.rhs_count == 2
    multi_metadata = multi.block_metadata
    assert multi_metadata is not None
    assert len(multi_metadata["independent_rhs"]) == 2
    assert all("restart_cycles" in item for item in multi_metadata["independent_rhs"])


@pytest.mark.fake_gpu
def test_lgmres_cupy_matches_gmres_on_toy_system(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(22)
    n = 10
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (0.5 + 0j) * np.eye(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    out_g = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=10,
        maxiter=60,
        show_progress=False,
    )
    out_l = solvers.lgmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=10,
        maxiter=60,
        outer_k=2,
        show_progress=False,
    )
    assert int(out_g.info) == 0
    assert int(out_l.info) == 0
    np.testing.assert_allclose(np.asarray(out_l.x), np.asarray(out_g.x), atol=1e-8, rtol=1e-8)


@pytest.mark.fake_gpu
def test_bicgstab_cupy_matches_gmres_on_toy_system(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(23)
    n = 10
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (0.5 + 0j) * np.eye(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    out_g = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=10,
        maxiter=60,
        show_progress=False,
    )
    out_b = solvers.bicgstab_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        maxiter=80,
        show_progress=False,
    )
    assert int(out_g.info) == 0
    assert int(out_b.info) == 0
    np.testing.assert_allclose(np.asarray(out_b.x), np.asarray(out_g.x), atol=1e-8, rtol=1e-8)


@pytest.mark.fake_gpu
def test_bicgstab_cupy_separates_recursive_and_true_residual_histories(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.asarray([1.0 + 0.0j, -0.5 + 0.0j], dtype=np.complex128)
    out = solvers.bicgstab_cupy(
        lambda x: np.asarray(x),
        b,
        rtol=1e-12,
        maxiter=4,
        show_progress=False,
        compute_final_residual=True,
    )

    assert int(out.info) == 0
    assert out.residual_history is not None
    assert out.true_residual_history is not None
    assert len(np.asarray(out.residual_history)) >= 2
    assert len(np.asarray(out.true_residual_history)) >= 2
    assert out.block_metadata is not None
    assert out.block_metadata["residual_history_kind"] == "bicgstab_recursive_residual"
    # Identity reaches the s-step convergence branch: one recurrence matvec plus
    # one independent physical residual gate.
    assert out.block_metadata["operator_applications"] == 2


@pytest.mark.fake_gpu
def test_bicgstab_cupy_explicit_zero_warm_start_evaluates_initial_matvec(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.asarray([1.0 + 0.0j, -0.5 + 0.0j], dtype=np.complex128)
    out = solvers.bicgstab_cupy(
        lambda x: np.asarray(x),
        b,
        x0=np.zeros_like(b),
        rtol=1e-12,
        maxiter=4,
        show_progress=False,
    )

    assert int(out.info) == 0
    assert out.block_metadata is not None
    assert out.block_metadata["operator_applications"] == 3


@pytest.mark.fake_gpu
def test_bicgstab_cupy_breakdown_path_returns_failure_without_crash(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)

    out = solvers.bicgstab_cupy(
        lambda x: np.zeros_like(np.asarray(x)),
        b,
        rtol=1e-12,
        atol=0.0,
        maxiter=6,
        show_progress=False,
    )

    assert int(out.info) > 0
    assert int(out.iterations) >= 1
    assert np.isfinite(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_gmres_cupy_native_breakdown_path_returns_failure_without_crash(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)

    out = solvers.gmres_cupy(
        lambda x: np.zeros_like(np.asarray(x)),
        b,
        rtol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=6,
        show_progress=False,
    )

    assert int(out.info) > 0
    assert int(out.iterations) >= 1
    assert np.isfinite(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_gmres_cupy_native_zero_initial_guess_skips_extra_initial_matvec(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -2.0 + 0j, 0.5 + 0j], dtype=np.complex128)
    matvec_calls = 0

    def _a_mv(x: np.ndarray) -> np.ndarray:
        nonlocal matvec_calls
        matvec_calls += 1
        return np.asarray(x)

    out = solvers.gmres_cupy(
        _a_mv,
        b,
        x0=None,
        rtol=0.0,
        atol=0.0,
        restart=2,
        maxiter=1,
        show_progress=False,
    )
    assert int(out.info) == 0
    assert int(out.iterations) == 1
    # One A@v inside Arnoldi + one final true-residual check.
    assert matvec_calls == 2


def test_lsqr_scipy_uses_original_rhs_norm_for_continuation_accounting() -> None:
    matrix = np.asarray([[2.0 + 0.1j, 0.2 - 0.1j], [0.1 + 0.3j, 1.5 - 0.2j]], dtype=np.complex128)
    rhs = np.asarray([1.0 + 0.5j, -0.4 + 0.2j], dtype=np.complex128)
    x0 = np.asarray([0.1 - 0.2j, 0.3 + 0.1j], dtype=np.complex128)
    residual = rhs - matrix @ x0
    original_norm = float(np.linalg.norm(rhs))
    result = lsqr_scipy(
        lambda values: matrix @ values,
        lambda values: matrix.conj().T @ values,
        rhs,
        x0=x0,
        initial_residual=residual,
        rhs_norm=original_norm,
        rtol=1.0e-10,
        maxiter=20,
        show_progress=False,
    )
    assert result.info == 0
    assert result.converged_reason == "converged"
    assert result.relative_residual == pytest.approx(
        float(result.residual_norm) / original_norm, rel=1.0e-12
    )
    assert result.true_residual_history is not None
    assert result.true_residual_history.shape == (2,)
    assert result.block_metadata is not None
    assert result.block_metadata["rhs_norm"] == pytest.approx(original_norm)


def test_lsqr_scipy_marks_physical_tolerance_failure_after_scipy_stop() -> None:
    matrix = np.diag(np.asarray([1.0, 3.0], dtype=np.complex128))
    rhs = np.asarray([1.0, 1.0], dtype=np.complex128)
    result = lsqr_scipy(
        lambda values: matrix @ values,
        lambda values: matrix.conj().T @ values,
        rhs,
        rtol=1.0e-14,
        maxiter=1,
        show_progress=False,
    )
    assert result.info != 0
    assert result.converged_reason == "tolerance_not_met"
    assert result.true_residual_history is not None
    assert result.true_residual_history.shape == (1,)


@pytest.mark.fake_gpu
def test_gmres_cupy_native_clamps_restart_to_system_size(monkeypatch):
    cupy = _fake_cupy_numpy_backend()
    basis_shapes: list[tuple[int, ...]] = []
    orig_zeros = cupy.zeros
    orig_empty = cupy.empty

    def _zeros(shape, dtype=None):
        return orig_zeros(shape, dtype=dtype)

    def _empty(shape, dtype=None):
        if isinstance(shape, tuple):
            basis_shapes.append(tuple(int(v) for v in shape))
        return orig_empty(shape, dtype=dtype)

    cupy.zeros = _zeros
    cupy.empty = _empty
    monkeypatch.setattr(solvers, "import_cupy", lambda: (cupy, None))
    n = 4
    b = np.arange(1, n + 1, dtype=np.float64).astype(np.complex128)

    solvers.gmres_cupy(
        lambda x: np.asarray(x),
        b,
        rtol=1e-12,
        atol=0.0,
        restart=50,
        maxiter=9,
        show_progress=False,
    )
    # V has shape (cycle_steps + 1, n); with restart clamp and n=4 we expect (5, 4),
    # not an oversized (10, 4) allocation from restart=50/maxiter=9.
    assert (5, 4) in basis_shapes
    assert (10, 4) not in basis_shapes


@pytest.mark.fake_gpu
def test_gmres_cupy_releases_completed_basis_before_restart(monkeypatch):
    cupy = _fake_cupy_numpy_backend()
    original_empty = cupy.empty
    basis_refs: list[weakref.ReferenceType[np.ndarray]] = []
    live_basis_at_second_cycle: list[int] = []

    def tracked_empty(shape, dtype=None):
        shape_tuple = tuple(int(value) for value in shape) if isinstance(shape, tuple) else ()
        if shape_tuple == (2, 3):
            if basis_refs:
                live_basis_at_second_cycle.append(
                    sum(reference() is not None for reference in basis_refs)
                )
            array = original_empty(shape, dtype=dtype)
            basis_refs.append(weakref.ref(array))
            return array
        return original_empty(shape, dtype=dtype)

    cupy.empty = tracked_empty
    monkeypatch.setattr(solvers, "import_cupy", lambda: (cupy, None))
    diagonal = np.asarray([2.0, 3.0, 4.0], dtype=np.complex128)

    out = solvers.gmres_cupy(
        lambda x: diagonal * np.asarray(x),
        np.ones(3, dtype=np.complex128),
        rtol=0.0,
        atol=0.0,
        restart=1,
        maxiter=2,
        show_progress=False,
        compute_final_residual=False,
    )

    assert int(out.iterations) == 2
    assert live_basis_at_second_cycle == [0]


@pytest.mark.fake_gpu
def test_gmres_cupy_native_tracks_scipy_solution_quality_on_toy_system(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(9)
    n = 8
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (M + M.conj().T)
    A = (1.5 + 0.0j) * np.eye(n, dtype=np.complex128) + 0.05 * H
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    Aop = LinearOperator((n, n), matvec=lambda x: A @ np.asarray(x), dtype=np.complex128)
    x_ref, info_ref = scipy_gmres(
        Aop,
        b,
        rtol=1e-10,
        atol=0.0,
        restart=8,
        maxiter=40,
        callback=None,
        callback_type="legacy",
    )
    out_cupy = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=8,
        maxiter=40,
        show_progress=False,
    )

    assert int(info_ref) == 0
    assert int(out_cupy.info) == 0
    assert float(out_cupy.relative_residual) <= 1e-10
    np.testing.assert_allclose(np.asarray(out_cupy.x), np.asarray(x_ref), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("refine_policy", ["never", "ifneeded", "always"])
@pytest.mark.fake_gpu
def test_gmres_cupy_native_supports_cgs_refinement_policies(monkeypatch, refine_policy: str):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.0 + 0j, 2.0 + 0j]], dtype=np.complex128)
    x_true = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)
    b = A @ x_true

    out = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-10,
        atol=0.0,
        restart=2,
        maxiter=8,
        orthogonalization="cgs",
        cgs_refinement=cast(Literal["never", "ifneeded", "always"], refine_policy),
        show_progress=False,
    )

    assert int(out.info) == 0
    assert str(out.converged_reason) == "converged"
    np.testing.assert_allclose(np.asarray(out.x), x_true, atol=1e-9, rtol=1e-9)


@pytest.mark.fake_gpu
def test_gmres_cupy_monitor_channels_and_callbacks(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.array([[2.0 + 0j, 1.0 + 0j], [1.0 + 0j, 3.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)
    inner_hist: list[float] = []
    true_hist: list[float] = []

    out = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-12,
        atol=0.0,
        restart=2,
        maxiter=8,
        callback=inner_hist.append,
        callback_true=true_hist.append,
        monitor="both",
        show_progress=False,
    )

    assert int(out.iterations) == len(inner_hist)
    assert out.preconditioned_residual_history is not None
    assert out.true_residual_history is not None
    assert len(out.preconditioned_residual_history) == int(out.iterations)
    assert len(out.true_residual_history) >= 1
    # Backward-compatible primary history channel stays preconditioned.
    assert out.residual_history is not None
    np.testing.assert_allclose(
        np.asarray(out.residual_history, dtype=float),
        np.asarray(out.preconditioned_residual_history, dtype=float),
    )
    assert len(true_hist) == len(out.true_residual_history)

    out_true = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-12,
        atol=0.0,
        restart=2,
        maxiter=8,
        monitor="true",
        show_progress=False,
    )
    np.testing.assert_allclose(
        np.asarray(out_true.residual_history, dtype=float),
        np.asarray(out_true.true_residual_history, dtype=float),
    )


@pytest.mark.fake_gpu
def test_gmres_cupy_reports_nonconverged_reason(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.array([[2.0 + 0j, 1.0 + 0j], [1.0 + 0j, 3.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, -2.0 + 0j], dtype=np.complex128)

    out = solvers.gmres_cupy(
        lambda x: A @ np.asarray(x),
        b,
        rtol=1e-14,
        atol=0.0,
        restart=1,
        maxiter=1,
        show_progress=False,
    )
    assert int(out.info) > 0
    assert str(out.converged_reason) in {"maxiter_reached", "breakdown", "happy_breakdown"}


@pytest.mark.parametrize("solver_name", ["gmres_cupy", "fgmres_cupy", "lgmres_cupy"])
@pytest.mark.fake_gpu
def test_cupy_restart_solvers_reject_block_rhs_at_single_rhs_entrypoint(
    monkeypatch, solver_name: str
):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.eye(3, dtype=np.complex128)
    solver = getattr(solvers, solver_name)
    with pytest.raises(ValueError, match="expects a 1D RHS"):
        solver(lambda x: np.asarray(x), b, show_progress=False)


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_identity_shape_and_metadata(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    n, p = 7, 3
    rng = np.random.default_rng(1234)
    B = (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))).astype(np.complex128)

    out = solve_linear_system(
        lambda x: np.asarray(x),
        B,
        method="gmres",
        backend="cupy",
        rtol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=30,
        show_progress=False,
    )
    assert out.rhs_count == p
    assert str(out.method) == "gmres[cupy-block]"
    np.testing.assert_allclose(np.asarray(out.x), B, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(np.asarray(out.info, dtype=int), np.zeros((p,), dtype=int))
    assert out.block_metadata is not None
    assert int(out.block_metadata["batch_count"]) == 1

    b_vec = B[:, 0]
    out_vec = solve_linear_system(
        lambda x: np.asarray(x),
        b_vec,
        method="gmres",
        backend="cupy",
        rtol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=30,
        show_progress=False,
    )
    assert str(out_vec.method) == "gmres[cupy]"
    np.testing.assert_allclose(np.asarray(out_vec.x), b_vec, atol=1e-9, rtol=1e-9)


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_matches_direct_on_dense_system(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(4321)
    n, p = 10, 4
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (0.75 + 0j) * np.eye(n, dtype=np.complex128)
    B = rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))

    out = solve_linear_system(
        lambda x: A @ np.asarray(x),
        B,
        method="gmres",
        backend="cupy",
        rtol=1e-10,
        atol=0.0,
        restart=6,
        maxiter=60,
        show_progress=False,
    )
    x_ref = np.linalg.solve(A, B)
    np.testing.assert_allclose(np.asarray(out.x), x_ref, atol=1e-8, rtol=1e-8)
    assert np.all(np.asarray(out.info, dtype=int) == 0)


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_requires_block_operator(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(99)
    n, p = 8, 3
    diag = (1.0 + rng.random(n)) + 0j
    A = np.diag(diag.astype(np.complex128))
    B = (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))).astype(np.complex128)

    def _a_vec_only(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError("vector-only operator")
        return cast(np.ndarray, A @ arr)

    with pytest.raises(ValueError, match="vector-only operator"):
        solve_linear_system(
            _a_vec_only,
            B,
            method="gmres",
            backend="cupy",
            rtol=1e-10,
            atol=0.0,
            restart=4,
            maxiter=40,
            show_progress=False,
        )


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_deflation_and_batching(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(777)
    n = 9
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    B = np.column_stack([u, u, 2.0 * u, v, v + 1e-10 * u]).astype(np.complex128)

    out_full = solve_linear_system(
        lambda x: np.asarray(x),
        B,
        method="gmres",
        backend="cupy",
        rtol=1e-12,
        atol=0.0,
        restart=5,
        maxiter=30,
        gmres_block_deflation_tol=1e-8,
        show_progress=False,
    )
    out_batched = solve_linear_system(
        lambda x: np.asarray(x),
        B,
        method="gmres",
        backend="cupy",
        rtol=1e-12,
        atol=0.0,
        restart=5,
        maxiter=30,
        gmres_block_deflation_tol=1e-8,
        gmres_block_batch_size=2,
        show_progress=False,
    )

    np.testing.assert_allclose(np.asarray(out_full.x), B, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(out_batched.x), np.asarray(out_full.x), atol=1e-9, rtol=1e-9
    )
    assert out_full.block_metadata is not None
    assert any(bool(m["applied"]) for m in out_full.block_metadata["batches"])
    assert out_batched.block_metadata is not None
    assert int(out_batched.block_metadata["batch_count"]) == 3


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_enforces_per_rhs_tolerance(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))

    def _fake_native(*args, **kwargs):
        b_comp = np.asarray(args[1])
        p = int(b_comp.shape[1])
        return types.SimpleNamespace(
            x=np.zeros_like(b_comp),
            info=0,
            iterations=1,
            block_residual_norm=float(np.linalg.norm(b_comp)),
            block_relative_residual=1.0,
            residual_norms=np.linalg.norm(b_comp, axis=0),
            relative_residuals=np.ones((p,), dtype=float),
            converged_reason="converged",
            preconditioned_history=np.asarray([1.0], dtype=float),
            true_history=np.asarray([1.0], dtype=float),
            per_rhs_true_history=np.asarray([np.ones((p,), dtype=float)], dtype=float),
        )

    monkeypatch.setattr(solvers, "block_gmres_cupy_native", _fake_native)
    b = np.asarray(
        [
            [1.0 + 0.0j, 0.5 + 0.0j],
            [0.25 + 0.0j, -0.75 + 0.0j],
            [-0.2 + 0.0j, 0.1 + 0.0j],
        ],
        dtype=np.complex128,
    )
    out = solvers.gmres_cupy_block(
        lambda x: np.asarray(x),
        b,
        rtol=1e-6,
        atol=0.0,
        show_progress=False,
        compute_final_residual=True,
    )
    assert np.all(np.asarray(out.info, dtype=int) > 0)
    np.testing.assert_allclose(np.asarray(out.relative_residual, dtype=float), 1.0, atol=1e-12)
    assert np.all(np.asarray(out.converged_reason, dtype=object) == "tolerance_not_met")
    assert out.block_metadata is not None


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_callback_payload(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(222)
    n, p = 8, 2
    A = np.diag((1.0 + rng.random(n)).astype(np.complex128))
    B = rng.standard_normal((n, p)).astype(np.complex128)
    seen: list[solvers.BlockKrylovCallbackPayload] = []

    out = solvers.gmres_cupy_block(
        lambda x: A @ np.asarray(x),
        B,
        rtol=1e-12,
        atol=0.0,
        restart=4,
        maxiter=30,
        callback=seen.append,
        show_progress=False,
    )
    assert np.all(np.asarray(out.info, dtype=int) == 0)
    assert len(seen) >= 2
    assert any(p.stage == "inner" for p in seen)
    assert any(p.stage == "restart" for p in seen)
    for payload in seen:
        assert payload.batch_count >= 1
        assert payload.batch_index >= 0
        assert payload.iteration >= 0


@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_block_gmres_uses_incycle_true_gate(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    rng = np.random.default_rng(1)
    n, p = 8, 2
    m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = m.conj().T @ m + (0.5 + 0j) * np.eye(n, dtype=np.complex128)
    B = (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))).astype(np.complex128)
    seen: list[solvers.BlockKrylovCallbackPayload] = []

    out = solvers.gmres_cupy_block(
        lambda x: A @ np.asarray(x),
        B,
        preconditioner=lambda x: 1e-12 * np.asarray(x),
        rtol=1e-8,
        atol=0.0,
        restart=50,
        maxiter=100,
        callback=seen.append,
        show_progress=False,
    )
    assert np.all(np.asarray(out.info, dtype=int) == 0)
    rhs_target = 1e-8 * np.linalg.norm(B, axis=0)
    assert np.all(np.asarray(out.residual_norm, dtype=float) <= rhs_target)
    restart_events = [payload for payload in seen if payload.stage == "restart"]
    # With in-cycle true gating enabled, this setup should converge without
    # repeated restart-boundary checks caused by an over-optimistic proxy.
    assert len(restart_events) <= 2


def test_solve_linear_system_fgmres_backend_guard():
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="available only with backend='cupy'"):
        solve_linear_system(
            lambda x: np.asarray(x),
            b,
            method="fgmres",
            backend="numpy",
            show_progress=False,
        )


@pytest.mark.fake_gpu
def test_solve_linear_system_fgmres_cupy_smoke(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -1.5 + 0j], dtype=np.complex128)
    out = solve_linear_system(
        lambda x: np.asarray(x),
        b,
        method="fgmres",
        backend="cupy",
        restart=4,
        maxiter=10,
        rtol=1e-12,
        atol=0.0,
        show_progress=False,
    )
    assert int(out.info) == 0
    assert str(out.method) == "fgmres[cupy]"
    np.testing.assert_allclose(np.asarray(out.x), b, atol=1e-10, rtol=1e-10)


@pytest.mark.fake_gpu
def test_solve_linear_system_lgmres_cupy_smoke(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    b = np.array([1.0 + 0j, -1.5 + 0j], dtype=np.complex128)
    out = solve_linear_system(
        lambda x: np.asarray(x),
        b,
        method="lgmres",
        backend="cupy",
        restart=4,
        maxiter=10,
        lgmres_outer_k=2,
        rtol=1e-12,
        atol=0.0,
        show_progress=False,
    )
    assert int(out.info) == 0
    assert str(out.method) == "lgmres[cupy]"
    np.testing.assert_allclose(np.asarray(out.x), b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["gmres", "fgmres", "lgmres"])
@pytest.mark.fake_gpu
def test_solve_linear_system_cupy_restart_solvers_verify_true_residual_each_restart(
    monkeypatch, method: Literal["gmres", "fgmres", "lgmres"]
):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.asarray([[2.0 + 0.0j, 0.25 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)
    b = np.asarray([1.0 + 0.0j, -1.5 + 0.0j], dtype=np.complex128)
    calls = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, A @ np.asarray(x))

    if method == "lgmres":
        out = solve_linear_system(
            A_mv,
            b,
            method="lgmres",
            backend="cupy",
            restart=1,
            maxiter=2,
            rtol=1e-30,
            atol=0.0,
            show_progress=False,
            compute_final_residual=True,
            lgmres_outer_k=0,
            lgmres_store_outer_av=False,
        )
    else:
        out = solve_linear_system(
            A_mv,
            b,
            method=method,
            backend="cupy",
            restart=1,
            maxiter=2,
            rtol=1e-30,
            atol=0.0,
            show_progress=False,
            compute_final_residual=True,
        )
    assert calls == 4
    assert int(out.iterations) == 2
    assert np.isfinite(float(out.residual_norm))
    assert np.isfinite(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_solve_linear_system_lgmres_cupy_skip_final_residual_avoids_terminal_apply(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.asarray([[2.0 + 0.0j, 0.25 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex128)
    b = np.asarray([1.0 + 0.0j, -1.5 + 0.0j], dtype=np.complex128)
    calls = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, A @ np.asarray(x))

    out = solve_linear_system(
        A_mv,
        b,
        method="lgmres",
        backend="cupy",
        restart=10,
        maxiter=1,
        rtol=1e-12,
        atol=0.0,
        show_progress=False,
        compute_final_residual=False,
    )
    # No restart follows this one-step solve, so the terminal physical residual
    # is the one application that the flag is allowed to omit.
    assert calls == 1
    assert int(out.iterations) == 1
    assert np.isnan(float(out.residual_norm))
    assert np.isnan(float(out.relative_residual))


@pytest.mark.fake_gpu
def test_solve_linear_system_bicgstab_cupy_multi_rhs_smoke(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.array([[3.0 + 0j, 1.0 + 0j], [0.5 + 0j, 2.0 + 0j]], dtype=np.complex128)
    B = np.array([[1.0 + 0j, -0.5 + 0j], [2.0 + 0j, 1.5 + 0j]], dtype=np.complex128)
    out = solve_linear_system(
        lambda x: A @ np.asarray(x),
        B,
        method="bicgstab",
        backend="cupy",
        rtol=1e-11,
        atol=0.0,
        maxiter=40,
        show_progress=False,
    )
    assert out.rhs_count == 2
    assert np.all(np.asarray(out.info, dtype=int) == 0)
    np.testing.assert_allclose(A @ np.asarray(out.x), B, atol=1e-8, rtol=1e-8)


def test_independent_multi_rhs_reuses_finalized_child_diagnostics(monkeypatch):
    child_results = [
        solvers.LinearSolveResult(
            x=np.asarray([1.0, 2.0]),
            info=0,
            residual_norm=1.0e-7,
            relative_residual=2.0e-8,
            iterations=3,
            method="bicgstab",
            residual_history=np.asarray([0.5, 2.0e-8]),
            converged_reason="converged",
        ),
        solvers.LinearSolveResult(
            x=np.asarray([3.0, 4.0]),
            info=2,
            residual_norm=5.0e-4,
            relative_residual=6.0e-5,
            iterations=7,
            method="bicgstab",
            residual_history=np.asarray([0.7, 6.0e-5]),
            converged_reason="maxiter_reached",
        ),
    ]
    child_index = 0

    def fake_bicgstab(*args: Any, **kwargs: Any) -> solvers.LinearSolveResult:
        del args, kwargs
        nonlocal child_index
        result = child_results[child_index]
        child_index += 1
        return result

    parent_apply_count = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        del x
        nonlocal parent_apply_count
        parent_apply_count += 1
        raise AssertionError("finalized child diagnostics must not trigger another apply")

    monkeypatch.setattr(solvers, "bicgstab_scipy", fake_bicgstab)
    out = solve_linear_system(
        A_mv,
        np.ones((2, 2), dtype=np.complex128),
        method="bicgstab",
        backend="numpy",
        show_progress=False,
    )

    assert parent_apply_count == 0
    np.testing.assert_array_equal(out.x, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(out.info, [0, 2])
    np.testing.assert_array_equal(out.iterations, [3, 7])
    np.testing.assert_allclose(out.residual_norm, [1.0e-7, 5.0e-4])
    np.testing.assert_allclose(out.relative_residual, [2.0e-8, 6.0e-5])
    np.testing.assert_array_equal(out.converged_reason, ["converged", "maxiter_reached"])


@pytest.mark.fake_gpu
def test_independent_multi_rhs_retains_all_backend_columns(monkeypatch):
    cupy = _fake_cupy_numpy_backend()
    monkeypatch.setattr(solvers, "import_cupy", lambda: (cupy, None))

    def fake_bicgstab(A_mv: Any, b: np.ndarray, **kwargs: Any) -> solvers.LinearSolveResult:
        del A_mv, kwargs
        x = cupy.asarray(2.0 * np.asarray(b))
        solvers._record_backend_solution(x)
        return solvers.LinearSolveResult(
            x=np.asarray(x),
            info=0,
            residual_norm=0.0,
            relative_residual=0.0,
            iterations=1,
            method="bicgstab[cupy]",
            converged_reason="converged",
        )

    monkeypatch.setattr(solvers, "bicgstab_cupy", fake_bicgstab)
    rhs = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.complex128)
    with solvers._capture_backend_solution(enabled=True) as capture:
        out = solve_linear_system(
            lambda x: np.asarray(x),
            rhs,
            method="bicgstab",
            backend="cupy",
            show_progress=False,
        )

    assert capture is not None
    np.testing.assert_array_equal(out.x, 2.0 * rhs)
    np.testing.assert_array_equal(capture["x"], 2.0 * rhs)


@pytest.mark.fake_gpu
def test_solve_linear_system_bicgstab_cupy_skip_final_residual_avoids_extra_apply(monkeypatch):
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_fake_cupy_numpy_backend(), None))
    A = np.asarray(
        [
            [4.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 3.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    b = np.asarray([1.0 + 0.0j, -1.5 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)
    calls = 0

    def A_mv(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, A @ np.asarray(x))

    out = solve_linear_system(
        A_mv,
        b,
        method="bicgstab",
        backend="cupy",
        maxiter=1,
        rtol=1e-12,
        atol=0.0,
        show_progress=False,
        compute_final_residual=False,
    )
    assert calls == 2
    assert int(out.iterations) == 1
    assert np.isnan(float(out.residual_norm))
    assert np.isnan(float(out.relative_residual))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "residual_atol", "solution_rtol", "solution_atol"),
    [
        (np.complex64, 1e-6, 1e-6, 1e-6),
        (np.complex128, 1e-12, 1e-12, 1e-12),
    ],
)
def test_gmres_cupy_native_matches_builtin_cupy_on_real_device(
    dtype: np.dtype,
    residual_atol: float,
    solution_rtol: float,
    solution_atol: float,
    cupy_runtime: tuple[Any, Any],
) -> None:
    cupy, cupyx_sparse_linalg = cupy_runtime
    rng = np.random.default_rng(41)
    n = 80
    restart = 20
    maxiter = 40

    # Use a controlled Hermitian positive-definite system rather than a random
    # normal-equation matrix.  This keeps the test focused on the native CuPy
    # GMRES implementation and its iteration accounting instead of comparing
    # three libraries at a complex64 stagnation plateau.
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype)
    Q, _ = np.linalg.qr(Z.astype(np.complex128))
    eigs = np.linspace(1.0, 3.0, n, dtype=float)
    A = (Q @ np.diag(eigs) @ Q.conj().T).astype(dtype)
    b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
    A_gpu = cupy.asarray(A)
    b_gpu = cupy.asarray(b)
    Aop_cpu = LinearOperator((n, n), matvec=lambda v: A @ np.asarray(v, dtype=dtype), dtype=dtype)
    scipy_hist: list[float] = []

    def _scipy_cb(v: float) -> None:
        scipy_hist.append(float(v))

    x_scipy, info_scipy = scipy_gmres(
        Aop_cpu,
        b,
        rtol=0.0,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=_scipy_cb,
        callback_type="legacy",
    )
    native_hist: list[float] = []

    out_native = solvers.gmres_cupy(
        lambda x: A_gpu @ cupy.asarray(x),
        b,
        rtol=0.0,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=native_hist.append,
        show_progress=False,
    )

    Aop_gpu = cupyx_sparse_linalg.LinearOperator(
        (n, n),
        matvec=lambda v: A_gpu @ v,
        dtype=dtype,
    )
    builtin_restart_hist: list[float] = []

    x_builtin, info_builtin = cupyx_sparse_linalg.gmres(
        Aop_gpu,
        b_gpu,
        x0=cupy.zeros_like(b_gpu),
        M=None,
        rtol=0.0,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        callback=lambda v: builtin_restart_hist.append(float(v)),
        callback_type="pr_norm",
    )
    cupy.cuda.Stream.null.synchronize()

    x_builtin_np = cupy.asnumpy(x_builtin)
    rel_scipy = float(np.linalg.norm(A @ np.asarray(x_scipy) - b) / np.linalg.norm(b))
    rel_builtin = float(
        np.linalg.norm(A @ np.asarray(x_builtin_np, dtype=dtype) - b) / np.linalg.norm(b)
    )
    rel_native = float(out_native.relative_residual)
    residuals = np.asarray([rel_scipy, rel_builtin, rel_native], dtype=float)

    assert int(out_native.info) == int(info_builtin)
    assert int(info_scipy) == int(info_builtin)
    assert int(info_builtin) == maxiter
    assert int(out_native.iterations) == maxiter
    assert int(out_native.iterations) == len(native_hist)
    assert len(scipy_hist) == maxiter
    # CuPy's built-in GMRES reports restart-boundary residuals for pr_norm;
    # pyceles also records true residuals at the initial point and at restart
    # boundaries.  Compare external-cycle accounting, not unavailable CuPy
    # internal Arnoldi iterations.
    assert len(builtin_restart_hist) == maxiter // restart
    assert out_native.true_residual_history is not None
    assert len(out_native.true_residual_history) - 1 == len(builtin_restart_hist)
    assert np.all(np.isfinite(residuals))
    assert float(np.max(residuals)) <= residual_atol
    np.testing.assert_allclose(
        out_native.x,
        x_builtin_np,
        rtol=solution_rtol,
        atol=solution_atol,
    )


def test_solve_linear_system_preconditioner_hook_identity():
    n = 10
    rng = np.random.default_rng(123)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    out = solve_linear_system(
        lambda x: x.copy(),
        b,
        method="gmres",
        preconditioner=lambda x: np.asarray(x),
        rtol=1e-12,
        atol=0.0,
        show_progress=False,
    )
    np.testing.assert_allclose(out.x, b, atol=1e-12, rtol=1e-12)


@pytest.mark.hdf5
def test_gmres_warm_restart_roundtrip_from_h5(tmp_path):
    rng = np.random.default_rng(77)
    n = 80
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (5.0 + 0.0j) * np.eye(n, dtype=np.complex128)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def A_mv(x: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, A @ np.asarray(x))

    # Strict reference solve from scratch.
    strict_cold = solve_linear_system(
        A_mv,
        b,
        method="gmres",
        rtol=1e-8,
        atol=0.0,
        restart=40,
        maxiter=300,
        show_progress=False,
    )
    assert strict_cold.info == 0

    # Loose solve, persisted and reloaded as warm-start seed.
    loose = solve_linear_system(
        A_mv,
        b,
        method="gmres",
        rtol=1e-4,
        atol=0.0,
        restart=40,
        maxiter=300,
        show_progress=False,
    )
    assert loose.info == 0

    path = tmp_path / "warm_restart_seed.h5"
    save_solution_h5(path, coeffs=np.asarray(loose.x))
    warm_seed = np.asarray(load_solution_h5(path)["coeffs"]).reshape(-1)

    strict_warm = solve_linear_system(
        A_mv,
        b,
        method="gmres",
        x0=warm_seed,
        rtol=1e-8,
        atol=0.0,
        restart=40,
        maxiter=300,
        show_progress=False,
    )
    assert strict_warm.info == 0

    # Warm restart must preserve final strict solution quality.
    np.testing.assert_allclose(strict_warm.x, strict_cold.x, rtol=1e-7, atol=1e-7)
    # Warm restart should not require more iterations in this deterministic setup.
    assert int(strict_warm.iterations) <= int(strict_cold.iterations)


def test_apply_operator_falls_back_to_columnwise_vector_calls():
    calls: list[np.ndarray] = []

    def _vec_only(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError("vector-only operator")
        calls.append(arr.copy())
        return cast(np.ndarray, 2.0 * arr)

    x = np.arange(6, dtype=np.complex128).reshape(3, 2)
    out = solvers._apply_operator(_vec_only, x)
    np.testing.assert_allclose(out, 2.0 * x)
    assert len(calls) == 2


@pytest.mark.fake_gpu
def test_apply_operator_cupy_keeps_columnwise_inputs_on_backend():
    cupy = _fake_cupy_numpy_backend()
    calls: list[np.ndarray] = []

    def _vec_only(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError("vector-only operator")
        calls.append(arr.copy())
        return cast(np.ndarray, 3.0 * arr)

    x = np.arange(6, dtype=np.complex128).reshape(3, 2)
    out = solvers._apply_operator_cupy(_vec_only, x, cupy=cupy)
    np.testing.assert_allclose(out, 3.0 * x)
    assert len(calls) == 2


@pytest.mark.api_contract
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"b": np.zeros((2, 2, 2), dtype=np.complex128)}, "`b` must be 1D or 2D"),
        (
            {
                "b": np.ones((2,), dtype=np.complex128),
                "x0": np.zeros((2, 1, 1), dtype=np.complex128),
            },
            "`x0` must be 1D or 2D",
        ),
        (
            {"b": np.ones((2, 2), dtype=np.complex128), "x0": np.ones((2, 3), dtype=np.complex128)},
            "`x0` must match `b` shape",
        ),
        (
            {"b": np.ones((2,), dtype=np.complex128), "method": "gcrotmk", "backend": "cupy"},
            "currently supports only GMRES, FGMRES, BiCGSTAB, LGMRES, GCRO, LSQR, or direct solves",
        ),
        (
            {"b": np.ones((2,), dtype=np.complex128), "method": "auto"},
            "Unknown method",
        ),
    ],
)
def test_solve_linear_system_validates_public_dispatch_inputs(kwargs, match):
    b = kwargs.pop("b")
    with pytest.raises(ValueError, match=match):
        solve_linear_system(lambda x: np.asarray(x), b, show_progress=False, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"b": np.zeros((2, 1, 1), dtype=np.complex128)},
            "`b` must be 1D or 2D",
        ),
        (
            {"b": np.ones((2,), dtype=np.complex128), "A_dense": np.eye(3, dtype=np.complex128)},
            "A_dense must have shape \\(2,2\\)",
        ),
        (
            {
                "b": np.ones((2,), dtype=np.complex128),
                "A_factorized": (np.eye(3, dtype=np.complex128), np.array([0, 1], dtype=int)),
            },
            "LU matrix",
        ),
        (
            {
                "b": np.ones((2,), dtype=np.complex128),
                "A_factorized": (np.eye(2, dtype=np.complex128), np.array([0, 1, 2], dtype=int)),
            },
            "pivot vector",
        ),
    ],
)
def test_direct_dense_scipy_validates_public_input_shapes(kwargs, match):
    b = kwargs.pop("b")
    with pytest.raises(ValueError, match=match):
        direct_dense_scipy(lambda x: np.asarray(x), b, show_progress=False, **kwargs)
