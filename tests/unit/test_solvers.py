import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres

from pyceles._optional import import_cupy
from pyceles.io.hdf5 import load_solution_h5, save_solution_h5
from pyceles.linear import solvers
from pyceles.linear.solvers import (
    direct_dense_scipy,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    gcrotmk_scipy,
    gmres_scipy,
    lgmres_scipy,
    solve_linear_system,
)


def _fake_cupy_numpy_backend():
    class _FakeCuPy:
        @staticmethod
        def asarray(x, dtype=None):
            return np.asarray(x, dtype=dtype)

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
        def abs(x):
            return np.abs(x)

        @staticmethod
        def sqrt(x):
            return np.sqrt(x)

        @staticmethod
        def conj(x):
            return np.conj(x)

        @staticmethod
        def vdot(x, y):
            return np.vdot(x, y)

        class linalg:
            norm = staticmethod(np.linalg.norm)
            solve = staticmethod(np.linalg.solve)

    return _FakeCuPy()


def _cupy_available() -> bool:
    try:
        cupy, _ = import_cupy()
    except RuntimeError:
        return False
    try:
        x = cupy.arange(1, dtype=cupy.float32)
        cupy.cuda.Stream.null.synchronize()
        return int(cupy.asnumpy(x)[0]) == 0
    except Exception:
        return False


def _configure_cupy_tempdir() -> None:
    tmp_root = Path.cwd() / "outputs" / "test_cupy_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    tempfile.tempdir = str(tmp_root)


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


def test_direct_dense_solve_identity():
    n = 8
    rng = np.random.default_rng(5)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    out = direct_dense_scipy(lambda x: x.copy(), b, max_n=16, show_progress=False)
    np.testing.assert_allclose(out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.method == "direct"
    assert out.relative_residual <= 1e-12


def test_direct_dense_uses_preassembled_matrix():
    A = np.array([[2.0 + 0j, 1.0 - 1.0j], [0.5 + 0.2j, 3.0 + 0j]], dtype=np.complex128)
    b = np.array([1.0 + 0j, -2.0 + 0.5j], dtype=np.complex128)
    out = direct_dense_scipy(lambda x: x.copy(), b, A_dense=A, max_n=16, show_progress=False)
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
        max_n=16,
        show_progress=False,
    )
    np.testing.assert_allclose(A @ out.x, b, atol=1e-12, rtol=1e-12)
    assert out.info == 0
    assert out.method == "direct"


def test_solve_linear_system_auto_picks_direct_for_small_n():
    b = np.array([1.0 + 0j, 2.0 + 0j])
    out = solve_linear_system(
        lambda x: x.copy(), b, method="auto", direct_max_n=4, show_progress=False
    )
    assert out.method == "direct"
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
        return A @ np.asarray(x)

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
        return A @ np.asarray(x)

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


def test_solve_linear_system_cupy_backend_rejects_bicgstab():
    b = np.array([1.0 + 0j, 2.0 + 0j])
    with pytest.raises(ValueError, match="supports only GMRES or direct"):
        solve_linear_system(
            lambda x: x,
            b,
            method="bicgstab",
            backend="cupy",
            show_progress=False,
        )


def test_factorize_dense_matrix_cupy_requests_inplace_overwrite(monkeypatch):
    calls: list[tuple[bool, bool]] = []

    class _FakeCuPy:
        @staticmethod
        def asarray(x, dtype=None):
            return np.asarray(x, dtype=dtype)

    fake_linalg = types.SimpleNamespace()

    def _lu_factor(a, overwrite_a=False, check_finite=True):
        calls.append((bool(overwrite_a), bool(check_finite)))
        return (np.asarray(a), np.array([0], dtype=int))

    fake_linalg.lu_factor = _lu_factor
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_linalg)
    cupyx_pkg = sys.modules.get("cupyx")
    if cupyx_pkg is not None:
        scipy_pkg = getattr(cupyx_pkg, "scipy", None)
        if scipy_pkg is not None:
            monkeypatch.setattr(scipy_pkg, "linalg", fake_linalg, raising=False)
    scipy_pkg = sys.modules.get("cupyx.scipy")
    if scipy_pkg is not None:
        monkeypatch.setattr(scipy_pkg, "linalg", fake_linalg, raising=False)
    monkeypatch.setattr(solvers, "import_cupy", lambda: (_FakeCuPy(), None))

    A = np.eye(2, dtype=np.complex64)
    solvers.factorize_dense_matrix(
        A,
        dtype=np.complex64,
        backend="cupy",
        overwrite_input=True,
    )
    assert calls == [(True, True)]


def test_gmres_cupy_reports_clear_import_failure(monkeypatch):
    def fail_import():
        raise RuntimeError("broken cuda path")

    monkeypatch.setattr(solvers, "import_cupy", fail_import)
    b = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex128)
    with pytest.raises(RuntimeError, match="broken cuda path"):
        solvers.gmres_cupy(lambda x: x, b, show_progress=False)


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


def test_gmres_cupy_native_clamps_restart_to_system_size(monkeypatch):
    cupy = _fake_cupy_numpy_backend()
    zero_shapes: list[tuple[int, ...]] = []
    orig_zeros = cupy.zeros

    def _zeros(shape, dtype=None):
        if isinstance(shape, tuple):
            zero_shapes.append(tuple(int(v) for v in shape))
        return orig_zeros(shape, dtype=dtype)

    cupy.zeros = _zeros  # type: ignore[method-assign]
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
    assert (5, 4) in zero_shapes
    assert (10, 4) not in zero_shapes


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


@pytest.mark.skipif(not _cupy_available(), reason="CuPy runtime unavailable")
@pytest.mark.parametrize(
    ("dtype",),
    [
        (np.complex64,),
        (np.complex128,),
    ],
)
def test_gmres_cupy_native_matches_builtin_cupy_on_real_device(
    dtype: np.dtype,
) -> None:
    prev_tmp = os.environ.get("TMP")
    prev_temp = os.environ.get("TEMP")
    prev_tempdir = tempfile.tempdir
    _configure_cupy_tempdir()
    try:
        cupy, cupyx_sparse_linalg = import_cupy()
        rng = np.random.default_rng(41)
        n = 80
        M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype)
        A = M.conj().T @ M + (0.5 + 0.0j) * np.eye(n, dtype=dtype)
        b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
        A_gpu = cupy.asarray(A)
        b_gpu = cupy.asarray(b)
        Aop_cpu = LinearOperator(
            (n, n), matvec=lambda v: A @ np.asarray(v, dtype=dtype), dtype=dtype
        )
        scipy_hist: list[float] = []

        def _scipy_cb(v: float) -> None:
            scipy_hist.append(float(v))

        x_scipy, info_scipy = scipy_gmres(
            Aop_cpu,
            b,
            rtol=0.0,
            atol=0.0,
            restart=20,
            maxiter=40,
            callback=_scipy_cb,
            callback_type="legacy",
        )
        native_hist: list[float] = []

        out_native = solvers.gmres_cupy(
            lambda x: A_gpu @ cupy.asarray(x),
            b,
            rtol=0.0,
            atol=0.0,
            restart=20,
            maxiter=40,
            callback=native_hist.append,
            show_progress=False,
        )

        Aop_gpu = cupyx_sparse_linalg.LinearOperator(
            (n, n),
            matvec=lambda v: A_gpu @ v,
            dtype=dtype,
        )
        x_builtin, info_builtin = cupyx_sparse_linalg.gmres(
            Aop_gpu,
            b_gpu,
            x0=cupy.zeros_like(b_gpu),
            M=None,
            rtol=0.0,
            atol=0.0,
            restart=20,
            maxiter=40,
            callback=None,
            callback_type=None,
        )
        cupy.cuda.Stream.null.synchronize()
        rel_scipy = float(np.linalg.norm(A @ np.asarray(x_scipy) - b) / np.linalg.norm(b))
        rel_builtin = float(cupy.linalg.norm(A_gpu @ x_builtin - b_gpu) / cupy.linalg.norm(b_gpu))
        rel_native = float(out_native.relative_residual)
        rel_diff_native_builtin = abs(rel_native - rel_builtin) / max(abs(rel_builtin), 1e-30)
        rel_diff_scipy_builtin = abs(rel_scipy - rel_builtin) / max(abs(rel_builtin), 1e-30)

        assert int(out_native.info) == int(info_builtin)
        assert int(info_scipy) == int(info_builtin)
        assert int(out_native.iterations) == len(native_hist)
        assert len(scipy_hist) == 40
        # Native GMRES should match built-in CuPy at least at the same residual
        # agreement level observed between SciPy legacy-inner and built-in CuPy.
        assert rel_diff_native_builtin <= 1.1 * rel_diff_scipy_builtin + 1e-12
    finally:
        if prev_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = prev_tmp
        if prev_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = prev_temp
        tempfile.tempdir = prev_tempdir


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


def test_gmres_warm_restart_roundtrip_from_h5(tmp_path):
    rng = np.random.default_rng(77)
    n = 80
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M.conj().T @ M + (5.0 + 0.0j) * np.eye(n, dtype=np.complex128)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def A_mv(x: np.ndarray) -> np.ndarray:
        return A @ np.asarray(x)

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
