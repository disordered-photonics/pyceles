import numpy as np

from pyceles.io.hdf5 import load_solution_h5, save_solution_h5
from pyceles.linear.solvers import (
    direct_dense_scipy,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    gcrotmk_scipy,
    gmres_scipy,
    lgmres_scipy,
    solve_linear_system,
)


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
