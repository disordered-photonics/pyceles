"""Linear solvers for the many-sphere system.

Large systems are typically solved matrix-free, while a dense direct solve is
only practical for small systems. This module therefore provides:

- Krylov methods (`gmres`, `bicgstab`, `lgmres`, `gcrotmk`)
- optional dense direct solve for small systems
- a dispatcher (`solve_linear_system`) with `method='auto'`

Key requirements for development/debugging:
- determinism and correctness-first
- progress reporting
- reliable reporting of *true* final residuals
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm


@dataclass(frozen=True)
class LinearSolveResult:
    """Result bundle for linear solves of the many-sphere system."""

    x: np.ndarray
    info: int | np.ndarray
    residual_norm: float | np.ndarray
    relative_residual: float | np.ndarray
    iterations: int | np.ndarray
    method: str
    residual_history: np.ndarray | list[np.ndarray | None] | None = None
    rhs_count: int = 1


GmresResult = LinearSolveResult
DenseLUFactorization = tuple[np.ndarray, np.ndarray]


def estimate_dense_matrix_bytes(n: int, *, dtype: npt.DTypeLike = np.complex128) -> int:
    """Return bytes required to store an n-by-n dense matrix."""

    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * n * np.dtype(dtype).itemsize


def factorize_dense_matrix(
    A_dense: np.ndarray,
    *,
    dtype: npt.DTypeLike = np.complex128,
) -> DenseLUFactorization:
    """Return LU factorization payload for repeated direct solves."""
    import scipy.linalg

    solve_dtype = np.dtype(dtype)
    A = np.asarray(A_dense, dtype=solve_dtype)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"`A_dense` must be a square 2D matrix. Got shape {A.shape}.")
    lu, piv = scipy.linalg.lu_factor(A, overwrite_a=False, check_finite=False)
    return np.asarray(lu), np.asarray(piv)


def _apply_operator(op: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Apply a vector operator to vector or matrix inputs.

    If `op` does not natively support `(n, nrhs)` inputs, columns are applied
    independently.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        return np.asarray(op(x_arr))
    try:
        y = np.asarray(op(x_arr))
        if y.shape == x_arr.shape:
            return y
    except (TypeError, ValueError):
        pass
    cols = [np.asarray(op(x_arr[:, j])) for j in range(x_arr.shape[1])]
    return np.column_stack(cols)


def _make_linear_operator(A_mv: Callable[[np.ndarray], np.ndarray], n: int, dtype: np.dtype):
    """Wrap matrix-free system matvec as a SciPy `LinearOperator`."""
    from scipy.sparse.linalg import LinearOperator

    def mv(v):
        """SciPy-compatible matvec wrapper with controlled dtype and ownership."""
        # Copy output defensively to avoid aliasing issues with scipy wrappers.
        return np.asarray(_apply_operator(A_mv, np.asarray(v)), dtype=dtype).copy()

    return LinearOperator((n, n), matvec=mv, dtype=dtype)


def _make_preconditioner_operator(
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]],
    n: int,
    dtype: np.dtype,
):
    """Wrap an optional preconditioner apply callable as `LinearOperator`."""
    if preconditioner is None:
        return None
    from scipy.sparse.linalg import LinearOperator

    def mv(v):
        """SciPy-compatible preconditioner wrapper with controlled dtype."""
        return np.asarray(_apply_operator(preconditioner, np.asarray(v)), dtype=dtype).copy()

    return LinearOperator((n, n), matvec=mv, dtype=dtype)


def _finalize_result(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x: np.ndarray,
    *,
    info: int,
    iterations: int,
    method: str,
    residual_history: Optional[list[float]] = None,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Finalize single-RHS diagnostics using true residual `||Ax-b||/||b||`."""
    if compute_final_residual:
        # Compute final residual in complex128 for stable diagnostics even when
        # the iterative solve used a lower-precision operator dtype.
        x_ref = np.asarray(x, dtype=np.complex128)
        b_ref = np.asarray(b, dtype=np.complex128)
        residual = np.asarray(A_mv(x_ref), dtype=np.complex128) - b_ref
        residual_norm = float(np.linalg.norm(residual))
        b_norm = float(np.linalg.norm(b_ref))
        relative_residual = residual_norm / b_norm if b_norm > 0 else residual_norm
    else:
        residual_norm = float("nan")
        relative_residual = float("nan")
    return LinearSolveResult(
        x=np.asarray(x),
        info=int(info),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=int(iterations),
        method=str(method),
        residual_history=None
        if residual_history is None
        else np.asarray(residual_history, dtype=float),
        rhs_count=1,
    )


def _finalize_multi_result(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x: np.ndarray,
    *,
    info: np.ndarray,
    iterations: np.ndarray,
    method: str,
    residual_history: list[np.ndarray | None] | None = None,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Finalize multi-RHS diagnostics with per-column true residuals."""
    if compute_final_residual:
        x_ref = np.asarray(x, dtype=np.complex128)
        b_ref = np.asarray(b, dtype=np.complex128)
        residual = np.asarray(_apply_operator(A_mv, x_ref), dtype=np.complex128) - b_ref
        residual_norm = np.linalg.norm(residual, axis=0)
        b_norm = np.linalg.norm(b_ref, axis=0)
        relative_residual = np.divide(
            residual_norm,
            b_norm,
            out=np.asarray(residual_norm, dtype=float),
            where=b_norm > 0,
        )
    else:
        nrhs = int(np.asarray(x).shape[1])
        residual_norm = np.full((nrhs,), np.nan, dtype=float)
        relative_residual = np.full((nrhs,), np.nan, dtype=float)
    return LinearSolveResult(
        x=np.asarray(x),
        info=np.asarray(info, dtype=int),
        residual_norm=np.asarray(residual_norm, dtype=float),
        relative_residual=np.asarray(relative_residual, dtype=float),
        iterations=np.asarray(iterations, dtype=int),
        method=str(method),
        residual_history=residual_history,
        rhs_count=int(x.shape[1]),
    )


def _estimate_eta(
    history: list[float], *, target_rel: float, elapsed: float
) -> Optional[tuple[int, float]]:
    """Estimate time-to-target assuming multiplicative residual decay.

    For Krylov solvers, residuals often behave approximately as:
      r_{k+1} ~= q * r_k, with 0 < q < 1
    once the asymptotic regime is reached.
    """
    if target_rel <= 0 or elapsed <= 0 or len(history) < 2:
        return None

    # Use a short rolling window so ETA starts early and adapts quickly.
    r = np.maximum(np.asarray(history[-4:], dtype=float), 1e-300)
    current = float(r[-1])
    if (not np.isfinite(current)) or current <= 0:
        return None
    if current <= target_rel:
        return (0, 0.0)

    # Tail geometric decay factor estimate (robust to mild oscillations/noise).
    ratios = r[1:] / r[:-1]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return None
    q = float(np.exp(np.mean(np.log(ratios))))
    if (not np.isfinite(q)) or q <= 0:
        return None
    if q >= 1.0 - 1e-8:
        return None

    n_left = float(np.log(target_rel / current) / np.log(q))
    if n_left < 0:
        return (0, 0.0)

    sec_per_iter = elapsed / len(history)
    if sec_per_iter <= 0:
        return None
    # Progress happens in whole iterations; round up to avoid over-optimistic ETA.
    iters_left = int(np.ceil(n_left))
    return iters_left, float(iters_left * sec_per_iter)


def _make_progress_tracker(
    method: str,
    *,
    show_progress: bool,
    target_rel: float,
    residual_label: str = "rel_res",
    max_iters: int | None = None,
):
    """Create solver-progress callbacks for iterative methods."""
    history: list[float] = []
    if not show_progress:

        def _noop_update(_: float | None) -> None:
            return

        def _noop_close() -> None:
            return

        return _noop_update, _noop_close, history

    start = time.perf_counter()
    n_updates = 0
    # Start with unknown total. Once convergence-rate estimates become
    # available we switch to dynamic total = done + estimated_left.
    pbar = tqdm(total=None, desc=f"{method.upper():8s}", leave=True)

    def update(residual: float | None) -> None:
        """Record one progress sample and refresh tqdm output."""
        nonlocal n_updates
        n_updates += 1
        r = None if residual is None else float(residual)
        if r is not None and np.isfinite(r):
            history.append(r)
        elapsed = time.perf_counter() - start
        eta = (
            None
            if (r is None or (not np.isfinite(r)))
            else _estimate_eta(history, target_rel=target_rel, elapsed=elapsed)
        )
        if eta is not None:
            est_total = n_updates + int(eta[0])
            if max_iters is not None:
                est_total = min(est_total, int(max_iters))
            pbar.total = max(est_total, n_updates, 1)
        pbar.update(max(0, n_updates - int(pbar.n)))
        if r is not None and np.isfinite(r):
            pbar.set_postfix_str(f"{residual_label}={r:.3e}", refresh=True)

    def close() -> None:
        """Finalize progress display."""
        pbar.update(max(0, n_updates - int(pbar.n)))
        pbar.close()

    return update, close, history


def gmres_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: Optional[int] = None,
    callback: Optional[Callable[[float], None]] = None,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> GmresResult:
    """Solve Ax=b via SciPy GMRES using a matvec callable.

    Parameters
    ----------
    A_mv:
        Callable implementing y = A(x).
    b:
        Right-hand side.
    x0:
        Optional warm-start vector.
    rtol, atol:
        GMRES tolerances (SciPy semantics).
    restart, maxiter:
        GMRES restart and maximum iterations.
    callback:
        Optional callback receiving the residual norm each iteration.
    show_progress:
        If True, print a single-line progress update with residual norm.
    """

    from scipy.sparse.linalg import gmres

    b = np.asarray(b)
    n = b.size
    op_dtype = np.result_type(b.dtype, np.complex64)
    Aop = _make_linear_operator(A_mv, n, op_dtype)
    Mop = _make_preconditioner_operator(preconditioner, n, op_dtype)

    progress_update, progress_close, history = _make_progress_tracker(
        "gmres",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="pr_rel_res",
        max_iters=maxiter,
    )
    iterations = 0

    def _cb(res_norm):
        """GMRES callback tracking SciPy-provided preconditioned residual norm."""
        nonlocal iterations
        iterations += 1
        # SciPy GMRES callback_type="pr_norm" reports the (possibly
        # preconditioned) residual norm used internally by GMRES, which is
        # not the same quantity as the final true ||Ax-b||/||b|| diagnostics.
        progress_update(float(res_norm))
        if callback is not None:
            callback(float(res_norm))

    x, info = gmres(
        Aop,
        b,
        x0=x0,
        M=Mop,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        callback=_cb,
        callback_type="pr_norm",
    )

    progress_close()

    return _finalize_result(
        A_mv,
        b,
        x,
        info=info,
        iterations=iterations,
        method="gmres",
        residual_history=history,
        compute_final_residual=compute_final_residual,
    )


def bicgstab_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve one RHS using BiCGSTAB on the matrix-free scattering operator."""
    from scipy.sparse.linalg import bicgstab

    b = np.asarray(b)
    n = b.size
    op_dtype = np.result_type(b.dtype, np.complex64)
    Aop = _make_linear_operator(A_mv, n, op_dtype)
    Mop = _make_preconditioner_operator(preconditioner, n, op_dtype)
    b_norm = float(np.linalg.norm(b))

    progress_update, progress_close, history = _make_progress_tracker(
        "bicgstab",
        show_progress=show_progress,
        target_rel=float(rtol),
        max_iters=maxiter,
    )
    iterations = 0

    def _cb(xk):
        """BiCGSTAB callback estimating true relative residual from iterate."""
        nonlocal iterations
        iterations += 1
        xk_1d = np.asarray(xk).reshape(-1)
        rk = b - np.asarray(A_mv(xk_1d)).reshape(-1)
        rrel = float(np.linalg.norm(rk) / b_norm) if b_norm > 0 else float(np.linalg.norm(rk))
        progress_update(rrel)

    x, info = bicgstab(Aop, b, x0=x0, M=Mop, rtol=rtol, atol=atol, maxiter=maxiter, callback=_cb)
    progress_close()
    return _finalize_result(
        A_mv,
        b,
        x,
        info=info,
        iterations=iterations,
        method="bicgstab",
        residual_history=history,
        compute_final_residual=compute_final_residual,
    )


def lgmres_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve one RHS using LGMRES on the matrix-free scattering operator."""
    from scipy.sparse.linalg import lgmres

    b = np.asarray(b)
    n = b.size
    op_dtype = np.result_type(b.dtype, np.complex64)
    Aop = _make_linear_operator(A_mv, n, op_dtype)
    Mop = _make_preconditioner_operator(preconditioner, n, op_dtype)

    progress_update, progress_close, history = _make_progress_tracker(
        "lgmres",
        show_progress=show_progress,
        target_rel=float(rtol),
        max_iters=maxiter,
    )
    iterations = 0

    def _cb(xk):
        """LGMRES callback (residual unavailable; progress is iteration-based)."""
        nonlocal iterations
        iterations += 1
        # SciPy's lgmres callback is invoked at points where "true residual"
        # tracking can be stale; avoid reporting misleading values.
        progress_update(None)

    x, info = lgmres(Aop, b, x0=x0, M=Mop, rtol=rtol, atol=atol, maxiter=maxiter, callback=_cb)
    progress_close()
    return _finalize_result(
        A_mv,
        b,
        x,
        info=info,
        iterations=iterations,
        method="lgmres",
        residual_history=history if history else None,
        compute_final_residual=compute_final_residual,
    )


def gcrotmk_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve one RHS using GCROTMK on the matrix-free scattering operator."""
    from scipy.sparse.linalg import gcrotmk

    b = np.asarray(b)
    n = b.size
    op_dtype = np.result_type(b.dtype, np.complex64)
    Aop = _make_linear_operator(A_mv, n, op_dtype)
    Mop = _make_preconditioner_operator(preconditioner, n, op_dtype)

    progress_update, progress_close, history = _make_progress_tracker(
        "gcrotmk",
        show_progress=show_progress,
        target_rel=float(rtol),
        max_iters=maxiter,
    )
    iterations = 0

    def _cb(xk):
        """GCROTMK callback (residual unavailable; progress is iteration-based)."""
        nonlocal iterations
        iterations += 1
        # SciPy's gcrotmk callback is invoked at points where "true residual"
        # tracking can be stale; avoid reporting misleading values.
        progress_update(None)

    x, info = gcrotmk(Aop, b, x0=x0, M=Mop, rtol=rtol, atol=atol, maxiter=maxiter, callback=_cb)
    progress_close()
    return _finalize_result(
        A_mv,
        b,
        x,
        info=info,
        iterations=iterations,
        method="gcrotmk",
        residual_history=history if history else None,
        compute_final_residual=compute_final_residual,
    )


def direct_dense_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    A_dense: Optional[np.ndarray] = None,
    A_factorized: DenseLUFactorization | None = None,
    max_n: int = 15000,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Dense direct solve by explicit matrix assembly.

    This is intended only for small systems.

    Parameters
    ----------
    A_dense:
        Optional preassembled dense matrix.
    A_factorized:
        Optional LU factorization payload `(lu, piv)` for repeated solves on
        the same operator. If provided, factorization is not recomputed.

    Notes
    -----
    `max_n` limits the *system size* `n` (number of unknowns in `A`), not the
    number of RHS columns. For many-sphere simulations, `n = N_spheres * N_modes`.
    """

    import scipy.linalg

    solve_dtype = np.dtype(dtype)
    b_arr = np.asarray(b, dtype=solve_dtype)
    if b_arr.ndim == 1:
        b_mat = b_arr.reshape(-1, 1)
        squeezed = True
    elif b_arr.ndim == 2:
        b_mat = b_arr
        squeezed = False
    else:
        raise ValueError(f"`b` must be 1D or 2D. Got shape {b_arr.shape}.")

    n = b_mat.shape[0]
    nrhs = b_mat.shape[1]
    if n > int(max_n):
        raise ValueError(
            f"Direct dense solve disabled for n={n} (> max_n={max_n}). "
            "Use an iterative method or raise `max_n` explicitly."
        )

    A_for_residual: np.ndarray | None = None
    if A_dense is not None:
        A_for_residual = np.asarray(A_dense, dtype=solve_dtype)
        if A_for_residual.shape != (n, n):
            raise ValueError(f"A_dense must have shape ({n},{n}), got {A_for_residual.shape}.")

    setup_mode = "assemble+factorize"
    if A_factorized is not None:
        setup_mode = "reuse_lu"
        lu, piv = A_factorized
        lu_arr = np.asarray(lu, dtype=solve_dtype)
        piv_arr = np.asarray(piv, dtype=np.int32).reshape(-1)
        if lu_arr.shape != (n, n):
            raise ValueError(
                f"`A_factorized[0]` (LU matrix) must have shape ({n},{n}). Got {lu_arr.shape}."
            )
        if piv_arr.shape != (n,):
            raise ValueError(
                f"`A_factorized[1]` (pivot vector) must have shape ({n},). Got {piv_arr.shape}."
            )
    else:
        if A_for_residual is None:
            A = np.empty((n, n), dtype=solve_dtype)
            eye = np.eye(n, dtype=solve_dtype)
            col_iter = range(n)
            if show_progress:
                col_iter = tqdm(col_iter, desc="Assemble A (dense via matvec)")
            for j in col_iter:
                A[:, j] = np.asarray(A_mv(eye[:, j]), dtype=solve_dtype)
            A_for_residual = A
        else:
            setup_mode = "factorize_dense"
        lu_arr, piv_arr = scipy.linalg.lu_factor(
            A_for_residual, overwrite_a=False, check_finite=False
        )

    if show_progress:
        residual_mode = "on" if compute_final_residual else "off"
        print(
            "[solver] Direct dense solve:"
            f" n={n} nrhs={nrhs} setup={setup_mode} final_residual_check={residual_mode}"
        )
    t0 = time.perf_counter()
    x_mat = scipy.linalg.lu_solve(
        (lu_arr, piv_arr),
        b_mat,
        overwrite_b=False,
        check_finite=False,
    )
    if show_progress:
        dt = time.perf_counter() - t0
        print(f"[solver] Direct dense solve completed in {dt:.3f} s")
    residual_op = (lambda v: A_for_residual @ np.asarray(v)) if A_for_residual is not None else A_mv
    if squeezed:
        x = x_mat[:, 0]
        return _finalize_result(
            residual_op,
            b_mat[:, 0],
            x,
            info=0,
            iterations=1,
            method="direct",
            compute_final_residual=compute_final_residual,
        )
    return _finalize_multi_result(
        residual_op,
        b_mat,
        x_mat,
        info=np.zeros((x_mat.shape[1],), dtype=int),
        iterations=np.ones((x_mat.shape[1],), dtype=int),
        method="direct",
        residual_history=[None] * x_mat.shape[1],
        compute_final_residual=compute_final_residual,
    )


def solve_linear_system(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    method: str = "auto",
    A_dense: Optional[np.ndarray] = None,
    A_factorized: DenseLUFactorization | None = None,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: Optional[int] = None,
    direct_max_n: int = 15000,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve Ax=b with selected method.

    Supported methods: `auto`, `gmres`, `bicgstab`, `lgmres`, `gcrotmk`, `direct`.
    If `method` resolves to `direct`, an optional preassembled `A_dense` can be
    supplied to avoid expensive column-by-column assembly via `A_mv`, and an
    optional `A_factorized=(lu, piv)` payload can be supplied to reuse LU
    factorization across repeated direct solves.

    Parameters
    ----------
    b:
        Right-hand side(s), shaped `(n,)` or `(n, nrhs)`.
    x0:
        Optional warm start, shaped like `b`.
    preconditioner:
        Optional callable approximating `M^{-1}` for iterative methods.
        It can accept vectors and may optionally accept batched `(n, nrhs)` inputs.
    compute_final_residual:
        If `True`, compute and store final true residual diagnostics
        `||Ax-b||/||b||` after the solve.
    """

    b_arr = np.asarray(b, dtype=np.dtype(dtype))
    if b_arr.ndim == 1:
        b_mat = b_arr.reshape(-1, 1)
        squeezed = True
    elif b_arr.ndim == 2:
        b_mat = b_arr
        squeezed = False
    else:
        raise ValueError(f"`b` must be 1D or 2D. Got shape {b_arr.shape}.")

    n = b_mat.shape[0]
    nrhs = b_mat.shape[1]
    m = str(method).lower()
    if m == "auto":
        m = "direct" if n <= int(direct_max_n) else "gmres"

    x0_mat: np.ndarray | None = None
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=np.dtype(dtype))
        if x0_arr.ndim == 1:
            x0_mat = x0_arr.reshape(-1, 1)
        elif x0_arr.ndim == 2:
            x0_mat = x0_arr
        else:
            raise ValueError(f"`x0` must be 1D or 2D. Got shape {x0_arr.shape}.")
        if x0_mat.shape != b_mat.shape:
            raise ValueError(f"`x0` must match `b` shape {b_mat.shape}. Got {x0_mat.shape}.")

    if m == "direct":
        out = direct_dense_scipy(
            A_mv,
            b_mat if nrhs > 1 else b_mat[:, 0],
            A_dense=A_dense,
            A_factorized=A_factorized,
            max_n=direct_max_n,
            dtype=dtype,
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )
        if squeezed:
            return out
        return out

    if nrhs > 1:
        xs: list[np.ndarray] = []
        infos: list[int] = []
        iters: list[int] = []
        histories: list[np.ndarray | None] = []
        for j in range(nrhs):
            if show_progress:
                print(f"[solver] RHS {j + 1}/{nrhs}")
            x0_j = None if x0_mat is None else x0_mat[:, j]
            rj = solve_linear_system(
                A_mv,
                b_mat[:, j],
                method=m,
                A_dense=A_dense,
                A_factorized=A_factorized,
                x0=x0_j,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                direct_max_n=direct_max_n,
                dtype=dtype,
                show_progress=show_progress,
                compute_final_residual=compute_final_residual,
            )
            xs.append(np.asarray(rj.x).reshape(-1))
            infos.append(int(rj.info))
            iters.append(int(rj.iterations))
            if isinstance(rj.residual_history, np.ndarray) or rj.residual_history is None:
                histories.append(rj.residual_history)
            else:
                histories.append(None)
        return _finalize_multi_result(
            A_mv,
            b_mat,
            np.column_stack(xs),
            info=np.asarray(infos, dtype=int),
            iterations=np.asarray(iters, dtype=int),
            method=m,
            residual_history=histories,
            compute_final_residual=compute_final_residual,
        )

    b_vec = b_mat[:, 0]
    x0_vec = None if x0_mat is None else x0_mat[:, 0]

    if m == "gmres":
        return gmres_scipy(
            A_mv,
            b_vec,
            x0=x0_vec,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )
    if m == "bicgstab":
        return bicgstab_scipy(
            A_mv,
            b_vec,
            x0=x0_vec,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )
    if m == "lgmres":
        return lgmres_scipy(
            A_mv,
            b_vec,
            x0=x0_vec,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )
    if m == "gcrotmk":
        return gcrotmk_scipy(
            A_mv,
            b_vec,
            x0=x0_vec,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )
    raise ValueError(
        f"Unknown method '{method}'. Use one of auto/gmres/bicgstab/lgmres/gcrotmk/direct."
    )
