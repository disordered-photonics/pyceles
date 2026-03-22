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
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy

from .krylov_cupy import fgmres_cupy_native, gmres_cupy_native, lgmres_cupy_native


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
    preconditioned_residual_history: np.ndarray | None = None
    true_residual_history: np.ndarray | None = None
    converged_reason: str | np.ndarray | None = None


GmresResult = LinearSolveResult
DenseLUFactorization = tuple[object, object]


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
    backend: Literal["numpy", "cupy"] = "numpy",
    overwrite_input: bool = False,
) -> DenseLUFactorization:
    """Return LU factorization payload for repeated direct solves."""
    solve_dtype = np.dtype(dtype)
    A = np.asarray(A_dense, dtype=solve_dtype)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"`A_dense` must be a square 2D matrix. Got shape {A.shape}.")
    if backend == "cupy":
        cupy, _ = import_cupy()
        import cupyx.scipy.linalg

        A_gpu = cupy.asarray(A)
        # For the GPU direct path, the LU payload is the persistent object we
        # actually want to keep. Allowing cuSOLVER to overwrite the dense matrix
        # avoids carrying both A and LU in device memory when callers are done
        # with the unfactorized operator.
        return cupyx.scipy.linalg.lu_factor(
            A_gpu,
            overwrite_a=bool(overwrite_input),
            check_finite=True,
        )
    import scipy.linalg

    lu, piv = scipy.linalg.lu_factor(
        A,
        overwrite_a=bool(overwrite_input),
        check_finite=False,
    )
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
    converged_reason: str | None = None,
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
        converged_reason=str(converged_reason)
        if converged_reason is not None
        else ("converged" if int(info) == 0 else "maxiter_reached"),
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
    converged_reason: np.ndarray | None = None,
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
        converged_reason=np.asarray(converged_reason, dtype=object)
        if converged_reason is not None
        else np.where(np.asarray(info, dtype=int) == 0, "converged", "maxiter_reached"),
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


def gmres_cupy(
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
    callback_true: Optional[Callable[[float], None]] = None,
    monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    progress_residual: Literal["preconditioned", "true"] = "preconditioned",
    orthogonalization: Literal["mgs", "cgs"] = "mgs",
    cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    happy_breakdown_tol: float = 0.0,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> GmresResult:
    """Solve Ax=b via native CuPy restarted GMRES.

    The Arnoldi basis, Hessenberg system, and Givens updates stay device-side.
    `callback` receives inner preconditioned residual updates. `callback_true`
    receives true residual updates at restart boundaries.
    """

    cupy, _ = import_cupy()
    b_arr = np.asarray(b)
    n = int(b_arr.size)
    op_dtype = np.dtype(np.result_type(b_arr.dtype, np.complex64))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10

    monitor_mode = str(monitor).lower()
    if monitor_mode not in {"preconditioned", "true", "both"}:
        raise ValueError("`monitor` must be 'preconditioned', 'true', or 'both'.")
    progress_mode = str(progress_residual).lower()
    if progress_mode not in {"preconditioned", "true"}:
        raise ValueError("`progress_residual` must be 'preconditioned' or 'true'.")

    progress_update, progress_close, _ = _make_progress_tracker(
        "gmres[cupy]",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="pr_rel_res" if progress_mode == "preconditioned" else "true_rel_res",
        max_iters=maxiter_total,
    )

    def _inner_callback(pr_rel: float) -> None:
        if progress_mode == "preconditioned":
            progress_update(pr_rel)
        if callback is not None:
            callback(pr_rel)

    def _restart_callback(true_rel: float) -> None:
        if progress_mode == "true":
            progress_update(true_rel)
        if callback_true is not None:
            callback_true(true_rel)

    native_callback = (
        _inner_callback
        if (callback is not None or (show_progress and progress_mode == "preconditioned"))
        else None
    )
    native_restart_callback = (
        _restart_callback
        if (callback_true is not None or (show_progress and progress_mode == "true"))
        else None
    )
    native = gmres_cupy_native(
        A_mv,
        b,
        cupy=cupy,
        x0=x0,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter_total,
        operator_dtype=op_dtype,
        callback=native_callback,
        restart_callback=native_restart_callback,
        record_preconditioned_history=monitor_mode in {"preconditioned", "both"},
        orthogonalization=orthogonalization,
        cgs_refinement=cgs_refinement,
        happy_breakdown_tol=float(happy_breakdown_tol),
    )
    progress_close()

    x_np = asnumpy(native.x)
    if compute_final_residual:
        residual_norm = float(native.residual_norm)
        relative_residual = float(native.relative_residual)
    else:
        residual_norm = float("nan")
        relative_residual = float("nan")

    pre_hist = np.asarray(native.preconditioned_history, dtype=float)
    true_hist = np.asarray(native.true_history, dtype=float)
    if monitor_mode == "preconditioned":
        residual_history: np.ndarray | None = pre_hist
    elif monitor_mode == "true":
        residual_history = true_hist
    else:
        # Keep backward-compatible single-channel field for code that still
        # reads `residual_history`; primary channel remains preconditioned.
        residual_history = pre_hist

    return LinearSolveResult(
        x=x_np,
        info=int(native.info),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=int(native.iterations),
        method="gmres[cupy]",
        residual_history=residual_history,
        rhs_count=1,
        preconditioned_residual_history=pre_hist,
        true_residual_history=true_hist,
        converged_reason=str(native.converged_reason),
    )


def fgmres_cupy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[..., np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: Optional[int] = None,
    callback: Optional[Callable[[float], None]] = None,
    callback_true: Optional[Callable[[float], None]] = None,
    monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    progress_residual: Literal["preconditioned", "true"] = "preconditioned",
    orthogonalization: Literal["mgs", "cgs"] = "mgs",
    cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    happy_breakdown_tol: float = 0.0,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> GmresResult:
    """Solve Ax=b via native CuPy restarted flexible GMRES.

    FGMRES uses a separate preconditioned basis `Z` and supports variable
    preconditioners (including callables that optionally consume iteration
    state dictionaries).
    """

    cupy, _ = import_cupy()
    b_arr = np.asarray(b)
    n = int(b_arr.size)
    op_dtype = np.dtype(np.result_type(b_arr.dtype, np.complex64))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10

    monitor_mode = str(monitor).lower()
    if monitor_mode not in {"preconditioned", "true", "both"}:
        raise ValueError("`monitor` must be 'preconditioned', 'true', or 'both'.")
    progress_mode = str(progress_residual).lower()
    if progress_mode not in {"preconditioned", "true"}:
        raise ValueError("`progress_residual` must be 'preconditioned' or 'true'.")

    progress_update, progress_close, _ = _make_progress_tracker(
        "fgmres[cupy]",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="pr_rel_res" if progress_mode == "preconditioned" else "true_rel_res",
        max_iters=maxiter_total,
    )

    def _inner_callback(pr_rel: float) -> None:
        if progress_mode == "preconditioned":
            progress_update(pr_rel)
        if callback is not None:
            callback(pr_rel)

    def _restart_callback(true_rel: float) -> None:
        if progress_mode == "true":
            progress_update(true_rel)
        if callback_true is not None:
            callback_true(true_rel)

    native_callback = (
        _inner_callback
        if (callback is not None or (show_progress and progress_mode == "preconditioned"))
        else None
    )
    native_restart_callback = (
        _restart_callback
        if (callback_true is not None or (show_progress and progress_mode == "true"))
        else None
    )
    native = fgmres_cupy_native(
        A_mv,
        b,
        cupy=cupy,
        x0=x0,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter_total,
        operator_dtype=op_dtype,
        callback=native_callback,
        restart_callback=native_restart_callback,
        record_preconditioned_history=monitor_mode in {"preconditioned", "both"},
        orthogonalization=orthogonalization,
        cgs_refinement=cgs_refinement,
        happy_breakdown_tol=float(happy_breakdown_tol),
    )
    progress_close()

    x_np = asnumpy(native.x)
    if compute_final_residual:
        residual_norm = float(native.residual_norm)
        relative_residual = float(native.relative_residual)
    else:
        residual_norm = float("nan")
        relative_residual = float("nan")

    pre_hist = np.asarray(native.preconditioned_history, dtype=float)
    true_hist = np.asarray(native.true_history, dtype=float)
    if monitor_mode == "preconditioned":
        residual_history: np.ndarray | None = pre_hist
    elif monitor_mode == "true":
        residual_history = true_hist
    else:
        residual_history = pre_hist

    return LinearSolveResult(
        x=x_np,
        info=int(native.info),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=int(native.iterations),
        method="fgmres[cupy]",
        residual_history=residual_history,
        rhs_count=1,
        preconditioned_residual_history=pre_hist,
        true_residual_history=true_hist,
        converged_reason=str(native.converged_reason),
    )


def lgmres_cupy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    preconditioner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 30,
    maxiter: Optional[int] = None,
    outer_k: int = 3,
    store_outer_av: bool = True,
    callback: Optional[Callable[[float], None]] = None,
    callback_true: Optional[Callable[[float], None]] = None,
    monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    progress_residual: Literal["preconditioned", "true"] = "preconditioned",
    orthogonalization: Literal["mgs", "cgs"] = "mgs",
    cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    happy_breakdown_tol: float = 0.0,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> GmresResult:
    """Solve Ax=b via native CuPy restarted LGMRES.

    LGMRES reuses a small set of correction vectors across restart cycles,
    reducing restart-stagnation risk compared with plain restarted GMRES.
    Monitor/callback semantics mirror ``gmres_cupy``.
    """

    cupy, _ = import_cupy()
    b_arr = np.asarray(b)
    n = int(b_arr.size)
    op_dtype = np.dtype(np.result_type(b_arr.dtype, np.complex64))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10

    monitor_mode = str(monitor).lower()
    if monitor_mode not in {"preconditioned", "true", "both"}:
        raise ValueError("`monitor` must be 'preconditioned', 'true', or 'both'.")
    progress_mode = str(progress_residual).lower()
    if progress_mode not in {"preconditioned", "true"}:
        raise ValueError("`progress_residual` must be 'preconditioned' or 'true'.")

    progress_update, progress_close, _ = _make_progress_tracker(
        "lgmres[cupy]",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="pr_rel_res" if progress_mode == "preconditioned" else "true_rel_res",
        max_iters=maxiter_total,
    )

    def _inner_callback(pr_rel: float) -> None:
        if progress_mode == "preconditioned":
            progress_update(pr_rel)
        if callback is not None:
            callback(pr_rel)

    def _restart_callback(true_rel: float) -> None:
        if progress_mode == "true":
            progress_update(true_rel)
        if callback_true is not None:
            callback_true(true_rel)

    native_callback = (
        _inner_callback
        if (callback is not None or (show_progress and progress_mode == "preconditioned"))
        else None
    )
    native_restart_callback = (
        _restart_callback
        if (callback_true is not None or (show_progress and progress_mode == "true"))
        else None
    )
    native = lgmres_cupy_native(
        A_mv,
        b,
        cupy=cupy,
        x0=x0,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter_total,
        outer_k=int(outer_k),
        store_outer_av=bool(store_outer_av),
        operator_dtype=op_dtype,
        callback=native_callback,
        restart_callback=native_restart_callback,
        record_preconditioned_history=monitor_mode in {"preconditioned", "both"},
        orthogonalization=orthogonalization,
        cgs_refinement=cgs_refinement,
        happy_breakdown_tol=float(happy_breakdown_tol),
    )
    progress_close()

    x_np = asnumpy(native.x)
    if compute_final_residual:
        residual_norm = float(native.residual_norm)
        relative_residual = float(native.relative_residual)
    else:
        residual_norm = float("nan")
        relative_residual = float("nan")

    pre_hist = np.asarray(native.preconditioned_history, dtype=float)
    true_hist = np.asarray(native.true_history, dtype=float)
    if monitor_mode == "preconditioned":
        residual_history: np.ndarray | None = pre_hist
    elif monitor_mode == "true":
        residual_history = true_hist
    else:
        residual_history = pre_hist

    return LinearSolveResult(
        x=x_np,
        info=int(native.info),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=int(native.iterations),
        method="lgmres[cupy]",
        residual_history=residual_history,
        rhs_count=1,
        preconditioned_residual_history=pre_hist,
        true_residual_history=true_hist,
        converged_reason=str(native.converged_reason),
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


def direct_dense_cupy(
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
    """Dense direct solve on GPU via CuPy/cuSOLVER.

    The dense matrix may still be assembled on CPU today; this path uploads it,
    factorizes it once on device, and reuses the LU payload across repeated RHS.
    """
    cupy, _ = import_cupy()
    import cupyx.scipy.linalg

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
        lu_payload = A_factorized
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
        A_gpu = cupy.asarray(A_for_residual)
        lu_payload = cupyx.scipy.linalg.lu_factor(A_gpu, overwrite_a=False, check_finite=True)

    if show_progress:
        residual_mode = "on" if compute_final_residual else "off"
        print(
            "[solver] Direct dense solve [cupy]:"
            f" n={n} nrhs={nrhs} setup={setup_mode} final_residual_check={residual_mode}"
        )
    b_gpu = cupy.asarray(b_mat)
    t0 = time.perf_counter()
    x_gpu = cupyx.scipy.linalg.lu_solve(lu_payload, b_gpu, overwrite_b=False, check_finite=True)
    cupy.cuda.Stream.null.synchronize()
    if show_progress:
        dt = time.perf_counter() - t0
        print(f"[solver] Direct dense solve [cupy] completed in {dt:.3f} s")

    x_mat = asnumpy(x_gpu)
    residual_op = (lambda v: A_for_residual @ np.asarray(v)) if A_for_residual is not None else A_mv
    if squeezed:
        x = x_mat[:, 0]
        return _finalize_result(
            residual_op,
            b_mat[:, 0],
            x,
            info=0,
            iterations=1,
            method="direct[cupy]",
            compute_final_residual=compute_final_residual,
        )
    return _finalize_multi_result(
        residual_op,
        b_mat,
        x_mat,
        info=np.zeros((x_mat.shape[1],), dtype=int),
        iterations=np.ones((x_mat.shape[1],), dtype=int),
        method="direct[cupy]",
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
    preconditioner: Optional[Callable[..., np.ndarray]] = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: Optional[int] = None,
    gmres_monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    gmres_progress_residual: Literal["preconditioned", "true"] = "preconditioned",
    gmres_orthogonalization: Literal["mgs", "cgs"] = "mgs",
    gmres_cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    gmres_happy_breakdown_tol: float = 0.0,
    lgmres_outer_k: int = 3,
    lgmres_store_outer_av: bool = True,
    direct_max_n: int = 15000,
    dtype: npt.DTypeLike = np.complex128,
    backend: Literal["numpy", "cupy"] = "numpy",
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve Ax=b with selected method.

    Supported methods: `auto`, `gmres`, `fgmres`, `bicgstab`, `lgmres`, `gcrotmk`, `direct`.
    If `method` resolves to `direct`, an optional preassembled `A_dense` can be
    supplied to avoid expensive column-by-column assembly via `A_mv`, and an
    optional `A_factorized=(lu, piv)` payload can be supplied to reuse LU
    factorization across repeated direct solves on either NumPy or CuPy backends.

    Parameters
    ----------
    b:
        Right-hand side(s), shaped `(n,)` or `(n, nrhs)`.
    x0:
        Optional warm start, shaped like `b`.
    preconditioner:
        Optional callable approximating `M^{-1}` for iterative methods.
        It can accept vectors and may optionally accept batched `(n, nrhs)` inputs.
        For native CuPy FGMRES, a two-argument form ``preconditioner(v, state)``
        is also accepted, where ``state`` carries iteration indices.
    gmres_monitor, gmres_progress_residual:
        CuPy-native GMRES monitor channels. `gmres_monitor` controls which
        residual history is retained in the result, while
        `gmres_progress_residual` controls tqdm reporting when enabled.
    gmres_orthogonalization, gmres_cgs_refinement, gmres_happy_breakdown_tol:
        CuPy-native GMRES Arnoldi controls. Orthogonalization can be modified
        Gram-Schmidt (`mgs`) or classical Gram-Schmidt (`cgs`) with optional
        iterative refinement policy.
    lgmres_outer_k, lgmres_store_outer_av:
        CuPy-native LGMRES recycle controls. ``lgmres_outer_k`` is the number
        of correction directions retained across restart cycles; enabling
        ``lgmres_store_outer_av`` caches ``A @ v`` for those recycled vectors.
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
    backend_name = backend
    if m == "auto":
        m = "direct" if n <= int(direct_max_n) else "gmres"
    if backend_name == "cupy" and m not in {"gmres", "fgmres", "lgmres", "direct"}:
        raise ValueError(
            "The CuPy linear-solver backend currently supports only GMRES, FGMRES, LGMRES, or direct solves."
        )

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
        direct_impl = direct_dense_cupy if backend_name == "cupy" else direct_dense_scipy
        out = direct_impl(
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
                gmres_monitor=gmres_monitor,
                gmres_progress_residual=gmres_progress_residual,
                gmres_orthogonalization=gmres_orthogonalization,
                gmres_cgs_refinement=gmres_cgs_refinement,
                gmres_happy_breakdown_tol=gmres_happy_breakdown_tol,
                lgmres_outer_k=lgmres_outer_k,
                lgmres_store_outer_av=lgmres_store_outer_av,
                direct_max_n=direct_max_n,
                dtype=dtype,
                backend=backend_name,
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
        if backend_name == "cupy":
            return gmres_cupy(
                A_mv,
                b_vec,
                x0=x0_vec,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                monitor=gmres_monitor,
                progress_residual=gmres_progress_residual,
                orthogonalization=gmres_orthogonalization,
                cgs_refinement=gmres_cgs_refinement,
                happy_breakdown_tol=gmres_happy_breakdown_tol,
                show_progress=show_progress,
                compute_final_residual=compute_final_residual,
            )
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
    if m == "fgmres":
        if backend_name != "cupy":
            raise ValueError("`method='fgmres'` is currently available only with backend='cupy'.")
        return fgmres_cupy(
            A_mv,
            b_vec,
            x0=x0_vec,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            monitor=gmres_monitor,
            progress_residual=gmres_progress_residual,
            orthogonalization=gmres_orthogonalization,
            cgs_refinement=gmres_cgs_refinement,
            happy_breakdown_tol=gmres_happy_breakdown_tol,
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
        if backend_name == "cupy":
            return lgmres_cupy(
                A_mv,
                b_vec,
                x0=x0_vec,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                outer_k=lgmres_outer_k,
                store_outer_av=lgmres_store_outer_av,
                monitor=gmres_monitor,
                progress_residual=gmres_progress_residual,
                orthogonalization=gmres_orthogonalization,
                cgs_refinement=gmres_cgs_refinement,
                happy_breakdown_tol=gmres_happy_breakdown_tol,
                show_progress=show_progress,
                compute_final_residual=compute_final_residual,
            )
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
        f"Unknown method '{method}'. Use one of auto/gmres/fgmres/bicgstab/lgmres/gcrotmk/direct."
    )
