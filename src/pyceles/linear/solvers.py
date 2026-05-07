"""Linear solvers for the many-sphere system.

Large systems are typically solved matrix-free, while a dense direct solve is
only practical for small systems. This module therefore provides:

- Krylov methods (`gmres`, `bicgstab`, `lgmres`, `gcrotmk`)
- optional dense direct solve for small systems
- native CuPy block-GMRES for multi-RHS iterative solves
- a dispatcher (`solve_linear_system`) with `method='auto'`

Key requirements for development/debugging:
- determinism and correctness-first
- progress reporting
- reliable reporting of requested true-residual diagnostics
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy

from .krylov_cupy import (
    bicgstab_cupy_native,
    block_gmres_cupy_native,
    fgmres_cupy_native,
    gmres_cupy_native,
    lgmres_cupy_native,
)


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
    block_residual_history: np.ndarray | list[np.ndarray] | None = None
    stopping_rule: str | None = None
    block_metadata: dict[str, Any] | None = None


GmresResult = LinearSolveResult
DenseLUFactorization = tuple[object, object]
_BACKEND_SOLUTION_CAPTURE: ContextVar[dict[str, Any] | None] = ContextVar(
    "_BACKEND_SOLUTION_CAPTURE", default=None
)


def _start_backend_solution_capture() -> tuple[Token[dict[str, Any] | None], dict[str, Any]]:
    """Start capturing backend-native solver output for internal postprocess handoff."""
    payload: dict[str, Any] = {}
    token = _BACKEND_SOLUTION_CAPTURE.set(payload)
    return token, payload


def _finish_backend_solution_capture(token: Token[dict[str, Any] | None]) -> None:
    """Stop capturing backend-native solver output."""
    _BACKEND_SOLUTION_CAPTURE.reset(token)


def _record_backend_solution(x: Any) -> None:
    payload = _BACKEND_SOLUTION_CAPTURE.get()
    if payload is not None:
        payload["x"] = x


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
        return cast(
            DenseLUFactorization,
            cupyx.scipy.linalg.lu_factor(
                A_gpu,
                overwrite_a=bool(overwrite_input),
                check_finite=True,
            ),
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
    preconditioner: Callable[[np.ndarray], np.ndarray] | None,
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
    residual_history: list[float] | None = None,
    compute_final_residual: bool = True,
    converged_reason: str | None = None,
) -> LinearSolveResult:
    """Finalize single-RHS diagnostics, computing `||Ax-b||/||b||` only when requested."""
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
    """Finalize multi-RHS diagnostics with per-column true residuals when requested."""
    if compute_final_residual:
        x_ref = np.asarray(x, dtype=np.complex128)
        b_ref = np.asarray(b, dtype=np.complex128)
        residual = np.asarray(_apply_operator(A_mv, x_ref), dtype=np.complex128) - b_ref
        residual_norm = np.linalg.norm(residual, axis=0)
        b_norm = np.linalg.norm(b_ref, axis=0)
        relative_residual = np.asarray(residual_norm, dtype=float).copy()
        np.divide(residual_norm, b_norm, out=relative_residual, where=b_norm > 0)
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
) -> tuple[int, float] | None:
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


def _prepare_cupy_restarted_solver(
    method: str,
    b: np.ndarray,
    *,
    rtol: float,
    maxiter: int | None,
    show_progress: bool,
    monitor: Literal["preconditioned", "true", "both"],
    progress_residual: Literal["preconditioned", "true"],
    callback: Callable[[float], None] | None,
    callback_true: Callable[[float], None] | None,
) -> tuple[
    Any,
    np.dtype[Any],
    int,
    Literal["preconditioned", "true", "both"],
    Callable[[float], None] | None,
    Callable[[float], None] | None,
    Callable[[], None],
]:
    """Prepare shared single-RHS CuPy restarted-Krylov wrapper plumbing."""
    b_arr = np.asarray(b)
    if b_arr.ndim != 1:
        raise ValueError(
            f"`{method}_cupy` expects a 1D RHS. For block solves use "
            "`solve_linear_system(..., method='gmres', backend='cupy')` with a 2D RHS."
        )

    cupy, _ = import_cupy()
    n = int(b_arr.size)
    op_dtype = np.dtype(np.result_type(b_arr.dtype, np.complex64))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10

    monitor_mode = cast(
        Literal["preconditioned", "true", "both"],
        str(monitor).lower(),
    )
    if monitor_mode not in {"preconditioned", "true", "both"}:
        raise ValueError("`monitor` must be 'preconditioned', 'true', or 'both'.")
    progress_mode = cast(
        Literal["preconditioned", "true"],
        str(progress_residual).lower(),
    )
    if progress_mode not in {"preconditioned", "true"}:
        raise ValueError("`progress_residual` must be 'preconditioned' or 'true'.")

    progress_update, progress_close, _ = _make_progress_tracker(
        f"{method}[cupy]",
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
    return (
        cupy,
        op_dtype,
        maxiter_total,
        monitor_mode,
        native_callback,
        native_restart_callback,
        progress_close,
    )


def _finalize_cupy_restarted_result(
    method: str,
    native: Any,
    *,
    monitor_mode: Literal["preconditioned", "true", "both"],
    compute_final_residual: bool,
) -> LinearSolveResult:
    """Build the common result payload for single-RHS CuPy restarted solvers."""
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
        method=f"{method}[cupy]",
        residual_history=residual_history,
        rhs_count=1,
        preconditioned_residual_history=pre_hist,
        true_residual_history=true_hist,
        converged_reason=str(native.converged_reason),
    )


@dataclass(frozen=True)
class BlockKrylovCallbackPayload:
    """Progress payload for block Krylov callbacks."""

    stage: Literal["inner", "restart"]
    iteration: int
    batch_index: int
    batch_count: int
    block_relative_residual: float
    per_rhs_relative_residual: np.ndarray | None = None


def _deflate_rhs_block_cupy(
    b_batch_gpu: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
    deflation_tol: float | None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    b_mat = cupy.asarray(b_batch_gpu, dtype=accum_dtype)
    p = int(b_mat.shape[1])
    if p <= 1 or deflation_tol is None or float(deflation_tol) <= 0.0:
        return (
            b_mat,
            None,
            {
                "enabled": bool(deflation_tol is not None and float(deflation_tol) > 0.0),
                "applied": False,
                "original_rhs": int(p),
                "effective_rhs": int(p),
                "deflation_tol": None if deflation_tol is None else float(deflation_tol),
            },
        )
    gram = b_mat.conj().T @ b_mat
    evals, evecs = cupy.linalg.eigh(gram)
    order = cupy.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    max_eval = float(cupy.abs(evals[0])) if p > 0 else 0.0
    cutoff = (float(deflation_tol) ** 2) * max_eval if max_eval > 0 else float("inf")
    keep = cupy.abs(evals) > cutoff
    rank = int(cupy.count_nonzero(keep))
    if rank <= 0:
        basis = cupy.asarray(evecs[:, :0], dtype=accum_dtype)
        compressed = b_mat[:, :0]
    else:
        basis = cupy.asarray(evecs[:, :rank], dtype=accum_dtype)
        compressed = b_mat @ basis
    return (
        compressed,
        basis,
        {
            "enabled": True,
            "applied": rank < p,
            "original_rhs": int(p),
            "effective_rhs": int(rank),
            "deflation_tol": float(deflation_tol),
        },
    )


def gmres_cupy_block(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 30,
    maxiter: int | None = None,
    block_batch_size: int | None = None,
    deflation_tol: float | None = None,
    callback: Callable[[BlockKrylovCallbackPayload], None] | None = None,
    monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    progress_residual: Literal["preconditioned", "true"] = "true",
    reorthogonalize: bool = True,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve ``A X = B`` with native CuPy block-GMRES.

    This path is active for CuPy GMRES when ``B`` is 2D. Convergence requires
    every RHS column to satisfy ``||r_j|| <= max(atol, rtol * ||b_j||)``,
    while block Frobenius residuals remain available as aggregate diagnostics.

    Operator and preconditioner callables may expose either 2D `(n, nrhs)` or
    legacy 1D `(n,)` interfaces. Legacy callables are adapted column-wise.
    The returned ``LinearSolveResult.block_metadata`` includes
    ``operator_block_adapter_used`` and ``preconditioner_block_adapter_used``.
    """
    cupy, _ = import_cupy()
    b_mat = np.asarray(b)
    if b_mat.ndim != 2:
        raise ValueError(f"`b` must be 2D for block-GMRES. Got shape {b_mat.shape}.")
    n, nrhs = int(b_mat.shape[0]), int(b_mat.shape[1])
    if nrhs < 1:
        raise ValueError("`b` must contain at least one RHS column for block-GMRES.")
    op_dtype = np.dtype(np.result_type(b_mat.dtype, np.complex64))
    x0_mat = None if x0 is None else np.asarray(x0, dtype=op_dtype)
    if x0_mat is not None and x0_mat.shape != b_mat.shape:
        raise ValueError(f"`x0` must match `b` shape {b_mat.shape}. Got {x0_mat.shape}.")
    monitor_mode = str(monitor).lower()
    if monitor_mode not in {"preconditioned", "true", "both"}:
        raise ValueError("`monitor` must be 'preconditioned', 'true', or 'both'.")
    progress_mode = str(progress_residual).lower()
    if progress_mode not in {"preconditioned", "true"}:
        raise ValueError("`progress_residual` must be 'preconditioned' or 'true'.")
    if block_batch_size is None:
        batch_size = int(nrhs)
    else:
        batch_size = max(1, min(int(block_batch_size), int(nrhs)))

    batch_ranges = [
        (start, min(start + batch_size, nrhs)) for start in range(0, int(nrhs), int(batch_size))
    ]
    maxiter_total = int(maxiter) if maxiter is not None else n * 10
    progress_update, progress_close, _ = _make_progress_tracker(
        "gmres[cupy-block]",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="pr_block_rel_res"
        if progress_mode == "preconditioned"
        else "true_block_rel_res",
        max_iters=maxiter_total * max(1, len(batch_ranges)),
    )

    x_out = np.zeros((n, nrhs), dtype=op_dtype)
    rhs_iterations = np.zeros((nrhs,), dtype=int)
    rhs_info = np.zeros((nrhs,), dtype=int)
    rhs_reason = np.full((nrhs,), "", dtype=object)
    batch_meta: list[dict[str, Any]] = []
    precond_hist_all: list[np.ndarray] = []
    true_hist_all: list[np.ndarray] = []

    for bidx, (start, stop) in enumerate(batch_ranges):
        b_slice = np.asarray(b_mat[:, start:stop], dtype=op_dtype)
        x0_slice = None if x0_mat is None else np.asarray(x0_mat[:, start:stop], dtype=op_dtype)
        b_gpu = cupy.asarray(b_slice, dtype=op_dtype)
        b_comp_gpu, basis_gpu, dmeta = _deflate_rhs_block_cupy(
            b_gpu,
            cupy=cupy,
            accum_dtype=np.dtype(np.result_type(op_dtype, np.complex128)),
            deflation_tol=deflation_tol,
        )
        eff_rhs = int(b_comp_gpu.shape[1])
        if eff_rhs == 0:
            x_slice = np.zeros_like(b_slice, dtype=op_dtype)
            rhs_info[start:stop] = 0
            rhs_iterations[start:stop] = 0
            rhs_reason[start:stop] = "converged"
            batch_meta.append(
                {
                    "batch_index": int(bidx),
                    "batch_start": int(start),
                    "batch_stop": int(stop),
                    **dmeta,
                }
            )
            continue
        x0_comp = None
        if x0_slice is not None:
            x0_gpu = cupy.asarray(x0_slice, dtype=op_dtype)
            x0_comp = x0_gpu if basis_gpu is None else x0_gpu @ basis_gpu

        batch_index = int(bidx)
        batch_count = len(batch_ranges)

        def _inner_cb(
            payload: dict[str, Any],
            *,
            _batch_index: int = batch_index,
            _batch_count: int = batch_count,
        ) -> None:
            r = float(payload["block_relative_residual"])
            if progress_mode == "preconditioned":
                progress_update(r)
            if callback is not None:
                callback(
                    BlockKrylovCallbackPayload(
                        stage="inner",
                        iteration=int(payload["iteration"]),
                        batch_index=_batch_index,
                        batch_count=_batch_count,
                        block_relative_residual=r,
                        per_rhs_relative_residual=None,
                    )
                )

        def _restart_cb(
            payload: dict[str, Any],
            *,
            _batch_index: int = batch_index,
            _batch_count: int = batch_count,
        ) -> None:
            r = float(payload["block_relative_residual"])
            if progress_mode == "true":
                progress_update(r)
            if callback is not None:
                per_rhs = payload.get("per_rhs_relative_residual")
                callback(
                    BlockKrylovCallbackPayload(
                        stage="restart",
                        iteration=int(payload["iteration"]),
                        batch_index=_batch_index,
                        batch_count=_batch_count,
                        block_relative_residual=r,
                        per_rhs_relative_residual=None
                        if per_rhs is None
                        else np.asarray(per_rhs, dtype=float),
                    )
                )

        native = block_gmres_cupy_native(
            A_mv,
            b_comp_gpu,
            cupy=cupy,
            x0=x0_comp,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter_total,
            operator_dtype=op_dtype,
            callback=_inner_cb
            if (show_progress and progress_mode == "preconditioned") or callback
            else None,
            restart_callback=_restart_cb
            if (show_progress and progress_mode == "true") or callback
            else None,
            record_preconditioned_history=monitor_mode in {"preconditioned", "both"},
            reorthogonalize=bool(reorthogonalize),
        )
        precond_hist_all.append(np.asarray(native.preconditioned_history, dtype=float))
        true_hist_all.append(np.asarray(native.true_history, dtype=float))
        x_comp_gpu = cupy.asarray(native.x, dtype=op_dtype)
        x_batch_gpu = (
            x_comp_gpu
            if basis_gpu is None
            else x_comp_gpu @ cupy.asarray(basis_gpu, dtype=op_dtype).conj().T
        )
        x_slice = np.asarray(asnumpy(x_batch_gpu), dtype=op_dtype)
        x_out[:, start:stop] = x_slice
        rhs_iterations[start:stop] = int(native.iterations)
        rhs_info[start:stop] = int(native.info)
        rhs_reason[start:stop] = str(native.converged_reason)
        batch_meta.append(
            {
                "batch_index": int(bidx),
                "batch_start": int(start),
                "batch_stop": int(stop),
                "iterations": int(native.iterations),
                "info": int(native.info),
                "converged_reason": str(native.converged_reason),
                "operator_supports_block": bool(native.operator_supports_block),
                "preconditioner_supports_block": bool(native.preconditioner_supports_block),
                **dmeta,
            }
        )

    progress_close()

    if compute_final_residual:
        residual = np.asarray(_apply_operator(A_mv, x_out), dtype=np.complex128) - np.asarray(
            b_mat, dtype=np.complex128
        )
        residual_norm = np.linalg.norm(residual, axis=0)
        b_norm = np.linalg.norm(np.asarray(b_mat, dtype=np.complex128), axis=0)
        relative_residual = np.asarray(residual_norm, dtype=float).copy()
        np.divide(residual_norm, b_norm, out=relative_residual, where=b_norm > 0)
        block_residual = float(np.linalg.norm(residual))
        b_frob = float(np.linalg.norm(np.asarray(b_mat, dtype=np.complex128)))
        block_relative = block_residual / b_frob if b_frob > 0 else block_residual
        rhs_target_abs = np.maximum(float(atol), float(rtol) * b_norm)
        rhs_converged = np.asarray(residual_norm <= rhs_target_abs, dtype=bool)
        rhs_info = np.where(
            rhs_converged,
            0,
            np.where(
                np.asarray(rhs_info, dtype=int) == 0,
                int(maxiter_total),
                np.asarray(rhs_info, dtype=int),
            ),
        )
        rhs_reason = np.where(
            rhs_converged,
            "converged",
            np.where(
                np.asarray(rhs_reason, dtype=object) == "converged",
                "tolerance_not_met",
                np.asarray(rhs_reason, dtype=object),
            ),
        )
    else:
        residual_norm = np.full((nrhs,), np.nan, dtype=float)
        relative_residual = np.full((nrhs,), np.nan, dtype=float)
        block_residual = float("nan")
        block_relative = float("nan")
        rhs_target_abs = np.full((nrhs,), np.nan, dtype=float)

    pre_hist = (
        np.concatenate([h for h in precond_hist_all if h.size > 0])
        if any(h.size > 0 for h in precond_hist_all)
        else np.asarray([], dtype=float)
    )
    true_hist = (
        np.concatenate([h for h in true_hist_all if h.size > 0])
        if any(h.size > 0 for h in true_hist_all)
        else np.asarray([], dtype=float)
    )
    if monitor_mode == "preconditioned":
        residual_history: np.ndarray | None = pre_hist
    elif monitor_mode == "true":
        residual_history = true_hist
    else:
        residual_history = pre_hist

    return LinearSolveResult(
        x=np.asarray(x_out),
        info=np.asarray(rhs_info, dtype=int),
        residual_norm=np.asarray(residual_norm, dtype=float),
        relative_residual=np.asarray(relative_residual, dtype=float),
        iterations=np.asarray(rhs_iterations, dtype=int),
        method="gmres[cupy-block]",
        residual_history=residual_history,
        rhs_count=int(nrhs),
        preconditioned_residual_history=pre_hist,
        true_residual_history=true_hist,
        converged_reason=np.asarray(rhs_reason, dtype=object),
        block_residual_history=true_hist if monitor_mode == "true" else pre_hist,
        stopping_rule="per_rhs_true_residual_at_restart",
        block_metadata={
            "batch_size": int(batch_size),
            "batch_count": len(batch_ranges),
            "deflation_tol": None if deflation_tol is None else float(deflation_tol),
            "batches": batch_meta,
            "block_residual_norm": float(block_residual),
            "block_relative_residual": float(block_relative),
            "per_rhs_target_abs": np.asarray(rhs_target_abs, dtype=float).tolist(),
            "operator_block_adapter_used": any(
                not bool(batch.get("operator_supports_block", False)) for batch in batch_meta
            ),
            "preconditioner_block_adapter_used": any(
                not bool(batch.get("preconditioner_supports_block", True)) for batch in batch_meta
            ),
        },
    )


def gmres_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    callback: Callable[[float], None] | None = None,
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
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    callback: Callable[[float], None] | None = None,
    callback_true: Callable[[float], None] | None = None,
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
    receives verified true residual updates only when the native solver
    performs an explicit true-residual `A @ x` check. With the default
    `compute_final_residual=True`, native CuPy restarted GMRES verifies true
    residuals at restart boundaries for robust stopping decisions.
    `compute_final_residual=False` disables those checks and leaves final
    true-residual scalars as `NaN`.
    """
    (
        cupy,
        op_dtype,
        maxiter_total,
        monitor_mode,
        native_callback,
        native_restart_callback,
        progress_close,
    ) = _prepare_cupy_restarted_solver(
        "gmres",
        b,
        rtol=rtol,
        maxiter=maxiter,
        show_progress=show_progress,
        monitor=monitor,
        progress_residual=progress_residual,
        callback=callback,
        callback_true=callback_true,
    )
    try:
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
            compute_final_residual=bool(compute_final_residual),
        )
    finally:
        progress_close()

    _record_backend_solution(native.x)
    return _finalize_cupy_restarted_result(
        "gmres",
        native,
        monitor_mode=monitor_mode,
        compute_final_residual=bool(compute_final_residual),
    )


def fgmres_cupy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[..., np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    callback: Callable[[float], None] | None = None,
    callback_true: Callable[[float], None] | None = None,
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
    state dictionaries). Verified true-residual callbacks are emitted only
    when the native solver performs an explicit true-residual `A @ x` check.
    With the default `compute_final_residual=True`, native CuPy restarted
    FGMRES verifies true residuals at restart boundaries for robust stopping
    decisions. `compute_final_residual=False` disables those checks and leaves
    final true-residual scalars as `NaN`.
    """
    (
        cupy,
        op_dtype,
        maxiter_total,
        monitor_mode,
        native_callback,
        native_restart_callback,
        progress_close,
    ) = _prepare_cupy_restarted_solver(
        "fgmres",
        b,
        rtol=rtol,
        maxiter=maxiter,
        show_progress=show_progress,
        monitor=monitor,
        progress_residual=progress_residual,
        callback=callback,
        callback_true=callback_true,
    )
    try:
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
            compute_final_residual=bool(compute_final_residual),
        )
    finally:
        progress_close()

    _record_backend_solution(native.x)
    return _finalize_cupy_restarted_result(
        "fgmres",
        native,
        monitor_mode=monitor_mode,
        compute_final_residual=bool(compute_final_residual),
    )


def lgmres_cupy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 30,
    maxiter: int | None = None,
    outer_k: int = 3,
    store_outer_av: bool = True,
    callback: Callable[[float], None] | None = None,
    callback_true: Callable[[float], None] | None = None,
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
    Monitor/callback semantics mirror ``gmres_cupy``: inner progress uses the
    projected/preconditioned recurrence, while verified true-residual callbacks
    are emitted when explicit true-residual checks are executed. With the
    default `compute_final_residual=True`, native CuPy restarted LGMRES
    verifies true residuals at restart boundaries for robust stopping
    decisions. `compute_final_residual=False` disables those checks and leaves
    final true-residual scalars as `NaN`.
    """
    (
        cupy,
        op_dtype,
        maxiter_total,
        monitor_mode,
        native_callback,
        native_restart_callback,
        progress_close,
    ) = _prepare_cupy_restarted_solver(
        "lgmres",
        b,
        rtol=rtol,
        maxiter=maxiter,
        show_progress=show_progress,
        monitor=monitor,
        progress_residual=progress_residual,
        callback=callback,
        callback_true=callback_true,
    )
    try:
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
            compute_final_residual=bool(compute_final_residual),
        )
    finally:
        progress_close()

    _record_backend_solution(native.x)
    return _finalize_cupy_restarted_result(
        "lgmres",
        native,
        monitor_mode=monitor_mode,
        compute_final_residual=bool(compute_final_residual),
    )


def bicgstab_cupy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    callback: Callable[[float], None] | None = None,
    show_progress: bool = True,
    compute_final_residual: bool = True,
) -> LinearSolveResult:
    """Solve Ax=b via native CuPy BiCGSTAB.

    This path keeps BiCGSTAB state vectors on device and reports true-residual
    progress each iteration. It is intentionally lighter than restarted GMRES
    because it does not store an explicit Krylov basis.
    """

    cupy, _ = import_cupy()
    b_arr = np.asarray(b)
    if b_arr.ndim != 1:
        raise ValueError(
            "`bicgstab_cupy` expects a 1D RHS. Multi-RHS runs should use "
            "`solve_linear_system(..., method='bicgstab', backend='cupy')`."
        )
    n = int(b_arr.size)
    op_dtype = np.dtype(np.result_type(b_arr.dtype, np.complex64))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10

    progress_update, progress_close, _ = _make_progress_tracker(
        "bicgstab[cupy]",
        show_progress=show_progress,
        target_rel=float(rtol),
        residual_label="true_rel_res",
        max_iters=maxiter_total,
    )

    def _native_callback(true_rel: float) -> None:
        progress_update(float(true_rel))
        if callback is not None:
            callback(float(true_rel))

    native = bicgstab_cupy_native(
        A_mv,
        b,
        cupy=cupy,
        x0=x0,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter_total,
        operator_dtype=op_dtype,
        callback=_native_callback if (show_progress or callback is not None) else None,
        compute_final_residual=bool(compute_final_residual),
    )
    progress_close()

    _record_backend_solution(native.x)
    x_np = asnumpy(native.x)
    if compute_final_residual:
        residual_norm = float(native.residual_norm)
        relative_residual = float(native.relative_residual)
    else:
        residual_norm = float("nan")
        relative_residual = float("nan")

    true_hist = np.asarray(native.true_history, dtype=float)
    return LinearSolveResult(
        x=x_np,
        info=int(native.info),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=int(native.iterations),
        method="bicgstab[cupy]",
        residual_history=true_hist,
        rhs_count=1,
        preconditioned_residual_history=None,
        true_residual_history=true_hist,
        converged_reason=str(native.converged_reason),
    )


def bicgstab_scipy(
    A_mv: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
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
        """BiCGSTAB callback.

        SciPy provides the current iterate, not the residual. Computing a true
        residual here costs one extra matvec per iteration, which is too
        expensive for periodic matrix-free operators. Only pay that diagnostic
        cost when progress output is actually requested.
        """
        nonlocal iterations
        iterations += 1
        if not show_progress:
            progress_update(None)
            return
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
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
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

    maxiter_resolved = int(maxiter) if maxiter is not None else 1000
    x, info = lgmres(
        Aop,
        b,
        x0=x0,
        M=Mop,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter_resolved,
        callback=_cb,
    )
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
    x0: np.ndarray | None = None,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
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

    maxiter_resolved = int(maxiter) if maxiter is not None else 1000
    x, info = gcrotmk(
        Aop,
        b,
        x0=x0,
        M=Mop,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter_resolved,
        callback=_cb,
    )
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
    A_dense: np.ndarray | None = None,
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
            col_iter: Iterable[int] = range(n)
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
    A_dense: np.ndarray | None = None,
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
            col_iter: Iterable[int] = range(n)
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

    _record_backend_solution(x_gpu[:, 0] if squeezed else x_gpu)
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
    A_dense: np.ndarray | None = None,
    A_factorized: DenseLUFactorization | None = None,
    x0: np.ndarray | None = None,
    preconditioner: Callable[..., np.ndarray] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    gmres_monitor: Literal["preconditioned", "true", "both"] = "preconditioned",
    gmres_progress_residual: Literal["preconditioned", "true"] = "preconditioned",
    gmres_orthogonalization: Literal["mgs", "cgs"] = "mgs",
    gmres_cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    gmres_happy_breakdown_tol: float = 0.0,
    gmres_block_batch_size: int | None = None,
    gmres_block_deflation_tol: float | None = None,
    gmres_block_reorthogonalize: bool = True,
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
    gmres_block_batch_size, gmres_block_deflation_tol, gmres_block_reorthogonalize:
        CuPy-native block GMRES controls used when `backend='cupy'`,
        `method='gmres'`, and `b` is 2D.
    lgmres_outer_k, lgmres_store_outer_av:
        CuPy-native LGMRES recycle controls. ``lgmres_outer_k`` is the number
        of correction directions retained across restart cycles; enabling
        ``lgmres_store_outer_av`` caches ``A @ v`` for those recycled vectors.
    compute_final_residual:
        If `True`, compute/store true-residual diagnostics. For native CuPy
        restarted GMRES/FGMRES/LGMRES this enables true-residual checks at
        restart boundaries (robust default); if `False`, those checks are
        disabled and final true-residual scalars are reported as `NaN`.

    Multi-RHS policy:
    - `backend='cupy', method='gmres', b.ndim==2` uses the native block-GMRES path.
    - other iterative paths currently process RHS columns independently.
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
    if backend_name == "cupy" and m not in {"gmres", "fgmres", "bicgstab", "lgmres", "direct"}:
        raise ValueError(
            "The CuPy linear-solver backend currently supports only GMRES, FGMRES, BiCGSTAB, LGMRES, or direct solves."
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

    if nrhs > 1 and m == "gmres" and backend_name == "cupy":
        return gmres_cupy_block(
            A_mv,
            b_mat,
            x0=x0_mat,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            block_batch_size=gmres_block_batch_size,
            deflation_tol=gmres_block_deflation_tol,
            monitor=gmres_monitor,
            progress_residual=gmres_progress_residual,
            reorthogonalize=bool(gmres_block_reorthogonalize),
            show_progress=show_progress,
            compute_final_residual=compute_final_residual,
        )

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
                gmres_block_batch_size=gmres_block_batch_size,
                gmres_block_deflation_tol=gmres_block_deflation_tol,
                gmres_block_reorthogonalize=gmres_block_reorthogonalize,
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
        if backend_name == "cupy":
            return bicgstab_cupy(
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
