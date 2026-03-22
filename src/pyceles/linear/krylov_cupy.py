from __future__ import annotations

"""Native CuPy Krylov implementations used by pyceles.

This module keeps restarted GMRES state on device memory and avoids delegating
inner-iteration linear-algebra work to host-side SciPy/NumPy routines.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CuPyGMRESNativeResult:
    """Result payload for the native CuPy GMRES implementation."""

    x: Any
    info: int
    iterations: int
    residual_norm: float
    relative_residual: float
    converged_reason: str
    preconditioned_history: np.ndarray
    true_history: np.ndarray


def _dtype_complex(dtype: npt.DTypeLike, *, name: str) -> np.dtype:
    out = np.dtype(dtype)
    if out.kind != "c":
        raise TypeError(f"`{name}` must be a complex dtype. Got {out}.")
    return out


def _as_device_vector(x: Any, *, cupy: Any, dtype: np.dtype, name: str) -> Any:
    out = cupy.asarray(x, dtype=dtype).reshape(-1)
    if int(out.ndim) != 1:
        raise ValueError(f"{name} must be 1D after reshape. Got ndim={int(out.ndim)}.")
    return out


def _dot(u: Any, v: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    return cupy.vdot(cupy.asarray(u, dtype=accum_dtype), cupy.asarray(v, dtype=accum_dtype))


def _norm(v: Any, *, cupy: Any, accum_dtype: np.dtype) -> float:
    return float(cupy.linalg.norm(cupy.asarray(v, dtype=accum_dtype)))


def _givens_complex(a: Any, b: Any, *, cupy: Any, accum_dtype: np.dtype) -> tuple[Any, Any]:
    """Return complex Givens factors `(c, s)` such that:

      [conj(c)  conj(s)] [a] = [r]
      [  -s        c   ] [b]   [0]

    with unitary rotation and `r` real/positive when possible.
    """
    abs_a = cupy.abs(a)
    abs_b = cupy.abs(b)
    if float(abs_b) == 0.0:
        return (
            cupy.asarray(1.0 + 0.0j, dtype=accum_dtype),
            cupy.asarray(0.0 + 0.0j, dtype=accum_dtype),
        )
    if float(abs_a) == 0.0:
        return (
            cupy.asarray(0.0 + 0.0j, dtype=accum_dtype),
            cupy.asarray(1.0 + 0.0j, dtype=accum_dtype),
        )
    scale = abs_a + abs_b
    norm = scale * cupy.sqrt((abs_a / scale) ** 2 + (abs_b / scale) ** 2)
    alpha = a / abs_a
    c = abs_a / norm
    s = alpha * cupy.conj(b) / norm
    return cupy.asarray(c, dtype=accum_dtype), cupy.asarray(s, dtype=accum_dtype)


def gmres_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    preconditioner: Callable[[Any], Any] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    operator_dtype: npt.DTypeLike | None = None,
    accum_dtype: npt.DTypeLike | None = None,
    callback: Callable[[float], None] | None = None,
    restart_callback: Callable[[float], None] | None = None,
    record_preconditioned_history: bool = False,
    orthogonalization: Literal["mgs", "cgs"] = "mgs",
    cgs_refinement: Literal["never", "ifneeded", "always"] = "ifneeded",
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-30,
    happy_breakdown_tol: float = 0.0,
) -> CuPyGMRESNativeResult:
    """Run restarted left-preconditioned GMRES fully on CuPy arrays.

    The convergence criterion follows SciPy-style semantics: convergence is
    decided on the true residual `||b - A x||` at restart boundaries.
    Inner-iteration callbacks receive the GMRES preconditioned residual proxy
    from the Arnoldi/Givens recurrence.
    """
    b_dtype_obj = getattr(b, "dtype", None)
    b_dtype = np.dtype(np.asarray(b).dtype if b_dtype_obj is None else b_dtype_obj)
    orth_mode = str(orthogonalization).lower()
    if orth_mode not in {"mgs", "cgs"}:
        raise ValueError("`orthogonalization` must be 'mgs' or 'cgs'.")
    cgs_refine_mode = str(cgs_refinement).lower()
    if cgs_refine_mode not in {"never", "ifneeded", "always"}:
        raise ValueError("`cgs_refinement` must be 'never', 'ifneeded', or 'always'.")
    if float(happy_breakdown_tol) < 0.0:
        raise ValueError("`happy_breakdown_tol` must be >= 0.")
    op_dtype = _dtype_complex(
        operator_dtype if operator_dtype is not None else np.result_type(b_dtype, np.complex64),
        name="operator_dtype",
    )
    acc_dtype = _dtype_complex(
        accum_dtype if accum_dtype is not None else np.result_type(op_dtype, np.complex128),
        name="accum_dtype",
    )
    b_vec = _as_device_vector(b, cupy=cupy, dtype=op_dtype, name="b")
    n = int(b_vec.size)
    restart_n = min(max(1, int(restart)), max(1, n))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10
    if maxiter_total < 1:
        raise ValueError("`maxiter` must be >= 1 when provided.")
    x_vec = (
        cupy.zeros_like(b_vec, dtype=op_dtype)
        if x0 is None
        else _as_device_vector(x0, cupy=cupy, dtype=op_dtype, name="x0")
    )
    if int(x_vec.size) != n:
        raise ValueError(f"`x0` size {int(x_vec.size)} does not match `b` size {n}.")
    x0_is_zero = x0 is None
    if not x0_is_zero and n > 0:
        # Mirror SciPy's zero-initial-guess shortcut and avoid one expensive
        # `A @ x0` matvec when the provided warm start is exactly zero.
        x0_is_zero = bool(float(cupy.max(cupy.abs(x_vec))) == 0.0)

    def _apply(op: Callable[[Any], Any], vec: Any) -> Any:
        return _as_device_vector(
            op(cupy.asarray(vec, dtype=op_dtype)), cupy=cupy, dtype=op_dtype, name="op(x)"
        )

    def _apply_minv(vec: Any) -> Any:
        if preconditioner is None:
            return cupy.asarray(vec, dtype=op_dtype)
        return _as_device_vector(
            preconditioner(cupy.asarray(vec, dtype=op_dtype)),
            cupy=cupy,
            dtype=op_dtype,
            name="M^-1(x)",
        )

    def _true_residual_stats(x_curr: Any) -> tuple[float, float, Any]:
        r_true = b_vec - _apply(A_mv, x_curr)
        abs_norm = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        rel_norm = abs_norm / b_norm if b_norm > 0 else abs_norm
        return abs_norm, rel_norm, r_true

    b_norm = _norm(b_vec, cupy=cupy, accum_dtype=acc_dtype)
    target_abs = max(float(atol), float(rtol) * b_norm)
    eps = float(np.finfo(op_dtype.char).eps)
    breakdown_tol_f = float(breakdown_tol)
    Mb_norm = _norm(_apply_minv(b_vec), cupy=cupy, accum_dtype=acc_dtype)
    ptol_max_factor = 1.0
    ptol = Mb_norm * min(ptol_max_factor, target_abs / b_norm) if b_norm > 0 else target_abs
    precond_hist: list[float] = []
    true_hist: list[float] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    residual_norm = float("nan")
    relative_residual = float("nan")
    r_true = None
    need_pr_rel = bool(callback is not None or record_preconditioned_history)

    while iterations < maxiter_total:
        if r_true is None:
            if x0_is_zero:
                residual_norm = float(b_norm)
                relative_residual = 1.0 if b_norm > 0 else 0.0
                r_true = cupy.asarray(b_vec, dtype=op_dtype)
                x0_is_zero = False
            else:
                residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
            true_hist.append(relative_residual)
            if restart_callback is not None:
                restart_callback(relative_residual)
        if residual_norm <= target_abs:
            info = 0
            converged_reason = "converged"
            break

        r0 = r_true
        z0 = _apply_minv(r0)
        beta = _norm(z0, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        V = cupy.zeros((cycle_steps + 1, n), dtype=op_dtype)
        H = cupy.zeros((cycle_steps, cycle_steps + 1), dtype=acc_dtype)
        cs = cupy.zeros((cycle_steps,), dtype=acc_dtype)
        sn = cupy.zeros((cycle_steps,), dtype=acc_dtype)
        g = cupy.zeros((cycle_steps + 1,), dtype=acc_dtype)
        V[0, :] = z0 / beta
        g[0] = beta
        k_used = 0
        cycle_breakdown = False
        cycle_breakdown_reason = "breakdown"
        cycle_presid = float("nan")

        for col in range(cycle_steps):
            w = _apply_minv(_apply(A_mv, V[col, :]))
            h0 = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
            if orth_mode == "mgs":
                for k in range(col + 1):
                    hik = _dot(V[k, :], w, cupy=cupy, accum_dtype=acc_dtype)
                    H[col, k] = hik
                    w = w - cupy.asarray(hik, dtype=op_dtype) * V[k, :]
                if reorthogonalize:
                    for k in range(col + 1):
                        hik2 = _dot(V[k, :], w, cupy=cupy, accum_dtype=acc_dtype)
                        H[col, k] = H[col, k] + hik2
                        w = w - cupy.asarray(hik2, dtype=op_dtype) * V[k, :]
            else:
                h_row = cupy.zeros((col + 1,), dtype=acc_dtype)
                for k in range(col + 1):
                    h_row[k] = _dot(V[k, :], w, cupy=cupy, accum_dtype=acc_dtype)
                H[col, : col + 1] = h_row
                w = w - cupy.asarray(h_row @ V[: col + 1, :], dtype=op_dtype)
                run_cgs_refine = False
                if cgs_refine_mode == "always":
                    run_cgs_refine = True
                elif cgs_refine_mode == "ifneeded":
                    h1 = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
                    run_cgs_refine = h1 <= 0.5 * h0
                if run_cgs_refine:
                    h_row2 = cupy.zeros((col + 1,), dtype=acc_dtype)
                    for k in range(col + 1):
                        h_row2[k] = _dot(V[k, :], w, cupy=cupy, accum_dtype=acc_dtype)
                    H[col, : col + 1] = H[col, : col + 1] + h_row2
                    w = w - cupy.asarray(h_row2 @ V[: col + 1, :], dtype=op_dtype)

            h_next = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
            H[col, col + 1] = h_next
            happy_tol = max(breakdown_tol_f, float(happy_breakdown_tol) * max(h0, eps))
            if h_next > happy_tol:
                V[col + 1, :] = w / h_next
            if h_next <= eps * max(h0, eps):
                H[col, col + 1] = cupy.asarray(0.0 + 0.0j, dtype=acc_dtype)
                cycle_breakdown = True
                cycle_breakdown_reason = "breakdown"
            elif h_next <= happy_tol:
                H[col, col + 1] = cupy.asarray(0.0 + 0.0j, dtype=acc_dtype)
                cycle_breakdown = True
                cycle_breakdown_reason = "happy_breakdown"

            for k in range(col):
                c = cs[k].copy()
                s = sn[k].copy()
                n0 = H[col, k].copy()
                n1 = H[col, k + 1].copy()
                H[col, k] = c * n0 + s * n1
                H[col, k + 1] = -cupy.conj(s) * n0 + c * n1

            c_new, s_new = _givens_complex(
                H[col, col], H[col, col + 1], cupy=cupy, accum_dtype=acc_dtype
            )
            cs[col] = c_new
            sn[col] = s_new
            h_col_col = H[col, col].copy()
            h_col_colp1 = H[col, col + 1].copy()
            H[col, col] = c_new * h_col_col + s_new * h_col_colp1
            H[col, col + 1] = cupy.asarray(0.0 + 0.0j, dtype=acc_dtype)

            g_col = g[col].copy()
            g[col] = c_new * g_col
            g[col + 1] = -cupy.conj(s_new) * g_col

            k_used = col + 1
            iterations += 1
            pr_abs = float(cupy.abs(g[col + 1]))
            pr_rel = (pr_abs / b_norm if b_norm > 0 else pr_abs) if need_pr_rel else float("nan")
            cycle_presid = pr_abs
            if record_preconditioned_history:
                precond_hist.append(pr_rel)
            if callback is not None:
                callback(pr_rel)
            if cycle_breakdown or pr_abs <= ptol or iterations >= maxiter_total:
                break

        if k_used <= 0:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        y = cupy.array(g[:k_used], dtype=acc_dtype)
        last = k_used - 1
        if float(cupy.abs(H[last, last])) <= max(breakdown_tol_f, eps):
            y[last] = cupy.asarray(0.0 + 0.0j, dtype=acc_dtype)
        for k in range(last, 0, -1):
            yk = y[k].copy()
            if float(cupy.abs(yk)) == 0.0:
                continue
            yk = yk / H[k, k].copy()
            y[k] = yk
            h_row = cupy.asarray(H[k, :k], dtype=acc_dtype)
            y[:k] = y[:k] - yk * h_row
        if float(cupy.abs(y[0])) > 0.0:
            y[0] = y[0] / H[0, 0].copy()
        x_vec = x_vec + cupy.asarray(y @ V[:k_used, :], dtype=op_dtype)
        residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
        true_hist.append(relative_residual)
        if restart_callback is not None:
            restart_callback(relative_residual)
        if residual_norm <= target_abs:
            info = 0
            converged_reason = "converged"
            break
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = cycle_breakdown_reason
            break
        if cycle_presid <= ptol:
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        if residual_norm > 0:
            ptol = cycle_presid * min(ptol_max_factor, target_abs / residual_norm)
        if iterations >= maxiter_total:
            info = iterations
            converged_reason = "maxiter_reached"
            break

    return CuPyGMRESNativeResult(
        x=x_vec,
        info=int(info),
        iterations=int(iterations),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        converged_reason=str(converged_reason),
        preconditioned_history=np.asarray(precond_hist, dtype=float),
        true_history=np.asarray(true_hist, dtype=float),
    )


__all__ = ["CuPyGMRESNativeResult", "gmres_cupy_native"]
