from __future__ import annotations

"""Native CuPy Krylov implementations used by pyceles.

This module keeps restarted GMRES state on device memory and avoids delegating
inner-iteration linear-algebra work to host-side SciPy/NumPy routines.
"""

import inspect
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


@dataclass(frozen=True)
class CuPyBlockGMRESNativeResult:
    """Result payload for native CuPy block-GMRES solves."""

    x: Any
    info: int
    iterations: int
    block_residual_norm: float
    block_relative_residual: float
    residual_norms: np.ndarray
    relative_residuals: np.ndarray
    converged_reason: str
    preconditioned_history: np.ndarray
    true_history: np.ndarray
    per_rhs_true_history: np.ndarray
    operator_supports_block: bool
    preconditioner_supports_block: bool


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


def _as_device_matrix(
    x: Any,
    *,
    cupy: Any,
    dtype: np.dtype,
    name: str,
    n_rows: int | None = None,
) -> Any:
    arr = cupy.asarray(x, dtype=dtype)
    if int(arr.ndim) == 1:
        arr = arr.reshape(-1, 1)
    elif int(arr.ndim) != 2:
        raise ValueError(
            f"{name} must be 2D (or 1D promoted to one column). Got ndim={int(arr.ndim)}."
        )
    if n_rows is not None and int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"{name} first dimension must be {int(n_rows)}. Got {int(arr.shape[0])}.")
    return arr


def _dot(u: Any, v: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    return cupy.vdot(cupy.asarray(u, dtype=accum_dtype), cupy.asarray(v, dtype=accum_dtype))


def _norm(v: Any, *, cupy: Any, accum_dtype: np.dtype) -> float:
    return float(cupy.linalg.norm(cupy.asarray(v, dtype=accum_dtype)))


def _dot_block(v: Any, w: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    v_acc = cupy.asarray(v, dtype=accum_dtype)
    w_acc = cupy.asarray(w, dtype=accum_dtype)
    return v_acc.conj().T @ w_acc


def _norms_block(v: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    return cupy.linalg.norm(cupy.asarray(v, dtype=accum_dtype), axis=0)


def _solve_block_least_squares(
    h: Any,
    g: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
) -> Any:
    """Solve ``min ||g - h y||_F`` on device using QR-first arithmetic.

    CuPy's generic ``lstsq`` path currently relies on SVD and can become
    brittle on some driver/cusolver stacks under long restarted runs. The
    block-GMRES projected systems are tall-and-thin and small, so reduced QR
    followed by a triangular solve is the natural fast path here.
    """
    h_acc = cupy.asarray(h, dtype=accum_dtype)
    g_acc = cupy.asarray(g, dtype=accum_dtype)
    q, r = cupy.linalg.qr(h_acc, mode="reduced")
    rhs = q.conj().T @ g_acc
    try:
        return cupy.linalg.solve(r, rhs)
    except Exception:
        # Defensive fallback for near-rank-deficient projected systems.
        hth = h_acc.conj().T @ h_acc
        eye = cupy.eye(int(hth.shape[0]), dtype=accum_dtype)
        ridge = cupy.asarray(np.finfo(accum_dtype).eps, dtype=accum_dtype)
        htg = h_acc.conj().T @ g_acc
        return cupy.linalg.solve(hth + ridge * eye, htg)


def _pack_block_basis(v_blocks: Any, *, cupy: Any, dtype: np.dtype) -> Any:
    """Pack block basis tensor `(nb, n, p)` into matrix `(n, nb*p)`."""
    v = cupy.asarray(v_blocks, dtype=dtype)
    return v.transpose(1, 0, 2).reshape(int(v.shape[1]), int(v.shape[0] * v.shape[2]))


def _apply_block_op(
    op: Callable[[Any], Any],
    x: Any,
    *,
    cupy: Any,
    dtype: np.dtype,
    supports_block: bool | None = None,
) -> tuple[Any, bool]:
    x_mat = _as_device_matrix(x, cupy=cupy, dtype=dtype, name="x_block")
    if supports_block is not False:
        try:
            y_try = _as_device_matrix(
                op(cupy.asarray(x_mat, dtype=dtype)),
                cupy=cupy,
                dtype=dtype,
                name="op(X)",
                n_rows=int(x_mat.shape[0]),
            )
            if int(y_try.shape[1]) == int(x_mat.shape[1]):
                return y_try, True
        except (TypeError, ValueError):
            pass
    cols = [
        _as_device_vector(
            op(cupy.asarray(x_mat[:, j], dtype=dtype)),
            cupy=cupy,
            dtype=dtype,
            name="op(x_j)",
        )[:, None]
        for j in range(int(x_mat.shape[1]))
    ]
    return cupy.concatenate(cols, axis=1), False


def _apply_block_preconditioner(
    preconditioner: Callable[[Any], Any] | None,
    x: Any,
    *,
    cupy: Any,
    dtype: np.dtype,
    supports_block: bool | None = None,
) -> tuple[Any, bool]:
    x_mat = _as_device_matrix(x, cupy=cupy, dtype=dtype, name="X")
    if preconditioner is None:
        return cupy.asarray(x_mat, dtype=dtype), True
    if supports_block is not False:
        try:
            y_try = _as_device_matrix(
                preconditioner(cupy.asarray(x_mat, dtype=dtype)),
                cupy=cupy,
                dtype=dtype,
                name="M^-1(X)",
                n_rows=int(x_mat.shape[0]),
            )
            if int(y_try.shape[1]) == int(x_mat.shape[1]):
                return y_try, True
        except (TypeError, ValueError):
            pass
    cols = [
        _as_device_vector(
            preconditioner(cupy.asarray(x_mat[:, j], dtype=dtype)),
            cupy=cupy,
            dtype=dtype,
            name="M^-1(x_j)",
        )[:, None]
        for j in range(int(x_mat.shape[1]))
    ]
    return cupy.concatenate(cols, axis=1), False


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


def _preconditioner_accepts_state(preconditioner: Callable[[Any], Any] | None) -> bool:
    if preconditioner is None:
        return False
    try:
        sig = inspect.signature(preconditioner)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def fgmres_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    preconditioner: Callable[..., Any] | None = None,
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
    """Run restarted right-preconditioned flexible GMRES on CuPy arrays.

    Compared with standard GMRES, FGMRES stores both Krylov basis vectors `V`
    and preconditioned vectors `Z_j = M_j^{-1} V_j`, allowing the
    preconditioner to vary by iteration.
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
        x0_is_zero = bool(float(cupy.max(cupy.abs(x_vec))) == 0.0)

    accepts_state = _preconditioner_accepts_state(preconditioner)

    def _apply(op: Callable[[Any], Any], vec: Any) -> Any:
        return _as_device_vector(
            op(cupy.asarray(vec, dtype=op_dtype)), cupy=cupy, dtype=op_dtype, name="op(x)"
        )

    def _apply_flexible_minv(vec: Any, *, iteration: int, cycle_iteration: int) -> Any:
        if preconditioner is None:
            return cupy.asarray(vec, dtype=op_dtype)
        v = cupy.asarray(vec, dtype=op_dtype)
        if accepts_state:
            iter_state = {
                "iteration": int(iteration),
                "cycle_iteration": int(cycle_iteration),
            }
            out = preconditioner(v, iter_state)
        else:
            out = preconditioner(v)
        return _as_device_vector(out, cupy=cupy, dtype=op_dtype, name="M_i^-1(x)")

    def _true_residual_stats(x_curr: Any) -> tuple[float, float, Any]:
        r_true = b_vec - _apply(A_mv, x_curr)
        abs_norm = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        rel_norm = abs_norm / b_norm if b_norm > 0 else abs_norm
        return abs_norm, rel_norm, r_true

    b_norm = _norm(b_vec, cupy=cupy, accum_dtype=acc_dtype)
    target_abs = max(float(atol), float(rtol) * b_norm)
    eps = float(np.finfo(op_dtype.char).eps)
    breakdown_tol_f = float(breakdown_tol)
    ptol_max_factor = 1.0
    ptol = target_abs
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

        beta = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        V = cupy.zeros((cycle_steps + 1, n), dtype=op_dtype)
        Z = cupy.zeros((cycle_steps, n), dtype=op_dtype)
        H = cupy.zeros((cycle_steps, cycle_steps + 1), dtype=acc_dtype)
        cs = cupy.zeros((cycle_steps,), dtype=acc_dtype)
        sn = cupy.zeros((cycle_steps,), dtype=acc_dtype)
        g = cupy.zeros((cycle_steps + 1,), dtype=acc_dtype)
        V[0, :] = r_true / beta
        g[0] = beta
        k_used = 0
        cycle_breakdown = False
        cycle_breakdown_reason = "breakdown"
        cycle_presid = float("nan")

        for col in range(cycle_steps):
            z = _apply_flexible_minv(V[col, :], iteration=iterations, cycle_iteration=col)
            Z[col, :] = z
            w = _apply(A_mv, z)
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
        x_vec = x_vec + cupy.asarray(y @ Z[:k_used, :], dtype=op_dtype)
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


def lgmres_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    preconditioner: Callable[[Any], Any] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 30,
    maxiter: int | None = None,
    outer_k: int = 3,
    store_outer_av: bool = True,
    prepend_outer_v: bool = False,
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
    """Run restarted right-preconditioned LGMRES on CuPy arrays.

    LGMRES augments each restarted cycle with up to ``outer_k`` normalized
    correction directions from previous cycles, which often mitigates restart
    stagnation versus plain restarted GMRES at similar memory footprint.
    """
    if prepend_outer_v:
        raise ValueError("`prepend_outer_v=True` is not supported in the native CuPy LGMRES path.")
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
    outer_keep = max(0, int(outer_k))
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
    ptol_max_factor = 1.0
    ptol = target_abs
    precond_hist: list[float] = []
    true_hist: list[float] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    residual_norm = float("nan")
    relative_residual = float("nan")
    r_true = None
    need_pr_rel = bool(callback is not None or record_preconditioned_history)
    outer_v: list[tuple[Any, Any | None]] = []

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

        z0 = _apply_minv(r_true)
        beta = _norm(z0, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        aug_count = min(len(outer_v), outer_keep)
        total_steps = cycle_steps + aug_count
        V = cupy.zeros((total_steps + 1, n), dtype=op_dtype)
        Z = cupy.zeros((total_steps, n), dtype=op_dtype)
        H = cupy.zeros((total_steps, total_steps + 1), dtype=acc_dtype)
        cs = cupy.zeros((total_steps,), dtype=acc_dtype)
        sn = cupy.zeros((total_steps,), dtype=acc_dtype)
        g = cupy.zeros((total_steps + 1,), dtype=acc_dtype)
        V[0, :] = z0 / beta
        g[0] = beta
        k_used = 0
        cycle_breakdown = False
        cycle_breakdown_reason = "breakdown"
        cycle_presid = float("nan")

        for col in range(total_steps):
            if col < cycle_steps:
                z = _apply_minv(V[col, :])
                w = _apply(A_mv, z)
            else:
                idx_aug = col - cycle_steps
                z, w_cached = outer_v[idx_aug]
                z = cupy.asarray(z, dtype=op_dtype)
                w = _apply(A_mv, z) if w_cached is None else cupy.asarray(w_cached, dtype=op_dtype)
            Z[col, :] = z
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
        dx = cupy.asarray(y @ Z[:k_used, :], dtype=op_dtype)
        x_vec = x_vec + dx

        if outer_keep > 0:
            dx_norm = _norm(dx, cupy=cupy, accum_dtype=acc_dtype)
            if dx_norm > breakdown_tol_f:
                dx_unit = cupy.asarray(dx / dx_norm, dtype=op_dtype)
                ax_unit = _apply(A_mv, dx_unit) if bool(store_outer_av) else None
                outer_v.append((dx_unit, ax_unit))
                while len(outer_v) > outer_keep:
                    del outer_v[0]

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


def block_gmres_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    preconditioner: Callable[[Any], Any] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 30,
    maxiter: int | None = None,
    operator_dtype: npt.DTypeLike | None = None,
    accum_dtype: npt.DTypeLike | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    restart_callback: Callable[[dict[str, Any]], None] | None = None,
    record_preconditioned_history: bool = False,
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-30,
) -> CuPyBlockGMRESNativeResult:
    """Run native CuPy block-GMRES on a right-hand-side matrix ``B``.

    The solver keeps block Arnoldi state on device and uses a block residual
    proxy for inexpensive inner monitoring. True convergence is accepted only
    when all RHS columns satisfy their per-column true-residual tolerances.
    When the proxy first drops below the target, the solver performs an
    immediate in-cycle true-residual gate so large restart values can still
    stop promptly without waiting for the restart boundary.
    """
    b_dtype_obj = getattr(b, "dtype", None)
    b_dtype = np.dtype(np.asarray(b).dtype if b_dtype_obj is None else b_dtype_obj)
    op_dtype = _dtype_complex(
        operator_dtype if operator_dtype is not None else np.result_type(b_dtype, np.complex64),
        name="operator_dtype",
    )
    acc_dtype = _dtype_complex(
        accum_dtype if accum_dtype is not None else np.result_type(op_dtype, np.complex128),
        name="accum_dtype",
    )
    b_mat = _as_device_matrix(b, cupy=cupy, dtype=op_dtype, name="b")
    n = int(b_mat.shape[0])
    p = int(b_mat.shape[1])
    restart_n = min(max(1, int(restart)), max(1, n))
    maxiter_total = int(maxiter) if maxiter is not None else n * 10
    if maxiter_total < 1:
        raise ValueError("`maxiter` must be >= 1 when provided.")
    x_mat = (
        cupy.zeros((n, p), dtype=op_dtype)
        if x0 is None
        else _as_device_matrix(x0, cupy=cupy, dtype=op_dtype, name="x0", n_rows=n)
    )
    if int(x_mat.shape[1]) != p:
        raise ValueError(f"`x0` column count {int(x_mat.shape[1])} does not match `b` columns {p}.")

    op_supports_block: bool | None = None
    minv_supports_block: bool | None = None

    def _apply(x_curr: Any) -> Any:
        nonlocal op_supports_block
        out, op_supports_block = _apply_block_op(
            A_mv,
            x_curr,
            cupy=cupy,
            dtype=op_dtype,
            supports_block=op_supports_block,
        )
        return out

    def _apply_minv(x_curr: Any) -> Any:
        nonlocal minv_supports_block
        out, minv_supports_block = _apply_block_preconditioner(
            preconditioner,
            x_curr,
            cupy=cupy,
            dtype=op_dtype,
            supports_block=minv_supports_block,
        )
        return out

    def _true_residual_stats(x_curr: Any) -> tuple[float, float, np.ndarray, np.ndarray, Any]:
        r_true = b_mat - _apply(x_curr)
        rhs_norms = _norms_block(r_true, cupy=cupy, accum_dtype=acc_dtype)
        rhs_norms_np = np.asarray(cupy.asnumpy(rhs_norms), dtype=float)
        rhs_rel = np.divide(
            rhs_norms_np,
            b_norms_np,
            out=np.asarray(rhs_norms_np, dtype=float).copy(),
            where=b_norms_np > 0,
        )
        block_abs = float(cupy.linalg.norm(cupy.asarray(r_true, dtype=acc_dtype)))
        block_rel = block_abs / b_norm_frob if b_norm_frob > 0 else block_abs
        return block_abs, block_rel, rhs_norms_np, rhs_rel, r_true

    b_norms = _norms_block(b_mat, cupy=cupy, accum_dtype=acc_dtype)
    b_norms_np = np.asarray(cupy.asnumpy(b_norms), dtype=float)
    b_norm_frob = float(cupy.linalg.norm(cupy.asarray(b_mat, dtype=acc_dtype)))
    rhs_target_abs = np.maximum(float(atol), float(rtol) * b_norms_np)
    target_abs_block = max(float(atol), float(rtol) * b_norm_frob)
    breakdown_tol_f = float(breakdown_tol)
    eps = float(np.finfo(op_dtype.char).eps)
    ptol_max_factor = 1.0
    ptol = target_abs_block
    precond_hist: list[float] = []
    true_hist: list[float] = []
    per_rhs_true_hist: list[np.ndarray] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    block_residual_norm = float("nan")
    block_relative_residual = float("nan")
    residual_norms = np.full((p,), np.nan, dtype=float)
    relative_residuals = np.full((p,), np.nan, dtype=float)
    r_true = None
    x0_is_zero = x0 is None
    if not x0_is_zero and n * p > 0:
        x0_is_zero = bool(float(cupy.max(cupy.abs(x_mat))) == 0.0)

    def _all_rhs_converged(rhs_abs: np.ndarray) -> bool:
        return bool(np.all(np.asarray(rhs_abs, dtype=float) <= rhs_target_abs))

    while iterations < maxiter_total:
        if r_true is None:
            if x0_is_zero:
                r_true = cupy.asarray(b_mat, dtype=op_dtype)
                residual_norms = np.asarray(b_norms_np, dtype=float)
                relative_residuals = np.where(b_norms_np > 0, 1.0, 0.0).astype(float)
                block_residual_norm = float(b_norm_frob)
                block_relative_residual = 1.0 if b_norm_frob > 0 else 0.0
                x0_is_zero = False
            else:
                (
                    block_residual_norm,
                    block_relative_residual,
                    residual_norms,
                    relative_residuals,
                    r_true,
                ) = _true_residual_stats(x_mat)
            true_hist.append(float(block_relative_residual))
            per_rhs_true_hist.append(np.asarray(relative_residuals, dtype=float))
            if restart_callback is not None:
                restart_callback(
                    {
                        "stage": "restart",
                        "iteration": int(iterations),
                        "block_relative_residual": float(block_relative_residual),
                        "per_rhs_relative_residual": np.asarray(relative_residuals, dtype=float),
                    }
                )
        if _all_rhs_converged(residual_norms):
            info = 0
            converged_reason = "converged"
            break

        z0 = _apply_minv(r_true)
        q0, r0 = cupy.linalg.qr(cupy.asarray(z0, dtype=acc_dtype), mode="reduced")
        q0 = cupy.asarray(q0, dtype=op_dtype)
        r0 = cupy.asarray(r0, dtype=acc_dtype)
        if float(cupy.linalg.norm(r0)) <= max(breakdown_tol_f, eps):
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        V = cupy.zeros((cycle_steps + 1, n, p), dtype=op_dtype)
        V[0] = q0
        H = cupy.zeros(((cycle_steps + 1) * p, cycle_steps * p), dtype=acc_dtype)
        G = cupy.zeros(((cycle_steps + 1) * p, p), dtype=acc_dtype)
        G[:p, :] = r0
        k_used = 0
        cycle_breakdown = False
        cycle_presid = float("nan")
        y_last = None
        y_last_k_used = -1
        cycle_converged = False
        inner_proxy_interval = 1 if (record_preconditioned_history or callback is not None) else 4

        for col in range(cycle_steps):
            w_acc = cupy.asarray(_apply_minv(_apply(V[col])), dtype=acc_dtype)
            v_prev = _pack_block_basis(V[: col + 1], cupy=cupy, dtype=acc_dtype)
            cs = slice(col * p, (col + 1) * p)
            h_block = v_prev.conj().T @ w_acc
            H[: (col + 1) * p, cs] = h_block
            w_acc = w_acc - v_prev @ h_block
            if reorthogonalize:
                h_block2 = v_prev.conj().T @ w_acc
                H[: (col + 1) * p, cs] = H[: (col + 1) * p, cs] + h_block2
                w_acc = w_acc - v_prev @ h_block2

            q_next, r_next = cupy.linalg.qr(w_acc, mode="reduced")
            if float(cupy.linalg.norm(r_next)) <= max(breakdown_tol_f, eps):
                cycle_breakdown = True
                k_used = col + 1
                y_last = None
                iterations += 1
                cycle_presid = 0.0
                if record_preconditioned_history:
                    precond_hist.append(0.0)
                if callback is not None:
                    callback(
                        {
                            "stage": "inner",
                            "iteration": int(iterations),
                            "block_relative_residual": 0.0,
                        }
                    )
                break
            V[col + 1] = cupy.asarray(q_next, dtype=op_dtype)
            H[(col + 1) * p : (col + 2) * p, col * p : (col + 1) * p] = cupy.asarray(
                r_next, dtype=acc_dtype
            )

            k_used = col + 1
            iterations += 1
            should_update_proxy = (
                (col + 1) % inner_proxy_interval == 0
                or (col + 1) == cycle_steps
                or iterations >= maxiter_total
            )
            if should_update_proxy:
                h_curr = H[: (col + 2) * p, : (col + 1) * p]
                g_curr = G[: (col + 2) * p, :]
                y_curr = _solve_block_least_squares(
                    h_curr, g_curr, cupy=cupy, accum_dtype=acc_dtype
                )
                y_last = cupy.asarray(y_curr, dtype=acc_dtype)
                y_last_k_used = k_used
                proj = g_curr - h_curr @ y_last
                pr_abs = float(cupy.linalg.norm(proj))
                pr_rel = pr_abs / b_norm_frob if b_norm_frob > 0 else pr_abs
                cycle_presid = pr_abs
                if record_preconditioned_history:
                    precond_hist.append(pr_rel)
                if callback is not None:
                    callback(
                        {
                            "stage": "inner",
                            "iteration": int(iterations),
                            "block_relative_residual": float(pr_rel),
                        }
                    )
                # Cheap proxy check every inner step; run the expensive
                # per-RHS true-residual gate only when the block proxy is at or
                # below the requested target.
                if pr_abs <= target_abs_block:
                    v_trial = _pack_block_basis(V[:k_used], cupy=cupy, dtype=op_dtype)
                    y_trial = cupy.asarray(y_last[: k_used * p, :], dtype=op_dtype)
                    x_trial = x_mat + v_trial @ y_trial
                    (
                        block_trial_abs,
                        block_trial_rel,
                        rhs_trial_abs,
                        rhs_trial_rel,
                        r_true_trial,
                    ) = _true_residual_stats(x_trial)
                    if _all_rhs_converged(rhs_trial_abs):
                        x_mat = x_trial
                        block_residual_norm = float(block_trial_abs)
                        block_relative_residual = float(block_trial_rel)
                        residual_norms = np.asarray(rhs_trial_abs, dtype=float)
                        relative_residuals = np.asarray(rhs_trial_rel, dtype=float)
                        r_true = r_true_trial
                        true_hist.append(float(block_relative_residual))
                        per_rhs_true_hist.append(np.asarray(relative_residuals, dtype=float))
                        if restart_callback is not None:
                            restart_callback(
                                {
                                    "stage": "restart",
                                    "iteration": int(iterations),
                                    "block_relative_residual": float(block_relative_residual),
                                    "per_rhs_relative_residual": np.asarray(
                                        relative_residuals, dtype=float
                                    ),
                                }
                            )
                        info = 0
                        converged_reason = "converged"
                        cycle_converged = True
                        break
            if iterations >= maxiter_total:
                break

        if cycle_converged:
            break

        if k_used <= 0:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        if y_last is None or y_last_k_used != k_used:
            h_curr = H[: (k_used + 1) * p, : k_used * p]
            g_curr = G[: (k_used + 1) * p, :]
            y_curr = _solve_block_least_squares(h_curr, g_curr, cupy=cupy, accum_dtype=acc_dtype)
            y_last = cupy.asarray(y_curr, dtype=acc_dtype)
            y_last_k_used = k_used

        v_used = _pack_block_basis(V[:k_used], cupy=cupy, dtype=op_dtype)
        y_used = cupy.asarray(y_last[: k_used * p, :], dtype=op_dtype)
        dx = v_used @ y_used
        x_mat = x_mat + dx

        (
            block_residual_norm,
            block_relative_residual,
            residual_norms,
            relative_residuals,
            r_true,
        ) = _true_residual_stats(x_mat)
        true_hist.append(float(block_relative_residual))
        per_rhs_true_hist.append(np.asarray(relative_residuals, dtype=float))
        if restart_callback is not None:
            restart_callback(
                {
                    "stage": "restart",
                    "iteration": int(iterations),
                    "block_relative_residual": float(block_relative_residual),
                    "per_rhs_relative_residual": np.asarray(relative_residuals, dtype=float),
                }
            )
        if _all_rhs_converged(residual_norms):
            info = 0
            converged_reason = "converged"
            break
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break
        if np.isfinite(cycle_presid):
            if cycle_presid <= ptol:
                ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
            else:
                ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
            if block_residual_norm > 0:
                ptol = cycle_presid * min(ptol_max_factor, target_abs_block / block_residual_norm)
        if iterations >= maxiter_total:
            info = iterations
            converged_reason = "maxiter_reached"
            break

    return CuPyBlockGMRESNativeResult(
        x=x_mat,
        info=int(info),
        iterations=int(iterations),
        block_residual_norm=float(block_residual_norm),
        block_relative_residual=float(block_relative_residual),
        residual_norms=np.asarray(residual_norms, dtype=float),
        relative_residuals=np.asarray(relative_residuals, dtype=float),
        converged_reason=str(converged_reason),
        preconditioned_history=np.asarray(precond_hist, dtype=float),
        true_history=np.asarray(true_hist, dtype=float),
        per_rhs_true_history=np.asarray(per_rhs_true_hist, dtype=float),
        operator_supports_block=bool(op_supports_block) if op_supports_block is not None else False,
        preconditioner_supports_block=(
            bool(minv_supports_block) if minv_supports_block is not None else True
        ),
    )


__all__ = [
    "CuPyGMRESNativeResult",
    "CuPyBlockGMRESNativeResult",
    "block_gmres_cupy_native",
    "gmres_cupy_native",
    "fgmres_cupy_native",
    "lgmres_cupy_native",
]
