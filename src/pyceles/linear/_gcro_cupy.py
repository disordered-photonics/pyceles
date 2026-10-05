"""Native CuPy harmonic GCRO-DR.

This module contains a forward-only recycling path.  The durable recycle pair
``U, C`` satisfies ``C = A @ U`` and ``C.conj().T @ C = I``.  Every completed
restart extracts a bounded set of small-magnitude harmonic Ritz directions
from the full augmented GCRO search space ``span(U, V_m)``.

Large basis vectors remain in the operator dtype.  Restart-boundary spectral
algebra is performed only on small host complex128 matrices, and products that
would otherwise widen or conjugate the complete Arnoldi basis use a bounded
workspace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .krylov_cupy import (
    _absolute_residual_target,
    _apply_givens_rotation,
    _as_device_matrix,
    _as_device_vector,
    _dot,
    _dtype_complex,
    _mgs_subtract_scaled,
    _norm,
    _resolve_accum_dtype,
    _solve_rotated_upper,
)


@dataclass(frozen=True)
class CuPyRecycleSpace:
    """Paired solution/image recycle bases for one fixed operator."""

    U: Any
    C: Any


@dataclass(frozen=True)
class CuPyGCRONativeResult:
    """Backend-native result from :func:`gcro_cupy_native`."""

    x: Any
    info: int
    iterations: int
    residual_norm: float
    relative_residual: float
    converged_reason: str
    preconditioned_history: np.ndarray
    true_history: np.ndarray
    operator_applications: int
    restart_cycles: int
    recycle_dimension: int


def _chunked_conjugate_rows_product(
    left_rows: Any,
    right_columns: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
    workspace_bytes: int = 64 * 1024**2,
) -> Any:
    """Return ``conj(left_rows) @ right_columns`` with bounded widening.

    ``left_rows`` stores full-space vectors as rows.  Only a slice of the
    unknown dimension is promoted to accumulation precision at one time, so a
    restart never creates a complex128 copy of the full Arnoldi basis.
    """

    rows = int(left_rows.shape[0])
    unknowns = int(left_rows.shape[1])
    columns = int(right_columns.shape[1])
    out = cupy.zeros((rows, columns), dtype=accum_dtype)
    if rows == 0 or columns == 0 or unknowns == 0:
        return out

    itemsize = int(np.dtype(accum_dtype).itemsize)
    live_columns = max(1, rows + columns)
    chunk = max(1, min(unknowns, int(workspace_bytes) // (live_columns * itemsize)))
    for start in range(0, unknowns, chunk):
        stop = min(unknowns, start + chunk)
        left_chunk = cupy.array(left_rows[:, start:stop], dtype=accum_dtype, copy=True)
        cupy.conj(left_chunk, out=left_chunk)
        right_chunk = cupy.asarray(right_columns[start:stop, :], dtype=accum_dtype)
        out += left_chunk @ right_chunk
        del left_chunk, right_chunk
    return out


def _recycle_projection(
    C: Any,
    vector: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
) -> Any:
    """Return ``C^H vector`` without widening a full-space temporary.

    The recycle rank is deliberately small (normally O(10)), so a handful of
    fused wide-accumulation reductions is preferable to converting the entire
    vector and recycle basis to complex128 on every Arnoldi step.
    """

    rank = int(C.shape[1])
    out = cupy.empty((rank,), dtype=accum_dtype)
    for index in range(rank):
        out[index] = _dot(C[:, index], vector, cupy=cupy, accum_dtype=accum_dtype)
    return out


def _paired_canonicalize(
    U: Any,
    C: Any,
    *,
    cupy: Any,
    operator_dtype: np.dtype,
    accum_dtype: np.dtype,
    max_rank: int,
) -> CuPyRecycleSpace | None:
    """Orthonormalize ``C`` while applying the same transform to ``U``."""

    u_mat = _as_device_matrix(U, cupy=cupy, dtype=operator_dtype, name="recycle U")
    c_mat = _as_device_matrix(C, cupy=cupy, dtype=operator_dtype, name="recycle C")
    if tuple(u_mat.shape) != tuple(c_mat.shape):
        raise ValueError(
            "Recycle U and C must have identical shapes. "
            f"Got U={tuple(u_mat.shape)}, C={tuple(c_mat.shape)}."
        )
    count = int(u_mat.shape[1])
    if count == 0 or int(max_rank) <= 0:
        return None

    gram_dev = _chunked_conjugate_rows_product(
        c_mat.T,
        c_mat,
        cupy=cupy,
        accum_dtype=accum_dtype,
    )
    gram = np.asarray(cupy.asnumpy(gram_dev), dtype=np.complex128)
    del gram_dev
    gram = 0.5 * (gram + gram.conj().T)
    try:
        evals, evecs = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        return None
    if evals.size == 0:
        return None
    scale = float(np.max(np.abs(evals)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    rel_tol = max(100.0 * np.finfo(operator_dtype).eps, 10.0 * np.finfo(float).eps)
    keep = np.flatnonzero(evals > rel_tol * scale)
    if keep.size == 0:
        return None
    keep = keep[np.argsort(evals[keep])[::-1]][: int(max_rank)]
    transform_host = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    transform = cupy.asarray(transform_host, dtype=operator_dtype)
    return CuPyRecycleSpace(
        U=cupy.asarray(u_mat @ transform, dtype=operator_dtype),
        C=cupy.asarray(c_mat @ transform, dtype=operator_dtype),
    )


def _harmonic_recycle_pair(
    *,
    arnoldi_rows: Any,
    hbar: Any,
    old_space: CuPyRecycleSpace | None,
    coupling: Any | None,
    recycle_dim: int,
    cupy: Any,
    operator_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> CuPyRecycleSpace | None:
    """Extract harmonic Ritz recycle vectors from the full augmented space."""

    steps = int(hbar.shape[1])
    old_rank = 0 if old_space is None else int(old_space.U.shape[1])
    keep = min(max(0, int(recycle_dim)), old_rank + steps)
    if steps == 0 or keep == 0:
        return None

    v_rows = cupy.asarray(arnoldi_rows[: steps + 1, :], dtype=operator_dtype)
    v_search_rows = v_rows[:steps, :]
    h_host = np.asarray(cupy.asnumpy(hbar[: steps + 1, :steps]), dtype=np.complex128)

    if old_rank > 0:
        assert old_space is not None
        if coupling is None:
            raise ValueError("Projected GCRO extraction requires C^H A V coefficients.")
        b_host = np.asarray(cupy.asnumpy(coupling[:old_rank, :steps]), dtype=np.complex128)
        old_u = cupy.asarray(old_space.U, dtype=operator_dtype)
        old_c = cupy.asarray(old_space.C, dtype=operator_dtype)
        u_norms = np.asarray(
            [_norm(old_u[:, j], cupy=cupy, accum_dtype=accum_dtype) for j in range(old_rank)],
            dtype=float,
        )
        if not np.all(np.isfinite(u_norms) & (u_norms > 0.0)):
            return old_space
        u_scale_host = 1.0 / u_norms
        u_scaled = old_u * cupy.asarray(u_scale_host, dtype=operator_dtype)[None, :]

        v_h_u_dev = _chunked_conjugate_rows_product(
            v_rows,
            u_scaled,
            cupy=cupy,
            accum_dtype=accum_dtype,
        )
        v_h_u = np.asarray(cupy.asnumpy(v_h_u_dev), dtype=np.complex128)
        del v_h_u_dev

        c_h_u_dev = _chunked_conjugate_rows_product(
            old_c.T,
            u_scaled,
            cupy=cupy,
            accum_dtype=accum_dtype,
        )
        c_h_u = np.asarray(cupy.asnumpy(c_h_u_dev), dtype=np.complex128)
        del c_h_u_dev
    else:
        b_host = np.empty((0, steps), dtype=np.complex128)
        u_scale_host = np.empty((0,), dtype=float)
        u_scaled = None
        old_c = None
        v_h_u = np.empty((steps + 1, 0), dtype=np.complex128)
        c_h_u = np.empty((0, 0), dtype=np.complex128)

    # With S=[U_hat,V_m], W=[C,V_(m+1)] and A S = W G,
    # harmonic Ritz vectors satisfy G^H G z = theta G^H W^H S z.
    total = old_rank + steps
    g_small = np.zeros((old_rank + steps + 1, total), dtype=np.complex128)
    if old_rank > 0:
        g_small[:old_rank, :old_rank] = np.diag(u_scale_host)
        g_small[:old_rank, old_rank:] = b_host
    g_small[old_rank:, old_rank:] = h_host

    w_h_s = np.zeros_like(g_small)
    if old_rank > 0:
        w_h_s[:old_rank, :old_rank] = c_h_u
        w_h_s[old_rank:, :old_rank] = v_h_u
    w_h_s[old_rank : old_rank + steps, old_rank:] = np.eye(steps, dtype=np.complex128)

    image_gram = g_small.conj().T @ g_small
    image_gram = 0.5 * (image_gram + image_gram.conj().T)
    image_h_search = g_small.conj().T @ w_h_s

    import scipy.linalg

    try:
        values, vectors = scipy.linalg.eig(image_gram, image_h_search, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        # A nearly singular projected pencil should not abort an otherwise
        # valid solve.  Retain the previous space and let the next restart
        # refresh it from a better-conditioned Arnoldi sample.
        return old_space
    finite = np.flatnonzero(
        np.isfinite(values.real) & np.isfinite(values.imag) & np.all(np.isfinite(vectors), axis=0)
    )
    if finite.size == 0:
        return old_space
    selected = finite[np.argsort(np.abs(values[finite]))[:keep]]
    coefficients = np.asarray(vectors[:, selected], dtype=np.complex128)
    coefficients_op = cupy.asarray(coefficients, dtype=operator_dtype)

    top = coefficients_op[:old_rank, :]
    bottom = coefficients_op[old_rank:, :]
    u_raw = cupy.asarray(v_search_rows.T @ bottom, dtype=operator_dtype)
    if old_rank > 0 and u_scaled is not None:
        u_raw += cupy.asarray(u_scaled @ top, dtype=operator_dtype)

    image_coefficients = g_small @ coefficients
    image_coefficients_op = cupy.asarray(image_coefficients, dtype=operator_dtype)
    c_raw = cupy.asarray(v_rows.T @ image_coefficients_op[old_rank:, :], dtype=operator_dtype)
    if old_rank > 0 and old_c is not None:
        c_raw += cupy.asarray(old_c @ image_coefficients_op[:old_rank, :], dtype=operator_dtype)

    return _paired_canonicalize(
        u_raw,
        c_raw,
        cupy=cupy,
        operator_dtype=operator_dtype,
        accum_dtype=accum_dtype,
        max_rank=keep,
    )


def gcro_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 100,
    maxiter: int | None = None,
    recycle_dim: int = 8,
    operator_dtype: npt.DTypeLike | None = None,
    accum_dtype: npt.DTypeLike | None = None,
    callback: Callable[[float], None] | None = None,
    restart_callback: Callable[[float], None] | None = None,
    record_proxy_history: bool = False,
    reorthogonalize: bool = True,
    breakdown_tol: float = 1e-30,
    happy_breakdown_tol: float = 0.0,
    compute_final_residual: bool = True,
) -> CuPyGCRONativeResult:
    """Run forward-only harmonic GCRO-DR on one CuPy right-hand side.

    ``restart`` is the complete GCRO-DR augmented dimension.  The first cycle
    is ordinary GMRES; subsequent cycles retain at most ``recycle_dim``
    harmonic Ritz vectors and therefore generate at most ``restart-k`` new
    Arnoldi directions.  A true physical residual is rebuilt at every restart
    because it is required to continue a restarted solve correctly.
    ``compute_final_residual=False`` suppresses only the terminal scalar
    diagnostics exposed by the public result; it does not replace the restart
    state with the cheaper Arnoldi proxy.
    """

    b_dtype_obj = getattr(b, "dtype", None)
    b_dtype = np.dtype(np.asarray(b).dtype if b_dtype_obj is None else b_dtype_obj)
    op_dtype = _dtype_complex(
        operator_dtype if operator_dtype is not None else np.result_type(b_dtype, np.complex64),
        name="operator_dtype",
    )
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
    if float(rtol) < 0.0 or float(atol) < 0.0:
        raise ValueError("`rtol` and `atol` must be non-negative.")
    if int(recycle_dim) < 1:
        raise ValueError("`recycle_dim` must be >= 1.")
    if float(happy_breakdown_tol) < 0.0:
        raise ValueError("`happy_breakdown_tol` must be non-negative.")

    b_vec = _as_device_vector(b, cupy=cupy, dtype=op_dtype, name="b")
    n = int(b_vec.size)
    restart_n = min(max(2, int(restart)), max(2, n))
    recycle_limit = min(int(recycle_dim), max(0, restart_n - 1))
    maxiter_total = int(maxiter) if maxiter is not None else max(1, 10 * n)
    if maxiter_total < 1:
        raise ValueError("`maxiter` must be >= 1 when provided.")

    x = (
        cupy.zeros_like(b_vec, dtype=op_dtype)
        if x0 is None
        else cupy.array(
            _as_device_vector(x0, cupy=cupy, dtype=op_dtype, name="x0"),
            dtype=op_dtype,
            copy=True,
        )
    )
    if int(x.size) != n:
        raise ValueError(f"`x0` size {int(x.size)} does not match `b` size {n}.")

    x0_is_zero = x0 is None

    operator_applications = 0

    def apply(values: Any) -> Any:
        nonlocal operator_applications
        out = _as_device_vector(
            A_mv(cupy.asarray(values, dtype=op_dtype)),
            cupy=cupy,
            dtype=op_dtype,
            name="A(x)",
        )
        operator_applications += 1
        return out

    b_norm = _norm(b_vec, cupy=cupy, accum_dtype=acc_dtype)
    target_abs = _absolute_residual_target(b_norm, rtol=rtol, atol=atol)
    eps = float(np.finfo(op_dtype).eps)
    breakdown = max(float(breakdown_tol), np.finfo(float).tiny)
    givens_residual = cupy.empty(
        (), dtype=np.float64 if acc_dtype == np.dtype(np.complex128) else np.float32
    )
    proxy_history: list[float] = []
    true_history: list[float] = []

    residual = cupy.array(b_vec, dtype=op_dtype, copy=True) if x0_is_zero else b_vec - apply(x)
    residual_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
    true_history.append(float(relative_residual))
    if restart_callback is not None:
        restart_callback(float(relative_residual))

    recycle_space: CuPyRecycleSpace | None = None
    iterations = 0
    cycles = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    if residual_norm <= target_abs:
        info = 0
        converged_reason = "converged"

    while iterations < maxiter_total and info != 0:
        recycle_rank = 0 if recycle_space is None else int(recycle_space.U.shape[1])
        if recycle_space is not None and recycle_rank > 0:
            gamma = _recycle_projection(recycle_space.C, residual, cupy=cupy, accum_dtype=acc_dtype)
            gamma_op = cupy.asarray(gamma, dtype=op_dtype)
            x += recycle_space.U @ gamma_op
            residual -= recycle_space.C @ gamma_op
            projected_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
            if projected_norm <= target_abs:
                residual = b_vec - apply(x)
                residual_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
                relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
                true_history.append(float(relative_residual))
                if restart_callback is not None:
                    restart_callback(float(relative_residual))
                if residual_norm <= target_abs:
                    info = 0
                    converged_reason = "converged"
                    break

        beta = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown or not np.isfinite(beta):
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        new_directions = restart_n if recycle_rank == 0 else max(1, restart_n - recycle_rank)
        cycle_steps = min(new_directions, maxiter_total - iterations)
        V = cupy.empty((cycle_steps + 1, n), dtype=op_dtype)
        H_rot = cupy.empty((cycle_steps, cycle_steps + 1), dtype=acc_dtype)
        Hbar = cupy.zeros((cycle_steps + 1, cycle_steps), dtype=acc_dtype)
        coupling = (
            cupy.empty((recycle_rank, cycle_steps), dtype=acc_dtype) if recycle_rank > 0 else None
        )
        cs = cupy.empty((cycle_steps,), dtype=acc_dtype)
        sn = cupy.empty((cycle_steps,), dtype=acc_dtype)
        g = cupy.empty((cycle_steps + 1,), dtype=acc_dtype)
        V[0, :] = residual / beta
        g[0] = beta

        k_used = 0
        cycle_breakdown = False
        cycle_breakdown_reason = "breakdown"
        proxy_relative = relative_residual

        for col in range(cycle_steps):
            w = apply(V[col, :])
            initial_w_norm = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
            w_owned = False
            if recycle_space is not None and coupling is not None:
                b_col = _recycle_projection(recycle_space.C, w, cupy=cupy, accum_dtype=acc_dtype)
                coupling[:, col] = b_col
                # ``apply`` may return an alias of ``V[col, :]``.  The
                # recycle correction is the first mutating operation, so make
                # ownership explicit before applying it in place.
                w = cupy.array(w, dtype=op_dtype, copy=True)
                w_owned = True
                w -= recycle_space.C @ cupy.asarray(b_col, dtype=op_dtype)

            for row in range(col + 1):
                value = _dot(V[row, :], w, cupy=cupy, accum_dtype=acc_dtype)
                H_rot[col, row] = value
                Hbar[row, col] = value
                w, w_owned = _mgs_subtract_scaled(w, V[row, :], value, cupy=cupy, owns_w=w_owned)
            if reorthogonalize:
                for row in range(col + 1):
                    value = _dot(V[row, :], w, cupy=cupy, accum_dtype=acc_dtype)
                    H_rot[col, row] += value
                    Hbar[row, col] += value
                    w, w_owned = _mgs_subtract_scaled(
                        w, V[row, :], value, cupy=cupy, owns_w=w_owned
                    )

            h_next = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
            H_rot[col, col + 1] = h_next
            Hbar[col + 1, col] = h_next
            happy_tol = max(breakdown, float(happy_breakdown_tol) * max(initial_w_norm, eps))
            if h_next > happy_tol:
                V[col + 1, :] = w / h_next
            else:
                # The row is part of the augmented relation used by harmonic
                # extraction.  ``empty`` storage must never leak an
                # uninitialized vector into that small restart calculation.
                V[col + 1, :] = cupy.asarray(0.0, dtype=op_dtype)
            if h_next <= eps * max(initial_w_norm, eps):
                H_rot[col, col + 1] = cupy.asarray(0.0, dtype=acc_dtype)
                cycle_breakdown = True
                cycle_breakdown_reason = "breakdown"
            elif h_next <= happy_tol:
                H_rot[col, col + 1] = cupy.asarray(0.0, dtype=acc_dtype)
                cycle_breakdown = True
                cycle_breakdown_reason = "happy_breakdown"

            proxy_abs = _apply_givens_rotation(H_rot, cs, sn, g, col, residual=givens_residual)

            k_used = col + 1
            iterations += 1
            proxy_relative = proxy_abs / b_norm if b_norm > 0.0 else proxy_abs
            if record_proxy_history:
                proxy_history.append(float(proxy_relative))
            if callback is not None:
                callback(float(proxy_relative))
            if cycle_breakdown or proxy_abs <= target_abs or iterations >= maxiter_total:
                break

        if k_used <= 0:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        y = _solve_rotated_upper(
            H_rot,
            g,
            k_used=k_used,
            cupy=cupy,
            accum_dtype=acc_dtype,
            breakdown_tol=breakdown,
        )
        y_op = cupy.asarray(y, dtype=op_dtype)
        correction = y_op @ V[:k_used, :]
        if recycle_space is not None and coupling is not None:
            correction -= recycle_space.U @ cupy.asarray(coupling[:, :k_used] @ y, dtype=op_dtype)
        x += cupy.asarray(correction, dtype=op_dtype)

        terminal_by_proxy = bool(
            cycle_breakdown or proxy_abs <= target_abs or iterations >= maxiter_total
        )
        # GCRO needs the physical residual whenever another augmented cycle may
        # follow.  At a terminal boundary the public flag may omit only that
        # final diagnostic/application.
        verify_true = (not terminal_by_proxy) or bool(compute_final_residual)
        if verify_true:
            residual = b_vec - apply(x)
            residual_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
            relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
            true_history.append(float(relative_residual))
            if restart_callback is not None:
                restart_callback(float(relative_residual))
        else:
            residual_norm = float("nan")
            relative_residual = float("nan")
        cycles += 1

        if verify_true and residual_norm <= target_abs:
            info = 0
            converged_reason = "converged"
            del V, H_rot, Hbar, coupling, cs, sn, g, y
            break

        if verify_true and not cycle_breakdown and iterations < maxiter_total:
            candidate = _harmonic_recycle_pair(
                arnoldi_rows=V[: k_used + 1, :],
                hbar=Hbar[: k_used + 1, :k_used],
                old_space=recycle_space,
                coupling=None if coupling is None else coupling[:, :k_used],
                recycle_dim=recycle_limit,
                cupy=cupy,
                operator_dtype=op_dtype,
                accum_dtype=acc_dtype,
            )
            if candidate is not None:
                recycle_space = candidate

        del V, H_rot, Hbar, coupling, cs, sn, g, y
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = cycle_breakdown_reason
            break
        if not verify_true and proxy_abs <= target_abs:
            info = 0
            converged_reason = "converged"
            break
        if iterations >= maxiter_total:
            info = iterations
            converged_reason = "maxiter_reached"
            break

    if compute_final_residual and (not true_history or true_history[-1] != relative_residual):
        residual = b_vec - apply(x)
        residual_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
        relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
        true_history.append(float(relative_residual))
    elif not compute_final_residual:
        residual_norm = float("nan")
        relative_residual = float("nan")

    if residual_norm <= target_abs:
        info = 0
        converged_reason = "converged"
    recycle_rank = 0 if recycle_space is None else int(recycle_space.U.shape[1])
    return CuPyGCRONativeResult(
        x=x,
        info=int(info),
        iterations=int(iterations),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        converged_reason=str(converged_reason),
        preconditioned_history=np.asarray(proxy_history, dtype=float),
        true_history=np.asarray(true_history, dtype=float),
        operator_applications=int(operator_applications),
        restart_cycles=int(cycles),
        recycle_dimension=int(recycle_rank),
    )


__all__ = ["CuPyGCRONativeResult", "CuPyRecycleSpace", "gcro_cupy_native"]
