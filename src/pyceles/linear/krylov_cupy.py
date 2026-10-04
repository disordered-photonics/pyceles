"""Native CuPy Krylov implementations used by pyceles.

This module keeps restarted GMRES state on device memory and avoids delegating
inner-iteration linear-algebra work to host-side SciPy/NumPy routines.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyceles._optional import is_cupy_array


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


@dataclass(frozen=True)
class CuPyBiCGSTABNativeResult:
    """Result payload for native CuPy BiCGSTAB solves.

    ``residual_history`` stores the inexpensive recursive residual norm.
    ``true_history`` stores only independently evaluated physical residuals.
    Keeping the channels separate avoids presenting finite-precision recurrence
    drift as a true-residual history.
    """

    x: Any
    info: int
    iterations: int
    residual_norm: float
    relative_residual: float
    converged_reason: str
    residual_history: np.ndarray
    true_history: np.ndarray
    operator_applications: int
    preconditioner_applications: int


@dataclass(frozen=True)
class CuPyLSQRNativeResult:
    """Result payload for the native matrix-free CuPy LSQR correction solve."""

    x: Any
    info: int
    iterations: int
    residual_norm: float
    relative_residual: float
    rhs_norm: float
    converged_reason: str
    residual_history: np.ndarray
    true_history: np.ndarray
    operator_applications: int
    adjoint_applications: int
    anorm: float
    acond: float
    arnorm: float
    correction_norm: float


def _dtype_complex(dtype: npt.DTypeLike, *, name: str) -> np.dtype:
    out = np.dtype(dtype)
    if out.kind != "c":
        raise TypeError(f"`{name}` must be a complex dtype. Got {out}.")
    return out


def _resolve_accum_dtype(operator_dtype: np.dtype, accum_dtype: npt.DTypeLike | None) -> np.dtype:
    """Resolve accumulation precision without allowing a silent downcast."""

    out = _dtype_complex(
        accum_dtype if accum_dtype is not None else np.result_type(operator_dtype, np.complex128),
        name="accum_dtype",
    )
    if out.itemsize < operator_dtype.itemsize:
        raise ValueError(
            f"`accum_dtype` ({out.name}) must be at least as precise as "
            f"`operator_dtype` ({operator_dtype.name})."
        )
    return out


def _as_device_vector(x: Any, *, cupy: Any, dtype: np.dtype, name: str) -> Any:
    out = cupy.asarray(x, dtype=dtype)
    if int(out.ndim) != 1:
        raise ValueError(f"{name} must be 1D. Got shape={tuple(out.shape)}.")
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


@cache
def _dot_reduction_kernel(input_dtype_name: str, accum_dtype_name: str) -> Any:
    """Return a fused conjugate-dot reduction without widening full vectors."""
    from pyceles._optional import import_cupy

    cupy, _ = import_cupy()
    input_dtype = np.dtype(input_dtype_name)
    accum_dtype = np.dtype(accum_dtype_name)
    reduce_type = "complex<double>" if accum_dtype == np.dtype(np.complex128) else "complex<float>"
    return cupy.ReductionKernel(
        f"{input_dtype.name} x, {input_dtype.name} y",
        f"{accum_dtype.name} z",
        # Widen before multiplication, not just when reducing the product.
        # A complex64 product can lose bits or overflow before a complex128
        # accumulator sees it; casts here require no widened device vectors.
        f"conj(({reduce_type})x) * ({reduce_type})y",
        "a + b",
        "z = a",
        "0",
        f"pyceles_dot_{input_dtype.name}_accum_{accum_dtype.name}",
        reduce_type=reduce_type,
    )


@cache
def _norm_reduction_kernel(input_dtype_name: str, accum_dtype_name: str) -> Any:
    """Return a fused Euclidean-norm reduction without widening the input."""
    from pyceles._optional import import_cupy

    cupy, _ = import_cupy()
    input_dtype = np.dtype(input_dtype_name)
    accum_dtype = np.dtype(accum_dtype_name)
    real_output = "float64" if accum_dtype == np.dtype(np.complex128) else "float32"
    scalar = "double" if real_output == "float64" else "float"
    return cupy.ReductionKernel(
        f"{input_dtype.name} x",
        f"{real_output} y",
        f"(({scalar})x.real() * ({scalar})x.real() + ({scalar})x.imag() * ({scalar})x.imag())",
        "a + b",
        "y = sqrt(a)",
        "0",
        f"pyceles_norm_{input_dtype.name}_accum_{accum_dtype.name}",
    )


def _dot(u: Any, v: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    left = cupy.asarray(u)
    right = cupy.asarray(v)
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"Dot-product operands must have matching shapes. Got {left.shape} and {right.shape}."
        )
    dtype = np.dtype(left.dtype)
    if (
        dtype == np.dtype(right.dtype)
        and dtype in {np.dtype(np.complex64), np.dtype(np.complex128)}
        and hasattr(cupy, "ReductionKernel")
    ):
        kernel = _dot_reduction_kernel(dtype.name, np.dtype(accum_dtype).name)
        return kernel(left, right)
    return cupy.vdot(cupy.asarray(left, dtype=accum_dtype), cupy.asarray(right, dtype=accum_dtype))


def _norm(v: Any, *, cupy: Any, accum_dtype: np.dtype) -> float:
    values = cupy.asarray(v)
    if int(values.size) == 0:
        return 0.0
    dtype = np.dtype(values.dtype)
    if dtype in {np.dtype(np.complex64), np.dtype(np.complex128)} and hasattr(
        cupy, "ReductionKernel"
    ):
        kernel = _norm_reduction_kernel(dtype.name, np.dtype(accum_dtype).name)
        return float(kernel(values))
    return float(cupy.linalg.norm(cupy.asarray(values, dtype=accum_dtype)))


def _norms_block(v: Any, *, cupy: Any, accum_dtype: np.dtype) -> Any:
    """Column norms using the scalar norm's fused map/reduce precision policy."""
    values = cupy.asarray(v)
    dtype = np.dtype(values.dtype)
    if dtype in {np.dtype(np.complex64), np.dtype(np.complex128)} and hasattr(
        cupy, "ReductionKernel"
    ):
        kernel = _norm_reduction_kernel(dtype.name, np.dtype(accum_dtype).name)
        return kernel(values, axis=0)
    return cupy.linalg.norm(cupy.asarray(values, dtype=accum_dtype), axis=0)


def _solve_upper_triangular(r_upper: Any, rhs: Any, *, cupy: Any) -> Any:
    """Solve a small upper-triangular system on the active array backend."""

    if not is_cupy_array(r_upper):
        return cupy.linalg.solve(r_upper, rhs)
    from cupyx.scipy.linalg import solve_triangular

    return solve_triangular(
        r_upper,
        rhs,
        lower=False,
        overwrite_b=False,
        check_finite=False,
    )


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

    # Keep the regular path triangular. If the projected problem has actually
    # lost rank, solve that problem directly rather than hiding it behind
    # normal equations plus an arbitrary ridge (which squares the condition
    # number and changes the GMRES minimization problem).
    diagonal = cupy.abs(cupy.diag(r))
    scale = float(cupy.max(diagonal)) if int(diagonal.size) else 0.0
    smallest = float(cupy.min(diagonal)) if int(diagonal.size) else 0.0
    real_dtype = np.float64 if np.dtype(accum_dtype) == np.dtype(np.complex128) else np.float32
    threshold = np.finfo(real_dtype).eps * max(h_acc.shape) * scale
    if scale > 0.0 and smallest > threshold:
        return _solve_upper_triangular(r, rhs, cupy=cupy)
    return cupy.linalg.lstsq(h_acc, g_acc, rcond=None)[0]


def _row_basis_combination(
    coefficients: Any,
    v_rows: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
    workspace_bytes: int = 64 * 1024**2,
) -> Any:
    """Return ``coefficients @ V`` without promoting the entire row basis.

    CGS coefficients and this contraction retain accumulation precision; only
    the final vector is narrowed to the stored basis dtype. Chunking output
    columns bounds the widened basis plus product to ``workspace_bytes``
    (apart from the minimum one-column case and library workspace), without
    splitting any dot product across chunks. The result allocation is separate.
    """

    basis = cupy.asarray(v_rows)
    weights = cupy.asarray(coefficients, dtype=accum_dtype)
    if np.dtype(basis.dtype) == np.dtype(accum_dtype):
        return weights @ basis

    n_rows, n = (int(value) for value in basis.shape)
    result = cupy.empty((n,), dtype=basis.dtype)
    if n == 0:
        return result
    if n_rows == 0:
        result.fill(0)
        return result
    bytes_per_column = (n_rows + 1) * int(np.dtype(accum_dtype).itemsize)
    chunk = max(1, min(n, int(workspace_bytes) // bytes_per_column))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        # Copy/cast a possibly strided slice directly into one packed buffer.
        widened = cupy.empty((n_rows, stop - start), dtype=accum_dtype)
        widened[...] = basis[:, start:stop]
        result[start:stop] = weights @ widened
        del widened
    return result


def _block_basis_projection(
    v_blocks: Any,
    w: Any,
    *,
    cupy: Any,
    accum_dtype: np.dtype,
    workspace_bytes: int = 64 * 1024**2,
) -> Any:
    """Return ``V^H w`` for a block basis without widening all of ``V``.

    ``v_blocks`` has shape ``(n_blocks, n, p)``.  Only bounded slices of the
    unknown dimension are promoted to accumulation precision; this avoids the
    ``O(n * restart * p)`` complex128 temporary previously created on every
    block-Arnoldi step.
    """

    basis = cupy.asarray(v_blocks)
    values = cupy.asarray(w)
    n_blocks, n, p = (int(value) for value in basis.shape)
    if tuple(values.shape) != (n, p):
        raise ValueError(
            f"Block projection input must have shape ({n}, {p}). Got {tuple(values.shape)}."
        )
    coefficients = cupy.zeros((n_blocks * p, p), dtype=accum_dtype)
    if n_blocks == 0 or n == 0:
        return coefficients

    itemsize = int(np.dtype(accum_dtype).itemsize)
    live_columns = max(1, n_blocks * p + p)
    chunk = max(1, min(n, int(workspace_bytes) // (live_columns * itemsize)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        # Conjugate, pack and widen in one ufunc write into owned workspace.
        # Reshaping the transposed basis first can copy it; then astype and
        # conj would each allocate another full chunk. Never conjugate a view
        # returned by asarray in place: it may still alias the Krylov basis.
        left = cupy.empty((stop - start, n_blocks, p), dtype=accum_dtype)
        cupy.conj(basis[:, start:stop, :].transpose(1, 0, 2), out=left)
        right = cupy.asarray(values[start:stop, :], dtype=accum_dtype)
        coefficients += left.reshape(stop - start, n_blocks * p).T @ right
        # Do not retain the previous chunk while allocating the next one.
        del left, right
    return coefficients


def _subtract_block_basis_projection(
    w: Any,
    v_blocks: Any,
    coefficients: Any,
    *,
    cupy: Any,
) -> None:
    """Apply ``w -= V @ coefficients`` without packing the full basis."""

    basis = cupy.asarray(v_blocks)
    n_blocks, _, p = (int(value) for value in basis.shape)
    coeff_blocks = cupy.asarray(coefficients, dtype=basis.dtype).reshape(n_blocks, p, p)
    correction = cupy.einsum("knp,kpq->nq", basis, coeff_blocks, optimize=True)
    cupy.subtract(w, correction, out=w)


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
) -> Any:
    """Apply a block-capable operator and validate its matrix shape."""

    x_mat = _as_device_matrix(x, cupy=cupy, dtype=dtype, name="x_block")
    y = _as_device_matrix(
        op(x_mat),
        cupy=cupy,
        dtype=dtype,
        name="op(X)",
        n_rows=int(x_mat.shape[0]),
    )
    if int(y.shape[1]) != int(x_mat.shape[1]):
        raise ValueError(
            "Block operator must preserve the RHS column count. "
            f"Got input {int(x_mat.shape[1])}, output {int(y.shape[1])}."
        )
    return y


def _apply_block_preconditioner(
    preconditioner: Callable[[Any], Any] | None,
    x: Any,
    *,
    cupy: Any,
    dtype: np.dtype,
) -> Any:
    """Apply a block-capable preconditioner without legacy column adapters."""

    x_mat = _as_device_matrix(x, cupy=cupy, dtype=dtype, name="X")
    if preconditioner is None:
        return x_mat
    y = _as_device_matrix(
        preconditioner(x_mat),
        cupy=cupy,
        dtype=dtype,
        name="M^-1(X)",
        n_rows=int(x_mat.shape[0]),
    )
    if int(y.shape[1]) != int(x_mat.shape[1]):
        raise ValueError(
            "Block preconditioner must preserve the RHS column count. "
            f"Got input {int(x_mat.shape[1])}, output {int(y.shape[1])}."
        )
    return y


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


def _solve_rotated_upper(
    h_rows: Any,
    g: Any,
    *,
    k_used: int,
    cupy: Any,
    accum_dtype: np.dtype,
    breakdown_tol: float,
) -> Any:
    """Solve one rotated GMRES projected system entirely on device.

    The regular path is triangular.  Only an actual near-singular projected
    system falls back to a small least-squares solve; there is no Python
    scalar back-substitution or broad exception-based compatibility path.
    """

    if int(k_used) <= 0:
        raise ValueError("`k_used` must be positive.")
    r_upper = cupy.asarray(h_rows[:k_used, :k_used].T, dtype=accum_dtype)
    rhs = cupy.asarray(g[:k_used], dtype=accum_dtype)
    last = int(k_used) - 1
    if float(cupy.abs(r_upper[last, last])) > float(breakdown_tol):
        return _solve_upper_triangular(r_upper, rhs, cupy=cupy)
    return cupy.linalg.lstsq(r_upper, rhs, rcond=None)[0]


def lsqr_cupy_native(
    A_mv: Callable[[Any], Any],
    A_h_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    initial_residual: Any | None = None,
    rhs_norm: float | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    operator_dtype: npt.DTypeLike | None = None,
    accum_dtype: npt.DTypeLike | None = None,
    callback: Callable[[float], None] | None = None,
    true_residual_every: int = 0,
    compute_final_residual: bool = True,
    condition_limit: float | None = None,
) -> CuPyLSQRNativeResult:
    """Run a matrix-free LSQR correction solve on CuPy.

    ``A_mv`` and ``A_h_mv`` must be the production action and its exact
    Hermitian adjoint.  Vector storage follows ``operator_dtype`` while all
    norms, inner products, and scalar recurrences follow ``accum_dtype``.
    The returned iterate solves ``A x = b`` from ``x0``; LSQR internally
    applies the correction equation to ``b - A x0`` (or the supplied
    ``initial_residual``) and never stores a Krylov basis.  ``rhs_norm`` keeps
    tolerance normalization tied to the original right-hand side when a
    correction residual is supplied.  ``true_residual_every`` is opt-in
    because a physical residual check costs another production forward
    action.
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
    if int(true_residual_every) < 0:
        raise ValueError("`true_residual_every` must be non-negative.")
    if condition_limit is not None and not (float(condition_limit) > 0.0):
        raise ValueError("`condition_limit` must be positive when provided.")
    b_vec = _as_device_vector(b, cupy=cupy, dtype=op_dtype, name="b")
    n = int(b_vec.size)
    maxiter_total = int(maxiter) if maxiter is not None else max(1, 2 * n)
    if maxiter_total < 1:
        raise ValueError("`maxiter` must be >= 1 when provided.")
    x0_vec = None if x0 is None else _as_device_vector(x0, cupy=cupy, dtype=op_dtype, name="x0")

    def _apply(action: Callable[[Any], Any], values: Any, name: str) -> Any:
        return _as_device_vector(
            action(cupy.asarray(values, dtype=op_dtype)),
            cupy=cupy,
            dtype=op_dtype,
            name=name,
        )

    def _scalar(value: Any) -> Any:
        return cupy.asarray(value, dtype=acc_dtype)

    if rhs_norm is None:
        b_norm = _norm(b_vec, cupy=cupy, accum_dtype=acc_dtype)
    else:
        b_norm = float(rhs_norm)
        if not np.isfinite(b_norm) or b_norm < 0.0:
            raise ValueError("`rhs_norm` must be finite and non-negative when provided.")
    target_abs = max(float(atol), float(rtol) * b_norm)
    operator_applications = 0
    adjoint_applications = 0
    true_history: list[float] = []

    if initial_residual is not None:
        residual = _as_device_vector(
            initial_residual, cupy=cupy, dtype=op_dtype, name="initial_residual"
        )
        if int(residual.size) != n:
            raise ValueError(
                f"`initial_residual` size {int(residual.size)} does not match `b` size {n}."
            )
    elif x0_vec is None:
        residual = cupy.asarray(b_vec, dtype=op_dtype)
    else:
        # A supplied warm start is semantically a real warm start.  Do not
        # scan the complete device vector merely to special-case an all-zero
        # value; callers that already know the residual can pass
        # ``initial_residual`` and avoid this production action entirely.
        residual = b_vec - _apply(A_mv, x0_vec, "A(x0)")
        operator_applications += 1
    residual_norm = _norm(residual, cupy=cupy, accum_dtype=acc_dtype)
    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
    true_history.append(float(relative_residual))
    if residual_norm <= target_abs:
        # The scattering system is square.  Avoid an unnecessary adjoint probe
        # merely to infer the solution shape when a supplied warm start already
        # has the correct dimension.
        solution = cupy.zeros_like(b_vec, dtype=op_dtype) if x0_vec is None else x0_vec
        return CuPyLSQRNativeResult(
            x=solution,
            info=0,
            iterations=0,
            residual_norm=float(residual_norm),
            relative_residual=float(relative_residual),
            rhs_norm=float(b_norm),
            converged_reason="converged",
            residual_history=np.asarray([relative_residual], dtype=float),
            true_history=np.asarray(true_history, dtype=float),
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            anorm=0.0,
            acond=0.0,
            arnorm=0.0,
            correction_norm=0.0,
        )

    beta = residual_norm
    u = residual / beta
    v = _apply(A_h_mv, u, "A^H(u)")
    adjoint_applications += 1
    if x0_vec is not None and int(x0_vec.size) != int(v.size):
        raise ValueError(
            f"`x0` size {int(x0_vec.size)} does not match the solution dimension {int(v.size)}."
        )
    # Keep the caller's warm-start array immutable.  The update below is
    # deliberately performed in-place into ``x_vec`` to avoid allocating a
    # full solution-sized temporary on every bidiagonalization step.
    x_vec = (
        cupy.zeros_like(v, dtype=op_dtype)
        if x0_vec is None
        else cupy.asarray(x0_vec, dtype=op_dtype).copy()
    )
    alpha = _norm(v, cupy=cupy, accum_dtype=acc_dtype)
    if alpha == 0.0:
        return CuPyLSQRNativeResult(
            x=x_vec,
            info=maxiter_total,
            iterations=0,
            residual_norm=float(residual_norm),
            relative_residual=float(relative_residual),
            rhs_norm=float(b_norm),
            converged_reason="breakdown",
            residual_history=np.asarray([relative_residual], dtype=float),
            true_history=np.asarray(true_history, dtype=float),
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            anorm=0.0,
            acond=float("inf"),
            arnorm=0.0,
            correction_norm=0.0,
        )
    cupy.divide(v, alpha, out=v)
    # ``w`` evolves independently of the Lanczos vector.  One additional
    # solution-sized scratch buffer is sufficient for all vector recurrences:
    # it is used first for bidiagonal subtraction, then for the solution
    # increment, and finally becomes the next ``w`` after a reference swap.
    w = cupy.asarray(v, dtype=op_dtype).copy()
    scratch = cupy.empty_like(w)
    phibar = _scalar(beta)
    rhobar = _scalar(alpha)
    residual_history: list[float] = []
    # Paige--Saunders diagnostics.  ``anorm`` starts at zero and each
    # bidiagonalization step contributes the *current* alpha and newly formed
    # beta.  Initializing with alpha^2+beta^2 and then adding alpha_next/beta_next
    # double-counts the bidiagonal terms and overestimates both anorm and acond.
    anorm_sq = _scalar(0.0)
    ddnorm = _scalar(0.0)
    anorm = 0.0
    acond = 0.0
    arnorm = float(alpha * beta)
    correction_norm = 0.0
    xxnorm = _scalar(0.0)
    z = _scalar(0.0)
    cs2 = _scalar(-1.0)
    sn2 = _scalar(0.0)
    info = maxiter_total
    converged_reason = "maxiter_reached"
    iterations = 0

    for iteration in range(1, maxiter_total + 1):
        u_next = _apply(A_mv, v, "A(v)")
        operator_applications += 1
        cupy.multiply(u, _scalar(alpha).astype(op_dtype), out=scratch)
        cupy.subtract(u_next, scratch, out=u_next)
        beta_next = _norm(u_next, cupy=cupy, accum_dtype=acc_dtype)
        # Paige--Saunders ||A|| estimate includes the current bidiagonal
        # alpha even when the newly formed beta is exactly zero.  Keeping the
        # update outside the normalization branch matters for condition-limit
        # stopping on exact/happy bidiagonal termination.
        anorm_sq = anorm_sq + _scalar(alpha) ** 2 + _scalar(beta_next) ** 2
        anorm = float(cupy.sqrt(cupy.asarray(anorm_sq).real))
        if beta_next > 0.0:
            cupy.divide(u_next, beta_next, out=u_next)

        v_next = _apply(A_h_mv, u_next, "A^H(u)")
        adjoint_applications += 1
        cupy.multiply(v, _scalar(beta_next).astype(op_dtype), out=scratch)
        cupy.subtract(v_next, scratch, out=v_next)
        alpha_next = _norm(v_next, cupy=cupy, accum_dtype=acc_dtype)
        if alpha_next > 0.0:
            cupy.divide(v_next, alpha_next, out=v_next)
        rho = float(cupy.sqrt(cupy.abs(rhobar) ** 2 + beta_next**2))
        if rho == 0.0 or not np.isfinite(rho):
            converged_reason = "breakdown"
            info = iterations if iterations else maxiter_total
            break
        c = rhobar / rho
        s = _scalar(beta_next / rho)
        theta = s * _scalar(alpha_next)
        rhobar = -c * _scalar(alpha_next)
        phi = c * phibar
        phibar = s * phibar
        tau = s * phi
        w_norm = _norm(w, cupy=cupy, accum_dtype=acc_dtype)
        ddnorm = ddnorm + _scalar(w_norm / rho) ** 2
        # Reuse one scratch vector for both the solution increment and the
        # next search direction.  The old ``w`` becomes the scratch buffer
        # after the reference swap.
        cupy.multiply(w, _scalar(phi / rho).astype(op_dtype), out=scratch)
        cupy.add(x_vec, scratch, out=x_vec)
        cupy.multiply(w, _scalar(theta / rho).astype(op_dtype), out=scratch)
        cupy.subtract(v_next, scratch, out=scratch)
        w, scratch = scratch, w
        u, v = u_next, v_next
        alpha, beta = alpha_next, beta_next
        iterations = iteration

        # Right rotation used by the original LSQR recurrence to estimate the
        # correction norm without another full-vector reduction.
        delta = sn2 * _scalar(rho)
        gambar = -cs2 * _scalar(rho)
        rotated_rhs = phi - delta * z
        if float(cupy.abs(gambar)) > 0.0:
            zbar = rotated_rhs / gambar
            correction_norm = float(cupy.sqrt(cupy.asarray(xxnorm + cupy.abs(zbar) ** 2).real))
        gamma = float(cupy.sqrt(cupy.abs(gambar) ** 2 + cupy.abs(theta) ** 2))
        if gamma > 0.0:
            cs2 = gambar / _scalar(gamma)
            sn2 = theta / _scalar(gamma)
            z = rotated_rhs / _scalar(gamma)
            xxnorm = xxnorm + cupy.abs(z) ** 2

        acond = anorm * float(cupy.sqrt(cupy.asarray(ddnorm).real))
        arnorm = float(alpha_next * cupy.abs(tau))
        recursive_rel = (
            float(cupy.abs(phibar)) / b_norm if b_norm > 0.0 else float(cupy.abs(phibar))
        )
        residual_history.append(float(recursive_rel))
        if callback is not None:
            callback(float(recursive_rel))

        if condition_limit is not None and acond >= float(condition_limit):
            info = iterations
            converged_reason = "condition_limit"
            break

        should_check = bool(true_residual_every) and iteration % int(true_residual_every) == 0
        if should_check:
            candidate = x_vec
            true_vec = b_vec - _apply(A_mv, candidate, "A(x)")
            operator_applications += 1
            true_abs = _norm(true_vec, cupy=cupy, accum_dtype=acc_dtype)
            true_rel = true_abs / b_norm if b_norm > 0.0 else true_abs
            true_history.append(float(true_rel))
            if true_abs <= target_abs:
                residual_norm, relative_residual = true_abs, true_rel
                info = 0
                converged_reason = "converged"
                break

        if recursive_rel <= (
            max(float(atol), float(rtol) * b_norm) / b_norm if b_norm > 0.0 else float(atol)
        ):
            if compute_final_residual:
                true_vec = b_vec - _apply(A_mv, x_vec, "A(x)")
                operator_applications += 1
                true_abs = _norm(true_vec, cupy=cupy, accum_dtype=acc_dtype)
                true_rel = true_abs / b_norm if b_norm > 0.0 else true_abs
                true_history.append(float(true_rel))
                residual_norm, relative_residual = true_abs, true_rel
                if true_abs <= target_abs:
                    info = 0
                    converged_reason = "converged"
                    break
            else:
                residual_norm = float(cupy.abs(phibar))
                relative_residual = recursive_rel
                info = 0
                converged_reason = "converged_recursive"
                break
        if beta_next == 0.0 and alpha_next == 0.0:
            info = iterations
            converged_reason = "breakdown"
            break

    if info != 0:
        if compute_final_residual:
            true_vec = b_vec - _apply(A_mv, x_vec, "A(x)")
            operator_applications += 1
            residual_norm = _norm(true_vec, cupy=cupy, accum_dtype=acc_dtype)
            relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
            true_history.append(float(relative_residual))
        else:
            residual_norm = float(cupy.abs(phibar))
            relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm

    return CuPyLSQRNativeResult(
        x=x_vec,
        info=int(info),
        iterations=int(iterations),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        rhs_norm=float(b_norm),
        converged_reason=str(converged_reason),
        residual_history=np.asarray(residual_history, dtype=float),
        true_history=np.asarray(true_history, dtype=float),
        operator_applications=int(operator_applications),
        adjoint_applications=int(adjoint_applications),
        anorm=float(anorm),
        acond=float(acond),
        arnorm=float(arnorm),
        correction_norm=float(correction_norm),
    )


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
    compute_final_residual: bool = True,
) -> CuPyGMRESNativeResult:
    """Run restarted left-preconditioned GMRES fully on CuPy arrays.

    Inner-iteration callbacks receive the GMRES preconditioned residual proxy
    from the Arnoldi/Givens recurrence. A physical residual is rebuilt whenever
    another restarted cycle is required. ``compute_final_residual=False`` skips
    only a terminal residual evaluation when no further cycle will be started.
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
    # ``compute_final_residual`` controls only the optional terminal diagnostic.
    # A physical residual is always rebuilt when another restarted cycle is
    # required; otherwise the next Krylov space would start from the wrong
    # vector even if the Arnoldi residual norm proxy were accurate.
    op_dtype = _dtype_complex(
        operator_dtype if operator_dtype is not None else np.result_type(b_dtype, np.complex64),
        name="operator_dtype",
    )
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
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
    ptol = Mb_norm * min(1.0, target_abs / b_norm) if b_norm > 0 else target_abs
    precond_hist: list[float] = []
    true_hist: list[float] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    residual_norm: float
    relative_residual: float
    need_pr_rel = bool(callback is not None or record_preconditioned_history)

    def _record_true_residual(rel_norm: float) -> None:
        true_hist.append(float(rel_norm))
        if restart_callback is not None:
            restart_callback(float(rel_norm))

    if x0_is_zero:
        residual_norm = float(b_norm)
        relative_residual = 1.0 if b_norm > 0 else 0.0
        r_true = cupy.asarray(b_vec, dtype=op_dtype)
        x0_is_zero = False
    else:
        residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
    _record_true_residual(relative_residual)
    if residual_norm <= target_abs:
        info = 0
        converged_reason = "converged"

    while iterations < maxiter_total and info != 0:
        z0 = _apply_minv(r_true)
        beta = _norm(z0, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        # Every Arnoldi vector is written before it is read.  Avoid clearing
        # the full basis (which can be a substantial device-memory bandwidth
        # cost for large restarts); Hessenberg and recurrence arrays below
        # remain zero-initialized because their structural zeros are used.
        V = cupy.empty((cycle_steps + 1, n), dtype=op_dtype)
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
                w = w - _row_basis_combination(
                    h_row, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                )
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
                    w = w - _row_basis_combination(
                        h_row2, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                    )

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

        y = _solve_rotated_upper(
            H,
            g,
            k_used=k_used,
            cupy=cupy,
            accum_dtype=acc_dtype,
            breakdown_tol=max(breakdown_tol_f, eps),
        )
        # The projected coefficients are solved in accumulation precision, but
        # the stored Arnoldi basis remains in operator precision. Cast the
        # small coefficient vector first; otherwise CuPy promotes the entire
        # ``(k_used, n)`` basis to complex128 for this product.
        y_op = cupy.asarray(y, dtype=op_dtype)
        x_vec = x_vec + cupy.asarray(y_op @ V[:k_used, :], dtype=op_dtype)
        # The next restart allocates a fresh basis.  Drop the completed cycle
        # before any physical-residual matvec so two full Arnoldi bases cannot
        # overlap at the restart boundary.
        del V, H, cs, sn, g, y, z0
        terminal_by_proxy = bool(
            cycle_breakdown or cycle_presid <= ptol or iterations >= maxiter_total
        )
        # A physical residual is mandatory whenever another restarted cycle may
        # follow.  At a terminal boundary it is optional and controlled by the
        # public ``compute_final_residual`` policy.
        if (not terminal_by_proxy) or bool(compute_final_residual):
            residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
            _record_true_residual(relative_residual)
            if residual_norm <= target_abs:
                info = 0
                converged_reason = "converged"
                break
            if cycle_breakdown:
                info = iterations if iterations > 0 else maxiter_total
                converged_reason = cycle_breakdown_reason
                break
            if iterations >= maxiter_total:
                info = iterations
                converged_reason = "maxiter_reached"
                break
            continue
        residual_norm = float("nan")
        relative_residual = float("nan")
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = cycle_breakdown_reason
        elif cycle_presid <= ptol:
            info = 0
            converged_reason = "converged"
        else:
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
    compute_final_residual: bool = True,
) -> CuPyGMRESNativeResult:
    """Run restarted right-preconditioned flexible GMRES on CuPy arrays.

    Compared with standard GMRES, FGMRES stores both Krylov basis vectors `V`
    and preconditioned vectors `Z_j = M_j^{-1} V_j`, allowing the
    preconditioner to vary by iteration. A physical residual is rebuilt whenever
    another restarted cycle is required. ``compute_final_residual=False`` skips
    only a terminal residual evaluation when no further cycle will be started.
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
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
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
    ptol = target_abs
    precond_hist: list[float] = []
    true_hist: list[float] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    residual_norm: float
    relative_residual: float
    need_pr_rel = bool(callback is not None or record_preconditioned_history)

    def _record_true_residual(rel_norm: float) -> None:
        true_hist.append(float(rel_norm))
        if restart_callback is not None:
            restart_callback(float(rel_norm))

    if x0_is_zero:
        residual_norm = float(b_norm)
        relative_residual = 1.0 if b_norm > 0 else 0.0
        r_true = cupy.asarray(b_vec, dtype=op_dtype)
        x0_is_zero = False
    else:
        residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
    _record_true_residual(relative_residual)
    if residual_norm <= target_abs:
        info = 0
        converged_reason = "converged"

    while iterations < maxiter_total and info != 0:
        beta = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        V = cupy.empty((cycle_steps + 1, n), dtype=op_dtype)
        Z = cupy.empty((cycle_steps, n), dtype=op_dtype)
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
            # An absent/identity preconditioner returns a view of V[col].
            # Keeping z until the next iteration would pin the completed V
            # across restart cleanup and the next basis allocation.
            del z
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
                w = w - _row_basis_combination(
                    h_row, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                )
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
                    w = w - _row_basis_combination(
                        h_row2, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                    )

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

        y = _solve_rotated_upper(
            H,
            g,
            k_used=k_used,
            cupy=cupy,
            accum_dtype=acc_dtype,
            breakdown_tol=max(breakdown_tol_f, eps),
        )
        y_op = cupy.asarray(y, dtype=op_dtype)
        x_vec = x_vec + cupy.asarray(y_op @ Z[:k_used, :], dtype=op_dtype)
        # The next restart allocates a fresh basis.  Drop the completed cycle
        # before any physical-residual matvec so two full Arnoldi bases cannot
        # overlap at the restart boundary.
        del V, Z, H, cs, sn, g, y
        terminal_by_proxy = bool(
            cycle_breakdown or cycle_presid <= ptol or iterations >= maxiter_total
        )
        # A physical residual is mandatory whenever another restarted cycle may
        # follow.  At a terminal boundary it is optional and controlled by the
        # public ``compute_final_residual`` policy.
        if (not terminal_by_proxy) or bool(compute_final_residual):
            residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
            _record_true_residual(relative_residual)
            if residual_norm <= target_abs:
                info = 0
                converged_reason = "converged"
                break
            if cycle_breakdown:
                info = iterations if iterations > 0 else maxiter_total
                converged_reason = cycle_breakdown_reason
                break
            if iterations >= maxiter_total:
                info = iterations
                converged_reason = "maxiter_reached"
                break
            continue
        residual_norm = float("nan")
        relative_residual = float("nan")
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = cycle_breakdown_reason
        elif cycle_presid <= ptol:
            info = 0
            converged_reason = "converged"
        else:
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
    compute_final_residual: bool = True,
) -> CuPyGMRESNativeResult:
    """Run restarted right-preconditioned LGMRES on CuPy arrays.

    LGMRES augments each restarted cycle with up to ``outer_k`` normalized
    correction directions from previous cycles, which often mitigates restart
    stagnation versus plain restarted GMRES at similar memory footprint. A
    physical residual is rebuilt whenever another restarted cycle is required.
    ``compute_final_residual=False`` skips only a terminal residual evaluation
    when no further cycle will be started.
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
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
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
    ptol = target_abs
    precond_hist: list[float] = []
    true_hist: list[float] = []
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"
    residual_norm: float
    relative_residual: float
    need_pr_rel = bool(callback is not None or record_preconditioned_history)
    outer_v: list[tuple[Any, Any | None]] = []

    def _record_true_residual(rel_norm: float) -> None:
        true_hist.append(float(rel_norm))
        if restart_callback is not None:
            restart_callback(float(rel_norm))

    if x0_is_zero:
        residual_norm = float(b_norm)
        relative_residual = 1.0 if b_norm > 0 else 0.0
        r_true = cupy.asarray(b_vec, dtype=op_dtype)
        x0_is_zero = False
    else:
        residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
    _record_true_residual(relative_residual)
    if residual_norm <= target_abs:
        info = 0
        converged_reason = "converged"

    while iterations < maxiter_total and info != 0:
        z0 = _apply_minv(r_true)
        beta = _norm(z0, cupy=cupy, accum_dtype=acc_dtype)
        if beta <= breakdown_tol_f:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = "breakdown"
            break

        cycle_steps = min(restart_n, maxiter_total - iterations)
        aug_count = min(len(outer_v), outer_keep)
        total_steps = cycle_steps + aug_count
        V = cupy.empty((total_steps + 1, n), dtype=op_dtype)
        Z = cupy.empty((total_steps, n), dtype=op_dtype)
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
            # z may be a view of the Arnoldi basis, not an owned vector.
            del z
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
                w = w - _row_basis_combination(
                    h_row, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                )
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
                    w = w - _row_basis_combination(
                        h_row2, V[: col + 1, :], cupy=cupy, accum_dtype=acc_dtype
                    )

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

        y = _solve_rotated_upper(
            H,
            g,
            k_used=k_used,
            cupy=cupy,
            accum_dtype=acc_dtype,
            breakdown_tol=max(breakdown_tol_f, eps),
        )
        y_op = cupy.asarray(y, dtype=op_dtype)
        dx = cupy.asarray(y_op @ Z[:k_used, :], dtype=op_dtype)
        x_vec = x_vec + dx

        if outer_keep > 0 and iterations < maxiter_total:
            dx_norm = _norm(dx, cupy=cupy, accum_dtype=acc_dtype)
            if dx_norm > breakdown_tol_f:
                dx_unit = cupy.asarray(dx / dx_norm, dtype=op_dtype)
                ax_unit = _apply(A_mv, dx_unit) if bool(store_outer_av) else None
                outer_v.append((dx_unit, ax_unit))
                while len(outer_v) > outer_keep:
                    del outer_v[0]

        # The next restart allocates a fresh basis.  Drop the completed cycle
        # before any physical-residual matvec so two full Arnoldi bases cannot
        # overlap at the restart boundary.
        del V, Z, H, cs, sn, g, y, z0
        terminal_by_proxy = bool(
            cycle_breakdown or cycle_presid <= ptol or iterations >= maxiter_total
        )
        # A physical residual is mandatory whenever another restarted cycle may
        # follow.  At a terminal boundary it is optional and controlled by the
        # public ``compute_final_residual`` policy.
        if (not terminal_by_proxy) or bool(compute_final_residual):
            residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
            _record_true_residual(relative_residual)
            if residual_norm <= target_abs:
                info = 0
                converged_reason = "converged"
                break
            if cycle_breakdown:
                info = iterations if iterations > 0 else maxiter_total
                converged_reason = cycle_breakdown_reason
                break
            if iterations >= maxiter_total:
                info = iterations
                converged_reason = "maxiter_reached"
                break
            continue
        residual_norm = float("nan")
        relative_residual = float("nan")
        if cycle_breakdown:
            info = iterations if iterations > 0 else maxiter_total
            converged_reason = cycle_breakdown_reason
        elif cycle_presid <= ptol:
            info = 0
            converged_reason = "converged"
        else:
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


def bicgstab_cupy_native(
    A_mv: Callable[[Any], Any],
    b: Any,
    *,
    cupy: Any,
    x0: Any | None = None,
    preconditioner: Callable[[Any], Any] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int | None = None,
    operator_dtype: npt.DTypeLike | None = None,
    accum_dtype: npt.DTypeLike | None = None,
    callback: Callable[[float], None] | None = None,
    callback_true: Callable[[float], None] | None = None,
    breakdown_tol: float = 1e-30,
    compute_final_residual: bool = True,
) -> CuPyBiCGSTABNativeResult:
    """Run right-preconditioned BiCGSTAB fully on CuPy arrays.

    The update order follows the canonical stabilized BiCG recurrence used by
    PETSc.  Recursive residuals are cheap and are reported every iteration,
    but they are not labelled as physical residuals: when the recurrence first
    meets the requested tolerance, an independently evaluated ``b - A @ x``
    gates convergence.  If finite-precision drift makes that gate fail, the
    recurrence is restarted from the true residual rather than returning a
    false convergence result.

    Vector recurrences are updated in place where dependencies permit.  This
    avoids several Krylov-sized temporaries per iteration on the CuPy backend.
    """
    b_dtype_obj = getattr(b, "dtype", None)
    b_dtype = np.dtype(np.asarray(b).dtype if b_dtype_obj is None else b_dtype_obj)
    op_dtype = _dtype_complex(
        operator_dtype if operator_dtype is not None else np.result_type(b_dtype, np.complex64),
        name="operator_dtype",
    )
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
    b_vec = _as_device_vector(b, cupy=cupy, dtype=op_dtype, name="b")
    n = int(b_vec.size)
    maxiter_total = int(maxiter) if maxiter is not None else n * 10
    if n == 0:
        return CuPyBiCGSTABNativeResult(
            x=cupy.asarray(b_vec, dtype=op_dtype),
            info=0,
            iterations=0,
            residual_norm=0.0,
            relative_residual=0.0,
            converged_reason="converged",
            residual_history=np.asarray([0.0], dtype=float),
            true_history=np.asarray([0.0], dtype=float),
            operator_applications=0,
            preconditioner_applications=0,
        )
    if maxiter_total < 1:
        raise ValueError("`maxiter` must be >= 1 when provided.")
    if float(rtol) < 0.0 or float(atol) < 0.0:
        raise ValueError("`rtol` and `atol` must be non-negative.")

    x_vec = (
        cupy.zeros_like(b_vec, dtype=op_dtype)
        if x0 is None
        else _as_device_vector(x0, cupy=cupy, dtype=op_dtype, name="x0").copy()
    )
    if int(x_vec.size) != n:
        raise ValueError(f"`x0` size {int(x_vec.size)} does not match `b` size {n}.")
    x0_is_zero = x0 is None

    operator_applications = 0
    preconditioner_applications = 0

    def _apply(vec: Any) -> Any:
        nonlocal operator_applications
        operator_applications += 1
        return _as_device_vector(
            A_mv(cupy.asarray(vec, dtype=op_dtype)),
            cupy=cupy,
            dtype=op_dtype,
            name="op(x)",
        )

    def _apply_minv(vec: Any) -> Any:
        nonlocal preconditioner_applications
        if preconditioner is None:
            return cupy.asarray(vec, dtype=op_dtype)
        preconditioner_applications += 1
        return _as_device_vector(
            preconditioner(cupy.asarray(vec, dtype=op_dtype)),
            cupy=cupy,
            dtype=op_dtype,
            name="M^-1(x)",
        )

    def _true_residual_stats(x_curr: Any) -> tuple[float, float, Any]:
        r_true = b_vec - _apply(x_curr)
        abs_norm = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        rel_norm = abs_norm / b_norm if b_norm > 0.0 else abs_norm
        return abs_norm, rel_norm, r_true

    b_norm = _norm(b_vec, cupy=cupy, accum_dtype=acc_dtype)
    target_abs = max(float(atol), float(rtol) * b_norm)
    breakdown_tol_f = float(max(0.0, breakdown_tol))
    residual_hist: list[float] = []
    true_hist: list[float] = []

    if x0_is_zero:
        r_vec = cupy.asarray(b_vec, dtype=op_dtype).copy()
        residual_norm = float(b_norm)
        relative_residual = 1.0 if b_norm > 0.0 else 0.0
    else:
        residual_norm, relative_residual, r_vec = _true_residual_stats(x_vec)
    residual_hist.append(relative_residual)
    true_hist.append(relative_residual)
    if residual_norm <= target_abs:
        return CuPyBiCGSTABNativeResult(
            x=x_vec,
            info=0,
            iterations=0,
            residual_norm=float(residual_norm),
            relative_residual=float(relative_residual),
            converged_reason="converged",
            residual_history=np.asarray(residual_hist, dtype=float),
            true_history=np.asarray(true_hist, dtype=float),
            operator_applications=operator_applications,
            preconditioner_applications=preconditioner_applications,
        )

    r_hat = cupy.asarray(r_vec, dtype=op_dtype).copy()
    p_vec: Any | None = None
    v_vec: Any | None = None
    rho_old = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
    alpha = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
    omega = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
    recurrence_fresh = True
    iterations = 0
    info = maxiter_total
    converged_reason = "maxiter_reached"

    def _record_recursive(norm_abs: float) -> float:
        rel = norm_abs / b_norm if b_norm > 0.0 else norm_abs
        residual_hist.append(rel)
        if callback is not None:
            callback(rel)
        return rel

    def _true_gate() -> tuple[bool, Any]:
        nonlocal residual_norm, relative_residual
        residual_norm, relative_residual, r_true = _true_residual_stats(x_vec)
        true_hist.append(relative_residual)
        if callback_true is not None:
            callback_true(relative_residual)
        return bool(np.isfinite(residual_norm) and residual_norm <= target_abs), r_true

    for k in range(maxiter_total):
        rho = _dot(r_hat, r_vec, cupy=cupy, accum_dtype=acc_dtype)
        if float(cupy.abs(rho)) <= breakdown_tol_f:
            iterations = k + 1
            info = k + 1
            converged_reason = "breakdown_rho"
            break

        if recurrence_fresh:
            p_vec = cupy.asarray(r_vec, dtype=op_dtype).copy()
            recurrence_fresh = False
        else:
            if p_vec is None or v_vec is None:
                raise RuntimeError("Internal CuPy BiCGSTAB error: missing recurrence state.")
            if float(cupy.abs(omega)) <= breakdown_tol_f:
                iterations = k + 1
                info = k + 1
                converged_reason = "breakdown_omega"
                break
            beta = (rho / rho_old) * (alpha / omega)
            beta_op = cupy.asarray(beta, dtype=op_dtype)
            omega_op = cupy.asarray(omega, dtype=op_dtype)
            # p <- r + beta * (p - omega*v), in place to avoid two n-vectors.
            p_vec -= omega_op * v_vec
            p_vec *= beta_op
            p_vec += r_vec
            v_vec = None

        phat = _apply_minv(p_vec)
        v_vec = _apply(phat)
        d1 = _dot(r_hat, v_vec, cupy=cupy, accum_dtype=acc_dtype)
        if float(cupy.abs(d1)) <= breakdown_tol_f:
            iterations = k + 1
            info = k + 1
            converged_reason = "breakdown_alpha"
            break

        alpha = rho / d1
        alpha_op = cupy.asarray(alpha, dtype=op_dtype)
        # r is owned: initialization copies b, and reliable restarts take
        # ownership of b - A(x). Its old value is dead once p and alpha are
        # formed; r_hat and the fresh p were copied separately. Reuse r for s
        # rather than copying a full vector every iteration.
        s_vec = r_vec
        s_vec -= alpha_op * v_vec
        s_norm = _norm(s_vec, cupy=cupy, accum_dtype=acc_dtype)
        if s_norm <= target_abs:
            x_vec += alpha_op * phat
            # The true-residual gate can need substantial operator workspace.
            # phat is dead after this update (and can alias p without M).
            phat = None
            iterations = k + 1
            _record_recursive(s_norm)
            if not compute_final_residual:
                residual_norm = float(s_norm)
                relative_residual = residual_hist[-1]
                info = 0
                converged_reason = "converged_recursive"
                break
            converged, r_true = _true_gate()
            if converged:
                info = 0
                converged_reason = "converged"
                break
            # Reliable restart after a false recursive convergence gate.
            r_vec = cupy.asarray(r_true, dtype=op_dtype)
            r_hat = cupy.asarray(r_true, dtype=op_dtype).copy()
            del r_true
            s_vec = None
            p_vec = None
            v_vec = None
            rho_old = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            alpha = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            omega = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            recurrence_fresh = True
            continue

        shat = _apply_minv(s_vec)
        t_vec = _apply(shat)
        tt = _dot(t_vec, t_vec, cupy=cupy, accum_dtype=acc_dtype)
        if float(cupy.abs(tt)) <= breakdown_tol_f:
            iterations = k + 1
            info = k + 1
            converged_reason = "breakdown_tt"
            break

        omega = _dot(t_vec, s_vec, cupy=cupy, accum_dtype=acc_dtype) / tt
        omega_op = cupy.asarray(omega, dtype=op_dtype)
        x_vec += alpha_op * phat
        x_vec += omega_op * shat
        # Reuse s as the next residual instead of allocating s - omega*t.
        s_vec -= omega_op * t_vec
        r_vec = s_vec
        s_vec = None
        residual_norm = _norm(r_vec, cupy=cupy, accum_dtype=acc_dtype)
        relative_residual = _record_recursive(residual_norm)
        iterations = k + 1

        # Release non-recurrence temporaries before the next matrix-free apply.
        phat = None
        shat = None
        t_vec = None

        if residual_norm <= target_abs:
            if not compute_final_residual:
                info = 0
                converged_reason = "converged_recursive"
                break
            converged, r_true = _true_gate()
            if converged:
                info = 0
                converged_reason = "converged"
                break
            r_vec = cupy.asarray(r_true, dtype=op_dtype)
            r_hat = cupy.asarray(r_true, dtype=op_dtype).copy()
            del r_true
            s_vec = None
            p_vec = None
            v_vec = None
            rho_old = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            alpha = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            omega = cupy.asarray(1.0 + 0.0j, dtype=acc_dtype)
            recurrence_fresh = True
            continue

        if float(cupy.abs(omega)) <= breakdown_tol_f:
            info = k + 1
            converged_reason = "breakdown_omega"
            break
        rho_old = rho
        info = iterations
    else:
        info = maxiter_total
        converged_reason = "maxiter_reached"

    if compute_final_residual and info != 0:
        residual_norm, relative_residual, _ = _true_residual_stats(x_vec)
        true_hist.append(relative_residual)
        if callback_true is not None:
            callback_true(relative_residual)
    elif not compute_final_residual and info != 0:
        residual_norm = float("nan")
        relative_residual = float("nan")

    return CuPyBiCGSTABNativeResult(
        x=x_vec,
        info=int(info),
        iterations=int(iterations),
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        converged_reason=str(converged_reason),
        residual_history=np.asarray(residual_hist, dtype=float),
        true_history=np.asarray(true_hist, dtype=float),
        operator_applications=int(operator_applications),
        preconditioner_applications=int(preconditioner_applications),
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
    acc_dtype = _resolve_accum_dtype(op_dtype, accum_dtype)
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

    def _apply(x_curr: Any) -> Any:
        return _apply_block_op(A_mv, x_curr, cupy=cupy, dtype=op_dtype)

    def _apply_minv(x_curr: Any) -> Any:
        return _apply_block_preconditioner(preconditioner, x_curr, cupy=cupy, dtype=op_dtype)

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
        block_abs = _norm(r_true, cupy=cupy, accum_dtype=acc_dtype)
        block_rel = block_abs / b_norm_frob if b_norm_frob > 0 else block_abs
        return block_abs, block_rel, rhs_norms_np, rhs_rel, r_true

    b_norms = _norms_block(b_mat, cupy=cupy, accum_dtype=acc_dtype)
    b_norms_np = np.asarray(cupy.asnumpy(b_norms), dtype=float)
    b_norm_frob = _norm(b_mat, cupy=cupy, accum_dtype=acc_dtype)
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
        V = cupy.empty((cycle_steps + 1, n, p), dtype=op_dtype)
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
            # Projection is performed in place. Own the action result so a
            # legitimate operator that returns an input view cannot mutate
            # the accumulated basis during orthogonalization.
            w = cupy.asarray(_apply_minv(_apply(V[col])), dtype=op_dtype).copy()
            v_prev = V[: col + 1]
            cs = slice(col * p, (col + 1) * p)
            h_block = _block_basis_projection(v_prev, w, cupy=cupy, accum_dtype=acc_dtype)
            H[: (col + 1) * p, cs] = h_block
            _subtract_block_basis_projection(w, v_prev, h_block, cupy=cupy)
            if reorthogonalize:
                h_block2 = _block_basis_projection(v_prev, w, cupy=cupy, accum_dtype=acc_dtype)
                H[: (col + 1) * p, cs] = H[: (col + 1) * p, cs] + h_block2
                _subtract_block_basis_projection(w, v_prev, h_block2, cupy=cupy)
            # Deleting V at restart is insufficient while this view is live.
            del v_prev

            q_next, r_next = cupy.linalg.qr(cupy.asarray(w, dtype=acc_dtype), mode="reduced")
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
                    # The packed basis can be as large as V. It is no longer
                    # needed when the physical-residual operator starts.
                    del v_trial, y_trial
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
                    # Failed gates must not retain full trial/residual blocks
                    # throughout the remaining Arnoldi steps or next restart.
                    del x_trial, r_true_trial
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
        # Release cycle-local Krylov storage before the residual matvec and
        # before a subsequent cycle allocates another block basis.
        del V, H, G, v_used, y_used, y_last, q0, r0, z0, dx

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
    )


__all__ = [
    "CuPyBiCGSTABNativeResult",
    "CuPyBlockGMRESNativeResult",
    "CuPyGMRESNativeResult",
    "CuPyLSQRNativeResult",
    "bicgstab_cupy_native",
    "block_gmres_cupy_native",
    "fgmres_cupy_native",
    "gmres_cupy_native",
    "lgmres_cupy_native",
    "lsqr_cupy_native",
]
