"""Backend-independent scalar stopping policy for iterative solvers."""

from __future__ import annotations

import numpy as np


def _validated_tolerances(*, rtol: float, atol: float) -> tuple[float, float]:
    """Validate scalar stopping controls without inspecting solver vectors."""
    relative, absolute = float(rtol), float(atol)
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("`rtol` must be finite and non-negative.")
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("`atol` must be finite and non-negative.")
    return relative, absolute


def _absolute_residual_target(rhs_norm: float, *, rtol: float, atol: float) -> float:
    """Build a finite stopping threshold from an already computed host norm.

    This is a setup check, not an overflow-safe reduction: leave the hot norm
    kernels unchanged and reject unusable scaling instead of accepting inf <=
    inf as convergence. No vector scan, device readback, or retry is added.
    """
    relative, absolute = _validated_tolerances(rtol=rtol, atol=atol)
    norm = float(rhs_norm)
    if not np.isfinite(norm) or norm < 0.0:
        raise FloatingPointError(
            "The RHS norm is not finite and non-negative. "
            "Check source scaling and solver precision before solving."
        )
    target = max(absolute, relative * norm)
    if not np.isfinite(target):
        raise FloatingPointError(
            "The absolute residual target overflowed. Check `rtol` and RHS scaling."
        )
    return target
