"""Chebyshev-shell accumulation helpers for periodic Ewald series.

A shell is a layer in integer lattice-index space, defined by
``max(abs(p), abs(q)) == shell``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .scalar import validate_optional_shell_count, validate_shell_count, validate_shell_tolerance

Array = np.ndarray


class PeriodicEwaldConvergenceError(RuntimeError):
    """Raised when adaptive Ewald shell accumulation fails to converge."""


@dataclass(frozen=True)
class LatticeShellControl:
    """Control payload for one Chebyshev-index shell series."""

    fixed_shells: int | None
    max_shells: int
    tolerance: float
    start_shell: int = 0
    min_converged_shell: int = 0
    atol: float = 1.0e-14

    @property
    def adaptive(self) -> bool:
        return self.fixed_shells is None

    @property
    def last_shell(self) -> int:
        return self.max_shells if self.fixed_shells is None else self.fixed_shells


def make_lattice_shell_control(
    *,
    shells: int | None,
    max_shells: int,
    shell_tolerance: float,
    name: str,
    start_shell: int = 0,
    min_converged_shell: int = 0,
    atol: float = 1.0e-14,
) -> LatticeShellControl:
    """Build validated shell control for fixed or adaptive accumulation."""
    fixed = validate_optional_shell_count(shells, name=name)
    if fixed is None:
        max_count = validate_shell_count(max_shells, name="max_shells")
        if max_count <= 0:
            raise ValueError(f"`max_shells` must be > 0 for adaptive mode. Got {max_shells!r}.")
    else:
        max_count = validate_shell_count(max_shells, name="max_shells")
    start = validate_shell_count(start_shell, name="start_shell")
    min_shell = validate_shell_count(min_converged_shell, name="min_converged_shell")
    tol = validate_shell_tolerance(shell_tolerance)
    atol_f = float(atol)
    if not np.isfinite(atol_f) or atol_f < 0.0:
        raise ValueError(f"`atol` must be finite and >= 0. Got {atol!r}.")
    if fixed is not None and start > fixed:
        return LatticeShellControl(
            fixed_shells=fixed,
            max_shells=max_count,
            tolerance=tol,
            start_shell=start,
            min_converged_shell=min_shell,
            atol=atol_f,
        )
    if min_shell < start:
        min_shell = start
    return LatticeShellControl(
        fixed_shells=fixed,
        max_shells=max_count,
        tolerance=tol,
        start_shell=start,
        min_converged_shell=min_shell,
        atol=atol_f,
    )


def _norm(value: complex | Array) -> float:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 0:
        return float(abs(complex(arr)))
    return float(np.linalg.norm(arr.reshape(-1)))


def accumulate_lattice_shell_series(
    *,
    control: LatticeShellControl,
    name: str,
    zero: complex | Array,
    evaluate_shell: Callable[[int], complex | Array],
) -> complex | Array:
    """Accumulate one Chebyshev-index shell series in fixed or adaptive mode."""
    total = zero
    if control.start_shell > control.last_shell:
        return total

    running_scale = 0.0
    prev_increment_norm: float | None = None
    for shell in range(control.start_shell, control.last_shell + 1):
        increment = evaluate_shell(shell)
        total = total + increment
        if not control.adaptive:
            continue
        increment_norm = _norm(increment)
        running_scale = max(running_scale, increment_norm)
        if shell < control.min_converged_shell:
            prev_increment_norm = None
            continue
        threshold = control.atol + control.tolerance * max(_norm(total), running_scale)
        if prev_increment_norm is not None:
            recent_norm = max(prev_increment_norm, increment_norm)
            if recent_norm <= threshold:
                return total
        prev_increment_norm = increment_norm

    if control.adaptive:
        raise PeriodicEwaldConvergenceError(
            "Adaptive periodic Ewald shell accumulation did not converge for "
            f"`{name}` within max_shells={control.max_shells} "
            f"(start={control.start_shell}, min_converged_shell={control.min_converged_shell}, "
            f"shell_tolerance={control.tolerance:.3e}, atol={control.atol:.3e})."
        )
    return total


__all__ = [
    "LatticeShellControl",
    "PeriodicEwaldConvergenceError",
    "accumulate_lattice_shell_series",
    "make_lattice_shell_control",
]
