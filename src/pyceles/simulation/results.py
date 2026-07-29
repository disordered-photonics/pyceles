"""Simulation result containers and result-shaping helpers.

Result field bindings are frozen, but numerical arrays and diagnostic mappings
remain mutable. This avoids duplicating potentially GiB-scale payloads merely
to claim deep immutability. In particular, ``SimulationResult.coeffs`` may
share storage with ``solver_result.x``; callers that need an independently
mutable snapshot should copy the array explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from pyceles.core.particles import Particle, ParticleCollection
from pyceles.core.sources import Source
from pyceles.linear.solvers import LinearSolveResult
from pyceles.postprocessing.farfield import (
    FarFieldPatterns,
    PeriodicFarFieldPayload,
    PowerBalance,
)

from .config import SimulationConfig


@dataclass(frozen=True, slots=True)
class ResultRetention:
    """Control optional large arrays retained by completed run results.

    The default preserves the full diagnostic payload. Solved coefficients
    remain available in every mode because they are the restart and deferred
    postprocessing state.
    """

    initial_coeffs: bool = True
    rhs: bool = True
    residual_history: bool = True
    polarization_basis_coeffs: bool = True

    @classmethod
    def minimal(cls) -> ResultRetention:
        """Retain solved coefficients and compact solver diagnostics only."""
        return cls(
            initial_coeffs=False,
            rhs=False,
            residual_history=False,
            polarization_basis_coeffs=False,
        )


def _apply_solver_result_retention(
    result: LinearSolveResult,
    *,
    retention: ResultRetention,
) -> LinearSolveResult:
    if retention.residual_history:
        return result
    return replace(
        result,
        residual_history=None,
        preconditioned_residual_history=None,
        true_residual_history=None,
        block_residual_history=None,
    )


@dataclass(frozen=True)
class SimulationResult:
    """Container for solved multipole coefficients and derived observables.

    For periodic runs, finite-cluster far-field payloads remain empty placeholders
    and periodic diffraction-order observables live under `periodic`.
    """

    config: SimulationConfig
    k: float
    k0: float
    coeffs: np.ndarray
    rhs: np.ndarray | None
    initial_coeffs: np.ndarray | None
    initial_coeffs_basis: dict[str, np.ndarray] | None
    coeffs_basis: dict[str, np.ndarray] | None
    solver_result: LinearSolveResult
    solver_result_basis: LinearSolveResult | None
    farfield: FarFieldPatterns
    farfield_basis: dict[str, FarFieldPatterns] | None
    power: PowerBalance | None
    power_basis: dict[str, PowerBalance] | None
    cross_sections: dict[str, float] | None
    cross_sections_basis: dict[str, dict[str, float]] | None
    unpolarized: dict[str, PowerBalance | dict[str, float]] | None
    decomposition_forward: dict[str, float] | None
    decomposition_backward: dict[str, float] | None
    decomposition_forward_basis: dict[str, dict[str, float]] | None
    decomposition_backward_basis: dict[str, dict[str, float]] | None
    particles: ParticleCollection | Sequence[Particle]
    periodic: PeriodicFarFieldPayload | None = None
    polarization_jones: tuple[complex, complex] | None = None
    compute_dtype: str = "complex128"
    accum_dtype: str = "complex128"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "particles",
            ParticleCollection.from_particles(self.particles),
        )

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def positions(self) -> np.ndarray:
        particles = self.particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("SimulationResult particles were not normalized.")
        return particles.positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        particles = self.particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("SimulationResult particles were not normalized.")
        return particles.circumscribing_radii


@dataclass(frozen=True)
class SolvedSourcesResult:
    """Solve-only outputs from one shared-operator multi-RHS solve."""

    labels: tuple[str, ...]
    sources: dict[str, Source]
    solver_result: LinearSolveResult
    initial_coeffs: dict[str, np.ndarray]
    rhs: dict[str, np.ndarray]
    coeffs: dict[str, np.ndarray]
    k: float
    k0: float
    compute_dtype: str
    accum_dtype: str


@dataclass(frozen=True)
class MultiSourceSimulationResult:
    """Postprocessed channel results for one solved multi-source payload."""

    labels: tuple[str, ...]
    sources: dict[str, Source]
    runs: dict[str, SimulationResult]
    solver_result: LinearSolveResult
    initial_coeffs: dict[str, np.ndarray] | None
    rhs: dict[str, np.ndarray] | None
    coeffs: dict[str, np.ndarray]

    def __getitem__(self, label: str) -> SimulationResult:
        return self.runs[label]


def avg_numeric_dict(d1: Mapping[str, object], d2: Mapping[str, object]) -> dict[str, float]:
    """Average overlapping scalar diagnostics from two channels."""
    keys = set(d1).intersection(set(d2))
    out: dict[str, float] = {}
    for key in keys:
        v1, v2 = d1[key], d2[key]
        if isinstance(v1, (int, float, np.floating)) and isinstance(v2, (int, float, np.floating)):
            out[key] = float(0.5 * (float(v1) + float(v2)))
    return out


def average_power_balances(first: PowerBalance, second: PowerBalance) -> PowerBalance:
    """Return the incoherent arithmetic average of two power balances."""

    def avg_optional(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(0.5 * (float(a) + float(b)))

    particles = None
    if (
        first.local_absorbed_power_per_particle is not None
        and second.local_absorbed_power_per_particle is not None
    ):
        a = np.asarray(first.local_absorbed_power_per_particle, dtype=np.float64)
        b = np.asarray(second.local_absorbed_power_per_particle, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(
                "Cannot average power balances with different per-particle shapes: "
                f"{a.shape} and {b.shape}."
            )
        particles = np.asarray(0.5 * (a + b), dtype=np.float64)
    return PowerBalance(
        incident_power=avg_optional(first.incident_power, second.incident_power),
        reflected_power=avg_optional(first.reflected_power, second.reflected_power),
        transmitted_power=avg_optional(first.transmitted_power, second.transmitted_power),
        local_absorbed_power=avg_optional(
            first.local_absorbed_power,
            second.local_absorbed_power,
        ),
        local_absorbed_power_per_particle=particles,
    )


def empty_pwp(dtype: npt.DTypeLike) -> dict[str, np.ndarray]:
    """Return an empty PWP payload used when far-field postprocessing is disabled."""
    return {
        "beta": np.zeros((0,), dtype=float),
        "alpha": np.zeros((0,), dtype=float),
        "kx": np.zeros((0, 0), dtype=float),
        "ky": np.zeros((0, 0), dtype=float),
        "kz": np.zeros((0, 0), dtype=float),
        "coeff": np.zeros((0, 0), dtype=np.dtype(dtype)),
    }


def empty_farfield_patterns(dtype: npt.DTypeLike) -> FarFieldPatterns:
    """Return an empty far-field payload for solve-only workflows."""
    return FarFieldPatterns(
        initial_te=None,
        initial_tm=None,
        scattered_te=empty_pwp(dtype),
        scattered_tm=empty_pwp(dtype),
        total_te=None,
        total_tm=None,
    )


def single_rhs_result_from_multi(result: LinearSolveResult, col: int) -> LinearSolveResult:
    """Extract one RHS column from a multi-RHS `LinearSolveResult`."""
    if int(result.rhs_count) <= 1:
        return result

    x_arr = np.asarray(result.x)
    if x_arr.ndim != 2:
        raise ValueError(
            f"Expected multi-RHS solver output with 2D `x` array. Got shape {x_arr.shape}."
        )
    if col < 0 or col >= x_arr.shape[1]:
        raise IndexError(f"RHS column index {col} out of bounds for shape {x_arr.shape}.")

    def _pick_scalar(value: int | float | np.ndarray, index: int) -> int | float:
        arr = np.asarray(value)
        selected = arr.item() if arr.ndim == 0 else arr.reshape(-1)[index]
        return float(selected) if arr.dtype.kind == "f" else int(selected)

    def _pick_optional_text(value: str | np.ndarray | None, index: int) -> str | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=object)
        selected = arr.item() if arr.ndim == 0 else arr.reshape(-1)[index]
        return str(selected)

    residual_history = None
    if isinstance(result.residual_history, list):
        if col < len(result.residual_history):
            residual_history = result.residual_history[col]
    elif isinstance(result.residual_history, np.ndarray):
        residual_history = result.residual_history

    return LinearSolveResult(
        x=np.asarray(x_arr[:, col]),
        info=int(_pick_scalar(result.info, col)),
        residual_norm=float(_pick_scalar(result.residual_norm, col)),
        relative_residual=float(_pick_scalar(result.relative_residual, col)),
        iterations=int(_pick_scalar(result.iterations, col)),
        method=str(result.method),
        residual_history=residual_history,
        rhs_count=1,
        preconditioned_residual_history=result.preconditioned_residual_history,
        true_residual_history=result.true_residual_history,
        converged_reason=_pick_optional_text(result.converged_reason, col),
        block_residual_history=result.block_residual_history,
        stopping_rule=result.stopping_rule,
        block_metadata=result.block_metadata,
    )


__all__ = [
    "MultiSourceSimulationResult",
    "ResultRetention",
    "SimulationResult",
    "SolvedSourcesResult",
    "average_power_balances",
    "avg_numeric_dict",
    "empty_farfield_patterns",
    "empty_pwp",
    "single_rhs_result_from_multi",
]
