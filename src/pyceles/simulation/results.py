"""Simulation result containers and result-shaping helpers.

These dataclasses are intentionally frozen because they represent completed
solve/postprocess payloads that should be safe to pass around without hidden
mutation. The mutable state lives in the `Simulation` orchestrator caches, not
in result objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyceles.core.particles import Particle, ParticleCollection
from pyceles.core.sources import Source
from pyceles.linear.solvers import LinearSolveResult
from pyceles.postprocessing.farfield import FarFieldPatterns, PeriodicFarFieldPayload

from .config import SimulationConfig


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
    rhs: np.ndarray
    initial_coeffs: np.ndarray
    initial_coeffs_basis: dict[str, np.ndarray] | None
    coeffs_basis: dict[str, np.ndarray] | None
    solver_result: LinearSolveResult
    solver_result_basis: LinearSolveResult | None
    farfield: FarFieldPatterns
    farfield_basis: dict[str, FarFieldPatterns] | None
    power: dict[str, float | np.ndarray] | None
    power_basis: dict[str, dict[str, float | np.ndarray]] | None
    cross_sections: dict[str, float] | None
    cross_sections_basis: dict[str, dict[str, float]] | None
    unpolarized: dict[str, dict[str, float]] | None
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
    initial_coeffs: dict[str, np.ndarray]
    rhs: dict[str, np.ndarray]
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
        if arr.ndim == 0:
            return float(arr) if arr.dtype.kind == "f" else int(arr)
        return float(arr[index]) if arr.dtype.kind == "f" else int(arr[index])

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
    )


__all__ = [
    "MultiSourceSimulationResult",
    "SimulationResult",
    "SolvedSourcesResult",
    "avg_numeric_dict",
    "empty_farfield_patterns",
    "empty_pwp",
    "single_rhs_result_from_multi",
]
