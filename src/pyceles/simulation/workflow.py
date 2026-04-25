"""High-level `Simulation` orchestrator.

The class holds geometry/config state plus reusable prepared-operator caches,
while the actual solve and postprocess logic lives in sibling modules so those
policies can grow without collapsing the workflow layer back into one file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyceles.core.operators import PreparedOperator
from pyceles.core.particles import Particle
from pyceles.core.sources import Source
from pyceles.linear.solvers import DenseLUFactorization

from .config import SimulationConfig
from .helpers import first_overlapping_circumscribing_pair, normalize_particle_geometry
from .postprocess import postprocess_sources_impl, run_impl
from .results import MultiSourceSimulationResult, SimulationResult, SolvedSourcesResult
from .solve import normalize_sources_argument, solve_sources_core


class Simulation:
    """High-level orchestrator for one many-particle scattering experiment."""

    particles: tuple[Particle, ...]

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def __init__(
        self,
        config: SimulationConfig,
        *,
        particles: Sequence[Particle],
    ):
        self.config = config
        part, pos, rad = normalize_particle_geometry(particles)

        self.positions = pos
        self.circumscribing_radii = rad
        self.particles = part
        self._prepared_operator_cache: PreparedOperator | None = None
        self._prepared_operator_dtype: np.dtype | None = None
        self._dense_operator_cache: np.ndarray | None = None
        self._dense_operator_dtype: np.dtype | None = None
        self._dense_lu_cache: DenseLUFactorization | None = None
        self._dense_lu_dtype: np.dtype | None = None
        self._solve_backend_handoffs: dict[int, dict[str, Any]] = {}
        if bool(self.config.check_circumscribing_sphere_overlap):
            overlap = first_overlapping_circumscribing_pair(
                self.positions,
                self.circumscribing_radii,
                atol=float(self.config.circumscribing_sphere_overlap_atol),
                show_progress=bool(self.config.verbose),
            )
            if overlap is not None:
                i, j, d, rsum = overlap
                raise ValueError(
                    "Invalid geometry: circumscribing spheres overlap for particle pair "
                    f"({i}, {j}) with center distance {d:.6g} and required minimum {rsum:.6g}. "
                    "The current T-matrix formulation requires disjoint circumscribing spheres. "
                    "If this is intentional for an experimental workflow, set "
                    "`check_circumscribing_sphere_overlap=False`."
                )

    def _validate_ready_to_run(self) -> Source:
        if self.config.source is None:
            raise ValueError(
                "Cannot run simulation: missing required `source` in SimulationConfig. "
                "Set `source` before calling `run()`."
            )
        return self.config.source

    def solve_sources(
        self,
        sources: Mapping[str, Source],
        *,
        solver_compute_final_residual: bool | None = None,
    ) -> SolvedSourcesResult:
        labeled = normalize_sources_argument(self, sources)
        return solve_sources_core(
            self,
            labeled,
            solver_compute_final_residual=solver_compute_final_residual,
        )

    def _solve_sources_for_immediate_postprocess(
        self,
        sources: Mapping[str, Source],
        *,
        solver_compute_final_residual: bool | None = None,
    ) -> SolvedSourcesResult:
        labeled = normalize_sources_argument(self, sources)
        return solve_sources_core(
            self,
            labeled,
            solver_compute_final_residual=solver_compute_final_residual,
            retain_backend_handoff=True,
        )

    def postprocess_sources(
        self,
        solved: SolvedSourcesResult,
        *,
        include_farfield: bool = True,
        farfield_polar_angles: np.ndarray | None = None,
        farfield_azimuthal_angles: np.ndarray | None = None,
    ) -> MultiSourceSimulationResult:
        return postprocess_sources_impl(
            self,
            solved,
            include_farfield=include_farfield,
            farfield_polar_angles=farfield_polar_angles,
            farfield_azimuthal_angles=farfield_azimuthal_angles,
        )

    def run(self, *, include_farfield: bool = True) -> SimulationResult:
        return run_impl(self, include_farfield=include_farfield)


__all__ = ["Simulation"]
