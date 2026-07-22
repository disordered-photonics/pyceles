"""High-level `Simulation` orchestrator.

The class holds geometry/config state plus reusable prepared-operator caches,
while the actual solve and postprocess logic lives in sibling modules so those
policies can grow without collapsing the workflow layer back into one file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pyceles.core.operators import PreparedOperator
from pyceles.core.particles import Particle
from pyceles.core.sources import Source
from pyceles.linear.solvers import DenseLUFactorization

from .config import SimulationConfig
from .helpers import (
    first_overlapping_circumscribing_pair,
    first_periodic_overlapping_circumscribing_pair,
    normalize_particle_geometry,
)
from .postprocess import postprocess_sources_impl, run_impl
from .results import MultiSourceSimulationResult, SimulationResult, SolvedSourcesResult
from .solve import (
    _solve_sources_for_immediate_postprocess_core,
    _SolvedSourcesExecution,
    normalize_sources_argument,
    solve_sources_core,
)


class Simulation:
    """High-level orchestrator for one many-particle scattering experiment."""

    _config: SimulationConfig
    _particles: tuple[Particle, ...]
    _positions: np.ndarray
    _circumscribing_radii: np.ndarray
    _prepared_operator_cache: PreparedOperator | None
    _prepared_operator_dtype: np.dtype | None
    _prepared_operator_periodic_key: tuple[float, float] | None
    _dense_operator_cache: np.ndarray | None
    _dense_operator_dtype: np.dtype | None
    _dense_lu_cache: DenseLUFactorization | None
    _dense_lu_dtype: np.dtype | None

    @property
    def config(self) -> SimulationConfig:
        """Immutable configuration identity used by prepared-operator caches."""
        return self._config

    @property
    def particles(self) -> tuple[Particle, ...]:
        """Canonical immutable particle descriptors for this simulation."""
        return self._particles

    @property
    def positions(self) -> np.ndarray:
        """Read-only particle-center array derived from ``particles``."""
        return self._positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        """Read-only circumscribing radii derived from ``particles``."""
        return self._circumscribing_radii

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    def clear_caches(self) -> None:
        """Release prepared operator, dense matrix, and LU references.

        The immutable experiment specification remains intact, so a later run
        rebuilds equivalent execution state. CuPy may retain freed device
        blocks in its process-wide allocator pool; use CuPy's pool controls
        separately when returning cached device memory to the driver matters.
        """
        self._prepared_operator_cache = None
        self._prepared_operator_dtype = None
        self._prepared_operator_periodic_key = None
        self._dense_operator_cache = None
        self._dense_operator_dtype = None
        self._dense_lu_cache = None
        self._dense_lu_dtype = None

    def __init__(
        self,
        config: SimulationConfig,
        *,
        particles: Sequence[Particle],
    ):
        self._config = config
        part, pos, rad = normalize_particle_geometry(particles)

        # Keep owning arrays out of the public object. A read-only view backed
        # by a read-only owner cannot have writes re-enabled through `.flags`.
        pos.setflags(write=False)
        rad.setflags(write=False)
        self._positions = pos.view()
        self._circumscribing_radii = rad.view()
        self._particles = part
        self.clear_caches()
        if bool(self.config.check_circumscribing_sphere_overlap):
            if self.config.periodic is None:
                finite_overlap = first_overlapping_circumscribing_pair(
                    self.positions,
                    self.circumscribing_radii,
                    atol=float(self.config.circumscribing_sphere_overlap_atol),
                    show_progress=bool(self.config.verbose),
                )
                if finite_overlap is not None:
                    i, j, d, rsum = finite_overlap
                    detail = f"particle pair ({i}, {j}) with center distance {d:.6g}"
                    raise ValueError(
                        "Invalid geometry: circumscribing spheres overlap for "
                        f"{detail} and required minimum {rsum:.6g}. "
                        "The current T-matrix formulation requires disjoint circumscribing spheres. "
                        "If this is intentional for an experimental workflow, set "
                        "`check_circumscribing_sphere_overlap=False`."
                    )
            else:
                periodic_overlap = first_periodic_overlapping_circumscribing_pair(
                    self.positions,
                    self.circumscribing_radii,
                    lattice=self.config.periodic.lattice,
                    atol=float(self.config.circumscribing_sphere_overlap_atol),
                )
                if periodic_overlap is not None:
                    i, j, p, q, d, rsum = periodic_overlap
                    detail = (
                        f"particle pair ({i}, {j}) under lattice shift ({p}, {q}) "
                        f"with image distance {d:.6g}"
                    )
                    raise ValueError(
                        "Invalid geometry: circumscribing spheres overlap for "
                        f"{detail} and required minimum {rsum:.6g}. "
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
    ) -> _SolvedSourcesExecution:
        labeled = normalize_sources_argument(self, sources)
        return _solve_sources_for_immediate_postprocess_core(
            self,
            labeled,
            solver_compute_final_residual=solver_compute_final_residual,
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
