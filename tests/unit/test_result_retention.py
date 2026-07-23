from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.indexing import n_modes
from pyceles.core.particles import Sphere
from pyceles.linear.solvers import LinearSolveResult
from pyceles.simulation import (
    ResultRetention,
    Simulation,
    SimulationConfig,
    SolvedSourcesResult,
)


def _simulation(*, solve_polarization_basis: bool = False) -> Simulation:
    polarization: Literal["TE"] | tuple[complex, complex] = (
        (1.0 + 0.0j, 0.5j) if solve_polarization_basis else "TE"
    )
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0.0j,
        polarization=polarization,
        polar_angle=0.2,
        azimuthal_angle=0.3,
    )
    return Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0.0j,
            lmax=1,
            source=source,
            solver_method="direct",
            solve_polarization_basis=solve_polarization_basis,
            verbose=False,
        ),
        particles=[
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=50.0,
                refractive_index=1.5 + 0.01j,
            )
        ],
    )


def _solved_single_channel(sim: Simulation) -> SolvedSourcesResult:
    source = sim.config.source
    assert source is not None
    shape = (sim.n_particles, n_modes(sim.config.lmax))
    initial = np.arange(np.prod(shape), dtype=np.float64).reshape(shape).astype(np.complex128)
    rhs = initial + (1.0 + 0.0j)
    coeffs = initial + (2.0 + 0.0j)
    history = np.array([1.0, 0.25], dtype=np.float64)
    solver_result = LinearSolveResult(
        x=coeffs.reshape(-1),
        info=0,
        residual_norm=1.0e-8,
        relative_residual=1.0e-9,
        iterations=2,
        method="gmres",
        residual_history=history,
        preconditioned_residual_history=history,
        true_residual_history=history,
        block_residual_history=history.reshape(-1, 1),
    )
    return SolvedSourcesResult(
        labels=("source",),
        sources={"source": source},
        solver_result=solver_result,
        initial_coeffs={"source": initial},
        rhs={"source": rhs},
        coeffs={"source": coeffs},
        k=2.0 * np.pi / 550.0,
        k0=2.0 * np.pi / 550.0,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )


def test_default_result_retention_preserves_complete_payload() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim)

    multi = sim.postprocess_sources(solved, include_farfield=False)
    run = multi["source"]

    assert run.initial_coeffs is solved.initial_coeffs["source"]
    assert run.rhs is not None
    np.testing.assert_array_equal(run.rhs, solved.rhs["source"])
    assert run.solver_result is solved.solver_result
    assert multi.initial_coeffs is not None
    assert multi.rhs is not None
    assert multi.solver_result is solved.solver_result


def test_minimal_result_retention_drops_optional_arrays_without_copying_solution() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim)

    multi = sim.postprocess_sources(
        solved,
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )
    run = multi["source"]

    assert run.initial_coeffs is None
    assert run.rhs is None
    assert multi.initial_coeffs is None
    assert multi.rhs is None
    assert run.coeffs is solved.coeffs["source"]
    assert np.shares_memory(run.coeffs, run.solver_result.x)
    assert np.shares_memory(multi.coeffs["source"], multi.solver_result.x)
    assert run.solver_result.residual_history is None
    assert run.solver_result.preconditioned_residual_history is None
    assert run.solver_result.true_residual_history is None
    assert run.solver_result.block_residual_history is None
    assert multi.solver_result.residual_history is None


def test_run_minimal_retention_omits_polarization_basis_coefficients() -> None:
    sim = _simulation(solve_polarization_basis=True)

    run = sim.run(
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )

    assert run.initial_coeffs is None
    assert run.rhs is None
    assert run.initial_coeffs_basis is None
    assert run.coeffs_basis is None
    assert run.farfield_basis is not None
    assert run.polarization_jones is not None
    assert run.coeffs.shape == (1, n_modes(sim.config.lmax))


def test_result_retention_rejects_untyped_policy() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim)

    with pytest.raises(TypeError, match="ResultRetention"):
        sim.postprocess_sources(
            solved,
            include_farfield=False,
            retention=cast(Any, "minimal"),
        )
