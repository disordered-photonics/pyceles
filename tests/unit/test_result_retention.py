from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.indexing import n_modes
from pyceles.core.particles import Sphere
from pyceles.linear.solvers import LinearSolveResult
from pyceles.postprocessing.farfield import PowerFluxDecomposition
from pyceles.simulation import (
    MultiSourceSolveResult,
    ResultRetention,
    Simulation,
    SimulationConfig,
)


def _source(*, polarized: bool = False) -> PlaneWave:
    polarization: Literal["TE"] | tuple[complex, complex] = (
        (1.0 + 0.0j, 0.5j) if polarized else "TE"
    )
    return PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0.0j,
        polarization=polarization,
        polar_angle=0.2,
        azimuthal_angle=0.3,
    )


def _simulation() -> Simulation:
    return Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0.0j,
            lmax=1,
            solver_method="direct",
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


def _solved_single_channel(sim: Simulation, source: PlaneWave) -> MultiSourceSolveResult:
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
    return MultiSourceSolveResult(
        config=sim.config,
        particles=sim.particles,
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
    solved = _solved_single_channel(sim, _source())

    multi = sim.postprocess_sources(solved, include_farfield=False)
    run = multi["source"]

    assert run.initial_coeffs is solved.initial_coeffs["source"]
    assert run.rhs is not None
    np.testing.assert_array_equal(run.rhs, solved.rhs["source"])
    assert multi.solver_result is solved.solver_result
    assert multi.channels["source"] is run


def test_solve_result_rejects_malformed_channel_shapes_at_construction() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim, _source())

    with pytest.raises(ValueError, match=r"RHS for channel 'source'.*shape"):
        replace(solved, rhs={"source": np.zeros((2, 3), dtype=np.complex128)})


def test_channel_result_rejects_reshaped_solver_view() -> None:
    sim = _simulation()
    run = sim.run(_source(), include_farfield=False)

    with pytest.raises(ValueError, match=r"Solved coefficients must have shape"):
        replace(run, coeffs=run.coeffs.reshape(2, 3))

    backward = PowerFluxDecomposition(
        direction="backward",
        initial_power=1.0,
        scattered_power=0.5,
        interference_power=-0.25,
    )
    with pytest.raises(ValueError, match=r"Forward decomposition must have direction"):
        replace(run, decomposition_forward=backward)


def test_minimal_result_retention_drops_optional_arrays_without_copying_solution() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim, _source())

    multi = sim.postprocess_sources(
        solved,
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )
    run = multi["source"]

    assert run.initial_coeffs is None
    assert run.rhs is None
    assert run.coeffs is solved.coeffs["source"]
    assert np.shares_memory(run.coeffs, multi.solver_result.x)
    assert multi.solver_result.residual_history is None


def test_polarization_minimal_retention_keeps_only_essential_basis_state() -> None:
    sim = _simulation()
    source = _source(polarized=True)

    result = sim.run_polarizations(
        source,
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )

    assert result.te.initial_coeffs is None
    assert result.tm.initial_coeffs is None
    assert result.te.rhs is None
    assert result.tm.rhs is None
    assert result.te.coeffs.shape == (1, n_modes(sim.config.lmax))
    assert result.tm.coeffs.shape == (1, n_modes(sim.config.lmax))
    assert int(result.solver_result.rhs_count) == 2
    mixed = result.mixed
    assert mixed.initial_coeffs is None
    assert mixed.rhs is None
    np.testing.assert_allclose(
        mixed.coeffs,
        result.jones[0] * result.te.coeffs + result.jones[1] * result.tm.coeffs,
    )


def test_result_retention_rejects_untyped_policy() -> None:
    sim = _simulation()
    solved = _solved_single_channel(sim, _source())

    with pytest.raises(TypeError, match="ResultRetention"):
        sim.postprocess_sources(
            solved,
            include_farfield=False,
            retention=cast(Any, "minimal"),
        )


@pytest.mark.api_contract
def test_single_source_result_owns_matching_single_rhs_solver() -> None:
    sim = _simulation()
    source = _source(polarized=True)

    run = sim.run(source, include_farfield=False)

    assert int(run.solver_result.rhs_count) == 1
    assert np.shares_memory(run.coeffs, run.solver_result.x)
    np.testing.assert_array_equal(
        run.coeffs.reshape(-1),
        np.asarray(run.solver_result.x).reshape(-1),
    )


@pytest.mark.api_contract
def test_multi_source_channels_share_block_solution_without_solver_aliases() -> None:
    sim = _simulation()
    source = _source(polarized=True)

    result = sim.run_sources(
        {
            "te": source.with_polarization("TE"),
            "tm": source.with_polarization("TM"),
        },
        include_farfield=False,
    )

    assert result.labels == ("te", "tm")
    assert int(result.solver_result.rhs_count) == 2
    assert not hasattr(result["te"], "solver_result")
    assert not hasattr(result["tm"], "solver_result")
    assert np.shares_memory(result["te"].coeffs, result.solver_result.x)
    assert np.shares_memory(result["tm"].coeffs, result.solver_result.x)
    np.testing.assert_array_equal(result["te"].coeffs.reshape(-1), result.solver_result.x[:, 0])
    np.testing.assert_array_equal(result["tm"].coeffs.reshape(-1), result.solver_result.x[:, 1])


@pytest.mark.api_contract
def test_polarization_mixed_channel_is_derived_and_not_retained() -> None:
    sim = _simulation()
    result = sim.run_polarizations(
        _source(polarized=True),
        include_farfield=False,
        retention=ResultRetention.minimal(),
    )

    assert int(result.solver_result.rhs_count) == 2
    assert np.shares_memory(result.te.coeffs, result.solver_result.x)
    assert np.shares_memory(result.tm.coeffs, result.solver_result.x)
    assert not hasattr(result.te, "solver_result")
    assert not hasattr(result.tm, "solver_result")

    mixed_first = result.mixed
    mixed_second = result.mixed
    assert not hasattr(mixed_first, "solver_result")
    assert not np.shares_memory(mixed_first.coeffs, result.solver_result.x)
    assert not np.shares_memory(mixed_first.coeffs, mixed_second.coeffs)
    np.testing.assert_allclose(
        mixed_first.coeffs,
        result.jones[0] * result.te.coeffs + result.jones[1] * result.tm.coeffs,
    )


@pytest.mark.api_contract
def test_run_sources_preserves_arbitrary_multi_rhs_block_execution(monkeypatch) -> None:
    import pyceles.simulation.solve as simulation_solve_module

    sim = _simulation()
    source = _source(polarized=True)
    sources = {
        f"channel_{index}": source.with_polarization("TE" if index % 2 == 0 else "TM")
        for index in range(5)
    }
    rhs_shapes: list[tuple[int, ...]] = []
    solve_linear_system_real = simulation_solve_module.solve_linear_system

    def _solve_linear_system_wrapped(A_mv, b, **kwargs):
        rhs_shapes.append(np.asarray(b).shape)
        return solve_linear_system_real(A_mv, b, **kwargs)

    monkeypatch.setattr(
        simulation_solve_module,
        "solve_linear_system",
        _solve_linear_system_wrapped,
    )

    result = sim.run_sources(sources, include_farfield=False)

    assert rhs_shapes == [(n_modes(sim.config.lmax), 5)]
    assert result.labels == tuple(sources)
    assert int(result.solver_result.rhs_count) == 5
    assert tuple(result.channels) == tuple(sources)
    with pytest.raises(TypeError):
        cast(Any, result.channels)["new"] = result["channel_0"]


@pytest.mark.api_contract
def test_solve_result_label_mappings_are_read_only() -> None:
    sim = _simulation()
    solved = sim.solve_sources({"source": _source()})

    with pytest.raises(TypeError):
        cast(Any, solved.sources)["other"] = _source()
    with pytest.raises(TypeError):
        cast(Any, solved.coeffs)["source"] = solved.coeffs["source"].copy()


@pytest.mark.api_contract
def test_postprocess_sources_rejects_foreign_simulation_context() -> None:
    sim = _simulation()
    source = _source()
    solved = sim.solve_sources({"source": source})
    foreign = Simulation(
        sim.config,
        particles=[
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=65.0,
                refractive_index=1.5 + 0.01j,
            )
        ],
    )

    with pytest.raises(ValueError, match="must come from this Simulation"):
        foreign.postprocess_sources(solved, include_farfield=False)


@pytest.mark.api_contract
def test_solver_owned_channels_reject_copied_coefficient_storage() -> None:
    from dataclasses import replace

    sim = _simulation()
    source = _source()
    direct = sim.run(source, include_farfield=False)
    with pytest.raises(ValueError, match="zero-copy view"):
        replace(direct, coeffs=direct.coeffs.copy())

    solved = sim.solve_sources({"source": source})
    with pytest.raises(ValueError, match="zero-copy view"):
        replace(
            solved,
            coeffs={"source": solved.coeffs["source"].copy()},
        )


@pytest.mark.api_contract
def test_result_reprs_are_compact_and_do_not_expand_large_arrays() -> None:
    sim = _simulation()
    source = _source(polarized=True)

    single = sim.run(source, include_farfield=False)
    multi = sim.run_sources(
        {
            "te": source.with_polarization("TE"),
            "tm": source.with_polarization("TM"),
        },
        include_farfield=False,
    )
    polarized = sim.run_polarizations(source, include_farfield=False)

    for result in (single, multi, polarized, single.solver_result):
        text = repr(result)
        assert len(text) < 600
        assert "array([" not in text

    assert "coeffs=(shape=(1, 6), dtype=complex128)" in repr(single)
    assert "labels=('te', 'tm')" in repr(multi)
    assert "jones=" in repr(polarized)
    assert "x=(shape=(6,), dtype=complex128)" in repr(single.solver_result)


@pytest.mark.api_contract
def test_polarization_basis_indexing_uses_exact_labels() -> None:
    result = _simulation().run_polarizations(
        _source(polarized=True),
        include_farfield=False,
    )

    assert result["te"] is result.te
    assert result["tm"] is result.tm
    with pytest.raises(KeyError):
        _ = result["TE"]
