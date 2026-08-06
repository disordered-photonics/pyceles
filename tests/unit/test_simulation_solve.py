from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.fields import LocalExpansionSource, PlaneWave
from pyceles.core.indexing import n_modes
from pyceles.core.particles import Sphere
from pyceles.linear.solvers import LinearSolveResult
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation import solve as sim_solve


def _plane_wave(*, wavelength: float = 550.0, medium_n: complex = 1.0 + 0j) -> PlaneWave:
    return PlaneWave(
        wavelength=wavelength,
        medium_n=medium_n,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )


def _single_sphere_sim(**cfg_overrides) -> Simulation:
    cfg_kwargs: dict[str, object] = {
        "wavelength": 550.0,
        "n_medium": 1.0 + 0j,
        "lmax": 1,
        "solver_method": "gmres",
        "verbose": False,
    }
    cfg_kwargs.update(cfg_overrides)
    cfg = cast(SimulationConfig, cast(Any, SimulationConfig)(**cfg_kwargs))
    return Simulation(
        cfg,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=50.0, refractive_index=1.5 + 0j)],
    )


@dataclass(frozen=True)
class _ToyLocalExpansionSource:
    wavelength: float = 550.0
    medium_n: complex = 1.0 + 0.0j
    amplitude: float = 1.0

    def has_finite_incident_power(self) -> bool:
        return False

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype=np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        ns = np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]
        return np.zeros((ns, n_modes(int(lmax))), dtype=np.dtype(dtype))

    def source_positions(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0]], dtype=float)

    def outgoing_coeffs(self, lmax: int, *, dtype=np.complex128) -> np.ndarray:
        return np.zeros((1, n_modes(int(lmax))), dtype=np.dtype(dtype))


def test_normalize_sources_argument_validates_mapping_duplicates_and_source_mismatch():
    sim = _single_sphere_sim()
    fn = cast(Any, sim_solve.normalize_sources_argument)
    with pytest.raises(TypeError, match="must be a mapping"):
        fn(sim, [_plane_wave()])
    with pytest.raises(ValueError, match="at least one source"):
        fn(sim, {})
    with pytest.raises(TypeError, match="labels must be strings"):
        fn(sim, {1: _plane_wave()})
    with pytest.raises(ValueError, match="non-empty strings"):
        fn(sim, {"": _plane_wave()})
    with pytest.raises(ValueError, match="wavelength mismatch"):
        fn(sim, {"src": _plane_wave(wavelength=551.0)})
    with pytest.raises(ValueError, match="medium_n mismatch"):
        fn(sim, {"src": _plane_wave(medium_n=1.2 + 0j)})


def test_normalize_sources_argument_warns_for_local_expansion_source_inside_circumsphere():
    sim = _single_sphere_sim()
    fn = cast(Any, sim_solve.normalize_sources_argument)
    source = _ToyLocalExpansionSource()
    assert isinstance(source, LocalExpansionSource)
    with pytest.warns(UserWarning, match="local source center lies inside"):
        out = fn(sim, {"src": source})
    assert out["src"] is source


def test_assemble_dense_operator_via_matvec_builds_dense_columns():
    fn = cast(Any, sim_solve._assemble_dense_operator_via_matvec)

    def _op(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.complex128)
        return np.array([2.0 * arr[0] + arr[1], arr[0] - 3.0 * arr[1]], dtype=np.complex128)

    timings: dict[str, float] = {}
    A = fn(_op, n=2, dtype=np.dtype(np.complex128), show_progress=False, timings=timings)
    np.testing.assert_allclose(A, np.array([[2.0, 1.0], [1.0, -3.0]], dtype=np.complex128))
    assert timings["dense_operator_assembly_s"] >= 0.0


def test_solve_sources_core_reuses_prepared_cache_and_broadcasts_warm_start(monkeypatch):
    sim = _single_sphere_sim()
    warm_start = np.arange(6, dtype=np.complex128).reshape(1, 6)
    prepare_calls = {"count": 0}
    recorded: dict[str, object] = {}

    class _Prepared:
        def __init__(self) -> None:
            self.coupling = object()

        def apply_A(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(x)

        def rhs_Tb(self, b: np.ndarray) -> np.ndarray:
            return np.asarray(b)

    def _fake_prepare_matvec(**kwargs):
        del kwargs
        prepare_calls["count"] += 1
        return _Prepared()

    def _fake_project_source_to_svwf(positions, lmax, source, **kwargs):
        del source, kwargs
        nm = sim_solve.n_modes(lmax)
        return np.arange(positions.shape[0] * nm, dtype=np.complex128).reshape(
            positions.shape[0], nm
        )

    def _fake_solve_linear_system(A_mv, b, **kwargs):
        del A_mv
        recorded["b_shape"] = np.asarray(b).shape
        recorded["x0"] = kwargs["x0"]
        recorded["preconditioner"] = kwargs["preconditioner"]
        return LinearSolveResult(
            x=np.zeros_like(np.asarray(b), dtype=np.complex128),
            info=np.zeros((2,), dtype=int),
            residual_norm=np.zeros((2,), dtype=float),
            relative_residual=np.zeros((2,), dtype=float),
            iterations=np.ones((2,), dtype=int),
            method="gmres",
            residual_history=[np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)],
            rhs_count=2,
        )

    monkeypatch.setattr(sim_solve, "prepare_matvec", _fake_prepare_matvec)
    monkeypatch.setattr(sim_solve, "project_source_to_svwf", _fake_project_source_to_svwf)
    monkeypatch.setattr(sim_solve, "solve_linear_system", _fake_solve_linear_system)

    sources = {"first": _plane_wave(), "second": _plane_wave()}
    out0 = sim_solve.solve_sources_core(sim, sources, warm_start=warm_start)
    out1 = sim_solve.solve_sources_core(sim, sources, warm_start=warm_start)

    assert prepare_calls["count"] == 1
    assert recorded["b_shape"] == (6, 2)
    np.testing.assert_allclose(
        np.asarray(recorded["x0"]),
        np.repeat(warm_start.reshape(-1, 1), 2, axis=1),
    )
    assert recorded["preconditioner"] is None
    assert out0.coeffs["first"].shape == (1, 6)
    assert out0.coeffs["second"].shape == (1, 6)
    assert out1.rhs["first"].shape == (1, 6)
    metadata = out0.solver_result.block_metadata
    assert metadata is not None
    timings = cast(dict[str, float], metadata["simulation_phase_timings_s"])
    assert timings["source_projection_s"] >= 0.0
    assert timings["prepare_operator_s"] >= 0.0
    assert timings["rhs_Tb_s"] >= 0.0
    assert timings["linear_solve_s"] >= 0.0
    assert timings["solve_sources_core_s"] >= 0.0


@pytest.mark.parametrize(
    ("warm_start", "match"),
    [
        (np.zeros((5,), dtype=np.complex128), "one coefficient vector"),
        (np.zeros((5, 1), dtype=np.complex128), "one coefficient vector"),
        (np.zeros((6, 3), dtype=np.complex128), "final axis matching"),
    ],
)
def test_solve_sources_core_validates_warm_start_shapes(monkeypatch, warm_start, match):
    sim = _single_sphere_sim()

    class _Prepared:
        def __init__(self) -> None:
            self.coupling = object()

        def apply_A(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(x)

        def rhs_Tb(self, b: np.ndarray) -> np.ndarray:
            return np.asarray(b)

    monkeypatch.setattr(sim_solve, "prepare_matvec", lambda **kwargs: _Prepared())
    monkeypatch.setattr(
        sim_solve,
        "project_source_to_svwf",
        lambda positions, lmax, source, **kwargs: np.zeros(
            (positions.shape[0], sim_solve.n_modes(lmax)), dtype=np.complex128
        ),
    )
    monkeypatch.setattr(
        sim_solve,
        "solve_linear_system",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("solve_linear_system should not run")
        ),
    )

    with pytest.raises(ValueError, match=match):
        sim_solve.solve_sources_core(
            sim,
            {"first": _plane_wave(), "second": _plane_wave()},
            warm_start=warm_start,
        )


def test_solve_sources_core_zero_particle_case_skips_operator_and_solver(monkeypatch):
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        verbose=False,
    )
    sim = Simulation(cfg, particles=[])

    def _fake_project_source_to_svwf(positions, lmax, source, **kwargs):
        del source, kwargs
        nm = sim_solve.n_modes(lmax)
        return np.zeros((positions.shape[0], nm), dtype=np.complex128)

    monkeypatch.setattr(sim_solve, "project_source_to_svwf", _fake_project_source_to_svwf)
    monkeypatch.setattr(
        sim_solve,
        "prepare_matvec",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("prepare_matvec should not run")),
    )
    monkeypatch.setattr(
        sim_solve,
        "solve_linear_system",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("solve_linear_system should not run")
        ),
    )

    out = sim_solve.solve_sources_core(sim, {"src": _plane_wave()})
    assert out.solver_result.rhs_count == 1
    assert out.coeffs["src"].shape == (0, 6)
    assert out.rhs["src"].shape == (0, 6)


@pytest.mark.fake_gpu
def test_solve_sources_core_direct_cupy_dense_fallback_releases_dense_cache(monkeypatch):
    sim = _single_sphere_sim(operator_backend="cupy", solver_method="direct")
    recorded: dict[str, object] = {}
    lu_payload = ("lu", "piv")

    class _Prepared:
        def __init__(self) -> None:
            self.coupling = object()

        def apply_A(self, x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=np.complex128)
            return 2.0 * arr

        def rhs_Tb(self, b: np.ndarray) -> np.ndarray:
            return np.asarray(b)

    monkeypatch.setattr(sim_solve, "prepare_matvec", lambda **kwargs: _Prepared())
    monkeypatch.setattr(
        sim_solve,
        "project_source_to_svwf",
        lambda positions, lmax, source, **kwargs: np.zeros(
            (positions.shape[0], sim_solve.n_modes(lmax)), dtype=np.complex128
        ),
    )

    def _fake_factorize_dense_matrix(A_dense, **kwargs):
        recorded["factorized_shape"] = np.asarray(A_dense).shape
        recorded["overwrite_input"] = kwargs["overwrite_input"]
        return lu_payload

    def _fake_solve_linear_system(A_mv, b, **kwargs):
        del A_mv
        recorded["A_factorized"] = kwargs["A_factorized"]
        return LinearSolveResult(
            x=np.zeros_like(np.asarray(b), dtype=np.complex128),
            info=0,
            residual_norm=0.0,
            relative_residual=0.0,
            iterations=1,
            method="direct",
            residual_history=None,
            rhs_count=1,
        )

    monkeypatch.setattr(sim_solve, "factorize_dense_matrix", _fake_factorize_dense_matrix)
    monkeypatch.setattr(sim_solve, "solve_linear_system", _fake_solve_linear_system)

    sim_solve.solve_sources_core(sim, {"src": _plane_wave()})

    assert recorded["factorized_shape"] == (6, 6)
    assert recorded["overwrite_input"] is True
    assert recorded["A_factorized"] == lu_payload
    assert sim._dense_operator_cache is None
    assert sim._dense_lu_cache == lu_payload


@pytest.mark.fake_gpu
def test_solve_sources_core_direct_cupy_reuses_lu_without_reassembling_dense_A(monkeypatch):
    sim = _single_sphere_sim(operator_backend="cupy", solver_method="direct")
    apply_calls = 0
    factorize_calls = 0
    solve_factorized_payloads: list[tuple[str, str]] = []
    lu_payload = ("lu", "piv")

    class _Prepared:
        def __init__(self) -> None:
            self.coupling = object()

        def apply_A(self, x: np.ndarray) -> np.ndarray:
            nonlocal apply_calls
            apply_calls += 1
            return np.asarray(x, dtype=np.complex128)

        def rhs_Tb(self, b: np.ndarray) -> np.ndarray:
            return np.asarray(b)

    monkeypatch.setattr(sim_solve, "prepare_matvec", lambda **kwargs: _Prepared())
    monkeypatch.setattr(
        sim_solve,
        "project_source_to_svwf",
        lambda positions, lmax, source, **kwargs: np.zeros(
            (positions.shape[0], sim_solve.n_modes(lmax)), dtype=np.complex128
        ),
    )

    def _fake_factorize_dense_matrix(A_dense, **kwargs):
        del A_dense, kwargs
        nonlocal factorize_calls
        factorize_calls += 1
        return lu_payload

    def _fake_solve_linear_system(A_mv, b, **kwargs):
        del A_mv
        solve_factorized_payloads.append(kwargs["A_factorized"])
        return LinearSolveResult(
            x=np.zeros_like(np.asarray(b), dtype=np.complex128),
            info=0,
            residual_norm=0.0,
            relative_residual=0.0,
            iterations=1,
            method="direct",
            residual_history=None,
            rhs_count=1,
        )

    monkeypatch.setattr(sim_solve, "factorize_dense_matrix", _fake_factorize_dense_matrix)
    monkeypatch.setattr(sim_solve, "solve_linear_system", _fake_solve_linear_system)

    sim_solve.solve_sources_core(
        sim,
        {"src": _plane_wave()},
        solver_compute_final_residual=False,
    )
    first_apply_calls = apply_calls

    sim_solve.solve_sources_core(
        sim,
        {"src": _plane_wave()},
        solver_compute_final_residual=False,
    )

    assert factorize_calls == 1
    assert solve_factorized_payloads == [lu_payload, lu_payload]
    assert first_apply_calls == 6
    assert apply_calls == first_apply_calls


def test_simulation_clear_caches_rebuilds_equivalent_operator_state() -> None:
    sim = _single_sphere_sim(operator_backend="numpy", solver_method="direct")
    source = _plane_wave()

    first = sim.solve_sources({"src": source})
    assert sim._prepared_operator_cache is not None
    assert sim._dense_lu_cache is not None

    sim.clear_caches()
    assert sim._prepared_operator_cache is None
    assert sim._prepared_operator_dtype is None
    assert sim._prepared_operator_periodic_key is None
    assert sim._dense_operator_cache is None
    assert sim._dense_operator_dtype is None
    assert sim._dense_lu_cache is None
    assert sim._dense_lu_dtype is None

    second = sim.solve_sources({"src": source})
    np.testing.assert_allclose(
        second.coeffs["src"],
        first.coeffs["src"],
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.fake_gpu
def test_solve_sources_core_rejects_custom_preconditioner_on_cupy_backend(monkeypatch):
    sim = _single_sphere_sim(
        operator_backend="cupy",
        solver_preconditioner=lambda x: np.asarray(x),
    )

    class _Prepared:
        def __init__(self) -> None:
            self.coupling = object()

        def apply_A(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(x)

        def rhs_Tb(self, b: np.ndarray) -> np.ndarray:
            return np.asarray(b)

    monkeypatch.setattr(sim_solve, "prepare_matvec", lambda **kwargs: _Prepared())
    monkeypatch.setattr(
        sim_solve,
        "project_source_to_svwf",
        lambda positions, lmax, source, **kwargs: np.zeros(
            (positions.shape[0], sim_solve.n_modes(lmax)), dtype=np.complex128
        ),
    )

    with pytest.raises(NotImplementedError, match="does not support custom preconditioner"):
        sim_solve.solve_sources_core(sim, {"src": _plane_wave()})


@pytest.mark.api_contract
def test_normalize_warm_start_mapping_is_label_aligned() -> None:
    first = np.arange(6, dtype=np.complex128).reshape(1, 6)
    second = 10.0 + np.arange(6, dtype=np.complex128)

    normalized = sim_solve.normalize_warm_start_argument(
        {"second": second, "first": first},
        labels=("first", "second"),
        unknowns=6,
        dtype=np.dtype(np.complex128),
    )

    assert normalized is not None
    assert normalized.shape == (6, 2)
    np.testing.assert_array_equal(normalized[:, 0], first.reshape(-1))
    np.testing.assert_array_equal(normalized[:, 1], second)

    with pytest.raises(ValueError, match="keys must exactly match"):
        sim_solve.normalize_warm_start_argument(
            {"first": first},
            labels=("first", "second"),
            unknowns=6,
            dtype=np.dtype(np.complex128),
        )

    with pytest.raises(TypeError, match="labels must be strings"):
        sim_solve.normalize_warm_start_argument(
            {1: first, "second": second},  # type: ignore[dict-item]
            labels=("first", "second"),
            unknowns=6,
            dtype=np.dtype(np.complex128),
        )
