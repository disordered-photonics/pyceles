from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import MLFMMOptions
from pyceles.core.periodic import PeriodicSpec
from pyceles.simulation import SimulationConfig

pytestmark = pytest.mark.api_contract


def test_simulation_config_owns_read_only_array_inputs() -> None:
    polar = np.linspace(0.0, np.pi, 9)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    cfg = SimulationConfig(
        polar_angles=polar,
        azimuthal_angles=azimuth,
        verbose=False,
    )

    polar[:] = -1.0
    azimuth[:] = -1.0

    np.testing.assert_allclose(cfg.polar_angles, np.linspace(0.0, np.pi, 9))
    np.testing.assert_allclose(
        cfg.azimuthal_angles,
        np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
    )
    assert not cfg.polar_angles.flags.writeable
    assert not cfg.azimuthal_angles.flags.writeable


def test_simulation_config_is_execution_state_independent() -> None:
    config_fields = {field.name for field in fields(SimulationConfig)}
    assert "source" not in config_fields
    assert "solve_polarization_basis" not in config_fields
    assert "solver_warm_start" not in config_fields


def test_simulation_config_accepts_cupy_operator_backend() -> None:
    cfg = SimulationConfig(operator_backend="cupy", verbose=False)
    assert cfg.operator_backend == "cupy"
    assert cfg.resolved_postprocessing_backend() == "cupy"


def test_simulation_config_accepts_cupy_gcro_and_recycle_dimension() -> None:
    cfg = SimulationConfig(
        operator_backend="cupy",
        solver_method="gcro",
        solver_restart=16,
        solver_recycle_dim=4,
        verbose=False,
    )
    assert cfg.solver_method == "gcro"
    assert cfg.solver_recycle_dim == 4


def test_simulation_config_rejects_gcro_on_numpy_backend() -> None:
    with pytest.raises(NotImplementedError, match="only with"):
        SimulationConfig(solver_method="gcro", verbose=False)


def test_simulation_config_rejects_gcros_full_restart_recycle_space() -> None:
    with pytest.raises(ValueError, match="smaller than `solver_restart`"):
        SimulationConfig(
            operator_backend="cupy",
            solver_method="gcro",
            solver_restart=8,
            solver_recycle_dim=8,
            verbose=False,
        )


def test_simulation_config_selects_solver_default_by_periodicity() -> None:
    finite = SimulationConfig(verbose=False)
    periodic = SimulationConfig(
        periodic=PeriodicSpec(lattice=RectangularLattice2D(300.0, 300.0)),
        verbose=False,
    )

    assert finite.solver_method == "bicgstab"
    assert periodic.solver_method == "gmres"


def test_simulation_config_accepts_explicit_postprocessing_backend() -> None:
    cfg = SimulationConfig(
        operator_backend="cupy",
        postprocessing_backend="numpy",
        verbose=False,
    )
    assert cfg.postprocessing_backend == "numpy"
    assert cfg.resolved_postprocessing_backend() == "numpy"


def test_simulation_config_rejects_unknown_operator_backend() -> None:
    with pytest.raises(ValueError, match="operator_backend"):
        SimulationConfig(operator_backend="cuda", verbose=False)  # type: ignore[arg-type]


def test_simulation_config_rejects_unknown_postprocessing_backend() -> None:
    with pytest.raises(ValueError, match="postprocessing_backend"):
        SimulationConfig(postprocessing_backend="cuda", verbose=False)  # type: ignore[arg-type]


def test_simulation_config_rejects_finite_cupy_translation_block_cache() -> None:
    with pytest.raises(ValueError, match="cache_translation_blocks=True"):
        SimulationConfig(operator_backend="cupy", cache_translation_blocks=True, verbose=False)


def test_simulation_config_allows_periodic_cupy_translation_block_cache() -> None:
    cfg = SimulationConfig(
        periodic=PeriodicSpec(lattice=RectangularLattice2D(300.0, 300.0)),
        operator_backend="cupy",
        cache_translation_blocks=True,
        verbose=False,
    )

    assert cfg.cache_translation_blocks is True


def test_simulation_config_accepts_mlfmm_coupling_backend_and_options() -> None:
    options = MLFMMOptions(max_leaf_particles=4, max_depth=6)
    cfg = SimulationConfig(
        coupling_backend="mlfmm",
        mlfmm_options=options,
        verbose=False,
    )
    assert cfg.coupling_backend == "mlfmm"
    assert cfg.mlfmm_options == options


def test_simulation_config_rejects_unknown_coupling_backend() -> None:
    with pytest.raises(ValueError, match="coupling_backend"):
        SimulationConfig(coupling_backend="fmm", verbose=False)  # type: ignore[arg-type]


def test_simulation_config_canonicalizes_backend_and_solver_names() -> None:
    cfg = SimulationConfig(
        solver_method="GMRES",  # type: ignore[arg-type]
        operator_backend="NUMPY",  # type: ignore[arg-type]
        coupling_backend="PAIRWISE",  # type: ignore[arg-type]
        postprocessing_backend="INHERIT",  # type: ignore[arg-type]
        verbose=False,
    )

    assert cfg.solver_method == "gmres"
    assert cfg.operator_backend == "numpy"
    assert cfg.coupling_backend == "pairwise"
    assert cfg.postprocessing_backend == "inherit"
    assert cfg.resolved_postprocessing_backend() == "numpy"


def test_simulation_config_accepts_cupy_mlfmm_combination() -> None:
    cfg = SimulationConfig(
        operator_backend="cupy",
        coupling_backend="mlfmm",
        compute_dtype="complex128",
        verbose=False,
    )
    assert cfg.operator_backend == "cupy"
    assert cfg.coupling_backend == "mlfmm"


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_simulation_config_accepts_mlfmm_complex64(backend: str) -> None:
    cfg = SimulationConfig(
        operator_backend=backend,  # type: ignore[arg-type]
        coupling_backend="mlfmm",
        compute_dtype="complex64",
        verbose=False,
    )
    assert cfg.operator_backend == backend
    assert cfg.coupling_backend == "mlfmm"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wavelength": 0.0}, "wavelength"),
        ({"lmax": 0}, "lmax"),
        ({"n_medium": 1.0 + 0.1j}, "n_medium"),
        ({"n_medium": 0.0 + 0j}, "n_medium"),
        ({"radial_lut_dr": -1.0}, "radial_lut_dr"),
        ({"circumscribing_sphere_overlap_atol": -1.0}, "circumscribing_sphere_overlap_atol"),
        ({"check_circumscribing_sphere_overlap": "yes"}, "boolean"),
        ({"force_general_initial_field": "yes"}, "boolean"),
        ({"solver_rtol": 0.0}, "solver_rtol"),
        ({"solver_compute_final_residual": "yes"}, "boolean"),
        ({"solver_restart": 0}, "solver_restart"),
        ({"solver_recycle_dim": 0}, "solver_recycle_dim"),
        ({"solver_maxiter": 0}, "solver_maxiter"),
        ({"solver_direct_max_n": 0}, "solver_direct_max_n"),
        ({"solver_preconditioner": object()}, "callable"),
    ],
)
def test_simulation_config_rejects_invalid_public_policy_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    fn = cast(Any, SimulationConfig)
    with pytest.raises(ValueError, match=match):
        fn(verbose=False, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "source_polar_angles": np.array([0.0, np.pi / 2.0, np.pi]),
                "source_azimuthal_angles": None,
            },
            "Set both `source_polar_angles` and `source_azimuthal_angles`",
        ),
        (
            {
                "farfield_polar_angles": np.array([0.0, np.pi / 2.0, np.pi]),
                "farfield_azimuthal_angles": None,
            },
            "Set both `farfield_polar_angles` and `farfield_azimuthal_angles`",
        ),
    ],
)
def test_simulation_config_requires_complete_override_grid_pairs(
    kwargs: dict[str, object], match: str
) -> None:
    fn = cast(Any, SimulationConfig)
    with pytest.raises(ValueError, match=match):
        fn(verbose=False, **kwargs)


def test_simulation_config_uses_override_grids_and_warns_on_periodic_endpoint() -> None:
    polar = np.array([0.0, np.pi / 2.0, np.pi], dtype=float)
    azimuth = np.array([0.0, np.pi, 2.0 * np.pi], dtype=float)
    with pytest.warns(UserWarning, match="includes both 0 and 2\\*pi"):
        cfg = SimulationConfig(
            verbose=False,
            source_polar_angles=polar,
            source_azimuthal_angles=azimuth,
            farfield_polar_angles=polar,
            farfield_azimuthal_angles=azimuth,
        )
    src_beta, src_alpha = cfg.source_angular_grids()
    ff_beta, ff_alpha = cfg.farfield_angular_grids()
    np.testing.assert_allclose(src_beta, polar)
    np.testing.assert_allclose(src_alpha, azimuth)
    np.testing.assert_allclose(ff_beta, polar)
    np.testing.assert_allclose(ff_alpha, azimuth)
