from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import MLFMMOptions
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.sources import PlaneWave
from pyceles.simulation import SimulationConfig


def test_simulation_config_accepts_cupy_operator_backend() -> None:
    cfg = SimulationConfig(operator_backend="cupy", verbose=False)
    assert cfg.operator_backend == "cupy"
    assert cfg.resolved_postprocessing_backend() == "cupy"


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
        source=PlaneWave(wavelength=550.0, medium_n=1.0 + 0j),
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
        ({"solver_maxiter": 0}, "solver_maxiter"),
        ({"solver_direct_max_n": 0}, "solver_direct_max_n"),
        ({"solver_preconditioner": object()}, "callable"),
        ({"solver_warm_start": np.zeros((1, 1, 1))}, "1D, 2D, or None"),
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
                "source": PlaneWave(
                    wavelength=551.0,
                    medium_n=1.0 + 0j,
                    polarization="TE",
                    polar_angle=0.0,
                    azimuthal_angle=0.0,
                    amplitude=1.0,
                )
            },
            "source.wavelength",
        ),
        (
            {
                "source": PlaneWave(
                    wavelength=550.0,
                    medium_n=1.2 + 0j,
                    polarization="TE",
                    polar_angle=0.0,
                    azimuthal_angle=0.0,
                    amplitude=1.0,
                )
            },
            "source.medium_n",
        ),
    ],
)
def test_simulation_config_rejects_inconsistent_embedded_source(
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
