from __future__ import annotations

import pytest

from pyceles.core.operators import MLFMMOptions
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


def test_simulation_config_rejects_cupy_translation_block_cache() -> None:
    with pytest.raises(ValueError, match="cache_translation_blocks=True"):
        SimulationConfig(operator_backend="cupy", cache_translation_blocks=True, verbose=False)


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
