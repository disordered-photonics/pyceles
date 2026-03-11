from __future__ import annotations

import pytest

from pyceles.simulation import SimulationConfig


def test_simulation_config_accepts_cupy_operator_backend() -> None:
    cfg = SimulationConfig(operator_backend="cupy", verbose=False)
    assert cfg.operator_backend == "cupy"


def test_simulation_config_rejects_unknown_operator_backend() -> None:
    with pytest.raises(ValueError, match="operator_backend"):
        SimulationConfig(operator_backend="cuda", verbose=False)  # type: ignore[arg-type]


def test_simulation_config_rejects_cupy_translation_block_cache() -> None:
    with pytest.raises(ValueError, match="cache_translation_blocks=True"):
        SimulationConfig(operator_backend="cupy", cache_translation_blocks=True, verbose=False)
