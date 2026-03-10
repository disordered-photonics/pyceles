from __future__ import annotations

"""Public simulation package.

This package owns the high-level workflow layer above the numerical kernels:
configuration, solve-only multi-source payloads, postprocessed run results, and
the `Simulation` orchestrator itself.
"""

from .config import SimulationConfig
from .results import MultiSourceSimulationResult, SimulationResult, SolvedSourcesResult
from .workflow import Simulation

__all__ = [
    "MultiSourceSimulationResult",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "SolvedSourcesResult",
]
