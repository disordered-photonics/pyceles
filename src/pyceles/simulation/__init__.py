"""Public simulation package.

This package owns the high-level workflow layer above the numerical kernels:
configuration, solve-only multi-source payloads, postprocessed run results, and
the `Simulation` orchestrator itself.
"""

from __future__ import annotations

from .config import SimulationConfig
from .results import (
    ChannelResult,
    MultiSourceResult,
    MultiSourceSolveResult,
    PolarizationResult,
    ResultRetention,
    SimulationResult,
    UnpolarizedDiagnostics,
)
from .workflow import Simulation

__all__ = [
    "ChannelResult",
    "MultiSourceResult",
    "MultiSourceSolveResult",
    "PolarizationResult",
    "ResultRetention",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "UnpolarizedDiagnostics",
]
