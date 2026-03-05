from __future__ import annotations

import inspect

from pyceles.simulation import Simulation


def test_simulation_constructor_is_particle_only() -> None:
    """Constructor contract: geometry is passed only through `particles`."""
    sig = inspect.signature(Simulation.__init__)
    params = sig.parameters

    assert "particles" in params
    assert params["particles"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["particles"].default is inspect.Parameter.empty
    assert "positions" not in params
    assert "radii" not in params
    assert "n_particle" not in params
