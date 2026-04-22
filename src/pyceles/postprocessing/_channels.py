"""Internal helpers for selecting and interpreting result polarization channels."""

from __future__ import annotations

from pyceles.core.sources import JonesPolarizedSource


def is_pure_channel_result(run: object, channel: str, *, atol: float = 1e-12) -> bool:
    """Return True when `run` already represents one pure TE/TM Jones channel.

    The helper stays duck-typed instead of depending on a concrete
    `SimulationResult`, because plotting helpers and lightweight tests often
    work with result-like stubs carrying only the relevant fields.
    """
    src = getattr(getattr(run, "config", None), "source", None)
    if src is None or not isinstance(src, JonesPolarizedSource):
        return False
    pol_jones = getattr(run, "polarization_jones", None)
    if pol_jones is None:
        return False
    a_te, a_tm = pol_jones
    if channel == "te":
        return bool(abs(complex(a_tm)) <= atol and abs(complex(a_te)) > atol)
    if channel == "tm":
        return bool(abs(complex(a_te)) <= atol and abs(complex(a_tm)) > atol)
    return False


__all__ = ["is_pure_channel_result"]
