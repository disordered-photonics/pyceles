from __future__ import annotations

from typing import Literal

Polarization = Literal["TE", "TM"]


def pure_polarization_label(
    a_te: complex,
    a_tm: complex,
    *,
    atol: float = 1e-15,
) -> Polarization | None:
    """Return pure-channel label when Jones weights represent TE-only or TM-only."""
    if abs(a_tm) <= atol and abs(a_te) > atol:
        return "TE"
    if abs(a_te) <= atol and abs(a_tm) > atol:
        return "TM"
    return None
