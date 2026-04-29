"""Configuration payloads for periodic operator preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.sources import PlaneWave


@dataclass(frozen=True)
class PeriodicOptions:
    """Numerical policy for periodic coupling evaluation."""

    method: Literal["ewald", "directsum"] = "ewald"
    eta: float | None = None
    real_shells: int = 12
    reciprocal_shells: int = 12
    directsum_window: int = 3

    def __post_init__(self) -> None:
        if self.method not in {"ewald", "directsum"}:
            raise ValueError(
                f"`method` must be one of {{'ewald', 'directsum'}}. Got {self.method!r}."
            )
        if self.eta is not None and (not np.isfinite(float(self.eta)) or float(self.eta) <= 0.0):
            raise ValueError(f"`eta` must be finite and positive when set. Got {self.eta!r}.")
        for name in ("real_shells", "reciprocal_shells", "directsum_window"):
            raw = getattr(self, name)
            value = int(raw)
            if value != raw:
                raise ValueError(f"`{name}` must be an integer. Got {raw!r}.")
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0. Got {raw!r}.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PeriodicSpec:
    """Periodic-boundary-condition specification for one reduced unit cell."""

    lattice: RectangularLattice2D
    options: PeriodicOptions = field(default_factory=PeriodicOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.lattice, RectangularLattice2D):
            raise TypeError(
                "`periodic.lattice` must be a RectangularLattice2D instance. "
                f"Got {type(self.lattice).__name__}."
            )
        if not isinstance(self.options, PeriodicOptions):
            raise TypeError(
                "`periodic.options` must be a PeriodicOptions instance. "
                f"Got {type(self.options).__name__}."
            )


def plane_wave_k_parallel(source: PlaneWave) -> np.ndarray:
    """Return the incident in-plane Bloch wavevector for a plane wave."""
    k = 2.0 * np.pi / float(source.wavelength) * float(np.real(source.medium_n))
    beta = float(source.polar_angle)
    alpha = float(source.azimuthal_angle)
    return np.array(
        [
            k * np.sin(beta) * np.cos(alpha),
            k * np.sin(beta) * np.sin(alpha),
        ],
        dtype=float,
    )


__all__ = ["PeriodicOptions", "PeriodicSpec", "plane_wave_k_parallel"]
