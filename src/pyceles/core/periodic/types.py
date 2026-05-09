"""Configuration payloads for periodic operator preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.sources import PlaneWave


@dataclass(frozen=True)
class PeriodicOptions:
    """Numerical policy for periodic coupling and output evaluation.

    ``eta`` is an inverse-length Ewald splitting parameter. Leaving it as
    ``None`` selects pyceles's automatic Ewald split: the canonical 2D
    rectangular-lattice value ``sqrt(pi / area)`` is used when stable, and is
    increased only when a cheap geometry-aware structural-sum preflight finds
    it unsafe. Set a numeric ``eta`` to force an expert/manual split.
    ``real_shells`` and ``reciprocal_shells`` are optional explicit
    Chebyshev-index shell truncation counts. Leaving either as ``None``
    enables adaptive shell accumulation with ``shell_tolerance`` and ``max_shells``
    (used as a safety cap, not as an accuracy target). Accelerated evaluators
    that cannot adapt without host/device synchronization may resolve these
    options once to fixed shell counts using the same tolerance.

    ``output_bmax`` controls optional evanescent diffraction orders in periodic
    output bases. ``None`` means propagating orders only for far-field power
    balances; near-field exterior evaluation still requires an explicit output
    basis because evanescent content is an expert convergence knob.
    """

    method: Literal["ewald", "directsum"] = "ewald"
    eta: float | None = None
    real_shells: int | None = None
    reciprocal_shells: int | None = None
    directsum_window: int = 3
    shell_tolerance: float = 1.0e-10
    max_shells: int = 32
    output_bmax: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"ewald", "directsum"}:
            raise ValueError(
                f"`method` must be one of {{'ewald', 'directsum'}}. Got {self.method!r}."
            )
        if self.eta is not None and (not np.isfinite(float(self.eta)) or float(self.eta) <= 0.0):
            raise ValueError(f"`eta` must be finite and positive when set. Got {self.eta!r}.")
        if self.output_bmax is not None:
            bmax = float(self.output_bmax)
            if not np.isfinite(bmax) or bmax <= 0.0:
                raise ValueError(
                    f"`output_bmax` must be finite and positive when set. Got {self.output_bmax!r}."
                )
            object.__setattr__(self, "output_bmax", bmax)
        for name in ("real_shells", "reciprocal_shells"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = int(raw)
            if value != raw:
                raise ValueError(f"`{name}` must be an integer or None. Got {raw!r}.")
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0 when set. Got {raw!r}.")
            object.__setattr__(self, name, value)
        for name in ("directsum_window", "max_shells"):
            raw = getattr(self, name)
            value = int(raw)
            if value != raw:
                raise ValueError(f"`{name}` must be an integer. Got {raw!r}.")
            if name == "max_shells":
                if value <= 0:
                    raise ValueError(f"`{name}` must be > 0. Got {raw!r}.")
            elif value < 0:
                raise ValueError(f"`{name}` must be >= 0. Got {raw!r}.")
            object.__setattr__(self, name, value)
        shell_tolerance = float(self.shell_tolerance)
        if not np.isfinite(shell_tolerance) or shell_tolerance <= 0.0:
            raise ValueError(
                f"`shell_tolerance` must be finite and positive. Got {self.shell_tolerance!r}."
            )
        object.__setattr__(self, "shell_tolerance", shell_tolerance)


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
