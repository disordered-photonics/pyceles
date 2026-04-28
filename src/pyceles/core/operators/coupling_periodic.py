"""Periodic coupling-operator descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.directsum import apply_periodic_direct_sum

Array = np.ndarray


@dataclass
class PeriodicCouplingOperator:
    """Bloch-reduced periodic coupling descriptor for rectangular 2D lattices."""

    lmax: int
    k: float
    positions: Array
    ab5: Array
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype

    def apply(self, x: Array) -> Array:
        """Apply the configured periodic coupling model to stacked coefficients."""
        if self.periodic.options.method == "directsum":
            return apply_periodic_direct_sum(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                lattice=self.periodic.lattice,
                k_parallel=self.k_parallel,
                window=int(self.periodic.options.directsum_window),
                ab5=self.ab5,
                dtype=self.dtype,
            )
        raise NotImplementedError(
            "Ewald periodic coupling is not implemented yet. "
            "For small validation runs, use PeriodicOptions(method='directsum')."
        )


__all__ = ["PeriodicCouplingOperator"]
