"""Periodic coupling-operator descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import PeriodicSpec

Array = np.ndarray


@dataclass
class PeriodicCouplingOperator:
    """Bloch-reduced periodic coupling descriptor for rectangular 2D lattices."""

    lmax: int
    k: float
    positions: Array
    lattice: RectangularLattice2D
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype

    def apply(self, x: Array) -> Array:
        """Apply periodic coupling once a lattice-sum kernel is available."""
        raise NotImplementedError(
            "Periodic coupling application is not implemented yet. "
            "Use this operator only for periodic setup validation until a "
            "lattice-sum implementation is available."
        )


__all__ = ["PeriodicCouplingOperator"]
