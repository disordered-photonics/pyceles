"""Periodic coupling-operator descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.directsum import apply_periodic_direct_sum
from pyceles.core.periodic.ewald import apply_periodic_ewald_sum, default_ewald_eta

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
    _ewald_block_cache: dict[tuple[int, int], Array] = field(default_factory=dict)

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
        options = self.periodic.options
        eta = (
            default_ewald_eta(self.periodic.lattice) if options.eta is None else float(options.eta)
        )
        return apply_periodic_ewald_sum(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            x=x,
            lattice=self.periodic.lattice,
            k_parallel=self.k_parallel,
            eta=eta,
            real_shells=int(options.real_shells),
            reciprocal_shells=int(options.reciprocal_shells),
            ab5=self.ab5,
            dtype=self.dtype,
            block_cache=self._ewald_block_cache,
        )


__all__ = ["PeriodicCouplingOperator"]
