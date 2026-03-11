from __future__ import annotations

"""Core operator protocols and prepared-operator boundary."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from pyceles._optional import coerce_array

from .single_body import ParticleTOperator

Array = np.ndarray


@runtime_checkable
class CouplingOperator(Protocol):
    """Prepared many-body coupling operator `W`."""

    def apply(self, x: Array) -> Array: ...


@runtime_checkable
class PrecomputableCouplingOperator(Protocol):
    """Optional coupling protocol for backends that support eager precomputation."""

    def populate(self, *, show_progress: bool = False) -> None: ...


@dataclass
class PreparedOperator:
    """Prepared linear operator `A = I - T W` with explicit `T`/`W` boundaries."""

    lmax: int
    k: float
    positions: Array
    particle_t: ParticleTOperator
    coupling: CouplingOperator
    dtype: np.dtype = np.dtype(np.complex128)

    def apply_W(self, x: Array) -> Array:
        return self.coupling.apply(x)

    def apply_A(self, x: Array) -> Array:
        wx = self.apply_W(x)
        x_arr = coerce_array(x, dtype=self.dtype, prefer_cupy=False)
        return x_arr - self.particle_t.apply(wx)

    def rhs(self, b: Array) -> Array:
        return self.particle_t.rhs(b)

    def rhs_Tb(self, b: Array) -> Array:
        return self.rhs(b)

    def apply_particle_block(self, particle_index: int, block: Array) -> Array:
        return self.particle_t.apply_particle_block(particle_index, block)

    def populate_coupling(self, *, show_progress: bool = False) -> None:
        if isinstance(self.coupling, PrecomputableCouplingOperator):
            self.coupling.populate(show_progress=show_progress)

    @property
    def T_diag(self) -> Array:
        diag = self.particle_t.mode_diagonal()
        if diag is None:
            raise NotImplementedError("Prepared operator does not expose diagonal per-mode T data.")
        return diag

    @property
    def T_M(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[0]

    @property
    def T_N(self) -> Array:
        diags = self.particle_t.degree_diagonals()
        if diags is None:
            raise NotImplementedError("Prepared operator does not expose diagonal T_M/T_N data.")
        return diags[1]


__all__ = ["Array", "CouplingOperator", "PrecomputableCouplingOperator", "PreparedOperator"]
