from __future__ import annotations

"""CuPy-backed diagonal single-particle scattering operator."""

from dataclasses import dataclass, field

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes

Array = np.ndarray


@dataclass
class CuPyDiagonalParticleTOperator:
    """GPU-backed diagonal single-particle `T` operator for sphere families."""

    lmax: int
    n_particles: int
    T_diag: Array
    T_M: Array
    T_N: Array
    dtype: np.dtype = np.dtype(np.complex128)
    _T_diag_gpu: object | None = field(default=None, init=False, repr=False)

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    def _diag_gpu(self):
        cupy, _ = import_cupy()
        if self._T_diag_gpu is None:
            self._T_diag_gpu = cupy.asarray(self.T_diag, dtype=self.dtype)
        return self._T_diag_gpu

    def _apply_impl(self, x: Array | object) -> object:
        arr = coerce_array(x, dtype=self.dtype, prefer_cupy=True).reshape(
            self.n_particles, self.n_modes
        )
        out = self._diag_gpu() * arr
        return out.reshape(self.n_particles * self.n_modes)

    def apply(self, x: Array | object) -> Array | object:
        out = self._apply_impl(x)
        return out if is_cupy_array(x) else asnumpy(out)

    def rhs(self, b: Array | object) -> Array | object:
        out = self._apply_impl(b)
        return out if is_cupy_array(b) else asnumpy(out)

    def apply_particle_block(self, particle_index: int, block: Array | object) -> Array | object:
        cupy, _ = import_cupy()
        local_diag = self._diag_gpu()[int(particle_index)]
        block_arr = cupy.asarray(block, dtype=self.dtype)
        out = local_diag[:, None] * block_arr
        return out if is_cupy_array(block) else asnumpy(out)

    def mode_diagonal(self) -> Array | None:
        return np.asarray(self.T_diag, dtype=self.dtype)

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        return (
            np.asarray(self.T_M, dtype=self.dtype),
            np.asarray(self.T_N, dtype=self.dtype),
        )


__all__ = ["CuPyDiagonalParticleTOperator"]
