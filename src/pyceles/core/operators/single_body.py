"""Prepared single-particle scattering-operator boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pyceles.core.indexing import n_modes

from .groups import PreparedParticleTGroup

Array = np.ndarray
COMPLEX128_DTYPE = np.dtype(np.complex128)


def build_T_mode_diagonal(lmax: int, T_M: Array, T_N: Array) -> Array:
    """Expand per-particle degree diagonals into flattened SVWF mode order."""

    lmax_i = int(lmax)
    t_m = np.asarray(T_M)
    t_n = np.asarray(T_N)
    if t_m.shape != t_n.shape:
        raise ValueError(
            f"T_M and T_N must have identical shapes. Got {t_m.shape} and {t_n.shape}."
        )

    n_particles = int(t_m.shape[0])
    n_scalar = lmax_i * (lmax_i + 2)
    out = np.zeros(
        (n_particles, n_modes(lmax_i)),
        dtype=np.result_type(t_m.dtype, t_n.dtype, np.complex64),
    )
    for degree in range(1, lmax_i + 1):
        start = (degree - 1) * (degree + 1)
        stop = start + (2 * degree + 1)
        out[:, start:stop] = t_m[:, degree : degree + 1]
        out[:, n_scalar + start : n_scalar + stop] = t_n[:, degree : degree + 1]
    return out


class ParticleTOperator(Protocol):
    """Prepared particle-local (single-particle) scattering operator `T`."""

    lmax: int
    n_particles: int
    dtype: np.dtype

    def apply(self, x: Array) -> Array: ...

    def apply_adjoint(self, x: Array) -> Array: ...

    def rhs(self, b: Array) -> Array: ...

    def apply_particle_block(self, particle_index: int, block: Array) -> Array: ...

    def mode_diagonal(self) -> Array | None: ...

    def degree_diagonals(self) -> tuple[Array, Array] | None: ...


@dataclass
class CompositeParticleTOperator:
    """Composite particle-local / single-particle T operator."""

    lmax: int
    n_particles: int
    groups: Sequence[PreparedParticleTGroup]
    dtype: np.dtype = COMPLEX128_DTYPE
    _particle_to_group: np.ndarray = field(init=False, repr=False)
    _particle_to_local: np.ndarray = field(init=False, repr=False)
    _full_group: PreparedParticleTGroup | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ns = int(self.n_particles)
        group_of = np.full(ns, -1, dtype=np.int64)
        local_of = np.full(ns, -1, dtype=np.int64)
        for gidx, group in enumerate(self.groups):
            ids = np.asarray(group.particle_indices, dtype=np.int64).reshape(-1)
            if ids.size == 0:
                raise ValueError("Particle-T operator groups must be non-empty.")
            if np.any(ids < 0) or np.any(ids >= ns):
                raise ValueError("Particle-T operator group indices out of bounds.")
            if np.any(group_of[ids] != -1):
                raise ValueError("Particle-T operator groups must not overlap.")
            group_of[ids] = gidx
            local_of[ids] = np.arange(ids.size, dtype=np.int64)
        if np.any(group_of < 0):
            raise ValueError("Particle-T operator groups must cover all particles.")
        self._particle_to_group = group_of
        self._particle_to_local = local_of
        if len(self.groups) == 1:
            ids = np.asarray(self.groups[0].particle_indices, dtype=np.int64).reshape(-1)
            if np.array_equal(ids, np.arange(ns, dtype=np.int64)):
                self._full_group = self.groups[0]

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    def _reshape_input(self, x: Array) -> tuple[Array, tuple[int, ...]]:
        raw = np.asarray(x, dtype=self.dtype)
        expected = self.n_particles * self.n_modes
        if raw.ndim == 1:
            if raw.size != expected:
                raise ValueError(f"Input length must be {expected}. Got {raw.size}.")
            return raw.reshape(self.n_particles, self.n_modes), raw.shape
        elif raw.ndim == 2:
            if raw.shape == (self.n_particles, self.n_modes):
                return raw, raw.shape
            elif raw.shape[0] == expected:
                return (
                    raw.reshape(self.n_particles, self.n_modes, raw.shape[1]),
                    raw.shape,
                )
            else:
                raise ValueError(
                    "2D input must have shape "
                    f"({self.n_particles}, {self.n_modes}) or ({expected}, nrhs). Got {raw.shape}."
                )
        elif raw.ndim == 3 and raw.shape[:2] == (self.n_particles, self.n_modes):
            return raw, raw.shape
        else:
            raise ValueError(
                "Input must have shape "
                f"({expected},), ({self.n_particles}, {self.n_modes}), "
                f"({expected}, nrhs), or ({self.n_particles}, {self.n_modes}, nrhs). "
                f"Got {raw.shape}."
            )

    def apply(self, x: Array) -> Array:
        arr, output_shape = self._reshape_input(x)
        if self._full_group is not None:
            return np.asarray(self._full_group.apply_subset(arr), dtype=self.dtype).reshape(
                output_shape
            )
        # Groups are disjoint and exhaustive, so no output element needs a
        # zero default before the group assignments below.
        out = np.empty_like(arr, dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            out[ids] = group.apply_subset(arr[ids])
        return out.reshape(output_shape)

    def apply_adjoint(self, x: Array) -> Array:
        arr, output_shape = self._reshape_input(x)
        if self._full_group is not None:
            return np.asarray(self._full_group.apply_adjoint_subset(arr), dtype=self.dtype).reshape(
                output_shape
            )
        out = np.empty_like(arr, dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            out[ids] = group.apply_adjoint_subset(arr[ids])
        return out.reshape(output_shape)

    def rhs(self, b: Array) -> Array:
        arr, output_shape = self._reshape_input(b)
        if self._full_group is not None:
            return np.asarray(self._full_group.rhs_subset(arr), dtype=self.dtype).reshape(
                output_shape
            )
        out = np.empty_like(arr, dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            out[ids] = group.rhs_subset(arr[ids])
        return out.reshape(output_shape)

    def apply_particle_block(self, particle_index: int, block: Array) -> Array:
        i = int(particle_index)
        gidx = int(self._particle_to_group[i])
        local = int(self._particle_to_local[i])
        return self.groups[gidx].apply_local_block(local, block)

    def mode_diagonal(self) -> Array | None:
        out = np.empty((self.n_particles, self.n_modes), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diag = group.mode_diagonal()
            if diag is None:
                return None
            out[ids] = np.asarray(diag, dtype=self.dtype)
        return out

    def degree_diagonals(self) -> tuple[Array, Array] | None:
        out_m = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        out_n = np.empty((self.n_particles, self.lmax + 1), dtype=self.dtype)
        for group in self.groups:
            ids = np.asarray(group.particle_indices, dtype=np.int64)
            diags = group.degree_diagonals()
            if diags is None:
                return None
            out_m[ids] = np.asarray(diags[0], dtype=self.dtype)
            out_n[ids] = np.asarray(diags[1], dtype=self.dtype)
        return out_m, out_n


__all__ = ["CompositeParticleTOperator", "ParticleTOperator", "build_T_mode_diagonal"]
