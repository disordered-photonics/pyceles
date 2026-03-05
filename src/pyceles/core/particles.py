from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Particle:
    """Base particle descriptor.

    Notes
    -----
    Only `Sphere` is currently supported in the active numerical kernels.
    Other shapes are included as explicit placeholders to expose planned API.
    """

    position: Tuple[float, float, float]

    def pos_array(self, dtype=float) -> np.ndarray:
        """Return particle center as numeric array for kernel consumption."""
        return np.asarray(self.position, dtype=dtype)

    @property
    def kind(self) -> str:
        """Lower-case particle type label used by dispatch utilities."""
        return type(self).__name__.lower()

    def circumscribing_radius(self) -> float:
        """Radius of a sphere enclosing the particle (for broad-phase geometry)."""
        raise NotImplementedError(f"{type(self).__name__} must implement circumscribing_radius().")


@dataclass(frozen=True)
class Sphere(Particle):
    """Homogeneous sphere supported by current scattering kernels."""

    radius: float
    refractive_index: complex = 1.5 + 0j

    def circumscribing_radius(self) -> float:
        """For spheres the circumscribing radius is the physical radius."""
        return float(self.radius)


@dataclass(frozen=True)
class LayeredSphere(Particle):
    """Concentric multilayer sphere for exact multilayer Mie kernels."""

    layer_radii: Tuple[float, ...]
    layer_refractive_indices: Tuple[complex, ...]

    def __post_init__(self) -> None:
        if len(self.layer_radii) == 0:
            raise ValueError("layer_radii must be non-empty.")
        if len(self.layer_radii) != len(self.layer_refractive_indices):
            raise ValueError("layer_radii and layer_refractive_indices must have the same length.")
        if any(float(r) <= 0.0 for r in self.layer_radii):
            raise ValueError("All layer radii must be positive.")
        if any(
            self.layer_radii[i] <= self.layer_radii[i - 1] for i in range(1, len(self.layer_radii))
        ):
            raise ValueError("layer_radii must be strictly increasing (inner to outer).")

    def circumscribing_radius(self) -> float:
        """Outermost shell radius, useful for overlap checks and bounding boxes."""
        return float(self.layer_radii[-1])


@dataclass(frozen=True)
class Ellipsoid(Particle):
    """Placeholder for future ellipsoidal T-matrix support."""

    semi_axes: Tuple[float, float, float]
    refractive_index: complex = 1.5 + 0j
    euler_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.semi_axes) != 3:
            raise ValueError("semi_axes must be a 3-tuple (a, b, c).")
        if any(float(a) <= 0.0 for a in self.semi_axes):
            raise ValueError("All semi-axes must be positive.")

    def circumscribing_radius(self) -> float:
        """Largest semi-axis, i.e. radius of the minimal enclosing sphere."""
        return float(max(self.semi_axes))


def _as_positions_array(positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Normalize particle-center inputs to shape `(N, 3)` float arrays."""
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (N, 3). Got {pos.shape}.")
    if not np.all(np.isfinite(pos)):
        raise ValueError("`positions` must contain only finite values.")
    return pos


def _as_complex_vector(
    values: Sequence[complex] | np.ndarray | complex,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize scalar-or-vector complex inputs to length `n`."""
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim == 0:
        out = np.full((n,), complex(arr), dtype=np.complex128)
    else:
        out = arr.reshape(-1).astype(np.complex128, copy=False)
        if out.shape[0] != n:
            raise ValueError(
                f"`{name}` length ({out.shape[0]}) must match number of particles ({n})."
            )
    if not np.all(np.isfinite(out.real)) or not np.all(np.isfinite(out.imag)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.any(out.real <= 0.0):
        raise ValueError(f"Real part of `{name}` must be strictly positive.")
    return out


def spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
    refractive_indices: Sequence[complex] | np.ndarray | complex,
    into: list[Particle] | None = None,
) -> list[Particle]:
    """Create/extend particle lists with homogeneous spheres from dense arrays.

    This helper is the canonical bridge from array-form geometry generators to
    the explicit particle-descriptor API used by `Simulation`.
    """
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n:
        raise ValueError(f"`radii` length ({rad.shape[0]}) must match number of particles ({n}).")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")
    n_part = _as_complex_vector(refractive_indices, n=n, name="refractive_indices")

    out = [] if into is None else into
    out.extend(
        Sphere(
            position=(float(p[0]), float(p[1]), float(p[2])),
            radius=float(r),
            refractive_index=complex(nr),
        )
        for p, r, nr in zip(pos, rad, n_part)
    )
    return out


def _as_layer_matrix(
    values: Sequence[Sequence[float | complex]] | Sequence[float | complex] | np.ndarray,
    *,
    n_particles: int,
    name: str,
) -> np.ndarray:
    """Normalize layer-value inputs to shape `(N, L)` with optional broadcast."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError(f"`{name}` must contain at least one layer.")
        return np.broadcast_to(arr.reshape(1, -1), (n_particles, int(arr.size))).copy()
    if arr.ndim == 2:
        if arr.shape[0] != n_particles:
            raise ValueError(
                f"`{name}` first dimension ({arr.shape[0]}) must match number of particles ({n_particles})."
            )
        if arr.shape[1] == 0:
            raise ValueError(f"`{name}` must contain at least one layer.")
        return np.asarray(arr).copy()
    raise ValueError(f"`{name}` must be shaped (L,) or (N, L). Got shape {arr.shape}.")


def layered_spheres_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    layer_radii: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    layer_refractive_indices: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
    into: list[Particle] | None = None,
) -> list[Particle]:
    """Create/extend particle lists with concentric layered spheres.

    For convenience, `(L,)` layer inputs are broadcast to all particles; use
    repeated calls with `into=` to append families with different `L`.
    """
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    lr = _as_layer_matrix(layer_radii, n_particles=n, name="layer_radii").astype(float, copy=False)
    li = _as_layer_matrix(
        layer_refractive_indices,
        n_particles=n,
        name="layer_refractive_indices",
    ).astype(np.complex128, copy=False)

    if lr.shape != li.shape:
        raise ValueError(
            f"`layer_radii` and `layer_refractive_indices` must share shape. Got {lr.shape} vs {li.shape}."
        )
    if np.any(~np.isfinite(lr)) or np.any(lr <= 0.0):
        raise ValueError("`layer_radii` entries must be finite and strictly positive.")
    if np.any(np.diff(lr, axis=1) <= 0.0):
        raise ValueError("Each particle `layer_radii` row must be strictly increasing.")
    if not np.all(np.isfinite(li.real)) or not np.all(np.isfinite(li.imag)):
        raise ValueError("`layer_refractive_indices` must contain only finite values.")
    if np.any(li.real <= 0.0):
        raise ValueError("Real part of `layer_refractive_indices` must be strictly positive.")

    out = [] if into is None else into
    out.extend(
        LayeredSphere(
            position=(float(p[0]), float(p[1]), float(p[2])),
            layer_radii=tuple(float(v) for v in r_row),
            layer_refractive_indices=tuple(complex(v) for v in n_row),
        )
        for p, r_row, n_row in zip(pos, lr, li)
    )
    return out


def ellipsoids_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    semi_axes: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    refractive_indices: Sequence[complex] | np.ndarray | complex,
    euler_angles: Sequence[Sequence[float]] | Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    into: list[Particle] | None = None,
) -> list[Particle]:
    """Create/extend particle lists with homogeneous ellipsoid descriptors.

    This is currently a geometry helper only; full ellipsoidal scattering
    kernels are planned for future integration.
    """
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    axes = np.asarray(semi_axes, dtype=float)
    if axes.ndim == 1:
        if axes.shape[0] != 3:
            raise ValueError(f"`semi_axes` must have shape (3,) or (N,3). Got {axes.shape}.")
        axes = np.broadcast_to(axes.reshape(1, 3), (n, 3)).copy()
    elif axes.ndim == 2 and axes.shape == (n, 3):
        axes = axes.copy()
    else:
        raise ValueError(f"`semi_axes` must have shape (3,) or (N,3). Got {axes.shape}.")
    if np.any(~np.isfinite(axes)) or np.any(axes <= 0.0):
        raise ValueError("`semi_axes` entries must be finite and strictly positive.")

    eul = np.asarray(euler_angles, dtype=float)
    if eul.ndim == 1:
        if eul.shape[0] != 3:
            raise ValueError(f"`euler_angles` must have shape (3,) or (N,3). Got {eul.shape}.")
        eul = np.broadcast_to(eul.reshape(1, 3), (n, 3)).copy()
    elif eul.ndim == 2 and eul.shape == (n, 3):
        eul = eul.copy()
    else:
        raise ValueError(f"`euler_angles` must have shape (3,) or (N,3). Got {eul.shape}.")
    if np.any(~np.isfinite(eul)):
        raise ValueError("`euler_angles` must contain only finite values.")

    n_part = _as_complex_vector(refractive_indices, n=n, name="refractive_indices")
    out = [] if into is None else into
    out.extend(
        Ellipsoid(
            position=(float(p[0]), float(p[1]), float(p[2])),
            semi_axes=(float(a[0]), float(a[1]), float(a[2])),
            refractive_index=complex(nr),
            euler_angles=(float(ang[0]), float(ang[1]), float(ang[2])),
        )
        for p, a, nr, ang in zip(pos, axes, n_part, eul)
    )
    return out
