from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

ParticleTRepresentation = Literal["diagonal", "axisymmetric", "dense"]


@dataclass(frozen=True)
class Particle:
    """Base particle descriptor.

    Notes
    -----
    `Sphere`, `LayeredSphere`, and `Spheroid` are supported by active solver
    kernels, though not every downstream postprocessing path is equally mature
    for every particle family.
    """

    position: tuple[float, float, float]

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

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Preferred prepared-operator representation for this particle family.

        The representation only describes how the particle-local `T` operator
        should be stored/applied. It does not decide the coupling backend.
        Keeping this declaration on the particle descriptor lets operator
        preparation choose specialized groups in one place instead of growing
        solver-wide type checks.
        """
        return "dense"


@dataclass(frozen=True)
class Sphere(Particle):
    """Homogeneous sphere supported by current scattering kernels."""

    radius: float
    refractive_index: complex = 1.5 + 0j

    def circumscribing_radius(self) -> float:
        """For spheres the circumscribing radius is the physical radius."""
        return float(self.radius)

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Spheres use the diagonal Mie fast path."""
        return "diagonal"


@dataclass(frozen=True)
class LayeredSphere(Particle):
    """Concentric multilayer sphere for exact multilayer Mie kernels."""

    layer_radii: tuple[float, ...]
    layer_refractive_indices: tuple[complex, ...]

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

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Layered spheres remain diagonal in the SVWF basis."""
        return "diagonal"


@dataclass(frozen=True)
class Spheroid(Particle):
    """Homogeneous axisymmetric particle with spherical-basis T-block support."""

    equatorial_radius: float
    polar_radius: float
    refractive_index: complex = 1.5 + 0j
    euler_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if float(self.equatorial_radius) <= 0.0:
            raise ValueError("equatorial_radius must be positive.")
        if float(self.polar_radius) <= 0.0:
            raise ValueError("polar_radius must be positive.")

    def circumscribing_radius(self) -> float:
        """Largest semi-axis of the spheroid."""
        return float(max(float(self.equatorial_radius), float(self.polar_radius)))

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        """Axisymmetric particles admit a narrower-than-dense T representation."""
        return "axisymmetric"


def _rotation_matrix_zyz_lab_to_body(euler_angles: tuple[float, float, float]) -> np.ndarray:
    """Return the lab-to-body rotation matrix for particle Euler angles.

    The convention matches the particle-orientation usage elsewhere in
    `pyceles`: a `z-y'-z''` Euler triplet with the body frame aligned to the
    particle at zero angles.
    """

    alpha = float(euler_angles[0])
    beta = float(euler_angles[1])
    gamma = float(euler_angles[2])
    rot_1 = np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0.0],
            [-np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_2 = np.array(
        [
            [np.cos(beta), 0.0, -np.sin(beta)],
            [0.0, 1.0, 0.0],
            [np.sin(beta), 0.0, np.cos(beta)],
        ],
        dtype=float,
    )
    rot_3 = np.array(
        [
            [np.cos(gamma), np.sin(gamma), 0.0],
            [-np.sin(gamma), np.cos(gamma), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return np.asarray(rot_3 @ rot_2 @ rot_1, dtype=float)


def particle_contains_points(
    particle: Particle,
    points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for points inside the physical particle boundary."""

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    center = np.asarray(particle.position, dtype=float).reshape(3)
    rel = pts - center[None, :]

    if isinstance(particle, Sphere):
        radius = float(particle.radius)
        return np.asarray(np.sum(rel * rel, axis=1) < (radius**2), dtype=bool)

    if isinstance(particle, LayeredSphere):
        radius = float(particle.layer_radii[-1])
        return np.asarray(np.sum(rel * rel, axis=1) < (radius**2), dtype=bool)

    if isinstance(particle, Spheroid):
        body = rel @ _rotation_matrix_zyz_lab_to_body(particle.euler_angles).T
        a = float(particle.equatorial_radius)
        c = float(particle.polar_radius)
        rho2 = (body[:, 0] / a) ** 2 + (body[:, 1] / a) ** 2 + (body[:, 2] / c) ** 2
        return np.asarray(rho2 < 1.0, dtype=bool)

    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


def particle_t_signature(particle: Particle) -> tuple[object, ...]:
    """Return a position-independent cache key for solver-facing particle-T data.

    The particle-local `T` operator depends on shape, material, and orientation,
    but not on the particle center. This helper lets preparation paths reuse one
    computed particle kernel across many identical particles placed at different
    positions in the cluster.

    Notes
    -----
    This signature intentionally includes every particle field except
    `position`, so a rotated spheroid is treated as distinct from the same
    spheroid at another orientation. That is the correct cache key when the
    reused object is the final lab-frame operator consumed by the solver.
    """

    payload = []
    for field in fields(type(particle)):
        if field.name == "position":
            continue
        payload.append(getattr(particle, field.name))
    return (type(particle), *payload)


def particle_intrinsic_t_signature(particle: Particle) -> tuple[object, ...]:
    """Return a position/orientation-independent key for intrinsic particle data.

    This narrower signature is useful when particle preparation naturally splits
    into:
    - an intrinsic body-frame object that depends only on shape/material, and
    - a solver-facing lab-frame object that additionally depends on orientation.

    Spheres and layered spheres do not currently need that distinction, but
    axisymmetric particles do: one aligned spheroid model can be reused across
    many differently oriented copies before the final SVWF rotation step.
    """

    payload = []
    for field in fields(type(particle)):
        if field.name in {"position", "euler_angles"}:
            continue
        payload.append(getattr(particle, field.name))
    return (type(particle), *payload)


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
        for p, r, nr in zip(pos, rad, n_part, strict=True)
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
        for p, r_row, n_row in zip(pos, lr, li, strict=True)
    )
    return out


def _as_positive_float_vector(
    values: Sequence[float] | np.ndarray | float,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Normalize scalar-or-vector positive float inputs to length `n`."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        out = np.full((n,), float(arr), dtype=float)
    else:
        out = arr.reshape(-1).astype(float, copy=False)
        if out.shape[0] != n:
            raise ValueError(
                f"`{name}` length ({out.shape[0]}) must match number of particles ({n})."
            )
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError(f"`{name}` must contain finite strictly positive values.")
    return out


def spheroids_from_arrays(
    *,
    positions: Sequence[Sequence[float]] | np.ndarray,
    equatorial_radii: Sequence[float] | np.ndarray | float,
    polar_radii: Sequence[float] | np.ndarray | float,
    refractive_indices: Sequence[complex] | np.ndarray | complex,
    euler_angles: Sequence[Sequence[float]] | Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    into: list[Particle] | None = None,
) -> list[Particle]:
    """Create/extend particle lists with homogeneous spheroid descriptors."""
    pos = _as_positions_array(positions)
    n = int(pos.shape[0])
    eq = _as_positive_float_vector(equatorial_radii, n=n, name="equatorial_radii")
    po = _as_positive_float_vector(polar_radii, n=n, name="polar_radii")

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
        Spheroid(
            position=(float(p[0]), float(p[1]), float(p[2])),
            equatorial_radius=float(a_eq),
            polar_radius=float(a_po),
            refractive_index=complex(nr),
            euler_angles=(float(ang[0]), float(ang[1]), float(ang[2])),
        )
        for p, a_eq, a_po, nr, ang in zip(pos, eq, po, n_part, eul, strict=True)
    )
    return out
