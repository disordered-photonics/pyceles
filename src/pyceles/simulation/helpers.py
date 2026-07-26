"""Shared non-public helpers for simulation geometry and startup behavior."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._logo import print_logo
from pyceles._version import __version__
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import Particle, ParticleCollection
from pyceles.linear.solvers import LinearSolveResult

_STARTUP_LOGO_PRINTED = False


@dataclass(frozen=True)
class _CircumsphereOverlap:
    """First invalid pair found by the circumscribing-sphere validator."""

    particle_i: int
    particle_j: int
    lattice_shift: tuple[int, int]
    distance: float
    required_minimum: float


def print_startup_logo_once() -> None:
    """Print pyceles logo once per process for verbose user-facing runs."""
    global _STARTUP_LOGO_PRINTED
    if _STARTUP_LOGO_PRINTED:
        return
    print_logo(__version__)
    _STARTUP_LOGO_PRINTED = True


def normalize_particle_geometry(
    particles: Sequence[Particle] | ParticleCollection,
) -> tuple[ParticleCollection, np.ndarray, np.ndarray]:
    """Normalize particle inputs into one immutable geometry owner."""
    part = ParticleCollection.from_particles(particles)
    pos = part.positions
    rad = part.circumscribing_radii
    if pos.shape != (len(part), 3):
        raise ValueError(f"Particle positions must have shape ({len(part)}, 3). Got {pos.shape}.")
    if not np.all(np.isfinite(pos)):
        raise ValueError("`particles` positions must contain only finite values.")
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("Particle circumscribing radii must be finite and strictly positive.")

    for n_eff in part.outer_refractive_index_batches():
        if not np.all(np.isfinite(n_eff.real)) or not np.all(np.isfinite(n_eff.imag)):
            raise ValueError("Particle refractive indices must be finite.")
        if np.any(n_eff.real <= 0.0):
            raise ValueError("Real part of particle refractive indices must be strictly positive.")
    return part, pos, rad


def make_empty_solver_result(
    *, dtype: npt.DTypeLike, nrhs: int = 1, method: str = "none"
) -> LinearSolveResult:
    """Build a successful zero-unknown solver result for no-scatterer runs."""
    ncols = int(nrhs)
    if ncols <= 1:
        x = np.zeros((0,), dtype=np.dtype(dtype))
        info: int | np.ndarray = 0
        residual_norm: float | np.ndarray = 0.0
        relative_residual: float | np.ndarray = 0.0
        iterations: int | np.ndarray = 0
        residual_history: np.ndarray | list[np.ndarray | None] | None = np.zeros((0,), dtype=float)
    else:
        x = np.zeros((0, ncols), dtype=np.dtype(dtype))
        info = np.zeros((ncols,), dtype=int)
        residual_norm = np.zeros((ncols,), dtype=float)
        relative_residual = np.zeros((ncols,), dtype=float)
        iterations = np.zeros((ncols,), dtype=int)
        residual_history = [np.zeros((0,), dtype=float) for _ in range(ncols)]
    return LinearSolveResult(
        x=x,
        info=info,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        iterations=iterations,
        method=str(method),
        residual_history=residual_history,
        rhs_count=max(1, ncols),
    )


def _minimum_image_delta_and_shift(
    delta: np.ndarray,
    *,
    lattice: RectangularLattice2D,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return one minimum-image displacement and its rectangular-cell shift."""
    out = np.asarray(delta, dtype=float).copy()
    p = int(np.rint(float(out[0]) / float(lattice.ax)))
    q = int(np.rint(float(out[1]) / float(lattice.ay)))
    out[0] -= p * float(lattice.ax)
    out[1] -= q * float(lattice.ay)
    return out, (p, q)


def _first_periodic_self_overlap(
    radii: np.ndarray,
    *,
    lattice: RectangularLattice2D,
    atol: float,
) -> _CircumsphereOverlap | None:
    """Check one particle against its nearest nonzero rectangular image."""
    if float(lattice.ax) <= float(lattice.ay):
        nearest_distance = float(lattice.ax)
        shift = (1, 0)
    else:
        nearest_distance = float(lattice.ay)
        shift = (0, 1)

    for i, radius in enumerate(radii):
        required_minimum = 2.0 * float(radius)
        if nearest_distance + atol < required_minimum:
            return _CircumsphereOverlap(
                particle_i=int(i),
                particle_j=int(i),
                lattice_shift=shift,
                distance=nearest_distance,
                required_minimum=required_minimum,
            )
    return None


def first_overlapping_circumscribing_pair(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    lattice: RectangularLattice2D | None = None,
    atol: float = 0.0,
    show_progress: bool = False,
) -> _CircumsphereOverlap | None:
    """Return the first finite or x-y-periodic circumsphere overlap.

    A cKDTree supplies a memory-bounded broad phase for both geometry kinds.
    Periodic cells use SciPy's per-axis box lengths with a zero z box length,
    so x and y follow the minimum-image convention while z remains finite.
    Particle self-images are checked separately because a periodic tree stores
    each center only once.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if pos.shape[0] != rad.size:
        raise ValueError(
            "`positions` and `radii` must describe the same number of particles. "
            f"Got {pos.shape[0]} and {rad.size}."
        )
    n = int(rad.size)
    if n == 0:
        return None

    atol_f = float(atol)
    if not np.isfinite(atol_f) or atol_f < 0.0:
        raise ValueError(f"`atol` must be finite and non-negative. Got {atol!r}.")

    search_positions = pos
    boxsize: tuple[float, float, float] | None = None
    if lattice is not None:
        self_overlap = _first_periodic_self_overlap(rad, lattice=lattice, atol=atol_f)
        if self_overlap is not None:
            return self_overlap
        search_positions = pos.copy()
        search_positions[:, 0] = np.mod(search_positions[:, 0], float(lattice.ax))
        search_positions[:, 1] = np.mod(search_positions[:, 1], float(lattice.ay))
        boxsize = (float(lattice.ax), float(lattice.ay), 0.0)

    if n < 2:
        return None

    from scipy.spatial import cKDTree

    tree: Any
    if boxsize is None:
        tree = cKDTree(search_positions)
    else:
        tree = cKDTree(search_positions, boxsize=np.asarray(boxsize, dtype=float))
    r_max = float(np.max(rad))
    i_iter: Iterable[int] = range(n - 1)
    if show_progress:
        label = "periodic circumspheres" if lattice is not None else "circumspheres"
        i_iter = tqdm(i_iter, total=n - 1, desc=f"Geometry check ({label})")

    for i in i_iter:
        ri = float(rad[i])
        candidates = tree.query_ball_point(search_positions[i], ri + r_max + atol_f)
        for j in sorted(int(candidate) for candidate in candidates if int(candidate) > i):
            delta = np.asarray(pos[i] - pos[j], dtype=float)
            shift = (0, 0)
            if lattice is not None:
                delta, shift = _minimum_image_delta_and_shift(delta, lattice=lattice)
            distance = float(np.linalg.norm(delta))
            required_minimum = ri + float(rad[j])
            if distance + atol_f < required_minimum:
                return _CircumsphereOverlap(
                    particle_i=int(i),
                    particle_j=int(j),
                    lattice_shift=shift,
                    distance=distance,
                    required_minimum=required_minimum,
                )
    return None


def warn_local_sources_inside_circumspheres(
    *,
    label: str,
    source_positions: np.ndarray,
    positions: np.ndarray,
    circumscribing_radii: np.ndarray,
) -> None:
    """Warn when local source centers lie inside particle circumscribing spheres."""
    if source_positions.size == 0 or positions.shape[0] == 0:
        return
    deltas = source_positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(deltas, axis=2)
    inside = dist < circumscribing_radii[None, :]
    if np.any(inside):
        j, i = np.argwhere(inside)[0]
        warnings.warn(
            "Untested configuration: local source center lies inside a particle circumscribing sphere. "
            f"Source '{label}', source index {int(j)}, particle index {int(i)}. "
            "Current pyceles local-emitter formulation is validated for emitters in the homogeneous "
            "host outside particles; interior placement may produce unreliable results.",
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "first_overlapping_circumscribing_pair",
    "make_empty_solver_result",
    "normalize_particle_geometry",
    "print_startup_logo_once",
    "warn_local_sources_inside_circumspheres",
]
