"""Shared non-public helpers for simulation geometry and startup behavior."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._logo import print_logo
from pyceles._version import __version__
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import LayeredSphere, Particle, Sphere, Spheroid
from pyceles.linear.solvers import LinearSolveResult

_STARTUP_LOGO_PRINTED = False


def print_startup_logo_once() -> None:
    """Print pyceles logo once per process for verbose user-facing runs."""
    global _STARTUP_LOGO_PRINTED
    if _STARTUP_LOGO_PRINTED:
        return
    print_logo(__version__)
    _STARTUP_LOGO_PRINTED = True


def normalize_particle_geometry(
    particles: Sequence[Particle],
) -> tuple[tuple[Particle, ...], np.ndarray, np.ndarray]:
    """Normalize explicit particle descriptors into solver-ready geometry arrays."""
    part = tuple(particles)
    if len(part) == 0:
        empty_pos = np.zeros((0, 3), dtype=float)
        empty_rad = np.zeros((0,), dtype=float)
        return part, empty_pos, empty_rad
    if not all(isinstance(p, Particle) for p in part):
        bad = [type(p).__name__ for p in part if not isinstance(p, Particle)]
        raise TypeError(f"All entries in `particles` must be Particle instances. Got {bad}.")

    pos = np.asarray([np.asarray(p.position, dtype=float) for p in part], dtype=float).reshape(
        -1, 3
    )
    if not np.all(np.isfinite(pos)):
        raise ValueError("`particles` positions must contain only finite values.")

    rad = np.asarray([float(p.circumscribing_radius()) for p in part], dtype=float).reshape(-1)
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("Particle circumscribing radii must be finite and strictly positive.")

    n_eff: list[complex] = []
    for p in part:
        if isinstance(p, Sphere):
            n_eff.append(complex(p.refractive_index))
        elif isinstance(p, LayeredSphere):
            n_eff.append(complex(p.layer_refractive_indices[-1]))
        elif isinstance(p, Spheroid):
            n_eff.append(complex(p.refractive_index))
        else:
            raise TypeError(
                f"Unsupported particle type {type(p).__name__!r} for refractive-index checks."
            )
    n_eff_arr = np.asarray(n_eff, dtype=np.complex128)
    if not np.all(np.isfinite(n_eff_arr.real)) or not np.all(np.isfinite(n_eff_arr.imag)):
        raise ValueError("Particle refractive indices must be finite.")
    if np.any(n_eff_arr.real <= 0.0):
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


def first_overlapping_circumscribing_pair(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    atol: float = 0.0,
    show_progress: bool = False,
) -> tuple[int, int, float, float] | None:
    """Return first overlapping circumscribing-sphere pair, if any."""
    pos = np.asarray(positions, dtype=float)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    n = int(rad.size)
    if n < 2:
        return None

    atol_f = float(atol)
    r_max = float(np.max(rad))

    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    i_iter: Iterable[int] = range(n - 1)
    if show_progress:
        i_iter = tqdm(i_iter, total=n - 1, desc="Geometry check (circumspheres)")

    for i in i_iter:
        ri = float(rad[i])
        cand = tree.query_ball_point(pos[i], ri + r_max + atol_f)
        for j in cand:
            j = int(j)
            if j <= i:
                continue
            rsum = ri + float(rad[j])
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d + atol_f < rsum:
                return i, j, d, rsum
    return None


def first_periodic_overlapping_circumscribing_pair(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    lattice: RectangularLattice2D,
    atol: float = 0.0,
) -> tuple[int, int, int, int, float, float] | None:
    """Return first circumsphere overlap across nearest periodic images."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    n = int(rad.size)
    if n == 0:
        return None
    atol_f = float(atol)
    shifts = (-1, 0, 1)
    for p in shifts:
        for q in shifts:
            shift = lattice.lattice_vector(p, q)
            for i in range(n):
                for j in range(n):
                    if p == 0 and q == 0 and j <= i:
                        continue
                    if i == j and p == 0 and q == 0:
                        continue
                    rsum = float(rad[i]) + float(rad[j])
                    d = float(np.linalg.norm(pos[i] - pos[j] - shift))
                    if d + atol_f < rsum:
                        return int(i), int(j), int(p), int(q), d, rsum
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
    "first_periodic_overlapping_circumscribing_pair",
    "make_empty_solver_result",
    "normalize_particle_geometry",
    "print_startup_logo_once",
    "warn_local_sources_inside_circumspheres",
]
