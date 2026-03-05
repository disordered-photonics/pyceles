from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._logo import print_logo
from pyceles._version import __version__
from pyceles.core.angular import uniform_periodic_azimuth_grid, uniform_polar_grid
from pyceles.core.indexing import n_modes
from pyceles.core.matvec import (
    PreparedMatvec,
    assemble_dense_A_numpy,
    prepare_matvec,
)
from pyceles.core.particles import LayeredSphere, Particle, Sphere, Spheroid
from pyceles.core.projection import project_source_to_svwf
from pyceles.core.sources import (
    DipoleCollection,
    DipoleSource,
    JonesPolarizedSource,
    PlaneWave,
    Source,
)
from pyceles.linear.preconditioner import make_grid_block_preconditioner
from pyceles.linear.solvers import (
    DenseLUFactorization,
    LinearSolveResult,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    solve_linear_system,
)
from pyceles.postprocessing.farfield import (
    FarFieldPatterns,
    compute_far_field_patterns,
    finite_beam_power_fractions,
    plane_wave_cross_sections,
    pwp_power_decomposition,
)

_STARTUP_LOGO_PRINTED = False


def _print_startup_logo_once() -> None:
    """Print pyceles logo once per process for verbose user-facing runs."""
    global _STARTUP_LOGO_PRINTED
    if _STARTUP_LOGO_PRINTED:
        return
    print_logo(__version__)
    _STARTUP_LOGO_PRINTED = True


def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    """Validate monotone angular quadrature nodes used in field integrations.

    These arrays usually represent sampled polar/azimuthal angles.
    Requiring a finite, strictly increasing 1D grid avoids ambiguous trapezoidal
    weights and prevents non-physical duplicate angular directions.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array; got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(f"`{name}` must contain at least 2 samples; got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return arr


def _validate_angular_grid_pair(
    *,
    polar_name: str,
    azimuthal_name: str,
    polar_values: np.ndarray,
    azimuthal_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one `(beta, alpha)` angular-grid pair used in quadratures/PWPs."""
    polar = _as_1d_float_array(polar_name, polar_values)
    azimuth = _as_1d_float_array(azimuthal_name, azimuthal_values)
    if polar[0] < -1e-12 or polar[-1] > np.pi + 1e-12:
        raise ValueError(f"`{polar_name}` must lie within [0, pi].")
    if azimuth[0] < -1e-12 or azimuth[-1] > 2.0 * np.pi + 1e-12:
        raise ValueError(f"`{azimuthal_name}` must lie within [0, 2*pi].")
    return polar, azimuth


def _warn_redundant_periodic_azimuth_endpoint(*, azimuth_name: str, azimuth: np.ndarray) -> None:
    """Warn on duplicated periodic endpoints (0 and 2*pi) in azimuth grids."""
    if azimuth.size < 2:
        return
    a0 = float(azimuth[0])
    a1 = float(azimuth[-1])
    if (
        np.isclose(a0, 0.0, rtol=0.0, atol=1e-12)
        and np.isclose(a1, 2.0 * np.pi, rtol=0.0, atol=1e-12)
        and np.isclose(a1 - a0, 2.0 * np.pi, rtol=0.0, atol=1e-12)
    ):
        warnings.warn(
            f"`{azimuth_name}` includes both 0 and 2*pi. For periodic angular integrals, "
            "prefer `endpoint=False` on [0, 2*pi) to avoid redundant work and keep periodic fast paths enabled.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_particle_geometry(
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


def _make_empty_solver_result(
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


def _first_overlapping_circumscribing_pair(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    atol: float = 0.0,
    show_progress: bool = False,
) -> tuple[int, int, float, float] | None:
    """Return first overlapping circumscribing-sphere pair, if any.

    The multiple-scattering T-matrix superposition assumes disjoint
    circumscribing spheres. This helper checks the condition
    ``||r_i-r_j|| + atol >= R_i + R_j`` and returns the first violating pair.

    The implementation uses a KD-tree candidate search so memory does not scale
    as a full pairwise distance matrix.
    """
    pos = np.asarray(positions, dtype=float)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    n = int(rad.size)
    if n < 2:
        return None

    atol_f = float(atol)
    r_max = float(np.max(rad))

    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    i_iter = range(n - 1)
    if show_progress:
        i_iter = tqdm(i_iter, total=n - 1, desc="Geometry check (circumspheres)")

    for i in i_iter:
        ri = float(rad[i])
        # Candidate pruning: no sphere farther than ri + r_max can overlap i.
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


def _mix_pwp_dict(
    p1: dict,
    p2: dict,
    *,
    a1: complex,
    a2: complex,
    dtype: npt.DTypeLike,
) -> dict:
    """Coherently superpose two plane-wave spectra on the same angular grid.

    Used when TE/TM basis far fields are already available and the user asks for
    a Jones-combined channel, so we reuse existing PWPs instead of recomputing.
    """
    out = dict(p1)
    out["coeff"] = np.asarray(a1 * p1["coeff"] + a2 * p2["coeff"], dtype=np.dtype(dtype))
    return out


def _mix_optional_pwp_dict(
    p1: dict | None,
    p2: dict | None,
    *,
    a1: complex,
    a2: complex,
    dtype: npt.DTypeLike,
) -> dict | None:
    """Mix optional PWP channels while preserving `None` for unavailable fields."""
    if p1 is None or p2 is None:
        return None
    return _mix_pwp_dict(p1, p2, a1=a1, a2=a2, dtype=dtype)


def _mix_farfield_patterns(
    ff_te: FarFieldPatterns,
    ff_tm: FarFieldPatterns,
    *,
    a_te: complex,
    a_tm: complex,
    dtype: npt.DTypeLike,
) -> FarFieldPatterns:
    """Build requested Jones far field by coherent TE/TM basis recombination."""
    initial_te = _mix_optional_pwp_dict(
        ff_te.initial_te,
        ff_tm.initial_te,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )
    initial_tm = _mix_optional_pwp_dict(
        ff_te.initial_tm,
        ff_tm.initial_tm,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )
    scattered_te = _mix_pwp_dict(
        ff_te.scattered_te,
        ff_tm.scattered_te,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )
    scattered_tm = _mix_pwp_dict(
        ff_te.scattered_tm,
        ff_tm.scattered_tm,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )
    total_te = _mix_optional_pwp_dict(
        ff_te.total_te,
        ff_tm.total_te,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )
    total_tm = _mix_optional_pwp_dict(
        ff_te.total_tm,
        ff_tm.total_tm,
        a1=a_te,
        a2=a_tm,
        dtype=dtype,
    )

    return FarFieldPatterns(
        initial_te=initial_te,
        initial_tm=initial_tm,
        scattered_te=scattered_te,
        scattered_tm=scattered_tm,
        total_te=total_te,
        total_tm=total_tm,
    )


@dataclass(frozen=True)
class SimulationConfig:
    """High-level configuration for one homogeneous-medium many-sphere run.

    Polarization/multi-source options:
    - Propagating TE/TM sources (for example `PlaneWave`, `GaussianBeam`,
      `SLMSource`, `BesselBeam`) use
      `source.polarization` as `"TE"`, `"TM"`, or Jones `(a_te, a_tm)`.
    - Local sources (`DipoleSource`, `DipoleCollection`) use dipole moments
      and positions instead of TE/TM polarization labels.
    - `Simulation.solve_sources(...)` is the canonical solve-only API for any
      labeled source set (shared operator, multi-RHS solve).
    - `Simulation.postprocess_sources(...)` turns a solve payload into per-channel
      `SimulationResult` outputs, with optional far-field/power diagnostics.
    - `solve_polarization_basis=True` keeps a convenience `run()` wrapper that
      internally solves TE/TM basis channels and adds mixed+basis+unpolarized
      outputs.
    Angular-grid policy:
    - `polar_angles`/`azimuthal_angles` define the shared CELES-style default
      grid used by source projection, near-field initial-field quadrature, and
      far-field outputs.
    - `source_*` and `farfield_*` optional overrides keep these stages
      separable in the API (SMUTHI-style flexibility) while preserving one
      shared default behavior out of the box.
    - Accuracy implication: the source-projection grid controls RHS fidelity for
      finite-width beams; a finer far-field grid cannot recover source angular
      content that was undersampled in the source projection.
    - Plane-wave source projection is analytic and does not rely on angular
      quadrature nodes; split grids mainly matter for Gaussian/AS sources.

    Solver extensibility options:
    - `solver_warm_start`: optional initial guess for iterative/direct solves,
      shape `(unknowns,)` or `(unknowns, nrhs)`.
    - `solver_compute_final_residual`: if `True`, report final
      `||Ax-b||/||b||` diagnostics after each solve; disable for
      high-throughput repeated direct solves when this extra check is not
      needed.
    - `solver_preconditioner`: optional callable preconditioner operator.
    - `solver_preconditioner_kind`: built-in preconditioner selection
      (`"none"` or `"grid_block"`).
    Geometry validity options:
    - `check_circumscribing_sphere_overlap=True` enforces disjoint
      circumscribing spheres (T-matrix superposition validity condition).
    - `circumscribing_sphere_overlap_atol` controls absolute geometric tolerance.
    The object gathers physical knobs (wavelength, source, truncation) and
    numerical policy (solver, precision, caching, preconditioning) so a run is
    reproducible and explicit.
    """

    wavelength: float = 550.0
    n_medium: complex = 1.0 + 0j
    lmax: int = 3
    source: Source | None = None
    # Shared CELES-like default angular grid. If no stage-specific grids are
    # provided, this pair is used for both source projection and far-field bins.
    # Use periodic azimuth sampling on [0, 2*pi) (endpoint=False) so periodic
    # fast-path detection remains active by default.
    polar_angles: np.ndarray = field(default_factory=lambda: uniform_polar_grid(5001))
    azimuthal_angles: np.ndarray = field(default_factory=lambda: uniform_periodic_azimuth_grid(201))
    # Optional source-projection quadrature grid (RHS / initial-field projection).
    # Set both or neither.
    source_polar_angles: np.ndarray | None = None
    source_azimuthal_angles: np.ndarray | None = None
    # Optional far-field output sampling grid (scattered/initial/total PWPs).
    # Set both or neither.
    farfield_polar_angles: np.ndarray | None = None
    farfield_azimuthal_angles: np.ndarray | None = None
    radial_lut_dr: float = 1.0
    force_general_initial_field: bool = False
    solver_method: Literal["auto", "gmres", "bicgstab", "lgmres", "gcrotmk", "direct"] = "direct"
    solver_direct_max_n: int = 15_000
    solver_rtol: float = 1e-5
    solver_compute_final_residual: bool = True
    solver_restart: int = 100
    solver_maxiter: int = 1000
    solver_warm_start: np.ndarray | None = None
    solver_preconditioner: Callable[[np.ndarray], np.ndarray] | None = None
    solver_preconditioner_kind: Literal["none", "grid_block"] = "none"
    solver_preconditioner_subdivisions: int | tuple[int, int, int] = 2
    solver_preconditioner_cubic_bbox: bool = True
    solver_preconditioner_max_block_unknowns: int | None = None
    compute_dtype: Literal["complex64", "complex128"] = "complex128"
    accum_dtype: Literal["complex64", "complex128"] = "complex128"
    cache_translation_blocks: bool = False
    check_circumscribing_sphere_overlap: bool = True
    circumscribing_sphere_overlap_atol: float = 0.0
    solve_polarization_basis: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate physical consistency and numerical-policy constraints."""
        if not (float(self.wavelength) > 0.0):
            raise ValueError(f"`wavelength` must be > 0. Got {self.wavelength!r}.")
        if int(self.lmax) < 1:
            raise ValueError(f"`lmax` must be >= 1. Got {self.lmax!r}.")
        n_medium = complex(self.n_medium)
        if abs(n_medium.imag) > 0.0:
            raise ValueError(f"`n_medium` must be real for this solver path. Got {n_medium!r}.")
        if not (float(n_medium.real) > 0.0):
            raise ValueError(f"`n_medium` must be positive. Got {n_medium!r}.")
        if float(self.radial_lut_dr) <= 0.0:
            raise ValueError(f"`radial_lut_dr` must be > 0. Got {self.radial_lut_dr!r}.")
        if float(self.circumscribing_sphere_overlap_atol) < 0.0:
            raise ValueError(
                "`circumscribing_sphere_overlap_atol` must be >= 0. "
                f"Got {self.circumscribing_sphere_overlap_atol!r}."
            )
        if not isinstance(self.check_circumscribing_sphere_overlap, (bool, np.bool_)):
            raise ValueError("`check_circumscribing_sphere_overlap` must be a boolean.")
        if not isinstance(self.force_general_initial_field, (bool, np.bool_)):
            raise ValueError("`force_general_initial_field` must be a boolean.")
        if float(self.solver_rtol) <= 0.0:
            raise ValueError(f"`solver_rtol` must be > 0. Got {self.solver_rtol!r}.")
        if not isinstance(self.solver_compute_final_residual, (bool, np.bool_)):
            raise ValueError("`solver_compute_final_residual` must be a boolean.")
        if int(self.solver_restart) < 1:
            raise ValueError(f"`solver_restart` must be >= 1. Got {self.solver_restart!r}.")
        if int(self.solver_maxiter) < 1:
            raise ValueError(f"`solver_maxiter` must be >= 1. Got {self.solver_maxiter!r}.")
        if int(self.solver_direct_max_n) < 1:
            raise ValueError(
                f"`solver_direct_max_n` must be >= 1. Got {self.solver_direct_max_n!r}."
            )
        if self.solver_preconditioner is not None and not callable(self.solver_preconditioner):
            raise ValueError("`solver_preconditioner` must be callable or None.")
        if self.solver_preconditioner_kind not in {"none", "grid_block"}:
            raise ValueError(
                "`solver_preconditioner_kind` must be one of {'none', 'grid_block'}. "
                f"Got {self.solver_preconditioner_kind!r}."
            )
        if self.solver_preconditioner is not None and self.solver_preconditioner_kind != "none":
            raise ValueError(
                "Set either custom `solver_preconditioner` or built-in "
                "`solver_preconditioner_kind`, not both."
            )
        subdiv = self.solver_preconditioner_subdivisions
        if isinstance(subdiv, (int, np.integer)):
            if int(subdiv) < 1:
                raise ValueError(
                    "`solver_preconditioner_subdivisions` must be >= 1. "
                    f"Got {self.solver_preconditioner_subdivisions!r}."
                )
        elif isinstance(subdiv, (tuple, list)) and len(subdiv) == 3:
            if any(int(v) < 1 for v in subdiv):
                raise ValueError(
                    "`solver_preconditioner_subdivisions` tuple entries must be >= 1. "
                    f"Got {self.solver_preconditioner_subdivisions!r}."
                )
        else:
            raise ValueError(
                "`solver_preconditioner_subdivisions` must be an int or length-3 tuple/list. "
                f"Got {self.solver_preconditioner_subdivisions!r}."
            )
        if self.solver_preconditioner_max_block_unknowns is not None:
            if int(self.solver_preconditioner_max_block_unknowns) < 1:
                raise ValueError(
                    "`solver_preconditioner_max_block_unknowns` must be >= 1 when set. "
                    f"Got {self.solver_preconditioner_max_block_unknowns!r}."
                )
        if self.solver_warm_start is not None:
            ws = np.asarray(self.solver_warm_start)
            if ws.ndim not in (1, 2):
                raise ValueError("`solver_warm_start` must be 1D, 2D, or None.")
        resolve_compute_accum_dtypes(
            compute_dtype=self.compute_dtype,
            accum_dtype=self.accum_dtype,
        )

        method = str(self.solver_method).lower()
        allowed = {"auto", "gmres", "bicgstab", "lgmres", "gcrotmk", "direct"}
        if method not in allowed:
            raise ValueError(
                f"`solver_method` must be one of {sorted(allowed)}. Got {self.solver_method!r}."
            )

        _, az_shared = _validate_angular_grid_pair(
            polar_name="polar_angles",
            azimuthal_name="azimuthal_angles",
            polar_values=self.polar_angles,
            azimuthal_values=self.azimuthal_angles,
        )
        _warn_redundant_periodic_azimuth_endpoint(
            azimuth_name="azimuthal_angles", azimuth=az_shared
        )

        has_source_polar = self.source_polar_angles is not None
        has_source_azimuth = self.source_azimuthal_angles is not None
        if has_source_polar != has_source_azimuth:
            raise ValueError(
                "Set both `source_polar_angles` and `source_azimuthal_angles`, or set neither."
            )
        if has_source_polar:
            _, az_source = _validate_angular_grid_pair(
                polar_name="source_polar_angles",
                azimuthal_name="source_azimuthal_angles",
                polar_values=np.asarray(self.source_polar_angles),
                azimuthal_values=np.asarray(self.source_azimuthal_angles),
            )
            _warn_redundant_periodic_azimuth_endpoint(
                azimuth_name="source_azimuthal_angles", azimuth=az_source
            )

        has_farfield_polar = self.farfield_polar_angles is not None
        has_farfield_azimuth = self.farfield_azimuthal_angles is not None
        if has_farfield_polar != has_farfield_azimuth:
            raise ValueError(
                "Set both `farfield_polar_angles` and `farfield_azimuthal_angles`, or set neither."
            )
        if has_farfield_polar:
            _, az_farfield = _validate_angular_grid_pair(
                polar_name="farfield_polar_angles",
                azimuthal_name="farfield_azimuthal_angles",
                polar_values=np.asarray(self.farfield_polar_angles),
                azimuthal_values=np.asarray(self.farfield_azimuthal_angles),
            )
            _warn_redundant_periodic_azimuth_endpoint(
                azimuth_name="farfield_azimuthal_angles", azimuth=az_farfield
            )

        if self.source is not None:
            if not isinstance(self.source, Source):
                raise TypeError(
                    "`source` must satisfy the pyceles Source protocol "
                    "(wavelength/medium_n + incident_coeffs + has_finite_incident_power APIs). "
                    f"Got {type(self.source).__name__}."
                )
            source_wavelength = float(self.source.wavelength)
            source_n_medium = complex(self.source.medium_n)
            if not np.isclose(source_wavelength, float(self.wavelength), rtol=0.0, atol=0.0):
                raise ValueError(
                    "Configuration mismatch: `source.wavelength` must match `SimulationConfig.wavelength` "
                    f"({source_wavelength!r} != {self.wavelength!r})."
                )
            if not np.isclose(source_n_medium, n_medium, rtol=0.0, atol=0.0):
                raise ValueError(
                    "Configuration mismatch: `source.medium_n` must match `SimulationConfig.n_medium` "
                    f"({source_n_medium!r} != {n_medium!r})."
                )

    def source_angular_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(beta, alpha)` grid used for incident-source projection.

        By default this returns the shared `polar_angles`/`azimuthal_angles`.
        """
        polar_values = (
            self.polar_angles if self.source_polar_angles is None else self.source_polar_angles
        )
        azimuthal_values = (
            self.azimuthal_angles
            if self.source_azimuthal_angles is None
            else self.source_azimuthal_angles
        )
        return _validate_angular_grid_pair(
            polar_name="source_polar_angles",
            azimuthal_name="source_azimuthal_angles",
            polar_values=np.asarray(polar_values),
            azimuthal_values=np.asarray(azimuthal_values),
        )

    def farfield_angular_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(beta, alpha)` grid used for far-field PWPs and power fluxes.

        By default this returns the shared `polar_angles`/`azimuthal_angles`.
        """
        polar_values = (
            self.polar_angles if self.farfield_polar_angles is None else self.farfield_polar_angles
        )
        azimuthal_values = (
            self.azimuthal_angles
            if self.farfield_azimuthal_angles is None
            else self.farfield_azimuthal_angles
        )
        return _validate_angular_grid_pair(
            polar_name="farfield_polar_angles",
            azimuthal_name="farfield_azimuthal_angles",
            polar_values=np.asarray(polar_values),
            azimuthal_values=np.asarray(azimuthal_values),
        )


@dataclass(frozen=True)
class SimulationResult:
    """Container for solved multipole coefficients and derived observables.

    In `Simulation.run()`, mixed outputs (`coeffs`, `farfield`, `power`,
    `cross_sections`) correspond to the source polarization requested by the
    user. In `Simulation.postprocess_sources(...)`, each channel result
    corresponds to its own source label.

    When `solve_polarization_basis=True`, basis and unpolarized diagnostics are
    also provided (`*_basis`, `unpolarized`). In that mode the solver computes
    TE/TM basis channels only (multi-RHS solve), then reconstructs the requested
    Jones channel by linear combination; no third mixed solve is run.
    `solver_result` and `solver_result_basis` therefore carry the same TE/TM
    solve diagnostics.
    Includes both solved unknowns and derived diagnostics:
    power, cross sections, far field, basis channels, and unpolarized averages.
    When `Simulation.run(..., include_farfield=False)` or
    `Simulation.postprocess_sources(..., include_farfield=False)` is used, the
    solve is still performed and postprocessing can be skipped; far-field
    payloads are returned as empty arrays and power/cross-section diagnostics
    remain `None`.
    `polarization_jones` is populated for propagating TE/TM sources that expose
    Jones metadata. For local dipole sources it is `None`.

    Naming note:
    `k0` stores the vacuum wavenumber `2*pi/wavelength` (not angular frequency).
    This intentionally avoids CELES-style `omega` naming ambiguity ahead of
    dipole/LDOS workflows where true angular frequency may also appear.
    """

    config: SimulationConfig
    positions: np.ndarray
    radii: np.ndarray
    k: float
    k0: float
    coeffs: np.ndarray
    rhs: np.ndarray
    initial_coeffs: np.ndarray
    initial_coeffs_basis: dict[str, np.ndarray] | None
    coeffs_basis: dict[str, np.ndarray] | None
    solver_result: LinearSolveResult
    solver_result_basis: LinearSolveResult | None
    farfield: FarFieldPatterns
    farfield_basis: dict[str, FarFieldPatterns] | None
    power: dict[str, float] | None
    power_basis: dict[str, dict[str, float]] | None
    cross_sections: dict[str, float] | None
    cross_sections_basis: dict[str, dict[str, float]] | None
    unpolarized: dict[str, dict[str, float]] | None
    decomposition_forward: dict[str, float] | None
    decomposition_backward: dict[str, float] | None
    decomposition_forward_basis: dict[str, dict[str, float]] | None
    decomposition_backward_basis: dict[str, dict[str, float]] | None
    particles: tuple[Particle, ...]
    polarization_jones: tuple[complex, complex] | None = None
    compute_dtype: str = "complex128"
    accum_dtype: str = "complex128"

    @property
    def n_particles(self) -> int:
        """Return the number of scattering particles represented by this result."""
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class SolvedSourcesResult:
    """Solve-only outputs from one shared-operator multi-RHS solve.

    This payload is the canonical output of `Simulation.solve_sources(...)`.
    It intentionally contains no far-field/power/cross-section postprocessing.
    Use `Simulation.postprocess_sources(...)` when channel `SimulationResult`
    objects are needed.
    """

    labels: tuple[str, ...]
    sources: dict[str, Source]
    solver_result: LinearSolveResult
    initial_coeffs: dict[str, np.ndarray]
    rhs: dict[str, np.ndarray]
    coeffs: dict[str, np.ndarray]
    k: float
    k0: float
    compute_dtype: str
    accum_dtype: str


@dataclass(frozen=True)
class MultiSourceSimulationResult:
    """Postprocessed channel results for one solved multi-source payload.

    All channels share one `SolvedSourcesResult` solve. Per-channel
    `SimulationResult` payloads are exposed under `runs`.
    """

    labels: tuple[str, ...]
    sources: dict[str, Source]
    runs: dict[str, SimulationResult]
    solver_result: LinearSolveResult
    initial_coeffs: dict[str, np.ndarray]
    rhs: dict[str, np.ndarray]
    coeffs: dict[str, np.ndarray]

    def __getitem__(self, label: str) -> SimulationResult:
        """Return one channel run by label."""
        return self.runs[label]


def _avg_numeric_dict(d1: Mapping[str, object], d2: Mapping[str, object]) -> dict[str, float]:
    """Average overlapping scalar diagnostics from two channels."""
    keys = set(d1).intersection(set(d2))
    out: dict[str, float] = {}
    for key in keys:
        v1, v2 = d1[key], d2[key]
        if isinstance(v1, (int, float, np.floating)) and isinstance(v2, (int, float, np.floating)):
            out[key] = float(0.5 * (float(v1) + float(v2)))
    return out


def _empty_pwp(dtype: npt.DTypeLike) -> dict[str, np.ndarray]:
    """Return an empty PWP payload used when far-field postprocessing is disabled."""
    return {
        "beta": np.zeros((0,), dtype=float),
        "alpha": np.zeros((0,), dtype=float),
        "kx": np.zeros((0, 0), dtype=float),
        "ky": np.zeros((0, 0), dtype=float),
        "kz": np.zeros((0, 0), dtype=float),
        "coeff": np.zeros((0, 0), dtype=np.dtype(dtype)),
    }


def _empty_farfield_patterns(dtype: npt.DTypeLike) -> FarFieldPatterns:
    """Return an empty far-field payload for solve-only workflows."""
    return FarFieldPatterns(
        initial_te=None,
        initial_tm=None,
        scattered_te=_empty_pwp(dtype),
        scattered_tm=_empty_pwp(dtype),
        total_te=None,
        total_tm=None,
    )


def _single_rhs_result_from_multi(result: LinearSolveResult, col: int) -> LinearSolveResult:
    """Extract one RHS column from a multi-RHS `LinearSolveResult`."""
    if int(result.rhs_count) <= 1:
        return result

    x_arr = np.asarray(result.x)
    if x_arr.ndim != 2:
        raise ValueError(
            f"Expected multi-RHS solver output with 2D `x` array. Got shape {x_arr.shape}."
        )
    if col < 0 or col >= x_arr.shape[1]:
        raise IndexError(f"RHS column index {col} out of bounds for shape {x_arr.shape}.")

    def _pick_scalar(value: int | float | np.ndarray, index: int) -> int | float:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr) if arr.dtype.kind == "f" else int(arr)
        return float(arr[index]) if arr.dtype.kind == "f" else int(arr[index])

    residual_history = None
    if isinstance(result.residual_history, list):
        if col < len(result.residual_history):
            residual_history = result.residual_history[col]
    elif isinstance(result.residual_history, np.ndarray):
        residual_history = result.residual_history

    return LinearSolveResult(
        x=np.asarray(x_arr[:, col]),
        info=int(_pick_scalar(result.info, col)),
        residual_norm=float(_pick_scalar(result.residual_norm, col)),
        relative_residual=float(_pick_scalar(result.relative_residual, col)),
        iterations=int(_pick_scalar(result.iterations, col)),
        method=str(result.method),
        residual_history=residual_history,
        rhs_count=1,
    )


class Simulation:
    """High-level orchestrator for one many-particle scattering experiment."""

    particles: tuple[Particle, ...]

    @property
    def n_particles(self) -> int:
        """Return the number of particles in this simulation geometry."""
        return int(self.positions.shape[0])

    def __init__(
        self,
        config: SimulationConfig,
        *,
        particles: Sequence[Particle],
    ):
        """Bind configuration plus particle geometry for a single simulation.

        Canonical geometry input is `particles=[...]` with explicit particle
        descriptors (`Sphere`, `LayeredSphere`, ...). Source-only runs are
        represented by an explicit empty particle list (`particles=[]`).
        """
        self.config = config
        part, pos, rad = _normalize_particle_geometry(particles)

        self.positions = pos
        self.radii = rad
        self.particles = part
        # Reuse operator-side precomputations across repeated solves on the same
        # geometry/config (e.g. moving-dipole LDOS maps).
        self._prepared_operator_cache: PreparedMatvec | None = None
        self._prepared_operator_dtype: np.dtype | None = None
        self._dense_operator_cache: np.ndarray | None = None
        self._dense_operator_dtype: np.dtype | None = None
        self._dense_lu_cache: DenseLUFactorization | None = None
        self._dense_lu_dtype: np.dtype | None = None
        if bool(self.config.check_circumscribing_sphere_overlap):
            overlap = _first_overlapping_circumscribing_pair(
                self.positions,
                self.radii,
                atol=float(self.config.circumscribing_sphere_overlap_atol),
                show_progress=bool(self.config.verbose),
            )
            if overlap is not None:
                i, j, d, rsum = overlap
                raise ValueError(
                    "Invalid geometry: circumscribing spheres overlap for particle pair "
                    f"({i}, {j}) with center distance {d:.6g} and required minimum {rsum:.6g}. "
                    "The current T-matrix formulation requires disjoint circumscribing spheres. "
                    "If this is intentional for an experimental workflow, set "
                    "`check_circumscribing_sphere_overlap=False`."
                )

    def _validate_ready_to_run(
        self,
    ) -> Source:
        """Ensure excitation is defined before assembling and solving the system."""
        if self.config.source is None:
            raise ValueError(
                "Cannot run simulation: missing required `source` in SimulationConfig. "
                "Set `source` before calling `run()`."
            )
        return self.config.source

    def _validate_source_compatibility(
        self,
        source: Source,
        *,
        label: str,
    ) -> None:
        """Validate one source against this simulation's wavelength/medium settings."""
        cfg = self.config
        if not isinstance(source, Source):
            raise TypeError(
                f"Source '{label}' must satisfy the pyceles Source protocol "
                "(wavelength/medium_n + incident_coeffs + has_finite_incident_power APIs). "
                f"Got {type(source).__name__}."
            )
        wl = float(source.wavelength)
        if not np.isclose(wl, float(cfg.wavelength), rtol=0.0, atol=0.0):
            raise ValueError(f"Source '{label}' wavelength mismatch: {wl!r} != {cfg.wavelength!r}.")
        n_src = complex(source.medium_n)
        if not np.isclose(n_src, complex(cfg.n_medium), rtol=0.0, atol=0.0):
            raise ValueError(f"Source '{label}' medium_n mismatch: {n_src!r} != {cfg.n_medium!r}.")
        if isinstance(source, (DipoleSource, DipoleCollection)):
            dip_pos = np.asarray(source.dipole_positions(), dtype=float).reshape(-1, 3)
            if dip_pos.size > 0 and self.positions.shape[0] > 0:
                # Current dipole workflow models local sources in the homogeneous
                # host and translates to sphere centers. It does not implement an
                # interior-source formulation for dipoles embedded inside particles.
                deltas = dip_pos[:, None, :] - self.positions[None, :, :]
                dist = np.linalg.norm(deltas, axis=2)
                inside = dist < self.radii[None, :]
                if np.any(inside):
                    j, i = np.argwhere(inside)[0]
                    warnings.warn(
                        "Untested configuration: dipole center lies inside a particle circumscribing sphere. "
                        f"Source '{label}', dipole index {int(j)}, particle index {int(i)}. "
                        "Current pyceles dipole formulation is validated for dipoles in the homogeneous host "
                        "outside particles; interior dipole placement may produce unreliable results.",
                        UserWarning,
                        stacklevel=3,
                    )

    def _normalize_sources_argument(
        self,
        sources: Mapping[str, Source] | Sequence[Source],
        *,
        labels: Sequence[str] | None = None,
    ) -> dict[str, Source]:
        """Normalize multi-source inputs to a deterministic labeled dictionary."""
        if isinstance(sources, Mapping):
            if labels is not None:
                raise ValueError("`labels` must be omitted when `sources` is a mapping.")
            out: dict[str, Source] = {}
            for key, src in sources.items():
                lbl = str(key)
                if lbl in out:
                    raise ValueError(f"Duplicate source label '{lbl}'.")
                out[lbl] = src
        else:
            src_list = list(sources)
            if len(src_list) == 0:
                raise ValueError("`sources` must contain at least one source.")
            if labels is None:
                labels_eff = [f"source_{j}" for j in range(len(src_list))]
            else:
                labels_eff = [str(v) for v in labels]
                if len(labels_eff) != len(src_list):
                    raise ValueError(
                        "`labels` length must match number of sources. "
                        f"Got {len(labels_eff)} labels for {len(src_list)} sources."
                    )
            if len(set(labels_eff)) != len(labels_eff):
                raise ValueError("`labels` must be unique.")
            out = {labels_eff[j]: src_list[j] for j in range(len(src_list))}

        if len(out) == 0:
            raise ValueError("`sources` must contain at least one source.")
        for label, src in out.items():
            self._validate_source_compatibility(src, label=label)
        return out

    def _build_single_channel_result(
        self,
        *,
        source: Source,
        initial_coeffs: np.ndarray,
        rhs_flat: np.ndarray,
        coeffs: np.ndarray,
        solver_result: LinearSolveResult,
        k: float,
        k0: float,
        compute_dtype: np.dtype,
        accum_dtype: np.dtype,
        farfield_polar_angles: np.ndarray,
        farfield_azimuthal_angles: np.ndarray,
        include_farfield: bool,
    ) -> SimulationResult:
        """Assemble one channel `SimulationResult` from solved coefficients."""
        cfg = self.config
        positions = self.positions
        radii = self.radii
        Ns = positions.shape[0]
        Nm = n_modes(cfg.lmax)

        power = None
        cross_sections = None
        decomposition_forward = None
        decomposition_backward = None
        if include_farfield:
            ff = compute_far_field_patterns(
                positions,
                coeffs,
                k=k,
                lmax=cfg.lmax,
                polar_angles=farfield_polar_angles,
                azimuthal_angles=farfield_azimuthal_angles,
                source=source,
                dtype=compute_dtype,
                show_progress=bool(cfg.verbose),
            )
            if isinstance(source, PlaneWave):
                cross_sections = plane_wave_cross_sections(
                    source,
                    initial_coeffs,
                    coeffs,
                    k0=k0,
                    n_medium=cfg.n_medium,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                )
            elif (
                ff.initial_te is not None
                and ff.initial_tm is not None
                and source.has_finite_incident_power()
            ):
                power = finite_beam_power_fractions(
                    source,
                    ff.initial_te,
                    ff.initial_tm,
                    ff.scattered_te,
                    ff.scattered_tm,
                    k0=k0,
                    k_medium=k,
                )
                decomposition_forward = pwp_power_decomposition(
                    direction="forward",
                    initial_pwp_te=ff.initial_te,
                    initial_pwp_tm=ff.initial_tm,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                    k0=k0,
                    k_medium=k,
                    source=source,
                )
                decomposition_backward = pwp_power_decomposition(
                    direction="backward",
                    initial_pwp_te=ff.initial_te,
                    initial_pwp_tm=ff.initial_tm,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                    k0=k0,
                    k_medium=k,
                    source=source,
                )
        else:
            ff = _empty_farfield_patterns(compute_dtype)

        pol_jones = (
            source.jones_coefficients() if isinstance(source, JonesPolarizedSource) else None
        )
        config_out = cfg if cfg.source is source else replace(cfg, source=source)
        return SimulationResult(
            config=config_out,
            positions=positions,
            radii=radii,
            particles=self.particles,
            k=k,
            k0=k0,
            coeffs=np.asarray(coeffs),
            rhs=np.asarray(rhs_flat).reshape(Ns, Nm),
            initial_coeffs=np.asarray(initial_coeffs),
            initial_coeffs_basis=None,
            coeffs_basis=None,
            solver_result=solver_result,
            solver_result_basis=None,
            farfield=ff,
            farfield_basis=None,
            power=power,
            power_basis=None,
            cross_sections=cross_sections,
            cross_sections_basis=None,
            unpolarized=None,
            decomposition_forward=decomposition_forward,
            decomposition_backward=decomposition_backward,
            decomposition_forward_basis=None,
            decomposition_backward_basis=None,
            polarization_jones=pol_jones,
            compute_dtype=str(compute_dtype),
            accum_dtype=str(accum_dtype),
        )

    def _solve_sources_core(
        self,
        labeled_sources: Mapping[str, Source],
        *,
        solver_compute_final_residual: bool | None = None,
    ) -> SolvedSourcesResult:
        """Solve labeled sources with one shared operator build (solve-only)."""
        cfg = self.config
        positions = self.positions
        labels = tuple(labeled_sources.keys())
        n_channels = len(labels)
        compute_final_residual = (
            bool(cfg.solver_compute_final_residual)
            if solver_compute_final_residual is None
            else bool(solver_compute_final_residual)
        )

        compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
            compute_dtype=cfg.compute_dtype,
            accum_dtype=cfg.accum_dtype,
        )

        Ns = positions.shape[0]
        Nm = n_modes(cfg.lmax)
        unknowns = Ns * Nm
        k0 = 2.0 * np.pi / float(cfg.wavelength)
        k = k0 * float(np.real(cfg.n_medium))

        solver_name = str(cfg.solver_method).lower()
        will_use_direct = solver_name == "direct" or (
            solver_name == "auto" and unknowns <= int(cfg.solver_direct_max_n)
        )
        if cfg.verbose:
            _print_startup_logo_once()
            print(
                "System:"
                f" particles={Ns} lmax={cfg.lmax} modes_per_particle={Nm} unknowns={unknowns} channels={n_channels}"
            )
            print(f"Dtypes: compute={compute_dtype.name} accum={accum_dtype.name}")
            dense_bytes = estimate_dense_matrix_bytes(unknowns, dtype=compute_dtype)
            dense_gib = dense_bytes / 1024**3
            if will_use_direct:
                print(f"Dense direct-solver A footprint: ~{dense_gib:.2f} GiB")
            else:
                print(
                    "Equivalent dense A footprint (for reference): "
                    f"~{dense_gib:.2f} GiB | current run: matrix-free iterative"
                )

        source_polar_angles, source_azimuthal_angles = cfg.source_angular_grids()
        if cfg.verbose:
            print(
                "Source angular grid:"
                f" beta={source_polar_angles.size}, alpha={source_azimuthal_angles.size}"
            )

        initial_coeffs: dict[str, np.ndarray] = {}
        for label in labels:
            src = labeled_sources[label]
            initial_coeffs[label] = project_source_to_svwf(
                positions,
                cfg.lmax,
                src,
                polar_angles=source_polar_angles,
                azimuthal_angles=source_azimuthal_angles,
                dtype=compute_dtype,
            )
            initial_coeffs[label] = np.asarray(initial_coeffs[label], dtype=accum_dtype)

        rhs_flat: dict[str, np.ndarray] = {
            label: np.zeros((unknowns,), dtype=accum_dtype) for label in labels
        }
        prepared = None
        A_mv = None
        A_dense = None
        if unknowns > 0:
            need_prepared = (
                self._prepared_operator_cache is None
                or self._prepared_operator_dtype is None
                or self._prepared_operator_dtype != compute_dtype
            )
            if need_prepared:
                prepared = prepare_matvec(
                    lmax=cfg.lmax,
                    k=k,
                    particles=list(self.particles),
                    n_medium=cfg.n_medium,
                    radial_lut_dr=cfg.radial_lut_dr,
                    cache_translation_blocks=cfg.cache_translation_blocks,
                    operator_dtype=compute_dtype,
                )
                self._prepared_operator_cache = prepared
                self._prepared_operator_dtype = np.dtype(compute_dtype)
                self._dense_operator_cache = None
                self._dense_operator_dtype = None
                self._dense_lu_cache = None
                self._dense_lu_dtype = None
            else:
                prepared = self._prepared_operator_cache
            if prepared is None:
                raise RuntimeError("Internal error: prepared operator cache not initialized.")
            A_mv = prepared.apply_A
            for label in labels:
                rhs_flat[label] = prepared.rhs_Tb(initial_coeffs[label].reshape(Ns * Nm))
            if will_use_direct:
                need_dense = (
                    self._dense_operator_cache is None
                    or self._dense_operator_dtype is None
                    or self._dense_operator_dtype != compute_dtype
                )
                if need_dense:
                    A_dense = assemble_dense_A_numpy(
                        prepared,
                        show_progress=bool(cfg.verbose),
                        use_cache=bool(cfg.cache_translation_blocks),
                        store_blocks=False,
                    )
                    self._dense_operator_cache = A_dense
                    self._dense_operator_dtype = np.dtype(compute_dtype)
                else:
                    A_dense = self._dense_operator_cache
                need_dense_lu = (
                    self._dense_lu_cache is None
                    or self._dense_lu_dtype is None
                    or self._dense_lu_dtype != compute_dtype
                )
                if need_dense_lu:
                    if A_dense is None:
                        raise RuntimeError("Internal error: direct solve requires dense operator.")
                    # Cache LU once per (geometry, config, dtype) so repeated
                    # direct solves with changed RHS avoid O(n^3) refactorization.
                    self._dense_lu_cache = factorize_dense_matrix(A_dense, dtype=compute_dtype)
                    self._dense_lu_dtype = np.dtype(compute_dtype)
                A_lu = self._dense_lu_cache
            else:
                A_lu = None

        rhs_matrix = np.column_stack([rhs_flat[label] for label in labels])
        rhs_arg = rhs_matrix[:, 0] if n_channels == 1 else rhs_matrix

        warm_start: np.ndarray | None = None
        if cfg.solver_warm_start is not None:
            ws = np.asarray(cfg.solver_warm_start, dtype=compute_dtype)
            if ws.ndim == 1:
                if ws.size != unknowns:
                    raise ValueError(
                        f"`solver_warm_start` length must match unknown count ({unknowns}). Got {ws.size}."
                    )
                warm_start = np.repeat(ws[:, None], n_channels, axis=1) if n_channels > 1 else ws
            elif ws.ndim == 2:
                if ws.shape[0] != unknowns:
                    raise ValueError(
                        f"`solver_warm_start` first dimension must match unknown count ({unknowns}). "
                        f"Got {ws.shape}."
                    )
                if ws.shape[1] == n_channels:
                    warm_start = ws
                elif ws.shape[1] == 1 and n_channels > 1:
                    warm_start = np.repeat(ws, n_channels, axis=1)
                else:
                    raise ValueError(
                        "`solver_warm_start` 2D second dimension must be 1 or match the number of channels. "
                        f"Got {ws.shape[1]} for {n_channels} channels."
                    )
            else:
                raise ValueError("`solver_warm_start` must be 1D or 2D.")
        if n_channels == 1 and warm_start is not None and np.ndim(warm_start) == 2:
            warm_start = np.asarray(warm_start)[:, 0]

        solver_preconditioner = cfg.solver_preconditioner
        if (
            solver_preconditioner is None
            and str(cfg.solver_preconditioner_kind).lower() == "grid_block"
            and not will_use_direct
            and unknowns > 0
        ):
            if prepared is None:
                raise RuntimeError(
                    "Internal error: prepared matvec is required for grid preconditioner."
                )
            solver_preconditioner = make_grid_block_preconditioner(
                prepared,
                subdivisions=cfg.solver_preconditioner_subdivisions,
                cubic_bbox=bool(cfg.solver_preconditioner_cubic_bbox),
                max_block_unknowns=cfg.solver_preconditioner_max_block_unknowns,
                show_progress=bool(cfg.verbose),
            )
            if cfg.verbose:
                sizes = np.asarray(solver_preconditioner.block_sizes, dtype=int)
                print(
                    "Preconditioner grid_block:"
                    f" blocks={solver_preconditioner.n_blocks} "
                    f"particles/block(min,mean,max)=({sizes.min()},{sizes.mean():.1f},{sizes.max()})"
                )

        if unknowns == 0:
            solver_result = _make_empty_solver_result(
                dtype=compute_dtype, nrhs=n_channels, method=cfg.solver_method
            )
            x_matrix = np.zeros((unknowns, n_channels), dtype=compute_dtype)
        else:
            if A_mv is None:
                raise RuntimeError("Internal error: A_mv not prepared for non-empty system.")
            solver_result = solve_linear_system(
                A_mv,
                rhs_arg,
                method=cfg.solver_method,
                A_dense=A_dense,
                A_factorized=A_lu,
                x0=warm_start,
                preconditioner=solver_preconditioner,
                rtol=float(cfg.solver_rtol),
                atol=0.0,
                restart=int(cfg.solver_restart),
                maxiter=int(cfg.solver_maxiter),
                direct_max_n=int(cfg.solver_direct_max_n),
                dtype=compute_dtype,
                show_progress=bool(cfg.verbose),
                compute_final_residual=compute_final_residual,
            )
            x_arr = np.asarray(solver_result.x)
            x_matrix = (
                x_arr.reshape(unknowns, 1)
                if n_channels == 1
                else x_arr.reshape(unknowns, n_channels)
            )

        coeffs: dict[str, np.ndarray] = {}
        rhs_out: dict[str, np.ndarray] = {}
        for j, label in enumerate(labels):
            x_col = x_matrix[:, j].reshape(Ns, Nm)
            coeffs[label] = x_col
            rhs_out[label] = np.asarray(rhs_flat[label]).reshape(Ns, Nm)

        return SolvedSourcesResult(
            labels=labels,
            sources=dict(labeled_sources),
            solver_result=solver_result,
            initial_coeffs=initial_coeffs,
            rhs=rhs_out,
            coeffs=coeffs,
            k=float(k),
            k0=float(k0),
            compute_dtype=str(compute_dtype),
            accum_dtype=str(accum_dtype),
        )

    def solve_sources(
        self,
        sources: Mapping[str, Source] | Sequence[Source],
        *,
        labels: Sequence[str] | None = None,
        solver_compute_final_residual: bool | None = None,
    ) -> SolvedSourcesResult:
        """Canonical solve-only API for one labeled source set.

        Parameters
        ----------
        sources:
            Either a mapping `{label: source}` (recommended) or a sequence of
            sources. Sequence inputs can be labeled via `labels`; otherwise
            `source_0`, `source_1`, ... are used.
        labels:
            Optional labels for sequence inputs.
        solver_compute_final_residual:
            Optional override for final residual diagnostics at this call.
            `None` uses `SimulationConfig.solver_compute_final_residual`.

        Returns
        -------
        SolvedSourcesResult
            Solve-only multipole payload: coefficients, RHS, initial
            coefficients, and multi-RHS solver diagnostics.
        """
        labeled = self._normalize_sources_argument(sources, labels=labels)
        return self._solve_sources_core(
            labeled,
            solver_compute_final_residual=solver_compute_final_residual,
        )

    def postprocess_sources(
        self,
        solved: SolvedSourcesResult,
        *,
        include_farfield: bool = True,
        farfield_polar_angles: np.ndarray | None = None,
        farfield_azimuthal_angles: np.ndarray | None = None,
    ) -> MultiSourceSimulationResult:
        """Postprocess solved channels into per-channel `SimulationResult` payloads.

        Parameters
        ----------
        solved:
            Solve-only payload returned by `solve_sources(...)`.
        include_farfield:
            If `False`, skip far-field/power/cross-section diagnostics.
        farfield_polar_angles, farfield_azimuthal_angles:
            Optional override grids for far-field postprocessing. Set both or neither.
            If omitted, `SimulationConfig.farfield_angular_grids()` is used.
        """
        cfg = self.config
        labels = tuple(solved.labels)
        n_channels = len(labels)

        if (farfield_polar_angles is None) != (farfield_azimuthal_angles is None):
            raise ValueError(
                "Set both `farfield_polar_angles` and `farfield_azimuthal_angles`, or set neither."
            )
        if farfield_polar_angles is None:
            ff_polar, ff_azimuth = cfg.farfield_angular_grids()
        else:
            ff_polar, ff_azimuth = _validate_angular_grid_pair(
                polar_name="farfield_polar_angles",
                azimuthal_name="farfield_azimuthal_angles",
                polar_values=np.asarray(farfield_polar_angles),
                azimuthal_values=np.asarray(farfield_azimuthal_angles),
            )

        Ns = self.positions.shape[0]
        Nm = n_modes(cfg.lmax)
        compute_dtype = np.dtype(solved.compute_dtype)
        accum_dtype = np.dtype(solved.accum_dtype)

        runs: dict[str, SimulationResult] = {}
        for j, label in enumerate(labels):
            if label not in solved.sources:
                raise KeyError(f"Missing source payload for label '{label}'.")
            if label not in solved.initial_coeffs:
                raise KeyError(f"Missing initial coefficients for label '{label}'.")
            if label not in solved.rhs:
                raise KeyError(f"Missing RHS payload for label '{label}'.")
            if label not in solved.coeffs:
                raise KeyError(f"Missing solved coefficients for label '{label}'.")

            x_col = np.asarray(solved.coeffs[label], dtype=accum_dtype)
            if x_col.shape != (Ns, Nm):
                raise ValueError(
                    f"Solved coefficients for label '{label}' must have shape {(Ns, Nm)}. "
                    f"Got {x_col.shape}."
                )
            rhs_col = np.asarray(solved.rhs[label], dtype=accum_dtype)
            if rhs_col.shape != (Ns, Nm):
                raise ValueError(
                    f"Solved RHS for label '{label}' must have shape {(Ns, Nm)}. Got {rhs_col.shape}."
                )
            solver_col = (
                solved.solver_result
                if n_channels == 1
                else _single_rhs_result_from_multi(solved.solver_result, j)
            )
            runs[label] = self._build_single_channel_result(
                source=solved.sources[label],
                initial_coeffs=np.asarray(solved.initial_coeffs[label], dtype=accum_dtype),
                rhs_flat=rhs_col.reshape(Ns * Nm),
                coeffs=x_col,
                solver_result=solver_col,
                k=float(solved.k),
                k0=float(solved.k0),
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
                farfield_polar_angles=ff_polar,
                farfield_azimuthal_angles=ff_azimuth,
                include_farfield=include_farfield,
            )

        return MultiSourceSimulationResult(
            labels=labels,
            sources=dict(solved.sources),
            runs=runs,
            solver_result=solved.solver_result,
            initial_coeffs=dict(solved.initial_coeffs),
            rhs=dict(solved.rhs),
            coeffs=dict(solved.coeffs),
        )

    def run(self, *, include_farfield: bool = True) -> SimulationResult:
        """Run one simulation for `config.source`.

        If `solve_polarization_basis=False`, this is a single-channel solve.
        If `solve_polarization_basis=True`, TE/TM channels are solved through
        `solve_sources(...)` and then combined into the requested Jones
        channel, while basis and unpolarized diagnostics are retained. This
        convenience mode applies to propagating TE/TM sources that expose Jones
        metadata and `with_polarization('TE'/'TM')`.

        Parameters
        ----------
        include_farfield:
            If `False`, skip far-field/power/cross-section postprocessing.
        """
        cfg = self.config
        source = self._validate_ready_to_run()

        if not bool(cfg.solve_polarization_basis):
            solved = self.solve_sources({"mixed": source})
            multi = self.postprocess_sources(solved, include_farfield=include_farfield)
            return multi["mixed"]
        if not isinstance(source, JonesPolarizedSource):
            raise ValueError(
                "`solve_polarization_basis=True` is only defined for TE/TM polarization sources "
                "that expose Jones metadata and `with_polarization('TE'/'TM')`."
            )
        a_te, a_tm = source.jones_coefficients()
        src_te = source.with_polarization("TE")
        src_tm = source.with_polarization("TM")

        basis_sources = {
            "te": src_te,
            "tm": src_tm,
        }
        basis_solved = self.solve_sources(basis_sources)
        basis_multi = self.postprocess_sources(basis_solved, include_farfield=include_farfield)
        run_te = basis_multi["te"]
        run_tm = basis_multi["tm"]
        compute_dtype = np.dtype(run_te.compute_dtype)
        accum_dtype = np.dtype(run_te.accum_dtype)

        b_te = np.asarray(basis_solved.initial_coeffs["te"])
        b_tm = np.asarray(basis_solved.initial_coeffs["tm"])
        rhs_te = np.asarray(basis_solved.rhs["te"])
        rhs_tm = np.asarray(basis_solved.rhs["tm"])
        x_te = np.asarray(basis_solved.coeffs["te"])
        x_tm = np.asarray(basis_solved.coeffs["tm"])

        b = np.asarray(a_te * b_te + a_tm * b_tm, dtype=accum_dtype)
        rhs = np.asarray(a_te * rhs_te + a_tm * rhs_tm, dtype=accum_dtype)
        x = np.asarray(a_te * x_te + a_tm * x_tm, dtype=accum_dtype)

        ff_basis = {"te": run_te.farfield, "tm": run_tm.farfield}
        if include_farfield:
            ff = _mix_farfield_patterns(
                ff_basis["te"],
                ff_basis["tm"],
                a_te=a_te,
                a_tm=a_tm,
                dtype=compute_dtype,
            )
        else:
            ff = _empty_farfield_patterns(compute_dtype)

        power = None
        cross_sections = None
        decomposition_forward = None
        decomposition_backward = None
        if include_farfield:
            if isinstance(source, PlaneWave):
                cross_sections = plane_wave_cross_sections(
                    source,
                    b,
                    x,
                    k0=run_te.k0,
                    n_medium=cfg.n_medium,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                )
            elif (
                ff.initial_te is not None
                and ff.initial_tm is not None
                and source.has_finite_incident_power()
            ):
                power = finite_beam_power_fractions(
                    source,
                    ff.initial_te,
                    ff.initial_tm,
                    ff.scattered_te,
                    ff.scattered_tm,
                    k0=run_te.k0,
                    k_medium=run_te.k,
                )
                decomposition_forward = pwp_power_decomposition(
                    direction="forward",
                    initial_pwp_te=ff.initial_te,
                    initial_pwp_tm=ff.initial_tm,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                    k0=run_te.k0,
                    k_medium=run_te.k,
                    source=source,
                )
                decomposition_backward = pwp_power_decomposition(
                    direction="backward",
                    initial_pwp_te=ff.initial_te,
                    initial_pwp_tm=ff.initial_tm,
                    scattered_pwp_te=ff.scattered_te,
                    scattered_pwp_tm=ff.scattered_tm,
                    k0=run_te.k0,
                    k_medium=run_te.k,
                    source=source,
                )

        power_basis = None
        cross_sections_basis = None
        decomposition_forward_basis = None
        decomposition_backward_basis = None
        unpolarized = None

        if run_te.power is not None and run_tm.power is not None:
            power_basis = {"te": run_te.power, "tm": run_tm.power}
            unpolarized = dict(unpolarized or {})
            unpolarized["power"] = _avg_numeric_dict(run_te.power, run_tm.power)
        if run_te.cross_sections is not None and run_tm.cross_sections is not None:
            cross_sections_basis = {"te": run_te.cross_sections, "tm": run_tm.cross_sections}
            unpolarized = dict(unpolarized or {})
            unpolarized["cross_sections"] = _avg_numeric_dict(
                run_te.cross_sections, run_tm.cross_sections
            )
        if run_te.decomposition_forward is not None and run_tm.decomposition_forward is not None:
            decomposition_forward_basis = {
                "te": run_te.decomposition_forward,
                "tm": run_tm.decomposition_forward,
            }
        if run_te.decomposition_backward is not None and run_tm.decomposition_backward is not None:
            decomposition_backward_basis = {
                "te": run_te.decomposition_backward,
                "tm": run_tm.decomposition_backward,
            }

        return SimulationResult(
            config=cfg,
            positions=self.positions,
            radii=self.radii,
            particles=self.particles,
            k=run_te.k,
            k0=run_te.k0,
            coeffs=x,
            rhs=rhs,
            initial_coeffs=b,
            initial_coeffs_basis={"te": b_te, "tm": b_tm},
            coeffs_basis={"te": x_te, "tm": x_tm},
            solver_result=basis_solved.solver_result,
            solver_result_basis=basis_solved.solver_result,
            farfield=ff,
            farfield_basis=ff_basis,
            power=power,
            power_basis=power_basis,
            cross_sections=cross_sections,
            cross_sections_basis=cross_sections_basis,
            unpolarized=unpolarized,
            decomposition_forward=decomposition_forward,
            decomposition_backward=decomposition_backward,
            decomposition_forward_basis=decomposition_forward_basis,
            decomposition_backward_basis=decomposition_backward_basis,
            polarization_jones=(a_te, a_tm),
            compute_dtype=str(compute_dtype),
            accum_dtype=str(accum_dtype),
        )
