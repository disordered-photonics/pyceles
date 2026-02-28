from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._logo import print_logo
from pyceles._version import __version__
from pyceles.core.indexing import n_modes
from pyceles.core.matvec import assemble_dense_A_numpy, prepare_matvec
from pyceles.core.projection import project_source_basis_to_svwf
from pyceles.core.sources import GaussianBeam, PlaneWave, source_jones
from pyceles.linear.preconditioner import make_grid_block_preconditioner
from pyceles.linear.solvers import (
    LinearSolveResult,
    estimate_dense_matrix_bytes,
    solve_linear_system,
)
from pyceles.postprocessing.farfield import (
    FarFieldPatterns,
    compute_far_field_patterns,
    finite_beam_power_fractions,
    plane_wave_cross_sections,
    pwp_power_decomposition,
)

_SOURCE_TYPES = (GaussianBeam, PlaneWave)
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


def _normalize_geometry(
    positions: np.ndarray,
    radii: np.ndarray,
    n_particle: np.ndarray | complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize particle geometry inputs for a physically valid sphere ensemble.

    This enforces the minimum constraints needed by the multiple-scattering
    solver: each sphere has one center, one positive radius, and one finite
    refractive index with positive real part.
    """
    if positions is None:
        raise ValueError(
            "`positions` cannot be None. "
            "Use an explicit array with shape (N, 3); for no scatterers use np.zeros((0, 3))."
        )
    if radii is None:
        raise ValueError(
            "`radii` cannot be None. "
            "Use an explicit array with shape (N,); for no scatterers use np.zeros((0,))."
        )
    if n_particle is None:
        raise ValueError(
            "`n_particle` cannot be None. "
            "Use an explicit scalar or array with shape (N,); for no scatterers use np.zeros((0,), complex)."
        )

    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (N, 3). Got {pos.shape}.")
    if not np.all(np.isfinite(pos)):
        raise ValueError("`positions` must contain only finite values.")

    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != pos.shape[0]:
        raise ValueError(
            f"`radii` length ({rad.shape[0]}) must match number of positions ({pos.shape[0]})."
        )
    if np.any(~np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("`radii` must be finite and strictly positive.")

    n_part_arr = np.asarray(n_particle, dtype=np.complex128)
    if n_part_arr.ndim == 0:
        n_part = np.full((pos.shape[0],), complex(n_part_arr), dtype=np.complex128)
    else:
        n_part = n_part_arr.reshape(-1)
        if n_part.shape[0] != pos.shape[0]:
            raise ValueError(
                f"`n_particle` length ({n_part.shape[0]}) must match number of positions ({pos.shape[0]})."
            )
    if not np.all(np.isfinite(n_part.real)) or not np.all(np.isfinite(n_part.imag)):
        raise ValueError("`n_particle` must contain only finite values.")
    if np.any(n_part.real <= 0.0):
        raise ValueError("Real part of `n_particle` must be strictly positive.")

    return pos, rad, n_part.astype(np.complex128, copy=False)


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

    Polarization/basis options:
    - `source.polarization` may be `"TE"`, `"TM"`, or Jones `(a_te, a_tm)`.
    - `solve_polarization_basis=True` solves TE/TM basis channels and stores both.
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
    source: GaussianBeam | PlaneWave | None = None
    # Shared CELES-like default angular grid. If no stage-specific grids are
    # provided, this pair is used for both source projection and far-field bins.
    # Use periodic azimuth sampling on [0, 2*pi) (endpoint=False) so periodic
    # fast-path detection remains active by default.
    polar_angles: np.ndarray = field(default_factory=lambda: np.linspace(0.0, np.pi, 5001))
    azimuthal_angles: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 2 * np.pi, 201, endpoint=False)
    )
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
            if not isinstance(self.source, _SOURCE_TYPES):
                raise TypeError(
                    "`source` must be a PlaneWave or GaussianBeam. "
                    f"Got {type(self.source).__name__}."
                )
            source_wavelength = float(self.source.wavelength)
            source_n_medium = complex(self.source.medium_n)
            if not np.isclose(source_wavelength, float(self.wavelength), rtol=0.0, atol=0.0):
                raise ValueError(
                    "Configuration mismatch: `source.wavelength` must match `SimulationConfig.wavelength` "
                    f"({source_wavelength!r} != {self.wavelength!r})."
                )
            if not np.isclose(source_n_medium.real, n_medium.real, rtol=0.0, atol=0.0):
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

    Mixed outputs (`coeffs`, `farfield`, `power`, `cross_sections`) correspond to
    the source polarization requested by the user.

    When `solve_polarization_basis=True`, basis and unpolarized diagnostics are
    also provided (`*_basis`, `unpolarized`). In that mode the solver computes
    TE/TM basis channels only (multi-RHS solve), then reconstructs the requested
    Jones channel by linear combination; no third mixed solve is run.
    `solver_result` and `solver_result_basis` therefore carry the same TE/TM
    solve diagnostics.
    Includes both solved unknowns and derived diagnostics:
    power, cross sections, far field, basis channels, and unpolarized averages.
    """

    config: SimulationConfig
    positions: np.ndarray
    radii: np.ndarray
    n_particle: np.ndarray
    k: float
    omega: float
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
    polarization_jones: tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)
    compute_dtype: str = "complex128"
    accum_dtype: str = "complex128"


class Simulation:
    """High-level orchestrator for one many-sphere scattering experiment."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        positions: np.ndarray,
        radii: np.ndarray,
        n_particle: np.ndarray | complex,
    ):
        """Bind configuration plus particle geometry for a single simulation.

        No-scatterer (source-only) runs are supported by passing explicit empty
        arrays:
        `positions.shape==(0,3)`, `radii.shape==(0,)`, `n_particle.shape==(0,)`.
        `None` inputs are rejected to avoid accidental empty runs.
        """
        self.config = config
        self.positions, self.radii, self.n_particle = _normalize_geometry(
            positions, radii, n_particle
        )
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

    def _validate_ready_to_run(self) -> GaussianBeam | PlaneWave:
        """Ensure excitation is defined before assembling and solving the system."""
        if self.config.source is None:
            raise ValueError(
                "Cannot run simulation: missing required `source` in SimulationConfig. "
                "Set `source` before calling `run()`."
            )
        return self.config.source

    def run(self) -> SimulationResult:
        """Solve for multipole coefficients and evaluate far-field diagnostics.

        Workflow:
        1. project source to incident SVWF coefficients,
        2. build/apply the operator for `(I - T W) x = T b`,
        3. solve the linear system (single RHS, or TE/TM basis multi-RHS only),
        4. evaluate far-field/power/cross-section diagnostics,
        5. assemble mixed, basis, and optional unpolarized outputs.

        For `solve_polarization_basis=True`, the requested Jones channel is
        formed as `x = a_te*x_te + a_tm*x_tm` and far-field PWPs are formed by
        coherent TE/TM basis recombination, avoiding an extra mixed solve and a
        third full far-field evaluation.
        """
        cfg = self.config
        source = self._validate_ready_to_run()
        positions = self.positions
        radii = self.radii
        n_particle = self.n_particle

        compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
            compute_dtype=cfg.compute_dtype,
            accum_dtype=cfg.accum_dtype,
        )

        Ns = positions.shape[0]
        Nm = n_modes(cfg.lmax)
        unknowns = Ns * Nm

        k = 2.0 * np.pi / float(cfg.wavelength) * float(np.real(cfg.n_medium))
        omega = 2.0 * np.pi / float(cfg.wavelength)

        solver_name = str(cfg.solver_method).lower()
        will_use_direct = solver_name == "direct" or (
            solver_name == "auto" and unknowns <= int(cfg.solver_direct_max_n)
        )
        if cfg.verbose:
            _print_startup_logo_once()
            print(
                "System:"
                f" particles={Ns} lmax={cfg.lmax} modes_per_particle={Nm} unknowns={unknowns}"
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
        farfield_polar_angles, farfield_azimuthal_angles = cfg.farfield_angular_grids()
        if cfg.verbose:
            same_beta = np.array_equal(source_polar_angles, farfield_polar_angles)
            same_alpha = np.array_equal(source_azimuthal_angles, farfield_azimuthal_angles)
            if same_beta and same_alpha:
                print(
                    "Angular grid (shared CELES-style): "
                    f"beta={source_polar_angles.size}, alpha={source_azimuthal_angles.size}"
                )
            else:
                print(
                    "Angular grids (split API): "
                    f"source beta/alpha={source_polar_angles.size}/{source_azimuthal_angles.size}, "
                    f"farfield beta/alpha={farfield_polar_angles.size}/{farfield_azimuthal_angles.size}"
                )
        a_te, a_tm = source_jones(source)

        b_basis = {
            key: val.astype(accum_dtype, copy=False)
            for key, val in project_source_basis_to_svwf(
                positions,
                cfg.lmax,
                source,
                polar_angles=source_polar_angles,
                azimuthal_angles=source_azimuthal_angles,
                dtype=compute_dtype,
            ).items()
        }
        b = np.asarray(a_te * b_basis["te"] + a_tm * b_basis["tm"], dtype=accum_dtype)

        rhs = np.zeros((unknowns,), dtype=accum_dtype)
        rhs_basis = {
            "te": np.zeros((unknowns,), dtype=accum_dtype),
            "tm": np.zeros((unknowns,), dtype=accum_dtype),
        }
        prepared = None
        A_mv = None
        A_dense = None
        if unknowns > 0:
            prepared = prepare_matvec(
                lmax=cfg.lmax,
                k=k,
                positions=positions,
                radii=radii,
                n_particle=n_particle,
                n_medium=cfg.n_medium,
                radial_lut_dr=cfg.radial_lut_dr,
                cache_translation_blocks=cfg.cache_translation_blocks,
                operator_dtype=compute_dtype,
            )
            A_mv = prepared.apply_A
            rhs = prepared.rhs_Tb(b.reshape(Ns * Nm))
            rhs_basis = {
                "te": prepared.rhs_Tb(b_basis["te"].reshape(Ns * Nm)),
                "tm": prepared.rhs_Tb(b_basis["tm"].reshape(Ns * Nm)),
            }
            if will_use_direct:
                A_dense = assemble_dense_A_numpy(
                    prepared,
                    show_progress=bool(cfg.verbose),
                    use_cache=bool(cfg.cache_translation_blocks),
                    store_blocks=False,
                )

        warm_start = None
        if cfg.solver_warm_start is not None:
            ws = np.asarray(cfg.solver_warm_start, dtype=compute_dtype)
            if ws.ndim == 1:
                if ws.size != unknowns:
                    raise ValueError(
                        f"`solver_warm_start` length must match unknown count ({unknowns}). Got {ws.size}."
                    )
                warm_start = ws
            elif ws.ndim == 2:
                if ws.shape[0] != unknowns:
                    raise ValueError(
                        f"`solver_warm_start` first dimension must match unknown count ({unknowns}). "
                        f"Got {ws.shape}."
                    )
                warm_start = ws
            else:
                raise ValueError("`solver_warm_start` must be 1D or 2D.")

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

        solver_basis = None
        coeffs_basis = None
        if unknowns == 0:
            if bool(cfg.solve_polarization_basis):
                solver_basis = _make_empty_solver_result(
                    dtype=compute_dtype, nrhs=2, method=cfg.solver_method
                )
                coeffs_basis = {
                    "te": np.zeros((Ns, Nm), dtype=compute_dtype),
                    "tm": np.zeros((Ns, Nm), dtype=compute_dtype),
                }
                x = np.asarray(
                    a_te * coeffs_basis["te"] + a_tm * coeffs_basis["tm"],
                    dtype=accum_dtype,
                )
                solver_result = solver_basis
            else:
                solver_result = _make_empty_solver_result(
                    dtype=compute_dtype, nrhs=1, method=cfg.solver_method
                )
                x = np.zeros((Ns, Nm), dtype=accum_dtype)
        elif bool(cfg.solve_polarization_basis):
            if A_mv is None:
                raise RuntimeError("Internal error: A_mv not prepared for non-empty system.")
            # Basis mode solves only pure TE/TM RHS channels. The requested Jones
            # mixed solution is reconstructed linearly from these basis solutions.
            rhs_mat = np.column_stack([rhs_basis["te"], rhs_basis["tm"]])
            if warm_start is not None and np.ndim(warm_start) == 1:
                warm_start = np.column_stack([warm_start, warm_start])
            solver_basis = solve_linear_system(
                A_mv,
                rhs_mat,
                method=cfg.solver_method,
                A_dense=A_dense,
                x0=warm_start,
                preconditioner=solver_preconditioner,
                rtol=float(cfg.solver_rtol),
                atol=0.0,
                restart=int(cfg.solver_restart),
                maxiter=int(cfg.solver_maxiter),
                direct_max_n=int(cfg.solver_direct_max_n),
                dtype=compute_dtype,
                show_progress=bool(cfg.verbose),
            )
            x_mat = np.asarray(solver_basis.x).reshape(unknowns, 2)
            x_te = x_mat[:, 0].reshape(Ns, Nm)
            x_tm = x_mat[:, 1].reshape(Ns, Nm)
            coeffs_basis = {"te": x_te, "tm": x_tm}
            x = np.asarray(a_te * x_te + a_tm * x_tm, dtype=accum_dtype)
            solver_result = solver_basis
        else:
            if A_mv is None:
                raise RuntimeError("Internal error: A_mv not prepared for non-empty system.")
            solver_result = solve_linear_system(
                A_mv,
                rhs,
                method=cfg.solver_method,
                A_dense=A_dense,
                x0=warm_start,
                preconditioner=solver_preconditioner,
                rtol=float(cfg.solver_rtol),
                atol=0.0,
                restart=int(cfg.solver_restart),
                maxiter=int(cfg.solver_maxiter),
                direct_max_n=int(cfg.solver_direct_max_n),
                dtype=compute_dtype,
                show_progress=bool(cfg.verbose),
            )
            x = np.asarray(solver_result.x).reshape(Ns, Nm)

        power = None
        power_basis = None
        cross_sections = None
        cross_sections_basis = None
        unpolarized = None
        diag_fwd = None
        diag_bwd = None
        diag_fwd_basis = None
        diag_bwd_basis = None
        ff_basis = None

        def _avg_numeric_dict(d1: dict, d2: dict) -> dict:
            """Average TE/TM scalar diagnostics for incoherent unpolarized reports.

            Unpolarized observables combine intensity-like quantities, not complex
            fields. This helper is used for scalar summaries such as total powers
            and cross sections when both pure basis channels are available.
            """
            keys = set(d1).intersection(set(d2))
            out = {}
            for k_ in keys:
                v1, v2 = d1[k_], d2[k_]
                if isinstance(v1, (int, float, np.floating)) and isinstance(
                    v2, (int, float, np.floating)
                ):
                    out[k_] = float(0.5 * (float(v1) + float(v2)))
            return out

        if coeffs_basis is not None:
            ff_basis = {}
            power_basis = {}
            cross_sections_basis = {}
            diag_fwd_basis = {}
            diag_bwd_basis = {}
            pol_pairs: tuple[tuple[str, Literal["TE", "TM"]], ...] = (
                ("te", "TE"),
                ("tm", "TM"),
            )
            for pol_key, pol_label in pol_pairs:
                src_pol = source.with_polarization(pol_label)
                ff_pol = compute_far_field_patterns(
                    positions,
                    coeffs_basis[pol_key],
                    k=k,
                    lmax=cfg.lmax,
                    polar_angles=farfield_polar_angles,
                    azimuthal_angles=farfield_azimuthal_angles,
                    source=src_pol,
                    dtype=compute_dtype,
                    show_progress=bool(cfg.verbose),
                )
                ff_basis[pol_key] = ff_pol
                if ff_pol.initial_te is not None and ff_pol.initial_tm is not None:
                    power_basis[pol_key] = finite_beam_power_fractions(
                        src_pol,
                        ff_pol.initial_te,
                        ff_pol.initial_tm,
                        ff_pol.scattered_te,
                        ff_pol.scattered_tm,
                        omega=omega,
                        k_medium=k,
                    )
                    diag_fwd_basis[pol_key] = pwp_power_decomposition(
                        direction="forward",
                        initial_pwp_te=ff_pol.initial_te,
                        initial_pwp_tm=ff_pol.initial_tm,
                        scattered_pwp_te=ff_pol.scattered_te,
                        scattered_pwp_tm=ff_pol.scattered_tm,
                        omega=omega,
                        k_medium=k,
                        source=src_pol,
                    )
                    diag_bwd_basis[pol_key] = pwp_power_decomposition(
                        direction="backward",
                        initial_pwp_te=ff_pol.initial_te,
                        initial_pwp_tm=ff_pol.initial_tm,
                        scattered_pwp_te=ff_pol.scattered_te,
                        scattered_pwp_tm=ff_pol.scattered_tm,
                        omega=omega,
                        k_medium=k,
                        source=src_pol,
                    )
                elif isinstance(src_pol, PlaneWave):
                    cross_sections_basis[pol_key] = plane_wave_cross_sections(
                        src_pol,
                        b_basis[pol_key],
                        coeffs_basis[pol_key],
                        omega=omega,
                        n_medium=cfg.n_medium,
                        scattered_pwp_te=ff_pol.scattered_te,
                        scattered_pwp_tm=ff_pol.scattered_tm,
                    )
            # Reuse TE/TM basis PWPs to build the requested Jones channel and
            # avoid a third full far-field evaluation pass.
            ff = _mix_farfield_patterns(
                ff_basis["te"],
                ff_basis["tm"],
                a_te=a_te,
                a_tm=a_tm,
                dtype=compute_dtype,
            )
            if "te" in cross_sections_basis and "tm" in cross_sections_basis:
                unpolarized = {
                    "cross_sections": _avg_numeric_dict(
                        cross_sections_basis["te"], cross_sections_basis["tm"]
                    )
                }
            if "te" in power_basis and "tm" in power_basis:
                unpolarized = dict(unpolarized or {})
                unpolarized["power"] = _avg_numeric_dict(power_basis["te"], power_basis["tm"])
        else:
            ff = compute_far_field_patterns(
                positions,
                x,
                k=k,
                lmax=cfg.lmax,
                polar_angles=farfield_polar_angles,
                azimuthal_angles=farfield_azimuthal_angles,
                source=source,
                dtype=compute_dtype,
                show_progress=bool(cfg.verbose),
            )
        if ff.initial_te is not None and ff.initial_tm is not None:
            # Power diagnostics are quadratic/intensity-like, so they are
            # evaluated from the coherently mixed Jones channel PWPs.
            power = finite_beam_power_fractions(
                source,
                ff.initial_te,
                ff.initial_tm,
                ff.scattered_te,
                ff.scattered_tm,
                omega=omega,
                k_medium=k,
            )
            diag_fwd = pwp_power_decomposition(
                direction="forward",
                initial_pwp_te=ff.initial_te,
                initial_pwp_tm=ff.initial_tm,
                scattered_pwp_te=ff.scattered_te,
                scattered_pwp_tm=ff.scattered_tm,
                omega=omega,
                k_medium=k,
                source=source,
            )
            diag_bwd = pwp_power_decomposition(
                direction="backward",
                initial_pwp_te=ff.initial_te,
                initial_pwp_tm=ff.initial_tm,
                scattered_pwp_te=ff.scattered_te,
                scattered_pwp_tm=ff.scattered_tm,
                omega=omega,
                k_medium=k,
                source=source,
            )
        elif isinstance(source, PlaneWave):
            cross_sections = plane_wave_cross_sections(
                source,
                b,
                x,
                omega=omega,
                n_medium=cfg.n_medium,
                scattered_pwp_te=ff.scattered_te,
                scattered_pwp_tm=ff.scattered_tm,
            )

        return SimulationResult(
            config=cfg,
            positions=positions,
            radii=radii,
            n_particle=n_particle,
            k=k,
            omega=omega,
            coeffs=x,
            rhs=rhs.reshape(Ns, Nm),
            initial_coeffs=b,
            initial_coeffs_basis=b_basis if coeffs_basis is not None else None,
            coeffs_basis=coeffs_basis,
            solver_result=solver_result,
            solver_result_basis=solver_basis,
            farfield=ff,
            farfield_basis=ff_basis,
            power=power,
            power_basis=power_basis if power_basis else None,
            cross_sections=cross_sections,
            cross_sections_basis=cross_sections_basis if cross_sections_basis else None,
            unpolarized=unpolarized,
            decomposition_forward=diag_fwd,
            decomposition_backward=diag_bwd,
            decomposition_forward_basis=diag_fwd_basis if diag_fwd_basis else None,
            decomposition_backward_basis=diag_bwd_basis if diag_bwd_basis else None,
            polarization_jones=(a_te, a_tm),
            compute_dtype=str(compute_dtype),
            accum_dtype=str(accum_dtype),
        )
