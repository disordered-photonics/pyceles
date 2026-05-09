"""Simulation configuration model and angular-grid validation helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles.core.angular import uniform_periodic_azimuth_grid, uniform_polar_grid
from pyceles.core.operators.mlfmm import MLFMMOptions
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.sources import PlaneWave, Source


def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    """Validate one monotone angular node vector used in quadratures or PWPs.

    These arrays represent sampled polar or azimuthal directions. Requiring a
    finite, strictly increasing 1D grid avoids ambiguous trapezoidal weights
    and prevents duplicate angular directions from silently polluting source
    projection or far-field integrations.
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


def validate_angular_grid_pair(
    *,
    polar_name: str,
    azimuthal_name: str,
    polar_values: np.ndarray,
    azimuthal_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one `(beta, alpha)` angular-grid pair used in quadratures or PWPs."""
    polar = _as_1d_float_array(polar_name, polar_values)
    azimuth = _as_1d_float_array(azimuthal_name, azimuthal_values)
    if polar[0] < -1e-12 or polar[-1] > np.pi + 1e-12:
        raise ValueError(f"`{polar_name}` must lie within [0, pi].")
    if azimuth[0] < -1e-12 or azimuth[-1] > 2.0 * np.pi + 1e-12:
        raise ValueError(f"`{azimuthal_name}` must lie within [0, 2*pi].")
    return polar, azimuth


def warn_redundant_periodic_azimuth_endpoint(*, azimuth_name: str, azimuth: np.ndarray) -> None:
    """Warn when a periodic azimuth grid duplicates both 0 and 2*pi endpoints."""
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


@dataclass(frozen=True)
class SimulationConfig:
    """High-level configuration for one homogeneous-medium many-particle run.

    The config gathers physical inputs (wavelength, embedding medium, source,
    truncation) together with numerical policy (angular grids, solver strategy,
    precision, caching, preconditioning) so a run remains explicit and
    reproducible.

    Grid policy:
    - `polar_angles` / `azimuthal_angles` are the shared CELES-like defaults
      used by source projection and far-field sampling unless stage-specific
      overrides are provided.
    - The defaults intentionally use an odd-count polar grid and a periodic
      endpoint-excluded azimuth grid so their different quadrature roles are
      visible from the counts alone.

    Backend policy:
    - `operator_backend` controls the many-body solve backend.
    - `postprocessing_backend` defaults to `"inherit"`, which reuses the
      chosen operator backend so a CuPy solve naturally prefers CuPy
      postprocessing where accelerated kernels exist.
    - Stage-specific postprocessing kernels may still fall back to the NumPy
      reference implementation when no accelerated path has shipped yet.
    - For `coupling_backend="mlfmm"`, `compute_dtype="complex64"` applies to
      exact-near interactions while sampled-far MLFMM operators stay on
      `complex128`.

    Geometry / physics policy:
    - the current homogeneous-medium solver path assumes real `n_medium`
    - circumscribing-sphere overlap checks remain enabled by default because
      the multiple-scattering T-matrix formulation assumes disjoint
      circumscribing spheres

    Near-field limitation:
    - spheroid postprocessing still uses the spherical outgoing SVWF expansion
      outside the particle, so points inside the circumscribing shell but
      outside the physical spheroid remain unreliable.
    """

    wavelength: float = 550.0
    n_medium: complex = 1.0 + 0j
    lmax: int = 3
    source: Source | None = None
    polar_angles: np.ndarray = field(default_factory=lambda: uniform_polar_grid(1801))
    azimuthal_angles: np.ndarray = field(default_factory=lambda: uniform_periodic_azimuth_grid(360))
    source_polar_angles: np.ndarray | None = None
    source_azimuthal_angles: np.ndarray | None = None
    farfield_polar_angles: np.ndarray | None = None
    farfield_azimuthal_angles: np.ndarray | None = None
    radial_lut_dr: float = 0.0
    force_general_initial_field: bool = False
    solver_method: Literal["auto", "gmres", "fgmres", "bicgstab", "lgmres", "gcrotmk", "direct"] = (
        "direct"
    )
    solver_direct_max_n: int = 15_000
    solver_rtol: float = 1e-5
    solver_compute_final_residual: bool = True
    solver_restart: int = 100
    solver_maxiter: int = 1000
    solver_warm_start: np.ndarray | None = None
    solver_preconditioner: Callable[[np.ndarray], np.ndarray] | None = None
    operator_backend: Literal["numpy", "cupy"] = "numpy"
    coupling_backend: Literal["pairwise", "mlfmm"] = "pairwise"
    periodic: PeriodicSpec | None = None
    postprocessing_backend: Literal["inherit", "numpy", "cupy"] = "inherit"
    mlfmm_options: MLFMMOptions | None = None
    compute_dtype: Literal["complex64", "complex128"] = "complex128"
    accum_dtype: Literal["complex64", "complex128"] = "complex128"
    cache_translation_blocks: bool = False
    check_circumscribing_sphere_overlap: bool = True
    circumscribing_sphere_overlap_atol: float = 0.0
    solve_polarization_basis: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if not (float(self.wavelength) > 0.0):
            raise ValueError(f"`wavelength` must be > 0. Got {self.wavelength!r}.")
        if int(self.lmax) < 1:
            raise ValueError(f"`lmax` must be >= 1. Got {self.lmax!r}.")
        n_medium = complex(self.n_medium)
        if abs(n_medium.imag) > 0.0:
            raise ValueError(f"`n_medium` must be real for this solver path. Got {n_medium!r}.")
        if not (float(n_medium.real) > 0.0):
            raise ValueError(f"`n_medium` must be positive. Got {n_medium!r}.")
        if float(self.radial_lut_dr) < 0.0:
            raise ValueError(f"`radial_lut_dr` must be >= 0. Got {self.radial_lut_dr!r}.")
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
        if self.solver_warm_start is not None:
            ws = np.asarray(self.solver_warm_start)
            if ws.ndim not in (1, 2):
                raise ValueError("`solver_warm_start` must be 1D, 2D, or None.")
        resolve_compute_accum_dtypes(
            compute_dtype=self.compute_dtype,
            accum_dtype=self.accum_dtype,
        )

        method = str(self.solver_method).lower()
        allowed = {"auto", "gmres", "fgmres", "bicgstab", "lgmres", "gcrotmk", "direct"}
        if method not in allowed:
            raise ValueError(
                f"`solver_method` must be one of {sorted(allowed)}. Got {self.solver_method!r}."
            )
        backend = str(self.operator_backend).lower()
        if backend not in {"numpy", "cupy"}:
            raise ValueError(
                "`operator_backend` must be one of {'numpy', 'cupy'}. "
                f"Got {self.operator_backend!r}."
            )
        coupling_backend = str(self.coupling_backend).lower()
        if coupling_backend not in {"pairwise", "mlfmm"}:
            raise ValueError(
                "`coupling_backend` must be one of {'pairwise', 'mlfmm'}. "
                f"Got {self.coupling_backend!r}."
            )
        if self.periodic is not None:
            if not isinstance(self.periodic, PeriodicSpec):
                raise TypeError(
                    "`periodic` must be a PeriodicSpec instance or None. "
                    f"Got {type(self.periodic).__name__}."
                )
            if coupling_backend != "pairwise":
                raise NotImplementedError("Periodic MLFMM coupling is not implemented yet.")
            if backend == "cupy" and self.periodic.options.method != "ewald":
                raise NotImplementedError(
                    "CuPy periodic workflows currently support only Ewald coupling."
                )
        post_backend = str(self.postprocessing_backend).lower()
        if post_backend not in {"inherit", "numpy", "cupy"}:
            raise ValueError(
                "`postprocessing_backend` must be one of {'inherit', 'numpy', 'cupy'}. "
                f"Got {self.postprocessing_backend!r}."
            )
        if backend == "cupy" and bool(self.cache_translation_blocks) and self.periodic is None:
            raise ValueError(
                "`cache_translation_blocks=True` is not supported with finite `operator_backend='cupy'`. "
                "The finite CuPy direct-coupling path does not expose translation-block caching."
            )
        _, az_shared = validate_angular_grid_pair(
            polar_name="polar_angles",
            azimuthal_name="azimuthal_angles",
            polar_values=self.polar_angles,
            azimuthal_values=self.azimuthal_angles,
        )
        warn_redundant_periodic_azimuth_endpoint(azimuth_name="azimuthal_angles", azimuth=az_shared)

        has_source_polar = self.source_polar_angles is not None
        has_source_azimuth = self.source_azimuthal_angles is not None
        if has_source_polar != has_source_azimuth:
            raise ValueError(
                "Set both `source_polar_angles` and `source_azimuthal_angles`, or set neither."
            )
        if has_source_polar:
            _, az_source = validate_angular_grid_pair(
                polar_name="source_polar_angles",
                azimuthal_name="source_azimuthal_angles",
                polar_values=np.asarray(self.source_polar_angles),
                azimuthal_values=np.asarray(self.source_azimuthal_angles),
            )
            warn_redundant_periodic_azimuth_endpoint(
                azimuth_name="source_azimuthal_angles", azimuth=az_source
            )

        has_farfield_polar = self.farfield_polar_angles is not None
        has_farfield_azimuth = self.farfield_azimuthal_angles is not None
        if has_farfield_polar != has_farfield_azimuth:
            raise ValueError(
                "Set both `farfield_polar_angles` and `farfield_azimuthal_angles`, or set neither."
            )
        if has_farfield_polar:
            _, az_farfield = validate_angular_grid_pair(
                polar_name="farfield_polar_angles",
                azimuthal_name="farfield_azimuthal_angles",
                polar_values=np.asarray(self.farfield_polar_angles),
                azimuthal_values=np.asarray(self.farfield_azimuthal_angles),
            )
            warn_redundant_periodic_azimuth_endpoint(
                azimuth_name="farfield_azimuthal_angles", azimuth=az_farfield
            )

        if self.source is not None:
            if not isinstance(self.source, Source):
                raise TypeError(
                    "`source` must satisfy the pyceles Source protocol "
                    "(wavelength/medium_n + incident_coeffs + has_finite_incident_power APIs). "
                    f"Got {type(self.source).__name__}."
                )
            if self.periodic is not None and not isinstance(self.source, PlaneWave):
                raise NotImplementedError(
                    "Periodic workflows currently support PlaneWave excitation only."
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
        polar_values = (
            self.polar_angles if self.source_polar_angles is None else self.source_polar_angles
        )
        azimuthal_values = (
            self.azimuthal_angles
            if self.source_azimuthal_angles is None
            else self.source_azimuthal_angles
        )
        return validate_angular_grid_pair(
            polar_name="source_polar_angles",
            azimuthal_name="source_azimuthal_angles",
            polar_values=np.asarray(polar_values),
            azimuthal_values=np.asarray(azimuthal_values),
        )

    def farfield_angular_grids(self) -> tuple[np.ndarray, np.ndarray]:
        polar_values = (
            self.polar_angles if self.farfield_polar_angles is None else self.farfield_polar_angles
        )
        azimuthal_values = (
            self.azimuthal_angles
            if self.farfield_azimuthal_angles is None
            else self.farfield_azimuthal_angles
        )
        return validate_angular_grid_pair(
            polar_name="farfield_polar_angles",
            azimuthal_name="farfield_azimuthal_angles",
            polar_values=np.asarray(polar_values),
            azimuthal_values=np.asarray(azimuthal_values),
        )

    def resolved_postprocessing_backend(self) -> Literal["numpy", "cupy"]:
        """Return the effective postprocessing backend for this run.

        `"inherit"` keeps the common case terse: users who opt into the CuPy
        solve backend usually want postprocessing to stay on the same array
        backend whenever an accelerated implementation exists. Explicit
        overrides remain available for diagnostics and mixed-backend
        experiments.
        """
        if self.postprocessing_backend == "inherit":
            return self.operator_backend
        return self.postprocessing_backend


__all__ = [
    "SimulationConfig",
    "validate_angular_grid_pair",
    "warn_redundant_periodic_azimuth_endpoint",
]
