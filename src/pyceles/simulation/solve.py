"""Solve-phase helpers for the high-level simulation workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    PairwiseCouplingOperator,
    PreparedOperator,
    assemble_dense_A_numpy,
    prepare_matvec,
)
from pyceles.core.projection import project_source_to_svwf
from pyceles.core.sources import DipoleCollection, DipoleSource, Source
from pyceles.linear.preconditioner import make_grid_block_preconditioner
from pyceles.linear.solvers import (
    DenseLUFactorization,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    solve_linear_system,
)

from .helpers import (
    make_empty_solver_result,
    print_startup_logo_once,
    warn_dipoles_inside_circumspheres,
)
from .results import SolvedSourcesResult


def _assemble_dense_operator_via_matvec(
    A_mv,
    *,
    n: int,
    dtype: np.dtype,
    show_progress: bool,
) -> np.ndarray:
    """Assemble a dense operator by applying the prepared matvec to basis vectors.

    This is the generic dense fallback used when a backend-specific blockwise
    assembler is not available yet. It keeps the direct-solve/LU-cache workflow
    usable on the CuPy backend without forcing a separate dense-assembly kernel.
    """
    A = np.empty((n, n), dtype=dtype)
    eye = np.eye(n, dtype=dtype)
    col_iter: Iterable[int] = range(n)
    if show_progress:
        col_iter = tqdm(col_iter, desc="Assemble A (dense via matvec)")
    for j in col_iter:
        A[:, j] = np.asarray(A_mv(eye[:, j]), dtype=dtype)
    return A


if TYPE_CHECKING:
    from .workflow import Simulation


def validate_source_compatibility(
    sim: Simulation,
    source: Source,
    *,
    label: str,
) -> None:
    """Validate one source against this simulation's wavelength/medium settings."""
    cfg = sim.config
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
        warn_dipoles_inside_circumspheres(
            label=label,
            dipole_positions=dip_pos,
            positions=sim.positions,
            circumscribing_radii=sim.circumscribing_radii,
        )


def normalize_sources_argument(sim: Simulation, sources: Mapping[str, Source]) -> dict[str, Source]:
    """Normalize multi-source mapping inputs to a deterministic labeled dictionary."""
    if not isinstance(sources, Mapping):
        raise TypeError("`sources` must be a mapping `{label: source}`.")
    out: dict[str, Source] = {}
    for key, src in sources.items():
        lbl = str(key)
        if lbl in out:
            raise ValueError(f"Duplicate source label '{lbl}'.")
        out[lbl] = src
    if len(out) == 0:
        raise ValueError("`sources` must contain at least one source.")
    for label, src in out.items():
        validate_source_compatibility(sim, src, label=label)
    return out


def solve_sources_core(
    sim: Simulation,
    labeled_sources: Mapping[str, Source],
    *,
    solver_compute_final_residual: bool | None = None,
) -> SolvedSourcesResult:
    """Solve labeled sources with one shared operator build (solve-only)."""
    cfg = sim.config
    positions = sim.positions
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
    operator_backend = cfg.operator_backend
    will_use_direct = solver_name == "direct" or (
        solver_name == "auto" and unknowns <= int(cfg.solver_direct_max_n)
    )
    if cfg.verbose:
        print_startup_logo_once()
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
        coeff = project_source_to_svwf(
            positions,
            cfg.lmax,
            src,
            polar_angles=source_polar_angles,
            azimuthal_angles=source_azimuthal_angles,
            dtype=compute_dtype,
        )
        initial_coeffs[label] = np.asarray(coeff, dtype=accum_dtype)

    rhs_flat: dict[str, np.ndarray] = {
        label: np.zeros((unknowns,), dtype=accum_dtype) for label in labels
    }
    prepared: PreparedOperator | None = None
    A_mv = None
    A_dense = None
    if unknowns > 0:
        need_prepared = (
            sim._prepared_operator_cache is None
            or sim._prepared_operator_dtype is None
            or sim._prepared_operator_dtype != compute_dtype
        )
        if need_prepared:
            prepared = prepare_matvec(
                lmax=cfg.lmax,
                k=k,
                particles=list(sim.particles),
                n_medium=cfg.n_medium,
                radial_lut_dr=cfg.radial_lut_dr,
                cache_translation_blocks=cfg.cache_translation_blocks,
                operator_dtype=compute_dtype,
                coupling_backend=cfg.coupling_backend,
                mlfmm_options=cfg.mlfmm_options,
                backend=operator_backend,
                show_progress=bool(cfg.verbose),
            )
            sim._prepared_operator_cache = prepared
            sim._prepared_operator_dtype = np.dtype(compute_dtype)
            sim._dense_operator_cache = None
            sim._dense_operator_dtype = None
            sim._dense_lu_cache = None
            sim._dense_lu_dtype = None
        else:
            prepared = sim._prepared_operator_cache
        if prepared is None:
            raise RuntimeError("Internal error: prepared operator cache not initialized.")
        A_mv = prepared.apply_A
        for label in labels:
            rhs_flat[label] = prepared.rhs_Tb(initial_coeffs[label].reshape(Ns * Nm))
        if will_use_direct:
            if operator_backend == "numpy" and not isinstance(
                prepared.coupling, PairwiseCouplingOperator
            ):
                raise NotImplementedError(
                    "Dense/direct NumPy solves currently require the pairwise coupling backend. "
                    "The resolved MLFMM coupling stages remain matrix-free only."
                )
            need_dense = (
                sim._dense_operator_cache is None
                or sim._dense_operator_dtype is None
                or sim._dense_operator_dtype != compute_dtype
            )
            if need_dense:
                if operator_backend == "numpy":
                    A_dense = assemble_dense_A_numpy(
                        prepared,
                        show_progress=bool(cfg.verbose),
                        use_cache=bool(cfg.cache_translation_blocks),
                        store_blocks=False,
                    )
                else:
                    if A_mv is None:
                        raise RuntimeError(
                            "Internal error: direct dense assembly requires prepared A_mv."
                        )
                    A_dense = _assemble_dense_operator_via_matvec(
                        A_mv,
                        n=unknowns,
                        dtype=np.dtype(compute_dtype),
                        show_progress=bool(cfg.verbose),
                    )
                sim._dense_operator_cache = A_dense
                sim._dense_operator_dtype = np.dtype(compute_dtype)
            else:
                A_dense = sim._dense_operator_cache
            need_dense_lu = (
                sim._dense_lu_cache is None
                or sim._dense_lu_dtype is None
                or sim._dense_lu_dtype != compute_dtype
            )
            if need_dense_lu:
                if A_dense is None:
                    raise RuntimeError("Internal error: direct solve requires dense operator.")
                sim._dense_lu_cache = factorize_dense_matrix(
                    A_dense,
                    dtype=compute_dtype,
                    backend=operator_backend,
                    overwrite_input=(operator_backend == "cupy"),
                )
                sim._dense_lu_dtype = np.dtype(compute_dtype)
                # On the CuPy direct path, the cached LU payload is the useful
                # repeated-RHS asset. Releasing the unfactorized dense operator
                # after in-place GPU LU factorization keeps VRAM available for
                # the factorization workspace and subsequent postprocessing.
                if operator_backend == "cupy":
                    sim._dense_operator_cache = None
                    sim._dense_operator_dtype = None
            A_lu: DenseLUFactorization | None = sim._dense_lu_cache
        else:
            A_lu = None
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
    if operator_backend == "cupy" and solver_preconditioner is not None:
        raise NotImplementedError(
            "The CuPy operator backend does not support yet custom preconditioner callables. "
            "Use the built-in `solver_preconditioner_kind='grid_block'` path instead."
        )
    if (
        solver_preconditioner is None
        and cfg.solver_preconditioner_kind == "grid_block"
        and not will_use_direct
        and unknowns > 0
    ):
        if prepared is None:
            raise RuntimeError(
                "Internal error: prepared matvec is required for grid preconditioner."
            )
        solver_preconditioner = make_grid_block_preconditioner(
            prepared,
            backend=operator_backend,
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
        solver_result = make_empty_solver_result(
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
            backend=operator_backend,
            show_progress=bool(cfg.verbose),
            compute_final_residual=compute_final_residual,
        )
        x_arr = np.asarray(solver_result.x)
        x_matrix = (
            x_arr.reshape(unknowns, 1) if n_channels == 1 else x_arr.reshape(unknowns, n_channels)
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


__all__ = ["normalize_sources_argument", "solve_sources_core", "validate_source_compatibility"]
