"""Solve-phase helpers for the high-level simulation workflow."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

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
from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.projection import project_source_to_svwf
from pyceles.core.sources import LocalExpansionSource, PlaneWave, Source
from pyceles.linear.solvers import (
    DenseLUFactorization,
    _finish_backend_solution_capture,
    _start_backend_solution_capture,
    estimate_dense_matrix_bytes,
    factorize_dense_matrix,
    solve_linear_system,
)

from .helpers import (
    make_empty_solver_result,
    print_startup_logo_once,
    warn_local_sources_inside_circumspheres,
)
from .results import SolvedSourcesResult


def _source_projection_uses_angular_grid(source: Source) -> bool:
    """Return whether RHS projection uses the configured angular quadrature grid."""
    return not isinstance(source, (LocalExpansionSource, PlaneWave))


def _record_elapsed(timings: dict[str, float] | None, key: str, start: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - float(start)


def _assemble_dense_operator_via_matvec(
    A_mv,
    *,
    n: int,
    dtype: np.dtype,
    show_progress: bool,
    timings: dict[str, float] | None = None,
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
    t0 = time.perf_counter()
    for j in col_iter:
        A[:, j] = np.asarray(A_mv(eye[:, j]), dtype=dtype)
    _record_elapsed(timings, "dense_operator_assembly_s", t0)
    return A


def _assemble_dense_operator_for_prepared(
    *,
    prepared: PreparedOperator,
    A_mv,
    n: int,
    dtype: np.dtype,
    show_progress: bool,
    timings: dict[str, float] | None = None,
) -> Any:
    """Assemble dense `A`, preferring operator-owned source-block streaming."""
    source_block_assembler = getattr(prepared, "assemble_dense_from_source_blocks", None)
    source_blocks = (
        None
        if source_block_assembler is None
        else source_block_assembler(show_progress=show_progress)
    )
    if source_blocks is not None:
        if timings is not None:
            timings["periodic_w_block_generation_s"] = (
                timings.get("periodic_w_block_generation_s", 0.0)
                + source_blocks.block_generation_seconds
            )
            timings["dense_operator_assembly_s"] = (
                timings.get("dense_operator_assembly_s", 0.0) + source_blocks.assembly_seconds
            )
        return source_blocks.matrix
    return _assemble_dense_operator_via_matvec(
        A_mv,
        n=n,
        dtype=dtype,
        show_progress=show_progress,
        timings=timings,
    )


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
    if isinstance(source, LocalExpansionSource):
        src_pos = np.asarray(source.source_positions(), dtype=float).reshape(-1, 3)
        warn_local_sources_inside_circumspheres(
            label=label,
            source_positions=src_pos,
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


def periodic_shared_k_parallel(
    sim: Simulation,
    labeled_sources: Mapping[str, Source],
) -> np.ndarray | None:
    """Return shared periodic Bloch wavevector or reject unsupported source sets."""
    if sim.config.periodic is None:
        return None
    ref_label: str | None = None
    ref_kp: np.ndarray | None = None
    for label, source in labeled_sources.items():
        if not isinstance(source, PlaneWave):
            raise NotImplementedError(
                "Periodic workflows currently support PlaneWave excitation only. "
                f"Source '{label}' is {type(source).__name__}."
            )
        kp = plane_wave_k_parallel(source)
        if ref_kp is None:
            ref_label = str(label)
            ref_kp = kp
            continue
        if not np.allclose(kp, ref_kp, rtol=1e-12, atol=1e-12):
            raise NotImplementedError(
                "Periodic multi-source solves require all PlaneWave sources to share "
                f"one in-plane Bloch wavevector. Source '{label}' differs from "
                f"source '{ref_label}'."
            )
    if ref_kp is None:
        raise ValueError("Periodic source mapping must contain at least one source.")
    return ref_kp


@dataclass(frozen=True)
class _SolvedSourcesExecution:
    """Private solve result plus short-lived backend data for immediate postprocessing."""

    solved: SolvedSourcesResult
    backend_coeffs: dict[str, Any] | None = None


def _solve_sources_impl(
    sim: Simulation,
    labeled_sources: Mapping[str, Source],
    *,
    solver_compute_final_residual: bool | None = None,
    retain_backend_handoff: bool,
) -> _SolvedSourcesExecution:
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
    phase_timings: dict[str, float] = {}
    solve_sources_t0 = time.perf_counter()

    Ns = positions.shape[0]
    Nm = n_modes(cfg.lmax)
    unknowns = Ns * Nm
    k0 = 2.0 * np.pi / float(cfg.wavelength)
    k = k0 * float(np.real(cfg.n_medium))

    solver_method = cfg.solver_method
    if solver_method is None:
        raise RuntimeError("SimulationConfig did not resolve its default solver method.")
    solver_name = str(solver_method).lower()
    operator_backend = cfg.operator_backend
    k_parallel = periodic_shared_k_parallel(sim, labeled_sources)
    periodic_key = (
        None
        if k_parallel is None
        else (float(np.asarray(k_parallel)[0]), float(np.asarray(k_parallel)[1]))
    )
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
        uses_source_grid = any(
            _source_projection_uses_angular_grid(labeled_sources[label]) for label in labels
        )
        if uses_source_grid:
            print(
                "Source angular grid:"
                f" beta={source_polar_angles.size}, alpha={source_azimuthal_angles.size}"
            )
        else:
            print("Source projection: analytic/local coefficients (no angular quadrature)")

    initial_coeffs: dict[str, np.ndarray] = {}
    source_projection_t0 = time.perf_counter()
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
    phase_timings["source_projection_s"] = time.perf_counter() - source_projection_t0

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
            or sim._prepared_operator_periodic_key != periodic_key
        )
        if need_prepared:
            prepare_t0 = time.perf_counter()
            prepared = prepare_matvec(
                lmax=cfg.lmax,
                k=k,
                particles=sim.particles,
                n_medium=cfg.n_medium,
                radial_lut_dr=cfg.radial_lut_dr,
                cache_translation_blocks=cfg.cache_translation_blocks,
                operator_dtype=compute_dtype,
                coupling_backend=cfg.coupling_backend,
                mlfmm_options=cfg.mlfmm_options,
                periodic=cfg.periodic,
                k_parallel=k_parallel,
                backend=operator_backend,
                show_progress=bool(cfg.verbose),
            )
            phase_timings["prepare_operator_s"] = time.perf_counter() - prepare_t0
            sim._prepared_operator_cache = prepared
            sim._prepared_operator_dtype = np.dtype(compute_dtype)
            sim._prepared_operator_periodic_key = periodic_key
            sim._dense_operator_cache = None
            sim._dense_operator_dtype = None
            sim._dense_lu_cache = None
            sim._dense_lu_dtype = None
        else:
            prepared = sim._prepared_operator_cache
        if prepared is None:
            raise RuntimeError("Internal error: prepared operator cache not initialized.")
        A_mv = prepared.apply_A
        rhs_t0 = time.perf_counter()
        for label in labels:
            rhs_flat[label] = prepared.rhs_Tb(initial_coeffs[label].reshape(Ns * Nm))
        phase_timings["rhs_Tb_s"] = time.perf_counter() - rhs_t0
        rayleigh_preparation = (
            cfg.periodic is not None and cfg.periodic.options.method == "rayleigh"
        )
        if (bool(cfg.cache_translation_blocks) and not will_use_direct) or rayleigh_preparation:
            populate_t0 = time.perf_counter()
            prepared.populate_coupling(show_progress=bool(cfg.verbose))
            populate_elapsed = time.perf_counter() - populate_t0
            if rayleigh_preparation:
                phase_timings["periodic_rayleigh_preparation_s"] = populate_elapsed
            elif bool(cfg.cache_translation_blocks) and not will_use_direct:
                phase_timings["periodic_w_block_generation_s"] = populate_elapsed
        if will_use_direct:
            need_dense_lu = (
                sim._dense_lu_cache is None
                or sim._dense_lu_dtype is None
                or sim._dense_lu_dtype != compute_dtype
            )
            if need_dense_lu:
                need_dense = (
                    sim._dense_operator_cache is None
                    or sim._dense_operator_dtype is None
                    or sim._dense_operator_dtype != compute_dtype
                )
                if need_dense:
                    if operator_backend == "numpy" and isinstance(
                        prepared.coupling, PairwiseCouplingOperator
                    ):
                        dense_t0 = time.perf_counter()
                        A_dense = assemble_dense_A_numpy(
                            prepared,
                            show_progress=bool(cfg.verbose),
                            use_cache=bool(cfg.cache_translation_blocks),
                            store_blocks=False,
                        )
                        _record_elapsed(phase_timings, "dense_operator_assembly_s", dense_t0)
                    else:
                        if A_mv is None:
                            raise RuntimeError(
                                "Internal error: direct dense assembly requires prepared A_mv."
                            )
                        A_dense = _assemble_dense_operator_for_prepared(
                            prepared=prepared,
                            A_mv=A_mv,
                            n=unknowns,
                            dtype=np.dtype(compute_dtype),
                            show_progress=bool(cfg.verbose),
                            timings=phase_timings,
                        )
                    sim._dense_operator_cache = A_dense
                    sim._dense_operator_dtype = np.dtype(compute_dtype)
                else:
                    A_dense = sim._dense_operator_cache
                if A_dense is None:
                    raise RuntimeError("Internal error: direct solve requires dense operator.")
                factor_t0 = time.perf_counter()
                sim._dense_lu_cache = factorize_dense_matrix(
                    A_dense,
                    dtype=compute_dtype,
                    backend=operator_backend,
                    overwrite_input=(operator_backend == "cupy"),
                )
                phase_timings["dense_factorization_s"] = time.perf_counter() - factor_t0
                sim._dense_lu_dtype = np.dtype(compute_dtype)
                # On the CuPy direct path, the cached LU payload is the useful
                # repeated-RHS asset. Releasing the unfactorized dense operator
                # after in-place GPU LU factorization keeps VRAM available for
                # the factorization workspace and subsequent postprocessing.
                if operator_backend == "cupy":
                    sim._dense_operator_cache = None
                    sim._dense_operator_dtype = None
                    # The LU payload is now the owner of the device matrix.  Do
                    # not pass the unfactorized dense matrix into the direct
                    # solver again; that would only trigger an avoidable copy
                    # for residual bookkeeping.
                    A_dense = None
            else:
                # Repeated direct solves only need the cached LU payload.
                # Reassembling dense A here would defeat the intended repeated-RHS fast path.
                A_dense = None
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
            "The high-level CuPy simulation path does not support custom "
            "preconditioner callables yet. Use the low-level linear solver "
            "API for backend-native experimental preconditioners."
        )

    backend_capture: dict[str, Any] | None = None
    if unknowns == 0:
        solver_result = make_empty_solver_result(
            dtype=compute_dtype, nrhs=n_channels, method=solver_method
        )
        x_matrix = np.zeros((unknowns, n_channels), dtype=compute_dtype)
    else:
        if A_mv is None:
            raise RuntimeError("Internal error: A_mv not prepared for non-empty system.")
        linear_solve_t0 = time.perf_counter()
        if operator_backend == "cupy" and bool(retain_backend_handoff):
            token, backend_capture = _start_backend_solution_capture()
            try:
                solver_result = solve_linear_system(
                    A_mv,
                    rhs_arg,
                    method=solver_method,
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
            finally:
                _finish_backend_solution_capture(token)
        else:
            solver_result = solve_linear_system(
                A_mv,
                rhs_arg,
                method=solver_method,
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
        phase_timings["linear_solve_s"] = time.perf_counter() - linear_solve_t0
        x_arr = np.asarray(solver_result.x)
        x_matrix = (
            x_arr.reshape(unknowns, 1) if n_channels == 1 else x_arr.reshape(unknowns, n_channels)
        )

    phase_timings["solve_sources_core_s"] = time.perf_counter() - solve_sources_t0
    metadata = dict(solver_result.block_metadata or {})
    metadata["simulation_phase_timings_s"] = dict(phase_timings)
    solver_result = replace(solver_result, block_metadata=metadata)

    coeffs: dict[str, np.ndarray] = {}
    rhs_out: dict[str, np.ndarray] = {}
    backend_coeffs: dict[str, Any] | None = None
    backend_x = None if backend_capture is None else backend_capture.get("x")
    if backend_x is not None and unknowns > 0:
        x_backend_arr = backend_x
        x_backend_matrix = (
            x_backend_arr.reshape(unknowns, 1)
            if n_channels == 1
            else x_backend_arr.reshape(unknowns, n_channels)
        )
        backend_coeffs = {}
    for j, label in enumerate(labels):
        x_col = x_matrix[:, j].reshape(Ns, Nm)
        coeffs[label] = x_col
        rhs_out[label] = np.asarray(rhs_flat[label]).reshape(Ns, Nm)
        if backend_coeffs is not None:
            backend_coeffs[label] = x_backend_matrix[:, j].reshape(Ns, Nm)

    solved = SolvedSourcesResult(
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
    return _SolvedSourcesExecution(solved=solved, backend_coeffs=backend_coeffs)


def solve_sources_core(
    sim: Simulation,
    labeled_sources: Mapping[str, Source],
    *,
    solver_compute_final_residual: bool | None = None,
) -> SolvedSourcesResult:
    """Solve labeled sources with one shared operator build (solve-only)."""
    return _solve_sources_impl(
        sim,
        labeled_sources,
        solver_compute_final_residual=solver_compute_final_residual,
        retain_backend_handoff=False,
    ).solved


def _solve_sources_for_immediate_postprocess_core(
    sim: Simulation,
    labeled_sources: Mapping[str, Source],
    *,
    solver_compute_final_residual: bool | None = None,
) -> _SolvedSourcesExecution:
    """Solve sources while retaining backend coefficients for the immediate consumer."""
    return _solve_sources_impl(
        sim,
        labeled_sources,
        solver_compute_final_residual=solver_compute_final_residual,
        retain_backend_handoff=sim.config.periodic is None,
    )


__all__ = [
    "normalize_sources_argument",
    "periodic_shared_k_parallel",
    "solve_sources_core",
    "validate_source_compatibility",
]
