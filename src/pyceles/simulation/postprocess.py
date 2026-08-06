"""Postprocessing-phase helpers for solved simulation channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.particles import LayeredSphere, Particle, PECSphere, Sphere, Spheroid
from pyceles.core.sources import (
    DipoleCollection,
    DipoleSource,
    JonesPolarizedSource,
    PlaneWave,
    Source,
)
from pyceles.postprocessing.farfield import (
    CrossSectionBalance,
    FarFieldPatterns,
    PeriodicFarFieldPayload,
    PowerBalance,
    compute_far_field_patterns,
    finite_beam_power_balance,
    local_absorption_cross_section_from_exciting,
    local_power_balance_from_exciting,
    mix_periodic_farfield_payloads,
    plane_wave_cross_section_balance,
    pwp_power_decomposition,
)
from pyceles.postprocessing.farfield.periodic import periodic_plane_wave_orders

from .config import validate_angular_grid_pair
from .results import (
    ChannelResult,
    MultiSourceResult,
    MultiSourceSolveResult,
    PolarizationResult,
    ResultRetention,
    SimulationResult,
    UnpolarizedDiagnostics,
    _apply_solver_result_retention,
    _CoherentChannelPayload,
    average_cross_section_balances,
    average_power_balances,
    empty_farfield_patterns,
    simulation_result_from_channel,
)

if TYPE_CHECKING:
    from .workflow import Simulation


def _is_numerically_lossless_cluster(
    particles: Sequence[Particle], *, imag_tol: float = 0.0
) -> bool:
    """Return whether every known particle material is numerically lossless."""
    tol = float(imag_tol)
    for particle in particles:
        if isinstance(particle, PECSphere):
            continue
        if isinstance(particle, (Sphere, Spheroid)):
            if abs(complex(particle.refractive_index).imag) > tol:
                return False
            continue
        if isinstance(particle, LayeredSphere):
            if any(abs(complex(n).imag) > tol for n in particle.layer_refractive_indices):
                return False
            continue
        return False
    return True


def _build_exciting_scattered_flat_generic_route(
    sim: Simulation,
    *,
    initial_coeffs: Any,
    coeffs: Any,
    accum_dtype: np.dtype,
):
    """Build flattened `(e, x)` payload for generic post-solve local-dissipation routes."""
    coeffs_size = int(getattr(coeffs, "size", 0))
    if coeffs_size == 0:
        return np.zeros((0,), dtype=np.dtype(accum_dtype)), np.zeros(
            (0,), dtype=np.dtype(accum_dtype)
        )
    prepared = sim._prepared_operator_cache
    if prepared is None:
        return None
    operator_dtype = np.dtype(prepared.dtype)
    use_cupy = bool(
        str(sim.config.operator_backend).lower() == "cupy"
        or is_cupy_array(coeffs)
        or is_cupy_array(initial_coeffs)
    )
    if use_cupy:
        cupy, _ = import_cupy()
        # Keep the operator application in its configured compute dtype.  Only
        # the final local-power reduction needs accumulation-precision copies.
        # Immediate postprocessing normally receives backend-native solve
        # coefficients, so this also avoids a host round trip.
        x_operator = cupy.asarray(coeffs, dtype=operator_dtype).reshape(-1)
        wx = prepared.apply_W(x_operator)
        x_flat = cupy.asarray(x_operator, dtype=accum_dtype).reshape(-1)
        b_flat = cupy.asarray(initial_coeffs, dtype=accum_dtype).reshape(-1)
        e_flat = b_flat + cupy.asarray(wx, dtype=accum_dtype).reshape(-1)
        return e_flat, x_flat
    x_operator = np.asarray(coeffs, dtype=operator_dtype).reshape(-1)
    wx = prepared.apply_W(x_operator)
    x_flat = np.asarray(x_operator, dtype=accum_dtype).reshape(-1)
    e_flat = np.asarray(initial_coeffs, dtype=accum_dtype).reshape(-1) + np.asarray(
        wx, dtype=accum_dtype
    ).reshape(-1)
    return e_flat, x_flat


def _plane_wave_local_absorption_generic_route(
    sim: Simulation,
    *,
    source: Source,
    initial_coeffs: Any,
    coeffs: Any,
    k0: float,
    accum_dtype: np.dtype,
) -> float | None:
    """Compute local plane-wave absorption through the generic `e=b+W x` route.

    For CuPy runs this keeps the extra post-solve `W x` apply and local reduction
    on-device, returning only the final scalar to host.

    CuPy solve outputs are retained as backend-native side payloads during
    immediate postprocessing, so this route can avoid re-uploading solved
    coefficients after the public NumPy result arrays have been materialized.
    """
    if not isinstance(source, PlaneWave):
        return None
    payload = _build_exciting_scattered_flat_generic_route(
        sim,
        initial_coeffs=initial_coeffs,
        coeffs=coeffs,
        accum_dtype=accum_dtype,
    )
    if payload is None:
        return None
    e_flat, x_flat = payload
    return local_absorption_cross_section_from_exciting(
        source,
        e_flat,
        x_flat,
        k0=float(k0),
        n_medium=sim.config.n_medium,
    )


def _local_power_balance_generic_route(
    sim: Simulation,
    *,
    initial_coeffs: Any,
    coeffs: Any,
    k0: float,
    accum_dtype: np.dtype,
) -> PowerBalance | None:
    """Compute local absorbed power via the generic `e=b+W x` route."""
    payload = _build_exciting_scattered_flat_generic_route(
        sim,
        initial_coeffs=initial_coeffs,
        coeffs=coeffs,
        accum_dtype=accum_dtype,
    )
    if payload is None:
        return None
    e_flat, x_flat = payload
    return local_power_balance_from_exciting(
        e_flat,
        x_flat,
        k0=float(k0),
        n_medium=sim.config.n_medium,
        n_particles=int(sim.positions.shape[0]),
        nmodes_per_particle=int(n_modes(sim.config.lmax)),
    )


def _print_power_balance(power: PowerBalance, *, label: str | None = None) -> None:
    """Print one concise backend-independent power-balance summary."""
    prefix = "Power balance" if label is None else f"Power balance [{label}]"

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{float(value):.6g}"

    print(
        f"{prefix}: R={fmt(power.reflectance)} T={fmt(power.transmittance)} "
        f"local_absorptance={fmt(power.local_absorptance)} "
        f"flux_defect_fraction={fmt(power.flux_defect_fraction)} "
        f"closure_error_fraction={fmt(power.closure_error_fraction)}"
    )


def _build_periodic_result(
    sim: Simulation,
    *,
    source: Source,
    coeffs: np.ndarray,
    k: float,
) -> PeriodicFarFieldPayload:
    """Compute periodic diffraction orders and the common unit-cell power balance."""
    periodic = sim.config.periodic
    if periodic is None:
        raise RuntimeError("Internal error: periodic postprocess called without periodic config.")
    if not isinstance(source, PlaneWave):
        raise NotImplementedError(
            "Periodic postprocessing currently supports PlaneWave excitation only."
        )

    return periodic_plane_wave_orders(
        source=source,
        lattice=periodic.lattice,
        positions=sim.positions,
        coeffs=coeffs,
        lmax=int(sim.config.lmax),
        k=float(k),
        n_medium=sim.config.n_medium,
        output_bmax=periodic.options.output_bmax,
    )


def mix_farfield_patterns(
    ff_te: FarFieldPatterns,
    ff_tm: FarFieldPatterns,
    *,
    a_te: complex,
    a_tm: complex,
    dtype: np.dtype,
) -> FarFieldPatterns:
    """Build requested Jones far field by coherent TE/TM basis recombination."""
    initial = None
    if ff_te.initial is not None and ff_tm.initial is not None:
        initial = ff_te.initial.linear_combination(
            ff_tm.initial,
            weight_self=a_te,
            weight_other=a_tm,
            dtype=dtype,
        )

    return FarFieldPatterns(
        initial=initial,
        scattered=ff_te.scattered.linear_combination(
            ff_tm.scattered,
            weight_self=a_te,
            weight_other=a_tm,
            dtype=dtype,
        ),
    )


def _build_channel_diagnostics(
    sim: Simulation,
    *,
    source: Source,
    initial_coeffs: np.ndarray,
    coeffs: np.ndarray,
    local_initial_coeffs: Any | None = None,
    local_coeffs: Any | None = None,
    farfield: FarFieldPatterns,
    k: float,
    k0: float,
    accum_dtype: np.dtype,
) -> tuple[
    PowerBalance | None,
    CrossSectionBalance | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Build optional power/cross-section diagnostics for one channel payload.

    This centralizes the physics-policy branching so the single-channel path
    and the mixed-from-basis path cannot silently drift apart when we extend
    diagnostics in the future.
    """
    cfg = sim.config
    local_b = initial_coeffs if local_initial_coeffs is None else local_initial_coeffs
    local_x = coeffs if local_coeffs is None else local_coeffs

    power: PowerBalance | None = None
    cross_sections: CrossSectionBalance | None = None
    decomposition_forward: dict[str, float] | None = None
    decomposition_backward: dict[str, float] | None = None

    if isinstance(source, PlaneWave):
        # Plane-wave cross sections are physical observables rather than raw
        # solver diagnostics. Preserve the exact zero for known lossless
        # materials and avoid an additional W @ x operator application. Finite
        # beams and periodic runs expose the un-clipped local power estimator
        # through PowerBalance, where it is useful as a closure diagnostic.
        c_abs_local = (
            0.0
            if _is_numerically_lossless_cluster(sim.particles)
            else _plane_wave_local_absorption_generic_route(
                sim,
                source=source,
                initial_coeffs=local_b,
                coeffs=local_x,
                k0=float(k0),
                accum_dtype=accum_dtype,
            )
        )
        if c_abs_local is None:
            raise RuntimeError("Plane-wave local absorption could not be evaluated.")
        cross_sections = plane_wave_cross_section_balance(
            source,
            initial_coeffs,
            coeffs,
            farfield.scattered,
            k0=k0,
            n_medium=cfg.n_medium,
            local_absorption=c_abs_local,
        )
        return power, cross_sections, decomposition_forward, decomposition_backward

    if farfield.initial is not None and source.has_finite_incident_power():
        p_abs_diag = _local_power_balance_generic_route(
            sim,
            initial_coeffs=local_b,
            coeffs=local_x,
            k0=float(k0),
            accum_dtype=accum_dtype,
        )
        p_abs_local_particles = (
            None if p_abs_diag is None else p_abs_diag.local_absorbed_power_per_particle
        )
        power = finite_beam_power_balance(
            source,
            farfield.initial,
            farfield.scattered,
            k0=k0,
            k_medium=k,
            local_absorbed_power=(None if p_abs_diag is None else p_abs_diag.local_absorbed_power),
            local_absorbed_power_per_particle=p_abs_local_particles,
        )
        decomposition_forward = pwp_power_decomposition(
            direction="forward",
            initial=farfield.initial,
            scattered=farfield.scattered,
            k0=k0,
            k_medium=k,
            source=source,
        )
        decomposition_backward = pwp_power_decomposition(
            direction="backward",
            initial=farfield.initial,
            scattered=farfield.scattered,
            k0=k0,
            k_medium=k,
            source=source,
        )
        return power, cross_sections, decomposition_forward, decomposition_backward

    if isinstance(source, (DipoleSource, DipoleCollection)):
        p_abs_diag = _local_power_balance_generic_route(
            sim,
            initial_coeffs=local_b,
            coeffs=local_x,
            k0=float(k0),
            accum_dtype=accum_dtype,
        )
        if p_abs_diag is not None:
            power = p_abs_diag
    return power, cross_sections, decomposition_forward, decomposition_backward


def _unpolarized_diagnostics(
    run_te: ChannelResult,
    run_tm: ChannelResult,
) -> UnpolarizedDiagnostics:
    """Build incoherent TE/TM scalar averages for one polarization solve."""
    power = None
    cross_sections = None
    if run_te.power is not None and run_tm.power is not None:
        power = average_power_balances(run_te.power, run_tm.power)
    if run_te.cross_sections is not None and run_tm.cross_sections is not None:
        cross_sections = average_cross_section_balances(
            run_te.cross_sections,
            run_tm.cross_sections,
        )
    return UnpolarizedDiagnostics(power=power, cross_sections=cross_sections)


def _assemble_channel_result(
    sim: Simulation,
    *,
    source: Source,
    retention: ResultRetention,
    initial_coeffs: np.ndarray,
    rhs: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    k0: float,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    farfield: FarFieldPatterns,
    power: PowerBalance | None,
    cross_sections: CrossSectionBalance | None,
    decomposition_forward: dict[str, float] | None,
    decomposition_backward: dict[str, float] | None,
    periodic: PeriodicFarFieldPayload | None = None,
) -> ChannelResult:
    """Assemble one physical channel without assigning solve provenance."""
    ns = int(sim.positions.shape[0])
    nm = int(n_modes(sim.config.lmax))
    return ChannelResult(
        config=sim.config,
        source=source,
        particles=sim.particles,
        k=k,
        k0=k0,
        coeffs=np.asarray(coeffs),
        rhs=np.asarray(rhs).reshape(ns, nm) if retention.rhs else None,
        initial_coeffs=np.asarray(initial_coeffs) if retention.initial_coeffs else None,
        farfield=farfield,
        power=power,
        cross_sections=cross_sections,
        decomposition_forward=decomposition_forward,
        decomposition_backward=decomposition_backward,
        periodic=periodic,
        compute_dtype=str(compute_dtype),
        accum_dtype=str(accum_dtype),
    )


def build_single_channel_result(
    sim: Simulation,
    *,
    source: Source,
    retention: ResultRetention,
    initial_coeffs: np.ndarray,
    rhs_flat: np.ndarray,
    coeffs: np.ndarray,
    backend_coeffs: Any | None = None,
    k: float,
    k0: float,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    farfield_polar_angles: np.ndarray,
    farfield_azimuthal_angles: np.ndarray,
    include_farfield: bool,
) -> ChannelResult:
    """Assemble one solved channel from its coefficients and observables."""
    cfg = sim.config
    positions = sim.positions
    Ns = positions.shape[0]
    Nm = n_modes(cfg.lmax)

    power: PowerBalance | None = None
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
            backend=cfg.resolved_postprocessing_backend(),
            dtype=compute_dtype,
            show_progress=bool(cfg.verbose),
        )
        power, cross_sections, decomposition_forward, decomposition_backward = (
            _build_channel_diagnostics(
                sim,
                source=source,
                initial_coeffs=initial_coeffs,
                coeffs=coeffs,
                local_initial_coeffs=None,
                local_coeffs=backend_coeffs,
                farfield=ff,
                k=k,
                k0=k0,
                accum_dtype=accum_dtype,
            )
        )
    else:
        ff = empty_farfield_patterns(compute_dtype)

    return _assemble_channel_result(
        sim,
        source=source,
        retention=retention,
        k=k,
        k0=k0,
        coeffs=coeffs,
        rhs=np.asarray(rhs_flat).reshape(Ns, Nm),
        initial_coeffs=initial_coeffs,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        farfield=ff,
        power=power,
        cross_sections=cross_sections,
        decomposition_forward=decomposition_forward,
        decomposition_backward=decomposition_backward,
    )


def postprocess_sources_impl(
    sim: Simulation,
    solved: MultiSourceSolveResult,
    *,
    include_farfield: bool = True,
    farfield_polar_angles: np.ndarray | None = None,
    farfield_azimuthal_angles: np.ndarray | None = None,
    backend_coeffs_by_label: Mapping[str, Any] | None = None,
    retention: ResultRetention | None = None,
) -> MultiSourceResult:
    """Postprocess one block solve into explicit per-source channel payloads."""
    retained = ResultRetention() if retention is None else retention
    if not isinstance(retained, ResultRetention):
        raise TypeError("`retention` must be a ResultRetention instance or None.")
    if solved.config is not sim.config or solved.particles is not sim.particles:
        raise ValueError(
            "`solved` must come from this Simulation instance or another Simulation "
            "sharing the exact same immutable config and particle collection."
        )
    cfg = sim.config
    labels = tuple(solved.labels)
    n_channels = len(labels)
    periodic_run = cfg.periodic is not None

    Ns = sim.positions.shape[0]
    Nm = n_modes(cfg.lmax)
    compute_dtype = np.dtype(solved.compute_dtype)
    accum_dtype = np.dtype(solved.accum_dtype)

    if periodic_run:
        periodic_runs: dict[str, ChannelResult] = {}
        for label in labels:
            if label not in solved.sources:
                raise KeyError(f"Missing source payload for label '{label}'.")
            if label not in solved.initial_coeffs:
                raise KeyError(f"Missing initial coefficients for label '{label}'.")
            if label not in solved.rhs:
                raise KeyError(f"Missing RHS payload for label '{label}'.")
            if label not in solved.coeffs:
                raise KeyError(f"Missing solved coefficients for label '{label}'.")

            x_col = solved.coeffs[label]
            if tuple(np.shape(x_col)) != (Ns, Nm):
                raise ValueError(
                    f"Solved coefficients for label '{label}' must have shape {(Ns, Nm)}. "
                    f"Got {np.shape(x_col)}."
                )
            rhs_col = solved.rhs[label]
            if tuple(np.shape(rhs_col)) != (Ns, Nm):
                raise ValueError(
                    f"Solved RHS for label '{label}' must have shape {(Ns, Nm)}. "
                    f"Got {np.shape(rhs_col)}."
                )
            periodic_payload = _build_periodic_result(
                sim,
                source=solved.sources[label],
                coeffs=np.asarray(x_col),
                k=float(solved.k),
            )
            backend_coeffs = (
                None if backend_coeffs_by_label is None else backend_coeffs_by_label[label]
            )
            local_components = _local_power_balance_generic_route(
                sim,
                initial_coeffs=solved.initial_coeffs[label],
                coeffs=x_col if backend_coeffs is None else backend_coeffs,
                k0=float(solved.k0),
                accum_dtype=accum_dtype,
            )
            if local_components is not None:
                periodic_payload = replace(
                    periodic_payload,
                    power=replace(
                        periodic_payload.power,
                        local_absorbed_power=local_components.local_absorbed_power,
                        local_absorbed_power_per_particle=(
                            local_components.local_absorbed_power_per_particle
                        ),
                    ),
                )
            if cfg.verbose:
                order_count = int(np.asarray(periodic_payload.order_mn).shape[0])
                propagating_count = int(
                    np.count_nonzero(np.asarray(periodic_payload.order_propagating, dtype=bool))
                )
                prefix = "Periodic orders"
                if n_channels > 1:
                    prefix = f"{prefix} [{label}]"
                print(f"{prefix}: total={order_count} propagating={propagating_count}")
                _print_power_balance(
                    periodic_payload.power,
                    label=label if n_channels > 1 else None,
                )
            periodic_runs[label] = _assemble_channel_result(
                sim,
                source=solved.sources[label],
                retention=retained,
                k=float(solved.k),
                k0=float(solved.k0),
                coeffs=np.asarray(x_col),
                rhs=np.asarray(rhs_col, dtype=accum_dtype).reshape(Ns, Nm),
                initial_coeffs=np.asarray(solved.initial_coeffs[label]),
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
                # Periodic runs expose far-field observables through
                # `ChannelResult.periodic`; keep finite-cluster PWP families
                # as intentional empty placeholders.
                farfield=empty_farfield_patterns(compute_dtype),
                power=periodic_payload.power,
                cross_sections=None,
                decomposition_forward=None,
                decomposition_backward=None,
                periodic=periodic_payload,
            )
        return MultiSourceResult(
            labels=labels,
            channels=periodic_runs,
            solver_result=_apply_solver_result_retention(
                solved.solver_result,
                retention=retained,
            ),
        )

    if (farfield_polar_angles is None) != (farfield_azimuthal_angles is None):
        raise ValueError(
            "Set both `farfield_polar_angles` and `farfield_azimuthal_angles`, or set neither."
        )
    if farfield_polar_angles is None:
        ff_polar, ff_azimuth = cfg.farfield_angular_grids()
    else:
        ff_polar, ff_azimuth = validate_angular_grid_pair(
            polar_name="farfield_polar_angles",
            azimuthal_name="farfield_azimuthal_angles",
            polar_values=np.asarray(farfield_polar_angles),
            azimuthal_values=np.asarray(farfield_azimuthal_angles),
        )

    runs: dict[str, ChannelResult] = {}
    for label in labels:
        if label not in solved.sources:
            raise KeyError(f"Missing source payload for label '{label}'.")
        if label not in solved.initial_coeffs:
            raise KeyError(f"Missing initial coefficients for label '{label}'.")
        if label not in solved.rhs:
            raise KeyError(f"Missing RHS payload for label '{label}'.")
        if label not in solved.coeffs:
            raise KeyError(f"Missing solved coefficients for label '{label}'.")

        x_col = solved.coeffs[label]
        if tuple(np.shape(x_col)) != (Ns, Nm):
            raise ValueError(
                f"Solved coefficients for label '{label}' must have shape {(Ns, Nm)}. "
                f"Got {np.shape(x_col)}."
            )
        rhs_col = solved.rhs[label]
        if tuple(np.shape(rhs_col)) != (Ns, Nm):
            raise ValueError(
                f"Solved RHS for label '{label}' must have shape {(Ns, Nm)}. "
                f"Got {np.shape(rhs_col)}."
            )
        backend_coeffs = None if backend_coeffs_by_label is None else backend_coeffs_by_label[label]
        run = build_single_channel_result(
            sim,
            source=solved.sources[label],
            retention=retained,
            initial_coeffs=solved.initial_coeffs[label],
            rhs_flat=np.asarray(rhs_col, dtype=accum_dtype).reshape(Ns * Nm),
            coeffs=x_col,
            backend_coeffs=backend_coeffs,
            k=float(solved.k),
            k0=float(solved.k0),
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            farfield_polar_angles=ff_polar,
            farfield_azimuthal_angles=ff_azimuth,
            include_farfield=include_farfield,
        )
        runs[label] = run
        if cfg.verbose and run.power is not None:
            _print_power_balance(run.power, label=label if n_channels > 1 else None)

    return MultiSourceResult(
        labels=labels,
        channels=runs,
        solver_result=_apply_solver_result_retention(
            solved.solver_result,
            retention=retained,
        ),
    )


def run_impl(
    sim: Simulation,
    source: Source,
    *,
    include_farfield: bool = True,
    retention: ResultRetention | None = None,
    warm_start: np.ndarray | None = None,
    solver_compute_final_residual: bool | None = None,
) -> SimulationResult:
    """Solve and postprocess one explicit source channel."""
    execution = sim._solve_sources_for_immediate_postprocess(
        {"source": source},
        warm_start=warm_start,
        solver_compute_final_residual=solver_compute_final_residual,
    )
    multi = postprocess_sources_impl(
        sim,
        execution.solved,
        include_farfield=include_farfield,
        backend_coeffs_by_label=execution.backend_coeffs,
        retention=retention,
    )
    return simulation_result_from_channel(multi["source"], multi.solver_result)


def run_polarizations_impl(
    sim: Simulation,
    source: JonesPolarizedSource,
    *,
    include_farfield: bool = True,
    retention: ResultRetention | None = None,
    warm_start: np.ndarray | Mapping[str, np.ndarray] | None = None,
    solver_compute_final_residual: bool | None = None,
) -> PolarizationResult:
    """Solve TE/TM basis channels together and build a typed polarization result."""
    retained = ResultRetention() if retention is None else retention
    if not isinstance(retained, ResultRetention):
        raise TypeError("`retention` must be a ResultRetention instance or None.")
    if not isinstance(source, JonesPolarizedSource):
        raise TypeError(
            "`run_polarizations()` requires a source with Jones metadata and "
            "`with_polarization('TE'/'TM')`."
        )

    a_te, a_tm = source.jones_coefficients()
    src_te = source.with_polarization("TE")
    src_tm = source.with_polarization("TM")
    execution = sim._solve_sources_for_immediate_postprocess(
        {"te": src_te, "tm": src_tm},
        warm_start=warm_start,
        solver_compute_final_residual=solver_compute_final_residual,
    )
    solved = execution.solved
    basis = postprocess_sources_impl(
        sim,
        solved,
        include_farfield=include_farfield,
        backend_coeffs_by_label=execution.backend_coeffs,
        retention=retained,
    )
    run_te = basis["te"]
    run_tm = basis["tm"]
    compute_dtype = np.dtype(run_te.compute_dtype)
    accum_dtype = np.dtype(run_te.accum_dtype)

    mixed_initial_coeffs: np.ndarray | None = None
    mixed_coeffs: np.ndarray | None = None
    local_mixed_coeffs: Any | None = None
    if sim.config.periodic is not None or include_farfield:
        mixed_initial_coeffs = np.asarray(
            a_te * solved.initial_coeffs["te"] + a_tm * solved.initial_coeffs["tm"],
            dtype=accum_dtype,
        )
        mixed_coeffs = np.asarray(
            a_te * solved.coeffs["te"] + a_tm * solved.coeffs["tm"],
            dtype=compute_dtype,
        )
        local_mixed_coeffs = mixed_coeffs
        if execution.backend_coeffs is not None:
            local_mixed_coeffs = (
                a_te * execution.backend_coeffs["te"] + a_tm * execution.backend_coeffs["tm"]
            )

    mixed_periodic: PeriodicFarFieldPayload | None = None
    mixed_power: PowerBalance | None = None
    mixed_cross_sections: CrossSectionBalance | None = None
    mixed_forward: dict[str, float] | None = None
    mixed_backward: dict[str, float] | None = None

    if sim.config.periodic is not None:
        # Periodic diffraction orders are the canonical far-field result and
        # are always produced by periodic postprocessing, matching `run()`.
        mixed_farfield = empty_farfield_patterns(compute_dtype)
        if not isinstance(source, PlaneWave):
            raise NotImplementedError(
                "Periodic polarization postprocessing currently requires PlaneWave excitation."
            )
        if run_te.periodic is None or run_tm.periodic is None:
            raise RuntimeError("Periodic basis channels are missing order payloads.")
        mixed_periodic = mix_periodic_farfield_payloads(
            run_te.periodic,
            run_tm.periodic,
            source=source,
            a_first=a_te,
            a_second=a_tm,
            k=run_te.k,
            n_medium=sim.config.n_medium,
        )
        if mixed_initial_coeffs is None or local_mixed_coeffs is None:
            raise RuntimeError("Periodic mixed-channel coefficients were not prepared.")
        local_components = _local_power_balance_generic_route(
            sim,
            initial_coeffs=mixed_initial_coeffs,
            coeffs=local_mixed_coeffs,
            k0=run_te.k0,
            accum_dtype=accum_dtype,
        )
        if local_components is not None:
            mixed_periodic = replace(
                mixed_periodic,
                power=replace(
                    mixed_periodic.power,
                    local_absorbed_power=local_components.local_absorbed_power,
                    local_absorbed_power_per_particle=(
                        local_components.local_absorbed_power_per_particle
                    ),
                ),
            )
        mixed_power = mixed_periodic.power
    else:
        mixed_farfield = (
            mix_farfield_patterns(
                run_te.farfield,
                run_tm.farfield,
                a_te=a_te,
                a_tm=a_tm,
                dtype=compute_dtype,
            )
            if include_farfield
            else empty_farfield_patterns(compute_dtype)
        )
        if include_farfield:
            if mixed_initial_coeffs is None or mixed_coeffs is None or local_mixed_coeffs is None:
                raise RuntimeError("Mixed-channel coefficients were not prepared.")
            (
                mixed_power,
                mixed_cross_sections,
                mixed_forward,
                mixed_backward,
            ) = _build_channel_diagnostics(
                sim,
                source=source,
                initial_coeffs=mixed_initial_coeffs,
                coeffs=mixed_coeffs,
                local_initial_coeffs=mixed_initial_coeffs,
                local_coeffs=local_mixed_coeffs,
                farfield=mixed_farfield,
                k=run_te.k,
                k0=run_te.k0,
                accum_dtype=accum_dtype,
            )

    return PolarizationResult(
        source=source,
        te=run_te,
        tm=run_tm,
        solver_result=basis.solver_result,
        unpolarized=_unpolarized_diagnostics(run_te, run_tm),
        _mixed_payload=_CoherentChannelPayload(
            farfield=mixed_farfield,
            power=mixed_power,
            cross_sections=mixed_cross_sections,
            decomposition_forward=mixed_forward,
            decomposition_backward=mixed_backward,
            periodic=mixed_periodic,
        ),
    )


__all__ = [
    "build_single_channel_result",
    "mix_farfield_patterns",
    "postprocess_sources_impl",
    "run_impl",
    "run_polarizations_impl",
]
