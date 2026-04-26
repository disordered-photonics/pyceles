"""Postprocessing-phase helpers for solved simulation channels."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.particles import LayeredSphere, Particle, Sphere, Spheroid
from pyceles.core.sources import (
    DipoleCollection,
    DipoleSource,
    JonesPolarizedSource,
    PlaneWave,
    Source,
)
from pyceles.postprocessing.farfield import (
    FarFieldPatterns,
    compute_far_field_patterns,
    finite_beam_power_fractions,
    local_absorbed_power_components_from_exciting,
    local_absorption_cross_section_from_exciting,
    plane_wave_cross_sections,
    pwp_power_decomposition,
)

from .config import validate_angular_grid_pair
from .results import (
    MultiSourceSimulationResult,
    SimulationResult,
    SolvedSourcesResult,
    avg_numeric_dict,
    empty_farfield_patterns,
    single_rhs_result_from_multi,
)

if TYPE_CHECKING:
    from .workflow import Simulation


def _is_numerically_lossless_cluster(
    particles: tuple[Particle, ...], *, imag_tol: float = 0.0
) -> bool:
    """Return True when all particle materials are numerically lossless."""
    tol = float(imag_tol)
    for particle in particles:
        if isinstance(particle, (Sphere, Spheroid)):
            if abs(complex(particle.refractive_index).imag) > tol:
                return False
            continue
        if isinstance(particle, LayeredSphere):
            if any(abs(complex(n).imag) > tol for n in particle.layer_refractive_indices):
                return False
            continue
        # Unknown particle families default to "possibly lossy" so we do not
        # skip local-absorption evaluation incorrectly.
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
    use_cupy = bool(
        str(sim.config.operator_backend).lower() == "cupy"
        or is_cupy_array(coeffs)
        or is_cupy_array(initial_coeffs)
    )
    if use_cupy:
        cupy, _ = import_cupy()
        # Solved coefficients may arrive through a private CuPy handoff, while
        # incident coefficients are public host payloads. Upload the latter only
        # when a local-dissipation route actually needs them.
        x_flat = cupy.asarray(coeffs, dtype=accum_dtype).reshape(-1)
        b_flat = cupy.asarray(initial_coeffs, dtype=accum_dtype).reshape(-1)
        wx = prepared.apply_W(x_flat)
        return b_flat + wx, x_flat
    x_flat = np.asarray(coeffs, dtype=accum_dtype).reshape(-1)
    wx = prepared.apply_W(x_flat)
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


def _local_absorbed_power_components_generic_route(
    sim: Simulation,
    *,
    initial_coeffs: Any,
    coeffs: Any,
    k0: float,
    accum_dtype: np.dtype,
) -> dict[str, float | np.ndarray] | None:
    """Compute local absorbed-power diagnostics via the generic `e=b+W x` route."""
    payload = _build_exciting_scattered_flat_generic_route(
        sim,
        initial_coeffs=initial_coeffs,
        coeffs=coeffs,
        accum_dtype=accum_dtype,
    )
    if payload is None:
        return None
    e_flat, x_flat = payload
    return local_absorbed_power_components_from_exciting(
        e_flat,
        x_flat,
        k0=float(k0),
        n_medium=sim.config.n_medium,
        n_particles=int(sim.positions.shape[0]),
        nmodes_per_particle=int(n_modes(sim.config.lmax)),
    )


def _as_float_scalar(value: float | np.ndarray) -> float:
    """Convert a scalar-like numeric payload to Python float."""
    arr = np.asarray(value, dtype=np.float64).reshape(())
    return float(arr)


def _mix_pwp_dict(
    p1: dict,
    p2: dict,
    *,
    a1: complex,
    a2: complex,
    dtype: np.dtype,
) -> dict:
    out = dict(p1)
    out["coeff"] = np.asarray(a1 * p1["coeff"] + a2 * p2["coeff"], dtype=np.dtype(dtype))
    return out


def _mix_optional_pwp_dict(
    p1: dict | None,
    p2: dict | None,
    *,
    a1: complex,
    a2: complex,
    dtype: np.dtype,
) -> dict | None:
    if p1 is None or p2 is None:
        return None
    return _mix_pwp_dict(p1, p2, a1=a1, a2=a2, dtype=dtype)


def mix_farfield_patterns(
    ff_te: FarFieldPatterns,
    ff_tm: FarFieldPatterns,
    *,
    a_te: complex,
    a_tm: complex,
    dtype: np.dtype,
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
    dict[str, float | np.ndarray] | None,
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Build optional power/cross-section diagnostics for one channel payload.

    This centralizes the physics-policy branching so the single-channel path
    and the mixed-from-basis path cannot silently drift apart when we extend
    diagnostics in the future.
    """
    cfg = sim.config
    ns = int(sim.positions.shape[0])
    local_b = initial_coeffs if local_initial_coeffs is None else local_initial_coeffs
    local_x = coeffs if local_coeffs is None else local_coeffs

    power: dict[str, float | np.ndarray] | None = None
    cross_sections: dict[str, float] | None = None
    decomposition_forward: dict[str, float] | None = None
    decomposition_backward: dict[str, float] | None = None

    if isinstance(source, PlaneWave):
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
        cross_sections = plane_wave_cross_sections(
            source,
            initial_coeffs,
            coeffs,
            k0=k0,
            n_medium=cfg.n_medium,
            scattered_pwp_te=farfield.scattered_te,
            scattered_pwp_tm=farfield.scattered_tm,
            local_absorption=c_abs_local,
        )
        return power, cross_sections, decomposition_forward, decomposition_backward

    if (
        farfield.initial_te is not None
        and farfield.initial_tm is not None
        and source.has_finite_incident_power()
    ):
        p_abs_diag: dict[str, float | np.ndarray] | None
        if _is_numerically_lossless_cluster(sim.particles):
            p_abs_diag = {
                "P_abs_local": 0.0,
                "P_abs_local_particles": np.zeros((ns,), dtype=np.float64),
            }
        else:
            p_abs_diag = _local_absorbed_power_components_generic_route(
                sim,
                initial_coeffs=local_b,
                coeffs=local_x,
                k0=float(k0),
                accum_dtype=accum_dtype,
            )
        power_base = finite_beam_power_fractions(
            source,
            farfield.initial_te,
            farfield.initial_tm,
            farfield.scattered_te,
            farfield.scattered_tm,
            k0=k0,
            k_medium=k,
            local_absorbed_power=(
                None if p_abs_diag is None else _as_float_scalar(p_abs_diag["P_abs_local"])
            ),
        )
        power = cast(dict[str, float | np.ndarray], dict(power_base))
        if p_abs_diag is not None:
            p_abs_local_particles = np.asarray(
                p_abs_diag.get("P_abs_local_particles", np.zeros((ns,), dtype=np.float64)),
                dtype=np.float64,
            ).reshape(ns)
            power["P_abs_local_particles"] = p_abs_local_particles
            power["A_local_particles"] = p_abs_local_particles / float(power["P_initial"])
        decomposition_forward = pwp_power_decomposition(
            direction="forward",
            initial_pwp_te=farfield.initial_te,
            initial_pwp_tm=farfield.initial_tm,
            scattered_pwp_te=farfield.scattered_te,
            scattered_pwp_tm=farfield.scattered_tm,
            k0=k0,
            k_medium=k,
            source=source,
        )
        decomposition_backward = pwp_power_decomposition(
            direction="backward",
            initial_pwp_te=farfield.initial_te,
            initial_pwp_tm=farfield.initial_tm,
            scattered_pwp_te=farfield.scattered_te,
            scattered_pwp_tm=farfield.scattered_tm,
            k0=k0,
            k_medium=k,
            source=source,
        )
        return power, cross_sections, decomposition_forward, decomposition_backward

    if isinstance(source, (DipoleSource, DipoleCollection)):
        if _is_numerically_lossless_cluster(sim.particles):
            power = {
                "P_abs_local": 0.0,
                "P_abs_local_particles": np.zeros((ns,), dtype=np.float64),
            }
        else:
            p_abs_diag = _local_absorbed_power_components_generic_route(
                sim,
                initial_coeffs=local_b,
                coeffs=local_x,
                k0=float(k0),
                accum_dtype=accum_dtype,
            )
            if p_abs_diag is not None:
                power = {
                    "P_abs_local": _as_float_scalar(p_abs_diag["P_abs_local"]),
                    "P_abs_local_particles": np.asarray(
                        p_abs_diag.get(
                            "P_abs_local_particles",
                            np.zeros((ns,), dtype=np.float64),
                        ),
                        dtype=np.float64,
                    ).reshape(ns),
                }
    return power, cross_sections, decomposition_forward, decomposition_backward


def _basis_channel_payloads(
    run_te: SimulationResult,
    run_tm: SimulationResult,
) -> tuple[
    dict[str, dict[str, float | np.ndarray]] | None,
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, float]] | None,
]:
    """Collect basis-channel diagnostics and incoherent unpolarized averages."""
    power_basis = None
    cross_sections_basis = None
    decomposition_forward_basis = None
    decomposition_backward_basis = None
    unpolarized = None

    if run_te.power is not None and run_tm.power is not None:
        power_basis = {"te": run_te.power, "tm": run_tm.power}
        unpolarized = dict(unpolarized or {})
        unpolarized["power"] = avg_numeric_dict(run_te.power, run_tm.power)
    if run_te.cross_sections is not None and run_tm.cross_sections is not None:
        cross_sections_basis = {"te": run_te.cross_sections, "tm": run_tm.cross_sections}
        unpolarized = dict(unpolarized or {})
        unpolarized["cross_sections"] = avg_numeric_dict(
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
    return (
        power_basis,
        cross_sections_basis,
        unpolarized,
        decomposition_forward_basis,
        decomposition_backward_basis,
    )


def _assemble_simulation_result(
    sim: Simulation,
    *,
    source: Source,
    initial_coeffs: np.ndarray,
    rhs: np.ndarray,
    coeffs: np.ndarray,
    solver_result,
    k: float,
    k0: float,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    farfield: FarFieldPatterns,
    power: dict[str, float | np.ndarray] | None,
    cross_sections: dict[str, float] | None,
    decomposition_forward: dict[str, float] | None,
    decomposition_backward: dict[str, float] | None,
    initial_coeffs_basis: dict[str, np.ndarray] | None = None,
    coeffs_basis: dict[str, np.ndarray] | None = None,
    solver_result_basis=None,
    farfield_basis: dict[str, FarFieldPatterns] | None = None,
    power_basis: dict[str, dict[str, float | np.ndarray]] | None = None,
    cross_sections_basis: dict[str, dict[str, float]] | None = None,
    unpolarized: dict[str, dict[str, float]] | None = None,
    decomposition_forward_basis: dict[str, dict[str, float]] | None = None,
    decomposition_backward_basis: dict[str, dict[str, float]] | None = None,
    polarization_jones: tuple[complex, complex] | None = None,
) -> SimulationResult:
    """Assemble the canonical completed-run payload from solved coefficients."""
    cfg = sim.config
    ns = int(sim.positions.shape[0])
    nm = int(n_modes(cfg.lmax))
    pol_jones = (
        source.jones_coefficients()
        if polarization_jones is None and isinstance(source, JonesPolarizedSource)
        else polarization_jones
    )
    config_out = cfg if cfg.source is source else replace(cfg, source=source)
    return SimulationResult(
        config=config_out,
        particles=sim.particles,
        k=k,
        k0=k0,
        coeffs=np.asarray(coeffs),
        rhs=np.asarray(rhs).reshape(ns, nm),
        initial_coeffs=np.asarray(initial_coeffs),
        initial_coeffs_basis=initial_coeffs_basis,
        coeffs_basis=coeffs_basis,
        solver_result=solver_result,
        solver_result_basis=solver_result_basis,
        farfield=farfield,
        farfield_basis=farfield_basis,
        power=power,
        power_basis=power_basis,
        cross_sections=cross_sections,
        cross_sections_basis=cross_sections_basis,
        unpolarized=unpolarized,
        decomposition_forward=decomposition_forward,
        decomposition_backward=decomposition_backward,
        decomposition_forward_basis=decomposition_forward_basis,
        decomposition_backward_basis=decomposition_backward_basis,
        polarization_jones=pol_jones,
        compute_dtype=str(compute_dtype),
        accum_dtype=str(accum_dtype),
    )


def build_single_channel_result(
    sim: Simulation,
    *,
    source: Source,
    initial_coeffs: np.ndarray,
    rhs_flat: np.ndarray,
    coeffs: np.ndarray,
    solver_result,
    backend_coeffs: Any | None = None,
    k: float,
    k0: float,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    farfield_polar_angles: np.ndarray,
    farfield_azimuthal_angles: np.ndarray,
    include_farfield: bool,
) -> SimulationResult:
    """Assemble one channel `SimulationResult` from solved coefficients."""
    cfg = sim.config
    positions = sim.positions
    Ns = positions.shape[0]
    Nm = n_modes(cfg.lmax)

    power: dict[str, float | np.ndarray] | None = None
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

    return _assemble_simulation_result(
        sim,
        source=source,
        k=k,
        k0=k0,
        coeffs=coeffs,
        rhs=np.asarray(rhs_flat).reshape(Ns, Nm),
        initial_coeffs=initial_coeffs,
        solver_result=solver_result,
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
    solved: SolvedSourcesResult,
    *,
    include_farfield: bool = True,
    farfield_polar_angles: np.ndarray | None = None,
    farfield_azimuthal_angles: np.ndarray | None = None,
) -> MultiSourceSimulationResult:
    """Postprocess solved channels into per-channel `SimulationResult` payloads."""
    cfg = sim.config
    if cfg.periodic is not None:
        raise NotImplementedError(
            "Periodic postprocessing outputs are not implemented yet. "
            "Diffraction-order and R/T/A result payloads are not available yet."
        )
    labels = tuple(solved.labels)
    n_channels = len(labels)

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

    Ns = sim.positions.shape[0]
    Nm = n_modes(cfg.lmax)
    compute_dtype = np.dtype(solved.compute_dtype)
    accum_dtype = np.dtype(solved.accum_dtype)

    runs: dict[str, SimulationResult] = {}
    backend_handoff = sim._solve_backend_handoffs.pop(id(solved), {})
    backend_coeffs_by_label = backend_handoff.get("coeffs")
    for j, label in enumerate(labels):
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
        solver_col = (
            solved.solver_result
            if n_channels == 1
            else single_rhs_result_from_multi(solved.solver_result, j)
        )
        backend_coeffs = None if backend_coeffs_by_label is None else backend_coeffs_by_label[label]
        runs[label] = build_single_channel_result(
            sim,
            source=solved.sources[label],
            initial_coeffs=solved.initial_coeffs[label],
            rhs_flat=np.asarray(rhs_col, dtype=accum_dtype).reshape(Ns * Nm),
            coeffs=x_col,
            solver_result=solver_col,
            backend_coeffs=backend_coeffs,
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


def run_impl(sim: Simulation, *, include_farfield: bool = True) -> SimulationResult:
    """Run one simulation for `config.source`."""
    cfg = sim.config
    source = sim._validate_ready_to_run()

    if not bool(cfg.solve_polarization_basis):
        solved = sim._solve_sources_for_immediate_postprocess({"mixed": source})
        try:
            multi = sim.postprocess_sources(solved, include_farfield=include_farfield)
            return multi["mixed"]
        finally:
            sim._solve_backend_handoffs.pop(id(solved), None)
    if not isinstance(source, JonesPolarizedSource):
        raise ValueError(
            "`solve_polarization_basis=True` is only defined for TE/TM polarization sources "
            "that expose Jones metadata and `with_polarization('TE'/'TM')`."
        )
    a_te, a_tm = source.jones_coefficients()
    src_te = source.with_polarization("TE")
    src_tm = source.with_polarization("TM")

    basis_solved = sim._solve_sources_for_immediate_postprocess({"te": src_te, "tm": src_tm})
    try:
        basis_backend_handoff = sim._solve_backend_handoffs.get(id(basis_solved), {})
        basis_multi = sim.postprocess_sources(basis_solved, include_farfield=include_farfield)
    finally:
        sim._solve_backend_handoffs.pop(id(basis_solved), None)
    run_te = basis_multi["te"]
    run_tm = basis_multi["tm"]
    compute_dtype = np.dtype(run_te.compute_dtype)
    accum_dtype = np.dtype(run_te.accum_dtype)

    b_te = basis_solved.initial_coeffs["te"]
    b_tm = basis_solved.initial_coeffs["tm"]
    rhs_te = basis_solved.rhs["te"]
    rhs_tm = basis_solved.rhs["tm"]
    x_te = basis_solved.coeffs["te"]
    x_tm = basis_solved.coeffs["tm"]

    b = a_te * b_te + a_tm * b_tm
    rhs = a_te * rhs_te + a_tm * rhs_tm
    x = a_te * x_te + a_tm * x_tm
    local_b: Any = b
    local_x: Any = x
    backend_coeffs_by_label = basis_backend_handoff.get("coeffs")
    if backend_coeffs_by_label is not None:
        local_x = a_te * backend_coeffs_by_label["te"] + a_tm * backend_coeffs_by_label["tm"]

    ff_basis = {"te": run_te.farfield, "tm": run_tm.farfield}
    ff = (
        mix_farfield_patterns(
            ff_basis["te"], ff_basis["tm"], a_te=a_te, a_tm=a_tm, dtype=compute_dtype
        )
        if include_farfield
        else empty_farfield_patterns(compute_dtype)
    )

    power: dict[str, float | np.ndarray] | None = None
    cross_sections = None
    decomposition_forward = None
    decomposition_backward = None
    if include_farfield:
        power, cross_sections, decomposition_forward, decomposition_backward = (
            _build_channel_diagnostics(
                sim,
                source=source,
                initial_coeffs=b,
                coeffs=x,
                local_initial_coeffs=local_b,
                local_coeffs=local_x,
                farfield=ff,
                k=run_te.k,
                k0=run_te.k0,
                accum_dtype=accum_dtype,
            )
        )

    (
        power_basis,
        cross_sections_basis,
        unpolarized,
        decomposition_forward_basis,
        decomposition_backward_basis,
    ) = _basis_channel_payloads(run_te, run_tm)

    return _assemble_simulation_result(
        sim,
        source=source,
        k=run_te.k,
        k0=run_te.k0,
        coeffs=x,
        rhs=rhs,
        initial_coeffs=b,
        solver_result=basis_solved.solver_result,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        farfield=ff,
        power=power,
        cross_sections=cross_sections,
        decomposition_forward=decomposition_forward,
        decomposition_backward=decomposition_backward,
        initial_coeffs_basis={"te": b_te, "tm": b_tm},
        coeffs_basis={"te": x_te, "tm": x_tm},
        solver_result_basis=basis_solved.solver_result,
        farfield_basis=ff_basis,
        power_basis=power_basis,
        cross_sections_basis=cross_sections_basis,
        unpolarized=unpolarized,
        decomposition_forward_basis=decomposition_forward_basis,
        decomposition_backward_basis=decomposition_backward_basis,
        polarization_jones=(a_te, a_tm),
    )


__all__ = [
    "build_single_channel_result",
    "mix_farfield_patterns",
    "postprocess_sources_impl",
    "run_impl",
]
