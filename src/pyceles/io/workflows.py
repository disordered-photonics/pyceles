from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.sources import BlochPlaneWave, PlaneWave
from pyceles.core.sources.plane_wave import _plane_wave_geometry

from .hdf5 import (
    load_far_field_h5,
    load_geometry_h5,
    load_mapping_h5,
    load_near_field_components_h5,
    load_periodic_h5,
    load_solution_h5,
    save_far_field_h5,
    save_geometry_h5,
    save_mapping_h5,
    save_near_field_components_h5,
    save_periodic_h5,
    save_solution_h5,
)

if TYPE_CHECKING:
    from pyceles.postprocessing.nearfield.slice import NearFieldSlice
    from pyceles.simulation import ChannelResult


def save_simulation_h5(run: ChannelResult, near_field: NearFieldSlice, out_h5: str | Path) -> Path:
    """Persist one explicit channel, its fields, and scalar diagnostics.

    Multi-source and polarization envelopes are intentionally not serialized as
    disguised single runs. Save the desired channel explicitly, for example
    ``save_simulation_h5(result["left"], ...)`` or
    ``save_simulation_h5(polarized.mixed, ...)``. Periodic runs include a
    separate configuration payload; order and power results are stored only
    when the channel provides them.
    """
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    save_geometry_h5(
        out_h5,
        particles=run.particles,
        n_medium=run.config.n_medium,
        wavelength=run.config.wavelength,
        lmax=run.config.lmax,
        mode="w",
    )
    solver_result = getattr(run, "solver_result", None)
    solver_attrs: dict[str, object] = {}
    residual_history = None
    info = None
    if solver_result is not None:
        solver_attrs.update(
            solver=solver_result.method,
            relative_residual=solver_result.relative_residual,
            iterations=solver_result.iterations,
        )
        if solver_result.converged_reason is not None:
            reason = np.asarray(solver_result.converged_reason, dtype=object)
            solver_attrs["converged_reason"] = (
                str(reason.item()) if reason.ndim == 0 else np.asarray(reason, dtype="S")
            )
        if solver_result.stopping_rule is not None:
            solver_attrs["stopping_rule"] = solver_result.stopping_rule
        residual_history = (
            solver_result.residual_history
            if isinstance(solver_result.residual_history, np.ndarray)
            else None
        )
        info = solver_result.info
    save_solution_h5(
        out_h5,
        coeffs=run.coeffs,
        rhs=run.rhs,
        initial_coeffs=run.initial_coeffs,
        residual_history=residual_history,
        info=info,
        attrs=solver_attrs,
    )
    save_near_field_components_h5(
        out_h5,
        X=near_field.axis_0,
        Z=near_field.axis_1,
        fields={name: {"E": pair[0], "H": pair[1]} for name, pair in near_field.field_maps.items()},
        inside=near_field.inside,
        attrs={
            "plane": near_field.plane,
            "plane_value": near_field.plane_value,
            "axis_0_label": near_field.axis_0_label,
            "axis_1_label": near_field.axis_1_label,
        },
    )

    diagnostics: dict[str, object] = {}
    periodic_configuration: dict[str, object] | None = None
    periodic_spec = run.config.periodic
    if periodic_spec is not None:
        if not isinstance(run.source, (PlaneWave, BlochPlaneWave)):
            raise TypeError(
                "Periodic simulation serialization requires a PlaneWave or BlochPlaneWave source."
            )
        periodic_configuration = {
            "method": periodic_spec.options.method,
            "lattice_a1": periodic_spec.lattice.a1,
            "lattice_a2": periodic_spec.lattice.a2,
            "unit_cell_area": periodic_spec.lattice.area,
            "k_medium": run.k,
            "incident_k_parallel": plane_wave_k_parallel(run.source),
            "options": asdict(periodic_spec.options),
            "order_amplitudes_saved": run.periodic is not None,
            "power_diagnostics_available": run.power is not None,
            "incident_source_type": type(run.source).__name__,
        }
        geometry = _plane_wave_geometry(run.source, k=run.k)
        if isinstance(run.source, PlaneWave):
            periodic_configuration.update(
                incident_polar_angle=run.source.polar_angle,
                incident_azimuthal_angle=run.source.azimuthal_angle,
            )
        periodic_configuration.update(
            incident_direction="up" if geometry.direction_sign > 0 else "down",
            incident_reference_point=geometry.reference_point,
            incident_amplitude=run.source.amplitude,
            incident_jones_coefficients=run.source.jones_coefficients(),
        )
    if periodic_spec is None:
        ff = run.farfield
        source_beta, source_alpha = run.config.source_angular_grids()
        farfield_beta, farfield_alpha = run.config.farfield_angular_grids()
        save_far_field_h5(
            out_h5,
            farfield=ff,
            attrs={
                "k_medium": run.k,
                "source_beta_points": int(source_beta.size),
                "source_alpha_points": int(source_alpha.size),
                "farfield_beta_points": int(farfield_beta.size),
                "farfield_alpha_points": int(farfield_alpha.size),
                "source_farfield_grid_equal": bool(
                    np.array_equal(source_beta, farfield_beta)
                    and np.array_equal(source_alpha, farfield_alpha)
                ),
            },
        )

        same_source_farfield_grids = bool(
            np.array_equal(source_beta, farfield_beta)
            and np.array_equal(source_alpha, farfield_alpha)
        )
        angular_grids: dict[str, object] = {
            "source_farfield_grid_equal": same_source_farfield_grids,
        }
        if same_source_farfield_grids:
            # Common path: one shared grid is enough for both source projection and far-field output.
            angular_grids["beta"] = np.asarray(source_beta, dtype=float)
            angular_grids["alpha"] = np.asarray(source_alpha, dtype=float)
        else:
            angular_grids["source_beta"] = np.asarray(source_beta, dtype=float)
            angular_grids["source_alpha"] = np.asarray(source_alpha, dtype=float)
            angular_grids["farfield_beta"] = np.asarray(farfield_beta, dtype=float)
            angular_grids["farfield_alpha"] = np.asarray(farfield_alpha, dtype=float)
        diagnostics["angular_grids"] = angular_grids
    elif run.periodic is not None:
        # Periodic runs store order-resolved observables under `periodic`.
        # `run.farfield` intentionally stays empty in this workflow.
        save_periodic_h5(
            out_h5,
            periodic=run.periodic,
            group="periodic",
            mode="a",
            include_power=False,
        )
        diagnostics["periodic"] = {
            "order_count": int(np.asarray(run.periodic.order_mn).shape[0]),
            "propagating_order_count": int(
                np.count_nonzero(np.asarray(run.periodic.order_propagating, dtype=bool))
            ),
            "incident_flux": float(run.periodic.incident_flux),
        }
    if run.power is not None:
        diagnostics["power"] = run.power.to_mapping()
    if run.cross_sections is not None:
        diagnostics["cross_sections"] = run.cross_sections.to_mapping()
    if run.decomposition_forward is not None:
        diagnostics["decomposition_forward"] = run.decomposition_forward
    if run.decomposition_backward is not None:
        diagnostics["decomposition_backward"] = run.decomposition_backward
    save_mapping_h5(out_h5, mapping=diagnostics, group="diagnostics", mode="a")
    if periodic_configuration is not None:
        save_mapping_h5(
            out_h5,
            mapping=periodic_configuration,
            group="periodic_configuration",
            mode="a",
        )
    return out_h5


def load_simulation_h5(path: str | Path) -> dict[str, object]:
    """Load saved simulation artifacts from HDF5 into a plain dictionary.

    This is a lightweight loader for analysis/post-processing workflows that do
    not need to reconstruct a full `SimulationResult` instance. Periodic runs
    include order data and a separate configuration payload when available.
    Absorbing-host runs retain the periodic configuration while omitting
    undefined order/power results.
    """
    p = Path(path)
    out: dict[str, object] = {
        "geometry": load_geometry_h5(p, group="geometry"),
        "solution": load_solution_h5(p, group="solution"),
    }

    with h5py.File(str(p), "r") as h5:
        has_near_field_components = "near_field_components" in h5
        has_far_field = "far_field" in h5
        has_periodic = "periodic" in h5
        has_periodic_configuration = "periodic_configuration" in h5
        has_diagnostics = "diagnostics" in h5

    if has_near_field_components:
        out["near_field_components"] = load_near_field_components_h5(
            p, group="near_field_components"
        )
    if has_far_field:
        out["far_field"] = load_far_field_h5(p, group="far_field")
    if has_periodic:
        out["periodic"] = load_periodic_h5(p, group="periodic")
    if has_periodic_configuration:
        out["periodic_configuration"] = load_mapping_h5(p, group="periodic_configuration")
    if has_diagnostics:
        out["diagnostics"] = load_mapping_h5(p, group="diagnostics")

    return out
