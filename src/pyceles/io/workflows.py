from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .hdf5 import (
    load_far_field_h5,
    load_geometry_h5,
    load_mapping_h5,
    load_near_field_components_h5,
    load_solution_h5,
    save_far_field_h5,
    save_geometry_h5,
    save_mapping_h5,
    save_near_field_components_h5,
    save_solution_h5,
)

if TYPE_CHECKING:
    from pyceles.postprocessing.nearfield.slice import NearFieldSlice
    from pyceles.simulation import SimulationResult


def save_simulation_h5(
    run: SimulationResult, near_field: NearFieldSlice, out_h5: str | Path
) -> Path:
    """Persist geometry, solver outputs, near field, far field, and diagnostics.

    If polarization-basis channels are present (for example from
    `SimulationConfig(solve_polarization_basis=True)` + `Simulation.run()`
    convenience mode),
    this workflow also stores:
    - basis coefficient solutions (`solution_basis/te`, `solution_basis/tm`)
    - basis far-field families (`far_field_basis/te`, `far_field_basis/tm`)
    - basis and unpolarized diagnostics under `diagnostics`.

    For `Simulation.solve_sources(...)` + `Simulation.postprocess_sources(...)`,
    save each channel result
    individually (for example `save_simulation_h5(multi["te"], ...)`).
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
    save_solution_h5(
        out_h5,
        coeffs=run.coeffs,
        rhs=run.rhs,
        initial_coeffs=run.initial_coeffs,
        residual_history=(
            run.solver_result.residual_history
            if isinstance(run.solver_result.residual_history, np.ndarray)
            else None
        ),
        info=run.solver_result.info,
        attrs={
            "solver": run.solver_result.method,
            "relative_residual": run.solver_result.relative_residual,
            "iterations": run.solver_result.iterations,
        },
    )
    if run.coeffs_basis is not None and run.initial_coeffs_basis is not None:
        for pol in ("te", "tm"):
            if pol not in run.coeffs_basis or pol not in run.initial_coeffs_basis:
                continue
            save_solution_h5(
                out_h5,
                group=f"solution_basis/{pol}",
                coeffs=run.coeffs_basis[pol],
                initial_coeffs=run.initial_coeffs_basis[pol],
                mode="a",
                attrs={
                    "polarization_channel": pol,
                },
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

    ff = run.farfield
    source_beta, source_alpha = run.config.source_angular_grids()
    farfield_beta, farfield_alpha = run.config.farfield_angular_grids()
    k_medium = 2.0 * np.pi / float(run.config.wavelength) * float(np.real(run.config.n_medium))
    patterns = {"scattered": {"te": ff.scattered_te, "tm": ff.scattered_tm}}
    if ff.initial_te is not None and ff.initial_tm is not None:
        patterns["initial"] = {"te": ff.initial_te, "tm": ff.initial_tm}
    if ff.total_te is not None and ff.total_tm is not None:
        patterns["total"] = {"te": ff.total_te, "tm": ff.total_tm}
    save_far_field_h5(
        out_h5,
        patterns=patterns,
        attrs={
            "k_medium": float(k_medium),
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

    if run.farfield_basis is not None:
        for pol, ff_pol in run.farfield_basis.items():
            patt_pol = {"scattered": {"te": ff_pol.scattered_te, "tm": ff_pol.scattered_tm}}
            if ff_pol.initial_te is not None and ff_pol.initial_tm is not None:
                patt_pol["initial"] = {"te": ff_pol.initial_te, "tm": ff_pol.initial_tm}
            if ff_pol.total_te is not None and ff_pol.total_tm is not None:
                patt_pol["total"] = {"te": ff_pol.total_te, "tm": ff_pol.total_tm}
            save_far_field_h5(
                out_h5,
                patterns=patt_pol,
                group=f"far_field_basis/{pol}",
                mode="a",
                attrs={
                    "polarization_channel": pol,
                    "k_medium": float(k_medium),
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
        np.array_equal(source_beta, farfield_beta) and np.array_equal(source_alpha, farfield_alpha)
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

    diagnostics: dict[str, object] = {"angular_grids": angular_grids}
    if run.polarization_jones is not None:
        diagnostics["polarization_jones"] = {
            "a_te_real": float(np.real(run.polarization_jones[0])),
            "a_te_imag": float(np.imag(run.polarization_jones[0])),
            "a_tm_real": float(np.real(run.polarization_jones[1])),
            "a_tm_imag": float(np.imag(run.polarization_jones[1])),
        }
    if run.power is not None:
        diagnostics["power"] = run.power
    if run.cross_sections is not None:
        diagnostics["cross_sections"] = run.cross_sections
    if run.decomposition_forward is not None:
        diagnostics["decomposition_forward"] = run.decomposition_forward
    if run.decomposition_backward is not None:
        diagnostics["decomposition_backward"] = run.decomposition_backward
    if run.power_basis is not None:
        diagnostics["power_basis"] = run.power_basis
    if run.cross_sections_basis is not None:
        diagnostics["cross_sections_basis"] = run.cross_sections_basis
    if run.decomposition_forward_basis is not None:
        diagnostics["decomposition_forward_basis"] = run.decomposition_forward_basis
    if run.decomposition_backward_basis is not None:
        diagnostics["decomposition_backward_basis"] = run.decomposition_backward_basis
    if run.unpolarized is not None:
        diagnostics["unpolarized"] = run.unpolarized
    save_mapping_h5(out_h5, mapping=diagnostics, group="diagnostics", mode="a")
    return out_h5


def load_simulation_h5(path: str | Path) -> dict[str, object]:
    """Load saved simulation artifacts from HDF5 into a plain dictionary.

    This is a lightweight loader for analysis/post-processing workflows that do
    not need to reconstruct a full `SimulationResult` instance.
    """
    p = Path(path)
    out: dict[str, object] = {
        "geometry": load_geometry_h5(p, group="geometry"),
        "solution": load_solution_h5(p, group="solution"),
    }

    with h5py.File(str(p), "r") as h5:
        has_near_field_components = "near_field_components" in h5
        has_far_field = "far_field" in h5
        has_diagnostics = "diagnostics" in h5
        has_solution_basis = "solution_basis" in h5
        has_far_field_basis = "far_field_basis" in h5

    if has_near_field_components:
        out["near_field_components"] = load_near_field_components_h5(
            p, group="near_field_components"
        )
    if has_far_field:
        out["far_field"] = load_far_field_h5(p, group="far_field")
    if has_diagnostics:
        out["diagnostics"] = load_mapping_h5(p, group="diagnostics")

    # Optional basis groups.
    basis_solution: dict[str, object] = {}
    basis_farfield: dict[str, object] = {}
    with h5py.File(str(p), "r") as h5:
        if has_solution_basis:
            for pol in ("te", "tm"):
                group = f"solution_basis/{pol}"
                if group in h5:
                    basis_solution[pol] = load_solution_h5(p, group=group)
        if has_far_field_basis:
            for pol in ("te", "tm"):
                group = f"far_field_basis/{pol}"
                if group in h5:
                    basis_farfield[pol] = load_far_field_h5(p, group=group)
    if basis_solution:
        out["solution_basis"] = basis_solution
    if basis_farfield:
        out["far_field_basis"] = basis_farfield

    return out
