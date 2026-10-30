"""Shared rectangular periodic outgoing-to-regular reception."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles.core.geometry_bounds import coincident_point_mask
from pyceles.core.indexing import n_modes
from pyceles.core.lattice import canonical_cell_points

from .ewald import (
    EwaldShellWorkspace,
    _add_self_correction_to_batched_sums,
    ewald_structural_sums_2d_batch,
    resolve_ewald_eta,
)
from .structural import (
    direct_structural_sums_2d,
    rectangular_projection_data,
)
from .types import PeriodicSpec


def _check_reception_origins(
    points: np.ndarray, positions: np.ndarray, coefficients: np.ndarray
) -> None:
    """Check active origins once on host, using exact canonical coordinates."""
    active_positions = positions[np.any(coefficients != 0, axis=1)]
    if np.any(coincident_point_mask(points, active_positions, atol=0)):
        raise ValueError("An outgoing expansion is singular at its source position")


def apply_periodic_reception(
    points: np.ndarray,
    positions: np.ndarray,
    coefficients: np.ndarray,
    *,
    lmax_out: int = 1,
    lmax_in: int,
    k: complex,
    periodic: PeriodicSpec,
    k_parallel: np.ndarray,
    exclude_self: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
    backend: str = "numpy",
    circumscribing_radii: np.ndarray | None = None,
    point_batch_size: int = 128,
    show_progress: bool = False,
) -> np.ndarray:
    """Receive outgoing waves, retaining nonzero self images.

    ``exclude_self`` removes only a coincident reference-cell source; it never
    removes its other images. Coordinates are reduced to a common cell before
    finite-image summation, with source and receiver Bloch phases restored.
    Ewald supplies rectangular reception also for a Rayleigh solve: source
    setup and sparse interior diagnostics do not require a second scan plan.
    """
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    if backend == "cupy":
        from .reception_cupy import apply_periodic_reception_cupy

        return apply_periodic_reception_cupy(
            points=points,
            positions=positions,
            coefficients=coefficients,
            lmax_in=lmax_in,
            lmax_out=lmax_out,
            k=k,
            periodic=periodic,
            k_parallel=k_parallel,
            exclude_self=exclude_self,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            circumscribing_radii=circumscribing_radii,
            point_batch_size=point_batch_size,
            show_progress=show_progress,
        )
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    coeffs = np.asarray(coefficients, dtype=compute_dtype).reshape(len(positions), n_modes(lmax_in))
    result = np.zeros((len(points), n_modes(lmax_out)), dtype=accum_dtype)
    active = np.any(coeffs != 0, axis=1)
    if not len(points) or not np.any(active):
        return result
    # Inactive expansions set neither the Green-function domain nor the
    # Ewald/Rayleigh geometry policy, even outside the active source slab.
    if not np.all(active):
        positions, coeffs = positions[active], coeffs[active]
        if circumscribing_radii is not None:
            circumscribing_radii = np.asarray(circumscribing_radii)[active]
    points, point_shifts = canonical_cell_points(
        points, ax=periodic.lattice.ax, ay=periodic.lattice.ay
    )
    positions, source_shifts = canonical_cell_points(
        positions, ax=periodic.lattice.ax, ay=periodic.lattice.ay
    )
    coeffs = np.asarray(
        coeffs * np.exp(-1j * (source_shifts[:, :2] @ k_parallel))[:, None], dtype=accum_dtype
    )
    if not exclude_self:
        _check_reception_origins(points, positions, coeffs)
    order = lmax_out + lmax_in
    capacity = (order + 1) // 2
    degrees, orders, indptr, input_modes, values = rectangular_projection_data(lmax_out, lmax_in)
    projection = csr_matrix(
        (values.astype(accum_dtype, copy=False), input_modes, indptr),
        shape=(len(degrees) * n_modes(lmax_out), n_modes(lmax_in)),
    )
    options = periodic.options
    far_plan = None
    if options.method == "rayleigh" and lmax_out == 1:
        from .rayleigh import apply_rayleigh_far_to_points_numpy, prepare_rayleigh_plan

        far_plan = prepare_rayleigh_plan(
            lmax=lmax_in,
            k=k,
            positions=positions,
            periodic=periodic,
            circumscribing_radii=circumscribing_radii,
            k_parallel=k_parallel,
            dtype=accum_dtype,
        )
        result += apply_rayleigh_far_to_points_numpy(far_plan, coeffs, points)
    near_offsets = near_destinations = None
    if far_plan is not None:
        from .rayleigh import near_point_source_csr

        near_offsets, near_destinations, _ = near_point_source_csr(
            points, positions, far_plan.z_cut
        )
    workspace = None
    if options.method != "directsum":
        eta = resolve_ewald_eta(
            periodic=periodic,
            k=k,
            k_parallel=k_parallel,
            positions=np.concatenate((points, positions)),
            lmax=capacity,
            max_vertical_offset=far_plan.z_cut
            if far_plan is not None
            else float(np.ptp(np.r_[points[:, 2], positions[:, 2]])),
        )
        workspace = EwaldShellWorkspace(
            lattice=periodic.lattice, k=k, k_parallel=k_parallel, eta=eta
        )
    from tqdm.auto import tqdm

    batch = max(1, int(point_batch_size))
    sources = (
        tqdm(enumerate(positions), total=len(positions), desc="Periodic reception", unit="source")
        if show_progress
        else enumerate(positions)
    )
    for j, source in sources:
        if not np.any(coeffs[j]):
            continue
        # Angular/source contraction is independent of the receiver batch.
        weighted_source = (projection @ coeffs[j]).reshape(len(degrees), n_modes(lmax_out))
        rows_all = (
            near_destinations[near_offsets[j] : near_offsets[j + 1]]
            if near_offsets is not None and near_destinations is not None
            else np.arange(len(points))
        )
        for start in range(0, len(rows_all), batch):
            rows = rows_all[start : start + batch]
            targets = points[rows]
            coincident = np.all(targets == source, axis=1)
            if workspace is None:
                sums = np.stack(
                    [
                        direct_structural_sums_2d(
                            lmax=capacity,
                            structural_order=order,
                            k=k,
                            destination=point,
                            source=source,
                            lattice=periodic.lattice,
                            k_parallel=k_parallel,
                            window=options.directsum_window,
                            exclude_zero_shift=bool(same),
                        )
                        for point, same in zip(targets, coincident, strict=True)
                    ]
                )
            else:
                sums = ewald_structural_sums_2d_batch(
                    lmax_struct=capacity,
                    structural_order=order,
                    k=k,
                    destinations=targets,
                    source=source,
                    lattice=periodic.lattice,
                    k_parallel=k_parallel,
                    eta=workspace.eta,
                    workspace=workspace,
                    real_shells=options.real_shells,
                    reciprocal_shells=options.reciprocal_shells,
                    shell_tolerance=options.shell_tolerance,
                    max_shells=options.max_shells,
                )
                for local in np.flatnonzero(coincident):
                    _add_self_correction_to_batched_sums(
                        sums,
                        local_destination_index=int(local),
                        k=k,
                        eta=workspace.eta,
                    )
            compact_sums = np.asarray(sums[:, degrees, orders], dtype=accum_dtype)
            result[rows] += compact_sums @ weighted_source
    result *= np.exp(1j * (point_shifts[:, :2] @ k_parallel))[:, None]
    return result
