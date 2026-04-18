from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles.core.sources import JonesPolarizedSource

from .components import NearFieldComponents, compute_near_field_components
from .slice import (
    NearFieldSlice,
    interpolate_center_pixels,
    reshape_field_points,
    slice_plane_metadata,
)

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def _is_pure_channel_result(run: SimulationResult, channel: str, *, atol: float = 1e-12) -> bool:
    """Return True when `run` already represents one pure TE/TM Jones channel."""
    src = run.config.source
    if src is None:
        return False
    if not isinstance(src, JonesPolarizedSource):
        return False
    if run.polarization_jones is None:
        return False
    a_te, a_tm = run.polarization_jones
    if channel == "te":
        return bool(abs(complex(a_tm)) <= atol and abs(complex(a_te)) > atol)
    if channel == "tm":
        return bool(abs(complex(a_te)) <= atol and abs(complex(a_tm)) > atol)
    return False


def _mix_complex_vector_fields(
    a_te: complex,
    a_tm: complex,
    f_te: np.ndarray,
    f_tm: np.ndarray,
) -> np.ndarray:
    """Coherently mix TE/TM complex vector fields with Jones amplitudes."""
    return np.asarray(a_te * f_te + a_tm * f_tm, dtype=np.result_type(f_te, f_tm, np.complex64))


def mix_near_field_components(
    nf_te: NearFieldComponents,
    nf_tm: NearFieldComponents,
    *,
    a_te: complex,
    a_tm: complex,
) -> NearFieldComponents:
    """Coherently mix TE/TM near-field components into one Jones channel."""
    return NearFieldComponents(
        E_initial=_mix_complex_vector_fields(a_te, a_tm, nf_te.E_initial, nf_tm.E_initial),
        H_initial=_mix_complex_vector_fields(a_te, a_tm, nf_te.H_initial, nf_tm.H_initial),
        E_scattered=_mix_complex_vector_fields(a_te, a_tm, nf_te.E_scattered, nf_tm.E_scattered),
        H_scattered=_mix_complex_vector_fields(a_te, a_tm, nf_te.H_scattered, nf_tm.H_scattered),
        E_internal=_mix_complex_vector_fields(a_te, a_tm, nf_te.E_internal, nf_tm.E_internal),
        H_internal=_mix_complex_vector_fields(a_te, a_tm, nf_te.H_internal, nf_tm.H_internal),
        E_total=_mix_complex_vector_fields(a_te, a_tm, nf_te.E_total, nf_tm.E_total),
        H_total=_mix_complex_vector_fields(a_te, a_tm, nf_te.H_total, nf_tm.H_total),
        inside_mask=np.asarray(nf_te.inside_mask) | np.asarray(nf_tm.inside_mask),
    )


def mix_near_field_slices(
    slice_te: NearFieldSlice,
    slice_tm: NearFieldSlice,
    *,
    a_te: complex,
    a_tm: complex,
) -> NearFieldSlice:
    """Coherently mix TE/TM near-field slices into one Jones-channel slice."""
    if slice_te.plane != slice_tm.plane:
        raise ValueError("Cannot mix slices from different planes.")
    if not np.isclose(slice_te.plane_value, slice_tm.plane_value):
        raise ValueError("Cannot mix slices at different plane values.")
    if (
        slice_te.axis_0.shape != slice_tm.axis_0.shape
        or slice_te.axis_1.shape != slice_tm.axis_1.shape
    ):
        raise ValueError("Cannot mix slices with different grid shapes.")
    if not np.allclose(slice_te.axis_0, slice_tm.axis_0) or not np.allclose(
        slice_te.axis_1, slice_tm.axis_1
    ):
        raise ValueError("Cannot mix slices with different grid coordinates.")

    map_keys = set(slice_te.field_maps).intersection(set(slice_tm.field_maps))
    mixed_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in map_keys:
        e_te, h_te = slice_te.field_maps[key]
        e_tm, h_tm = slice_tm.field_maps[key]
        mixed_maps[key] = (
            _mix_complex_vector_fields(a_te, a_tm, e_te, e_tm),
            _mix_complex_vector_fields(a_te, a_tm, h_te, h_tm),
        )

    return NearFieldSlice(
        axis_0=np.asarray(slice_te.axis_0),
        axis_1=np.asarray(slice_te.axis_1),
        inside=np.asarray(slice_te.inside) | np.asarray(slice_tm.inside),
        field_maps=mixed_maps,
        plane=slice_te.plane,
        plane_value=float(slice_te.plane_value),
        axis_0_label=slice_te.axis_0_label,
        axis_1_label=slice_te.axis_1_label,
    )


def compute_near_field(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    show_progress: bool = True,
    force_general_initial_field: bool | None = None,
) -> NearFieldComponents:
    """Evaluate near-field components on arbitrary point coordinates."""
    pts_flat, lead_shape = reshape_field_points(points)

    source = run.config.source
    if source is None:
        raise RuntimeError(
            "SimulationResult has no source attached. "
            "Run a simulation with an explicit `SimulationConfig.source`."
        )

    channel_key = str(channel).lower()
    if channel_key not in {"mixed", "te", "tm"}:
        raise ValueError("`channel` must be one of {'mixed', 'te', 'tm'}.")

    coeffs = run.coeffs
    source_eff = source
    if channel_key in {"te", "tm"}:
        used_basis_payload = False
        if run.coeffs_basis is not None and channel_key in run.coeffs_basis:
            coeffs = run.coeffs_basis[channel_key]
            used_basis_payload = True
        elif _is_pure_channel_result(run, channel_key):
            coeffs = run.coeffs
        else:
            raise ValueError(
                "Requested basis near-field channel, but `run.coeffs_basis` is not available. "
                "Use `solve_polarization_basis=True` with `Simulation.run()`, or use a "
                "channel result from "
                "`Simulation.postprocess_sources(Simulation.solve_sources(...))` and query it with "
                "`channel='mixed'`."
            )
        if used_basis_payload:
            pol_label: Literal["TE", "TM"] = "TE" if channel_key == "te" else "TM"
            if not isinstance(source, JonesPolarizedSource):
                raise ValueError(
                    "Basis near-field channel requires a source with Jones metadata "
                    "and `with_polarization('TE'/'TM')`."
                )
            source_eff = source.with_polarization(pol_label)

    source_polar_angles, source_azimuthal_angles = run.config.source_angular_grids()
    postprocessing_backend = run.config.resolved_postprocessing_backend()
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=run.config.compute_dtype,
        accum_dtype=run.config.accum_dtype,
    )

    nf = compute_near_field_components(
        pts_flat,
        coeffs=coeffs,
        k=run.k,
        lmax=run.config.lmax,
        beam=source_eff,
        polar_angles=source_polar_angles,
        azimuthal_angles=source_azimuthal_angles,
        particles=run.particles,
        n_medium=run.config.n_medium,
        show_progress=show_progress,
        force_general_initial_field=(
            bool(run.config.force_general_initial_field)
            if force_general_initial_field is None
            else bool(force_general_initial_field)
        ),
        lut_dr=run.config.radial_lut_dr,
        backend=postprocessing_backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    if lead_shape == ():
        vec_shape: tuple[int, ...] = (3,)
        mask_shape: tuple[int, ...] = ()
    else:
        vec_shape = (*lead_shape, 3)
        mask_shape = lead_shape

    return NearFieldComponents(
        E_initial=nf.E_initial.reshape(vec_shape),
        H_initial=nf.H_initial.reshape(vec_shape),
        E_scattered=nf.E_scattered.reshape(vec_shape),
        H_scattered=nf.H_scattered.reshape(vec_shape),
        E_internal=nf.E_internal.reshape(vec_shape),
        H_internal=nf.H_internal.reshape(vec_shape),
        E_total=nf.E_total.reshape(vec_shape),
        H_total=nf.H_total.reshape(vec_shape),
        inside_mask=nf.inside_mask.reshape(mask_shape),
    )


def compute_near_field_slice(
    run: SimulationResult,
    *,
    axis_0_min: float = -4000.0,
    axis_0_max: float = 4000.0,
    axis_1_min: float = -3000.0,
    axis_1_max: float = 5000.0,
    dx: float = 40.0,
    plane: str = "y",
    plane_value: float = 0.0,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    show_progress: bool = True,
    force_general_initial_field: bool | None = None,
    center_pixel_policy: Literal["none", "interpolate"] = "interpolate",
) -> NearFieldSlice:
    """Evaluate near fields on an axis-aligned planar slice."""
    if float(dx) <= 0.0:
        raise ValueError(f"`dx` must be > 0. Got {dx!r}.")

    if float(axis_0_max) < float(axis_0_min):
        raise ValueError(
            f"`axis_0_max` must be >= `axis_0_min`. Got axis_0_min={axis_0_min!r}, axis_0_max={axis_0_max!r}."
        )
    if float(axis_1_max) < float(axis_1_min):
        raise ValueError(
            f"`axis_1_max` must be >= `axis_1_min`. Got axis_1_min={axis_1_min!r}, axis_1_max={axis_1_max!r}."
        )

    plane_name = str(plane).lower()
    if plane_name not in {"x", "y", "z"}:
        raise ValueError(f"`plane` must be one of 'x', 'y', or 'z'. Got {plane!r}.")
    _, _, _, axis_0_label, axis_1_label = slice_plane_metadata(plane_name)
    policy = str(center_pixel_policy).lower()
    if policy not in {"none", "interpolate"}:
        raise ValueError(
            f"`center_pixel_policy` must be 'none' or 'interpolate'. Got {center_pixel_policy!r}."
        )

    axis_0_vals = np.arange(float(axis_0_min), float(axis_0_max) + float(dx), float(dx))
    axis_1_vals = np.arange(float(axis_1_min), float(axis_1_max) + float(dx), float(dx))
    axis_0_grid, axis_1_grid = np.meshgrid(axis_0_vals, axis_1_vals, indexing="xy")
    axis_const = np.zeros_like(axis_0_grid)
    if plane_name == "x":
        pts = np.stack(
            [np.full_like(axis_0_grid, float(plane_value)), axis_0_grid, axis_1_grid], axis=-1
        )
    elif plane_name == "z":
        pts = np.stack(
            [axis_0_grid, axis_1_grid, np.full_like(axis_0_grid, float(plane_value))], axis=-1
        )
    else:
        pts = np.stack([axis_0_grid, axis_const + float(plane_value), axis_1_grid], axis=-1)

    nf = compute_near_field(
        run,
        points=pts,
        channel=channel,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
    )
    field_dtype = np.dtype(getattr(run.config, "compute_dtype", "complex128"))
    field_maps = {
        "initial": (
            nf.E_initial.astype(field_dtype, copy=False),
            nf.H_initial.astype(field_dtype, copy=False),
        ),
        "scattered": (
            nf.E_scattered.astype(field_dtype, copy=False),
            nf.H_scattered.astype(field_dtype, copy=False),
        ),
        "internal": (
            nf.E_internal.astype(field_dtype, copy=False),
            nf.H_internal.astype(field_dtype, copy=False),
        ),
        "total": (
            nf.E_total.astype(field_dtype, copy=False),
            nf.H_total.astype(field_dtype, copy=False),
        ),
    }
    if policy == "interpolate":
        interpolate_center_pixels(
            run=run,
            axis_0_grid=axis_0_grid,
            axis_1_grid=axis_1_grid,
            field_maps=field_maps,
            plane=plane_name,
            plane_value=float(plane_value),
        )
    return NearFieldSlice(
        axis_0=axis_0_grid,
        axis_1=axis_1_grid,
        inside=nf.inside_mask.reshape(axis_1_grid.shape),
        field_maps=field_maps,
        plane=plane_name,
        plane_value=float(plane_value),
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
    )


__all__ = [
    "NearFieldComponents",
    "NearFieldSlice",
    "compute_near_field",
    "compute_near_field_slice",
    "mix_near_field_components",
    "mix_near_field_slices",
]
