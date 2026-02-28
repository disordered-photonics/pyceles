from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles._dtypes import resolve_compute_accum_dtypes

from .nearfield_workflows import NearFieldComponents, compute_near_field_components

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


@dataclass(frozen=True)
class NearFieldSlice:
    """Near-field payload on a 2D slice for plotting and diagnostics.

    `field_maps` stores complex vector fields `(E, H)` for each family
    (`initial`, `scattered`, `internal`, `total`) sampled on the same grid.
    """

    axis_0: np.ndarray
    axis_1: np.ndarray
    inside: np.ndarray
    field_maps: dict[str, tuple[np.ndarray, np.ndarray]]
    plane: str
    plane_value: float
    axis_0_label: str
    axis_1_label: str


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
    """Coherently mix TE/TM near-field components into one Jones channel.

    This helper enables reuse when TE and TM channels were already evaluated
    and a requested mixed Jones field should be formed without recomputing
    near-field kernels.
    """
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
        E_te, H_te = slice_te.field_maps[key]
        E_tm, H_tm = slice_tm.field_maps[key]
        mixed_maps[key] = (
            _mix_complex_vector_fields(a_te, a_tm, E_te, E_tm),
            _mix_complex_vector_fields(a_te, a_tm, H_te, H_tm),
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


def _slice_plane_metadata(plane: str) -> tuple[int, int, int, str, str]:
    """Map plane label to normal/tangential coordinate indices and axis labels."""
    p = str(plane).lower()
    if p == "x":
        return 0, 1, 2, "y", "z"
    if p == "y":
        return 1, 0, 2, "x", "z"
    if p == "z":
        return 2, 0, 1, "x", "y"
    raise ValueError("plane must be one of 'x', 'y', or 'z'.")


def _reshape_field_points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize arbitrary point inputs to flat `(N,3)` plus original leading shape."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(
                f"1D `points` input must have exactly 3 entries (x,y,z). Got shape {arr.shape}."
            )
        return arr.reshape(1, 3), ()
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        lead_shape = tuple(arr.shape[:-1])
        return arr.reshape(-1, 3), lead_shape
    raise ValueError(f"`points` must be shaped (3,), (N,3), or (...,3). Got shape {arr.shape}.")


def compute_near_field(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    show_progress: bool = True,
    force_general_initial_field: bool | None = None,
) -> NearFieldComponents:
    """Evaluate near-field components on arbitrary point coordinates.

    This is the geometry-agnostic counterpart of `compute_near_field_slice`.
    Accepted `points` shapes are:
    - `(3,)` for a single point
    - `(N, 3)` for a point cloud
    - `(..., 3)` for structured grids/volumes

    Parameters
    ----------
    channel:
        - ``"mixed"``: use the source polarization requested by the user
          (default).
        - ``"te"`` / ``"tm"``: evaluate one pure basis channel. Requires
          `run.coeffs_basis` from `solve_polarization_basis=True`.
    """
    pts_flat, lead_shape = _reshape_field_points(points)

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
        if run.coeffs_basis is None:
            raise ValueError(
                "Requested basis near-field channel, but `run.coeffs_basis` is not available. "
                "Run simulation with `solve_polarization_basis=True`."
            )
        coeffs = run.coeffs_basis[channel_key]
        pol_label: Literal["TE", "TM"] = "TE" if channel_key == "te" else "TM"
        source_eff = source.with_polarization(pol_label)

    # Initial-field near-field evaluation must use the source-projection angular
    # quadrature, not necessarily the far-field display grid.
    source_polar_angles, source_azimuthal_angles = run.config.source_angular_grids()

    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=run.config.compute_dtype,
        accum_dtype=run.config.accum_dtype,
    )

    nf = compute_near_field_components(
        pts_flat,
        positions=run.positions,
        coeffs=coeffs,
        k=run.k,
        lmax=run.config.lmax,
        beam=source_eff,
        polar_angles=source_polar_angles,
        azimuthal_angles=source_azimuthal_angles,
        radii=run.radii,
        n_particle=run.n_particle,
        n_medium=run.config.n_medium,
        show_progress=show_progress,
        force_general_initial_field=(
            bool(run.config.force_general_initial_field)
            if force_general_initial_field is None
            else bool(force_general_initial_field)
        ),
        lut_dr=run.config.radial_lut_dr,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    vec_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    if lead_shape == ():
        vec_shape = (3,)
        mask_shape = ()
    else:
        vec_shape = lead_shape + (3,)
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


def _neighbor_mean_inplace(arr: np.ndarray, i0: int, i1: int) -> None:
    """Replace one sample with the mean of available 4-neighborhood samples."""
    n0, n1 = arr.shape[0], arr.shape[1]
    values = []
    for d0, d1 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        j0 = i0 + d0
        j1 = i1 + d1
        if 0 <= j0 < n0 and 0 <= j1 < n1:
            values.append(arr[j0, j1])
    if values:
        arr[i0, i1] = np.mean(np.asarray(values), axis=0)


def _interpolate_center_pixels(
    *,
    run: SimulationResult,
    axis_0_grid: np.ndarray,
    axis_1_grid: np.ndarray,
    field_maps: dict[str, tuple[np.ndarray, np.ndarray]],
    plane: str,
    plane_value: float,
) -> None:
    """Patch singular center pixels by local interpolation on exact center hits.

    When a slice grid node lands exactly on a sphere center, some spherical-field
    formulas can generate an isolated numerical artifact. This post-step replaces
    only those exact-center pixels by neighbor means, leaving all other samples
    untouched.
    """
    # Build 2D projected center coordinates for spheres whose centers lie on the slice.
    p = np.asarray(run.positions, dtype=float)
    normal_idx, axis_0_idx, axis_1_idx, _, _ = _slice_plane_metadata(plane)
    on_plane = np.isclose(p[:, normal_idx], plane_value, atol=1e-12)
    u = p[on_plane, axis_0_idx]
    v = p[on_plane, axis_1_idx]

    if u.size == 0:
        return

    axis_0_values = np.asarray(axis_0_grid[0, :], dtype=float)
    axis_1_values = np.asarray(axis_1_grid[:, 0], dtype=float)
    for uu, vv in zip(u, v):
        i0 = int(np.argmin(np.abs(axis_0_values - uu)))
        i1 = int(np.argmin(np.abs(axis_1_values - vv)))
        if not (
            np.isclose(axis_0_values[i0], uu, atol=1e-12)
            and np.isclose(axis_1_values[i1], vv, atol=1e-12)
        ):
            continue
        for key in field_maps:
            E_map, H_map = field_maps[key]
            _neighbor_mean_inplace(E_map, i1, i0)
            _neighbor_mean_inplace(H_map, i1, i0)


def compute_near_field_slice(
    run: SimulationResult,
    *,
    x_min: float = -4000.0,
    x_max: float = 4000.0,
    z_min: float = -3000.0,
    z_max: float = 5000.0,
    axis_0_min: float | None = None,
    axis_0_max: float | None = None,
    axis_1_min: float | None = None,
    axis_1_max: float | None = None,
    dx: float = 40.0,
    plane: str = "y",
    plane_value: float = 0.0,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    show_progress: bool = True,
    force_general_initial_field: bool | None = None,
    center_pixel_policy: Literal["none", "interpolate"] = "interpolate",
) -> NearFieldSlice:
    """Evaluate near fields on an axis-aligned planar slice.

    This is a convenience wrapper for visualization workflows. For arbitrary
    points/grids/volumes use `compute_near_field`.

    Notes
    -----
    Unpolarized near fields are not represented as a single complex vector
    field. For incoherent TE/TM averaging, evaluate `channel="te"` and
    `channel="tm"` separately and combine intensity-level observables.
    """
    if float(dx) <= 0.0:
        raise ValueError(f"`dx` must be > 0. Got {dx!r}.")

    if axis_0_min is None:
        axis_0_min = x_min
    if axis_0_max is None:
        axis_0_max = x_max
    if axis_1_min is None:
        axis_1_min = z_min
    if axis_1_max is None:
        axis_1_max = z_max

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
    _, _, _, axis_0_label, axis_1_label = _slice_plane_metadata(plane_name)
    policy = str(center_pixel_policy).lower()
    if policy not in {"none", "interpolate"}:
        raise ValueError(
            f"`center_pixel_policy` must be 'none' or 'interpolate'. Got {center_pixel_policy!r}."
        )

    axis_0_vals = np.arange(float(axis_0_min), float(axis_0_max) + float(dx), float(dx))
    axis_1_vals = np.arange(float(axis_1_min), float(axis_1_max) + float(dx), float(dx))
    A0, A1 = np.meshgrid(axis_0_vals, axis_1_vals, indexing="xy")
    A_const = np.zeros_like(A0)
    if plane_name == "x":
        pts = np.stack([np.full_like(A0, float(plane_value)), A0, A1], axis=-1)
    elif plane_name == "z":
        pts = np.stack([A0, A1, np.full_like(A0, float(plane_value))], axis=-1)
    else:
        pts = np.stack([A0, A_const + float(plane_value), A1], axis=-1)

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
        _interpolate_center_pixels(
            run=run,
            axis_0_grid=A0,
            axis_1_grid=A1,
            field_maps=field_maps,
            plane=plane_name,
            plane_value=float(plane_value),
        )
    return NearFieldSlice(
        axis_0=A0,
        axis_1=A1,
        inside=nf.inside_mask.reshape(A1.shape),
        field_maps=field_maps,
        plane=plane_name,
        plane_value=float(plane_value),
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
    )
