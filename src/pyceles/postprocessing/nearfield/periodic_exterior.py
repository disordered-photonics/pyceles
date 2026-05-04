from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles.core.periodic import PeriodicSpec
from pyceles.core.sources import JonesPolarizedSource, PlaneWave
from pyceles.postprocessing._channels import is_pure_channel_result
from pyceles.postprocessing.farfield import periodic_order_amplitudes

from .components import NearFieldComponents
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def _resolve_periodic_channel_payload(
    run: SimulationResult, *, channel: Literal["mixed", "te", "tm"]
) -> tuple[np.ndarray, PlaneWave]:
    """Resolve coefficient/source payload for one periodic near-field channel."""
    source = run.config.source
    if not isinstance(source, PlaneWave):
        raise NotImplementedError(
            "Periodic near-field evaluation currently supports PlaneWave sources only."
        )
    if channel == "mixed":
        return np.asarray(run.coeffs), source

    if run.coeffs_basis is not None and channel in run.coeffs_basis:
        coeffs = np.asarray(run.coeffs_basis[channel])
        if not isinstance(source, JonesPolarizedSource):
            raise ValueError(
                "Basis periodic near-field channel requires a source with Jones metadata "
                "and `with_polarization('TE'/'TM')`."
            )
        pol_label: Literal["TE", "TM"] = "TE" if channel == "te" else "TM"
        return coeffs, source.with_polarization(pol_label)

    if is_pure_channel_result(run, channel):
        return np.asarray(run.coeffs), source

    raise ValueError(
        "Requested basis periodic near-field channel, but `run.coeffs_basis` is not available. "
        "Use `solve_polarization_basis=True` with `Simulation.run()`, or use a channel result "
        "from `Simulation.postprocess_sources(Simulation.solve_sources(...))` and query it with "
        "`channel='mixed'`."
    )


def _slab_z_bounds(run: SimulationResult) -> tuple[float, float]:
    """Return conservative particle-slab z bounds from circumscribing spheres."""
    if run.n_particles == 0:
        return float("-inf"), float("inf")
    z = np.asarray(run.positions[:, 2], dtype=float).reshape(-1)
    r = np.asarray(run.circumscribing_radii, dtype=float).reshape(-1)
    z_min = float(np.min(z - r))
    z_max = float(np.max(z + r))
    return z_min, z_max


def _slab_classification_tolerance(
    *,
    points_z: np.ndarray,
    z_min: float,
    z_max: float,
    accum_dtype: str,
    user_tol: float,
) -> float:
    """Return a dtype-aware z-classification tolerance for slab-region dispatch."""
    tol_user = float(user_tol)
    if tol_user < 0.0:
        raise ValueError(f"`slab_tolerance` must be >= 0. Got {user_tol!r}.")
    eps = np.finfo(np.dtype(accum_dtype)).eps
    scale = max(
        1.0,
        float(abs(z_min)),
        float(abs(z_max)),
        float(np.max(np.abs(np.asarray(points_z, dtype=float)))) if points_z.size else 1.0,
    )
    tol_auto = 64.0 * float(eps) * scale
    return float(max(tol_user, tol_auto))


def _plane_wave_vectors(
    *,
    k: float,
    k_parallel: np.ndarray,
    kz: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return TE/TM unit vectors and corresponding H-vector factors."""
    kx = float(k_parallel[0])
    ky = float(k_parallel[1])
    alpha = float(np.arctan2(ky, kx))
    kp = float(np.hypot(kx, ky))
    sb = kp / float(k)
    cb = complex(kz) / float(k)

    e_te = np.asarray(
        [-np.sin(alpha), np.cos(alpha), 0.0],
        dtype=np.complex128,
    )
    e_tm = np.asarray(
        [cb * np.cos(alpha), cb * np.sin(alpha), -sb],
        dtype=np.complex128,
    )
    khat = np.asarray(
        [sb * np.cos(alpha), sb * np.sin(alpha), cb],
        dtype=np.complex128,
    )
    h_te = np.cross(khat, e_te)
    h_tm = np.cross(khat, e_tm)
    return e_te, e_tm, h_te, h_tm


def _initial_plane_wave_field(
    points: np.ndarray,
    *,
    source: PlaneWave,
    k: float,
    n_medium: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate incident plane-wave `E/H` fields at arbitrary points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    k_parallel = np.asarray(
        [
            float(k)
            * float(np.sin(float(source.polar_angle)))
            * float(np.cos(float(source.azimuthal_angle))),
            float(k)
            * float(np.sin(float(source.polar_angle)))
            * float(np.sin(float(source.azimuthal_angle))),
        ],
        dtype=float,
    )
    kz = complex(float(k) * float(np.cos(float(source.polar_angle))), 0.0)
    e_te, e_tm, h_te, h_tm = _plane_wave_vectors(k=float(k), k_parallel=k_parallel, kz=kz)
    a_te, a_tm = source.jones_coefficients()
    amplitude = complex(source.amplitude)

    e_vec = amplitude * (complex(a_te) * e_te + complex(a_tm) * e_tm)
    h_vec = amplitude * complex(n_medium) * (complex(a_te) * h_te + complex(a_tm) * h_tm)

    fp = np.asarray(getattr(source, "focal_point", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    phase = np.exp(
        1j
        * (
            float(k_parallel[0]) * (pts[:, 0] - float(fp[0]))
            + float(k_parallel[1]) * (pts[:, 1] - float(fp[1]))
            + float(np.real(kz)) * (pts[:, 2] - float(fp[2]))
        )
    )
    e = phase[:, None] * e_vec[None, :]
    h = phase[:, None] * h_vec[None, :]
    return e, h


def _accumulate_order_field(
    points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    order_k_parallel: np.ndarray,
    order_kz: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate one-hemisphere periodic order field on query points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = int(pts.shape[0])
    e = np.zeros((n_points, 3), dtype=np.complex128)
    h = np.zeros((n_points, 3), dtype=np.complex128)
    if n_points == 0:
        return e, h

    kpar = np.asarray(order_k_parallel, dtype=float).reshape(-1, 2)
    kz_arr = np.asarray(order_kz, dtype=np.complex128).reshape(-1)
    amp = np.asarray(amplitudes, dtype=np.complex128).reshape(-1, 2)
    for i in range(kpar.shape[0]):
        kx = float(kpar[i, 0])
        ky = float(kpar[i, 1])
        kz = complex(kz_arr[i])
        if abs(kz) <= 1e-15:
            continue
        a_te = complex(amp[i, 0])
        a_tm = complex(amp[i, 1])
        if abs(a_te) <= 0.0 and abs(a_tm) <= 0.0:
            continue
        e_te, e_tm, h_te, h_tm = _plane_wave_vectors(
            k=float(k),
            k_parallel=kpar[i],
            kz=kz,
        )
        e_vec = a_te * e_te + a_tm * e_tm
        h_vec = complex(n_medium) * (a_te * h_te + a_tm * h_tm)
        phase = np.exp(1j * (kx * pts[:, 0] + ky * pts[:, 1] + kz * pts[:, 2]))
        e += phase[:, None] * e_vec[None, :]
        h += phase[:, None] * h_vec[None, :]
    return e, h


def compute_periodic_near_field_exterior(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    field_bmax: float | None = None,
    slab_tolerance: float = 1e-12,
) -> NearFieldComponents:
    """Evaluate periodic near fields for points strictly outside the particle slab."""
    periodic = run.config.periodic
    if periodic is None:
        raise ValueError(
            "`compute_periodic_near_field` requires a periodic simulation result "
            "(`run.config.periodic` must be set)."
        )
    if not isinstance(periodic, PeriodicSpec):
        raise TypeError(
            "Periodic near-field evaluation requires `run.config.periodic` to be a PeriodicSpec."
        )
    channel_key = str(channel).lower()
    if channel_key not in {"mixed", "te", "tm"}:
        raise ValueError("`channel` must be one of {'mixed', 'te', 'tm'}.")
    if channel_key == "mixed":
        channel_literal: Literal["mixed", "te", "tm"] = "mixed"
    elif channel_key == "te":
        channel_literal = "te"
    else:
        channel_literal = "tm"
    coeffs, source = _resolve_periodic_channel_payload(run, channel=channel_literal)

    pts_flat, lead_shape = reshape_field_points(points)
    n_points = int(pts_flat.shape[0])
    if run.n_particles == 0:
        above_mask = np.ones((n_points,), dtype=bool)
        below_mask = np.zeros((n_points,), dtype=bool)
        slab_mask = np.zeros((n_points,), dtype=bool)
    else:
        z_min, z_max = _slab_z_bounds(run)
        tol = _slab_classification_tolerance(
            points_z=np.asarray(pts_flat[:, 2], dtype=float),
            z_min=z_min,
            z_max=z_max,
            accum_dtype=str(run.accum_dtype),
            user_tol=float(slab_tolerance),
        )
        above_mask = np.asarray(pts_flat[:, 2] > (z_max + tol), dtype=bool)
        below_mask = np.asarray(pts_flat[:, 2] < (z_min - tol), dtype=bool)
        slab_mask = ~(above_mask | below_mask)
    if np.any(slab_mask):
        raise NotImplementedError(
            "Periodic near-field local (in-slab) evaluation is not implemented yet. "
            f"Requested {int(np.count_nonzero(slab_mask))} point(s) inside or intersecting the slab."
        )

    k = float(run.k)
    output_bmax = periodic.options.output_bmax if field_bmax is None else float(field_bmax)
    if output_bmax is None:
        raise ValueError(
            "Periodic near-field exterior evaluation requires an explicit reciprocal output basis. "
            "Set `field_bmax` on the call or configure `periodic.options.output_bmax`."
        )
    order_payload = periodic_order_amplitudes(
        source=source,
        lattice=periodic.lattice,
        positions=run.positions,
        coeffs=np.asarray(coeffs),
        lmax=int(run.config.lmax),
        k=k,
        output_bmax=output_bmax,
    )

    e_initial, h_initial = _initial_plane_wave_field(
        pts_flat,
        source=source,
        k=k,
        n_medium=run.config.n_medium,
    )
    e_scattered = np.zeros_like(e_initial)
    h_scattered = np.zeros_like(h_initial)

    if np.any(above_mask):
        e_up, h_up = _accumulate_order_field(
            pts_flat[above_mask],
            k=k,
            n_medium=run.config.n_medium,
            order_k_parallel=order_payload.order_k_parallel,
            order_kz=order_payload.order_kz,
            amplitudes=order_payload.scattered_up_amplitudes,
        )
        e_scattered[above_mask] = e_up
        h_scattered[above_mask] = h_up
    if np.any(below_mask):
        e_down, h_down = _accumulate_order_field(
            pts_flat[below_mask],
            k=k,
            n_medium=run.config.n_medium,
            order_k_parallel=order_payload.order_k_parallel,
            order_kz=-np.asarray(order_payload.order_kz, dtype=np.complex128),
            amplitudes=order_payload.scattered_down_amplitudes,
        )
        e_scattered[below_mask] = e_down
        h_scattered[below_mask] = h_down

    e_internal = np.zeros_like(e_initial)
    h_internal = np.zeros_like(h_initial)
    e_total = e_initial + e_scattered
    h_total = h_initial + h_scattered
    inside = np.zeros((n_points,), dtype=bool)

    out_dtype = np.dtype(run.accum_dtype)
    if lead_shape == ():
        vec_shape: tuple[int, ...] = (3,)
        mask_shape: tuple[int, ...] = ()
    else:
        vec_shape = (*lead_shape, 3)
        mask_shape = lead_shape
    return NearFieldComponents(
        E_initial=np.asarray(e_initial, dtype=out_dtype).reshape(vec_shape),
        H_initial=np.asarray(h_initial, dtype=out_dtype).reshape(vec_shape),
        E_scattered=np.asarray(e_scattered, dtype=out_dtype).reshape(vec_shape),
        H_scattered=np.asarray(h_scattered, dtype=out_dtype).reshape(vec_shape),
        E_internal=np.asarray(e_internal, dtype=out_dtype).reshape(vec_shape),
        H_internal=np.asarray(h_internal, dtype=out_dtype).reshape(vec_shape),
        E_total=np.asarray(e_total, dtype=out_dtype).reshape(vec_shape),
        H_total=np.asarray(h_total, dtype=out_dtype).reshape(vec_shape),
        inside_mask=inside.reshape(mask_shape),
    )


__all__ = ["compute_periodic_near_field_exterior"]
