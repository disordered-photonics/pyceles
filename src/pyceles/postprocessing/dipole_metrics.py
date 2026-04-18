"""Dipole-emitter power and LDOS helpers for homogeneous-host pyceles runs.

These helpers intentionally evaluate *particle-scattered* fields at dipole
positions and avoid direct dipole self-field sampling at r=0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.postprocessing.nearfield.scattered import compute_scattered_field

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


@dataclass(frozen=True)
class DipolePowerLDOSResult:
    """Dissipated-power / LDOS payload evaluated at dipole source positions.

    Attributes
    ----------
    dipole_positions:
        Dipole coordinates `(Nd, 3)`.
    dipole_moments:
        Amplitude-scaled complex dipole moments `(Nd, 3)`.
    E_scattered_at_dipoles:
        Particle-scattered electric field sampled at each dipole position `(Nd, 3)`.
    delta_power:
        Scattering-induced power correction
        `DeltaP = (k0/2) Im( p* · E_scat(r_d) )` per dipole.
    power_homogeneous:
        Homogeneous-background dipole power `P0` per dipole.
    power_total:
        Total dissipated power estimate `P = P0 + DeltaP` per dipole.
    enhancement:
        Purcell/LDOS-like enhancement ratio `P / P0` per dipole.
    """

    dipole_positions: np.ndarray
    dipole_moments: np.ndarray
    E_scattered_at_dipoles: np.ndarray
    delta_power: np.ndarray
    power_homogeneous: np.ndarray
    power_total: np.ndarray
    enhancement: np.ndarray


def _dipole_arrays(source: DipoleSource | DipoleCollection) -> tuple[np.ndarray, np.ndarray]:
    """Return dipole positions and amplitude-scaled moments as `(Nd,3)` arrays."""
    if isinstance(source, DipoleSource):
        return (
            np.asarray(source.dipole_positions(), dtype=float).reshape(-1, 3),
            np.asarray(source.dipole_moments(), dtype=np.complex128).reshape(-1, 3),
        )
    return (
        np.asarray(source.dipole_positions(), dtype=float).reshape(-1, 3),
        np.asarray(source.dipole_moments_array(), dtype=np.complex128).reshape(-1, 3),
    )


def compute_dipole_power_ldos(
    run: SimulationResult,
    *,
    channel: Literal["mixed"] = "mixed",
    allow_inside_particle: bool = False,
    show_progress: bool = False,
) -> DipolePowerLDOSResult:
    r"""Compute dipole dissipated power and LDOS-like enhancement from `SimulationResult`.

    This helper is for local dipole-source runs (`DipoleSource` or
    `DipoleCollection`) and evaluates only particle-scattered fields at dipole
    positions. It avoids direct dipole self-field singular evaluations.

    Formulas (pyceles unit conventions):
    - `P0 = |p|^2 * k_medium * k0^3 / (12*pi)` (homogeneous background)
    - `DeltaP = (k0/2) * Im(p* · E_scat(r_d))`
    - `P = P0 + DeltaP`
    - `enhancement = P / P0`

    Notes
    -----
    - `enhancement` corresponds to projected electric-LDOS/Purcell-like ratio
      for the specified dipole orientation.
    - Interior dipole placement (inside circumscribing particle spheres) is
      currently untested in pyceles; by default this raises unless
      `allow_inside_particle=True`.
    """
    if channel != "mixed":
        raise ValueError(
            "Dipole power/LDOS helpers currently support only `channel='mixed'` "
            "because dipole sources do not define TE/TM basis channels."
        )
    source = run.config.source
    if not isinstance(source, (DipoleSource, DipoleCollection)):
        raise TypeError(
            "Dipole power/LDOS helpers require a run produced from DipoleSource "
            f"or DipoleCollection. Got {type(source).__name__}."
        )

    dip_pos, dip_mom = _dipole_arrays(source)
    if dip_pos.shape[0] == 0:
        empty = np.zeros((0,), dtype=float)
        return DipolePowerLDOSResult(
            dipole_positions=dip_pos,
            dipole_moments=dip_mom,
            E_scattered_at_dipoles=np.zeros((0, 3), dtype=np.complex128),
            delta_power=empty,
            power_homogeneous=empty,
            power_total=empty,
            enhancement=empty,
        )

    if run.positions.shape[0] > 0:
        dr = dip_pos[:, None, :] - np.asarray(run.positions, dtype=float)[None, :, :]
        dist = np.linalg.norm(dr, axis=2)
        inside = dist < np.asarray(run.circumscribing_radii, dtype=float)[None, :]
        if np.any(inside):
            j, i = np.argwhere(inside)[0]
            msg = (
                "Dipole index "
                f"{int(j)} lies inside particle index {int(i)}. "
                "This is currently untested in pyceles and may be unreliable."
            )
            if not bool(allow_inside_particle):
                raise ValueError(msg + " Pass `allow_inside_particle=True` to proceed explicitly.")
            warnings.warn(msg, UserWarning, stacklevel=2)

    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=run.config.compute_dtype,
        accum_dtype=run.config.accum_dtype,
    )
    # Dipole power/LDOS sampling is downstream postprocessing, so it should
    # inherit the same backend policy as far-field / near-field workflows.
    # This keeps local emitter studies on the CuPy scattered-field path when
    # the run was configured for GPU postprocessing.
    E_scat, _ = compute_scattered_field(
        dip_pos,
        np.asarray(run.positions, dtype=float),
        np.asarray(run.coeffs),
        k=float(run.k),
        lmax=int(run.config.lmax),
        n_medium=complex(run.config.n_medium),
        particle_distance_resolution=float(run.config.radial_lut_dr),
        show_progress=bool(show_progress),
        backend=run.config.resolved_postprocessing_backend(),
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    E_scat = np.asarray(E_scat, dtype=accum_dtype).reshape(-1, 3)

    p_dot_e = np.sum(np.conjugate(dip_mom) * E_scat, axis=1)
    delta_power = 0.5 * float(run.k0) * np.imag(p_dot_e)
    mu2 = np.sum(np.abs(dip_mom) ** 2, axis=1)
    p0 = mu2 * float(run.k) * (float(run.k0) ** 3) / (12.0 * np.pi)
    p_total = p0 + delta_power
    enhancement = np.divide(
        p_total,
        p0,
        out=np.full_like(p0, np.nan, dtype=float),
        where=p0 > 0.0,
    )

    return DipolePowerLDOSResult(
        dipole_positions=dip_pos,
        dipole_moments=dip_mom,
        E_scattered_at_dipoles=E_scat,
        delta_power=np.asarray(delta_power, dtype=float),
        power_homogeneous=np.asarray(p0, dtype=float),
        power_total=np.asarray(p_total, dtype=float),
        enhancement=np.asarray(enhancement, dtype=float),
    )


def compute_dipole_ldos_enhancement(
    run: SimulationResult,
    *,
    channel: Literal["mixed"] = "mixed",
    allow_inside_particle: bool = False,
    show_progress: bool = False,
    squeeze: bool = True,
) -> float | np.ndarray:
    """Return `P/P0` enhancement from `compute_dipole_power_ldos`.

    For one dipole and `squeeze=True`, returns a scalar float.
    """
    out = compute_dipole_power_ldos(
        run,
        channel=channel,
        allow_inside_particle=allow_inside_particle,
        show_progress=show_progress,
    ).enhancement
    if squeeze and out.size == 1:
        return float(out[0])
    return out
