from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyceles.core.conversions import svwf_outgoing_to_pwp
from pyceles.core.sources import AngularSpectrumSource, DipoleCollection, DipoleSource, Source

from .common import cast_pwp_coeff_dtype


@dataclass(frozen=True)
class FarFieldPatterns:
    """TE/TM plane-wave spectra on the common `(alpha,beta)` angular grid."""

    initial_te: dict | None
    initial_tm: dict | None
    scattered_te: dict
    scattered_tm: dict
    total_te: dict | None
    total_tm: dict | None


def scattered_field_plane_wave_pattern(
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    *,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> tuple[dict, dict]:
    """Compute far-field plane-wave patterns of the scattered field."""
    return svwf_outgoing_to_pwp(
        positions=np.asarray(positions, dtype=float),
        coeffs=np.asarray(coeffs, dtype=np.dtype(dtype)),
        k=float(k),
        lmax=int(lmax),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        dtype=np.dtype(dtype),
        show_progress=bool(show_progress),
    )


def total_field_plane_wave_pattern(
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
) -> tuple[dict, dict]:
    """Combine initial and scattered PWPs into total-field PWPs."""
    tot_te = dict(initial_pwp_te)
    tot_tm = dict(initial_pwp_tm)
    tot_te["coeff"] = initial_pwp_te["coeff"] + scattered_pwp_te["coeff"]
    tot_tm["coeff"] = initial_pwp_tm["coeff"] + scattered_pwp_tm["coeff"]
    return tot_te, tot_tm


def compute_far_field_patterns(
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    source: Source | None = None,
    dtype: npt.DTypeLike = np.complex128,
    show_progress: bool = False,
) -> FarFieldPatterns:
    """Compute scattered PWPs and, when available, initial/total PWPs."""
    ctype = np.dtype(dtype)

    p_s_te, p_s_tm = scattered_field_plane_wave_pattern(
        positions=positions,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        dtype=ctype,
        show_progress=show_progress,
    )

    p_i_te = None
    p_i_tm = None
    p_t_te = None
    p_t_tm = None

    if isinstance(source, (DipoleSource, DipoleCollection)):
        dip_pos = np.asarray(source.dipole_positions(), dtype=float).reshape(-1, 3)
        dip_coeffs = np.asarray(source.outgoing_coeffs(1, dtype=ctype), dtype=ctype)
        p_i_te, p_i_tm = scattered_field_plane_wave_pattern(
            positions=dip_pos,
            coeffs=dip_coeffs,
            k=k,
            lmax=1,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            dtype=ctype,
            show_progress=False,
        )
        p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)
    elif isinstance(source, AngularSpectrumSource):
        p_i_te, p_i_tm = source.angular_spectrum(
            k=float(k),
            polar_angles=np.asarray(polar_angles, dtype=float),
            azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        )
        p_i_te = cast_pwp_coeff_dtype(p_i_te, ctype)
        p_i_tm = cast_pwp_coeff_dtype(p_i_tm, ctype)
        p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)

    return FarFieldPatterns(
        initial_te=p_i_te,
        initial_tm=p_i_tm,
        scattered_te=p_s_te,
        scattered_tm=p_s_tm,
        total_te=p_t_te,
        total_tm=p_t_tm,
    )


__all__ = [
    "FarFieldPatterns",
    "compute_far_field_patterns",
    "scattered_field_plane_wave_pattern",
    "total_field_plane_wave_pattern",
]
