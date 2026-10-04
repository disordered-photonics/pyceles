from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Polarization = Literal["TE", "TM"]


def pure_polarization_label(
    a_te: complex,
    a_tm: complex,
) -> Polarization | None:
    """Classify exact single-channel support, without discarding a small component.

    The label does not imply unit amplitude: callers must retain the nonzero
    Jones coefficient, including its complex phase.
    """
    if a_tm == 0.0 and a_te != 0.0:
        return "TE"
    if a_te == 0.0 and a_tm != 0.0:
        return "TM"
    return None


def normalize_global_polarization_vector(
    vector: tuple[complex, complex, complex] | npt.ArrayLike,
) -> np.ndarray:
    """Normalize one global Cartesian polarization vector to unit norm.

    This helper is used by lab-frame polarized angular-spectrum sources where
    `amplitude` sets overall field strength and the polarization vector carries
    only direction/relative phase information.
    """
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            "`global_polarization` must have exactly 3 entries. "
            f"Got shape {np.asarray(vector).shape}."
        )
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError("`global_polarization` must contain only finite values.")
    # Scale real components before the norm: a finite direction may have an
    # unrepresentable norm, or a norm whose square underflows. Avoid complex
    # magnitude/division here too, including for subnormal input components.
    scale = float(max(np.max(np.abs(arr.real)), np.max(np.abs(arr.imag))))
    if scale == 0.0:
        raise ValueError("`global_polarization` must not be the zero vector.")
    scaled = np.empty(3, dtype=np.complex128)
    np.divide(arr.real, scale, out=scaled.real)
    np.divide(arr.imag, scale, out=scaled.imag)
    scaled /= np.linalg.norm(scaled)
    return scaled


def project_global_cartesian_to_te_tm(
    *,
    global_polarization: tuple[complex, complex, complex] | npt.ArrayLike,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    ephi_g: np.ndarray,
    etheta_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one lab-frame polarization state onto TE/TM per propagation ray.

    For each ray direction `s=(sx,sy,sz)`, the input global vector is first
    projected onto the transverse plane (`E.k=0`), then decomposed on the
    global spherical unit vectors (`e_phi`, `e_theta`) that define TE/TM.
    """
    p = normalize_global_polarization_vector(global_polarization)
    sx_arr = np.asarray(sx, dtype=float)
    sy_arr = np.asarray(sy, dtype=float)
    sz_arr = np.asarray(sz, dtype=float)
    ephi = np.asarray(ephi_g, dtype=float)
    etheta = np.asarray(etheta_g, dtype=float)

    if ephi.shape[-1] != 3 or etheta.shape[-1] != 3:
        raise ValueError("`ephi_g` and `etheta_g` must have trailing dimension 3.")
    if ephi.shape != etheta.shape:
        raise ValueError("`ephi_g` and `etheta_g` must have identical shapes.")
    if sx_arr.shape != sy_arr.shape or sx_arr.shape != sz_arr.shape:
        raise ValueError("`sx`, `sy`, and `sz` must have identical shapes.")
    if ephi.shape[:-1] != sx_arr.shape:
        raise ValueError("`sx/sy/sz` must match `ephi_g` and `etheta_g` leading dimensions.")

    dot_ps = p[0] * sx_arr + p[1] * sy_arr + p[2] * sz_arr
    ex_t = p[0] - dot_ps * sx_arr
    ey_t = p[1] - dot_ps * sy_arr
    ez_t = p[2] - dot_ps * sz_arr

    g_te = ex_t * ephi[..., 0] + ey_t * ephi[..., 1] + ez_t * ephi[..., 2]
    g_tm = ex_t * etheta[..., 0] + ey_t * etheta[..., 1] + ez_t * etheta[..., 2]
    return np.asarray(g_te, dtype=np.complex128), np.asarray(g_tm, dtype=np.complex128)
