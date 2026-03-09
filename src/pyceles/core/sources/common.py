from __future__ import annotations

import numpy as np
from scipy.special import eval_genlaguerre

from ..geometry_bounds import conservative_cross_set_max_distance
from ..indexing import index_vswf, n_modes
from ..translation import RadialLUT, translation_ab5_table, translation_block


def _as_complex_triplet(
    name: str, values: tuple[complex, complex, complex] | np.ndarray
) -> np.ndarray:
    """Normalize one 3-component complex vector and validate finiteness."""
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"`{name}` must have exactly 3 entries. Got shape {np.asarray(values).shape}."
        )
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr


def _as_float_triplet(name: str, values: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Normalize one 3-component float vector and validate finiteness."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"`{name}` must have exactly 3 entries. Got shape {np.asarray(values).shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return arr


def _normalize_dipole_collection_inputs(
    positions: np.ndarray,
    dipole_moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize dipole collection arrays."""
    pos = np.asarray(positions, dtype=float)
    mom = np.asarray(dipole_moments, dtype=np.complex128)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (Nd, 3). Got {pos.shape}.")
    if mom.ndim != 2 or mom.shape[1] != 3:
        raise ValueError(f"`dipole_moments` must have shape (Nd, 3). Got {mom.shape}.")
    if pos.shape[0] != mom.shape[0]:
        raise ValueError(
            "`positions` and `dipole_moments` must have the same first dimension. "
            f"Got {pos.shape[0]} and {mom.shape[0]}."
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("`positions` must contain only finite values.")
    if not np.all(np.isfinite(mom.real)) or not np.all(np.isfinite(mom.imag)):
        raise ValueError("`dipole_moments` must contain only finite values.")
    return pos, mom


def _dipole_outgoing_coeff_vector(
    *,
    lmax: int,
    k0: float,
    k_medium: float,
    dipole_moment: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Outgoing SVWF coefficients for one electric point dipole."""
    nm = n_modes(int(lmax))
    out = np.zeros((nm,), dtype=dtype)
    if int(lmax) < 1:
        return out

    mx, my, mz = dipole_moment
    c_xy = 1.0 / (2.0 * np.sqrt(3.0))
    c_z = 1.0 / np.sqrt(6.0)
    pref = (1j * float(k_medium) * (float(k0) ** 2)) / np.pi

    out[index_vswf(1, -1, 2, int(lmax))] = pref * c_xy * (mx + 1j * my)
    out[index_vswf(1, 0, 2, int(lmax))] = pref * c_z * mz
    out[index_vswf(1, 1, 2, int(lmax))] = pref * c_xy * (mx - 1j * my)
    return out


def _incident_coeffs_from_outgoing_expansion(
    *,
    receiver_positions: np.ndarray,
    source_positions: np.ndarray,
    outgoing_coeffs: np.ndarray,
    lmax: int,
    k_medium: float,
    radial_lut_dr: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Translate outgoing multipoles to regular SVWFs at receiver centers."""
    pos_rcv = np.asarray(receiver_positions, dtype=float)
    pos_src = np.asarray(source_positions, dtype=float)
    coeffs_src = np.asarray(outgoing_coeffs, dtype=dtype)
    dr_user = float(radial_lut_dr)
    if dr_user < 0.0:
        raise ValueError(f"`radial_lut_dr` must be >= 0. Got {radial_lut_dr!r}.")
    k_abs = float(abs(k_medium))
    if k_abs <= 0.0:
        raise ValueError(f"`k_medium` must be non-zero for radial LUT setup. Got {k_medium!r}.")
    ns = int(pos_rcv.shape[0])
    nm = n_modes(int(lmax))
    out = np.zeros((ns, nm), dtype=dtype)
    if ns == 0 or pos_src.shape[0] == 0:
        return out

    if np.any(np.all(np.isclose(pos_rcv[:, None, :], pos_src[None, :, :], atol=1e-12), axis=2)):
        raise ValueError(
            "Outgoing source center and receiver center coincide for at least one pair. "
            "Outgoing-to-regular translation is singular at zero separation."
        )

    ab5 = translation_ab5_table(int(lmax), dtype=dtype)
    r_max = conservative_cross_set_max_distance(pos_rcv, pos_src)
    radial_lut = RadialLUT(
        lmax=int(lmax),
        k=float(k_medium),
        r_max=float(r_max),
        dr=(1.0e-2 / k_abs) if dr_user == 0.0 else dr_user,
        dtype=dtype,
    )

    for j in range(pos_src.shape[0]):
        c_out = coeffs_src[j]
        for i in range(ns):
            rvec = pos_rcv[i] - pos_src[j]
            wij = translation_block(
                int(lmax),
                float(k_medium),
                rvec,
                ab5=ab5,
                radial_lut=radial_lut,
            )
            out[i] += np.asarray(wij @ c_out, dtype=dtype)

    return out


def _validated_int(name: str, value: int, *, minimum: int | None = None) -> int:
    """Validate one finite integer parameter (optionally with lower bound)."""
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be an integer. Got {value!r}.") from exc
    if not np.isfinite(value_f) or (not np.isclose(value_f, round(value_f), atol=0.0)):
        raise ValueError(f"`{name}` must be an integer. Got {value!r}.")
    value_i = int(round(value_f))
    if minimum is not None and value_i < int(minimum):
        raise ValueError(f"`{name}` must be >= {int(minimum)}. Got {value_i}.")
    return value_i


def _laguerre_profile_factor(
    *,
    radial_argument: np.ndarray,
    radial_order_p: int,
    azimuthal_order_l: int,
) -> np.ndarray:
    """Return Maxwellian Laguerre-Gaussian scalar profile P_pl(x)."""
    x = np.asarray(radial_argument, dtype=float)
    p = int(radial_order_p)
    l = int(azimuthal_order_l)
    l_abs = abs(l)
    poly = np.asarray(eval_genlaguerre(p, l_abs, 2.0 * (x**2)), dtype=float)
    return np.asarray(((-1.0) ** p) * ((np.sqrt(2.0) * x) ** l_abs) * poly * np.exp(-(x**2)))
