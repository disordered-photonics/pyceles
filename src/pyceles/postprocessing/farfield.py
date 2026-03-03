"""Far-field utilities for angular spectra, power fluxes, and cross sections.

The routines here evaluate scattered/initial/total plane-wave patterns on an
(`alpha`, `beta`) grid, integrate hemisphere power flow, and derive
plane-wave-normalized scattering observables.

Naming note: this module uses `k0 = 2*pi/lambda` for the vacuum wavenumber.
Older CELES-style formulas often call this quantity `omega`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyceles.core.conversions import svwf_outgoing_to_pwp
from pyceles.core.sources import (
    AngularSpectrumSource,
    DipoleCollection,
    DipoleSource,
    PlaneWave,
    Source,
    ensure_finite_power_diagnostics_supported,
)


@dataclass(frozen=True)
class FarFieldPatterns:
    """TE/TM plane-wave spectra on the common `(alpha,beta)` angular grid.

    Conventions:
    - `alpha`: azimuth in `[0, 2*pi)`
    - `beta`: polar angle from `+z` in `[0, pi]`
    - `coeff`: complex spectrum `g(alpha,beta)` per polarization
    """

    initial_te: dict | None
    initial_tm: dict | None
    scattered_te: dict
    scattered_tm: dict
    total_te: dict | None
    total_tm: dict | None


def _integrate_periodic_alpha(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Integrate azimuthal samples while enforcing 2*pi periodic closure."""
    values = np.asarray(values)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.size < 2:
        return np.zeros(values.shape[1:], dtype=np.result_type(values, np.float64))
    span = alpha[-1] - alpha[0]
    if np.isclose(span, 2.0 * np.pi):
        return np.trapezoid(values, alpha, axis=0)
    alpha_ext = np.concatenate([alpha, [alpha[0] + 2.0 * np.pi]])
    values_ext = np.concatenate([values, values[0:1, ...]], axis=0)
    return np.trapezoid(values_ext, alpha_ext, axis=0)


def _cast_pwp_coeff_dtype(pwp: dict, dtype: np.dtype) -> dict:
    """Return a shallow-copied PWP dict with `coeff` cast to dtype."""
    out = dict(pwp)
    # NumPy 2 raises if `copy=False` is requested but a copy is required.
    # Here we allow a copy when dtype conversion/layout demands it.
    out["coeff"] = np.asarray(out["coeff"], dtype=np.dtype(dtype))
    return out


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
    """Compute far-field plane-wave pattern (PWP) of the scattered field.

    Parameters
    ----------
    positions:
        (Ns,3) sphere centers.
    coeffs:
        (Ns, Nm) scattered coefficients, CELES ordering.
    k:
        Medium wavenumber k_medium.
    lmax:
        Truncation.
    polar_angles, azimuthal_angles:
        1D arrays of beta and alpha.

    Returns
    -------
    pwp_te, pwp_tm
        Each is a dict with keys: beta, alpha, kx, ky, kz, coeff.
        `coeff` has shape (Na, Nb).
    """

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
    """Combine initial + scattered PWPs into total-field PWP."""

    # shallow copies (keep grids)
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
    """Compute far-field PWPs for scattered field and, when available, initial/total fields.

    All returned angular grids/components are expressed in the user/source frame.
    (Near-field internals may use rotated local frames for acceleration, but
    far-field bins are not rotated or re-labeled.)
    This mirrors the standard workflow: scattered PWP first, then initial PWP
    (if source supports it), then coherent total-field composition.
    For dipole sources, `initial_*` denotes the direct dipole-emission PWP
    and `total_* = initial_* + scattered_*`.

    Notes
    -----
    `polar_angles`/`azimuthal_angles` here define output sampling bins for
    far-field PWPs. In principle they can be chosen independently from the source-projection
    quadrature grid used to build the linear-system RHS.
    """
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
        p_i_te = _cast_pwp_coeff_dtype(p_i_te, ctype)
        p_i_tm = _cast_pwp_coeff_dtype(p_i_tm, ctype)
        p_t_te, p_t_tm = total_field_plane_wave_pattern(p_i_te, p_i_tm, p_s_te, p_s_tm)

    return FarFieldPatterns(
        initial_te=p_i_te,
        initial_tm=p_i_tm,
        scattered_te=p_s_te,
        scattered_tm=p_s_tm,
        total_te=p_t_te,
        total_tm=p_t_tm,
    )


def pwp_power_decomposition(
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    k_medium: float,
    direction: str = "forward",
    source: Source | None = None,
) -> dict[str, float]:
    """Decompose power into initial/scattered/interference/total terms.

    The decomposition is computed for a selected hemisphere:
      P_total = P_initial + P_scattered + P_interference

    When `source` is provided, this diagnostic is allowed only for sources
    with finite incident power (see `Source.has_finite_incident_power()`).
    """
    if source is not None:
        ensure_finite_power_diagnostics_supported(
            source,
            diagnostic="Power decomposition",
        )
    total_te, total_tm = total_field_plane_wave_pattern(
        initial_pwp_te,
        initial_pwp_tm,
        scattered_pwp_te,
        scattered_pwp_tm,
    )
    p_initial_te = pwp_power_flux(
        initial_pwp_te,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    p_initial_tm = pwp_power_flux(
        initial_pwp_tm,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    P_initial = p_initial_te + p_initial_tm

    p_scattered_te = pwp_power_flux(
        scattered_pwp_te,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    p_scattered_tm = pwp_power_flux(
        scattered_pwp_tm,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    P_scattered = p_scattered_te + p_scattered_tm

    p_total_te = pwp_power_flux(
        total_te,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    p_total_tm = pwp_power_flux(
        total_tm,
        k0=k0,
        k_medium=k_medium,
        direction=direction,
    )
    P_total = p_total_te + p_total_tm
    P_interference = P_total - P_initial - P_scattered

    return {
        "P_initial": float(P_initial),
        "P_scattered": float(P_scattered),
        "P_interference": float(P_interference),
        "P_total": float(P_total),
    }


def pwp_power_flux(
    pwp: dict,
    *,
    k0: float,
    k_medium: float,
    direction: str,
) -> float:
    """Power flux through one hemisphere from a single-polarization PWP.

    Parameters
    ----------
    pwp:
        Plane wave pattern dict (keys: alpha, beta, coeff).
    k0:
        Vacuum wavenumber `2*pi/lambda` (CELES-style code often names this `omega`).
    k_medium:
        Medium wavenumber k0*n_medium.
    direction:
        'forward' or 'backward'. Forward corresponds to cos(beta) >= 0.

    Returns
    -------
    Power (float)
    """

    alpha = np.asarray(pwp["alpha"], dtype=float)
    beta = np.asarray(pwp["beta"], dtype=float)
    g = np.asarray(pwp["coeff"])

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")

    cb = np.cos(beta)
    # Use strict hemisphere splits to avoid double-counting beta=pi/2 samples.
    if direction == "forward":
        mask = cb > 0
    else:
        mask = cb < 0

    beta_m = beta[mask]
    g_m = g[:, mask]

    integrand = np.sin(beta_m)[None, :] * (np.abs(g_m) ** 2)

    # integrate alpha then beta (CELES ordering)
    int_alpha = _integrate_periodic_alpha(integrand, alpha)
    int_beta = np.trapezoid(int_alpha, beta_m)

    pref = 2 * np.pi**2 / (k0 * k_medium)
    return float(np.real(pref * int_beta))


def incident_power_from_pwp(
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    *,
    k0: float,
    k_medium: float,
) -> float:
    """Incident beam power from the initial TE/TM PWPs.

    This integrates the initial spectrum over both hemispheres. Unlike the old
    normal-incidence Gaussian closed form, this remains consistent for tilted
    beams because it uses the actual initial PWP provided to the solver.
    """
    p_initial = (
        pwp_power_flux(initial_pwp_te, k0=k0, k_medium=k_medium, direction="forward")
        + pwp_power_flux(initial_pwp_tm, k0=k0, k_medium=k_medium, direction="forward")
        + pwp_power_flux(initial_pwp_te, k0=k0, k_medium=k_medium, direction="backward")
        + pwp_power_flux(initial_pwp_tm, k0=k0, k_medium=k_medium, direction="backward")
    )
    if (not np.isfinite(p_initial)) or (p_initial <= 0.0):
        raise ValueError(
            "Incident power computed from initial PWPs is non-finite or non-positive; "
            "transmitted/reflected fractions are undefined."
        )
    return float(p_initial)


def finite_beam_power_fractions(
    source: Source,
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    k_medium: float,
) -> dict[str, float]:
    """Compute transmitted/reflected powers and fractions for finite-power beams.

    Normalization is always done by integrating the supplied initial PWP over
    solid angle, so the result is consistent for both normal and tilted beams.
    This is defined only when `source.has_finite_incident_power()` is true.
    """

    ensure_finite_power_diagnostics_supported(
        source,
        diagnostic="Finite-beam power fractions",
    )

    total_te, total_tm = total_field_plane_wave_pattern(
        initial_pwp_te,
        initial_pwp_tm,
        scattered_pwp_te,
        scattered_pwp_tm,
    )
    p_transmitted_te = pwp_power_flux(
        total_te,
        k0=k0,
        k_medium=k_medium,
        direction="forward",
    )
    p_transmitted_tm = pwp_power_flux(
        total_tm,
        k0=k0,
        k_medium=k_medium,
        direction="forward",
    )
    p_transmitted = p_transmitted_te + p_transmitted_tm

    p_reflected_te = pwp_power_flux(
        scattered_pwp_te,
        k0=k0,
        k_medium=k_medium,
        direction="backward",
    )
    p_reflected_tm = pwp_power_flux(
        scattered_pwp_tm,
        k0=k0,
        k_medium=k_medium,
        direction="backward",
    )
    p_reflected = p_reflected_te + p_reflected_tm

    p_initial = incident_power_from_pwp(
        initial_pwp_te,
        initial_pwp_tm,
        k0=k0,
        k_medium=k_medium,
    )

    return {
        "P_initial": float(p_initial),
        "P_transmitted": float(p_transmitted),
        "P_reflected": float(p_reflected),
        "T": float(p_transmitted / p_initial),
        "R": float(p_reflected / p_initial),
    }


def _validate_plane_wave_cross_section_inputs(
    source: Source,
    *,
    k0: float,
    n_medium: complex,
) -> tuple[float, float]:
    """Validate plane-wave cross-section normalization inputs.

    Returns
    -------
    n_real, k_medium
    """
    if not isinstance(source, PlaneWave):
        raise ValueError("Cross section only defined for PlaneWave excitation.")

    n_m = complex(n_medium)
    if abs(n_m.imag) > 0:
        raise ValueError("Cross section undefined for plane wave incident from absorbing medium.")
    n_real = float(np.real(n_m))
    if n_real <= 0:
        raise ValueError("n_medium must be positive and real for cross-section normalization.")

    k_medium = float(k0) * n_real
    return n_real, k_medium


def _plane_wave_incident_intensity_scale(source: Source) -> float:
    """Return the dimensionless incident-field scale ``|E0|^2 * (|a_te|^2 + |a_tm|^2)``.

    This captures how solved coefficient vectors scale with the source amplitude
    and Jones-vector magnitude. Cross sections must be normalized by this factor
    to remain intensity-independent.
    """
    if not isinstance(source, PlaneWave):
        raise ValueError("Cross section only defined for PlaneWave excitation.")

    a_te, a_tm = source.jones_coefficients()
    pol_norm2 = float(abs(a_te) ** 2 + abs(a_tm) ** 2)
    amp2 = float(abs(complex(source.amplitude)) ** 2)
    scale = amp2 * pol_norm2
    if (not np.isfinite(scale)) or (scale <= 0.0):
        raise ValueError(
            "Plane-wave incident intensity scale must be finite and positive for cross-section normalization."
        )
    return scale


def scattering_cross_section(
    source: Source,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
) -> dict[str, np.ndarray]:
    """Differential scattering cross section from scattered PWPs.

    Returns
    -------
    dict
        Keys are:
        - ``alpha``: azimuth samples
        - ``beta``: polar samples
        - ``te``: TE differential cross section, shape (Na, Nb)
        - ``tm``: TM differential cross section, shape (Na, Nb)
        - ``total``: ``te + tm``, shape (Na, Nb)

    Notes
    -----
    Cross sections are defined only for plane-wave illumination (finite incident
    intensity), with
    ``dC_sca/dOmega = I_sca(alpha,beta) / I_inc``.
    """
    n_real, k_medium = _validate_plane_wave_cross_section_inputs(
        source,
        k0=k0,
        n_medium=n_medium,
    )

    alpha = np.asarray(scattered_pwp_te["alpha"], dtype=float)
    beta = np.asarray(scattered_pwp_te["beta"], dtype=float)
    g_te = np.asarray(scattered_pwp_te["coeff"])
    g_tm = np.asarray(scattered_pwp_tm["coeff"])

    if g_te.shape != g_tm.shape:
        raise ValueError(
            "scattered TE/TM PWP coefficient arrays must have identical shapes. "
            f"Got {g_te.shape} and {g_tm.shape}."
        )

    # PWP intensity per solid angle for one polarization:
    # I_Omega = (2*pi^2 / (k0*k_medium)) * |g|^2
    # Differential scattering cross section:
    # dC_sca/dOmega = I_Omega / I_inc
    incident_scale = _plane_wave_incident_intensity_scale(source)
    initial_intensity = incident_scale * n_real / 2.0

    pref = (2.0 * np.pi**2) / (float(k0) * k_medium * initial_intensity)
    dcs_te = pref * (np.abs(g_te) ** 2)
    dcs_tm = pref * (np.abs(g_tm) ** 2)
    dcs_total = dcs_te + dcs_tm

    return {
        "alpha": alpha,
        "beta": beta,
        "te": np.asarray(dcs_te, dtype=float),
        "tm": np.asarray(dcs_tm, dtype=float),
        "total": np.asarray(dcs_total, dtype=float),
    }


def total_scattering_cross_section(
    source: Source,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Total scattering cross section from far-field PWPs.

    Cross section is defined only for plane-wave excitation.
    """
    dcs = scattering_cross_section(
        source,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    alpha = np.asarray(dcs["alpha"], dtype=float)
    beta = np.asarray(dcs["beta"], dtype=float)
    total = np.asarray(dcs["total"], dtype=float)

    int_alpha = _integrate_periodic_alpha(total * np.sin(beta)[None, :], alpha)
    return float(np.trapezoid(int_alpha, beta))


def extinction_cross_section(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Extinction cross section from solved incident/scattered coefficients.

    Notes
    -----
    In CELES SVWF normalization (unit-amplitude source):

    ``C_ext = -(pi / k^2) * Re(b^H x)``,

    where ``b`` are incident coefficients, ``x`` are scattered coefficients and
    ``k = k0 * n_medium``. For arbitrary source amplitude/Jones magnitude,
    this quantity is normalized by the incident intensity scale
    ``|E0|^2 * (|a_te|^2 + |a_tm|^2)``.
    """
    _, k_medium = _validate_plane_wave_cross_section_inputs(
        source,
        k0=k0,
        n_medium=n_medium,
    )
    incident_scale = _plane_wave_incident_intensity_scale(source)
    b = np.asarray(initial_coeffs, dtype=np.complex128).reshape(-1)
    x = np.asarray(scattered_coeffs, dtype=np.complex128).reshape(-1)
    if b.shape != x.shape:
        raise ValueError(
            "`initial_coeffs` and `scattered_coeffs` must have matching shapes. "
            f"Got {b.shape} and {x.shape}."
        )
    pref = np.pi / (k_medium**2 * incident_scale)
    return float(np.real(-pref * np.vdot(b, x)))


def absorption_cross_section(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Absorption cross section as ``C_ext - C_sca``.

    Cross sections follow the SMUTHI/cluster convention: ``C_sca`` is obtained
    from far-field integration, while ``C_ext`` is coefficient-based.
    """
    c_ext = extinction_cross_section(
        source,
        initial_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    c_sca = total_scattering_cross_section(
        source,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    return float(c_ext - c_sca)


def plane_wave_cross_sections(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
) -> dict[str, float]:
    """Return plane-wave cross sections with SMUTHI-style cluster scattering.

    Returns
    -------
    dict
        Contains:
        - ``C_ext`` (coefficient-based optical-theorem form)
        - ``C_sca`` (far-field-integrated cluster scattering cross section)
        - ``C_abs`` (``C_ext - C_sca``)

    Notes
    -----
    For multi-particle clusters solved in per-particle local SVWF bases, the
    physical scattering cross section is obtained from angular far-field
    integration, as done in SMUTHI.
    """
    c_ext = extinction_cross_section(
        source,
        initial_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    c_sca = total_scattering_cross_section(
        source,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    return {
        "C_ext": float(c_ext),
        "C_sca": float(c_sca),
        "C_abs": float(c_ext - c_sca),
    }
