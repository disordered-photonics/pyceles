from __future__ import annotations

import numpy as np

from pyceles.core.sources import PlaneWave, Source

from .common import integrate_periodic_alpha


def _validate_plane_wave_cross_section_inputs(
    source: Source,
    *,
    k0: float,
    n_medium: complex,
) -> tuple[float, float]:
    """Validate plane-wave cross-section normalization inputs."""
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
    """Return the dimensionless incident-field scale for plane-wave normalization."""
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
    """Differential scattering cross section from scattered PWPs."""
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
    """Total scattering cross section from far-field PWPs."""
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
    int_alpha = integrate_periodic_alpha(total * np.sin(beta)[None, :], alpha)
    return float(np.trapezoid(int_alpha, beta))


def extinction_cross_section(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Extinction cross section from solved incident/scattered coefficients."""
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
    """Absorption cross section as `C_ext - C_sca`."""
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
    """Return plane-wave cross sections with SMUTHI-style cluster scattering."""
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


__all__ = [
    "absorption_cross_section",
    "extinction_cross_section",
    "plane_wave_cross_sections",
    "scattering_cross_section",
    "total_scattering_cross_section",
]
