from __future__ import annotations

import numpy as np

from pyceles.core.sources import Source, ensure_finite_power_diagnostics_supported

from .common import integrate_periodic_alpha
from .patterns import total_field_plane_wave_pattern


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
    """Decompose power into initial/scattered/interference/total terms."""
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
    p_initial_te = pwp_power_flux(initial_pwp_te, k0=k0, k_medium=k_medium, direction=direction)
    p_initial_tm = pwp_power_flux(initial_pwp_tm, k0=k0, k_medium=k_medium, direction=direction)
    p_initial = p_initial_te + p_initial_tm

    p_scattered_te = pwp_power_flux(scattered_pwp_te, k0=k0, k_medium=k_medium, direction=direction)
    p_scattered_tm = pwp_power_flux(scattered_pwp_tm, k0=k0, k_medium=k_medium, direction=direction)
    p_scattered = p_scattered_te + p_scattered_tm

    p_total_te = pwp_power_flux(total_te, k0=k0, k_medium=k_medium, direction=direction)
    p_total_tm = pwp_power_flux(total_tm, k0=k0, k_medium=k_medium, direction=direction)
    p_total = p_total_te + p_total_tm
    p_interference = p_total - p_initial - p_scattered

    return {
        "P_initial": float(p_initial),
        "P_scattered": float(p_scattered),
        "P_interference": float(p_interference),
        "P_total": float(p_total),
    }


def pwp_power_flux(
    pwp: dict,
    *,
    k0: float,
    k_medium: float,
    direction: str,
) -> float:
    """Power flux through one hemisphere from a single-polarization PWP."""
    alpha = np.asarray(pwp["alpha"], dtype=float)
    beta = np.asarray(pwp["beta"], dtype=float)
    g = np.asarray(pwp["coeff"])

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")

    cb = np.cos(beta)
    mask = cb > 0 if direction == "forward" else cb < 0
    beta_m = beta[mask]
    g_m = g[:, mask]

    integrand = np.sin(beta_m)[None, :] * (np.abs(g_m) ** 2)
    int_alpha = integrate_periodic_alpha(integrand, alpha)
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
    """Incident beam power from the initial TE/TM PWPs."""
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
    """Compute transmitted/reflected powers and fractions for finite-power beams."""
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
    p_transmitted_te = pwp_power_flux(total_te, k0=k0, k_medium=k_medium, direction="forward")
    p_transmitted_tm = pwp_power_flux(total_tm, k0=k0, k_medium=k_medium, direction="forward")
    p_transmitted = p_transmitted_te + p_transmitted_tm

    p_reflected_te = pwp_power_flux(
        scattered_pwp_te, k0=k0, k_medium=k_medium, direction="backward"
    )
    p_reflected_tm = pwp_power_flux(
        scattered_pwp_tm, k0=k0, k_medium=k_medium, direction="backward"
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


__all__ = [
    "finite_beam_power_fractions",
    "incident_power_from_pwp",
    "pwp_power_decomposition",
    "pwp_power_flux",
]
