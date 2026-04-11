from __future__ import annotations

from typing import Any

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.sources import Source, ensure_finite_power_diagnostics_supported

from .common import integrate_periodic_alpha
from .patterns import total_field_plane_wave_pattern


def _validate_power_normalization_inputs(*, k0: float, n_medium: complex) -> tuple[float, float]:
    """Validate finite-power normalization inputs and return `(k_medium, n_real)`."""
    n_m = complex(n_medium)
    if abs(n_m.imag) > 0:
        raise ValueError("Power diagnostics are undefined for absorbing embedding media.")
    n_real = float(np.real(n_m))
    if n_real <= 0:
        raise ValueError("n_medium must be positive and real for power normalization.")
    return float(k0) * n_real, n_real


def local_absorbed_power_from_exciting(
    exciting_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Local absorbed power from exciting/scattered SVWF coefficients.

    This evaluates

    `P_abs_local = (pi * n_m / (2 k_m^2)) * (-Re<e, x> - <x, x>)`

    where `e` are local exciting coefficients and `x` are solved scattered
    coefficients in the same flattened mode ordering.
    """
    components = local_absorbed_power_components_from_exciting(
        exciting_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    return float(components["P_abs_local"])


def local_absorbed_power_components_from_exciting(
    exciting_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
    n_particles: int | None = None,
    nmodes_per_particle: int | None = None,
) -> dict[str, Any]:
    """Return local absorbed power plus optional per-particle contributions."""
    k_medium, n_real = _validate_power_normalization_inputs(k0=k0, n_medium=n_medium)
    pref = (np.pi * n_real) / (2.0 * (k_medium**2))
    if is_cupy_array(exciting_coeffs) or is_cupy_array(scattered_coeffs):
        cupy, _ = import_cupy()
        e_gpu = cupy.asarray(exciting_coeffs, dtype=cupy.complex128).reshape(-1)
        x_gpu = cupy.asarray(scattered_coeffs, dtype=cupy.complex128).reshape(-1)
        if e_gpu.shape != x_gpu.shape:
            raise ValueError(
                "`exciting_coeffs` and `scattered_coeffs` must have matching shapes. "
                f"Got {e_gpu.shape} and {x_gpu.shape}."
            )
        term_ex = cupy.real(cupy.vdot(e_gpu, x_gpu))
        term_xx = cupy.real(cupy.vdot(x_gpu, x_gpu))
        out: dict[str, Any] = {
            "P_abs_local": float(cupy.asnumpy(pref * (-term_ex - term_xx))),
        }
        if (n_particles is None) != (nmodes_per_particle is None):
            raise ValueError(
                "`n_particles` and `nmodes_per_particle` must be provided together for per-particle diagnostics."
            )
        if n_particles is not None and nmodes_per_particle is not None:
            ns = int(n_particles)
            nm = int(nmodes_per_particle)
            if ns < 0 or nm < 0:
                raise ValueError("`n_particles` and `nmodes_per_particle` must be nonnegative.")
            if ns == 0:
                out["P_abs_local_particles"] = np.zeros((0,), dtype=np.float64)
            elif int(e_gpu.size) != ns * nm:
                raise ValueError(
                    "Flattened coefficient size does not match requested `(n_particles, nmodes_per_particle)` shape. "
                    f"Got size {int(e_gpu.size)} vs {ns * nm}."
                )
            else:
                e_mat = e_gpu.reshape(ns, nm)
                x_mat = x_gpu.reshape(ns, nm)
                ex = cupy.real(cupy.sum(cupy.conj(e_mat) * x_mat, axis=1))
                xx = cupy.real(cupy.sum(cupy.conj(x_mat) * x_mat, axis=1))
                out["P_abs_local_particles"] = np.asarray(
                    cupy.asnumpy(pref * (-ex - xx)),
                    dtype=np.float64,
                )
        return out

    e = np.asarray(exciting_coeffs, dtype=np.complex128).reshape(-1)
    x = np.asarray(scattered_coeffs, dtype=np.complex128).reshape(-1)
    if e.shape != x.shape:
        raise ValueError(
            "`exciting_coeffs` and `scattered_coeffs` must have matching shapes. "
            f"Got {e.shape} and {x.shape}."
        )
    term_ex = float(np.real(np.vdot(e, x)))
    term_xx = float(np.real(np.vdot(x, x)))
    out = {"P_abs_local": float(pref * (-term_ex - term_xx))}
    if (n_particles is None) != (nmodes_per_particle is None):
        raise ValueError(
            "`n_particles` and `nmodes_per_particle` must be provided together for per-particle diagnostics."
        )
    if n_particles is not None and nmodes_per_particle is not None:
        ns = int(n_particles)
        nm = int(nmodes_per_particle)
        if ns < 0 or nm < 0:
            raise ValueError("`n_particles` and `nmodes_per_particle` must be nonnegative.")
        if ns == 0:
            out["P_abs_local_particles"] = np.zeros((0,), dtype=np.float64)
        elif int(e.size) != ns * nm:
            raise ValueError(
                "Flattened coefficient size does not match requested `(n_particles, nmodes_per_particle)` shape. "
                f"Got size {int(e.size)} vs {ns * nm}."
            )
        else:
            e_mat = e.reshape(ns, nm)
            x_mat = x.reshape(ns, nm)
            ex = np.real(np.sum(np.conj(e_mat) * x_mat, axis=1))
            xx = np.real(np.sum(np.conj(x_mat) * x_mat, axis=1))
            out["P_abs_local_particles"] = np.asarray(pref * (-ex - xx), dtype=np.float64)
    return out


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
    local_absorbed_power: float | None = None,
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

    p_abs_raw_diff = float(p_initial - p_transmitted - p_reflected)
    out = {
        "P_initial": float(p_initial),
        "P_transmitted": float(p_transmitted),
        "P_reflected": float(p_reflected),
        "T": float(p_transmitted / p_initial),
        "R": float(p_reflected / p_initial),
        "P_abs_raw_diff": p_abs_raw_diff,
        "A_raw_diff": float(p_abs_raw_diff / p_initial),
    }
    if local_absorbed_power is not None:
        p_abs_local = float(local_absorbed_power)
        out["P_abs_local"] = p_abs_local
        out["A_local"] = float(p_abs_local / p_initial)
        out["Delta_power_closure"] = float(p_abs_raw_diff - p_abs_local)
    return out


__all__ = [
    "finite_beam_power_fractions",
    "incident_power_from_pwp",
    "local_absorbed_power_components_from_exciting",
    "local_absorbed_power_from_exciting",
    "pwp_power_decomposition",
    "pwp_power_flux",
]
