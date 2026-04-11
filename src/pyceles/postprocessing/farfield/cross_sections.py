from __future__ import annotations

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
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


def local_absorption_cross_section_from_exciting(
    source: Source,
    exciting_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
) -> float:
    """Local dissipation estimate from exciting/scattered SVWF coefficients.

    This evaluates

    `C_abs_local = (pi/(k_m^2 I0)) * (-Re<e, x> - <x, x>)`

    where `e` are local exciting coefficients and `x` are solved scattered
    coefficients in the same flattened mode ordering.
    """
    _, k_medium = _validate_plane_wave_cross_section_inputs(
        source,
        k0=k0,
        n_medium=n_medium,
    )
    incident_scale = _plane_wave_incident_intensity_scale(source)
    pref = np.pi / (k_medium**2 * incident_scale)
    if is_cupy_array(exciting_coeffs) or is_cupy_array(scattered_coeffs):
        cupy, _ = import_cupy()
        e_gpu = cupy.asarray(exciting_coeffs, dtype=cupy.complex128).reshape(-1)
        x_gpu = cupy.asarray(scattered_coeffs, dtype=cupy.complex128).reshape(-1)
        if e_gpu.shape != x_gpu.shape:
            raise ValueError(
                "`exciting_coeffs` and `scattered_coeffs` must have matching shapes. "
                f"Got {e_gpu.shape} and {x_gpu.shape}."
            )
        val = pref * (-cupy.real(cupy.vdot(e_gpu, x_gpu)) - cupy.real(cupy.vdot(x_gpu, x_gpu)))
        return float(cupy.asnumpy(val))

    e = np.asarray(exciting_coeffs, dtype=np.complex128).reshape(-1)
    x = np.asarray(scattered_coeffs, dtype=np.complex128).reshape(-1)
    if e.shape != x.shape:
        raise ValueError(
            "`exciting_coeffs` and `scattered_coeffs` must have matching shapes. "
            f"Got {e.shape} and {x.shape}."
        )
    return float(pref * (-np.real(np.vdot(e, x)) - np.real(np.vdot(x, x))))


def plane_wave_cross_section_components(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
    local_absorption: float | None = None,
) -> dict[str, float]:
    """Return raw/local plane-wave cross-section components and closure defect.

    `C_ext_raw` and `C_sca_raw` keep the current global estimators. `C_abs_raw_diff`
    is their difference. `C_abs_local` can be supplied from a local exciting-field
    route (for example `e=b+W x`), and `Delta_closure` reports the residual
    mismatch between the two independent global estimators and the local
    dissipation estimate.
    """
    c_ext_raw = extinction_cross_section(
        source,
        initial_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    c_sca_raw = total_scattering_cross_section(
        source,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    c_abs_raw_diff = float(c_ext_raw - c_sca_raw)
    c_abs_local = c_abs_raw_diff if local_absorption is None else float(local_absorption)
    delta_closure = float(c_abs_raw_diff - c_abs_local)
    return {
        "C_ext_raw": float(c_ext_raw),
        "C_sca_raw": float(c_sca_raw),
        "C_abs_raw_diff": float(c_abs_raw_diff),
        "C_abs_local": float(c_abs_local),
        "Delta_closure": float(delta_closure),
    }


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
    """Raw-difference absorption estimator `C_ext_raw - C_sca_raw`.

    This helper keeps the legacy estimator available explicitly. For the
    physically local dissipation estimate, use
    `local_absorption_cross_section_from_exciting(...)` and pass it through
    `plane_wave_cross_sections(..., local_absorption=...)`.
    """
    components = plane_wave_cross_section_components(
        source,
        initial_coeffs,
        scattered_coeffs,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    return float(components["C_abs_raw_diff"])


def plane_wave_cross_sections(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
    local_absorption: float | None = None,
    allow_raw_diff_fallback: bool = False,
) -> dict[str, float]:
    """Return plane-wave cross sections with explicit raw/local decomposition.

    Public `C_abs` follows the local dissipation estimate (`C_abs_local`).
    The legacy raw-difference estimator remains available as `C_abs_raw_diff`,
    together with `Delta_closure = C_abs_raw_diff - C_abs_local`.

    By default this helper is strict about local semantics: callers must pass
    `local_absorption` explicitly. Legacy low-level fallback to
    `C_abs_local = C_abs_raw_diff` is available only when
    `allow_raw_diff_fallback=True`.
    """
    if local_absorption is None:
        b_arr = np.asarray(initial_coeffs)
        x_arr = np.asarray(scattered_coeffs)
        if int(b_arr.size) == 0 and int(x_arr.size) == 0:
            local_absorption = 0.0
    if local_absorption is None and not bool(allow_raw_diff_fallback):
        raise ValueError(
            "`plane_wave_cross_sections` requires `local_absorption` under the "
            "current local-absorption semantics. If you intentionally want the "
            "legacy raw-difference fallback, pass "
            "`allow_raw_diff_fallback=True` or call "
            "`plane_wave_cross_section_components(...)` directly."
        )
    components = plane_wave_cross_section_components(
        source,
        initial_coeffs,
        scattered_coeffs,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
        local_absorption=local_absorption,
    )
    return {
        "C_ext": float(components["C_ext_raw"]),
        "C_sca": float(components["C_sca_raw"]),
        "C_abs": float(components["C_abs_local"]),
        **components,
    }


__all__ = [
    "absorption_cross_section",
    "extinction_cross_section",
    "local_absorption_cross_section_from_exciting",
    "plane_wave_cross_section_components",
    "plane_wave_cross_sections",
    "scattering_cross_section",
    "total_scattering_cross_section",
]
