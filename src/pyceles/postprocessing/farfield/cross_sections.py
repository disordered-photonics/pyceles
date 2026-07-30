from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.sources import PlaneWave, Source

from .common import integrate_periodic_alpha


@dataclass(frozen=True, slots=True)
class CrossSectionBalance:
    """Plane-wave cross sections with an explicit closure diagnostic.

    Values use the simulation length unit squared. ``local_absorption`` is an
    independent local estimate; ``absorption_by_difference`` is the global
    ``extinction - scattering`` estimator. Their difference is retained as
    ``closure_error`` rather than being hidden behind an absorption alias.
    """

    extinction: float
    scattering: float
    local_absorption: float

    def __post_init__(self) -> None:
        for name in ("extinction", "scattering", "local_absorption"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"`{name}` must be finite. Got {value!r}.")
            object.__setattr__(self, name, value)

    @property
    def absorption_by_difference(self) -> float:
        """Return the global estimator ``extinction - scattering``."""
        return float(self.extinction - self.scattering)

    @property
    def closure_error(self) -> float:
        """Return ``absorption_by_difference - local_absorption``."""
        return float(self.absorption_by_difference - self.local_absorption)

    def to_mapping(self) -> dict[str, float]:
        """Return the canonical serialization mapping."""
        return {
            "extinction": float(self.extinction),
            "scattering": float(self.scattering),
            "local_absorption": float(self.local_absorption),
            "absorption_by_difference": self.absorption_by_difference,
            "closure_error": self.closure_error,
        }


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
    interference = float(np.vdot(e, x).real)
    scattered_norm = float(np.vdot(x, x).real)
    return pref * (-interference - scattered_norm)


def plane_wave_cross_section_balance(
    source: Source,
    initial_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    n_medium: complex,
    local_absorption: float,
) -> CrossSectionBalance:
    """Return plane-wave cross sections and their closure discrepancy.

    Extinction and scattering are independent global estimators. The caller
    supplies the local dissipation estimate explicitly, so the result cannot
    silently relabel a global flux difference as local absorption.
    """
    extinction = extinction_cross_section(
        source,
        initial_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    scattering = total_scattering_cross_section(
        source,
        scattered_pwp_te,
        scattered_pwp_tm,
        k0=k0,
        n_medium=n_medium,
    )
    return CrossSectionBalance(
        extinction=float(extinction),
        scattering=float(scattering),
        local_absorption=float(local_absorption),
    )


__all__ = [
    "CrossSectionBalance",
    "extinction_cross_section",
    "local_absorption_cross_section_from_exciting",
    "plane_wave_cross_section_balance",
    "scattering_cross_section",
    "total_scattering_cross_section",
]
