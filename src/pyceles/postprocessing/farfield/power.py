from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles._optional import import_cupy, is_cupy_array
from pyceles.core.sources import Source, ensure_finite_power_diagnostics_supported

from .common import integrate_periodic_alpha
from .patterns import total_field_plane_wave_pattern


@dataclass(frozen=True, slots=True)
class PowerBalance:
    """Common local and flux power accounting for simulation results.

    ``flux_defect`` is the raw flux balance
    ``incident - reflected - transmitted``. ``closure_error`` subtracts the
    local exciting/scattered-coefficient absorption estimate. It is therefore
    the residual between two independent power identities and can expose
    coupling-operator approximation, far-field/quadrature, basis-truncation,
    or normalization inconsistency. For an inexact solve,
    ``local_absorbed_power`` also contains the solve-equation defect; this is
    intentional because it makes convergence failures visible instead of
    silently forcing nominally lossless materials to zero absorption.

    Dipole workflows may provide only ``local_absorbed_power`` and its
    per-particle decomposition. Flux-derived properties then return ``None``.
    """

    incident_power: float | None = None
    reflected_power: float | None = None
    transmitted_power: float | None = None
    local_absorbed_power: float | None = None
    local_absorbed_power_per_particle: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "incident_power",
            "reflected_power",
            "transmitted_power",
            "local_absorbed_power",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"`{name}` must be finite when provided. Got {value!r}.")
            object.__setattr__(self, name, scalar)
        if self.incident_power is not None and float(self.incident_power) <= 0.0:
            raise ValueError("`incident_power` must be positive when provided.")
        if (self.reflected_power is None) != (self.transmitted_power is None):
            raise ValueError("`reflected_power` and `transmitted_power` must be provided together.")
        if self.reflected_power is not None and self.incident_power is None:
            raise ValueError(
                "`incident_power` is required when reflected/transmitted powers are provided."
            )
        if self.local_absorbed_power_per_particle is not None:
            if self.local_absorbed_power is None:
                raise ValueError(
                    "`local_absorbed_power` is required with a per-particle decomposition."
                )
            values = np.array(
                self.local_absorbed_power_per_particle,
                dtype=np.float64,
                copy=True,
            ).reshape(-1)
            if not bool(np.all(np.isfinite(values))):
                raise ValueError("Per-particle local absorbed powers must be finite.")
            values.setflags(write=False)
            object.__setattr__(self, "local_absorbed_power_per_particle", values)

    @property
    def has_flux_balance(self) -> bool:
        """Return whether incident/reflected/transmitted powers are available."""
        return self.incident_power is not None and self.reflected_power is not None

    @property
    def flux_defect(self) -> float | None:
        """Return ``incident - reflected - transmitted`` in power units."""
        if not self.has_flux_balance:
            return None
        assert self.incident_power is not None
        assert self.reflected_power is not None
        assert self.transmitted_power is not None
        return float(self.incident_power - self.reflected_power - self.transmitted_power)

    @property
    def closure_error(self) -> float | None:
        """Return ``flux_defect - local_absorbed_power`` in power units."""
        defect = self.flux_defect
        if defect is None or self.local_absorbed_power is None:
            return None
        return float(defect - self.local_absorbed_power)

    @property
    def reflectance(self) -> float | None:
        """Return reflected/incident power when flux data are available."""
        if self.reflected_power is None or self.incident_power is None:
            return None
        return float(self.reflected_power / self.incident_power)

    @property
    def transmittance(self) -> float | None:
        """Return transmitted/incident power when flux data are available."""
        if self.transmitted_power is None or self.incident_power is None:
            return None
        return float(self.transmitted_power / self.incident_power)

    @property
    def local_absorptance(self) -> float | None:
        """Return local absorbed/incident power when both are available."""
        if self.local_absorbed_power is None or self.incident_power is None:
            return None
        return float(self.local_absorbed_power / self.incident_power)

    @property
    def flux_defect_fraction(self) -> float | None:
        """Return the raw flux defect normalized by incident power."""
        defect = self.flux_defect
        if defect is None or self.incident_power is None:
            return None
        return float(defect / self.incident_power)

    @property
    def closure_error_fraction(self) -> float | None:
        """Return the power-closure error normalized by incident power."""
        closure = self.closure_error
        if closure is None or self.incident_power is None:
            return None
        return float(closure / self.incident_power)

    @property
    def local_absorptance_per_particle(self) -> np.ndarray | None:
        """Return per-particle local absorbed-power fractions when available."""
        if self.local_absorbed_power_per_particle is None or self.incident_power is None:
            return None
        return np.asarray(
            self.local_absorbed_power_per_particle / self.incident_power,
            dtype=np.float64,
        )

    def to_mapping(self) -> dict[str, float | np.ndarray | None]:
        """Return the canonical serialization mapping without redundant arrays.

        Scalar derived diagnostics are included for readable output files. The
        normalized per-particle vector is intentionally omitted because it is
        exactly derivable from ``local_absorbed_power_per_particle`` and
        ``incident_power`` and can otherwise double large diagnostic payloads.
        """
        return {
            "incident_power": self.incident_power,
            "reflected_power": self.reflected_power,
            "transmitted_power": self.transmitted_power,
            "local_absorbed_power": self.local_absorbed_power,
            "local_absorbed_power_per_particle": self.local_absorbed_power_per_particle,
            "flux_defect": self.flux_defect,
            "closure_error": self.closure_error,
            "reflectance": self.reflectance,
            "transmittance": self.transmittance,
            "local_absorptance": self.local_absorptance,
            "flux_defect_fraction": self.flux_defect_fraction,
            "closure_error_fraction": self.closure_error_fraction,
        }


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
    power = local_power_balance_from_exciting(
        exciting_coeffs,
        scattered_coeffs,
        k0=k0,
        n_medium=n_medium,
    )
    assert power.local_absorbed_power is not None
    return float(power.local_absorbed_power)


def local_power_balance_from_exciting(
    exciting_coeffs: np.ndarray,
    scattered_coeffs: np.ndarray,
    *,
    k0: float,
    n_medium: complex,
    n_particles: int | None = None,
    nmodes_per_particle: int | None = None,
) -> PowerBalance:
    """Return a local-only power balance with optional particle decomposition."""
    if (n_particles is None) != (nmodes_per_particle is None):
        raise ValueError(
            "`n_particles` and `nmodes_per_particle` must be provided together "
            "for per-particle diagnostics."
        )
    ns = None if n_particles is None else int(n_particles)
    nm = None if nmodes_per_particle is None else int(nmodes_per_particle)
    if ns is not None and nm is not None and (ns < 0 or nm < 0):
        raise ValueError("`n_particles` and `nmodes_per_particle` must be nonnegative.")

    k_medium, n_real = _validate_power_normalization_inputs(k0=k0, n_medium=n_medium)
    pref = (np.pi * n_real) / (2.0 * (k_medium**2))
    particle_power: np.ndarray | None = None

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
        total_power = float(cupy.asnumpy(pref * (-term_ex - term_xx)))
        if ns is not None and nm is not None:
            if ns == 0:
                particle_power = np.zeros((0,), dtype=np.float64)
            elif int(e_gpu.size) != ns * nm:
                raise ValueError(
                    "Flattened coefficient size does not match requested "
                    "`(n_particles, nmodes_per_particle)` shape. "
                    f"Got size {int(e_gpu.size)} vs {ns * nm}."
                )
            else:
                e_mat = e_gpu.reshape(ns, nm)
                x_mat = x_gpu.reshape(ns, nm)
                ex = cupy.real(cupy.sum(cupy.conj(e_mat) * x_mat, axis=1))
                xx = cupy.real(cupy.sum(cupy.conj(x_mat) * x_mat, axis=1))
                particle_power = np.asarray(
                    cupy.asnumpy(pref * (-ex - xx)),
                    dtype=np.float64,
                )
        return PowerBalance(
            local_absorbed_power=total_power,
            local_absorbed_power_per_particle=particle_power,
        )

    e = np.asarray(exciting_coeffs, dtype=np.complex128).reshape(-1)
    x = np.asarray(scattered_coeffs, dtype=np.complex128).reshape(-1)
    if e.shape != x.shape:
        raise ValueError(
            "`exciting_coeffs` and `scattered_coeffs` must have matching shapes. "
            f"Got {e.shape} and {x.shape}."
        )
    term_ex = float(np.real(np.vdot(e, x)))
    term_xx = float(np.real(np.vdot(x, x)))
    total_power = float(pref * (-term_ex - term_xx))
    if ns is not None and nm is not None:
        if ns == 0:
            particle_power = np.zeros((0,), dtype=np.float64)
        elif int(e.size) != ns * nm:
            raise ValueError(
                "Flattened coefficient size does not match requested "
                "`(n_particles, nmodes_per_particle)` shape. "
                f"Got size {int(e.size)} vs {ns * nm}."
            )
        else:
            e_mat = e.reshape(ns, nm)
            x_mat = x.reshape(ns, nm)
            ex = np.real(np.sum(np.conj(e_mat) * x_mat, axis=1))
            xx = np.real(np.sum(np.conj(x_mat) * x_mat, axis=1))
            particle_power = np.asarray(pref * (-ex - xx), dtype=np.float64)
    return PowerBalance(
        local_absorbed_power=total_power,
        local_absorbed_power_per_particle=particle_power,
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


def finite_beam_power_balance(
    source: Source,
    initial_pwp_te: dict,
    initial_pwp_tm: dict,
    scattered_pwp_te: dict,
    scattered_pwp_tm: dict,
    *,
    k0: float,
    k_medium: float,
    local_absorbed_power: float | None = None,
    local_absorbed_power_per_particle: np.ndarray | None = None,
) -> PowerBalance:
    """Compute common power accounting for a finite-power beam."""
    ensure_finite_power_diagnostics_supported(
        source,
        diagnostic="Finite-beam power balance",
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

    return PowerBalance(
        incident_power=float(p_initial),
        transmitted_power=float(p_transmitted),
        reflected_power=float(p_reflected),
        local_absorbed_power=(
            None if local_absorbed_power is None else float(local_absorbed_power)
        ),
        local_absorbed_power_per_particle=local_absorbed_power_per_particle,
    )


__all__ = [
    "PowerBalance",
    "finite_beam_power_balance",
    "incident_power_from_pwp",
    "local_absorbed_power_from_exciting",
    "local_power_balance_from_exciting",
    "pwp_power_decomposition",
    "pwp_power_flux",
]
