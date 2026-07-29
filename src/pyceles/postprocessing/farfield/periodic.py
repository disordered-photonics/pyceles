"""Periodic diffraction-order amplitudes and flux diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles.core.conversions import transformation_coefficients
from pyceles.core.indexing import iter_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.sources import PlaneWave
from pyceles.core.spherical import spherical_functions_trigon

from .orders import enumerate_diffraction_orders_rectangular


@dataclass(frozen=True)
class PeriodicFarFieldPayload:
    """Discrete periodic-order amplitudes and power diagnostics."""

    lattice_a1: np.ndarray
    lattice_a2: np.ndarray
    unit_cell_area: float
    incident_k_parallel: np.ndarray
    output_bmax: float | None
    order_mn: np.ndarray
    order_k_parallel: np.ndarray
    order_kz: np.ndarray
    order_propagating: np.ndarray
    reflected_amplitudes: np.ndarray
    transmitted_amplitudes: np.ndarray
    reflected_power_per_order: np.ndarray
    transmitted_power_per_order: np.ndarray
    incident_power_per_area: float
    reflectance: float
    transmittance: float
    absorptance_raw_diff: float
    local_absorptance: float | None = None
    power_closure_error: float | None = None

    @property
    def absorptance(self) -> float:
        """Return the historical alias for ``absorptance_raw_diff``.

        The canonical field is named ``absorptance_raw_diff`` because it is the
        flux deficit ``1 - R - T``. It is only a physical absorption estimate
        when the solved coefficients and periodic coupling satisfy the same
        power identity; ``local_absorptance`` and ``power_closure_error``
        separate those effects when available.
        """
        return float(self.absorptance_raw_diff)


@dataclass(frozen=True)
class PeriodicOrderAmplitudes:
    """Order-domain periodic amplitudes before flux normalization.

    The scattered payload is expressed on the two longitudinal branches:
    - `scattered_up_amplitudes`: order waves with `+kz`
    - `scattered_down_amplitudes`: order waves with `-kz`

    `reflected_amplitudes` and `transmitted_amplitudes` are derived from the
    branch amplitudes plus incidence direction bookkeeping, and include the
    incident `(m, n) = (0, 0)` component in `transmitted_amplitudes`.
    """

    order_mn: np.ndarray
    order_k_parallel: np.ndarray
    order_kz: np.ndarray
    order_propagating: np.ndarray
    scattered_up_amplitudes: np.ndarray
    scattered_down_amplitudes: np.ndarray
    reflected_amplitudes: np.ndarray
    transmitted_amplitudes: np.ndarray
    incident_polarization: np.ndarray
    incidence_sign: float
    incident_k_parallel: np.ndarray
    unit_cell_area: float
    output_bmax: float | None


def _order_mode_coefficients(
    *,
    lmax: int,
    k: float,
    k_parallel: np.ndarray,
    kz: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-mode TE/TM order coefficients for `+kz` and `-kz` branches."""
    kp = float(np.linalg.norm(np.asarray(k_parallel, dtype=float).reshape(2)))
    alpha = float(np.arctan2(float(k_parallel[1]), float(k_parallel[0])))
    st = np.asarray(kp / float(k), dtype=np.complex128)
    ct_up = np.asarray(complex(kz) / float(k), dtype=np.complex128)
    ct_down = -ct_up

    pi_up, tau_up = spherical_functions_trigon(ct_up, st, int(lmax), xp=np)
    pi_down, tau_down = spherical_functions_trigon(ct_down, st, int(lmax), xp=np)
    nmodes = sum(1 for _ in iter_modes(int(lmax)))
    b_up_te = np.zeros((nmodes,), dtype=np.complex128)
    b_up_tm = np.zeros((nmodes,), dtype=np.complex128)
    b_down_te = np.zeros((nmodes,), dtype=np.complex128)
    b_down_tm = np.zeros((nmodes,), dtype=np.complex128)

    for tau, l, m, idx in iter_modes(int(lmax)):
        phase = np.exp(1j * float(m) * alpha)
        b_up_te[idx] = (
            transformation_coefficients(pi_up, tau_up, tau, l, m, pol=1, dagger=False) * phase
        )
        b_up_tm[idx] = (
            transformation_coefficients(pi_up, tau_up, tau, l, m, pol=2, dagger=False) * phase
        )
        b_down_te[idx] = (
            transformation_coefficients(pi_down, tau_down, tau, l, m, pol=1, dagger=False) * phase
        )
        b_down_tm[idx] = (
            transformation_coefficients(pi_down, tau_down, tau, l, m, pol=2, dagger=False) * phase
        )
    return b_up_te, b_up_tm, b_down_te, b_down_tm


def periodic_order_amplitudes(
    *,
    source: PlaneWave,
    lattice: RectangularLattice2D,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    output_bmax: float | None = None,
) -> PeriodicOrderAmplitudes:
    """Compute branch-resolved periodic order amplitudes.

    This returns the reusable periodic-order payload used by both:
    - periodic far-field power diagnostics (`R/T/A`),
    - periodic exterior near-field evaluation via Rayleigh sums.

    The SVWF-to-PVWF prefactor follows the CELES/SMUTHI convention:
    `2*pi / (A * k * kz)`, where `A` is the unit-cell area.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    x = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], -1)
    k_f = float(k)
    if k_f <= 0.0 or not np.isfinite(k_f):
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")

    kp_inc = plane_wave_k_parallel(source)
    orders = enumerate_diffraction_orders_rectangular(
        lattice=lattice,
        k_parallel_incident=kp_inc,
        k=k_f,
        output_bmax=output_bmax,
    )

    n_orders = int(orders.m.size)
    reflected = np.zeros((n_orders, 2), dtype=np.complex128)
    transmitted = np.zeros((n_orders, 2), dtype=np.complex128)
    scattered_up = np.zeros((n_orders, 2), dtype=np.complex128)
    scattered_down = np.zeros((n_orders, 2), dtype=np.complex128)

    area = float(lattice.area)
    for i in range(n_orders):
        kp = orders.k_parallel[i]
        kz = complex(orders.kz[i])
        kx, ky = float(kp[0]), float(kp[1])
        if abs(kz) <= 1e-15:
            continue
        b_up_te, b_up_tm, b_down_te, b_down_tm = _order_mode_coefficients(
            lmax=int(lmax),
            k=k_f,
            k_parallel=kp,
            kz=kz,
        )
        phase_up = np.exp(-1j * (pos[:, 0] * kx + pos[:, 1] * ky + pos[:, 2] * kz))
        phase_down = np.exp(-1j * (pos[:, 0] * kx + pos[:, 1] * ky - pos[:, 2] * kz))
        pref = (2.0 * np.pi) / (area * k_f * kz)
        scattered_up[i, 0] = pref * np.sum(phase_up * (x @ b_up_te))
        scattered_up[i, 1] = pref * np.sum(phase_up * (x @ b_up_tm))
        scattered_down[i, 0] = pref * np.sum(phase_down * (x @ b_down_te))
        scattered_down[i, 1] = pref * np.sum(phase_down * (x @ b_down_tm))

    inc_sign = 1.0 if float(np.cos(float(source.polar_angle))) >= 0.0 else -1.0
    amp = complex(source.amplitude)
    a_te, a_tm = source.jones_coefficients()
    incident_pol = amp * np.asarray([a_te, a_tm], dtype=np.complex128)

    if inc_sign >= 0.0:
        transmitted[:, :] = scattered_up
        reflected[:, :] = scattered_down
    else:
        transmitted[:, :] = scattered_down
        reflected[:, :] = scattered_up

    matches_incident = np.where((orders.m == 0) & (orders.n == 0))[0]
    if matches_incident.size:
        transmitted[int(matches_incident[0]), :] += incident_pol

    return PeriodicOrderAmplitudes(
        order_mn=np.column_stack((orders.m, orders.n)).astype(np.int32, copy=False),
        order_k_parallel=np.asarray(orders.k_parallel, dtype=float).reshape(n_orders, 2),
        order_kz=np.asarray(orders.kz, dtype=np.complex128).reshape(n_orders),
        order_propagating=np.asarray(orders.propagating, dtype=bool).reshape(n_orders),
        scattered_up_amplitudes=np.asarray(scattered_up, dtype=np.complex128),
        scattered_down_amplitudes=np.asarray(scattered_down, dtype=np.complex128),
        reflected_amplitudes=np.asarray(reflected, dtype=np.complex128),
        transmitted_amplitudes=np.asarray(transmitted, dtype=np.complex128),
        incident_polarization=np.asarray(incident_pol, dtype=np.complex128).reshape(2),
        incidence_sign=float(inc_sign),
        incident_k_parallel=np.asarray(kp_inc, dtype=float).reshape(2),
        unit_cell_area=float(area),
        output_bmax=(None if output_bmax is None else float(output_bmax)),
    )


def periodic_plane_wave_orders(
    *,
    source: PlaneWave,
    lattice: RectangularLattice2D,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    n_medium: complex,
    output_bmax: float | None = None,
) -> PeriodicFarFieldPayload:
    """Compute periodic reflected/transmitted order amplitudes and `R/T/A`.

    This follows the CELES/SMUTHI SVWF-to-PVWF convention for periodic orders:
    each order coefficient uses the `2*pi/(A*k*kz)` prefactor, where `A` is
    unit-cell area and `kz` is the branch-selected longitudinal wavevector with
    non-negative imaginary part.

    `output_bmax=None` selects the propagating-order table only (`|k_parallel|<=k`).
    Set `output_bmax` explicitly to include evanescent orders in the output basis.
    """
    k_f = float(k)
    if k_f <= 0.0 or not np.isfinite(k_f):
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    n_real = float(np.real(complex(n_medium)))
    if n_real <= 0.0:
        raise ValueError(f"`n_medium` must be positive and real. Got {n_medium!r}.")

    orders_payload = periodic_order_amplitudes(
        source=source,
        lattice=lattice,
        positions=positions,
        coeffs=coeffs,
        lmax=int(lmax),
        k=k_f,
        output_bmax=output_bmax,
    )

    n_orders = int(orders_payload.order_mn.shape[0])
    reflected = np.asarray(orders_payload.reflected_amplitudes, dtype=np.complex128)
    transmitted = np.asarray(orders_payload.transmitted_amplitudes, dtype=np.complex128)
    order_kz = np.asarray(orders_payload.order_kz, dtype=np.complex128)
    order_propagating = np.asarray(orders_payload.order_propagating, dtype=bool)

    kz_abs = np.abs(np.real(order_kz))
    flux_prefactor = np.where(
        order_propagating,
        n_real * kz_abs / (2.0 * k_f),
        0.0,
    )
    reflected_power = np.asarray(
        flux_prefactor * np.sum(np.abs(reflected) ** 2, axis=1),
        dtype=np.float64,
    )
    transmitted_power = np.asarray(
        flux_prefactor * np.sum(np.abs(transmitted) ** 2, axis=1),
        dtype=np.float64,
    )

    incident_norm = float(np.sum(np.abs(orders_payload.incident_polarization) ** 2))
    incident_power = n_real * abs(float(np.cos(float(source.polar_angle)))) * incident_norm / 2.0
    if incident_power <= 0.0 or not np.isfinite(incident_power):
        raise ValueError("Incident periodic power normalization is non-finite or non-positive.")

    r_tot = float(np.sum(reflected_power) / incident_power)
    t_tot = float(np.sum(transmitted_power) / incident_power)
    a_tot = float(1.0 - r_tot - t_tot)
    return PeriodicFarFieldPayload(
        lattice_a1=np.asarray(lattice.a1, dtype=float),
        lattice_a2=np.asarray(lattice.a2, dtype=float),
        unit_cell_area=float(orders_payload.unit_cell_area),
        incident_k_parallel=np.asarray(orders_payload.incident_k_parallel, dtype=float).reshape(2),
        output_bmax=orders_payload.output_bmax,
        order_mn=np.asarray(orders_payload.order_mn, dtype=np.int32).reshape(n_orders, 2),
        order_k_parallel=np.asarray(orders_payload.order_k_parallel, dtype=float).reshape(
            n_orders, 2
        ),
        order_kz=np.asarray(orders_payload.order_kz, dtype=np.complex128).reshape(n_orders),
        order_propagating=np.asarray(orders_payload.order_propagating, dtype=bool).reshape(
            n_orders
        ),
        reflected_amplitudes=np.asarray(reflected, dtype=np.complex128),
        transmitted_amplitudes=np.asarray(transmitted, dtype=np.complex128),
        reflected_power_per_order=np.asarray(reflected_power, dtype=np.float64),
        transmitted_power_per_order=np.asarray(transmitted_power, dtype=np.float64),
        incident_power_per_area=float(incident_power),
        reflectance=float(r_tot),
        transmittance=float(t_tot),
        absorptance_raw_diff=float(a_tot),
    )


__all__ = [
    "PeriodicFarFieldPayload",
    "PeriodicOrderAmplitudes",
    "periodic_order_amplitudes",
    "periodic_plane_wave_orders",
]
