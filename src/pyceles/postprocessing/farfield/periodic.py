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
from .power import PowerBalance


@dataclass(frozen=True)
class PeriodicFarFieldPayload:
    """Discrete periodic-order amplitudes plus cell-normalized power balance."""

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
    reflected_flux_per_order: np.ndarray
    transmitted_flux_per_order: np.ndarray
    incident_flux: float
    power: PowerBalance


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


def _periodic_order_fluxes(
    *,
    reflected: np.ndarray,
    transmitted: np.ndarray,
    order_kz: np.ndarray,
    order_propagating: np.ndarray,
    k: float,
    n_medium: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normal power flux carried by each reflected/transmitted order."""
    k_f = float(k)
    if k_f <= 0.0 or not np.isfinite(k_f):
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    n_real = float(np.real(complex(n_medium)))
    if n_real <= 0.0:
        raise ValueError(f"`n_medium` must be positive and real. Got {n_medium!r}.")
    kz = np.asarray(order_kz, dtype=np.complex128)
    propagating = np.asarray(order_propagating, dtype=bool)
    prefactor = np.where(
        propagating,
        n_real * np.abs(np.real(kz)) / (2.0 * k_f),
        0.0,
    )
    reflected_flux = np.asarray(
        prefactor * np.sum(np.abs(np.asarray(reflected)) ** 2, axis=1),
        dtype=np.float64,
    )
    transmitted_flux = np.asarray(
        prefactor * np.sum(np.abs(np.asarray(transmitted)) ** 2, axis=1),
        dtype=np.float64,
    )
    return reflected_flux, transmitted_flux


def mix_periodic_farfield_payloads(
    first: PeriodicFarFieldPayload,
    second: PeriodicFarFieldPayload,
    *,
    source: PlaneWave,
    a_first: complex,
    a_second: complex,
    k: float,
    n_medium: complex,
) -> PeriodicFarFieldPayload:
    """Coherently combine two order-aligned periodic channel payloads.

    The two inputs must share one diffraction-order geometry, as TE/TM basis
    channels from the same periodic block solve do. Mixing their order
    amplitudes is exactly equivalent to re-evaluating the periodic far-field
    transform on the mixed coefficient vector.
    """
    structural_arrays = (
        "lattice_a1",
        "lattice_a2",
        "incident_k_parallel",
        "order_mn",
        "order_k_parallel",
        "order_kz",
        "order_propagating",
    )
    for name in structural_arrays:
        lhs = np.asarray(getattr(first, name))
        rhs = np.asarray(getattr(second, name))
        if lhs.shape != rhs.shape or not np.array_equal(lhs, rhs):
            raise ValueError(
                "Cannot mix periodic channels with different order geometry: "
                f"field '{name}' differs."
            )
    if not np.isclose(
        float(first.unit_cell_area),
        float(second.unit_cell_area),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Cannot mix periodic channels with different unit-cell areas.")
    if first.output_bmax != second.output_bmax:
        raise ValueError("Cannot mix periodic channels with different output_bmax values.")

    reflected = np.asarray(
        a_first * first.reflected_amplitudes + a_second * second.reflected_amplitudes,
        dtype=np.complex128,
    )
    transmitted = np.asarray(
        a_first * first.transmitted_amplitudes + a_second * second.transmitted_amplitudes,
        dtype=np.complex128,
    )
    reflected_flux, transmitted_flux = _periodic_order_fluxes(
        reflected=reflected,
        transmitted=transmitted,
        order_kz=first.order_kz,
        order_propagating=first.order_propagating,
        k=k,
        n_medium=n_medium,
    )

    n_real = float(np.real(complex(n_medium)))
    amp = complex(source.amplitude)
    incident_norm = float(
        abs(amp) ** 2 * (abs(complex(a_first)) ** 2 + abs(complex(a_second)) ** 2)
    )
    incident_flux = n_real * abs(float(np.cos(float(source.polar_angle)))) * incident_norm / 2.0
    if incident_flux <= 0.0 or not np.isfinite(incident_flux):
        raise ValueError("Incident periodic power normalization is non-finite or non-positive.")

    area = float(first.unit_cell_area)
    power = PowerBalance(
        incident_power=float(incident_flux * area),
        reflected_power=float(np.sum(reflected_flux) * area),
        transmitted_power=float(np.sum(transmitted_flux) * area),
    )
    return PeriodicFarFieldPayload(
        lattice_a1=np.asarray(first.lattice_a1),
        lattice_a2=np.asarray(first.lattice_a2),
        unit_cell_area=area,
        incident_k_parallel=np.asarray(first.incident_k_parallel),
        output_bmax=first.output_bmax,
        order_mn=np.asarray(first.order_mn),
        order_k_parallel=np.asarray(first.order_k_parallel),
        order_kz=np.asarray(first.order_kz, dtype=np.complex128),
        order_propagating=np.asarray(first.order_propagating, dtype=bool),
        reflected_amplitudes=reflected,
        transmitted_amplitudes=transmitted,
        reflected_flux_per_order=reflected_flux,
        transmitted_flux_per_order=transmitted_flux,
        incident_flux=float(incident_flux),
        power=power,
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

    reflected_flux, transmitted_flux = _periodic_order_fluxes(
        reflected=reflected,
        transmitted=transmitted,
        order_kz=order_kz,
        order_propagating=order_propagating,
        k=k_f,
        n_medium=n_medium,
    )

    incident_norm = float(np.sum(np.abs(orders_payload.incident_polarization) ** 2))
    incident_flux = n_real * abs(float(np.cos(float(source.polar_angle)))) * incident_norm / 2.0
    if incident_flux <= 0.0 or not np.isfinite(incident_flux):
        raise ValueError("Incident periodic power normalization is non-finite or non-positive.")

    unit_cell_area = float(orders_payload.unit_cell_area)
    power = PowerBalance(
        incident_power=float(incident_flux * unit_cell_area),
        reflected_power=float(np.sum(reflected_flux) * unit_cell_area),
        transmitted_power=float(np.sum(transmitted_flux) * unit_cell_area),
    )
    return PeriodicFarFieldPayload(
        lattice_a1=np.asarray(lattice.a1, dtype=float),
        lattice_a2=np.asarray(lattice.a2, dtype=float),
        unit_cell_area=unit_cell_area,
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
        reflected_flux_per_order=np.asarray(reflected_flux, dtype=np.float64),
        transmitted_flux_per_order=np.asarray(transmitted_flux, dtype=np.float64),
        incident_flux=float(incident_flux),
        power=power,
    )


__all__ = [
    "PeriodicFarFieldPayload",
    "PeriodicOrderAmplitudes",
    "mix_periodic_farfield_payloads",
    "periodic_order_amplitudes",
    "periodic_plane_wave_orders",
]
