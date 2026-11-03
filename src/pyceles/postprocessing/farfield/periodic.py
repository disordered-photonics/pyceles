"""Periodic diffraction-order amplitudes and flux diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyceles.core.conversions import plane_wave_basis, transformation_coefficients
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D, real_transverse_wavevector
from pyceles.core.particles import ParticleCollection
from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.periodic.types import periodic_source_k_parallel
from pyceles.core.radial import passive_wavenumber
from pyceles.core.sources import BlochPlaneWave, DipoleCollection, DipoleSource, PlaneWave
from pyceles.core.sources.embedded import locate_dipole_source
from pyceles.core.sources.plane_wave import _plane_wave_geometry
from pyceles.core.spherical import spherical_functions_trigon

from .orders import DiffractionOrders, enumerate_diffraction_orders_rectangular
from .power import PowerBalance

if TYPE_CHECKING:
    from pyceles.simulation import ChannelResult


@dataclass(frozen=True)
class PeriodicOutgoingAmplitudes:
    """TE/TM outgoing orders at finite upper/lower reference planes.

    Upward waves use +kz, downward waves use -kz. Their phases are
    exp(i*(q_mn.rho +/- kz*(z-reference_z))). Reference planes are ordered
    (upper, lower). No incident wave or plane-wave power normalization is
    implied, and absorbing hosts require an explicit reciprocal cutoff.
    """

    orders: DiffractionOrders
    up_amplitudes: np.ndarray
    down_amplitudes: np.ndarray
    reference_z: tuple[float, float]
    k_parallel: np.ndarray
    unit_cell_area: float
    output_bmax: float | None

    def to_mapping(self) -> dict[str, object]:
        """Return field amplitudes and their finite-plane interpretation for I/O."""
        return dict(
            order_mn=np.column_stack((self.orders.m, self.orders.n)),
            order_k_parallel=self.orders.k_parallel,
            order_kz=self.orders.kz,
            order_propagating=self.orders.propagating,
            up_amplitudes=self.up_amplitudes,
            down_amplitudes=self.down_amplitudes,
            reference_z=self.reference_z,
            k_parallel=self.k_parallel,
            unit_cell_area=self.unit_cell_area,
            output_bmax=self.output_bmax,
        )


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


def _incidence_sign(source: PlaneWave | BlochPlaneWave) -> float:
    return _plane_wave_geometry(source).direction_sign


def _incidence_cosine(source: PlaneWave | BlochPlaneWave) -> float:
    return abs(float(complex(_plane_wave_geometry(source).cos_beta).real))


def _incident_order_polarization(source: PlaneWave | BlochPlaneWave, *, k: complex) -> np.ndarray:
    """Express the incident zero order in its global-origin TE/TM frame.

    The source amplitude is specified at its own finite reference point. At
    an exact pole its azimuth also rotates the source Jones axes relative to
    the zero order's fixed Cartesian frame.
    """
    geometry = _plane_wave_geometry(source, k=k)
    a_te, a_tm = source.jones_coefficients()
    if isinstance(source, PlaneWave) and source.polar_angle in (0.0, np.pi):
        ca, sa = np.cos(geometry.azimuth), np.sin(geometry.azimuth)
        sign = geometry.direction_sign
        a_te, a_tm = ca * a_te + sign * sa * a_tm, -sign * sa * a_te + ca * a_tm
    phase = np.exp(-1j * np.dot(geometry.wavevector, geometry.reference_point))
    return np.asarray(
        complex(source.amplitude) * phase * np.asarray([a_te, a_tm]), dtype=np.complex128
    )


def _order_mode_coefficients(
    *,
    lmax: int,
    k: complex,
    k_parallel: np.ndarray,
    kz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return order-by-mode TE/TM coefficients for both longitudinal branches."""
    kpar = np.asarray(k_parallel, dtype=float).reshape(-1, 2)
    kp = np.linalg.norm(kpar, axis=1)
    alpha = np.arctan2(kpar[:, 1], kpar[:, 0])
    st = np.asarray(kp / complex(k), dtype=np.complex128)
    ct_up = np.asarray(kz, dtype=np.complex128).reshape(-1) / complex(k)
    ct_down = -ct_up

    pi_up, tau_up = spherical_functions_trigon(ct_up, st, int(lmax), xp=np)
    pi_down, tau_down = spherical_functions_trigon(ct_down, st, int(lmax), xp=np)
    nmodes = n_modes(int(lmax))
    b_up_te = np.empty((len(kpar), nmodes), dtype=np.complex128)
    b_up_tm = np.empty_like(b_up_te)
    b_down_te = np.empty_like(b_up_te)
    b_down_tm = np.empty_like(b_up_te)

    for tau, l, m, idx in iter_modes(int(lmax)):
        phase = np.exp(1j * float(m) * alpha)
        b_up_te[:, idx] = (
            transformation_coefficients(pi_up, tau_up, tau, l, m, pol=1, dagger=False) * phase
        )
        b_up_tm[:, idx] = (
            transformation_coefficients(pi_up, tau_up, tau, l, m, pol=2, dagger=False) * phase
        )
        b_down_te[:, idx] = (
            transformation_coefficients(pi_down, tau_down, tau, l, m, pol=1, dagger=False) * phase
        )
        b_down_tm[:, idx] = (
            transformation_coefficients(pi_down, tau_down, tau, l, m, pol=2, dagger=False) * phase
        )
    return b_up_te, b_up_tm, b_down_te, b_down_tm


def _add_outgoing_amplitudes(
    outgoing: PeriodicOutgoingAmplitudes,
    *,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: complex,
) -> None:
    """Add one outgoing expansion family to an owned order-amplitude sink."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    x = np.asarray(coeffs, dtype=np.complex128).reshape(len(pos), n_modes(lmax))
    active = np.any(x != 0, axis=1)
    pos, x = pos[active], x[active]
    if not len(pos):
        return
    k_c = complex(k)
    area = outgoing.unit_cell_area
    orders = outgoing.orders
    n_orders = len(orders.m)
    if np.any(orders.kz == 0):
        raise ValueError(
            "Outgoing amplitudes at a grazing Rayleigh-threshold order need a limiting prescription"
        )
    # Bound the four conversion tables to about 2 MiB in complex128. Particle
    # contractions remain streamed, without a particle-by-order workspace.
    batch_size = max(1, 32768 // n_modes(lmax))
    for start in range(0, n_orders, batch_size):
        stop = min(start + batch_size, n_orders)
        coefficients = _order_mode_coefficients(
            lmax=int(lmax),
            k=k_c,
            k_parallel=orders.k_parallel[start:stop],
            kz=orders.kz[start:stop],
        )
        for j, i in enumerate(range(start, stop)):
            kx, ky = orders.k_parallel[i]
            kz = orders.kz[i]
            b_up_te, b_up_tm, b_down_te, b_down_tm = (table[j] for table in coefficients)
            phase_up = np.exp(
                -1j * (pos[:, 0] * kx + pos[:, 1] * ky + (pos[:, 2] - outgoing.reference_z[0]) * kz)
            )
            phase_down = np.exp(
                -1j * (pos[:, 0] * kx + pos[:, 1] * ky - (pos[:, 2] - outgoing.reference_z[1]) * kz)
            )
            pref = (2.0 * np.pi) / (area * k_c * kz)
            outgoing.up_amplitudes[i, 0] += pref * np.sum(phase_up * (x @ b_up_te))
            outgoing.up_amplitudes[i, 1] += pref * np.sum(phase_up * (x @ b_up_tm))
            outgoing.down_amplitudes[i, 0] += pref * np.sum(phase_down * (x @ b_down_te))
            outgoing.down_amplitudes[i, 1] += pref * np.sum(phase_down * (x @ b_down_tm))


def periodic_outgoing_amplitudes(
    *,
    lattice: RectangularLattice2D,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: complex,
    k_parallel: np.ndarray,
    output_bmax: float | None = None,
    reference_z: tuple[float, float] = (0.0, 0.0),
) -> PeriodicOutgoingAmplitudes:
    """Transform outgoing SVWF arrays without imposing an incident source.

    The prefactor is 2*pi/(area*k*kz). Finite reference planes avoid growing
    evanescent phase factors when a source is far from the global origin.
    Choose them outside the source slab when reconstructing physical fields.
    """
    wave_number = passive_wavenumber(k)
    if len(reference_z) != 2 or not np.all(np.isfinite(reference_z)):
        raise ValueError("reference_z must contain two finite heights")
    q = real_transverse_wavevector(k_parallel)
    orders = enumerate_diffraction_orders_rectangular(
        lattice=lattice,
        k_parallel_incident=q,
        k=wave_number,
        output_bmax=output_bmax,
    )
    out = PeriodicOutgoingAmplitudes(
        orders,
        np.zeros((len(orders.m), 2), dtype=np.complex128),
        np.zeros((len(orders.m), 2), dtype=np.complex128),
        (float(reference_z[0]), float(reference_z[1])),
        q,
        float(lattice.area),
        output_bmax,
    )
    _add_outgoing_amplitudes(out, positions=positions, coeffs=coeffs, lmax=lmax, k=wave_number)
    return out


def compute_periodic_outgoing_amplitudes(
    run: ChannelResult,
    *,
    output_bmax: float | None = None,
) -> PeriodicOutgoingAmplitudes:
    """Outgoing orders of one periodic channel at its source-slab boundaries.

    Plane-wave channels contain particle-scattered waves. Dipole arrays also
    include direct host-source emission; embedded emission is already part of
    the solved outgoing coefficients. The same amplitude definition applies
    in absorbing hosts without assigning infinite-distance radiation power.
    """
    periodic = run.config.periodic
    if periodic is None:
        raise ValueError("Outgoing periodic orders require a periodic channel")
    q = periodic_source_k_parallel(periodic, run.source)
    host = None
    heights = np.r_[
        run.positions[:, 2] - run.circumscribing_radii,
        run.positions[:, 2] + run.circumscribing_radii,
    ]
    if isinstance(run.source, (DipoleSource, DipoleCollection)):
        # Outgoing waves already contain embedded emission. Only the host
        # subset is needed here, so no local interface solve is prepared.
        prepared = run._dipole_source
        placement = (
            prepared.placement
            if prepared is not None
            else locate_dipole_source(
                ParticleCollection.from_particles(run.particles),
                run.source,
                lattice=periodic.lattice,
                k_parallel=q,
            )
        )
        host = placement.host_source
        active = np.any(host.outgoing_coeffs() != 0, axis=1)
        heights = np.r_[heights, host.source_positions()[active, 2]]
    reference_z = (float(heights.max()), float(heights.min())) if len(heights) else (0.0, 0.0)
    out = periodic_outgoing_amplitudes(
        lattice=periodic.lattice,
        positions=run.positions,
        coeffs=run.coeffs,
        lmax=run.config.lmax,
        k=run.k,
        k_parallel=q,
        reference_z=reference_z,
        output_bmax=periodic.options.output_bmax if output_bmax is None else output_bmax,
    )
    if host is not None:
        _add_outgoing_amplitudes(
            out, positions=host.source_positions(), coeffs=host.outgoing_coeffs(), lmax=1, k=run.k
        )
    return out


def periodic_outgoing_flux(
    outgoing: PeriodicOutgoingAmplitudes,
    *,
    k: complex,
    n_medium: complex,
    z_above: float | None = None,
    z_below: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward power per cell/order through finite upper and lower planes.

    Integrating one cell eliminates cross-order interference for real q.
    Evaluate the full complex TE/TM Maxwell basis, retaining evanescent flux
    in absorbing hosts. This is finite-plane flux, not R/T normalization.
    """
    upper, lower = outgoing.reference_z
    z_above = upper if z_above is None else float(z_above)
    z_below = lower if z_below is None else float(z_below)
    if not np.all(np.isfinite([z_above, z_below])) or z_above < upper or z_below > lower:
        raise ValueError("Flux planes must be finite and outside the outgoing reference slab")
    fluxes = []
    for sign, amplitude, distance in (
        (1, outgoing.up_amplitudes, z_above - upper),
        (-1, outgoing.down_amplitudes, z_below - lower),
    ):
        flux = np.zeros(len(outgoing.orders.m), dtype=float)
        # Keep Maxwell-vector work bounded, as in exterior E/H reconstruction.
        for start in range(0, len(flux), 4096):
            stop = min(start + 4096, len(flux))
            kz = sign * outgoing.orders.kz[start:stop]
            e_te, e_tm, h_te, h_tm = plane_wave_basis(
                k=k, k_parallel=outgoing.orders.k_parallel[start:stop], kz=kz
            )
            weights = amplitude[start:stop] * np.exp(1j * kz * distance)[:, None]
            e = weights[:, 0, None] * e_te + weights[:, 1, None] * e_tm
            h = complex(n_medium) * (weights[:, 0, None] * h_te + weights[:, 1, None] * h_tm)
            flux[start:stop] = sign * outgoing.unit_cell_area / 2 * np.cross(e, h.conj())[:, 2].real
        fluxes.append(flux)
    return fluxes[0], fluxes[1]


def periodic_order_amplitudes(
    *,
    source: PlaneWave | BlochPlaneWave,
    lattice: RectangularLattice2D,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: complex,
    output_bmax: float | None = None,
) -> PeriodicOrderAmplitudes:
    """Compute branch-resolved periodic order amplitudes.

    This returns branch-resolved amplitudes for periodic exterior field
    reconstruction. Lossless-host power diagnostics are built on this payload;
    complex-host amplitudes are meaningful at finite distance but are not
    normalized into infinite-distance power.

    The SVWF-to-PVWF prefactor follows the CELES/SMUTHI convention:
    `2*pi / (A * k * kz)`, where `A` is the unit-cell area.
    """
    k_c = passive_wavenumber(k)
    kp_inc = plane_wave_k_parallel(source)
    outgoing = periodic_outgoing_amplitudes(
        lattice=lattice,
        positions=positions,
        coeffs=coeffs,
        lmax=lmax,
        k=k_c,
        k_parallel=kp_inc,
        output_bmax=output_bmax,
    )
    orders = outgoing.orders
    n_orders = len(orders.m)
    scattered_up, scattered_down = outgoing.up_amplitudes, outgoing.down_amplitudes
    reflected = np.zeros_like(scattered_up)
    transmitted = np.zeros_like(scattered_down)
    area = outgoing.unit_cell_area

    inc_sign = _incidence_sign(source)
    incident_pol = _incident_order_polarization(source, k=k_c)

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
    if complex(k).imag != 0.0:
        raise ValueError(
            "Periodic power normalization requires a lossless host; use "
            "`periodic_order_amplitudes` for finite-distance complex-host fields."
        )
    k_f = float(complex(k).real)
    if k_f <= 0.0 or not np.isfinite(k_f):
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    n_c = complex(n_medium)
    if n_c.imag != 0.0:
        raise ValueError("Periodic power normalization requires a lossless host medium.")
    n_real = float(n_c.real)
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
    source: PlaneWave | BlochPlaneWave,
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
    incident_flux = n_real * _incidence_cosine(source) * incident_norm / 2.0
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
    source: PlaneWave | BlochPlaneWave,
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

    `output_bmax=None` selects the propagating-order table only for a lossless
    host. A complex wavenumber requires an explicit reciprocal-radius cutoff.
    """
    if complex(n_medium).imag != 0.0:
        raise ValueError("Periodic power normalization requires a lossless host medium.")
    k_f = float(k)
    if k_f <= 0.0 or not np.isfinite(k_f):
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    n_real = float(np.real(complex(n_medium)))
    if n_real <= 0.0:
        raise ValueError(f"`n_medium` must be positive and real. Got {n_medium!r}.")
    if _incidence_cosine(source) == 0.0:
        raise ValueError(
            "Periodic power normalization requires a propagating BlochPlaneWave incidence."
        )

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
    incident_flux = n_real * _incidence_cosine(source) * incident_norm / 2.0
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
    "PeriodicOutgoingAmplitudes",
    "compute_periodic_outgoing_amplitudes",
    "mix_periodic_farfield_payloads",
    "periodic_order_amplitudes",
    "periodic_outgoing_amplitudes",
    "periodic_outgoing_flux",
    "periodic_plane_wave_orders",
]
