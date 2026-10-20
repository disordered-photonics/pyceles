from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

import pyceles as pcl
from pyceles.core.indexing import n_modes
from pyceles.core.operators import PeriodicCouplingOperator
from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.translation import translation_ab5_table
from pyceles.postprocessing.farfield import periodic_order_amplitudes
from pyceles.postprocessing.nearfield import NearFieldComponents, compute_initial_field


def _source(
    *,
    host: complex,
    q: tuple[float, float],
    direction: Literal["up", "down"] = "down",
    reference_point: tuple[float, float, float] = (0.0, 0.0, 250.0),
    amplitude: complex = 1.0,
) -> pcl.BlochPlaneWave:
    return pcl.BlochPlaneWave(
        wavelength=550.0,
        medium_n=host,
        k_parallel=q,
        direction=direction,
        reference_point=reference_point,
        polarization=(0.7 + 0.1j, -0.2 + 0.3j),
        amplitude=amplitude,
    )


def test_bloch_plane_wave_fields_satisfy_maxwell_and_decay_from_reference_plane() -> None:
    host = 1.33 + 0.2j
    source = _source(host=host, q=(0.0034, -0.0012))
    lattice = pcl.RectangularLattice2D(1100.0, 900.0)
    reference = np.asarray(source.reference_point, dtype=float)
    points = np.asarray(
        [
            reference,
            reference + np.asarray([0.0, 0.0, -80.0]),
            reference + np.asarray([0.0, 0.0, -230.0]),
            reference + np.asarray([lattice.ax, 0.0, -80.0]),
        ]
    )
    k = 2.0 * np.pi / source.wavelength * host
    e, h = compute_initial_field(
        points,
        k=k,
        n_medium=host,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        show_progress=False,
    )

    qx, qy = source.k_parallel
    q_magnitude = float(np.hypot(qx, qy))
    alpha = float(np.arctan2(qy, qx))
    kz = source.wavevector[2]
    khat = source.wavevector / k
    e_te = np.asarray([-np.sin(alpha), np.cos(alpha), 0.0], dtype=np.complex128)
    e_tm = np.asarray(
        [kz / k * np.cos(alpha), kz / k * np.sin(alpha), -q_magnitude / k],
        dtype=np.complex128,
    )
    a_te, a_tm = source.jones_coefficients()
    e_reference = source.amplitude * (a_te * e_te + a_tm * e_tm)
    h_reference = host * np.cross(khat, e_reference)
    np.testing.assert_allclose(e[0], e_reference, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(h[0], h_reference, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(np.dot(source.wavevector, e_reference), 0.0, atol=1e-16)
    np.testing.assert_allclose(np.dot(source.wavevector, h_reference), 0.0, atol=1e-16)

    decay = np.exp(-source.longitudinal_wavenumber.imag * 150.0)
    np.testing.assert_allclose(np.linalg.norm(e[2]) / np.linalg.norm(e[1]), decay, rtol=2e-14)
    np.testing.assert_allclose(
        e[3],
        np.exp(1j * qx * lattice.ax) * e[1],
        rtol=2e-14,
        atol=2e-14,
    )


def _periodic_coupling(
    *,
    method: str,
    positions: np.ndarray,
    k: complex,
    q: np.ndarray,
    lattice: pcl.RectangularLattice2D,
) -> PeriodicCouplingOperator:
    if method == "directsum":
        options = pcl.PeriodicOptions(method="directsum", directsum_window=10)
    elif method == "ewald":
        options = pcl.PeriodicOptions(
            method="ewald",
            eta=np.sqrt(np.pi / lattice.area),
            real_shells=12,
            reciprocal_shells=12,
        )
    else:
        options = pcl.PeriodicOptions(
            method="rayleigh",
            eta=np.sqrt(np.pi / lattice.area),
            real_shells=12,
            reciprocal_shells=12,
            rayleigh_z_cut=400.0,
            rayleigh_reciprocal_shells=10,
        )
    return PeriodicCouplingOperator(
        lmax=1,
        k=k,
        positions=positions,
        ab5=translation_ab5_table(1, dtype=np.complex128),
        periodic=pcl.PeriodicSpec(lattice, options),
        k_parallel=q,
        dtype=np.dtype(np.complex128),
        circumscribing_radii=np.full(positions.shape[0], 15.0),
    )


def test_periodic_operators_match_at_oblique_real_bloch_vector_and_adjoint() -> None:
    host = 1.33 + 0.3j
    k = 2.0 * np.pi / 550.0 * host
    q = np.asarray([0.0021, -0.0013])
    lattice = pcl.RectangularLattice2D(1100.0, 900.0)
    # Off-origin sites straddle the x seam and force both shifted and
    # noncoplanar translation blocks, including the Rayleigh far path.
    positions = np.asarray([[1060.0, 170.0, 10.0], [35.0, 230.0, 710.0]])
    rng = np.random.default_rng(20261019)
    size = positions.shape[0] * n_modes(1)
    x = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    y = rng.standard_normal(size) + 1j * rng.standard_normal(size)

    operators = {
        method: _periodic_coupling(
            method=method,
            positions=positions,
            k=k,
            q=q,
            lattice=lattice,
        )
        for method in ("directsum", "ewald", "rayleigh")
    }
    actions = {method: operator.apply(x) for method, operator in operators.items()}
    for method in ("ewald", "rayleigh"):
        relative_error = np.linalg.norm(actions[method] - actions["directsum"]) / np.linalg.norm(
            actions["directsum"]
        )
        assert relative_error < 2e-6, f"{method} relative action error {relative_error:.3e}"

    for method, operator in operators.items():
        lhs = np.vdot(operator.apply(x), y)
        rhs = np.vdot(x, operator.apply_adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=3e-12, atol=3e-12)

        shifted = _periodic_coupling(
            method=method,
            positions=positions + lattice.lattice_vector(1, -2),
            k=k,
            q=q,
            lattice=lattice,
        )
        np.testing.assert_allclose(shifted.apply(x), operator.apply(x), rtol=3e-12, atol=3e-12)


def test_periodic_total_field_obeys_bloch_phase_between_cells() -> None:
    host = 1.33 + 0.3j
    lattice = pcl.RectangularLattice2D(1100.0, 900.0)
    source = _source(host=host, q=(0.0021, -0.0013))
    config = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=host,
        lmax=1,
        periodic=pcl.PeriodicSpec(
            lattice,
            pcl.PeriodicOptions(
                method="ewald",
                eta=np.sqrt(np.pi / lattice.area),
                real_shells=8,
                reciprocal_shells=8,
                output_bmax=0.14,
            ),
        ),
        solver_method="direct",
        verbose=False,
    )
    particle = pcl.Sphere(
        position=(140.0, 120.0, 0.0),
        radius=25.0,
        refractive_index=1.5 + 0.01j,
    )
    result = pcl.Simulation(config, particles=[particle]).run(source, include_farfield=False)
    point = np.asarray([[230.0, 80.0, 100.0]])
    fields = pcl.compute_periodic_near_field(
        result,
        points=np.concatenate((point, point + lattice.a1), axis=0),
        field_bmax=0.14,
    )
    phase = np.exp(1j * source.k_parallel[0] * lattice.ax)
    for name in ("E_initial", "H_initial", "E_scattered", "H_scattered", "E_total", "H_total"):
        values = np.asarray(getattr(fields, name))
        np.testing.assert_allclose(values[1], phase * values[0], rtol=2e-11, atol=2e-11)


def test_lossless_bloch_incidence_uses_longitudinal_flux_for_power_normalization() -> None:
    wavelength = 550.0
    host = 1.0 + 0.0j
    k = 2.0 * np.pi / wavelength
    qx = 0.25 * k
    source = pcl.BlochPlaneWave(
        wavelength=wavelength,
        medium_n=host,
        k_parallel=(qx, 0.0),
        direction="down",
        polarization="TE",
        amplitude=0.7 + 0.2j,
    )
    lattice = pcl.RectangularLattice2D(1000.0, 900.0)
    config = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=host,
        lmax=1,
        periodic=pcl.PeriodicSpec(
            lattice,
            pcl.PeriodicOptions(method="directsum", directsum_window=3),
        ),
        solver_method="direct",
        verbose=False,
    )
    particle = pcl.Sphere(position=(250.0, 300.0, 0.0), radius=20.0, refractive_index=1.5)
    run = pcl.Simulation(config, particles=[particle]).run(source, include_farfield=False)

    assert run.periodic is not None
    expected_flux = np.sqrt(1.0 - 0.25**2) * abs(source.amplitude) ** 2 / 2.0
    np.testing.assert_allclose(run.periodic.incident_flux, expected_flux, rtol=2e-14)


def test_small_bloch_spectrum_reconstructs_a_finite_beam_response() -> None:
    host = 1.33 + 0.2j
    lattice = pcl.RectangularLattice2D(1100.0, 900.0)
    config = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=host,
        lmax=1,
        periodic=pcl.PeriodicSpec(
            lattice,
            pcl.PeriodicOptions(
                method="ewald",
                eta=np.sqrt(np.pi / lattice.area),
                real_shells=8,
                reciprocal_shells=8,
                output_bmax=0.14,
            ),
        ),
        solver_method="direct",
        verbose=False,
    )
    particle = pcl.Sphere(
        position=(lattice.ax / 2.0, lattice.ay / 2.0, 0.0),
        radius=22.0,
        refractive_index=1.5 + 0.01j,
    )
    sim = pcl.Simulation(config, particles=[particle])
    qx = np.pi / lattice.ax
    components: list[NearFieldComponents] = []
    for q, weight in ((-qx, -0.25), (0.0, 0.5), (qx, 0.25)):
        source = pcl.BlochPlaneWave(
            wavelength=550.0,
            medium_n=host,
            k_parallel=(q, 0.0),
            direction="down",
            reference_point=(lattice.ax / 2.0, lattice.ay / 2.0, 250.0),
            polarization="TE",
            amplitude=weight,
        )
        run = sim.run(source, include_farfield=False)
        points = np.asarray(
            [[lattice.ax / 2.0, lattice.ay / 2.0, 100.0], [0.0, lattice.ay / 2.0, 100.0]]
        )
        components.append(pcl.compute_periodic_near_field(run, points=points, field_bmax=0.14))

    e_initial = np.sum(np.stack([field.E_initial for field in components]), axis=0)
    e_scattered = np.sum(np.stack([field.E_scattered for field in components]), axis=0)
    e_total = np.sum(np.stack([field.E_total for field in components]), axis=0)
    assert np.linalg.norm(e_scattered) > 0.0
    np.testing.assert_allclose(e_total, e_initial + e_scattered, rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(e_initial[0]) > np.linalg.norm(e_initial[1])


def test_periodic_incident_zero_order_respects_source_phase_and_axes() -> None:
    """The zero-order Jones data and Cartesian fields must use one origin."""
    lattice = pcl.RectangularLattice2D(900.0, 1100.0)
    sources = (
        pcl.PlaneWave(
            wavelength=550.0,
            medium_n=1.0,
            polar_angle=0.0,
            azimuthal_angle=0.41,
            focal_point=(17.0, -9.0, 83.0),
            polarization=(0.7 + 0.1j, -0.3 + 0.2j),
        ),
        pcl.PlaneWave(
            wavelength=550.0,
            medium_n=1.0,
            polar_angle=0.37,
            azimuthal_angle=0.82,
            focal_point=(27.0, -16.0, 52.0),
            polarization=(0.2 - 0.1j, 0.6 + 0.3j),
        ),
        pcl.PlaneWave(
            wavelength=550.0,
            medium_n=1.0,
            polar_angle=np.pi,
            azimuthal_angle=0.63,
            focal_point=(-13.0, 8.0, -61.0),
            polarization=(0.3 + 0.2j, 0.4 - 0.1j),
        ),
        pcl.BlochPlaneWave(
            wavelength=550.0,
            medium_n=1.33 + 0.2j,
            k_parallel=(0.0021, -0.0013),
            direction="down",
            reference_point=(30.0, -17.0, 90.0),
            polarization=(0.5 + 0.1j, -0.2 + 0.4j),
        ),
        pcl.BlochPlaneWave(
            wavelength=550.0,
            medium_n=1.33 + 0.2j,
            k_parallel=(0.0021, -0.0013),
            direction="up",
            reference_point=(-22.0, 19.0, -75.0),
            polarization=(-0.3 + 0.2j, 0.4 + 0.1j),
        ),
    )
    for source in sources:
        host = complex(source.medium_n)
        k = 2.0 * np.pi / source.wavelength * host
        order = periodic_order_amplitudes(
            source=source,
            lattice=lattice,
            positions=np.zeros((1, 3)),
            coeffs=np.zeros((1, n_modes(1)), dtype=np.complex128),
            lmax=1,
            k=k,
            output_bmax=0.02,
        )
        zero = np.flatnonzero(np.all(order.order_mn == [0, 0], axis=1))
        assert zero.size == 1
        te, tm = order.transmitted_amplitudes[int(zero[0])]
        np.testing.assert_allclose(order.incident_polarization, [te, tm], rtol=0, atol=0)

        q = plane_wave_k_parallel(source)
        rho = float(np.hypot(*q))
        alpha = float(np.arctan2(q[1], q[0])) if rho else 0.0
        sign = order.incidence_sign
        gamma = complex(np.sqrt(k * k - rho * rho + 0j))
        kz = sign * gamma
        e_te = np.asarray([-np.sin(alpha), np.cos(alpha), 0.0], dtype=np.complex128)
        e_tm = np.asarray(
            [(kz / k) * np.cos(alpha), (kz / k) * np.sin(alpha), -rho / k],
            dtype=np.complex128,
        )
        electric_from_order = te * e_te + tm * e_tm
        electric_direct, _ = compute_initial_field(
            np.zeros((1, 3)),
            k=k,
            n_medium=host,
            beam=source,
            polar_angles=np.empty(0),
            azimuthal_angles=np.empty(0),
            show_progress=False,
        )
        np.testing.assert_allclose(
            electric_from_order,
            electric_direct[0],
            rtol=4e-14,
            atol=4e-14,
        )


def test_shifted_real_host_plane_wave_keeps_periodic_order_phase_coherent() -> None:
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0,
        polar_angle=0.0,
        azimuthal_angle=0.41,
        polarization=(0.7 + 0.1j, -0.3 + 0.2j),
    )
    shifted_source = replace(source, focal_point=(0.0, 0.0, 73.0))
    k = 2.0 * np.pi / source.wavelength
    phase_at_origin = np.exp(-1j * k * shifted_source.focal_point[2])
    rng = np.random.default_rng(20261020)
    coeffs = rng.standard_normal((1, n_modes(1))) + 1j * rng.standard_normal((1, n_modes(1)))
    lattice = pcl.RectangularLattice2D(900.0, 1100.0)
    positions = np.asarray([[20.0, -30.0, 40.0]])
    reference = periodic_order_amplitudes(
        source=source,
        lattice=lattice,
        positions=positions,
        coeffs=coeffs,
        lmax=1,
        k=k,
        output_bmax=0.02,
    )
    shifted = periodic_order_amplitudes(
        source=shifted_source,
        lattice=lattice,
        positions=positions,
        coeffs=phase_at_origin * coeffs,
        lmax=1,
        k=k,
        output_bmax=0.02,
    )

    zero_order = np.flatnonzero(np.all(reference.order_mn == [0, 0], axis=1))
    assert zero_order.size == 1
    assert np.linalg.norm(reference.scattered_up_amplitudes[int(zero_order[0])]) > 0.0
    np.testing.assert_array_equal(shifted.order_mn, reference.order_mn)
    np.testing.assert_allclose(
        shifted.reflected_amplitudes,
        phase_at_origin * reference.reflected_amplitudes,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        shifted.transmitted_amplitudes,
        phase_at_origin * reference.transmitted_amplitudes,
        rtol=2e-13,
        atol=2e-13,
    )
