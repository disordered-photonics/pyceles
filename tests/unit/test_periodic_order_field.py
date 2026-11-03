"""Plane-wave bases and streamed periodic E/H and flux reconstruction."""

import numpy as np
import pytest

from pyceles.core.conversions import plane_wave_basis
from pyceles.core.lattice import RectangularLattice2D
from pyceles.postprocessing.farfield import periodic_outgoing_amplitudes, periodic_outgoing_flux
from pyceles.postprocessing.nearfield.periodic_exterior import _accumulate_order_field


class _Progress:
    def __init__(self):
        self.updates = []

    def update(self, count):
        self.updates.append(count)


def _scalar_basis(k, k_parallel, kz):
    """Cartesian TE/TM vectors with H from the plane-wave Maxwell equation."""
    alpha = np.arctan2(k_parallel[1], k_parallel[0])
    e_te = np.array([-np.sin(alpha), np.cos(alpha), 0], dtype=complex)
    e_tm = np.array(
        [kz / k * np.cos(alpha), kz / k * np.sin(alpha), -np.linalg.norm(k_parallel) / k]
    )
    wavevector = np.r_[k_parallel, kz]
    return e_te, e_tm, np.cross(wavevector, e_te) / k, np.cross(wavevector, e_tm) / k


@pytest.mark.parametrize("n_medium", [1.0, 1.33 + 0.08j])
@pytest.mark.parametrize("sign", [-1, 1])
def test_plane_wave_basis_broadcasts_wavevector_axes(n_medium, sign):
    k = 2 * np.pi / 550 * n_medium
    base = np.array([[0, 0], [0.004, -0.003], [0.03, 0.01]])
    kpar = np.stack((base, np.stack((-base[:, 1], base[:, 0]), axis=-1)))
    kz = sign * np.sqrt(k * k - np.sum(base**2, axis=-1) + 0j)
    actual = plane_wave_basis(k=k, k_parallel=kpar, kz=kz)
    for row in range(2):
        for order in range(3):
            expected = _scalar_basis(k, kpar[row, order], kz[order])
            scalar = plane_wave_basis(k=k, k_parallel=kpar[row, order], kz=kz[order])
            for batched, single, reference in zip(actual, scalar, expected, strict=True):
                assert batched.shape == (2, 3, 3)
                assert single.shape == (3,)
                np.testing.assert_allclose(batched[row, order], single, rtol=3e-15, atol=3e-15)
                np.testing.assert_allclose(single, reference, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("n_medium", [1.0, 1.33 + 0.08j])
@pytest.mark.parametrize("sign", [-1, 1])
def test_streamed_periodic_fields_match_scalar_maxwell_fields(n_medium, sign):
    k = 2 * np.pi / 550 * n_medium
    kpar = np.array([[0, 0], [0.004, -0.003], [-0.009, 0.002], [0.03, 0.01]])
    kz = sign * np.sqrt(k * k - np.sum(kpar**2, axis=-1) + 0j)
    rng = np.random.default_rng(907)
    amplitudes = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    amplitudes[2] = 0
    reference_z = 1e6
    points = rng.uniform(-100, 100, (12, 6))[::2, ::2]
    points[:, 2] += reference_z + sign * 200
    expected_e, expected_h = np.zeros((2, len(points), 3), dtype=complex)
    for q, z, (a_te, a_tm) in zip(kpar, kz, amplitudes, strict=True):
        e_te, e_tm, h_te, h_tm = _scalar_basis(k, q, z)
        phase = np.exp(1j * (points[:, :2] @ q + z * (points[:, 2] - reference_z)))
        expected_e += phase[:, None] * (a_te * e_te + a_tm * e_tm)
        expected_h += phase[:, None] * n_medium * (a_te * h_te + a_tm * h_tm)
    progress = _Progress()
    actual = _accumulate_order_field(
        points,
        k=k,
        n_medium=n_medium,
        order_k_parallel=kpar,
        order_kz=kz,
        amplitudes=amplitudes,
        reference_z=reference_z,
        progress=progress,
    )
    for field, expected in zip(actual, (expected_e, expected_h), strict=True):
        np.testing.assert_allclose(field, expected, rtol=5e-14, atol=5e-15)
    assert progress.updates == [1] * len(kpar)


def test_zero_order_fields_skip_growing_phases_and_report_progress():
    progress = _Progress()
    with np.errstate(over="raise", invalid="raise"):
        e, h = _accumulate_order_field(
            np.array([[0, 0, 1e6]]),
            k=0.01,
            n_medium=1,
            order_k_parallel=np.array([[0.03, 0], [0, 0.04]]),
            order_kz=np.array([-0.02j, -0.03j]),
            amplitudes=np.zeros((2, 2)),
            progress=progress,
        )
    np.testing.assert_array_equal(e, 0)
    np.testing.assert_array_equal(h, 0)
    assert progress.updates == [1, 1]


def test_streamed_order_batches_sum_repeated_wavevectors():
    k, n_medium = 0.014 + 0.001j, 1.4 + 0.1j
    base = np.array([[0, 0], [0.004, -0.003], [-0.009, 0.002], [0.03, 0.01]])
    base_kz = np.sqrt(k * k - np.sum(base**2, axis=-1))
    # A repeated plane-wave direction is equivalent to summing its amplitudes.
    # The order count also crosses the bounded basis-workspace batch.
    labels = np.arange(4103) % len(base)
    rng = np.random.default_rng(911)
    amplitudes = rng.standard_normal((len(labels), 2)) + 1j * rng.standard_normal((len(labels), 2))
    amplitudes[4095:4097] = 0
    points = np.array([[20, -50, 200], [-30, 10, 250]])
    expected_e, expected_h = np.zeros((2, len(points), 3), dtype=complex)
    for i, (q, kz) in enumerate(zip(base, base_kz, strict=True)):
        a_te, a_tm = np.sum(amplitudes[labels == i], axis=0)
        e_te, e_tm, h_te, h_tm = _scalar_basis(k, q, kz)
        phase = np.exp(1j * (points[:, :2] @ q + kz * points[:, 2]))
        expected_e += phase[:, None] * (a_te * e_te + a_tm * e_tm)
        expected_h += phase[:, None] * n_medium * (a_te * h_te + a_tm * h_tm)
    progress = _Progress()
    actual = _accumulate_order_field(
        points,
        k=k,
        n_medium=n_medium,
        order_k_parallel=base[labels],
        order_kz=base_kz[labels],
        amplitudes=amplitudes,
        progress=progress,
    )
    for field, expected in zip(actual, (expected_e, expected_h), strict=True):
        np.testing.assert_allclose(field, expected, rtol=2e-12, atol=2e-12)
    assert progress.updates == [1] * len(labels)


def test_periodic_flux_batches_match_scalar_maxwell_fields():
    n_medium = 1.33 + 0.08j
    k = 2 * np.pi / 550 * n_medium
    outgoing = periodic_outgoing_amplitudes(
        lattice=RectangularLattice2D(4200, 4200),
        positions=np.zeros((1, 3)),
        coeffs=np.array([[1, 0.2j, -0.3, 0.5j, 0.8, -0.2j]]),
        lmax=1,
        k=k,
        k_parallel=np.array([0.0013, -0.0007]),
        output_bmax=0.055,
    )
    assert len(outgoing.orders.m) > 4096  # Cross the bounded Maxwell-vector workspace.
    expected = []
    for sign, amplitudes, distance in (
        (1, outgoing.up_amplitudes, 200),
        (-1, outgoing.down_amplitudes, -250),
    ):
        flux = np.empty(len(outgoing.orders.m))
        for i, (q, kz) in enumerate(
            zip(outgoing.orders.k_parallel, outgoing.orders.kz, strict=True)
        ):
            e_te, e_tm, h_te, h_tm = _scalar_basis(k, q, sign * kz)
            a_te, a_tm = amplitudes[i] * np.exp(1j * sign * kz * distance)
            electric = a_te * e_te + a_tm * e_tm
            magnetic = n_medium * (a_te * h_te + a_tm * h_tm)
            flux[i] = (
                sign * outgoing.unit_cell_area / 2 * np.cross(electric, magnetic.conj())[2].real
            )
        expected.append(flux)
    actual = periodic_outgoing_flux(outgoing, k=k, n_medium=n_medium, z_above=200, z_below=-250)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-15)
