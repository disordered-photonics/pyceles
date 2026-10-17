"""Tangential Fourier-plane boundaries, Maxwell fields and incident projection."""

from dataclasses import replace

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import index_vswf
from pyceles.postprocessing.nearfield.initial import compute_initial_field


def _gaussian_plane(host, *, polar_angle=0.0, azimuthal_angle=0.0):
    q = np.linspace(0, 0.019, 33)
    alpha = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    width = 180.0
    profile = width**2 / (4 * np.pi) * np.exp(-(width**2) * q**2 / 4)
    return pcl.FinitePlaneSource(
        600,
        q,
        alpha,
        np.cos(alpha)[:, None] * profile,
        np.sin(alpha)[:, None] * profile,
        medium_n=host,
        reference_point=(20, -30, -80),
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        amplitude=0.7 + 0.2j,
    )


def _fields(source, points, *, dtype=np.complex128, backend="numpy"):
    return compute_initial_field(
        points,
        k=2 * np.pi / source.wavelength * source.medium_n,
        n_medium=source.medium_n,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        compute_dtype=dtype,
        backend=backend,
        show_progress=False,
    )


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.01j, 1.33 + 0.5j])
@pytest.mark.parametrize("angles", [(0, 0), (0.6, 0.9), (np.pi, 0.3)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_plane_boundary_matches_cartesian_maxwell_waves_and_regular_projection(
    host, angles, dtype, backend, request
):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    source = _gaussian_plane(host, polar_angle=angles[0], azimuthal_angle=angles[1])
    points = np.array([[30, -20, -60], [130, 50, 80], [-40, -90, 10]], dtype=float)
    electric, magnetic = _fields(source, points, dtype=dtype, backend=backend)
    local = (points - source.reference_point) @ source.plane_frame()
    alpha, q = source.spectrum_azimuths, source.transverse_wavenumbers
    k = 2 * np.pi / 600 * host
    kz = np.sqrt(k * k - q * q + 0j)
    # Direct Cartesian Maxwell waves, independent of the spectrum's polarization
    # and quadrature helpers and of its spherical-wave projection.
    wq = np.full(len(q), q[1] - q[0])
    wq[[0, -1]] *= 0.5
    expected_e, expected_h = np.zeros_like(electric), np.zeros_like(magnetic)
    for ia, angle in enumerate(alpha):
        for iq, transverse in enumerate(q):
            vector = np.array([transverse * np.cos(angle), transverse * np.sin(angle), kz[iq]])
            te, tm = (
                source.amplitude * source.coeff_te[ia, iq],
                source.amplitude * source.coeff_tm[ia, iq],
            )
            e = np.array(
                [
                    -te * np.sin(angle) + tm * np.cos(angle),
                    te * np.cos(angle) + tm * np.sin(angle),
                    -transverse / kz[iq] * tm,
                ]
            )
            h = host * np.cross(vector / k, e)
            phase = (
                np.exp(1j * (local @ vector)) * transverse * wq[iq] * (2 * np.pi / len(alpha))
            )[:, None]
            expected_e += phase * (e @ source.plane_frame().T)
            expected_h += phase * (h @ source.plane_frame().T)
    tolerance = 2e-6 if dtype == np.complex64 else 3e-13
    for value, expected in ((electric, expected_e), (magnetic, expected_h)):
        assert np.linalg.norm(value - expected) / np.linalg.norm(expected) < tolerance
        assert value.dtype == np.complex128  # Default accumulation for both fields.
    coeffs = source.incident_coeffs(points, 1, dtype=dtype)
    for tau, field in ((2, expected_e), (1, 1j / host * expected_h)):
        reference = np.column_stack(
            (
                np.sqrt(3) * (field[:, 0] + 1j * field[:, 1]),
                np.sqrt(6) * field[:, 2],
                np.sqrt(3) * (field[:, 0] - 1j * field[:, 1]),
            )
        )
        columns = [index_vswf(1, m, tau, 1) for m in (-1, 0, 1)]
        assert (
            np.linalg.norm(coeffs[:, columns] - reference) / np.linalg.norm(reference) < tolerance
        )


def test_prescribed_tangential_plane_profile_is_independent_of_host():
    source = _gaussian_plane(1.33)
    points = np.array([[20, -30, -80], [80, -10, -80], [-60, 50, -80]])
    reference_e, _ = _fields(source, points)
    for host in (1.33 + 0.01j, 1.33 + 0.5j):
        electric, _ = _fields(replace(source, medium_n=host), points)
        np.testing.assert_allclose(electric[:, :2], reference_e[:, :2], rtol=2e-14, atol=2e-14)
    assert np.linalg.norm(electric[:, 2] - reference_e[:, 2]) > 0.01 * np.linalg.norm(
        reference_e[:, 2]
    )


def test_plane_boundary_real_host_pairwise_fields_and_source_blocks():
    source = _gaussian_plane(1.33, polar_angle=0.2, azimuthal_angle=0.4)
    sim = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=600,
            n_medium=source.medium_n,
            lmax=2,
            verbose=False,
            solver_method="direct",
            radial_lut_dr=600 / 4096,
            polar_angles=np.linspace(0, np.pi, 9),
            azimuthal_angles=np.arange(12) * np.pi / 6,
        ),
        particles=[
            pcl.Sphere((0, 0, 0), 35, 1.7 + 0.03j),
            pcl.Sphere((160, 20, 30), 40, 1.6 + 0.02j),
        ],
    )
    run = sim.run(source)
    assert run.farfield.initial is None and run.farfield.total is None
    assert np.isfinite(run.farfield.scattered.coeff_te).all()
    assert run.power is None and run.cross_sections is None
    fields = pcl.compute_near_field(
        run, points=np.array([[0, 0, 0], [160, 20, 30], [90, 30, 80]]), show_progress=False
    )
    assert np.isfinite(fields.E_total).all() and np.isfinite(fields.H_total).all()
    block = sim.run_sources({"a": source, "b": replace(source, amplitude=2j * source.amplitude)})
    np.testing.assert_allclose(block["b"].coeffs, 2j * block["a"].coeffs, rtol=1e-12, atol=1e-14)
