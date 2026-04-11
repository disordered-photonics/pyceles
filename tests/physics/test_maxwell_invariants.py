from __future__ import annotations

import numpy as np

import pyceles as pcl


def _div(field: np.ndarray, h: float) -> complex:
    return (
        np.gradient(field[..., 0], h, axis=0)
        + np.gradient(field[..., 1], h, axis=1)
        + np.gradient(field[..., 2], h, axis=2)
    ).mean()


def _curl(field: np.ndarray, h: float) -> np.ndarray:
    cx = np.gradient(field[..., 2], h, axis=1) - np.gradient(field[..., 1], h, axis=2)
    cy = np.gradient(field[..., 0], h, axis=2) - np.gradient(field[..., 2], h, axis=0)
    cz = np.gradient(field[..., 1], h, axis=0) - np.gradient(field[..., 0], h, axis=1)
    return np.array([cx.mean(), cy.mean(), cz.mean()], dtype=np.complex128)


def test_source_only_plane_wave_satisfies_local_maxwell_identities():
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.3j),
        polar_angle=0.4,
        azimuthal_angle=0.2,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        source=source,
        solver_method="direct",
        verbose=False,
    )
    run = pcl.Simulation(
        cfg,
        particles=[],
    ).run()

    eps = 0.2
    x = np.array([15.0, 15.0 + eps], dtype=float)
    y = np.array([-12.0, -12.0 + eps], dtype=float)
    z = np.array([8.0, 8.0 + eps], dtype=float)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1)

    nf = pcl.compute_near_field(run, points=points, channel="mixed", show_progress=False)
    E = np.asarray(nf.E_total, dtype=np.complex128)
    H = np.asarray(nf.H_total, dtype=np.complex128)

    div_e = _div(E, eps)
    div_h = _div(H, eps)
    curl_e = _curl(E, eps)
    curl_h = _curl(H, eps)

    e_mean = E.mean(axis=(0, 1, 2))
    h_mean = H.mean(axis=(0, 1, 2))
    k0 = float(run.k0)
    e_scale = k0 * np.linalg.norm(e_mean)
    h_scale = k0 * np.linalg.norm(h_mean)

    assert abs(div_e) / e_scale < 1e-4
    assert abs(div_h) / h_scale < 1e-4
    np.testing.assert_allclose(curl_e, 1j * k0 * h_mean, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(curl_h, -1j * k0 * e_mean, rtol=1e-4, atol=1e-6)
