from __future__ import annotations

import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.matvec import (
    apply_A_numpy,
    assemble_dense_A_numpy,
    estimate_translation_cache_bytes,
    precompute_T_diagonal,
    prepare_matvec,
    rhs_Tb_numpy,
)
from pyceles.core.particles import LayeredSphere, Particle, Sphere, spheres_from_arrays
from pyceles.core.tmatrix import sphere_T_diagonal
from pyceles.core.translation import RadialLUT, translation_ab5_table


def _sample_problem():
    lmax = 3
    k = 2 * np.pi / 550.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [120.0, 15.0, -40.0],
            [-60.0, 45.0, 35.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    n_medium = 1.0 + 0j
    Nm = n_modes(lmax)
    Ns = positions.shape[0]
    rng = np.random.default_rng(7)
    x = rng.standard_normal(Ns * Nm) + 1j * rng.standard_normal(Ns * Nm)
    b = rng.standard_normal(Ns * Nm) + 1j * rng.standard_normal(Ns * Nm)
    return lmax, k, positions, radii, n_particle, particles, n_medium, x, b


def test_prepare_matvec_matches_explicit_operator_kernels():
    lmax, k, positions, _, _, particles, n_medium, x, b = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)
    ab5 = translation_ab5_table(lmax)
    rmax = float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)))
    lut = RadialLUT(lmax=lmax, k=k, r_max=rmax, dr=0.5)

    A_ref = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=lut)
    rhs_ref = rhs_Tb_numpy(lmax, b, T_M=T_M, T_N=T_N)

    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=True,
    )

    A_prepared = prepared.apply_A(x)
    rhs_prepared = prepared.rhs_Tb(b)

    np.testing.assert_allclose(A_prepared, A_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rhs_prepared, rhs_ref, rtol=1e-12, atol=1e-12)


def test_precompute_t_diagonal_matches_per_sphere_reference():
    lmax, k, _, radii, n_particle, particles, n_medium, _, _ = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)

    for i in range(len(particles)):
        Td = sphere_T_diagonal(
            lmax=lmax,
            k_medium=k,
            radius=float(radii[i]),
            n_particle=complex(n_particle[i]),
            n_medium=n_medium,
        )
        np.testing.assert_allclose(T_M[i], Td[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(T_N[i], Td[2], rtol=1e-12, atol=1e-12)


def test_prepare_matvec_accepts_mixed_sphere_and_layered():
    lmax, k, positions, radii, n_particle, _, n_medium, x, _ = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
        Sphere(
            position=tuple(positions[2].tolist()),
            radius=float(radii[2]),
            refractive_index=complex(n_particle[2]),
        ),
    ]
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    y = prepared.apply_A(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_apply_A_lookup_vs_direct_coupling_agree():
    """Lookup-coupling path should match explicit per-distance coupling evaluation."""
    lmax, k, positions, _, _, particles, n_medium, x, _ = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)
    ab5 = translation_ab5_table(lmax)
    rmax = float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)))
    # Use a dense lookup spacing so interpolation error stays below strict
    # lookup-vs-direct tolerance.
    lut = RadialLUT(lmax=lmax, k=k, r_max=rmax, dr=0.05)

    y_lut = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=lut)
    y_direct = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=None)
    np.testing.assert_allclose(y_lut, y_direct, rtol=5e-6, atol=1e-9)


def test_block_cache_fills_once_and_reuses():
    lmax, k, positions, _, _, particles, n_medium, x, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=True,
    )

    Ns = positions.shape[0]
    expected_blocks = Ns * (Ns - 1)

    _ = prepared.apply_A(x)
    cache_size_after_first = len(prepared._W_cache)
    _ = prepared.apply_A(x)
    cache_size_after_second = len(prepared._W_cache)

    assert cache_size_after_first == expected_blocks
    assert cache_size_after_second == expected_blocks


def test_estimate_translation_cache_bytes():
    n = 500
    lmax = 3
    nm = n_modes(lmax)
    expected = n * (n - 1) * nm * nm * np.dtype(np.complex128).itemsize
    assert estimate_translation_cache_bytes(n, lmax) == expected


def test_assemble_dense_A_matches_apply_A():
    lmax, k, positions, _, _, particles, n_medium, _, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    A = assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
    Nm = n_modes(lmax)
    n = positions.shape[0] * Nm
    assert A.shape == (n, n)

    rng = np.random.default_rng(3)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y_dense = A @ x
    y_mv = prepared.apply_A(x)
    np.testing.assert_allclose(y_dense, y_mv, rtol=1e-12, atol=1e-12)
