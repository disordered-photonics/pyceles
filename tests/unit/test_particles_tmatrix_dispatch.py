import numpy as np
import pytest

from pyceles.core.particles import LayeredSphere, Sphere, Spheroid
from pyceles.core.tmatrix import (
    layered_internal_ab_ratios,
    layered_sphere_T_diagonal,
    particle_internal_ratios,
    particle_T_diagonal,
    particle_T_matrix_block,
    particle_T_matrix_blocks,
    sphere_T_diagonal,
)


def test_particle_t_diagonal_sphere_dispatch():
    lmax = 3
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    p = Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0.0j)
    out = particle_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        particle=p,
        n_medium=n_medium,
    )
    ref = sphere_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        radius=p.radius,
        n_particle=p.refractive_index,
        n_medium=n_medium,
    )
    assert 1 in out and 2 in out
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(out[2], ref[2], rtol=1e-13, atol=1e-13)


def test_particle_t_diagonal_layered_dispatch():
    p_layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(50.0, 100.0),
        layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
    )
    out = particle_T_diagonal(
        lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_layered, n_medium=1.0 + 0j
    )
    ref = layered_sphere_T_diagonal(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        layer_radii=p_layered.layer_radii,
        layer_refractive_indices=p_layered.layer_refractive_indices,
        n_medium=1.0 + 0j,
    )
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[2], ref[2], rtol=1e-12, atol=1e-12)


def test_particle_t_matrix_block_sphere_dispatch_is_exact_diagonal():
    lmax = 3
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    p = Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0.0j)
    block = particle_T_matrix_block(
        lmax=lmax,
        k_medium=k_medium,
        particle=p,
        n_medium=n_medium,
    )
    Td = particle_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        particle=p,
        n_medium=n_medium,
    )

    repeats = 2 * np.arange(1, lmax + 1) + 1
    diag = np.concatenate([np.repeat(Td[1][1:], repeats), np.repeat(Td[2][1:], repeats)])
    np.testing.assert_allclose(block, np.diag(diag), rtol=1e-13, atol=1e-13)


def test_particle_t_matrix_blocks_layered_stack_dispatch():
    particles = [
        Sphere(position=(0.0, 0.0, 0.0), radius=80.0, refractive_index=1.45 + 0.0j),
        LayeredSphere(
            position=(100.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
        ),
    ]
    blocks = particle_T_matrix_blocks(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
    )
    assert blocks.shape == (2, 30, 30)
    np.testing.assert_allclose(
        blocks[0],
        particle_T_matrix_block(
            lmax=3,
            k_medium=2.0 * np.pi / 550.0,
            particle=particles[0],
            n_medium=1.0 + 0j,
        ),
        rtol=1e-13,
        atol=1e-13,
    )


def test_particle_internal_ratios_layered_returns_core_regular_ratios():
    p_layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(50.0, 100.0),
        layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
    )
    out = particle_internal_ratios(
        lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_layered, n_medium=1.0 + 0j
    )
    ref = layered_internal_ab_ratios(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        layer_radii=p_layered.layer_radii,
        layer_refractive_indices=p_layered.layer_refractive_indices,
        n_medium=1.0 + 0j,
    )
    np.testing.assert_allclose(out[1], ref[1]["A"][0, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[2], ref[2]["A"][0, :], rtol=1e-12, atol=1e-12)


def test_particle_t_diagonal_spheroid_placeholder_raises():
    p_spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=110.0,
        refractive_index=1.5 + 0.0j,
    )
    with pytest.raises(NotImplementedError):
        particle_internal_ratios(
            lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_spheroid, n_medium=1.0 + 0j
        )
