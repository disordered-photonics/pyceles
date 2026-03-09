import numpy as np
import pytest

import pyceles.core.tmatrix as tmatrix_mod
from pyceles.core.particles import LayeredSphere, Particle, Sphere, Spheroid
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


def test_particle_t_matrix_blocks_reuses_identical_particle_blocks(monkeypatch):
    particles: list[Particle] = [
        LayeredSphere(
            position=(0.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
        ),
        LayeredSphere(
            position=(150.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
        ),
    ]

    calls = {"count": 0}
    original = particle_T_matrix_block

    def counted_particle_t_matrix_block(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pyceles.core.tmatrix.particle_T_matrix_block",
        counted_particle_t_matrix_block,
    )

    blocks = particle_T_matrix_blocks(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
    )

    assert calls["count"] == 1
    np.testing.assert_allclose(blocks[0], blocks[1], rtol=1e-13, atol=1e-13)


def test_particle_t_matrix_blocks_reuses_intrinsic_spheroid_block_across_orientations(monkeypatch):
    particles: list[Particle] = [
        Spheroid(
            position=(0.0, 0.0, 0.0),
            equatorial_radius=90.0,
            polar_radius=120.0,
            refractive_index=1.5 + 0.0j,
            euler_angles=(0.1, 0.2, 0.3),
        ),
        Spheroid(
            position=(150.0, 0.0, 0.0),
            equatorial_radius=90.0,
            polar_radius=120.0,
            refractive_index=1.5 + 0.0j,
            euler_angles=(0.4, 0.1, -0.2),
        ),
        Spheroid(
            position=(300.0, 0.0, 0.0),
            equatorial_radius=90.0,
            polar_radius=120.0,
            refractive_index=1.5 + 0.0j,
            euler_angles=(0.1, 0.2, 0.3),
        ),
    ]

    aligned_calls = {"count": 0}
    rotate_calls = {"count": 0}
    original_body = tmatrix_mod._aligned_spheroid_tmatrix_block
    original_rotate = tmatrix_mod.rotate_svwf_tmatrix_block

    def counted_body(*args, **kwargs):
        aligned_calls["count"] += 1
        return original_body(*args, **kwargs)

    def counted_rotate(*args, **kwargs):
        rotate_calls["count"] += 1
        return original_rotate(*args, **kwargs)

    monkeypatch.setattr(tmatrix_mod, "_aligned_spheroid_tmatrix_block", counted_body)
    monkeypatch.setattr(tmatrix_mod, "rotate_svwf_tmatrix_block", counted_rotate)

    blocks = particle_T_matrix_blocks(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
    )

    assert aligned_calls["count"] == 1
    assert rotate_calls["count"] == 2
    np.testing.assert_allclose(blocks[0], blocks[2], rtol=1e-12, atol=1e-12)


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


def test_particle_t_diagonal_spheroid_diagonal_path_raises():
    p_spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=110.0,
        refractive_index=1.5 + 0.0j,
    )
    with pytest.raises(NotImplementedError):
        particle_T_diagonal(
            lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_spheroid, n_medium=1.0 + 0j
        )
    with pytest.raises(NotImplementedError):
        particle_internal_ratios(
            lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_spheroid, n_medium=1.0 + 0j
        )


def test_particle_t_matrix_block_spheroid_reduces_to_sphere_for_equal_axes():
    lmax = 3
    k_medium = 2.0 * np.pi / 550.0
    n_medium = 1.0 + 0j
    n_particle = 1.5 + 0.0j
    sphere = Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=n_particle)
    spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=100.0,
        polar_radius=100.0,
        refractive_index=n_particle,
    )

    block_sphere = particle_T_matrix_block(
        lmax=lmax,
        k_medium=k_medium,
        particle=sphere,
        n_medium=n_medium,
    )
    block_spheroid = particle_T_matrix_block(
        lmax=lmax,
        k_medium=k_medium,
        particle=spheroid,
        n_medium=n_medium,
    )

    np.testing.assert_allclose(block_spheroid, block_sphere, rtol=5e-5, atol=5e-7)


def test_particle_t_matrix_block_rotated_spheroid_returns_dense_block():
    spheroid = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=120.0,
        refractive_index=1.5 + 0.0j,
        euler_angles=(0.1, 0.2, 0.3),
    )

    block = particle_T_matrix_block(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particle=spheroid,
        n_medium=1.0 + 0j,
    )
    assert block.shape == (30, 30)
