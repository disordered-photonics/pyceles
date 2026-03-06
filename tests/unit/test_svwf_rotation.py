import numpy as np

from pyceles.core.particles import Sphere, Spheroid
from pyceles.core.svwf_rotation import rotate_svwf_tmatrix_block, svwf_rotation_matrix
from pyceles.core.tmatrix import particle_T_matrix_block


def test_svwf_rotation_matrix_is_identity_for_zero_angles():
    R = svwf_rotation_matrix(3, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(R.shape[0]), rtol=0.0, atol=1e-13)


def test_svwf_rotation_matrix_is_unitary():
    R = svwf_rotation_matrix(3, 0.2, 0.4, -0.3)
    np.testing.assert_allclose(R.conj().T @ R, np.eye(R.shape[0]), rtol=1e-12, atol=1e-12)


def test_rotate_svwf_tmatrix_block_preserves_sphere_response():
    block = particle_T_matrix_block(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particle=Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0.0j),
        n_medium=1.0 + 0j,
    )
    rotated = rotate_svwf_tmatrix_block(block, 3, (0.2, 0.4, -0.3))
    np.testing.assert_allclose(rotated, block, rtol=1e-11, atol=1e-11)


def test_rotate_svwf_tmatrix_block_matches_explicit_dense_rotation():
    lmax = 3
    angles = (0.2, 0.4, -0.3)
    aligned = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=120.0,
        refractive_index=1.5 + 0.0j,
    )
    block = particle_T_matrix_block(
        lmax=lmax,
        k_medium=2.0 * np.pi / 550.0,
        particle=aligned,
        n_medium=1.0 + 0j,
    )
    rot_back = svwf_rotation_matrix(lmax, -angles[2], -angles[1], -angles[0])
    rot_fwd = svwf_rotation_matrix(lmax, *angles)
    explicit = rot_back.T @ block @ rot_fwd.T

    np.testing.assert_allclose(
        rotate_svwf_tmatrix_block(block, lmax, angles),
        explicit,
        rtol=1e-12,
        atol=1e-12,
    )


def test_particle_t_matrix_block_rotated_spheroid_matches_explicit_rotation_of_aligned_block():
    aligned = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=120.0,
        refractive_index=1.5 + 0.0j,
    )
    rotated = Spheroid(
        position=(0.0, 0.0, 0.0),
        equatorial_radius=90.0,
        polar_radius=120.0,
        refractive_index=1.5 + 0.0j,
        euler_angles=(0.2, 0.4, -0.3),
    )

    aligned_block = particle_T_matrix_block(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particle=aligned,
        n_medium=1.0 + 0j,
    )
    rotated_block = particle_T_matrix_block(
        lmax=3,
        k_medium=2.0 * np.pi / 550.0,
        particle=rotated,
        n_medium=1.0 + 0j,
    )

    np.testing.assert_allclose(
        rotated_block,
        rotate_svwf_tmatrix_block(aligned_block, 3, rotated.euler_angles),
        rtol=1e-12,
        atol=1e-12,
    )
