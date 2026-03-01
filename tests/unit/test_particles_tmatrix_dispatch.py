import numpy as np
import pytest

from pyceles.core.particles import Ellipsoid, LayeredSphere, Sphere
from pyceles.core.tmatrix import (
    particle_internal_ratios,
    particle_T_diagonal,
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


def test_particle_t_diagonal_placeholders_raise():
    p_layered = LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(50.0, 100.0),
        layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
    )
    p_ellip = Ellipsoid(
        position=(0.0, 0.0, 0.0),
        semi_axes=(80.0, 90.0, 110.0),
        refractive_index=1.5 + 0.0j,
    )
    with pytest.raises(NotImplementedError):
        particle_T_diagonal(
            lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_layered, n_medium=1.0 + 0j
        )
    with pytest.raises(NotImplementedError):
        particle_internal_ratios(
            lmax=3, k_medium=2.0 * np.pi / 550.0, particle=p_ellip, n_medium=1.0 + 0j
        )
