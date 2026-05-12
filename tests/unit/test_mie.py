import numpy as np
import pytest

from pyceles.core.tmatrix import mie_ab, mie_cross_sections, mie_efficiencies, sphere_T_diagonal


@pytest.mark.reference
def test_mie_coeffs_against_reference_values():
    """Reference values from the SMUTHI prototype test case."""
    l = 4
    k0 = 2 * 3.15 / 550
    n_medium = 1.6 + 0.1j
    n_particle = 2.6 + 0.4j
    radius = 260

    a, b = mie_ab(
        lmax=10,
        k_medium=k0 * n_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )

    np.testing.assert_allclose(
        a[l], 0.5112721702623048 + 0.061284547954858236j, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        b[l], 0.42146910417721567 + 0.3466983965849721j, rtol=1e-12, atol=1e-12
    )


def test_sphere_T_diagonal_sign_convention():
    """CELES convention: tau=1 -> -b_l, tau=2 -> -a_l."""
    lmax = 8
    k0 = 2 * np.pi / 550
    n_medium = 1.33 + 0.0j
    n_particle = 2.1 + 0.3j
    radius = 120

    a, b = mie_ab(
        lmax=lmax,
        k_medium=k0 * n_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    tdiag = sphere_T_diagonal(
        lmax=lmax,
        k_medium=k0 * n_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )

    np.testing.assert_allclose(tdiag[1][1:], -b[1:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tdiag[2][1:], -a[1:], rtol=1e-12, atol=1e-12)


def test_mie_cross_sections_lossless_sphere_has_near_zero_absorption():
    wl = 550.0
    n_medium = 1.0 + 0j
    n_particle = 1.5 + 0j
    radius = 100.0
    k = 2 * np.pi / wl * n_medium

    cs = mie_cross_sections(
        lmax=30,
        k_medium=k,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    assert cs["C_ext"] > 0
    assert cs["C_sca"] > 0
    assert abs(cs["C_abs"]) < 1e-10
    np.testing.assert_allclose(cs["C_ext"], cs["C_sca"] + cs["C_abs"], rtol=1e-13, atol=1e-13)


def test_mie_cross_sections_absorption_sign_for_passive_vs_active():
    wl = 550.0
    n_medium = 1.0 + 0j
    radius = 100.0
    k = 2 * np.pi / wl * n_medium

    cs_passive = mie_cross_sections(
        lmax=30,
        k_medium=k,
        radius=radius,
        n_particle=1.5 + 0.01j,
        n_medium=n_medium,
    )
    cs_active = mie_cross_sections(
        lmax=30,
        k_medium=k,
        radius=radius,
        n_particle=1.5 - 0.01j,
        n_medium=n_medium,
    )
    assert cs_passive["C_abs"] > 0
    assert cs_active["C_abs"] < 0


def test_mie_efficiencies_are_area_normalized_cross_sections():
    wl = 550.0
    n_medium = 1.33 + 0j
    n_particle = 2.0 + 0.1j
    radius = 120.0
    k = 2 * np.pi / wl * n_medium

    cs = mie_cross_sections(
        lmax=35,
        k_medium=k,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    qs = mie_efficiencies(
        lmax=35,
        k_medium=k,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    area = np.pi * radius**2
    np.testing.assert_allclose(qs["Q_ext"], cs["C_ext"] / area, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(qs["Q_sca"], cs["C_sca"] / area, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(qs["Q_abs"], cs["C_abs"] / area, rtol=1e-13, atol=1e-13)
