import numpy as np

from pyceles.core.fields import PlaneWave, incident_coeffs_planewave
from pyceles.core.matvec import make_prepared_A_and_rhs, prepare_matvec
from pyceles.core.particles import Sphere, spheres_from_arrays
from pyceles.postprocessing.nearfield import compute_internal_field


def _single_sphere_internal_field(
    *,
    vacuum_wavelength: float,
    n_medium: complex,
    n_particle: complex,
    sphere_radius: float,
    lmax: int,
    points: np.ndarray,
) -> np.ndarray:
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    radii = np.array([sphere_radius], dtype=float)

    source = PlaneWave(
        wavelength=vacuum_wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
        focal_point=(0.0, 0.0, 0.0),
    )
    b = incident_coeffs_planewave(positions, lmax, source)

    k_medium = 2 * np.pi / vacuum_wavelength * float(np.real(n_medium))
    prepared = prepare_matvec(
        lmax=lmax,
        k=k_medium,
        particles=spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=np.array([n_particle], dtype=np.complex128),
        ),
        n_medium=n_medium,
        radial_lut_dr=1.0,
        cache_translation_blocks=True,
    )
    _, rhs = make_prepared_A_and_rhs(prepared, b.reshape(-1))
    coeffs = rhs.reshape(1, -1)

    E, _, inside = compute_internal_field(
        points,
        coeffs,
        k=k_medium,
        lmax=lmax,
        particles=(
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=float(sphere_radius),
                refractive_index=complex(n_particle),
            ),
        ),
        n_medium=n_medium,
        show_progress=False,
    )
    assert np.all(inside)
    return E


def test_internal_field_against_smuthi_celes_reference_values():
    """Ported from SMUTHI `test_internal_field.py` (homogeneous medium case)."""
    vacuum_wavelength = 550.0
    sphere_radius = 200.0
    lmax = 3

    points = np.column_stack(
        [
            [1.0, 24.0, -18.0, 180.0],
            [0.0, -24.0, 29.0, 0.0],
            [1.0, 13.0, -43.0, 0.0],
        ]
    )

    E_diel_ref = np.array(
        [
            [0.0000000 + 0.0000000j, 0.7665766 + 0.8608202j, 0.0000000 + 0.0000000j],
            [0.0144948 - 0.0053389j, 0.5851322 + 0.9504231j, -0.1699689 - 0.1021124j],
            [0.0102967 - 0.0049106j, 1.1293162 + 0.2558453j, 0.1316240 + 0.0713801j],
            [0.0000000 + 0.0000000j, 0.8560891 + 0.1384031j, 0.0000000 + 0.0000000j],
        ],
        dtype=np.complex128,
    )

    E_metal_ref = np.array(
        [
            [0.0000000 + 0.0000000j, 0.0000015 - 0.0000010j, 0.0000000 - 0.0000000j],
            [-0.0000007 + 0.0000009j, 0.0000034 + 0.0000002j, 0.0000006 - 0.0000029j],
            [-0.0000010 - 0.0000000j, -0.0000052 - 0.0000343j, 0.0000142 - 0.0000098j],
            [0.0000000 + 0.0000000j, 0.0051606 - 0.0456204j, 0.0000000 + 0.0000000j],
        ],
        dtype=np.complex128,
    )

    E_diel = _single_sphere_internal_field(
        vacuum_wavelength=vacuum_wavelength,
        n_medium=1.0 + 0j,
        n_particle=1.5 + 0j,
        sphere_radius=sphere_radius,
        lmax=lmax,
        points=points,
    )
    E_metal = _single_sphere_internal_field(
        vacuum_wavelength=vacuum_wavelength,
        n_medium=1.33 + 0j,
        n_particle=1.1 + 6.1j,
        sphere_radius=sphere_radius,
        lmax=lmax,
        points=points,
    )

    np.testing.assert_allclose(E_diel, E_diel_ref, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(E_metal, E_metal_ref, rtol=2e-5, atol=1e-6)
