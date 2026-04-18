import numpy as np

from pyceles.core.particles import LayeredSphere, Sphere, Spheroid
from pyceles.io.hdf5 import (
    load_far_field_h5,
    load_geometry_h5,
    load_mapping_h5,
    load_near_field_components_h5,
    load_solution_h5,
    save_far_field_h5,
    save_geometry_h5,
    save_mapping_h5,
    save_near_field_components_h5,
    save_near_field_h5,
    save_solution_h5,
)


def _dummy_pwp(alpha: np.ndarray, beta: np.ndarray) -> dict:
    coeff = np.ones((alpha.size, beta.size), dtype=np.complex128)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": np.sin(bgrid) * np.cos(agrid),
        "ky": np.sin(bgrid) * np.sin(agrid),
        "kz": np.cos(bgrid) * np.ones_like(agrid),
        "coeff": coeff,
    }


def test_solution_roundtrip(tmp_path):
    path = tmp_path / "out.h5"
    rng = np.random.default_rng(4)
    coeffs = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    rhs = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    save_solution_h5(path, coeffs=coeffs, rhs=rhs, info=0, attrs={"solver": "gmres"})

    loaded = load_solution_h5(path)
    np.testing.assert_allclose(loaded["coeffs"], coeffs)
    np.testing.assert_allclose(loaded["rhs"], rhs)
    assert loaded["attrs"]["gmres_info"] == 0
    assert loaded["attrs"]["solver"] == "gmres"


def test_geometry_near_far_write(tmp_path):
    path = tmp_path / "run.h5"
    particles = (
        Sphere(position=(0.0, 0.0, 0.0), radius=0.5, refractive_index=1.5 + 0.01j),
        Sphere(position=(1.0, 2.0, 3.0), radius=0.6, refractive_index=1.6 + 0.02j),
    )

    X, Z = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-2, 2, 7), indexing="xy")
    E = np.zeros((*X.shape, 3), dtype=np.complex128)
    H = np.zeros_like(E)
    inside = np.zeros(X.shape, dtype=bool)

    alpha = np.linspace(0.0, 2 * np.pi, 17)
    beta = np.linspace(0.0, np.pi, 33)
    pwp = _dummy_pwp(alpha, beta)

    save_geometry_h5(
        path,
        particles=particles,
        n_medium=1.0 + 0j,
        wavelength=550.0,
        lmax=3,
        mode="w",
    )
    save_near_field_h5(path, X=X, Z=Z, E=E, H=H, inside=inside)
    save_far_field_h5(
        path,
        patterns={
            "initial": {"te": pwp, "tm": pwp},
            "scattered": {"te": pwp, "tm": pwp},
            "total": {"te": pwp, "tm": pwp},
        },
    )

    import h5py

    with h5py.File(path, "r") as h5:
        assert "geometry" in h5
        assert "near_field" in h5
        assert "far_field" in h5
        assert "positions" not in h5["geometry"]
        np.testing.assert_allclose(h5["near_field/X"][...], X)
        np.testing.assert_allclose(h5["far_field/initial/te/coeff"][...], pwp["coeff"])
        np.testing.assert_allclose(h5["far_field/scattered/te/coeff"][...], pwp["coeff"])
        assert bool(h5["far_field"].attrs["total_omitted_as_redundant"]) is True

    geom_loaded = load_geometry_h5(path)
    assert tuple(geom_loaded["particles"]) == particles
    ff_loaded = load_far_field_h5(path)
    np.testing.assert_allclose(ff_loaded["patterns"]["initial"]["te"]["coeff"], pwp["coeff"])


def test_geometry_particle_descriptor_roundtrip(tmp_path):
    path = tmp_path / "particles.h5"
    particles = (
        Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.01j),
        LayeredSphere(
            position=(200.0, 0.0, 0.0),
            layer_radii=(60.0, 110.0),
            layer_refractive_indices=(2.1 + 0.0j, 1.7 + 0.03j),
        ),
        Spheroid(
            position=(-150.0, 10.0, 25.0),
            equatorial_radius=80.0,
            polar_radius=40.0,
            refractive_index=1.8 + 0.0j,
            euler_angles=(0.2, 0.4, 0.6),
        ),
    )
    save_geometry_h5(
        path,
        particles=particles,
        n_medium=1.0 + 0j,
        wavelength=550.0,
        lmax=3,
        mode="w",
    )

    loaded = load_geometry_h5(path)
    loaded_particles = loaded["particles"]
    assert isinstance(loaded_particles, tuple)
    assert len(loaded_particles) == 3
    assert loaded_particles == particles


def test_near_field_components_write(tmp_path):
    path = tmp_path / "nf_components.h5"
    X, Z = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-2, 2, 3), indexing="xy")
    E = np.zeros((*X.shape, 3), dtype=np.complex128)
    H = np.ones_like(E)
    S = np.zeros((*X.shape, 3), dtype=np.float64)
    inside = np.zeros(X.shape, dtype=bool)

    save_near_field_components_h5(
        path,
        X=X,
        Z=Z,
        fields={
            "initial": {"E": E, "H": H},
            "scattered": {"E": E, "H": H},
            "total": {"E": E, "H": H, "S": S},
        },
        inside=inside,
    )

    import h5py

    with h5py.File(path, "r") as h5:
        assert "near_field_components" in h5
        np.testing.assert_allclose(h5["near_field_components/X"][...], X)
        np.testing.assert_allclose(h5["near_field_components/initial/H"][...], H)
        assert "total" not in h5["near_field_components"]
        assert bool(h5["near_field_components"].attrs["total_omitted_as_redundant"]) is True

    loaded = load_near_field_components_h5(path)
    np.testing.assert_allclose(loaded["X"], X)
    np.testing.assert_allclose(loaded["fields"]["initial"]["H"], H)


def test_save_mapping_h5_nested(tmp_path):
    path = tmp_path / "diag.h5"
    save_mapping_h5(
        path,
        mapping={
            "power": {"T": 1.2, "R": 0.3},
            "cross_sections": {"C_sca": 10.0, "C_ext": 12.0},
            "jones": {"a_te_real": 1.0, "a_tm_imag": 1.0},
        },
    )

    import h5py

    with h5py.File(path, "r") as h5:
        assert "diagnostics" in h5
        assert "power" in h5["diagnostics"]
        assert float(h5["diagnostics/power"].attrs["T"]) == 1.2
        assert float(h5["diagnostics/cross_sections"].attrs["C_sca"]) == 10.0

    loaded = load_mapping_h5(path)
    assert float(loaded["power"]["T"]) == 1.2
    assert float(loaded["cross_sections"]["C_sca"]) == 10.0
