import h5py
import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.tmatrix import particle_T_matrix_block
from pyceles.io.hdf5 import load_geometry_h5, save_geometry_h5
from pyceles.postprocessing.nearfield import compute_internal_field
from pyceles.postprocessing.nearfield.classification import classify_internal_points

pytestmark = [pytest.mark.filesystem, pytest.mark.hdf5]


def _storage_from_native(
    native: np.ndarray, lmax: int, *, basis: str = "parity"
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Encode a CELES block in the published interleaved convention."""
    if basis not in {"parity", "helicity"}:
        raise ValueError(basis)
    native_labels = tuple(
        (int(mode[1]), int(mode[2]), "magnetic" if int(mode[0]) == 1 else "electric")
        for mode in iter_modes(lmax)
    )
    if basis == "helicity":
        parity_labels = native_labels
        source_labels = tuple(
            (l, m, pol)
            for pol in ("positive", "negative")
            for l in range(1, lmax + 1)
            for m in range(-l, l + 1)
        )
        change = np.zeros((len(parity_labels), len(source_labels)), dtype=float)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for row, (l, m, pol) in enumerate(parity_labels):
            positive = source_labels.index((l, m, "positive"))
            negative = source_labels.index((l, m, "negative"))
            change[row, positive] = inv_sqrt2
            change[row, negative] = inv_sqrt2 if pol == "electric" else -inv_sqrt2
        source = change.T @ native @ change
        native_labels = source_labels
    else:
        source = native
    storage_polarizations = (
        ("electric", "magnetic")
        if basis == "parity"
        else (
            "positive",
            "negative",
        )
    )
    storage_labels = [
        (l, m, polarization)
        for l in range(1, lmax + 1)
        for m in range(-l, l + 1)
        for polarization in storage_polarizations
    ]
    native_m = np.asarray([label[1] for label in native_labels], dtype=float)
    phase = (1j / np.sqrt(np.pi)) * np.where(native_m > 0.0, (-1.0) ** native_m, 1.0)
    parity_storage = phase[None, :] * source / phase[:, None]
    native_index = {label: index for index, label in enumerate(native_labels)}
    order = [native_index[label] for label in storage_labels]
    return parity_storage[np.ix_(order, order)], storage_labels


def _write_standard_file(
    path, native: np.ndarray, lmax: int, *, wavelengths=(600.0,), basis: str = "parity"
) -> None:
    stored, labels = _storage_from_native(native, lmax, basis=basis)
    with h5py.File(path, "w") as root:
        root.create_dataset("tmatrix", data=np.stack([stored] * len(wavelengths)))
        root.create_dataset("vacuum_wavelength", data=np.asarray(wavelengths, dtype=float))
        embedding = root.create_group("embedding")
        embedding.create_dataset("relative_permittivity", data=1.0)
        embedding.create_dataset("relative_permeability", data=1.0)
        modes = root.create_group("modes")
        modes.create_dataset("l", data=np.asarray([label[0] for label in labels], dtype=np.int64))
        modes.create_dataset("m", data=np.asarray([label[1] for label in labels], dtype=np.int64))
        modes.create_dataset(
            "polarization", data=np.asarray([label[2] for label in labels], dtype="S8")
        )


def test_standard_tmatrix_import_reorders_and_normalizes(tmp_path):
    lmax = 2
    size = n_modes(lmax)
    rng = np.random.default_rng(12)
    native = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    path = tmp_path / "particle.tmat.h5"
    _write_standard_file(path, native, lmax)

    data = pcl.load_tmatrix_h5(path, wavelength=600.0)

    np.testing.assert_allclose(data.t_matrix, native)
    assert data.lmax == lmax
    assert data.wavelength == 600.0
    assert data.t_matrix.flags.writeable is False
    assert data.embedding_refractive_index == 1.0 + 0j


def test_standard_tmatrix_preserves_complex64_storage_dtype(tmp_path):
    path = tmp_path / "complex64.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        stored = np.asarray(root["tmatrix"][...], dtype=np.complex64)
        del root["tmatrix"]
        root.create_dataset("tmatrix", data=stored)
    data = pcl.load_tmatrix_h5(path, wavelength=600.0)
    assert data.t_matrix.dtype == np.dtype(np.complex64)


def test_standard_tmatrix_uses_published_mode_order_and_phase(tmp_path):
    path = tmp_path / "literal.tmat.h5"
    stored = np.zeros((6, 6), dtype=np.complex128)
    stored[np.diag_indices(6)] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    stored[4, 1] = 2.0 + 3.0j
    labels = [
        (1, -1, "electric"),
        (1, -1, "magnetic"),
        (1, 0, "electric"),
        (1, 0, "magnetic"),
        (1, 1, "electric"),
        (1, 1, "magnetic"),
    ]
    with h5py.File(path, "w") as root:
        root.create_dataset("tmatrix", data=stored)
        root.create_dataset("vacuum_wavelength", data=np.asarray([600.0]))
        modes = root.create_group("modes")
        modes.create_dataset("l", data=np.asarray([label[0] for label in labels]))
        modes.create_dataset("m", data=np.asarray([label[1] for label in labels]))
        modes.create_dataset(
            "polarization", data=np.asarray([label[2] for label in labels], dtype="S8")
        )

    data = pcl.load_tmatrix_h5(path)

    expected = np.diag((20.0, 40.0, 60.0, 10.0, 30.0, 50.0)).astype(np.complex128)
    # The m=1 row / m=-1 column phase ratio is -1 under the published c(m).
    expected[5, 0] = -2.0 - 3.0j
    np.testing.assert_allclose(data.t_matrix, expected)


def test_tmatrix_context_validation_requires_matching_metadata(tmp_path):
    path = tmp_path / "particle.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root["vacuum_wavelength"].attrs["unit"] = "nm"
    data = pcl.load_tmatrix_h5(path, wavelength_index=0)

    data.validate_context(wavelength=600.0, n_medium=1.0 + 0j)
    with pytest.raises(ValueError, match="incompatible"):
        data.validate_context(wavelength=601.0, n_medium=1.0 + 0j)
    with pytest.raises(ValueError, match="finite"):
        data.as_particle(
            position=(0.0, np.nan, 0.0),
            radius=1.0,
            wavelength=600.0,
            n_medium=1.0,
        )


def test_as_particle_validates_context_and_owns_selected_block(tmp_path):
    path = tmp_path / "particle.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    data = pcl.load_tmatrix_h5(path, wavelength=600.0)

    particle = data.as_particle(
        position=(0.0, 0.0, 0.0),
        radius=1.0,
        wavelength=600.0,
        n_medium=1.0,
    )
    assert particle.t_matrix is not data.t_matrix
    with pytest.raises(ValueError, match="incompatible"):
        data.as_particle(
            position=(0.0, 0.0, 0.0),
            radius=1.0,
            wavelength=601.0,
            n_medium=1.0,
        )


def test_spectral_embedding_metadata_follows_selected_matrix(tmp_path):
    path = tmp_path / "spectral-embedding.tmat.h5"
    _write_standard_file(
        path, np.eye(n_modes(1), dtype=np.complex128), 1, wavelengths=(500.0, 600.0)
    )
    with h5py.File(path, "r+") as root:
        embedding = root["embedding"]
        del embedding["relative_permittivity"]
        del embedding["relative_permeability"]
        embedding.create_dataset("relative_permittivity", data=np.asarray([1.0, 4.0]))
        embedding.create_dataset("relative_permeability", data=np.asarray([1.0, 1.0]))
    data = pcl.load_tmatrix_h5(path, wavelength=600.0)
    assert data.embedding_refractive_index == 2.0 + 0j
    data.validate_context(wavelength=600.0, n_medium=2.0)


def test_standard_import_rejects_displaced_and_split_mode_sets(tmp_path):
    path = tmp_path / "unsupported-modes.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root["modes"].create_dataset("positions", data=np.asarray([[1.0, 0.0, 0.0]]))
    with pytest.raises(NotImplementedError, match="center"):
        pcl.load_tmatrix_h5(path)

    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root["modes"].create_dataset("l_incident", data=np.asarray([1]))
    with pytest.raises(NotImplementedError, match="incident/scattered"):
        pcl.load_tmatrix_h5(path)


def test_standard_tmatrix_requires_unambiguous_spectral_selection(tmp_path):
    native = np.eye(n_modes(1), dtype=np.complex128)
    path = tmp_path / "particle.tmat.h5"
    _write_standard_file(path, native, 1, wavelengths=(500.0, 600.0))

    with pytest.raises(ValueError, match="choose wavelength"):
        pcl.load_tmatrix_h5(path)
    with pytest.raises(ValueError, match="No exact wavelength"):
        pcl.load_tmatrix_h5(path, wavelength=550.0)
    with pytest.raises(ValueError, match="only one"):
        pcl.load_tmatrix_h5(path, wavelength=600.0, wavelength_index=1)

    _write_standard_file(path, native, 1, wavelengths=(600.0, 600.0 + 1.0e-12))
    with pytest.raises(ValueError, match="matches 2"):
        pcl.load_tmatrix_h5(path, wavelength=600.0)


def test_frequency_axis_is_reported_as_si_wavelength(tmp_path):
    path = tmp_path / "frequency.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root.move("vacuum_wavelength", "frequency")
        root["frequency"][...] = np.asarray([500.0])
        root["frequency"].attrs["unit"] = "THz"

    data = pcl.load_tmatrix_h5(path, wavelength_index=0)

    assert data.wavelength_unit == "m"
    assert data.wavelength is not None
    np.testing.assert_allclose(data.wavelength, 299_792_458.0 / (500.0e12))


def test_inverse_time_frequency_unit_uses_reciprocal_prefix_scale(tmp_path):
    path = tmp_path / "inverse-time-frequency.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root.move("vacuum_wavelength", "frequency")
        root["frequency"][...] = np.asarray([2.0])
        root["frequency"].attrs["unit"] = "ps^{-1}"

    data = pcl.load_tmatrix_h5(path, wavelength_index=0)

    assert data.wavelength_unit == "m"
    assert data.wavelength is not None
    np.testing.assert_allclose(data.wavelength, 299_792_458.0 / (2.0e12))


def test_frequency_axis_requires_explicit_unit(tmp_path):
    path = tmp_path / "unitless-frequency.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root.move("vacuum_wavelength", "frequency")
        root["frequency"][...] = np.asarray([500.0])

    with pytest.raises(ValueError, match="explicit frequency unit"):
        pcl.load_tmatrix_h5(path, wavelength_index=0)


def test_reciprocal_length_unit_is_normalized(tmp_path):
    path = tmp_path / "wavenumber.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        del root["vacuum_wavelength"]
        axis = root.create_dataset("vacuum_wavenumber", data=np.asarray([1.0 / 600.0]))
        axis.attrs["unit"] = "1/nm"

    data = pcl.load_tmatrix_h5(path)

    assert data.wavelength_unit == "nm"
    assert data.wavelength is not None
    np.testing.assert_allclose(data.wavelength, 600.0)


def test_standard_tmatrix_rejects_invalid_spectral_axis_shape_and_values(tmp_path):
    path = tmp_path / "invalid-axis.tmat.h5"
    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        del root["vacuum_wavelength"]
        root.create_dataset("vacuum_wavelength", data=np.asarray([[600.0]]))
    with pytest.raises(ValueError, match="one-dimensional"):
        pcl.load_tmatrix_h5(path)

    _write_standard_file(path, np.eye(n_modes(1), dtype=np.complex128), 1)
    with h5py.File(path, "r+") as root:
        root["vacuum_wavelength"][...] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        pcl.load_tmatrix_h5(path)


def test_tmatrix_particle_uses_dense_operator_and_roundtrips_geometry(tmp_path):
    lmax = 1
    matrix = np.eye(n_modes(lmax), dtype=np.complex128)
    particle = pcl.TMatrixParticle(
        position=(1.0, 2.0, 3.0),
        radius=4.0,
        lmax=lmax,
        t_matrix=matrix,
    )
    assert particle.t_operator_representation == "dense"
    np.testing.assert_allclose(particle_T_matrix_block(1, 2.0, particle, 1.0), matrix)

    path = tmp_path / "geometry.h5"
    save_geometry_h5(path, particles=(particle,), n_medium=1.0, wavelength=600.0, lmax=lmax)
    loaded = load_geometry_h5(path)["particles"]
    assert isinstance(loaded[0], pcl.TMatrixParticle)
    np.testing.assert_allclose(loaded[0].t_matrix, matrix)


def test_imported_particle_near_field_is_explicitly_unavailable():
    particle = pcl.TMatrixParticle(
        position=(0.0, 0.0, 0.0),
        radius=2.0,
        lmax=1,
        t_matrix=np.eye(n_modes(1), dtype=np.complex128),
    )
    points = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    classification = classify_internal_points(points, (particle,))
    assert classification.inside_any.tolist() == [True, False]
    e, h, inside = compute_internal_field(
        points,
        np.zeros(n_modes(1), dtype=np.complex128),
        k=1.0,
        lmax=1,
        particles=(particle,),
        _point_classification=classification,
    )
    assert inside.tolist() == [True, False]
    assert np.isnan(e[0]).all() and np.isnan(h[0]).all()
    assert np.all(e[1] == 0.0) and np.all(h[1] == 0.0)


def test_helicity_storage_is_converted_to_celes_parity(tmp_path):
    lmax = 2
    size = n_modes(lmax)
    rng = np.random.default_rng(23)
    native = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    path = tmp_path / "helicity.tmat.h5"
    _write_standard_file(path, native, lmax, basis="helicity")

    data = pcl.load_tmatrix_h5(path)

    np.testing.assert_allclose(data.t_matrix, native)
    assert data.source_basis == "helicity"
