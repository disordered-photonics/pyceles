from __future__ import annotations

import numpy as np
import pytest

from pyceles import ParticleCollection, SimulationConfig, Sphere, TMatrixParticle
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import prepare_matvec
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.sources import DipoleCollection, PlaneWave
from pyceles.io.standard_tmatrix import TMatrixData
from pyceles.postprocessing.farfield.power import PowerBalance


def _assert_sealed(array: np.ndarray) -> None:
    assert not array.flags.owndata
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def test_public_immutable_arrays_do_not_expose_owning_buffers() -> None:
    matrix = np.eye(6, dtype=np.complex128)
    particle = TMatrixParticle(
        position=(0.0, 0.0, 0.0),
        radius=0.2,
        lmax=1,
        t_matrix=matrix,
    )
    collection = ParticleCollection.from_particles([particle])
    config = SimulationConfig(wavelength=1.0, lmax=1)
    dipoles = DipoleCollection(
        wavelength=1.0,
        positions=np.zeros((1, 3), dtype=float),
        dipole_moments=np.asarray([[1.0, 0.0, 0.0]], dtype=np.complex128),
    )
    data = TMatrixData(
        t_matrix=matrix,
        lmax=1,
        embedding={"relative_permittivity": np.asarray([1.0])},
    )
    balance = PowerBalance(
        local_absorbed_power=1.0,
        local_absorbed_power_per_particle=np.asarray([1.0]),
    )

    for array in (
        particle.t_matrix,
        collection.positions,
        collection.archetype_indices,
        collection.circumscribing_radii,
        config.polar_angles,
        config.azimuthal_angles,
        dipoles.positions,
        dipoles.dipole_moments,
        data.t_matrix,
        data.embedding["relative_permittivity"],
        balance.local_absorbed_power_per_particle,
    ):
        assert isinstance(array, np.ndarray)
        _assert_sealed(array)


def test_frozen_descriptors_detach_mutable_constructor_inputs() -> None:
    position = np.asarray([1.0, 2.0, 3.0])
    polarization = np.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    focal_point = np.asarray([4.0, 5.0, 6.0])

    sphere = Sphere(position=position, radius=1.0, refractive_index=1.5)
    source = PlaneWave(
        wavelength=1.0,
        polarization=polarization,
        focal_point=focal_point,
    )

    position[:] = 0.0
    polarization[:] = 0.0
    focal_point[:] = 0.0

    assert sphere.position == (1.0, 2.0, 3.0)
    assert source.polarization == (1.0 + 0.0j, 0.0 + 1.0j)
    assert source.focal_point == (4.0, 5.0, 6.0)


def test_prepared_periodic_operator_owns_bloch_vector() -> None:
    k_parallel = np.asarray([0.1, 0.2], dtype=float)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi,
        particles=[],
        radial_lut_dr=0.0,
        periodic=PeriodicSpec(RectangularLattice2D(ax=2.0, ay=2.0)),
        k_parallel=k_parallel,
    )

    k_parallel[:] = 9.0
    stored = prepared.coupling.k_parallel
    np.testing.assert_allclose(stored, [0.1, 0.2])
    _assert_sealed(stored)
