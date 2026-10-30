"""Saved source constructor data preserves excitation and embedded ownership."""

from typing import Any, cast

import numpy as np
import pytest

from pyceles.core import sources
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import LayeredSphere
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.io.hdf5 import load_mapping_h5, load_periodic_h5, save_mapping_h5, save_periodic_h5
from pyceles.io.workflows import _source_mapping, load_simulation_h5, save_simulation_h5
from pyceles.postprocessing.dipole_metrics import compute_dipole_source_work
from pyceles.postprocessing.farfield import compute_periodic_outgoing_amplitudes
from pyceles.postprocessing.nearfield import (
    compute_near_field_slice,
    compute_periodic_near_field_slice,
)
from pyceles.simulation import ResultRetention, Simulation, SimulationConfig

pytestmark = [pytest.mark.filesystem, pytest.mark.hdf5]


def test_outgoing_periodic_orders_keep_finite_references_in_hdf5(tmp_path):
    cfg = SimulationConfig(
        wavelength=550,
        n_medium=1.33 + 0.08j,
        lmax=1,
        verbose=False,
        periodic=PeriodicSpec(RectangularLattice2D(600, 650), k_parallel=(0.0013, -0.0007)),
    )
    run = Simulation(cfg, particles=()).run(
        sources.DipoleSource(
            550, medium_n=cfg.n_medium, position=(230, 120, 1e6), dipole_moment=(1, 0.2j, -0.3)
        )
    )
    outgoing = compute_periodic_outgoing_amplitudes(run, output_bmax=0.07)
    path = tmp_path / "outgoing.h5"
    save_periodic_h5(path, periodic=outgoing, mode="w")
    saved = load_periodic_h5(path)
    np.testing.assert_array_equal(saved["reference_z"], outgoing.reference_z)
    np.testing.assert_array_equal(saved["order_kz"], outgoing.orders.kz)
    np.testing.assert_array_equal(saved["up_amplitudes"], outgoing.up_amplitudes)
    np.testing.assert_array_equal(saved["down_amplitudes"], outgoing.down_amplitudes)
    assert cfg.periodic is not None
    np.testing.assert_array_equal(saved["k_parallel"], cfg.periodic.k_parallel)
    assert "reflected_amplitudes" not in saved and "power" not in saved


def test_periodic_embedded_source_preserves_boundary_without_plane_wave_metadata(tmp_path):
    cfg = SimulationConfig(
        wavelength=550,
        lmax=2,
        solver_method="direct",
        verbose=False,
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(
                method="ewald",
                real_shells=3,
                reciprocal_shells=4,
            ),
            k_parallel=(0.0013, -0.0007),
        ),
    )
    source = sources.DipoleSource(550, position=(10, 0, 0), dipole_moment=(1, 0.2j, 0))
    run = Simulation(cfg, particles=(LayeredSphere((0, 0, 0), (20, 35), (1.5, 1.8)),)).run(
        source,
        retention=ResultRetention.minimal(),
    )
    near = compute_periodic_near_field_slice(
        run, axis_0_min=40, axis_0_max=50, axis_1_min=-5, axis_1_max=5, dx=10
    )
    path = save_simulation_h5(run, near, tmp_path / "periodic_dipole.h5")
    saved = load_simulation_h5(path)
    boundary = cast(dict[str, Any], saved["periodic_configuration"])
    metadata = cast(dict[str, Any], saved["source"])
    assert cfg.periodic is not None
    np.testing.assert_array_equal(boundary["k_parallel"], cfg.periodic.k_parallel)
    assert not boundary["power_diagnostics_available"]
    assert "incident_direction" not in boundary and "incident_jones_coefficients" not in boundary
    assert "periodic" not in saved and "far_field" not in saved
    restored = sources.DipoleSource(**metadata["parameters"])
    np.testing.assert_array_equal(restored.outgoing_coeffs(), source.outgoing_coeffs())
    assert np.all(np.isfinite(compute_dipole_source_work(run).power_total))


@pytest.mark.parametrize(
    "source",
    [
        sources.PlaneWave(550, polarization=(0.7, 0.3j)),
        sources.BlochPlaneWave(550, k_parallel=(0.001, -0.002)),
        sources.GaussianBeam(550, beam_width=800),
        sources.LaguerreGaussianBeam(550, radial_order_p=1, azimuthal_order_l=2),
        sources.FocusedLaguerreGaussianBeam(550),
        sources.CartesianPolarizedFocusedLaguerreGaussianBeam(550),
        sources.BesselBeam(550),
        sources.CartesianPolarizedBesselBeam(550),
        sources.FinitePlaneSource(
            550,
            np.array([0.001, 0.002]),
            np.array([0.2, 0.8]),
            np.array([[0.7, 0.3j], [0.1, 0.2j]]),
            np.array([[0.2j, -0.4], [0.3, 0.1j]]),
        ),
        sources.DipoleSource(550, position=(2, 3, 5), amplitude=0.7 - 0.2j),
        sources.DipoleCollection(
            550,
            positions=np.array([[2, 3, 5], [7, 11, 13]]),
            dipole_moments=np.array([[1, 0.2j, 0], [0, 0.5, 1j]]),
        ),
        sources.SLMSource(sources.GaussianBeam(550, beam_width=800), modulation=0.7 - 0.2j),
        sources.AngularSpectrumSLMSource(
            sources.CartesianPolarizedBesselBeam(550), modulation=np.ones((8, 7)) * (0.7 - 0.2j)
        ),
    ],
    ids=lambda source: type(source).__name__,
)
def test_native_source_parameters_roundtrip_without_changing_incident_coefficients(
    tmp_path, source
):
    path = tmp_path / "source.h5"
    save_mapping_h5(path, group="source", mapping=_source_mapping(source), mode="w")
    saved = load_mapping_h5(path, group="source")
    assert saved["parameters_complete"]
    parameters = saved["parameters"]
    if isinstance(source, (sources.SLMSource, sources.AngularSpectrumSLMSource)):
        parameters["base_source"] = type(source.base_source)(
            **parameters["base_source"]["parameters"]
        )
    restored = type(source)(**parameters)
    assert saved["type"] == type(source).__name__
    options = dict(
        polar_angles=np.linspace(0.05, np.pi / 2, 7), azimuthal_angles=np.arange(8) * np.pi / 4
    )
    points = np.array([[200, 100, 300], [-300, 50, 200]])
    np.testing.assert_array_equal(
        source.incident_coeffs(points, 2, **options), restored.incident_coeffs(points, 2, **options)
    )


def test_embedded_minimal_result_retains_source_state_for_fields_and_work(tmp_path):
    source = sources.DipoleCollection(
        550,
        positions=np.array([[20, 0, 0], [200, 0, 0]]),
        dipole_moments=np.array([[1, 0.2j, 0], [0, 0.5, 1j]]),
        amplitude=0.7 - 0.2j,
    )
    simulation = Simulation(
        SimulationConfig(
            wavelength=550, lmax=3, solver_method="direct", radial_lut_dr=0.25, verbose=False
        ),
        particles=(LayeredSphere((0, 0, 0), (50, 80), (1.5, 1.7 + 0.15j)),),
    )
    run = simulation.run(source, include_farfield=False, retention=ResultRetention.minimal())
    near = compute_near_field_slice(
        run, axis_0_min=0, axis_0_max=100, axis_1_min=-10, axis_1_max=10, dx=10, show_progress=False
    )
    path = save_simulation_h5(run, near, tmp_path / "embedded.h5")
    saved = load_simulation_h5(path)
    metadata = cast(dict[str, Any], saved["source"])
    restored = sources.DipoleCollection(**metadata["parameters"])
    restored_run = simulation.run(restored, include_farfield=False)
    restored_near = compute_near_field_slice(
        restored_run,
        axis_0_min=0,
        axis_0_max=100,
        axis_1_min=-10,
        axis_1_max=10,
        dx=10,
        show_progress=False,
    )
    np.testing.assert_array_equal(near.field_maps["total"], restored_near.field_maps["total"])
    np.testing.assert_array_equal(
        compute_dipole_source_work(run, reaction_lmax=32).power_total,
        compute_dipole_source_work(restored_run, reaction_lmax=32).power_total,
    )
    assert "rhs" not in cast(dict[str, Any], saved["solution"])


def test_callable_source_keeps_artifacts_without_claiming_restorable_parameters(tmp_path):
    source = sources.SLMSource(
        sources.GaussianBeam(550, beam_width=800), modulation=lambda a, b: np.exp(1j * a)
    )
    simulation = Simulation(
        SimulationConfig(
            wavelength=550,
            lmax=1,
            polar_angles=np.linspace(0, np.pi / 2, 7),
            azimuthal_angles=np.arange(8) * np.pi / 4,
            verbose=False,
        ),
        particles=[],
    )
    run = simulation.run(source, include_farfield=False)
    near = compute_near_field_slice(
        run, axis_0_min=0, axis_0_max=1, axis_1_min=0, axis_1_max=1, dx=1, show_progress=False
    )
    path = save_simulation_h5(run, near, tmp_path / "callable.h5")
    saved = load_simulation_h5(path)
    metadata = cast(dict[str, Any], saved["source"])
    assert metadata["type"] == "SLMSource"
    assert not metadata["parameters_complete"]
    assert "parameters" not in metadata
    np.testing.assert_array_equal(cast(dict[str, Any], saved["solution"])["coeffs"], run.coeffs)


def test_custom_source_protocol_does_not_require_dataclass_state_for_saving(tmp_path):
    class CustomSource:
        wavelength = 550.0
        medium_n = 1.0 + 0j
        amplitude = 1.0 + 0j
        outgoing_lmax = 1

        def __init__(self):
            self.source = sources.DipoleSource(550, position=(0, 0, -100))

        def has_finite_incident_power(self):
            return False

        def incident_coeffs(self, *args, **kwargs):
            return self.source.incident_coeffs(*args, **kwargs)

        def source_positions(self):
            return self.source.source_positions()

        def outgoing_coeffs(self, **kwargs):
            return self.source.outgoing_coeffs(**kwargs)

    source = CustomSource()
    simulation = Simulation(SimulationConfig(wavelength=550, lmax=1, verbose=False), particles=[])
    run = simulation.run(source, include_farfield=False)
    near = compute_near_field_slice(
        run, axis_0_min=0, axis_0_max=1, axis_1_min=0, axis_1_max=1, dx=1, show_progress=False
    )
    saved = load_simulation_h5(save_simulation_h5(run, near, tmp_path / "custom.h5"))
    metadata = cast(dict[str, Any], saved["source"])
    assert metadata["type"] == "CustomSource"
    assert not metadata["parameters_complete"]
    assert "parameters" not in metadata
