"""Sparse MSTM-v4 Cartesian internal E and Z0*H in an absorbing host.

Normal incidence with E(0)=+x, exp(-i*omega*t), lmax=5. MSTM's complex(4)
near-field buffer was printed with five significant digits; the inline values
retain that precision. All refractive indices are absolute, nonmagnetic.
"""

import numpy as np
import pytest

from pyceles.core.particles import spheres_from_arrays
from pyceles.core.sources import PlaneWave
from pyceles.postprocessing.nearfield import compute_near_field
from pyceles.simulation import Simulation, SimulationConfig

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_four_sphere_internal_fields_against_mstm(dtype, backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    host = 1.33 + 0.5j
    positions = np.array([[-312, -108, -132], [246, -186, -48], [-114, 288, 126], [198, 162, 264]])
    particles = spheres_from_arrays(
        positions=positions,
        radii=np.array([45, 57, 39, 51]),
        refractive_indices=2 + 0.1j,
    )
    simulation = Simulation(
        SimulationConfig(
            wavelength=600,
            n_medium=host,
            lmax=5,
            radial_lut_dr=600 / 4096,
            operator_backend=backend,
            compute_dtype=dtype.__name__,
            solver_method="direct",
            verbose=False,
        ),
        particles=particles,
    )
    source = PlaneWave(600, medium_n=host, polarization="TM")
    run = simulation.run(source, include_farfield=False)
    # Two samples on a line inside the first sphere, on either side of its centre.
    points = np.array([[-312, -108, -159], [-312, -108, -123]])
    fields = compute_near_field(run, points=points, show_progress=False)
    assert fields.inside_mask.all()
    np.testing.assert_array_equal(fields.E_total, fields.E_internal)
    np.testing.assert_array_equal(fields.H_total, fields.H_internal)
    expected_e = np.array(
        [
            [-0.64791 - 1.7896j, 7.1805e-5 + 2.1075e-4j, 1.1897e-4 - 2.5737e-4j],
            [0.35924 - 1.7521j, 1.0312e-4 + 2.0720e-4j, 1.0622e-4 - 2.0245e-4j],
        ]
    )
    expected_h = np.array(
        [
            [-4.2754e-5 + 4.6859e-5j, -1.4551 - 3.1772j, -3.8761e-4 - 2.9200e-4j],
            [2.3473e-5 - 2.2614e-7j, 0.69841 - 2.9458j, -4.6160e-4 - 2.0825e-4j],
        ]
    )
    # The single-precision reference qualifies complete vectors; its weak
    # components do not provide independent high-precision relative anchors.
    for actual, expected in ((fields.E_total, expected_e), (fields.H_total, expected_h)):
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-4
