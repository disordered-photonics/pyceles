"""Compact periodic translation/affine-solve reference in a passive host."""

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import Sphere
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.sources import DipoleSource
from pyceles.simulation import Simulation, SimulationConfig

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
def test_periodic_host_and_embedded_forcing_match_external_expansions(request, backend, method):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    # TREAMS 0.4.8.dev1 ExpandLattice and I-TW, converted from parity to the
    # package SVWF basis. Embedded emission was supplied by independent centred
    # sphere boundary matching; these are not reference internal-emitter fields.
    embedded_coeffs = np.array(
        [
            8.970965180894399e-13 - 2.150654942802063e-13j,
            1.5493745511315398e-12 - 9.247808733055679e-13j,
            -5.240831198270143e-13 - 1.6762798048703876e-13j,
            -7.145737714770654e-8 + 1.2723059970749436e-7j,
            3.790497147756273e-8 - 6.743422274254082e-8j,
            -1.0721420537460487e-7 + 1.908144351187048e-7j,
        ]
    )
    host_incident = np.array(
        [
            -2.613407424964692e-10 + 2.9665625185351064e-9j,
            -9.787533818516607e-9 + 1.2682876143433577e-8j,
            -5.311327802338032e-9 - 1.8357876491410705e-9j,
            3.4651719203040176e-9 + 4.32633654896833e-9j,
            -2.775930559636669e-9 - 2.315621470336898e-9j,
            2.5912080990728004e-9 + 1.4116567962212247e-8j,
        ]
    )
    cfg = SimulationConfig(
        wavelength=550,
        n_medium=1.33 + 0.4j,
        lmax=1,
        verbose=False,
        solver_method="direct",
        operator_backend=backend,
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(method=method, eta=0.003, real_shells=4, reciprocal_shells=6),
            k_parallel=(0.0013, -0.0007),
        ),
    )
    embedded = DipoleSource(
        550, medium_n=cfg.n_medium, position=(30, 20, 0), dipole_moment=(1, 0.2j, -0.3)
    )
    host = DipoleSource(
        550, medium_n=cfg.n_medium, position=(200, 210, 20), dipole_moment=(0.1j, 0.6, 0.2)
    )
    result = Simulation(cfg, particles=(Sphere((30, 20, 0), 35, 1.7 + 0.03j),)).solve_sources(
        {"embedded": embedded, "host": host}
    )
    np.testing.assert_allclose(result.coeffs["embedded"][0], embedded_coeffs, rtol=1e-9, atol=1e-23)
    np.testing.assert_allclose(
        result.initial_coeffs["host"][0], host_incident, rtol=1e-9, atol=1e-23
    )
