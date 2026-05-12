"""Local regression against SMUTHI-generated isolated-spheroid observables.

This test intentionally hard-codes a compact set of reference cross sections
from a local SMUTHI diagnostic run so routine regression checks do not require
re-running SMUTHI.

The chosen spheroid is deliberately harder than the initial smoke-oracle:
- aspect ratio 3,
- refractive-index contrast 2.5:1,
- aligned and rotated orientations,
- TE and TM readout.

That makes it a better guard for both the spheroid backend and the far-field
``C_sca`` integration path.

Oracle note
-----------
The ``C_sca`` values here are not SMUTHI's built-in ``total_scattering_cross_section``
readout on the coarse open azimuth grid. They were obtained by re-integrating
the SMUTHI differential far field with periodic azimuth closure, which is the
same rule used by `pyceles` and is supported by sphere/Mie benchmarks.

The local SMUTHI/NFMDS reference run used:
- ``lmax = 14``,
- stable ``n_rank = 14``,
- ``n_beta = 721``,
- ``n_alpha = 28801``.

The public pyceles regression below intentionally uses a smaller output
sampling grid for speed. It is compared against that denser oracle, not against
SMUTHI totals evaluated on the same coarse bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

from pyceles.core.particles import Spheroid
from pyceles.core.sources import PlaneWave
from pyceles.simulation import Simulation, SimulationConfig, SimulationResult

pytestmark = [pytest.mark.reference, pytest.mark.slow]


@dataclass(frozen=True)
class SpheroidSmuthiOracle:
    name: str
    angles: tuple[float, float, float]
    cext_te: float
    csca_te: float
    cext_tm: float
    csca_tm: float


_ORACLES: tuple[SpheroidSmuthiOracle, ...] = (
    SpheroidSmuthiOracle(
        name="aligned",
        angles=(0.0, 0.0, 0.0),
        cext_te=267966.9656045385,
        csca_te=267866.3071677355,
        cext_tm=267966.9656045385,
        csca_tm=267866.3071677355,
    ),
    SpheroidSmuthiOracle(
        name="rotated",
        angles=(0.3, 0.6, 0.2),
        cext_te=232871.19236045482,
        csca_te=232784.94658405948,
        cext_tm=274472.1963556153,
        csca_tm=274363.5221125227,
    ),
)

_RUN_CACHE: dict[tuple[str, Literal["TE", "TM"]], SimulationResult] = {}


def _run_case(case: SpheroidSmuthiOracle, pol_name: Literal["TE", "TM"]) -> SimulationResult:
    key = (case.name, pol_name)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]

    src = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=pol_name,
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=10,
        source=src,
        # The oracle comes from a denser SMUTHI far-field reintegration. These
        # bins are only the public test's output sampling grid.
        polar_angles=np.linspace(0.0, np.pi, 61),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 121, endpoint=False),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    run = Simulation(
        cfg,
        particles=[
            Spheroid(
                position=(0.0, 0.0, 240.0),
                equatorial_radius=80.0,
                polar_radius=240.0,
                refractive_index=2.5 + 0.0j,
                euler_angles=case.angles,
            )
        ],
    ).run()
    _RUN_CACHE[key] = run
    return run


def test_spheroid_cross_sections_match_smuthi_oracles() -> None:
    for case in _ORACLES:
        run_te = _run_case(case, "TE")
        run_tm = _run_case(case, "TM")
        assert run_te.cross_sections is not None
        assert run_tm.cross_sections is not None

        np.testing.assert_allclose(
            run_te.cross_sections["C_ext"], case.cext_te, rtol=2e-5, atol=0.0
        )
        np.testing.assert_allclose(
            run_tm.cross_sections["C_ext"], case.cext_tm, rtol=2e-5, atol=0.0
        )
        np.testing.assert_allclose(
            run_te.cross_sections["C_sca"], case.csca_te, rtol=1e-3, atol=0.0
        )
        np.testing.assert_allclose(
            run_tm.cross_sections["C_sca"], case.csca_tm, rtol=1e-3, atol=0.0
        )
