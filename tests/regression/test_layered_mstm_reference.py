"""Layered-sphere regression against fixed MSTM-v4 oracle values.

MSTM-v4 text outputs print far-field/cross-section aggregates with fixed-format
precision (typically 5 significant digits for these blocks). We therefore store
those values exactly as printed by MSTM, while keeping full parsed precision for
near-field complex-field-derived quantities.

The three oracle configurations below are deliberately more challenging:
- each has at least one layer with `n_real > 2`,
- each mixes absorbing/non-absorbing layers,
- layer stacks are non-monotonic (alternating index trend across shells).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest

from pyceles.core.particles import LayeredSphere
from pyceles.core.sources import PlaneWave
from pyceles.io import far_field_intensity
from pyceles.postprocessing.nearfield import compute_near_field
from pyceles.simulation import Simulation, SimulationConfig

pytestmark = [pytest.mark.reference, pytest.mark.slow]


@dataclass(frozen=True)
class LayeredMSTMOracle:
    name: str
    radii: tuple[float, ...]
    indices: tuple[complex, ...]
    qext_mstm: float
    qsca_mstm: float
    s11_fwd_center_mstm_raw: float
    s11_bwd_center_mstm_raw: float
    near_points: tuple[tuple[float, float, float], ...]
    near_erms_unpol_mstm: tuple[float, ...]


_ORACLE: tuple[LayeredMSTMOracle, ...] = (
    LayeredMSTMOracle(
        name="L2",
        radii=(55.0, 120.0),
        indices=(1.38 + 0.01j, 2.2 + 0j),
        qext_mstm=4.5435e00,
        qsca_mstm=4.5354e00,
        s11_fwd_center_mstm_raw=1.6659e02,
        s11_bwd_center_mstm_raw=1.4465e01,
        near_points=(
            (-10.000023384349968, -10.000023384349968, 10.000023384349968),
            (-30.000070153049908, -10.000023384349968, 10.000023384349968),
            (-90.00371186789774, -10.000023384349968, 10.000023384349968),
            (-109.99675581910162, -10.000023384349968, 10.000023384349968),
            (-150.00035076524952, -10.000023384349968, 10.000023384349968),
        ),
        near_erms_unpol_mstm=(
            0.9800535748466713,
            1.0804485715622865,
            1.2374491799316285,
            1.1064771232619315,
            1.2680342863968623,
        ),
    ),
    LayeredMSTMOracle(
        name="L3",
        radii=(40.0, 80.0, 120.0),
        indices=(2.05 + 0j, 1.62 + 0.012j, 2.0 + 0j),
        qext_mstm=2.2066e00,
        qsca_mstm=2.1806e00,
        s11_fwd_center_mstm_raw=9.6928e01,
        s11_bwd_center_mstm_raw=4.2009e-01,
        near_points=(
            (-10.000023384349968, -10.000023384349968, 10.000023384349968),
            (-70.00016369044977, -10.000023384349968, 10.000023384349968),
            (-90.00371186789774, -10.000023384349968, 10.000023384349968),
            (-109.99675581910162, -10.000023384349968, 10.000023384349968),
            (-150.00035076524952, -10.000023384349968, 10.000023384349968),
        ),
        near_erms_unpol_mstm=(
            0.9646786051672822,
            1.1043522632883769,
            0.9658848952267346,
            0.900733413267383,
            1.1798904234737224,
        ),
    ),
    LayeredMSTMOracle(
        name="L4",
        radii=(30.0, 60.0, 90.0, 120.0),
        indices=(2.02 + 0j, 1.68 + 0j, 1.74 + 0j, 1.28 + 0.002j),
        qext_mstm=6.3885e-01,
        qsca_mstm=6.3347e-01,
        s11_fwd_center_mstm_raw=2.1175e01,
        s11_bwd_center_mstm_raw=4.1726e00,
        near_points=(
            (-10.000023384349968, -10.000023384349968, 10.000023384349968),
            (-50.00011692174984, -10.000023384349968, 10.000023384349968),
            (-70.00016369044977, -10.000023384349968, 10.000023384349968),
            (-109.99675581910162, -10.000023384349968, 10.000023384349968),
            (-150.00035076524952, -10.000023384349968, 10.000023384349968),
        ),
        near_erms_unpol_mstm=(
            0.9593048333744598,
            1.0041901604673789,
            0.8908507673148909,
            0.9900704460842673,
            1.089891970737754,
        ),
    ),
)

_RUN_CACHE: dict[tuple[str, Literal["TE", "TM"]], Any] = {}
_CROSS_SECTION_RUN_CACHE: dict[tuple[str, Literal["TE", "TM"]], Any] = {}


def _run_case(
    case: LayeredMSTMOracle,
    pol_name: Literal["TE", "TM"],
    *,
    polar_count: int = 361,
    azimuthal_count: int = 361,
    cache: dict[tuple[str, Literal["TE", "TM"]], Any] | None = None,
) -> Any:
    cache_obj = _RUN_CACHE if cache is None else cache
    key = (case.name, pol_name)
    if key in cache_obj:
        return cache_obj[key]

    src = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=pol_name,
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=10,
        polar_angles=np.linspace(0.0, np.pi, polar_count),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, azimuthal_count, endpoint=False),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    run = Simulation(
        cfg,
        particles=[
            LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=case.radii,
                layer_refractive_indices=case.indices,
            )
        ],
    ).run(src)
    cache_obj[key] = run
    return run


def _run_cross_section_case(case: LayeredMSTMOracle, pol_name: Literal["TE", "TM"]) -> Any:
    # Cross sections are much less sensitive to the far-field sampling density
    # than the on-axis S11 ratio used below, so keep a cheaper dedicated cache.
    return _run_case(
        case,
        pol_name,
        polar_count=141,
        azimuthal_count=141,
        cache=_CROSS_SECTION_RUN_CACHE,
    )


def _unpolarized_farfield_map(
    case: LayeredMSTMOracle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_te = _run_case(case, "TE")
    run_tm = _run_case(case, "TM")
    i_te = np.asarray(far_field_intensity(run_te.farfield.scattered), dtype=float)
    i_tm = np.asarray(far_field_intensity(run_tm.farfield.scattered), dtype=float)
    i_un = 0.5 * (i_te + i_tm)
    kx = np.asarray(run_te.farfield.scattered.kx, dtype=float)
    ky = np.asarray(run_te.farfield.scattered.ky, dtype=float)
    kz = np.asarray(run_te.farfield.scattered.kz, dtype=float)
    return i_un, kx, ky, kz


def _on_axis_s11(
    i_un: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray
) -> tuple[float, float]:
    d2 = kx**2 + ky**2
    fi, fj = np.unravel_index(int(np.argmin(np.where(kz >= 0.0, d2, np.inf))), d2.shape)
    bi, bj = np.unravel_index(int(np.argmin(np.where(kz <= 0.0, d2, np.inf))), d2.shape)
    return float(i_un[fi, fj]), float(i_un[bi, bj])


def _unpolarized_near_erms(case: LayeredMSTMOracle) -> np.ndarray:
    points = np.asarray(case.near_points, dtype=float)
    run_te = _run_case(case, "TE")
    run_tm = _run_case(case, "TM")
    e_te = np.asarray(
        compute_near_field(run_te, points=points, show_progress=False).E_total,
        dtype=np.complex128,
    )
    e_tm = np.asarray(
        compute_near_field(run_tm, points=points, show_progress=False).E_total,
        dtype=np.complex128,
    )
    return np.asarray(
        np.sqrt(0.5 * (np.sum(np.abs(e_te) ** 2, axis=-1) + np.sum(np.abs(e_tm) ** 2, axis=-1)))
    )


def test_layered_spheres_match_mstm_oracles_for_cross_sections():
    for case in _ORACLE:
        run_te = _run_cross_section_case(case, "TE")
        run_tm = _run_cross_section_case(case, "TM")
        assert run_te.cross_sections is not None
        assert run_tm.cross_sections is not None

        area = np.pi * float(case.radii[-1]) ** 2
        qext_py = 0.5 * (
            float(run_te.cross_sections.extinction / area)
            + float(run_tm.cross_sections.extinction / area)
        )
        qsca_py = 0.5 * (
            float(run_te.cross_sections.scattering / area)
            + float(run_tm.cross_sections.scattering / area)
        )

        np.testing.assert_allclose(qext_py, case.qext_mstm, rtol=1e-4, atol=0.0)
        np.testing.assert_allclose(qsca_py, case.qsca_mstm, rtol=1e-4, atol=0.0)


def test_layered_spheres_match_mstm_on_axis_s11_ratio():
    for case in _ORACLE:
        i_un, kx, ky, kz = _unpolarized_farfield_map(case)
        s11_fwd_py, s11_bwd_py = _on_axis_s11(i_un, kx, ky, kz)

        py_ratio = float(s11_fwd_py / s11_bwd_py)
        mstm_ratio = float(case.s11_fwd_center_mstm_raw / case.s11_bwd_center_mstm_raw)
        np.testing.assert_allclose(py_ratio, mstm_ratio, rtol=1e-4, atol=0.0)


def test_layered_spheres_match_mstm_off_axis_nearfield_unpolarized_erms():
    for case in _ORACLE:
        py_erms = _unpolarized_near_erms(case)
        mstm_erms = np.asarray(case.near_erms_unpol_mstm, dtype=float)
        np.testing.assert_allclose(py_erms, mstm_erms, rtol=1e-4, atol=0.0)
