from __future__ import annotations

"""Sphere regression against fixed miepython far-field and near-field oracles.

These tests intentionally hard-code a compact set of reference values derived
from local `miepython` runs so routine regression checks do not depend on the
external package at test time.

Convention note
---------------
The reference cases use a normal-incidence ``TM`` plane wave in `pyceles`.
For this incidence, `pyceles`' source convention corresponds to an incident
electric field polarized along ``+x``, which matches the fixed plane-wave
convention used by `miepython.field`.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

import pyceles as pcl
from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere
from pyceles.postprocessing.farfield import scattering_cross_section
from pyceles.simulation import Simulation, SimulationConfig, SimulationResult


@dataclass(frozen=True)
class SphereMiepythonFarfieldOracle:
    name: str
    radius: float
    refractive_index: complex
    n_medium: float
    wavelength: float
    lmax: int
    dcs_alpha0: tuple[float, ...]
    dcs_alpha90: tuple[float, ...]


@dataclass(frozen=True)
class SphereMiepythonNearfieldOracle:
    name: str
    radius: float
    refractive_index: complex
    n_medium: float
    wavelength: float
    lmax: int
    points: tuple[tuple[float, float, float], ...]
    e_total: tuple[tuple[complex, complex, complex], ...]
    h_total: tuple[tuple[complex, complex, complex], ...]


_BETA_SAMPLES = np.array([0.3, 0.9, 1.4, 2.2], dtype=float)
_ALPHA_SAMPLES = np.array([0.0, 0.5 * np.pi], dtype=float)

_FARFIELD_ORACLES: tuple[SphereMiepythonFarfieldOracle, ...] = (
    SphereMiepythonFarfieldOracle(
        name="dielectric",
        radius=50.0,
        refractive_index=1.45 + 0.01j,
        n_medium=1.0,
        wavelength=550.0,
        lmax=8,
        dcs_alpha0=(
            20.097609442135237,
            8.307063077513703,
            0.6607322496079077,
            5.8546496098537055,
        ),
        dcs_alpha90=(
            21.965597786151868,
            20.92014318480737,
            19.571441259842032,
            17.468848530864527,
        ),
    ),
    SphereMiepythonFarfieldOracle(
        name="absorbing_host",
        radius=70.0,
        refractive_index=2.0 + 0.15j,
        n_medium=1.33,
        wavelength=700.0,
        lmax=8,
        dcs_alpha0=(
            254.36189738807067,
            103.30907283033024,
            9.253000571411889,
            52.27577970409264,
        ),
        dcs_alpha90=(
            276.8921443766452,
            248.90463657589783,
            214.75723969767537,
            166.03024916582993,
        ),
    ),
    SphereMiepythonFarfieldOracle(
        name="metallic",
        radius=40.0,
        refractive_index=0.25 + 2.5j,
        n_medium=1.5,
        wavelength=650.0,
        lmax=8,
        dcs_alpha0=(
            2575.855956781428,
            1077.4487877058525,
            79.201823953243,
            964.953361558878,
        ),
        dcs_alpha90=(
            2824.6275656179323,
            2810.0066810202065,
            2790.644283049681,
            2759.2143442722345,
        ),
    ),
)

_NEARFIELD_ORACLES: tuple[SphereMiepythonNearfieldOracle, ...] = (
    SphereMiepythonNearfieldOracle(
        name="dielectric",
        radius=50.0,
        refractive_index=1.45 + 0.01j,
        n_medium=1.0,
        wavelength=550.0,
        lmax=8,
        points=((80.0, 0.0, 0.0), (0.0, 0.0, 80.0)),
        e_total=(
            (
                1.1741223176133293 + 0.03996595930586861j,
                0.0 + 0.0j,
                0.00071753349605823 - 0.016251889179238043j,
            ),
            (
                0.5557232098428037 + 0.8078975826379835j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ),
        ),
        h_total=(
            (
                0.0 + 0.0j,
                1.0044627012261083 + 0.0013831870860966566j,
                0.0 + 0.0j,
            ),
            (
                0.0 + 0.0j,
                0.5831357972398661 + 0.871072079863206j,
                0.0 + 0.0j,
            ),
        ),
    ),
    SphereMiepythonNearfieldOracle(
        name="metallic_n1",
        radius=40.0,
        refractive_index=0.25 + 2.5j,
        n_medium=1.0,
        wavelength=650.0,
        lmax=8,
        points=((75.0, 0.0, 0.0), (0.0, 0.0, 75.0)),
        e_total=(
            (
                1.698342729685021 + 0.22915110524023963j,
                0.0 + 0.0j,
                0.0022778014231423127 - 0.01917521472707826j,
            ),
            (
                0.4946014262913753 + 0.6473773683416005j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ),
        ),
        h_total=(
            (
                0.0 + 0.0j,
                1.0132969441183233 - 0.000827058868363369j,
                0.0 + 0.0j,
            ),
            (
                0.0 + 0.0j,
                0.6599749645849201 + 0.9131308926354853j,
                0.0 + 0.0j,
            ),
        ),
    ),
)

_RUN_CACHE: dict[tuple[str, bool], SimulationResult] = {}


def _run_case(
    *,
    name: str,
    radius: float,
    refractive_index: complex,
    n_medium: float,
    wavelength: float,
    lmax: int,
    polar_angles: np.ndarray | None = None,
    azimuthal_angles: np.ndarray | None = None,
    include_farfield: bool,
) -> SimulationResult:
    key = (name, include_farfield)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]

    source = PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium + 0j,
        polarization="TM",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium + 0j,
        lmax=lmax,
        source=source,
        # Keep this oracle pinned to the historical explicit LUT spacing so the
        # regression only tracks sphere-vs-miepython agreement, not default
        # radial-LUT policy changes.
        radial_lut_dr=1.0,
        polar_angles=(
            np.asarray(polar_angles, dtype=float)
            if polar_angles is not None
            else np.linspace(0.0, np.pi, 21)
        ),
        azimuthal_angles=(
            np.asarray(azimuthal_angles, dtype=float)
            if azimuthal_angles is not None
            else np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False)
        ),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    run = Simulation(
        cfg,
        particles=[
            Sphere(
                position=(0.0, 0.0, 0.0),
                radius=radius,
                refractive_index=refractive_index,
            )
        ],
    ).run(include_farfield=include_farfield)
    _RUN_CACHE[key] = run
    return run


def test_sphere_differential_scattering_matches_miepython_oracles() -> None:
    for case in _FARFIELD_ORACLES:
        run = _run_case(
            name=case.name,
            radius=case.radius,
            refractive_index=case.refractive_index,
            n_medium=case.n_medium,
            wavelength=case.wavelength,
            lmax=case.lmax,
            polar_angles=_BETA_SAMPLES,
            azimuthal_angles=_ALPHA_SAMPLES,
            include_farfield=True,
        )
        if run.config.source is None:
            raise AssertionError(
                "Sphere miepython regression expects an explicit PlaneWave source."
            )
        dcs = scattering_cross_section(
            cast(PlaneWave, run.config.source),
            run.farfield.scattered_te,
            run.farfield.scattered_tm,
            k0=run.k0,
            n_medium=run.config.n_medium,
        )
        np.testing.assert_allclose(
            dcs["total"][0],
            np.asarray(case.dcs_alpha0, dtype=float),
            rtol=1e-7,
            atol=0.0,
        )
        np.testing.assert_allclose(
            dcs["total"][1],
            np.asarray(case.dcs_alpha90, dtype=float),
            rtol=1e-7,
            atol=0.0,
        )


def test_sphere_total_near_field_matches_miepython_oracles() -> None:
    for case in _NEARFIELD_ORACLES:
        points = np.asarray(case.points, dtype=float)
        run = _run_case(
            name=case.name,
            radius=case.radius,
            refractive_index=case.refractive_index,
            n_medium=case.n_medium,
            wavelength=case.wavelength,
            lmax=case.lmax,
            include_farfield=False,
        )
        nf = pcl.compute_near_field(run, points=points, channel="mixed", show_progress=False)

        np.testing.assert_allclose(
            np.asarray(nf.E_total, dtype=np.complex128),
            np.asarray(case.e_total, dtype=np.complex128),
            rtol=1e-5,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(nf.H_total, dtype=np.complex128),
            np.asarray(case.h_total, dtype=np.complex128),
            rtol=1e-6,
            atol=1e-12,
        )
