"""Local regression against SMUTHI-generated dipole reference values.

This test intentionally hard-codes a small set of reference values extracted
from a local SMUTHI diagnostic run so routine regression checks do not require
re-running SMUTHI.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.particles import spheres_from_arrays

pytestmark = pytest.mark.reference


def _build_geometry(shift_z: float) -> list[pcl.core.Particle]:
    positions = np.array(
        [
            [-360.0, 0.0, -120.0],
            [-80.0, 0.0, 100.0],
            [180.0, 0.0, -80.0],
            [420.0, 0.0, 140.0],
        ],
        dtype=float,
    )
    positions[:, 2] += float(shift_z)
    radii = np.array([110.0, 90.0, 120.0, 80.0], dtype=float)
    n_particle = np.array([1.5 + 0.0j, 2.5 + 0.0j, 1.5 + 0.1j, 2.5 + 0.2j], dtype=np.complex128)
    return spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )


def _probe_points(shift_z: float) -> np.ndarray:
    return np.array(
        [
            [600.0, 0.0, shift_z + 320.0],
            [0.0, 0.0, shift_z + 540.0],
            [-500.0, 0.0, shift_z + 220.0],
        ],
        dtype=float,
    )


def _intensity(pwp_te: dict[str, np.ndarray], pwp_tm: dict[str, np.ndarray]) -> np.ndarray:
    return cast(
        np.ndarray,
        np.abs(np.asarray(pwp_te["coeff"])) ** 2 + np.abs(np.asarray(pwp_tm["coeff"])) ** 2,
    )


def _run_case(
    *,
    source: pcl.DipoleSource | pcl.DipoleCollection,
    particles: list[pcl.core.Particle],
    beta: np.ndarray,
    alpha: np.ndarray,
    probes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sim = pcl.Simulation(
        pcl.SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=2,
            source=source,
            polar_angles=beta,
            azimuthal_angles=alpha,
            solver_method="gmres",
            solver_rtol=1e-6,
            verbose=False,
        ),
        particles=particles,
    )
    run = sim.run()
    nf = pcl.compute_near_field(run, points=probes, channel="mixed", show_progress=False)
    if run.farfield.initial_te is None or run.farfield.initial_tm is None:
        raise AssertionError("Dipole regression expects initial far-field channels.")
    if run.farfield.total_te is None or run.farfield.total_tm is None:
        raise AssertionError("Dipole regression expects total far-field channels.")
    return (
        _intensity(run.farfield.initial_te, run.farfield.initial_tm),
        _intensity(run.farfield.scattered_te, run.farfield.scattered_tm),
        _intensity(run.farfield.total_te, run.farfield.total_tm),
        np.asarray(nf.E_total),
        float(source.dissipated_power_homogeneous_background()),
    )


def _assert_reference_samples(
    I_initial: np.ndarray,
    I_scattered: np.ndarray,
    I_total: np.ndarray,
    E_total: np.ndarray,
    P0: float,
    *,
    expected_samples: dict[str, dict[str, float]],
    expected_probe_E: list[list[tuple[float, float]]],
    expected_P0: float,
) -> None:
    for key, values in expected_samples.items():
        i_str, j_str = key.split(",")
        i = int(i_str)
        j = int(j_str)
        np.testing.assert_allclose(I_initial[i, j], values["initial"], rtol=1e-3, atol=0.0)
        np.testing.assert_allclose(I_scattered[i, j], values["scattered"], rtol=1e-3, atol=0.0)
        np.testing.assert_allclose(I_total[i, j], values["total"], rtol=1e-3, atol=0.0)

    E_ref = np.asarray(
        [[complex(re, im) for (re, im) in row] for row in expected_probe_E], dtype=np.complex128
    )
    np.testing.assert_allclose(np.asarray(E_total), E_ref, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(P0, expected_P0, rtol=1e-12, atol=0.0)


def test_dipole_single_smuthi_reference() -> None:
    shift_z = 800.0
    particles = _build_geometry(shift_z)
    probes = _probe_points(shift_z)
    beta = np.linspace(0.0, np.pi, 121, endpoint=False)
    alpha = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    source = pcl.DipoleSource(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        position=(0.0, 0.0, shift_z + 50.0),
        dipole_moment=(1.0 + 0j, 0.2 + 0.1j, -0.1 + 0.05j),
    )

    I_initial, I_scattered, I_total, E_total, P0 = _run_case(
        source=source,
        particles=particles,
        beta=beta,
        alpha=alpha,
        probes=probes,
    )

    _assert_reference_samples(
        I_initial,
        I_scattered,
        I_total,
        E_total,
        P0,
        expected_samples={
            "0,0": {
                "initial": 3.743802306289224e-16,
                "scattered": 1.0033529355800324e-15,
                "total": 2.332783978074537e-15,
            },
            "17,23": {
                "initial": 3.0590961135848804e-16,
                "scattered": 1.0922035269124379e-15,
                "total": 1.7187523004328296e-15,
            },
            "53,47": {
                "initial": 3.70195668014311e-16,
                "scattered": 8.830019583218075e-16,
                "total": 2.012077821453485e-15,
            },
        },
        expected_probe_E=[
            [
                (-4.888611130616936e-09, 6.5573337559727355e-09),
                (-8.334678409318016e-10, 2.0899983889067804e-09),
                (2.195426376806934e-08, -1.4073825020763899e-08),
            ],
            [
                (5.561860086105286e-08, 9.276088836645969e-10),
                (2.021569780452191e-09, 2.2400377276095085e-09),
                (-2.620541401888479e-09, 2.633570457855451e-09),
            ],
            [
                (-1.7776089345126607e-09, -1.9385394322612484e-08),
                (1.3105715973227465e-09, 5.535867748020203e-09),
                (-9.161934470738314e-09, 1.5278728199174993e-08),
            ],
        ],
        expected_P0=4.800279612641185e-10,
    )


def test_dipole_collection_smuthi_reference() -> None:
    shift_z = 800.0
    particles = _build_geometry(shift_z)
    probes = _probe_points(shift_z)
    beta = np.linspace(0.0, np.pi, 121, endpoint=False)
    alpha = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    source = pcl.DipoleCollection(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        positions=np.array(
            [[0.0, 0.0, shift_z + 50.0], [220.0, 0.0, shift_z + 150.0]], dtype=float
        ),
        dipole_moments=np.array(
            [[1.0 + 0j, 0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.7 + 0.2j, 0.1 + 0j]],
            dtype=np.complex128,
        ),
    )

    I_initial, I_scattered, I_total, E_total, P0 = _run_case(
        source=source,
        particles=particles,
        beta=beta,
        alpha=alpha,
        probes=probes,
    )

    _assert_reference_samples(
        I_initial,
        I_scattered,
        I_total,
        E_total,
        P0,
        expected_samples={
            "0,0": {
                "initial": 5.4552547891643e-16,
                "scattered": 9.921128270835695e-16,
                "total": 2.426035872499551e-15,
            },
            "17,23": {
                "initial": 4.718613629707642e-16,
                "scattered": 1.0282839445618385e-15,
                "total": 1.823792769087663e-15,
            },
            "53,47": {
                "initial": 4.65679766713742e-16,
                "scattered": 7.582243822529551e-16,
                "total": 2.000996824327449e-15,
            },
        },
        expected_probe_E=[
            [
                (-5.811984644550756e-09, 5.625316688893326e-09),
                (1.535307469083438e-08, -1.23842610971682e-08),
                (2.3752306403016733e-08, -1.3389334918786219e-08),
            ],
            [
                (5.566630069068345e-08, 2.942390187720629e-09),
                (9.950796247848799e-09, -1.4378873739168103e-08),
                (-3.4475998281786317e-09, 1.1063463693740614e-09),
            ],
            [
                (-4.3229331531263286e-10, -1.7928137445066098e-08),
                (-9.92521649685439e-09, 7.328094715481123e-09),
                (-7.868105260932773e-09, 1.669955084990348e-08),
            ],
        ],
        expected_P0=6.957581744439929e-10,
    )
