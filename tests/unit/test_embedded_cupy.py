"""Embedded finite sources share device operators and native local responses."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pyceles._optional import asnumpy
from pyceles.core.indexing import n_modes
from pyceles.core.operators.coupling_pairwise_cupy import CuPyPairwiseCouplingOperator
from pyceles.core.particles import LayeredSphere, PECSphere, Sphere, TMatrixParticle
from pyceles.core.sources import DipoleCollection, DipoleSource, Source
from pyceles.core.svwf import expansion_fields
from pyceles.core.translation import RadialLUT, translation_ab5_table, translation_block
from pyceles.postprocessing.dipole_metrics import (
    compute_dipole_reaction_field,
    compute_dipole_source_work,
)
from pyceles.postprocessing.nearfield import compute_electric_field, compute_near_field
from pyceles.simulation import ResultRetention, Simulation, SimulationConfig

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", params=["complex64", "complex128"])
def embedded_runs(request, cupy_runtime):
    dtype = request.param
    host = 1.33 + 0.08j
    matrix = np.eye(n_modes(2), dtype=complex) * (0.01 + 0.02j)
    matrix[0, 7], matrix[5, 2] = 0.03 - 0.01j, -0.02j
    particles = (
        LayeredSphere((0, 0, 0), (45, 70), (1.5, 1.9 + 0.12j)),
        Sphere((250, 0, 0), 60, 1.7),
        PECSphere((500, 0, 0), 40),
        TMatrixParticle((750, 0, 0), 30, 2, matrix),
    )
    config = SimulationConfig(
        wavelength=550,
        n_medium=host,
        lmax=2,
        radial_lut_dr=0.25,
        compute_dtype=dtype,
        accum_dtype="complex128",
        solver_method="gmres",
        solver_rtol=2e-6 if dtype == "complex64" else 1e-11,
        verbose=False,
        polar_angles=np.linspace(0, np.pi, 7),
        azimuthal_angles=np.arange(8) * np.pi / 4,
    )
    sources: dict[str, Source] = {
        "core": DipoleSource(
            550, medium_n=host, position=(10, 0, 0), dipole_moment=(1, 0.3j, -0.2)
        ),
        "mixed": DipoleCollection(
            550,
            medium_n=host,
            positions=np.array([[10, 0, 0], [260, 0, 0], [55, 0, 0], [900, 0, 0], [500, 0, 0]]),
            dipole_moments=np.array(
                [[1, 0.3j, -0.2], [0.1j, 0.5, 0], [0.2, -0.1j, 0.3], [-0.1j, 0.6, 0.2j], [0, 0, 0]]
            ),
            radial_lut_dr=0.25,
        ),
    }
    cpu = Simulation(config, particles=particles)
    gpu = Simulation(replace(config, operator_backend="cupy"), particles=particles)
    reference = cpu.run_sources(sources, retention=ResultRetention.minimal())
    actual = gpu.run_sources(sources, retention=ResultRetention.minimal())
    return cpu, gpu, reference, actual


def test_embedded_source_blocks_fields_and_work_match_numpy(embedded_runs):
    _, _, reference, actual = embedded_runs
    tolerance = 1e-5 if actual.config.compute_dtype == "complex64" else 2e-10
    points = np.array(
        [
            [0, 0, 0],
            [20, 5, 3],
            [55, 0, 0],
            [60, 5, 2],
            [250, 0, 0],
            [500, 0, 0],
            [100, 60, 30],
            [1000, 0, 0],
            [10, 0, 0],
        ]
    )
    for name in reference.channels:
        a, b = actual[name], reference[name]
        assert a.rhs is None and a.initial_coeffs is None
        assert np.linalg.norm(a.coeffs - b.coeffs) / np.linalg.norm(b.coeffs) < tolerance
        fields = compute_near_field(a, points=points, show_progress=False)
        expected = compute_near_field(b, points=points, show_progress=False)
        for component in (
            "E_initial",
            "H_initial",
            "E_scattered",
            "H_scattered",
            "E_internal",
            "H_internal",
            "E_total",
            "H_total",
        ):
            value, target = getattr(fields, component), getattr(expected, component)
            assert value.dtype == np.complex128
            np.testing.assert_array_equal(np.isnan(value), np.isnan(target))
            valid = np.isfinite(target)
            if np.linalg.norm(target[valid]):
                assert (
                    np.linalg.norm(value[valid] - target[valid]) / np.linalg.norm(target[valid])
                    < tolerance
                )
            else:
                np.testing.assert_array_equal(value[valid], target[valid])
        electric = compute_electric_field(a, points=points, show_progress=False)
        np.testing.assert_array_equal(electric.E_total, fields.E_total)
        reaction = compute_dipole_reaction_field(a)
        expected_reaction = compute_dipole_reaction_field(b)
        scales = np.linalg.norm(expected_reaction, axis=1)
        active = scales != 0
        # A weak Cartesian component may cancel; each emitter's vector field
        # supplies the physical comparison scale, including the host emitter.
        assert np.all(
            np.linalg.norm(reaction[active] - expected_reaction[active], axis=1) / scales[active]
            < tolerance
        )
        np.testing.assert_array_equal(reaction[~active], expected_reaction[~active])
        for polarization in ("coeff_te", "coeff_tm"):
            value, target = (
                getattr(a.farfield.total, polarization),
                getattr(b.farfield.total, polarization),
            )
            assert np.linalg.norm(value - target) / np.linalg.norm(target) < tolerance
    work = compute_dipole_source_work(actual["core"])
    expected_work = compute_dipole_source_work(reference["core"])
    np.testing.assert_allclose(work.power_total, expected_work.power_total, rtol=tolerance, atol=0)
    with pytest.raises(ValueError, match="nonabsorbing local material"):
        compute_dipole_source_work(actual["mixed"])


@pytest.mark.parametrize("method", ["direct", "bicgstab", "lsqr"])
def test_embedded_gpu_solvers_accept_native_emission(embedded_runs, method):
    _, gpu, reference, _ = embedded_runs
    config = replace(gpu.config, solver_method=method)
    run = Simulation(config, particles=gpu.particles).run(
        reference["core"].source, include_farfield=False
    )
    expected = reference["core"].coeffs
    tolerance = 1e-5 if config.compute_dtype == "complex64" else 2e-10
    assert run.solver_result.info == 0
    assert np.linalg.norm(run.coeffs - expected) / np.linalg.norm(expected) < tolerance


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_selected_reception_matches_independent_blocks_and_preserves_inputs(cupy_runtime, dtype):
    cupy, _ = cupy_runtime
    # Uneven target tail, repeats, nonsequential global indices, and block RHSs.
    positions = np.arange(39).reshape(13, 3) * np.array([31, -17, 23])
    k, lmax = 0.02 + 0.001j, 2
    rng = np.random.default_rng(713)
    x = np.asarray(
        rng.normal(size=(13, n_modes(lmax), 3)) + 1j * rng.normal(size=(13, n_modes(lmax), 3)),
        dtype=np.complex128,
    )
    targets = np.resize(np.array([12, 0, 8, 1], dtype=np.int32), 37)
    lut = RadialLUT(lmax=lmax, k=k, r_max=2000, dr=0.25, dtype=dtype)
    operator = CuPyPairwiseCouplingOperator(lmax, k, positions, None, lut, np.dtype(dtype))
    device_x = cupy.asarray(x.reshape(-1, 3))
    expected = np.zeros((len(targets), n_modes(lmax), 3), dtype=np.complex128)
    ab5 = translation_ab5_table(lmax, dtype=dtype)
    for row, target in enumerate(targets):
        for source in range(len(positions)):
            if source != target:
                block = translation_block(
                    lmax, k, positions[target] - positions[source], ab5=ab5, radial_lut=lut
                )
                expected[row] += block.astype(np.complex128) @ x[source].astype(dtype).astype(
                    np.complex128
                )
    received = operator.apply_rows(device_x, targets, accum_dtype=np.dtype(np.complex128))
    original = asnumpy(received).copy()
    operator.apply_adjoint(device_x)
    operator.apply(device_x)
    np.testing.assert_array_equal(asnumpy(device_x), x.reshape(-1, 3))
    np.testing.assert_array_equal(asnumpy(received), original)
    tolerance = 4e-6 if dtype == "complex64" else 3e-13
    np.testing.assert_allclose(
        original.reshape(expected.shape), expected, rtol=tolerance, atol=1e-11
    )


def test_cupy_expansion_crosses_batches_with_promoted_accumulation(cupy_runtime, monkeypatch):
    import pyceles.core.svwf as svwf

    monkeypatch.setattr(svwf, "_cupy_svwf_pair_batch_size", lambda **kwargs: 2)
    rng = np.random.default_rng(24)
    regular = rng.normal(size=n_modes(3)) + 1j * rng.normal(size=n_modes(3))
    outgoing = regular * (0.2 + 0.1j)
    points = np.array([[1, 2, 3], [30, 0, 0], [0, 50, 0], [4, -6, 90], [500, 0, 0]])
    options: dict[str, Any] = dict(
        centre=np.zeros(3),
        regular=regular,
        outgoing=outgoing,
        k=0.02 + 0.001j,
        n=1.33 + 0.08j,
        lmax=3,
        compute_dtype="complex64",
        accum_dtype="complex128",
    )
    reference = expansion_fields(points, **options)
    actual = expansion_fields(points, backend="cupy", **options)
    for value, target in zip(actual, reference, strict=True):
        assert value.dtype == np.complex128
        np.testing.assert_allclose(value, target, rtol=3e-6, atol=1e-8)


def test_high_order_cupy_reaction_combines_radial_factors_before_cast(cupy_runtime):
    source = DipoleSource(550, position=(65, 0, 0))
    particle = LayeredSphere((0, 0, 0), (50, 80), (1.5 + 0.1j, 1.7))
    config = SimulationConfig(
        wavelength=550,
        lmax=2,
        solver_method="gmres",
        compute_dtype="complex64",
        accum_dtype="complex128",
        solver_rtol=2e-6,
        verbose=False,
    )
    reference = Simulation(config, particles=(particle,)).run(source, include_farfield=False)
    actual = Simulation(replace(config, operator_backend="cupy"), particles=(particle,)).run(
        source, include_farfield=False
    )
    expected = compute_dipole_source_work(reference, reaction_lmax=32)
    work = compute_dipole_source_work(actual, reaction_lmax=32)
    assert np.isfinite(work.E_reaction_at_dipoles).all()
    np.testing.assert_allclose(work.power_ratio, expected.power_ratio, rtol=2e-6)
