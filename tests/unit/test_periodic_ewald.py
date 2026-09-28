from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

import pyceles as pcl
import pyceles.core.periodic.ewald as ewald_module
from pyceles.core.periodic import plane_wave_k_parallel
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    PeriodicEwaldConvergenceError,
    _same_plane_real_sum,
    _same_plane_reciprocal_sum,
    _self_correction,
    _shifted_real_sum,
    _shifted_reciprocal_sum,
    default_ewald_eta,
    ewald_structural_constant_2d,
    ewald_structural_sums_2d,
    ewald_structural_sums_2d_batch,
    resolve_ewald_eta,
    select_ewald_eta,
)
from pyceles.core.periodic.scalar import (
    real_integral_sequence,
    scaled_real_integral_sequence,
)
from pyceles.core.periodic.shells import (
    accumulate_lattice_shell_series,
    make_lattice_shell_control,
)
from pyceles.core.periodic.special import (
    shifted_delta_sequence,
    shifted_delta_sequence_batched,
    upper_incomplete_gamma_int_or_halfint,
)
from pyceles.core.periodic.structural import (
    apply_structural_sums_to_vector,
    blocks_from_structural_sums,
    direct_structural_sums_2d,
    translation_contraction_tensor,
)
from pyceles.core.translation import translation_ab5_table


def _lattice() -> pcl.RectangularLattice2D:
    return pcl.RectangularLattice2D(ax=430.0, ay=470.0)


def test_fixed_shell_ewald_is_quasiperiodic_across_lateral_seam() -> None:
    """Fixed Ewald shells must be independent of the chosen cell representative."""
    lattice = pcl.RectangularLattice2D(ax=3909.790257352941, ay=3909.790257352941)
    k = 2.0 * np.pi / (550.0 / 1.36)
    k_parallel = np.asarray([4.0e-4, -2.0e-4], dtype=float)
    positions = np.asarray([[0.0, 0.0, 0.0], [-3600.0, -67.0, -168.0]], dtype=float)
    periodic = pcl.PeriodicSpec(
        lattice=lattice,
        options=pcl.PeriodicOptions(
            method="ewald",
            shell_tolerance=1.0e-7,
            max_shells=64,
        ),
    )
    eta = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=k_parallel,
        positions=positions,
        lmax=1,
    )
    counts = ewald_module.resolve_ewald_shell_counts(
        periodic=periodic,
        k=k,
        k_parallel=k_parallel,
        positions=positions,
        lmax=1,
        eta=eta,
    )
    raw = positions[1] - positions[0]
    periods = np.asarray([lattice.ax, lattice.ay], dtype=float)
    shift = np.rint(raw[:2] / periods)
    wrapped = raw.copy()
    wrapped[:2] -= shift * periods
    phase = np.exp(-1j * np.dot(shift * periods, k_parallel))
    workspace = EwaldShellWorkspace(lattice, k, k_parallel, eta)

    def evaluate(displacement: np.ndarray) -> np.ndarray:
        values = ewald_structural_sums_2d_batch(
            lmax_struct=1,
            k=k,
            destinations=(-displacement)[None, :],
            source=np.zeros(3),
            lattice=lattice,
            k_parallel=k_parallel,
            eta=eta,
            real_shells=counts.real_shells,
            reciprocal_shells=counts.reciprocal_shells,
            shell_tolerance=1.0e-7,
            max_shells=64,
            workspace=workspace,
        )
        return cast(np.ndarray, values[0])

    np.testing.assert_allclose(
        evaluate(raw),
        phase * evaluate(wrapped),
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_scalar_ewald_is_quasiperiodic_across_lateral_seam() -> None:
    lattice = pcl.RectangularLattice2D(ax=3909.790257352941, ay=3909.790257352941)
    k = 2.0 * np.pi / (550.0 / 1.36)
    k_parallel = np.asarray([4.0e-4, -2.0e-4], dtype=float)
    eta = 1.8133492941951595e-3
    source = np.zeros(3, dtype=float)
    destination = np.asarray([3600.0, 67.0, 168.0], dtype=float)
    periods = np.asarray([lattice.ax, lattice.ay], dtype=float)
    shift = np.rint(destination[:2] / periods)
    wrapped_destination = destination.copy()
    wrapped_destination[:2] -= shift * periods
    phase = np.exp(1j * np.dot(shift * periods, k_parallel))

    raw = ewald_structural_constant_2d(
        0,
        0,
        k=k,
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        eta=eta,
        real_shells=0,
        reciprocal_shells=12,
        max_shells=64,
    )
    wrapped = ewald_structural_constant_2d(
        0,
        0,
        k=k,
        destination=wrapped_destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        eta=eta,
        real_shells=0,
        reciprocal_shells=12,
        max_shells=64,
    )

    assert raw == pytest.approx(phase * wrapped, rel=2.0e-11, abs=2.0e-11)


def test_scalar_same_plane_routing_ignores_lateral_coordinate_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large lateral coordinates must not alter a z-only routing decision."""
    routes: list[str] = []

    def shifted_reciprocal(*args: object, **kwargs: object) -> complex:
        del args, kwargs
        routes.append("shifted")
        return 0.0 + 0.0j

    def same_plane_reciprocal(*args: object, **kwargs: object) -> complex:
        del args, kwargs
        routes.append("same")
        return 0.0 + 0.0j

    monkeypatch.setattr(ewald_module, "_shifted_reciprocal_sum", shifted_reciprocal)
    monkeypatch.setattr(ewald_module, "_same_plane_reciprocal_sum", same_plane_reciprocal)
    monkeypatch.setattr(ewald_module, "_shifted_real_sum", lambda *args, **kwargs: 0.0 + 0.0j)
    monkeypatch.setattr(ewald_module, "_same_plane_real_sum", lambda *args, **kwargs: 0.0 + 0.0j)

    lattice = pcl.RectangularLattice2D(ax=430.0, ay=470.0)
    for lateral_origin in (0.0, 1.0e10):
        ewald_structural_constant_2d(
            0,
            0,
            k=1.0,
            destination=np.asarray([lateral_origin, 0.0, 1.0e-8]),
            source=np.asarray([lateral_origin, 0.0, 0.0]),
            lattice=lattice,
            k_parallel=np.zeros(2),
            eta=0.1,
            real_shells=0,
            reciprocal_shells=0,
        )

    assert routes == ["shifted", "shifted"]


def test_self_correction_uses_half_integer_origin_term() -> None:
    """The 3D origin correction uses Gamma(-1/2, x), not plain Gamma(0, x)."""
    k = 2.0 * np.pi / 550.0
    eta = 0.02
    x = -(k * k) / (4.0 * eta * eta)

    expected = upper_incomplete_gamma_int_or_halfint(-0.5, x) / (4.0 * np.pi)

    assert _self_correction(k, eta) == pytest.approx(expected, rel=1e-15, abs=1e-15)


def test_eta_probe_relative_difference_scales_large_finite_rows() -> None:
    """Finite large probe values must not become an inf/inf comparison."""
    lhs = np.full((1, 32), complex(1.0e308, 1.0e308), dtype=np.complex128)
    rhs = np.full((1, 32), complex(5.0e307, 5.0e307), dtype=np.complex128)
    with np.errstate(over="raise", invalid="raise"):
        relative_difference = ewald_module._eta_probe_relative_difference(lhs, rhs)
    assert relative_difference == pytest.approx(0.5, rel=1.0e-14)


@pytest.mark.reference
def test_off_plane_ewald_structural_constants_match_stabilized_references() -> None:
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([0.0012, -0.0007], dtype=float)
    destination = np.array([65.0, 32.0, 59.0], dtype=float)

    expected = {
        (0, 0): 0.11164659546263714 - 0.4923653691127005j,
        (1, -1): -0.3801412017227096 - 0.7240325221832068j,
        (2, 0): -0.29765008648181657 - 0.8869424347969169j,
        (4, 1): -4.267889014871487 + 8.891313748380995j,
    }

    for (degree, order), reference in expected.items():
        actual = ewald_structural_constant_2d(
            degree,
            order,
            k=k,
            destination=destination,
            source=np.zeros(3, dtype=float),
            lattice=_lattice(),
            k_parallel=k_parallel,
            eta=0.02,
            real_shells=12,
            reciprocal_shells=12,
        )
        assert actual == pytest.approx(reference, rel=1e-12, abs=1e-12)


@pytest.mark.reference
def test_off_plane_ewald_high_order_matches_reliable_direct_window() -> None:
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([0.0012, -0.0007], dtype=float)
    destination = np.array([65.0, 32.0, 59.0], dtype=float)
    lmax = 2

    ewald = ewald_structural_sums_2d(
        lmax=lmax,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=k_parallel,
        eta=0.02,
        real_shells=12,
        reciprocal_shells=12,
    )
    direct = direct_structural_sums_2d(
        lmax=lmax,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=k_parallel,
        window=16,
    )

    assert ewald[4, 1 + 4] == pytest.approx(direct[4, 1 + 4], rel=8e-5, abs=8e-4)


def test_ewald_structural_sums_are_eta_invariant_for_same_plane_case() -> None:
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([0.0012, -0.0007], dtype=float)
    destination = np.array([73.0, 28.0, 0.0], dtype=float)

    a = ewald_structural_constant_2d(
        2,
        0,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=k_parallel,
        eta=0.018,
        real_shells=16,
        reciprocal_shells=16,
    )
    b = ewald_structural_constant_2d(
        2,
        0,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=k_parallel,
        eta=0.024,
        real_shells=16,
        reciprocal_shells=16,
    )

    assert a == pytest.approx(b, rel=2e-9, abs=2e-9)


def test_batch_same_plane_rows_match_individual_evaluations() -> None:
    """Mixed same-plane/shifted batches must apply same-plane parity per point.

    Regular xz/yz diagnostic slices can contain one exact same-z row embedded in
    otherwise off-plane rows.  The batch implementation should match separate
    scalar-table evaluations for each point in that mixed batch.
    """
    k = 2.0 * np.pi / 550.0
    kp = np.array([0.0012, -0.0007], dtype=float)
    lattice = _lattice()
    source = np.zeros(3, dtype=float)
    destinations = np.array(
        [
            [73.0, 28.0, 0.0],
            [65.0, 32.0, 59.0],
            [-41.0, 109.0, 0.0],
        ],
        dtype=float,
    )
    batch = ewald_structural_sums_2d_batch(
        lmax_struct=2,
        k=k,
        destinations=destinations,
        source=source,
        lattice=lattice,
        k_parallel=kp,
        eta=0.02,
        real_shells=8,
        reciprocal_shells=8,
    )
    expected = np.stack(
        [
            ewald_structural_sums_2d(
                lmax=2,
                k=k,
                destination=dest,
                source=source,
                lattice=lattice,
                k_parallel=kp,
                eta=0.02,
                real_shells=8,
                reciprocal_shells=8,
            )
            for dest in destinations
        ],
        axis=0,
    )
    np.testing.assert_allclose(batch, expected, rtol=5e-12, atol=5e-12)


def test_periodic_structural_contraction_honors_complex64_compute_dtype() -> None:
    """c64/c128 contracts the completed c128 Ewald table in c64 arithmetic."""
    rng = np.random.default_rng(42)
    lmax = 1
    nm = 6
    order = 2 * lmax
    sums = rng.normal(size=(3, order + 1, 2 * order + 1)) + 1j * rng.normal(
        size=(3, order + 1, 2 * order + 1)
    )
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    tensor = translation_contraction_tensor(lmax=lmax, ab5=ab5, dtype=np.complex64)
    actual = blocks_from_structural_sums(
        lmax=lmax,
        structural_sums=np.asarray(sums, dtype=np.complex128),
        ab5=ab5,
        dtype=np.complex64,
        contraction_tensor=tensor,
    )
    expected = np.asarray(
        np.einsum(
            "dpm,ijpm->dij",
            np.asarray(sums, dtype=np.complex64),
            tensor,
            optimize=True,
        ),
        dtype=np.complex64,
    )
    assert actual.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(actual, expected)

    vector = rng.normal(size=nm) + 1j * rng.normal(size=nm)
    actual_vec = apply_structural_sums_to_vector(
        lmax=lmax,
        structural_sums=np.asarray(sums, dtype=np.complex128),
        ab5=ab5,
        vector=vector,
        dtype=np.complex64,
        contraction_tensor=tensor,
    )
    expected_vec = np.asarray(
        np.einsum(
            "dpm,ijpm,j->di",
            np.asarray(sums, dtype=np.complex64),
            tensor,
            np.asarray(vector, dtype=np.complex64),
            optimize=True,
        ),
        dtype=np.complex64,
    )
    assert actual_vec.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(actual_vec, expected_vec)


def test_batch_ewald_fails_fast_on_nonfinite_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_shifted_delta(
        order: int,
        gamma: np.ndarray,
        cz: np.ndarray,
        eta: float,
        **_kwargs: object,
    ) -> np.ndarray:
        del eta
        return np.full(
            (cz.size, gamma.size, order + 1),
            np.inf + 0.0j,
            dtype=np.complex128,
        )

    monkeypatch.setattr(
        ewald_module,
        "shifted_delta_sequence_batched",
        nonfinite_shifted_delta,
    )

    with pytest.raises(
        FloatingPointError,
        match=r"reciprocal shell produced non-finite.*order=2, shell=0",
    ):
        ewald_structural_sums_2d_batch(
            lmax_struct=1,
            k=2.0 * np.pi / 550.0,
            destinations=np.array([[65.0, 32.0, 59.0]], dtype=float),
            source=np.zeros((3,), dtype=float),
            lattice=_lattice(),
            k_parallel=np.array([0.0012, -0.0007], dtype=float),
            eta=0.02,
            real_shells=1,
            reciprocal_shells=1,
        )


def test_batch_roundoff_same_plane_rows_use_same_plane_limit() -> None:
    k = 2.0 * np.pi / 550.0
    kp = np.array([0.0012, -0.0007], dtype=float)
    lattice = _lattice()
    source = np.array([12.0, -8.0, 53.71408859848636], dtype=float)
    destinations = np.array(
        [
            [73.0, 28.0, 53.71408859848634],
            [65.0, 32.0, 59.0],
            [-41.0, 109.0, 53.71408859848634],
        ],
        dtype=float,
    )
    exact_destinations = np.asarray(destinations, dtype=float)
    exact_destinations[[0, 2], 2] = source[2]

    near = ewald_structural_sums_2d_batch(
        lmax_struct=2,
        k=k,
        destinations=destinations,
        source=source,
        lattice=lattice,
        k_parallel=kp,
        eta=0.02,
        real_shells=8,
        reciprocal_shells=8,
    )
    exact = ewald_structural_sums_2d_batch(
        lmax_struct=2,
        k=k,
        destinations=exact_destinations,
        source=source,
        lattice=lattice,
        k_parallel=kp,
        eta=0.02,
        real_shells=8,
        reciprocal_shells=8,
    )
    np.testing.assert_allclose(near, exact, rtol=5e-12, atol=5e-12)


def test_ewald_structural_batch_supports_high_mlfmm_closure_degree() -> None:
    """High box orders must not route large factorials through object ufuncs."""

    values = ewald_structural_sums_2d_batch(
        lmax_struct=6,
        k=2.0 * np.pi / 550.0,
        destinations=np.array([[35.0, -22.0, 11.0]], dtype=float),
        source=np.zeros((3,), dtype=float),
        lattice=pcl.RectangularLattice2D(300.0, 320.0),
        k_parallel=np.array([4.0e-4, -2.0e-4], dtype=float),
        eta=2.5e-3,
        real_shells=1,
        reciprocal_shells=1,
        dtype=np.complex128,
    )

    assert values.shape == (1, 13, 25)
    assert np.all(np.isfinite(values))


def test_reciprocal_high_closure_normalization_avoids_factorial_overflow() -> None:
    """High MLFMM closure orders must not materialize factorial roots."""
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(5317.0, 5317.0)
    workspace = EwaldShellWorkspace(
        lattice=lattice,
        k=k,
        k_parallel=np.array([4.0e-4, -2.0e-4]),
        eta=2.5e-3,
    )

    for degree in (172, 342):
        same_plane = _same_plane_reciprocal_sum(
            degree,
            0,
            k=k,
            k_parallel=workspace.k_parallel,
            lattice=lattice,
            eta=workspace.eta,
            shells=0,
            max_shells=1,
            c_xy=np.array([35.0, -22.0]),
            workspace=workspace,
        )
        shifted = _shifted_reciprocal_sum(
            degree,
            0,
            rvec=np.array([35.0, -22.0, 11.0]),
            k=k,
            k_parallel=workspace.k_parallel,
            lattice=lattice,
            eta=workspace.eta,
            shells=0,
            max_shells=1,
            workspace=workspace,
        )

        assert np.isfinite(same_plane)
        assert np.isfinite(shifted)


def test_scaled_real_integral_sequence_matches_low_order_reference() -> None:
    k = 2.0 * np.pi / 550.0
    eta = 2.5e-3
    radii = np.array([300.0, 1200.0, 5317.0], dtype=float)

    for degree in (0, 1, 4, 12):
        reference = real_integral_sequence(degree, eta, k, radii) * (0.5 * k * radii) ** (
            degree + 1
        )
        actual = scaled_real_integral_sequence(degree, eta, k, radii)
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=0.0)


def test_same_plane_real_sum_supports_order_342_closure() -> None:
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(5317.0, 5317.0)
    workspace = EwaldShellWorkspace(
        lattice=lattice,
        k=k,
        k_parallel=np.array([4.0e-4, -2.0e-4]),
        eta=2.5e-3,
    )

    value = _same_plane_real_sum(
        342,
        0,
        k=k,
        k_parallel=workspace.k_parallel,
        lattice=lattice,
        eta=workspace.eta,
        shells=1,
        max_shells=1,
        workspace=workspace,
    )
    assert np.isfinite(value)

    shifted = _shifted_real_sum(
        342,
        0,
        rvec=np.array([8750.0, 0.0, 8750.0]),
        k=k,
        k_parallel=workspace.k_parallel,
        lattice=lattice,
        eta=workspace.eta,
        shells=1,
        max_shells=1,
        workspace=workspace,
    )
    assert np.isfinite(shifted)


def test_shifted_delta_sequence_batched_matches_scalar_rows() -> None:
    gamma = np.asarray([0.011 - 0.0j, 0.006j, 0.003 + 0.004j], dtype=np.complex128)
    z_offsets = np.asarray([38.0, -117.0, 245.0], dtype=float)

    got = shifted_delta_sequence_batched(4, gamma, z_offsets, 0.02)
    expected = np.stack(
        [shifted_delta_sequence(4, gamma, float(z), 0.02) for z in z_offsets],
        axis=0,
    )

    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)


def test_ewald_structural_constant_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="order"):
        ewald_structural_constant_2d(
            1,
            2,
            k=2.0 * np.pi / 550.0,
            destination=np.zeros(3, dtype=float),
            source=np.zeros(3, dtype=float),
            lattice=_lattice(),
            k_parallel=np.zeros(2, dtype=float),
            eta=0.02,
            real_shells=1,
            reciprocal_shells=1,
        )


def test_adaptive_shells_do_not_stop_on_cancelling_large_shell_pair() -> None:
    seen: list[int] = []

    def evaluate(shell: int) -> complex:
        seen.append(int(shell))
        return {0: 1.0 + 0.0j, 1: -1.0 + 0.0j}.get(int(shell), 0.0 + 0.0j)

    control = make_lattice_shell_control(
        shells=None,
        max_shells=3,
        shell_tolerance=1.0e-6,
        name="demo_shells",
    )

    result = accumulate_lattice_shell_series(
        control=control,
        name="demo_shells",
        zero=0.0 + 0.0j,
        evaluate_shell=evaluate,
    )

    assert result == pytest.approx(0.0 + 0.0j)
    assert seen == [0, 1, 2, 3]


def test_fixed_shell_counts_ignore_adaptive_limits() -> None:
    k = 2.0 * np.pi / 550.0
    kp = np.array([0.0012, -0.0007], dtype=float)
    destination = np.array([65.0, 32.0, 59.0], dtype=float)
    a = ewald_structural_constant_2d(
        2,
        0,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=kp,
        eta=0.02,
        real_shells=4,
        reciprocal_shells=4,
        max_shells=1,
        shell_tolerance=1e-4,
    )
    b = ewald_structural_constant_2d(
        2,
        0,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=kp,
        eta=0.02,
        real_shells=4,
        reciprocal_shells=4,
        max_shells=32,
        shell_tolerance=1e-12,
    )
    assert a == pytest.approx(b, rel=1e-14, abs=1e-14)


def test_automatic_eta_raises_unstable_large_cell_split() -> None:
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(ax=3600.0, ay=3600.0)
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 10.0],
        ],
        dtype=float,
    )

    eta = select_ewald_eta(
        lattice=lattice,
        k=k,
        k_parallel=np.zeros(2, dtype=float),
        positions=positions,
        lmax=3,
        real_shells=12,
        reciprocal_shells=12,
    )

    assert eta > 2.0 * default_ewald_eta(lattice)
    assert eta <= 0.35 * k * (1.0 + 1.0e-12)


def test_oblique_eta_selection_uses_adaptive_shell_tolerance() -> None:
    """Oblique Bloch phases must not be rejected by a hidden low-shell probe."""

    lattice = pcl.RectangularLattice2D(ax=3544.8765, ay=3544.8765)
    k = 2.0 * np.pi / 366.6666666666667
    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.5,
        polarization="TE",
        polar_angle=math.radians(25.0),
        azimuthal_angle=0.0,
    )
    periodic = pcl.PeriodicSpec(
        lattice=lattice,
        options=pcl.PeriodicOptions(
            method="rayleigh",
            shell_tolerance=1.0e-8,
            max_shells=64,
        ),
    )
    # These offsets reproduce the near-plane and clipped far-z probes of the
    # 10-um production crop without loading its large geometry file.
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-201.856832, 521.510926, 0.0019],
            [1644.7113, -981.1352, 35_000.0],
        ],
        dtype=float,
    )

    eta = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=plane_wave_k_parallel(source),
        positions=positions,
        lmax=4,
        max_vertical_offset=366.6666667,
    )

    assert eta == pytest.approx(4.0 * default_ewald_eta(lattice), rel=0.0, abs=1.0e-15)


def test_rayleigh_eta_preflight_is_limited_to_the_exact_near_band() -> None:
    k = 2.0 * np.pi / 366.6666666666667
    lattice = pcl.RectangularLattice2D(ax=3544.8765, ay=3544.8765)
    options = pcl.PeriodicOptions(
        method="rayleigh",
        rayleigh_z_cut=400.0,
        shell_tolerance=1.0e-8,
        max_shells=64,
    )
    periodic = pcl.PeriodicSpec(lattice=lattice, options=options)
    near_positions = np.asarray([[0.0, 0.0, 0.0], [-300.0, 250.0, 400.0]], dtype=float)
    tall_positions = np.asarray([[0.0, 0.0, 0.0], [-300.0, 250.0, 35_000.0]], dtype=float)

    near_eta = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=np.zeros(2),
        positions=near_positions,
        lmax=2,
    )
    tall_eta = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=np.zeros(2),
        positions=tall_positions,
        lmax=2,
    )

    assert tall_eta == pytest.approx(near_eta, rel=0.0, abs=0.0)
    assert tall_eta > 2.0 * default_ewald_eta(lattice)


def test_automatic_ewald_eta_accepts_stable_tall_cell_after_large_height_fix() -> None:
    k = 2.0 * np.pi / 366.6666666666667
    lattice = pcl.RectangularLattice2D(ax=3544.8765, ay=3544.8765)
    periodic = pcl.PeriodicSpec(
        lattice=lattice,
        options=pcl.PeriodicOptions(method="ewald", shell_tolerance=1.0e-8, max_shells=64),
    )
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [100.0, -80.0, 35_000.0]],
        dtype=float,
    )

    eta = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=np.zeros(2),
        positions=positions,
        lmax=2,
    )

    # The tall offset is valid; the former rejection was caused by an
    # overflow in the lower-half-plane Faddeeva product rather than by an
    # unresolved eta split.
    assert eta > 2.0 * default_ewald_eta(lattice)
    assert np.isfinite(eta)


@pytest.mark.reference
def test_adaptive_shells_match_large_fixed_reference_on_benign_case() -> None:
    k = 2.0 * np.pi / 550.0
    kp = np.array([0.0012, -0.0007], dtype=float)
    destination = np.array([73.0, 28.0, 17.0], dtype=float)

    fixed = ewald_structural_sums_2d(
        lmax=2,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=kp,
        eta=0.02,
        real_shells=16,
        reciprocal_shells=16,
    )
    adaptive = ewald_structural_sums_2d(
        lmax=2,
        k=k,
        destination=destination,
        source=np.zeros(3, dtype=float),
        lattice=_lattice(),
        k_parallel=kp,
        eta=0.02,
        real_shells=None,
        reciprocal_shells=None,
        shell_tolerance=1e-10,
        max_shells=32,
    )
    np.testing.assert_allclose(adaptive, fixed, rtol=2e-9, atol=2e-9)


def test_adaptive_reciprocal_guard_includes_propagating_orders() -> None:
    k = 2.0 * np.pi / 550.0
    lattice = pcl.RectangularLattice2D(ax=700.0, ay=700.0)
    kp = np.zeros(2, dtype=float)
    destination = np.zeros(3, dtype=float)
    source = np.zeros(3, dtype=float)
    fixed = ewald_structural_constant_2d(
        0,
        0,
        k=k,
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=kp,
        eta=0.02,
        real_shells=0,
        reciprocal_shells=2,
        max_shells=2,
        shell_tolerance=1.0,
        exclude_zero_shift=True,
    )
    adaptive = ewald_structural_constant_2d(
        0,
        0,
        k=k,
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=kp,
        eta=0.02,
        real_shells=0,
        reciprocal_shells=None,
        max_shells=2,
        shell_tolerance=1.0,
        exclude_zero_shift=True,
    )
    assert adaptive == pytest.approx(fixed, rel=1e-14, abs=1e-14)


def test_adaptive_shells_raise_on_non_convergence() -> None:
    with pytest.raises(PeriodicEwaldConvergenceError, match="did not converge"):
        _ = ewald_structural_constant_2d(
            4,
            1,
            k=2.0 * np.pi / 550.0,
            destination=np.array([65.0, 32.0, 59.0], dtype=float),
            source=np.zeros(3, dtype=float),
            lattice=_lattice(),
            k_parallel=np.array([0.0012, -0.0007], dtype=float),
            eta=0.02,
            real_shells=None,
            reciprocal_shells=None,
            shell_tolerance=1e-14,
            max_shells=1,
        )


def test_resolve_ewald_shell_counts_honors_explicit_options():
    from pyceles.core.lattice import RectangularLattice2D
    from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
    from pyceles.core.periodic.ewald import resolve_ewald_shell_counts

    lattice = RectangularLattice2D(ax=3600.0, ay=3600.0)
    spec = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(real_shells=3, reciprocal_shells=5, max_shells=16),
    )
    counts = resolve_ewald_shell_counts(
        periodic=spec,
        k=2 * np.pi / 550.0,
        k_parallel=np.zeros(2),
        positions=np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 200.0]]),
        lmax=3,
    )
    assert counts.real_shells == 3
    assert counts.reciprocal_shells == 5
    assert counts.selected_by == "explicit"


def test_resolve_ewald_shell_counts_returns_bounded_probe_counts():
    from pyceles.core.lattice import RectangularLattice2D
    from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
    from pyceles.core.periodic.ewald import resolve_ewald_shell_counts

    lattice = RectangularLattice2D(ax=3600.0, ay=3600.0)
    spec = PeriodicSpec(
        lattice=lattice, options=PeriodicOptions(max_shells=8, shell_tolerance=1e-8)
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [120.0, 0.0, 200.0], [300.0, 50.0, 900.0]])
    k = 2 * np.pi / 550.0
    eta = 1.4e-3
    counts = resolve_ewald_shell_counts(
        periodic=spec,
        k=k,
        k_parallel=np.zeros(2),
        positions=positions,
        lmax=3,
        eta=eta,
    )
    assert 0 <= counts.real_shells <= 8
    assert 0 <= counts.reciprocal_shells <= 8
    assert counts.selected_by in {
        "probe",
        "max_shells_fallback",
    }


def test_shell_count_fallback_uses_configured_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate probe must not reintroduce a hidden fixed shell count."""

    from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
    from pyceles.core.periodic.ewald import resolve_ewald_shell_counts

    lattice = pcl.RectangularLattice2D(ax=3600.0, ay=3600.0)
    spec = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(max_shells=16, shell_tolerance=1.0e-8),
    )
    monkeypatch.setattr(
        ewald_module, "_representative_ewald_eta_offsets", lambda **_: np.empty((0, 3))
    )

    counts = resolve_ewald_shell_counts(
        periodic=spec,
        k=2.0 * np.pi / 550.0,
        k_parallel=np.zeros(2),
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        lmax=2,
        eta=1.0e-3,
    )

    assert counts.real_shells == 16
    assert counts.reciprocal_shells == 16
    assert counts.selected_by == "max_shells_fallback"


def test_shell_count_reference_failure_is_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed GPU counts must not hide a non-finite max-shell reference."""

    from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
    from pyceles.core.periodic.ewald import resolve_ewald_shell_counts

    lattice = pcl.RectangularLattice2D(ax=3600.0, ay=3600.0)
    spec = PeriodicSpec(
        lattice=lattice,
        options=PeriodicOptions(max_shells=16, shell_tolerance=1.0e-8),
    )
    monkeypatch.setattr(ewald_module, "_reference_shell_probe_vector", lambda **_: None)

    with pytest.raises(ewald_module.PeriodicEwaldConvergenceError, match="max-shell reference"):
        resolve_ewald_shell_counts(
            periodic=spec,
            k=2.0 * np.pi / 550.0,
            k_parallel=np.zeros(2),
            positions=np.asarray([[0.0, 0.0, 0.0]]),
            lmax=2,
            eta=1.0e-3,
        )


def test_near_coplanar_structural_sum_has_smooth_same_plane_limit() -> None:
    source = np.asarray([-1257.03296776, 2173.52998709, 3841.006126], dtype=float)
    dz = 2.9574860172942863e-4
    destination = np.asarray([-7.63834228, 2721.94247985, source[2] + dz], dtype=float)
    mirrored = destination.copy()
    mirrored[2] = source[2] - dz
    same_plane = destination.copy()
    same_plane[2] = source[2]
    k = 2.0 * np.pi / 632.8
    lattice = pcl.RectangularLattice2D(6117.812563520112, 6117.812563520112)
    eta = 1.1588807813266307e-3

    values = ewald_structural_sums_2d_batch(
        lmax_struct=3,
        k=k,
        destinations=np.stack([destination, mirrored, same_plane]),
        source=source,
        lattice=lattice,
        k_parallel=np.zeros(2),
        eta=eta,
        real_shells=12,
        reciprocal_shells=12,
        max_shells=12,
    )

    assert np.linalg.norm(values[0]) < 1.0
    np.testing.assert_allclose(values[0], values[2], rtol=2e-5, atol=3e-7)
    for degree in range(7):
        for order in range(-degree, degree + 1):
            parity = -1.0 if (degree - abs(order)) % 2 else 1.0
            np.testing.assert_allclose(
                values[1, degree, order + 6],
                parity * values[0, degree, order + 6],
                rtol=2e-11,
                atol=2e-11,
            )


def test_ewald_policy_preflights_are_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated operator/postprocess policy queries should reuse scalar probe results."""
    ewald_module._select_ewald_eta_from_offsets_cached.cache_clear()
    ewald_module._resolve_ewald_shell_counts_from_offsets_cached.cache_clear()
    lattice = pcl.RectangularLattice2D(ax=3600.0, ay=3600.0)
    periodic = pcl.PeriodicSpec(
        lattice=lattice,
        options=pcl.PeriodicOptions(method="rayleigh", shell_tolerance=1.0e-8, max_shells=16),
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [130.0, -75.0, 240.0], [90.0, 30.0, 700.0]])
    k = 2.0 * np.pi / 550.0
    kp = np.asarray([0.0002, -0.0001])

    eta_calls = 0
    original_eta_probe = ewald_module._eta_probe_vector

    def counted_eta_probe(**kwargs):
        nonlocal eta_calls
        eta_calls += 1
        return original_eta_probe(**kwargs)

    monkeypatch.setattr(ewald_module, "_eta_probe_vector", counted_eta_probe)
    eta_first = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=kp,
        positions=positions,
        lmax=3,
        max_vertical_offset=400.0,
    )
    first_eta_calls = eta_calls
    assert first_eta_calls > 0
    eta_second = resolve_ewald_eta(
        periodic=periodic,
        k=k,
        k_parallel=kp,
        positions=positions.copy(),
        lmax=3,
        max_vertical_offset=400.0,
    )
    assert eta_second == eta_first
    assert eta_calls == first_eta_calls

    # Shell selection uses the same compact offset key and is cached independently.
    counts_first = ewald_module.resolve_ewald_shell_counts(
        periodic=periodic,
        k=k,
        k_parallel=kp,
        positions=positions,
        lmax=3,
        eta=eta_first,
        max_vertical_offset=400.0,
    )
    calls_after_counts = eta_calls
    assert calls_after_counts > first_eta_calls
    counts_second = ewald_module.resolve_ewald_shell_counts(
        periodic=periodic,
        k=k,
        k_parallel=kp,
        positions=positions.copy(),
        lmax=3,
        eta=eta_first,
        max_vertical_offset=400.0,
    )
    assert counts_second == counts_first
    assert eta_calls == calls_after_counts


def test_shifted_reciprocal_plane_factorization_matches_scalar_reference() -> None:
    """Off-plane reciprocal Ewald sums factor exactly into source/destination phases."""
    from pyceles.core.periodic.scalar import (
        log_factorial,
        log_spherical_factorial_root,
        structural_sum_m_normalization,
    )

    k = 2.0 * np.pi / 550.0
    kp = np.array([0.0012, -0.0007], dtype=float)
    lattice = _lattice()
    eta = 0.02
    shells = 2
    source = np.array([31.0, -47.0, 63.0], dtype=float)
    plane_z = -19.0
    workspace = EwaldShellWorkspace(
        lattice=lattice,
        k=float(k),
        k_parallel=kp,
        eta=float(eta),
    )
    destinations = np.array(
        [
            [0.0, 0.0, plane_z],
            [17.0, -23.0, plane_z],
            [-81.0, 54.0, plane_z],
        ],
        dtype=float,
    )

    for degree, order_m in ((2, 1), (3, -1)):
        amplitudes: list[np.ndarray] = []
        wavevectors: list[np.ndarray] = []
        base_log = (
            log_spherical_factorial_root(degree, order_m)
            - degree * math.log(2.0)
            - math.log(lattice.area)
            - 2.0 * math.log(float(k))
        )
        cz = float(source[2] - plane_z)
        z_scaled = -float(k) * cz
        for shell in range(shells + 1):
            data = workspace.reciprocal_shell(shell)
            n_values = np.arange(0, degree - abs(order_m) + 1, dtype=np.int64)
            inner = np.zeros((data.rho.size, n_values.size), dtype=np.complex128)
            for n in n_values:
                s_values = np.arange(
                    int(n),
                    min(degree - abs(order_m), 2 * int(n)) + 1,
                    dtype=np.int64,
                )
                parity = (degree - abs(order_m)) & 1
                s_values = s_values[(s_values & 1) == parity]
                for s_value in s_values:
                    log_weight = base_log - (
                        log_factorial(2 * int(n) - int(s_value))
                        + log_factorial(int(s_value) - int(n))
                        + log_factorial((degree + abs(order_m) - int(s_value)) // 2)
                        + log_factorial((degree - abs(order_m) - int(s_value)) // 2)
                    )
                    q_power = 2 * int(n) - int(s_value)
                    rho_power = degree - int(s_value)
                    inner[:, int(n)] += (
                        z_scaled**q_power * (data.rho / float(k)) ** rho_power * np.exp(log_weight)
                    )

            def upper_gamma_provider(max_index: int, shell: int = shell) -> np.ndarray:
                return workspace.upper_gamma(shell, max_index)

            delta = shifted_delta_sequence(
                int(n_values[-1]),
                data.gamma,
                cz,
                eta,
                upper_gamma_provider=upper_gamma_provider,
                series_exclusion=data.rayleigh_zero,
            )
            radial = np.sum(
                (data.gamma / float(k))[:, None] ** (2 * n_values - 1) * delta * inner,
                axis=1,
            )
            source_phase = np.exp(-1j * (data.kgt @ source[:2]))
            term_values = (
                structural_sum_m_normalization(order_m)
                * ((-1j) ** order_m)
                * ((-1.0) ** degree)
                * source_phase
                * np.exp(1j * order_m * data.phi)
                * radial
            )
            amplitudes.append(term_values)
            wavevectors.append(data.kgt)

        amplitude = np.concatenate(amplitudes)
        kgt = np.concatenate(wavevectors, axis=0)
        factored = np.exp(1j * (destinations[:, :2] @ kgt.T)) @ amplitude
        reference = np.asarray(
            [
                structural_sum_m_normalization(order_m)
                * _shifted_reciprocal_sum(
                    degree,
                    order_m,
                    rvec=destination - source,
                    k=float(k),
                    k_parallel=kp,
                    lattice=lattice,
                    eta=float(eta),
                    shells=shells,
                    workspace=workspace,
                )
                for destination in destinations
            ],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(factored, reference, rtol=2.0e-13, atol=2.0e-13)
