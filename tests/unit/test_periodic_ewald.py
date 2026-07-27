from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.periodic.ewald import (
    PeriodicEwaldConvergenceError,
    _self_correction,
    default_ewald_eta,
    ewald_structural_constant_2d,
    ewald_structural_sums_2d,
    ewald_structural_sums_2d_batch,
    select_ewald_eta,
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
from pyceles.core.periodic.structural import direct_structural_sums_2d


def _lattice() -> pcl.RectangularLattice2D:
    return pcl.RectangularLattice2D(ax=430.0, ay=470.0)


def test_self_correction_uses_half_integer_origin_term() -> None:
    """The 3D origin correction uses Gamma(-1/2, x), not plain Gamma(0, x)."""
    k = 2.0 * np.pi / 550.0
    eta = 0.02
    x = -(k * k) / (4.0 * eta * eta)

    expected = upper_incomplete_gamma_int_or_halfint(-0.5, x) / (4.0 * np.pi)

    assert _self_correction(k, eta) == pytest.approx(expected, rel=1e-15, abs=1e-15)


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
    from pyceles.core.periodic.ewald import resolve_ewald_eta, resolve_ewald_shell_counts

    lattice = RectangularLattice2D(ax=3600.0, ay=3600.0)
    spec = PeriodicSpec(
        lattice=lattice, options=PeriodicOptions(max_shells=8, shell_tolerance=1e-8)
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [120.0, 0.0, 200.0], [300.0, 50.0, 900.0]])
    k = 2 * np.pi / 550.0
    eta = resolve_ewald_eta(periodic=spec, k=k, k_parallel=np.zeros(2), positions=positions, lmax=3)
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
    assert counts.selected_by in {"probe", "fallback", "max_shells_reference_failed"}
