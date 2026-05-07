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
    select_ewald_eta,
)
from pyceles.core.periodic.shells import (
    accumulate_lattice_shell_series,
    make_lattice_shell_control,
)
from pyceles.core.periodic.special import upper_incomplete_gamma_int_or_halfint
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
