from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.periodic.ewald import (
    ewald_structural_constant_2d,
    ewald_structural_sums_2d,
)
from pyceles.core.periodic.structural import direct_structural_sums_2d


def _lattice() -> pcl.RectangularLattice2D:
    return pcl.RectangularLattice2D(ax=430.0, ay=470.0)


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
