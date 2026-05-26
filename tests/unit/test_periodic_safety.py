from __future__ import annotations

import math

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import (
    rayleigh_report,
    rayleigh_threshold_scales,
    suggest_safe_period_scales,
)


def test_square_normal_incidence_threshold_scales() -> None:
    thresholds = rayleigh_threshold_scales(scale_min=0.5, scale_max=3.1)
    scales = [item.scale for item in thresholds]

    np.testing.assert_allclose(
        scales,
        [1.0, math.sqrt(2.0), 2.0, math.sqrt(5.0), math.sqrt(8.0), 3.0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    first = thresholds[0]
    assert first.multiplicity == 4
    assert set(first.orders) == {(-1, 0), (0, -1), (0, 1), (1, 0)}


@pytest.mark.parametrize(
    ("scale_min", "scale_max", "expected"),
    [
        (5.3, 5.8, 0.5 * (math.sqrt(29.0) + math.sqrt(32.0))),
        (7.2, 7.7, 0.5 * (math.sqrt(53.0) + math.sqrt(58.0))),
        (9.4, 9.9, 0.5 * (math.sqrt(90.0) + math.sqrt(97.0))),
    ],
)
def test_safe_period_suggestions_recover_wide_square_gaps(
    scale_min: float,
    scale_max: float,
    expected: float,
) -> None:
    candidates = suggest_safe_period_scales(
        scale_min=scale_min,
        scale_max=scale_max,
        max_results=3,
    )

    assert candidates
    assert candidates[0].scale == pytest.approx(expected, rel=1.0e-12)
    assert candidates[0].clearance > 0.0


def test_rayleigh_report_flags_near_and_exact_square_thresholds() -> None:
    k = 2.0 * math.pi

    near = rayleigh_report(
        RectangularLattice2D(7.2, 7.2),
        k=k,
        k_parallel=np.zeros(2),
        near_threshold=0.08,
    )
    assert near.warning_level == "near"
    assert near.min_clearance < 0.08
    assert {(abs(item.m), abs(item.n)) for item in near.nearest_orders} == {(4, 6), (6, 4)}

    exact_scale = math.sqrt(52.0)
    exact = rayleigh_report(
        RectangularLattice2D(exact_scale, exact_scale),
        k=k,
        k_parallel=np.zeros(2),
    )
    assert exact.warning_level == "at"
    assert exact.min_clearance == pytest.approx(0.0, abs=1.0e-10)


def test_rayleigh_report_uses_host_wavelength_scaling() -> None:
    wavelength_vacuum = 500.0
    n_medium = 1.5
    k = 2.0 * math.pi * n_medium / wavelength_vacuum

    report = rayleigh_report(
        RectangularLattice2D(1000.0, 2000.0),
        k=k,
        k_parallel=np.zeros(2),
    )

    assert report.period_over_host_wavelength_x == pytest.approx(3.0)
    assert report.period_over_host_wavelength_y == pytest.approx(6.0)


def test_oblique_rectangular_threshold_formula_hits_light_circle() -> None:
    aspect = 1.3
    u = (0.2, 0.1)
    m, n = (2, -1)
    hx = float(m)
    hy = float(n) / aspect
    c = 1.0 - u[0] ** 2 - u[1] ** 2
    scale = (
        u[0] * hx + u[1] * hy + math.sqrt((u[0] * hx + u[1] * hy) ** 2 + c * (hx * hx + hy * hy))
    ) / c

    thresholds = rayleigh_threshold_scales(
        scale_min=scale - 1.0e-9,
        scale_max=scale + 1.0e-9,
        aspect_y_over_x=aspect,
        u_parallel=u,
    )

    assert any((m, n) in item.orders for item in thresholds)
    q = np.array([u[0] + m / scale, u[1] + n / (aspect * scale)])
    assert float(np.linalg.norm(q)) == pytest.approx(1.0, rel=1.0e-12)


def test_threshold_grouping_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="grouping_rtol"):
        rayleigh_threshold_scales(
            scale_min=0.5,
            scale_max=3.0,
            grouping_rtol=0.0,
        )


def test_safe_period_suggestion_policy_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="min_gap_width"):
        suggest_safe_period_scales(scale_min=0.5, scale_max=3.0, min_gap_width=-1.0e-6)
    with pytest.raises(ValueError, match="prefer_smaller_period"):
        suggest_safe_period_scales(scale_min=0.5, scale_max=3.0, prefer_smaller_period=math.inf)
    with pytest.raises(ValueError, match="max_results"):
        suggest_safe_period_scales(scale_min=0.5, scale_max=3.0, max_results=0)
