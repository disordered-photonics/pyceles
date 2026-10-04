"""Host workspace ownership around the fused field evaluator (no GPU needed)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.postprocessing.nearfield import scattered, scattered_cupy


def _arguments() -> dict[str, Any]:
    return {
        "field_points": np.arange(18, dtype=float).reshape(6, 3),
        "positions": np.asarray([[0.1, 0.2, 0.3]]),
        "coeffs": np.ones((1, 6), dtype=np.complex64),
        "k": 2.0,
        "lmax": 1,
        "n_medium": 1.0,
        "particle_distance_resolution": 0.1,
        "lut": object(),
        "active_mask": None,
        "batch_size": 64,
        "compute_dtype": np.complex64,
        "accum_dtype": np.complex128,
    }


@pytest.mark.parametrize("all_active_mask", (False, True))
@pytest.mark.parametrize("magnetic", (False, True))
def test_full_grid_returns_owned_evaluator_outputs_without_placeholder(
    monkeypatch: pytest.MonkeyPatch, all_active_mask: bool, magnetic: bool
) -> None:
    kwargs = _arguments()
    points = kwargs["field_points"]
    if all_active_mask:
        kwargs["active_mask"] = np.ones(points.shape[0], dtype=bool)
    electric = np.full(points.shape, 2.0 + 3.0j)
    magnetic_result = np.full(points.shape, -1.0j) if magnetic else None

    def evaluate(**arguments):
        assert arguments["field_points"] is points
        return electric, magnetic_result

    def unexpected_zeros(*args, **kwargs):
        raise AssertionError("Full-grid evaluation must not allocate a zero placeholder.")

    monkeypatch.setattr(scattered_cupy, "compute_scattered_field_cupy_fused", evaluate)
    monkeypatch.setattr(scattered.np, "zeros", unexpected_zeros)
    e, h = scattered._compute_scattered_field_cupy(**kwargs, compute_magnetic=magnetic)
    assert e is electric
    assert h is magnetic_result


@pytest.mark.parametrize("magnetic", (False, True))
def test_masked_destinations_are_allocated_after_evaluation(
    monkeypatch: pytest.MonkeyPatch, magnetic: bool
) -> None:
    kwargs = _arguments()
    mask = np.asarray([True, False, False, True, False, True], dtype=bool)
    kwargs["active_mask"] = mask
    events: list[str] = []
    zeros = np.zeros

    def allocate(*args, **kwargs):
        events.append("allocate")
        return zeros(*args, **kwargs)

    def evaluate(**arguments):
        events.append("evaluate")
        np.testing.assert_array_equal(arguments["field_points"], kwargs["field_points"][mask])
        size = int(mask.sum())
        return np.full((size, 3), 2.0j), np.full((size, 3), -3.0j) if magnetic else None

    monkeypatch.setattr(scattered_cupy, "compute_scattered_field_cupy_fused", evaluate)
    monkeypatch.setattr(scattered.np, "zeros", allocate)
    e, h = scattered._compute_scattered_field_cupy(**kwargs, compute_magnetic=magnetic)
    assert events[0] == "evaluate"
    np.testing.assert_array_equal(e[mask], 2.0j)
    np.testing.assert_array_equal(e[~mask], 0.0)
    if magnetic:
        assert h is not None
        np.testing.assert_array_equal(h[mask], -3.0j)
        np.testing.assert_array_equal(h[~mask], 0.0)
        assert not np.shares_memory(e, h)
    else:
        assert h is None


@pytest.mark.parametrize("empty_case", ("source", "mask", "points"))
def test_empty_work_returns_distinct_zero_fields_without_evaluation(
    monkeypatch: pytest.MonkeyPatch, empty_case: str
) -> None:
    kwargs = _arguments()
    if empty_case == "source":
        kwargs["positions"] = np.empty((0, 3))
    elif empty_case == "points":
        kwargs["field_points"] = np.empty((0, 3))
        kwargs["active_mask"] = np.empty(0, dtype=bool)
    else:
        kwargs["active_mask"] = np.zeros(6, dtype=bool)

    def unexpected_evaluation(**arguments):
        raise AssertionError("No active work must not call the evaluator.")

    monkeypatch.setattr(scattered_cupy, "compute_scattered_field_cupy_fused", unexpected_evaluation)
    e, h = scattered._compute_scattered_field_cupy(**kwargs)
    assert h is not None
    assert e.shape == h.shape == kwargs["field_points"].shape
    np.testing.assert_array_equal(e, 0.0)
    np.testing.assert_array_equal(h, 0.0)
    assert not np.shares_memory(e, h)
