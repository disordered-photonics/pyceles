"""Polarization direction must not depend on its representable input scale."""

from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.polarization import normalize_global_polarization_vector


@pytest.mark.parametrize("scale", (1.0, 1.0e-200, 1.0e200, np.finfo(float).tiny / 16.0))
@pytest.mark.parametrize("phase", (1.0 + 0.0j, 0.0 + 1.0j))
def test_cartesian_polarization_preserves_direction_and_phase(scale, phase) -> None:
    direction = np.asarray([1.0 + 1.0j, -0.5j, 0.5], dtype=np.complex128)
    vector = (phase * direction) * scale
    before = vector.copy()
    result = normalize_global_polarization_vector(vector)
    expected = phase * direction / np.linalg.norm(direction)
    np.testing.assert_allclose(result, expected, rtol=3.0e-15, atol=3.0e-15)
    np.testing.assert_array_equal(vector, before)
    assert not np.shares_memory(vector, result)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, rtol=3.0e-15)


def test_cartesian_polarization_accepts_finite_components_with_unrepresentable_norm() -> None:
    value = 0.9 * np.finfo(float).max
    vector = np.asarray([complex(value, value), 0.0, 0.0])
    result = normalize_global_polarization_vector(vector)
    np.testing.assert_allclose(result, [(1.0 + 1.0j) / np.sqrt(2.0), 0.0, 0.0])


@pytest.mark.parametrize("vector", ([0, 0, 0], [np.nan, 0, 0], [1, np.inf, 0]))
def test_cartesian_polarization_still_rejects_invalid_directions(vector) -> None:
    with pytest.raises(ValueError):
        normalize_global_polarization_vector(vector)
