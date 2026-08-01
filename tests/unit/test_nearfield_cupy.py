from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.postprocessing.nearfield.scattered import (
    compute_scattered_electric_field,
    compute_scattered_field,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("lmax", [1, 3, 6, 8, 12])
@pytest.mark.parametrize(
    ("compute_dtype", "accum_dtype", "rtol", "atol"),
    [
        (np.complex64, np.complex64, 5e-5, 5e-6),
        (np.complex64, np.complex128, 5e-5, 5e-6),
        (np.complex128, np.complex128, 2e-11, 2e-12),
    ],
)
def test_fused_scattered_near_field_matches_numpy(
    cupy_runtime: tuple[Any, Any],
    lmax: int,
    compute_dtype: Any,
    accum_dtype: Any,
    rtol: float,
    atol: float,
) -> None:
    del cupy_runtime
    rng = np.random.default_rng(1024 + lmax)
    positions = rng.uniform(-300.0, 300.0, size=(11, 3))
    points = rng.uniform(-500.0, 500.0, size=(17, 3)) + np.asarray([0.0, 0.0, 900.0])
    coeffs = (
        rng.normal(size=(positions.shape[0], n_modes(lmax)))
        + 1j * rng.normal(size=(positions.shape[0], n_modes(lmax)))
    ).astype(compute_dtype)
    kwargs = dict(
        k=2.0 * np.pi / 700.0,
        lmax=lmax,
        n_medium=1.2 + 0.03j,
        particle_distance_resolution=0.5,
        show_progress=False,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    e_ref, h_ref = compute_scattered_field(
        points,
        positions,
        coeffs,
        backend="numpy",
        **kwargs,
    )
    e_gpu, h_gpu = compute_scattered_field(
        points,
        positions,
        coeffs,
        backend="cupy",
        **kwargs,
    )
    e_only = compute_scattered_electric_field(
        points,
        positions,
        coeffs,
        backend="cupy",
        **kwargs,
    )

    np.testing.assert_allclose(e_gpu, e_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(h_gpu, h_ref, rtol=rtol, atol=atol)
    np.testing.assert_allclose(e_only, e_ref, rtol=rtol, atol=atol)
