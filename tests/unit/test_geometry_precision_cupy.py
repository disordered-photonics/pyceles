"""Coordinate subtraction must precede narrowing to field compute precision."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    build_mlfmm_cupy_host_cache,
    prepare_matvec,
    prepare_mlfmm_cupy_data,
)
from pyceles.core.operators.mlfmm_cupy import _apply_exact_near_pairs
from pyceles.core.particles import spheres_from_arrays
from pyceles.postprocessing.nearfield.scattered import (
    compute_scattered_electric_field,
    compute_scattered_field,
)

pytestmark = pytest.mark.gpu

# All offsets and differences below remain exact in float64. In float32,
# differences of 128 around this origin disappear before any physics is done.
_SHIFT = np.asarray([2**34, -(2**34), 2**34], dtype=np.float64)
_POSITIONS = np.asarray([[0, 0, 0], [128, 256, 0], [-256, 128, 128]], dtype=np.float64)


def _prepare(
    positions: np.ndarray,
    *,
    backend: Literal["numpy", "cupy"],
    dtype: Any,
    mlfmm: bool = False,
):
    return prepare_matvec(
        lmax=1,
        k=2 * np.pi / 550,
        particles=spheres_from_arrays(
            positions=positions,
            radii=np.full(len(positions), 20.0),
            refractive_indices=np.full(len(positions), 1.59 + 0j),
        ),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        operator_dtype=dtype,
        backend=backend,
        coupling_backend="mlfmm" if mlfmm else "pairwise",
        mlfmm_options=(MLFMMOptions(max_leaf_particles=1, max_depth=4) if mlfmm else None),
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("nrhs", (1, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_pairwise_translation_preserves_small_separations_at_large_origins(
    dtype: Any, nrhs: int, adjoint: bool
) -> None:
    reference = _prepare(_POSITIONS, backend="numpy", dtype=dtype)
    shifted = _prepare(_POSITIONS + _SHIFT, backend="cupy", dtype=dtype)
    rng = np.random.default_rng(412)
    shape = (len(_POSITIONS) * n_modes(1), nrhs)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if nrhs == 1:
        x = x[:, 0]
    ref_action = reference.coupling.apply_adjoint if adjoint else reference.coupling.apply
    action = shifted.coupling.apply_adjoint if adjoint else shifted.coupling.apply
    expected = (
        ref_action(x) if nrhs == 1 else np.column_stack([ref_action(x[:, j]) for j in range(nrhs)])
    )
    tolerance = 3e-5 if dtype == np.complex64 else 1e-11
    np.testing.assert_allclose(asnumpy(action(x)), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("adjoint", (False, True))
def test_exact_near_kernel_preserves_coordinate_precision(adjoint: bool) -> None:
    cupy, _ = import_cupy()
    corners = np.asarray(
        [[x, y, z] for x in (-512, 512) for y in (-512, 512) for z in (-512, 512)],
        dtype=np.float64,
    )
    positions = np.vstack((corners, corners + 64))
    coupling = _prepare(positions, backend="numpy", dtype=np.complex64, mlfmm=True).coupling
    assert isinstance(coupling, MLFMMCouplingOperator)
    cache = build_mlfmm_cupy_host_cache(coupling)
    assert cache.near_positions_flat.dtype == np.dtype(np.float64)
    original = prepare_mlfmm_cupy_data(cache)
    # Isolate exact-near geometry from far-expansion/tree translation effects.
    shifted = prepare_mlfmm_cupy_data(
        replace(cache, near_positions_flat=np.ascontiguousarray((positions + _SHIFT).ravel()))
    )
    assert shifted.near_pairs.positions.dtype == cupy.float64
    rng = np.random.default_rng(413)
    shape = (len(positions), n_modes(1), 2)
    x = cupy.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=cupy.complex64
    )
    expected = asnumpy(
        _apply_exact_near_pairs(original, x, workspace=None, cupy=cupy, adjoint=adjoint)
    )
    actual = asnumpy(
        _apply_exact_near_pairs(shifted, x, workspace=None, cupy=cupy, adjoint=adjoint)
    )
    assert np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("dtype", "accum_dtype"),
    ((np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)),
)
def test_scattered_fields_subtract_coordinates_before_narrowing(
    dtype: Any, accum_dtype: Any
) -> None:
    points = np.asarray([[0, 0, 512], [128, -128, 256], [-256, 256, 512]], dtype=np.float64)
    rng = np.random.default_rng(414)
    shape = (len(_POSITIONS), n_modes(1))
    coeffs = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    kwargs = dict(
        k=2 * np.pi / 550,
        lmax=1,
        n_medium=1.0 + 0j,
        particle_distance_resolution=0.5,
        show_progress=False,
        compute_dtype=dtype,
        accum_dtype=accum_dtype,
    )
    e_ref, h_ref = compute_scattered_field(points, _POSITIONS, coeffs, backend="numpy", **kwargs)
    e, h = compute_scattered_field(
        points + _SHIFT, _POSITIONS + _SHIFT, coeffs, backend="cupy", **kwargs
    )
    e_only = compute_scattered_electric_field(
        points + _SHIFT, _POSITIONS + _SHIFT, coeffs, backend="cupy", **kwargs
    )
    tolerance = 5e-5 if dtype == np.complex64 else 2e-11
    np.testing.assert_allclose(e, e_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(h, h_ref, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(e_only, e_ref, rtol=tolerance, atol=tolerance)
