from __future__ import annotations

import numpy as np
import pytest
from scipy import special as scipy_special


def _cupy_or_skip():
    try:
        from pyceles._optional import import_cupy

        cupy, _ = import_cupy()
        # Trigger context creation so CI machines without a working CUDA runtime skip cleanly.
        _ = cupy.asarray([0.0]).sum().get()
        return cupy
    except Exception as exc:  # pragma: no cover - depends on optional GPU availability
        pytest.skip(f"CuPy/CUDA is not available: {exc}")


def test_wofz_cupy_matches_scipy_reference():
    cp = _cupy_or_skip()
    from pyceles.core.periodic.special_cupy import wofz_cupy

    z = np.asarray(
        [
            0.0 + 0.0j,
            0.1 + 0.2j,
            -0.5 + 0.7j,
            1.25 - 0.5j,
            -2.0 + 1.5j,
            3.0 + 0.1j,
        ],
        dtype=np.complex128,
    )
    got = cp.asnumpy(wofz_cupy(cp.asarray(z), cupy=cp))
    want = scipy_special.wofz(z)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)


def test_shifted_delta_sequence_cupy_matches_numpy_reference():
    cp = _cupy_or_skip()
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy

    gamma = np.asarray([0.008 + 0.001j, 0.011 + 0.003j, 0.004 + 0.009j], dtype=np.complex128)
    got = cp.asnumpy(shifted_delta_sequence_cupy(4, cp.asarray(gamma), 275.0, 0.0015, cupy=cp))
    want = shifted_delta_sequence(4, gamma, 275.0, 0.0015)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)


def test_real_integral_sequence_cupy_matches_numpy_reference():
    cp = _cupy_or_skip()
    from pyceles.core.periodic.scalar import real_integral_sequence
    from pyceles.core.periodic.special_cupy import real_integral_sequence_cupy

    radii = np.asarray([120.0, 350.0, 900.0], dtype=np.float64)
    got = cp.asnumpy(
        real_integral_sequence_cupy(4, 0.0015, 2 * np.pi / 550.0, cp.asarray(radii), cupy=cp)
    )
    want = real_integral_sequence(4, 0.0015, 2 * np.pi / 550.0, radii)
    # This recurrence is mildly ill-conditioned for the smallest radius in this
    # sample: near-machine-epsilon differences in the Faddeeva seed are
    # amplified into a few 1e-10 relative error in the final real-space term.
    np.testing.assert_allclose(got, want, rtol=5e-10, atol=1e-14)


def test_shifted_delta_sequence_cupy_batched_matches_reference_rows():
    cp = _cupy_or_skip()
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy_batched

    gamma = np.asarray([0.008 + 0.001j, 0.011 + 0.003j, 0.004 + 0.009j], dtype=np.complex128)
    z_offsets = np.asarray([125.0, 275.0, -410.0], dtype=np.float64)
    got = cp.asnumpy(
        shifted_delta_sequence_cupy_batched(
            4,
            cp.asarray(gamma),
            cp.asarray(z_offsets),
            0.0015,
            cupy=cp,
        )
    )
    want = np.stack(
        [shifted_delta_sequence(4, gamma, float(z_offset), 0.0015) for z_offset in z_offsets],
        axis=0,
    )
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=1e-14)
