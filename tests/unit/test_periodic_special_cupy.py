from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import special as scipy_special

pytestmark = [pytest.mark.gpu, pytest.mark.reference]


def test_wofz_cupy_matches_scipy_reference(cupy_runtime: tuple[Any, Any]) -> None:
    cp, _ = cupy_runtime
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


def test_shifted_delta_sequence_cupy_matches_numpy_reference(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy

    gamma = np.asarray([0.008 + 0.001j, 0.011 + 0.003j, 0.004 + 0.009j], dtype=np.complex128)
    got = cp.asnumpy(shifted_delta_sequence_cupy(4, cp.asarray(gamma), 275.0, 0.0015, cupy=cp))
    want = shifted_delta_sequence(4, gamma, 275.0, 0.0015)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)


def test_shifted_delta_sequence_cupy_stays_finite_for_tall_cell(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy

    k = 2.0 * np.pi / 550.0
    got = cp.asnumpy(
        shifted_delta_sequence_cupy(
            3,
            cp.asarray([k + 0.0j]),
            10550.4,
            2.67125e-3,
            cupy=cp,
        )
    )
    want = shifted_delta_sequence(3, np.asarray([k + 0.0j]), 10550.4, 2.67125e-3)
    np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
    assert np.all(np.isfinite(got.real))
    assert np.all(np.isfinite(got.imag))


def test_real_integral_sequence_cupy_matches_numpy_reference(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
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


def test_shifted_delta_sequence_cupy_batched_matches_reference_rows(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
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


def test_shifted_delta_sequence_cupy_near_plane_matches_numpy_high_order(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy

    eta = 1.1588807813266307e-3
    gamma = np.asarray([0.006 + 0.0j, 0.02j], dtype=np.complex128)
    z_offset = 2.9574860172942863e-4
    got = cp.asnumpy(shifted_delta_sequence_cupy(12, cp.asarray(gamma), z_offset, eta, cupy=cp))
    want = shifted_delta_sequence(12, gamma, z_offset, eta)
    np.testing.assert_allclose(got, want, rtol=3e-13, atol=3e-13)


def test_fixed_cupy_ewald_near_coplanar_matches_numpy(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    from pyceles.core.lattice import RectangularLattice2D
    from pyceles.core.periodic.ewald import ewald_structural_sums_2d_batch
    from pyceles.core.periodic.ewald_cupy import (
        CupyEwaldShellWorkspace,
        ewald_structural_sums_2d_fixed_cupy,
    )

    source = np.asarray([-1257.03296776, 2173.52998709, 3841.006126], dtype=float)
    destination = np.asarray([-7.63834228, 2721.94247985, 3841.0064217486017], dtype=float)
    k = 2.0 * np.pi / 632.8
    eta = 1.1588807813266307e-3
    lattice = RectangularLattice2D(6117.812563520112, 6117.812563520112)
    workspace = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=lattice,
        k=k,
        k_parallel=np.zeros(2),
        eta=eta,
    )

    got = cp.asnumpy(
        ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=cp.asarray(
                (source - destination)[None, :], dtype=cp.float64
            ),
            lmax_struct=3,
            workspace=workspace,
            real_shell_count=12,
            reciprocal_shell_count=12,
            coordinate_scale=max(abs(float(source[2])), abs(float(destination[2]))),
        )
    )
    want = ewald_structural_sums_2d_batch(
        lmax_struct=3,
        k=k,
        destinations=destination[None, :],
        source=source,
        lattice=lattice,
        k_parallel=np.zeros(2),
        eta=eta,
        real_shells=12,
        reciprocal_shells=12,
        max_shells=12,
    )
    np.testing.assert_allclose(got, want, rtol=2e-8, atol=2e-9)


def test_shifted_delta_sequence_cupy_honors_series_exclusion(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cp, _ = cupy_runtime
    from pyceles.core.periodic.special import shifted_delta_sequence
    from pyceles.core.periodic.special_cupy import shifted_delta_sequence_cupy

    gamma = np.asarray([1.0e-10j], dtype=np.complex128)
    exclusion = np.asarray([True])
    got = cp.asnumpy(
        shifted_delta_sequence_cupy(
            0,
            cp.asarray(gamma),
            z_offset=1.0e-3,
            eta=1.0e-4,
            series_exclusion=cp.asarray(exclusion),
            cupy=cp,
        )
    )
    want = shifted_delta_sequence(
        0,
        gamma,
        z_offset=1.0e-3,
        eta=1.0e-4,
        series_exclusion=exclusion,
    )
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-14)
