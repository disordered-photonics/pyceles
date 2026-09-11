from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr as scipy_lsqr

from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.operators import PeriodicCouplingOperator, prepare_matvec
from pyceles.core.operators.base import PreparedOperator
from pyceles.core.particles import Sphere, Spheroid
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.translation import translation_ab5_table
from pyceles.linear.krylov_cupy import lsqr_cupy_native


class _SyntheticParticleT:
    def __init__(self, diagonal: np.ndarray) -> None:
        self._diagonal = diagonal

    def mode_diagonal(self) -> np.ndarray:
        return self._diagonal

    def apply(self, values: np.ndarray) -> np.ndarray:
        factors = self._diagonal.reshape((-1,) + (1,) * (values.ndim - 1))
        return np.asarray(factors * np.asarray(values))

    def apply_adjoint(self, values: np.ndarray) -> np.ndarray:
        factors = np.conjugate(self._diagonal).reshape((-1,) + (1,) * (values.ndim - 1))
        return np.asarray(factors * np.asarray(values))


@dataclass
class _SyntheticCoupling:
    matrix: np.ndarray
    k_parallel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    periodic: PeriodicSpec = field(
        default_factory=lambda: PeriodicSpec(
            lattice=RectangularLattice2D(3000.0, 3000.0),
            options=PeriodicOptions(method="rayleigh"),
        )
    )

    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(values), dtype=np.complex128)

    def apply_adjoint(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix.conj().T @ np.asarray(values), dtype=np.complex128)


def _synthetic_prepared() -> PreparedOperator:
    lmax = 1
    nm = n_modes(lmax)
    permutation = np.empty(nm, dtype=np.int64)
    for tau, degree, order, index in iter_modes(lmax):
        permutation[index] = index_vswf(degree, -order, tau, lmax)
    # A real matrix commuting with J satisfies W^H = J conj(W) J exactly.
    matrix = np.eye(2 * nm) + 0.2 * np.kron(np.eye(2), np.eye(nm)[permutation])
    # Deliberately make T depend on m.  Spherical/layered-sphere T matrices
    # are m-independent and would not detect use of the wrong diagonal slot.
    diagonal = np.asarray(
        [0.03 + 0.01j + (0.004 - 0.002j) * index for index in range(2 * nm)],
        dtype=np.complex128,
    )
    particle_t = _SyntheticParticleT(diagonal)
    coupling = _SyntheticCoupling(matrix)
    return PreparedOperator(
        lmax=lmax,
        positions=np.zeros((2, 3), dtype=float),
        dtype=np.dtype(np.complex128),
        k=1.0,
        particle_t=cast(Any, particle_t),
        coupling=coupling,
    )


def test_periodic_rayleigh_adjoint_matches_synthetic_action() -> None:
    prepared = _synthetic_prepared()
    rng = np.random.default_rng(1729)
    for _ in range(3):
        x = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        y = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        action_x = prepared.apply_A(x)
        lhs = np.vdot(action_x, y)
        rhs = np.vdot(x, prepared.apply_adjoint(y))
        assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300) < 1e-14


def test_prepared_operator_exposes_shape_preserving_adjoint() -> None:
    prepared = _synthetic_prepared()
    first = prepared.make_adjoint(backend="numpy")
    assert callable(first)
    values = np.ones((12, 2), dtype=np.complex128)
    actual = first(values)
    assert actual.shape == values.shape
    np.testing.assert_allclose(actual, prepared.apply_adjoint(values))


def test_periodic_rayleigh_coupling_adjoint_matches_stored_operator() -> None:
    """The NumPy Rayleigh action and its adjoint satisfy the inner-product identity."""
    lmax = 1
    positions = np.asarray(
        [[0.0, 0.0, -120.0], [180.0, -75.0, 35.0], [-210.0, 130.0, 260.0]],
        dtype=float,
    )
    coupling = PeriodicCouplingOperator(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=np.complex128),
        periodic=PeriodicSpec(
            lattice=RectangularLattice2D(900.0, 850.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.002,
                real_shells=2,
                reciprocal_shells=2,
                rayleigh_reciprocal_shells=1,
                rayleigh_z_cut=161.0,
                shell_tolerance=1.0e-7,
                max_shells=2,
            ),
        ),
        k_parallel=np.asarray([0.001, -0.0004]),
        dtype=np.dtype(np.complex128),
        circumscribing_radii=np.full(positions.shape[0], 80.0),
    )
    rng = np.random.default_rng(20260909)
    x = rng.standard_normal(positions.shape[0] * n_modes(lmax)) + 1j * rng.standard_normal(
        positions.shape[0] * n_modes(lmax)
    )
    y = rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    lhs = np.vdot(coupling.apply(x), y)
    rhs = np.vdot(x, coupling.apply_adjoint(y))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300) < 2.0e-12


def test_periodic_rayleigh_adjoint_supports_materialized_spheroid_t_blocks() -> None:
    """Materialized non-diagonal axisymmetric T blocks are covered exactly."""
    lmax = 2
    particles = (
        Sphere(position=(5.0, -120.0, -10.0), radius=26.0, refractive_index=1.4 + 0.01j),
        Spheroid(
            position=(-90.0, -40.0, -75.0),
            equatorial_radius=30.0,
            polar_radius=44.0,
            refractive_index=1.7 + 0.02j,
            euler_angles=(0.1, 0.25, -0.2),
        ),
        Spheroid(
            position=(110.0, 55.0, 80.0),
            equatorial_radius=30.0,
            polar_radius=44.0,
            refractive_index=1.7 + 0.02j,
            euler_angles=(-0.3, 0.15, 0.4),
        ),
    )
    prepared = prepare_matvec(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=0.0,
        operator_dtype=np.complex128,
        accum_dtype=np.complex128,
        periodic=PeriodicSpec(
            lattice=RectangularLattice2D(520.0, 570.0),
            options=PeriodicOptions(
                method="rayleigh",
                eta=0.004,
                real_shells=2,
                reciprocal_shells=2,
                rayleigh_reciprocal_shells=1,
                rayleigh_z_cut=130.0,
                shell_tolerance=1.0e-10,
                max_shells=3,
            ),
        ),
        k_parallel=np.asarray([0.0011, -0.0007]),
        backend="numpy",
    )
    rng = np.random.default_rng(20260909)
    n = len(particles) * n_modes(lmax)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    lhs = np.vdot(prepared.apply_A(x), y)
    rhs = np.vdot(x, prepared.apply_adjoint(y))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300) < 2.0e-12


@pytest.mark.fake_gpu
def test_native_lsqr_forwards_operator_and_accumulation_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep this test self-contained: importing a helper from test_solvers makes
    # mypy discover that un-packaged test module under two different names.
    cupy = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        array=np.array,
        zeros_like=np.zeros_like,
        zeros=np.zeros,
        concatenate=np.concatenate,
        column_stack=np.column_stack,
        stack=np.stack,
        arange=np.arange,
        abs=np.abs,
        sqrt=np.sqrt,
        conj=np.conj,
        vdot=np.vdot,
        max=np.max,
        linalg=SimpleNamespace(norm=np.linalg.norm),
    )
    rng = np.random.default_rng(19)
    matrix = (rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))).astype(np.complex64)
    rhs = (rng.standard_normal(6) + 1j * rng.standard_normal(6)).astype(np.complex64)
    callbacks: list[float] = []
    result = lsqr_cupy_native(
        lambda values: matrix @ np.asarray(values),
        lambda values: matrix.conj().T @ np.asarray(values),
        rhs,
        cupy=cupy,
        operator_dtype=np.complex64,
        accum_dtype=np.complex128,
        maxiter=32,
        callback=callbacks.append,
        compute_final_residual=True,
    )
    assert result.x.dtype == np.dtype(np.complex64)
    assert result.iterations == len(callbacks)
    assert result.operator_applications >= result.iterations
    assert result.adjoint_applications == result.iterations + 1
    assert result.relative_residual < 1e-5
    np.testing.assert_allclose(matrix @ result.x, rhs, atol=2e-5, rtol=2e-5)
    reference = scipy_lsqr(
        cast(
            Any,
            LinearOperator(
                shape=matrix.shape,
                dtype=np.complex128,
                matvec=lambda values: matrix.astype(np.complex128) @ values,
                rmatvec=lambda values: matrix.astype(np.complex128).conj().T @ values,
            ),
        ),
        rhs.astype(np.complex128),
        atol=0.0,
        btol=0.0,
        conlim=1.0e20,
        iter_lim=32,
        show=False,
    )
    np.testing.assert_allclose(result.x, reference[0], atol=3e-5, rtol=3e-5)

    # Diagnostic recurrences are part of the switching policy, so check them
    # independently rather than only comparing the final solution.
    diagnostic = lsqr_cupy_native(
        lambda values: matrix.astype(np.complex128) @ np.asarray(values),
        lambda values: matrix.astype(np.complex128).conj().T @ np.asarray(values),
        rhs.astype(np.complex128),
        cupy=cupy,
        operator_dtype=np.complex128,
        accum_dtype=np.complex128,
        rtol=0.0,
        atol=0.0,
        maxiter=4,
        compute_final_residual=False,
    )
    diagnostic_reference = scipy_lsqr(
        cast(
            Any,
            LinearOperator(
                shape=matrix.shape,
                dtype=np.complex128,
                matvec=lambda values: matrix.astype(np.complex128) @ values,
                rmatvec=lambda values: matrix.astype(np.complex128).conj().T @ values,
            ),
        ),
        rhs.astype(np.complex128),
        atol=0.0,
        btol=0.0,
        conlim=0.0,
        iter_lim=4,
        show=False,
    )
    np.testing.assert_allclose(diagnostic.anorm, diagnostic_reference[5], rtol=2e-13)
    np.testing.assert_allclose(diagnostic.acond, diagnostic_reference[6], rtol=2e-13)
    np.testing.assert_allclose(diagnostic.arnorm, diagnostic_reference[7], rtol=2e-13)
    np.testing.assert_allclose(diagnostic.correction_norm, diagnostic_reference[8], rtol=2e-13)


@pytest.mark.fake_gpu
def test_native_lsqr_can_carry_a_known_correction_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    cupy = SimpleNamespace(
        asarray=np.asarray,
        asnumpy=np.asarray,
        array=np.array,
        zeros_like=np.zeros_like,
        zeros=np.zeros,
        concatenate=np.concatenate,
        column_stack=np.column_stack,
        stack=np.stack,
        arange=np.arange,
        abs=np.abs,
        sqrt=np.sqrt,
        conj=np.conj,
        vdot=np.vdot,
        max=np.max,
        linalg=SimpleNamespace(norm=np.linalg.norm),
    )
    matrix = np.asarray(
        [[1.4 + 0.1j, 0.2 - 0.1j], [-0.3 + 0.2j, 1.1 - 0.2j]],
        dtype=np.complex128,
    )
    x0 = np.asarray([0.2 - 0.1j, -0.4 + 0.3j], dtype=np.complex64)
    rhs = np.asarray(matrix @ x0 + np.asarray([0.1 - 0.05j, -0.07 + 0.02j]), dtype=np.complex64)
    residual = rhs - matrix @ x0
    forward_calls = 0

    def forward(values: Any) -> np.ndarray:
        nonlocal forward_calls
        forward_calls += 1
        return matrix @ np.asarray(values)

    result = lsqr_cupy_native(
        forward,
        lambda values: matrix.conj().T @ np.asarray(values),
        rhs,
        cupy=cupy,
        x0=x0,
        initial_residual=residual,
        rhs_norm=float(np.linalg.norm(rhs)),
        operator_dtype=np.complex64,
        accum_dtype=np.complex128,
        maxiter=2,
        compute_final_residual=False,
    )
    # Supplying the residual avoids the otherwise redundant A @ x0 action.
    assert result.operator_applications == 2
    assert forward_calls == result.operator_applications
    assert result.rhs_norm == pytest.approx(float(np.linalg.norm(rhs)))
    assert not np.allclose(result.x, x0)


def test_solve_linear_system_rejects_unvalidated_lsqr_requests() -> None:
    from pyceles.linear.solvers import solve_linear_system

    matrix = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match=r"only.*backend='cupy'"):
        solve_linear_system(
            lambda values: matrix @ values,
            np.ones(2, dtype=np.complex128),
            method="lsqr",
            backend="numpy",
        )
    with pytest.raises(ValueError, match="requires an exact `A_h_mv`"):
        solve_linear_system(
            lambda values: matrix @ values,
            np.ones(2, dtype=np.complex128),
            method="lsqr",
            backend="cupy",
        )
