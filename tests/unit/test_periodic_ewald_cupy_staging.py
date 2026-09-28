from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import ewald_cupy
from pyceles.core.periodic.ewald_cupy import CupyEwaldShellWorkspace

pytestmark = pytest.mark.fake_gpu


def _workspace() -> CupyEwaldShellWorkspace:
    return CupyEwaldShellWorkspace(
        cupy=np,
        lattice=RectangularLattice2D(430.0, 470.0),
        k=2.0 * np.pi / 550.0,
        k_parallel=np.asarray([0.0012, -0.0007], dtype=float),
        eta=0.02,
    )


def test_upper_gamma_staging_grows_one_table_per_reciprocal_layout() -> None:
    workspace = _workspace()

    terms_small = workspace.upper_gamma_terms(3, 4)
    terms_large = workspace.upper_gamma_terms(3, 9)
    assert terms_large.shape[1] == 10
    assert terms_small.shape[0] == terms_large.shape[0]
    assert len(workspace.upper_gamma_terms_cache) == 1


def test_real_term_staging_matches_host_bloch_phases() -> None:
    workspace = _workspace()

    terms = workspace.real_terms(2)
    shifts = np.asarray(terms.shifts)
    expected = np.exp(1j * (shifts[:, :2] @ workspace.k_parallel))

    np.testing.assert_allclose(terms.phase_xy, expected, rtol=0.0, atol=0.0)


def test_fixed_cupy_ewald_canonicalizes_pairs_at_evaluator_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = CupyEwaldShellWorkspace(
        cupy=np,
        lattice=RectangularLattice2D(3909.790257352941, 3909.790257352941),
        k=2.0 * np.pi / (550.0 / 1.36),
        k_parallel=np.asarray([4.0e-4, -2.0e-4], dtype=float),
        eta=1.8133492941951595e-3,
    )
    raw = np.asarray([[-3600.0, -67.0, -168.0]], dtype=float)
    periods = np.asarray([workspace.lattice.ax, workspace.lattice.ay], dtype=float)
    shift = np.rint(raw[0, :2] / periods)
    wrapped = raw.copy()
    wrapped[0, :2] -= shift * periods
    seen: list[np.ndarray] = []

    def add_shifted(*, c: Any, sums: Any, **_: Any) -> None:
        seen.append(np.asarray(c).copy())
        sums[...] = 1.0 + 0.0j

    monkeypatch.setattr(ewald_cupy, "_add_shifted_reciprocal_structural_sums_cupy", add_shifted)
    monkeypatch.setattr(ewald_cupy, "_add_real_space_structural_sums_cupy", lambda **_: None)

    got = ewald_cupy.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=raw,
        lmax_struct=1,
        workspace=workspace,
        real_shell_count=0,
        reciprocal_shell_count=0,
    )

    np.testing.assert_allclose(seen[0], wrapped, rtol=0.0, atol=1.0e-12)
    phase = np.exp(-1j * np.dot(shift * periods, workspace.k_parallel))
    np.testing.assert_allclose(got, phase, rtol=0.0, atol=1.0e-14)


def test_fixed_cupy_ewald_skips_shifted_preparation_for_same_plane_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shifted_calls = 0

    def fail_shifted(**_: Any) -> None:
        nonlocal shifted_calls
        shifted_calls += 1

    monkeypatch.setattr(ewald_cupy, "_add_shifted_reciprocal_structural_sums_cupy", fail_shifted)
    monkeypatch.setattr(ewald_cupy, "_add_real_space_structural_sums_cupy", lambda **_: None)

    got = ewald_cupy.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=np.asarray([[20.0, -15.0, 0.0]], dtype=float),
        lmax_struct=1,
        workspace=_workspace(),
        real_shell_count=1,
        reciprocal_shell_count=1,
    )

    assert got.shape == (1, 3, 5)
    assert shifted_calls == 0


def test_fixed_cupy_ewald_skips_same_plane_tables_for_shifted_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    shifted_calls = 0

    def add_shifted(**_: Any) -> None:
        nonlocal shifted_calls
        shifted_calls += 1

    monkeypatch.setattr(ewald_cupy, "_add_shifted_reciprocal_structural_sums_cupy", add_shifted)
    monkeypatch.setattr(ewald_cupy, "_add_real_space_structural_sums_cupy", lambda **_: None)
    monkeypatch.setattr(
        workspace,
        "reciprocal_terms",
        lambda _shell: (_ for _ in ()).throw(AssertionError("same-plane tables were requested")),
    )

    got = ewald_cupy.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=np.asarray([[20.0, -15.0, 10.0]], dtype=float),
        lmax_struct=1,
        workspace=workspace,
        real_shell_count=1,
        reciprocal_shell_count=1,
    )

    assert got.shape == (1, 3, 5)
    assert shifted_calls == 1


def test_fixed_cupy_ewald_rejects_shifted_exact_rayleigh_zero() -> None:
    workspace = CupyEwaldShellWorkspace(
        cupy=np,
        lattice=RectangularLattice2D(550.0, 700.0),
        k=2.0 * np.pi / 550.0,
        k_parallel=np.zeros(2),
        eta=0.002,
    )

    with pytest.raises(ValueError, match="Rayleigh/Wood anomaly"):
        ewald_cupy.ewald_structural_sums_2d_fixed_cupy(
            relative_source_minus_destination=np.asarray([[20.0, -15.0, 0.1]]),
            lmax_struct=1,
            workspace=workspace,
            real_shell_count=1,
            reciprocal_shell_count=1,
        )


def test_fixed_cupy_ewald_can_skip_shifted_reciprocal_but_keep_real_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shifted_calls = 0
    real_calls = 0

    def add_shifted(**_: Any) -> None:
        nonlocal shifted_calls
        shifted_calls += 1

    def add_real(**_: Any) -> None:
        nonlocal real_calls
        real_calls += 1

    workspace = _workspace()
    monkeypatch.setattr(ewald_cupy, "_add_shifted_reciprocal_structural_sums_cupy", add_shifted)
    monkeypatch.setattr(ewald_cupy, "_add_real_space_structural_sums_cupy", add_real)

    got = ewald_cupy.ewald_structural_sums_2d_fixed_cupy(
        relative_source_minus_destination=np.asarray([[20.0, -15.0, 10.0]], dtype=float),
        lmax_struct=1,
        workspace=workspace,
        real_shell_count=1,
        reciprocal_shell_count=1,
        include_shifted_reciprocal=False,
    )

    assert got.shape == (1, 3, 5)
    assert shifted_calls == 0
    assert real_calls == 1
