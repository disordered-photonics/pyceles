from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pyceles.postprocessing.nearfield import (
    NearFieldComponents,
    NearFieldSlice,
    compute_near_field,
    mix_near_field_components,
    mix_near_field_slices,
)
from pyceles.postprocessing.nearfield import workflows as nf_workflows


class _DummyJonesSource:
    def __init__(self, *, wavelength: float = 550.0, medium_n: complex = 1.0 + 0j) -> None:
        self.wavelength = wavelength
        self.medium_n = medium_n
        self.amplitude = 1.0
        self.polarization = "TE"
        self.with_polarization_calls: list[object] = []

    def has_finite_incident_power(self) -> bool:
        return False

    def incident_coeffs(self, *args, **kwargs):  # pragma: no cover - workflow monkeypatched below
        raise AssertionError("incident_coeffs should not be called in this test")

    def jones_coefficients(self) -> tuple[complex, complex]:
        return 1.0 + 0.0j, 0.0 + 0.0j

    def with_polarization(self, polarization):
        self.with_polarization_calls.append(polarization)
        return ("with_polarization", polarization)


class _DummyConfig:
    def __init__(self, *, source) -> None:
        self.source = source
        self.lmax = 1
        self.n_medium = 1.0 + 0j
        self.compute_dtype = "complex128"
        self.accum_dtype = "complex128"
        self.force_general_initial_field = False
        self.radial_lut_dr = 0.5

    def source_angular_grids(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0.0, np.pi / 2.0]), np.array([0.0, np.pi], dtype=float)

    def resolved_postprocessing_backend(self) -> str:
        return "numpy"


def _mock_components(scale: complex) -> NearFieldComponents:
    z = np.ones((2, 3), dtype=np.complex128) * scale
    return NearFieldComponents(
        E_initial=z,
        H_initial=2 * z,
        E_scattered=3 * z,
        H_scattered=4 * z,
        E_internal=5 * z,
        H_internal=6 * z,
        E_total=7 * z,
        H_total=8 * z,
        inside_mask=np.array([True, False]),
    )


def test_mix_near_field_components_linear_combination():
    nf_te = _mock_components(1.0 + 0.0j)
    nf_tm = _mock_components(0.0 + 1.0j)
    a_te = 1.0 + 0.0j
    a_tm = 1.0j
    mixed = mix_near_field_components(nf_te, nf_tm, a_te=a_te, a_tm=a_tm)
    np.testing.assert_allclose(mixed.E_total, a_te * nf_te.E_total + a_tm * nf_tm.E_total)
    np.testing.assert_allclose(mixed.H_initial, a_te * nf_te.H_initial + a_tm * nf_tm.H_initial)
    assert mixed.inside_mask.tolist() == [True, False]


def test_mix_near_field_slices_linear_combination():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), indexing="xy")
    te_map = np.ones((*axis_0.shape, 3), dtype=np.complex128)
    tm_map = 2j * np.ones_like(te_map)
    slice_te = NearFieldSlice(
        axis_0=axis_0,
        axis_1=axis_1,
        inside=np.array([[False, True], [False, False]], dtype=bool),
        field_maps={"total": (te_map, 3 * te_map)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )
    slice_tm = NearFieldSlice(
        axis_0=axis_0.copy(),
        axis_1=axis_1.copy(),
        inside=np.array([[False, False], [True, False]], dtype=bool),
        field_maps={"total": (tm_map, 4 * tm_map)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )
    mixed = mix_near_field_slices(slice_te, slice_tm, a_te=1.0 + 0j, a_tm=1.0j)
    E_mix, H_mix = mixed.field_maps["total"]
    np.testing.assert_allclose(
        E_mix, slice_te.field_maps["total"][0] + 1.0j * slice_tm.field_maps["total"][0]
    )
    np.testing.assert_allclose(
        H_mix, slice_te.field_maps["total"][1] + 1.0j * slice_tm.field_maps["total"][1]
    )
    assert mixed.inside.dtype == bool
    assert mixed.inside.sum() == 2


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda slc: NearFieldSlice(**{**slc.__dict__, "plane": "x"}), "different planes"),
        (
            lambda slc: NearFieldSlice(**{**slc.__dict__, "plane_value": 1.0}),
            "different plane values",
        ),
        (
            lambda slc: NearFieldSlice(**{**slc.__dict__, "axis_0": slc.axis_0[:, :-1]}),
            "different grid shapes",
        ),
        (
            lambda slc: NearFieldSlice(**{**slc.__dict__, "axis_1": slc.axis_1 + 0.1}),
            "different grid coordinates",
        ),
    ],
)
def test_mix_near_field_slices_rejects_mismatched_slice_metadata(mutator, match):
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), indexing="xy")
    vec = np.ones((*axis_0.shape, 3), dtype=np.complex128)
    base = NearFieldSlice(
        axis_0=axis_0,
        axis_1=axis_1,
        inside=np.zeros(axis_0.shape, dtype=bool),
        field_maps={"total": (vec, vec)},
        plane="y",
        plane_value=0.0,
        axis_0_label="x",
        axis_1_label="z",
    )
    with pytest.raises(ValueError, match=match):
        mix_near_field_slices(base, mutator(base), a_te=1.0 + 0j, a_tm=0.0 + 0j)


def test_compute_near_field_validates_source_and_channel():
    fn = cast(Any, compute_near_field)
    source = _DummyJonesSource()
    run = SimpleNamespace(
        config=_DummyConfig(source=source),
        coeffs=np.ones((1, 6), dtype=np.complex128),
        coeffs_basis=None,
        particles=tuple(),
        k=1.23,
        polarization_jones=(1.0 + 0.0j, 1.0 + 0.0j),
    )

    with pytest.raises(ValueError, match="`channel` must be one of"):
        fn(run, points=np.array([0.0, 0.0, 0.0]), channel="bad", show_progress=False)

    run_missing_source = SimpleNamespace(
        config=_DummyConfig(source=None),
        coeffs=run.coeffs,
        coeffs_basis=None,
        particles=tuple(),
        k=run.k,
        polarization_jones=None,
    )
    with pytest.raises(RuntimeError, match="has no source attached"):
        fn(
            run_missing_source,
            points=np.array([0.0, 0.0, 0.0]),
            channel="mixed",
            show_progress=False,
        )

    with pytest.raises(ValueError, match=r"`run\.coeffs_basis` is not available"):
        fn(run, points=np.array([0.0, 0.0, 0.0]), channel="te", show_progress=False)


def test_compute_near_field_selects_basis_payload_and_pure_channel_fallback(monkeypatch):
    fn = cast(Any, compute_near_field)
    captured: list[dict[str, Any]] = []

    def _fake_compute_components(points, **kwargs):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        captured.append(kwargs)
        vec = np.arange(pts.shape[0] * 3, dtype=np.complex128).reshape(pts.shape[0], 3)
        return NearFieldComponents(
            E_initial=vec,
            H_initial=2 * vec,
            E_scattered=3 * vec,
            H_scattered=4 * vec,
            E_internal=5 * vec,
            H_internal=6 * vec,
            E_total=7 * vec,
            H_total=8 * vec,
            inside_mask=np.zeros((pts.shape[0],), dtype=bool),
        )

    monkeypatch.setattr(nf_workflows, "compute_near_field_components", _fake_compute_components)

    basis_source = _DummyJonesSource()
    coeffs = np.arange(6, dtype=np.complex128).reshape(1, 6)
    coeffs_te = coeffs + 10.0
    run_basis = SimpleNamespace(
        config=_DummyConfig(source=basis_source),
        coeffs=coeffs,
        coeffs_basis={"te": coeffs_te},
        particles=tuple(),
        k=2.0,
        polarization_jones=(1.0 + 0.0j, 0.0 + 0.0j),
    )
    out_scalar = fn(
        run_basis,
        points=np.array([1.0, 2.0, 3.0]),
        channel="te",
        show_progress=False,
    )
    assert out_scalar.E_total.shape == (3,)
    assert out_scalar.inside_mask.shape == ()
    assert captured[-1]["coeffs"] is coeffs_te
    assert captured[-1]["beam"] == ("with_polarization", "TE")
    assert basis_source.with_polarization_calls == ["TE"]

    pure_source = _DummyJonesSource()
    run_pure = SimpleNamespace(
        config=_DummyConfig(source=pure_source),
        coeffs=coeffs,
        coeffs_basis=None,
        particles=tuple(),
        k=2.0,
        polarization_jones=(1.0 + 0.0j, 0.0 + 0.0j),
    )
    grid_points = np.zeros((2, 2, 3), dtype=float)
    out_grid = fn(run_pure, points=grid_points, channel="te", show_progress=False)
    assert out_grid.E_total.shape == (2, 2, 3)
    assert out_grid.inside_mask.shape == (2, 2)
    assert captured[-1]["coeffs"] is coeffs
    assert cast(object, captured[-1]["beam"]) is pure_source
    assert pure_source.with_polarization_calls == []
