from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere
from pyceles.postprocessing.nearfield import NearFieldComponents, compute_near_field_slice
from pyceles.postprocessing.nearfield import periodic as nf_periodic
from pyceles.postprocessing.nearfield import workflows as nf_workflows
from pyceles.postprocessing.nearfield.slice import (
    reshape_field_points,
    slice_plane_metadata,
)
from pyceles.simulation import Simulation, SimulationConfig

pytestmark = pytest.mark.api_contract


def _make_run():
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        polar_angles=np.linspace(0.0, np.pi, 31),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False),
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.0j)],
    )
    return sim.run(
        PlaneWave(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            polarization="TE",
            polar_angle=0.0,
            azimuthal_angle=0.0,
            amplitude=1.0,
            focal_point=(0.0, 0.0, 0.0),
        ),
        include_farfield=False,
    )


def test_slice_helpers_cover_metadata_and_reshape():
    assert slice_plane_metadata("x") == (0, 1, 2, "y", "z")
    assert slice_plane_metadata("y") == (1, 0, 2, "x", "z")
    assert slice_plane_metadata("z") == (2, 0, 1, "x", "y")
    with pytest.raises(ValueError, match="`plane` must be one of"):
        slice_plane_metadata("bad")

    pts_flat, lead_shape = reshape_field_points(np.array([1.0, 2.0, 3.0]))
    assert pts_flat.shape == (1, 3)
    assert lead_shape == ()

    pts_tensor, tensor_shape = reshape_field_points(np.zeros((2, 3, 3), dtype=float))
    assert pts_tensor.shape == (6, 3)
    assert tensor_shape == (2, 3)

    with pytest.raises(ValueError, match="`points` must be shaped"):
        reshape_field_points(np.zeros((2, 2), dtype=float))


@pytest.mark.parametrize("periodic", [False, True], ids=["finite", "periodic"])
@pytest.mark.parametrize(
    ("plane", "plane_value", "expected_first", "expected_last", "axis_labels"),
    [
        ("x", 2.0, np.array([2.0, -1.0, -2.0]), np.array([2.0, 1.0, 2.0]), ("y", "z")),
        ("y", -3.0, np.array([-1.0, -3.0, -2.0]), np.array([1.0, -3.0, 2.0]), ("x", "z")),
        ("z", 4.0, np.array([-1.0, -2.0, 4.0]), np.array([1.0, 2.0, 4.0]), ("x", "y")),
    ],
)
def test_near_field_slice_preserves_evaluator_samples_and_dtype(
    monkeypatch,
    plane: str,
    plane_value: float,
    expected_first: np.ndarray,
    expected_last: np.ndarray,
    axis_labels: tuple[str, str],
    periodic: bool,
):
    captured: dict[str, Any] = {}

    def _fake_compute_near_field(run, *, points, show_progress, **kwargs):
        del run
        pts = np.asarray(points, dtype=float)
        captured["points"] = pts.copy()
        captured["show_progress"] = show_progress
        captured.update(kwargs)
        shape = pts.shape[:-1]
        vec = np.zeros((*shape, 3), dtype=np.complex128)
        vec[shape[0] // 2, shape[1] // 2] = [1000 + 2.0**-30, -2j, np.nan]
        fields = NearFieldComponents(
            E_initial=vec + 1.0,
            H_initial=vec + 2.0,
            E_scattered=vec + 3.0,
            H_scattered=vec + 4.0,
            E_internal=vec + 5.0,
            H_internal=vec + 6.0,
            E_total=vec + 7.0,
            H_total=vec + 8.0,
            inside_mask=np.zeros(shape, dtype=bool),
        )
        captured["fields"] = fields
        captured["snapshots"] = {
            name: (getattr(fields, f"E_{name}").copy(), getattr(fields, f"H_{name}").copy())
            for name in ("initial", "scattered", "internal", "total")
        }
        return fields

    module = nf_periodic if periodic else nf_workflows
    evaluator = "compute_periodic_near_field" if periodic else "compute_near_field"
    monkeypatch.setattr(module, evaluator, _fake_compute_near_field)

    run = type(
        "Run",
        (),
        {
            "config": type("Cfg", (), {"compute_dtype": "complex64"})(),
            "particles": tuple(),
            "positions": np.zeros((1, 3)),
        },
    )()
    run.positions[0, slice_plane_metadata(plane)[0]] = plane_value

    slice_fn = (
        nf_periodic.compute_periodic_near_field_slice if periodic else compute_near_field_slice
    )
    options = {} if periodic else {"force_general_initial_field": True}
    out = slice_fn(
        run,
        axis_0_min=-1.0,
        axis_0_max=1.0,
        axis_1_min=-2.0,
        axis_1_max=2.0,
        dx=1.0,
        plane=plane,
        plane_value=plane_value,
        show_progress=False,
        **options,
    )

    pts = cast(np.ndarray, captured["points"])
    np.testing.assert_allclose(pts[0, 0], expected_first)
    np.testing.assert_allclose(pts[-1, -1], expected_last)
    assert captured["show_progress"] is False
    if not periodic:
        assert captured["force_general_initial_field"] is True
    assert out.axis_0_label == axis_labels[0]
    assert out.axis_1_label == axis_labels[1]
    for name, (e, h) in out.field_maps.items():
        assert e is getattr(captured["fields"], f"E_{name}")
        assert h is getattr(captured["fields"], f"H_{name}")
        np.testing.assert_array_equal(e, captured["snapshots"][name][0])
        np.testing.assert_array_equal(h, captured["snapshots"][name][1])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dx": 0.0}, "`dx` must be > 0"),
        ({"axis_0_min": 2.0, "axis_0_max": 1.0}, "`axis_0_max` must be >="),
        ({"axis_1_min": 2.0, "axis_1_max": 1.0}, "`axis_1_max` must be >="),
        ({"plane": "bad"}, "`plane` must be one of"),
    ],
)
def test_compute_near_field_slice_validates_public_inputs(kwargs: dict[str, Any], match: str):
    fn = cast(Any, compute_near_field_slice)
    with pytest.raises(ValueError, match=match):
        fn(type("Run", (), {})(), show_progress=False, **kwargs)


def test_compute_near_field_slice_uses_axis_bounds():
    run = _make_run()
    out = compute_near_field_slice(
        run,
        plane="y",
        plane_value=0.0,
        axis_0_min=-200.0,
        axis_0_max=200.0,
        axis_1_min=-100.0,
        axis_1_max=100.0,
        dx=100.0,
        show_progress=False,
    )
    assert out.axis_0.shape == out.axis_1.shape
    assert out.inside.shape == out.axis_0.shape


def test_compute_near_field_slice_rejects_legacy_xyz_bounds_kwargs():
    run = _make_run()
    fn = cast(Any, compute_near_field_slice)
    with np.testing.assert_raises(TypeError):
        fn(
            run,
            x_min=-200.0,
            x_max=200.0,
            z_min=-100.0,
            z_max=100.0,
            dx=100.0,
            plane="y",
            plane_value=0.0,
            show_progress=False,
        )
