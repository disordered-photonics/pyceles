from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere
from pyceles.postprocessing.nearfield import NearFieldComponents, compute_near_field_slice
from pyceles.postprocessing.nearfield import workflows as nf_workflows
from pyceles.postprocessing.nearfield.slice import (
    interpolate_center_pixels,
    reshape_field_points,
    slice_plane_metadata,
)
from pyceles.simulation import Simulation, SimulationConfig


def _make_run():
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=PlaneWave(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            polarization="TE",
            polar_angle=0.0,
            azimuthal_angle=0.0,
            amplitude=1.0,
            focal_point=(0.0, 0.0, 0.0),
        ),
        polar_angles=np.linspace(0.0, np.pi, 31),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False),
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.0j)],
    )
    return sim.run(include_farfield=False)


def test_slice_helpers_cover_metadata_reshape_and_center_interpolation():
    assert slice_plane_metadata("x") == (0, 1, 2, "y", "z")
    assert slice_plane_metadata("y") == (1, 0, 2, "x", "z")
    assert slice_plane_metadata("z") == (2, 0, 1, "x", "y")
    with pytest.raises(ValueError, match="plane must be one of"):
        slice_plane_metadata("bad")

    pts_flat, lead_shape = reshape_field_points(np.array([1.0, 2.0, 3.0]))
    assert pts_flat.shape == (1, 3)
    assert lead_shape == ()

    pts_tensor, tensor_shape = reshape_field_points(np.zeros((2, 3, 3), dtype=float))
    assert pts_tensor.shape == (6, 3)
    assert tensor_shape == (2, 3)

    with pytest.raises(ValueError, match="`points` must be shaped"):
        reshape_field_points(np.zeros((2, 2), dtype=float))

    axis_0_grid, axis_1_grid = np.meshgrid(np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]))
    e_map = np.zeros((3, 3, 3), dtype=np.complex128)
    h_map = np.zeros_like(e_map)
    neighbors = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [5.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [7.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    e_map[0, 1] = neighbors[0]
    e_map[2, 1] = neighbors[1]
    e_map[1, 0] = neighbors[2]
    e_map[1, 2] = neighbors[3]
    h_map[:] = 2.0 * e_map
    e_map[1, 1] = 99.0 + 0.0j
    h_map[1, 1] = 199.0 + 0.0j
    run = type(
        "Run", (), {"positions": np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=float)}
    )()
    field_maps = {"total": (e_map, h_map)}
    interpolate_center_pixels(
        run=run,
        axis_0_grid=axis_0_grid,
        axis_1_grid=axis_1_grid,
        field_maps=field_maps,
        plane="y",
        plane_value=0.0,
    )
    np.testing.assert_allclose(e_map[1, 1], np.mean(neighbors, axis=0))
    np.testing.assert_allclose(h_map[1, 1], 2.0 * np.mean(neighbors, axis=0))


@pytest.mark.parametrize(
    ("plane", "plane_value", "expected_first", "expected_last", "axis_labels"),
    [
        ("x", 2.0, np.array([2.0, -1.0, -2.0]), np.array([2.0, 1.0, 2.0]), ("y", "z")),
        ("y", -3.0, np.array([-1.0, -3.0, -2.0]), np.array([1.0, -3.0, 2.0]), ("x", "z")),
        ("z", 4.0, np.array([-1.0, -2.0, 4.0]), np.array([1.0, 2.0, 4.0]), ("x", "y")),
    ],
)
def test_compute_near_field_slice_builds_expected_plane_points_and_interpolates(
    monkeypatch,
    plane: str,
    plane_value: float,
    expected_first: np.ndarray,
    expected_last: np.ndarray,
    axis_labels: tuple[str, str],
):
    captured: dict[str, object] = {}

    def _fake_compute_near_field(
        run, *, points, channel, show_progress, force_general_initial_field
    ):
        del run
        pts = np.asarray(points, dtype=float)
        captured["points"] = pts.copy()
        captured["channel"] = channel
        captured["show_progress"] = show_progress
        captured["force_general_initial_field"] = force_general_initial_field
        shape = pts.shape[:-1]
        vec = np.zeros((*shape, 3), dtype=np.complex64)
        return NearFieldComponents(
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

    interp_calls: list[dict[str, object]] = []

    def _fake_interpolate_center_pixels(**kwargs):
        interp_calls.append(kwargs)

    monkeypatch.setattr(nf_workflows, "compute_near_field", _fake_compute_near_field)
    monkeypatch.setattr(nf_workflows, "interpolate_center_pixels", _fake_interpolate_center_pixels)

    run = type(
        "Run",
        (),
        {"config": type("Cfg", (), {"compute_dtype": "complex64"})(), "particles": tuple()},
    )()

    out = compute_near_field_slice(
        run,
        axis_0_min=-1.0,
        axis_0_max=1.0,
        axis_1_min=-2.0,
        axis_1_max=2.0,
        dx=1.0,
        plane=plane,
        plane_value=plane_value,
        channel="mixed",
        show_progress=False,
        force_general_initial_field=True,
        center_pixel_policy="interpolate",
    )

    pts = cast(np.ndarray, captured["points"])
    np.testing.assert_allclose(pts[0, 0], expected_first)
    np.testing.assert_allclose(pts[-1, -1], expected_last)
    assert captured["channel"] == "mixed"
    assert captured["show_progress"] is False
    assert captured["force_general_initial_field"] is True
    assert out.axis_0_label == axis_labels[0]
    assert out.axis_1_label == axis_labels[1]
    assert out.field_maps["total"][0].dtype == np.complex64
    assert len(interp_calls) == 1
    assert interp_calls[0]["plane"] == plane
    assert interp_calls[0]["plane_value"] == float(plane_value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dx": 0.0}, "`dx` must be > 0"),
        ({"axis_0_min": 2.0, "axis_0_max": 1.0}, "`axis_0_max` must be >="),
        ({"axis_1_min": 2.0, "axis_1_max": 1.0}, "`axis_1_max` must be >="),
        ({"plane": "bad"}, "`plane` must be one of"),
        ({"center_pixel_policy": "bad"}, "`center_pixel_policy` must be 'none' or 'interpolate'"),
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
        center_pixel_policy="none",
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
