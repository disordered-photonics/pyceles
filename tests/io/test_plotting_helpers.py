import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse

from pyceles.core.particles import LayeredSphere, Particle, Sphere, Spheroid
from pyceles.io.plotting import (
    far_field_intensity,
    far_field_intensity_from_result,
    near_field_component,
    plot_farfield_hemispheres,
    plot_field_component,
    plot_nearfield_panels,
    plot_nearfield_panels_channels,
    plot_nearfield_poynting_overlay,
    plot_poynting,
    plot_source_showcase_slices,
    plot_spheres,
    unpolarized_far_field_intensity,
    unpolarized_near_field_intensity,
)
from pyceles.postprocessing.nearfield import NearFieldSlice


def test_near_field_component_extracts_expected_channels():
    E = np.array([[[1 + 2j, 2 + 0j, 3 - 1j]]], dtype=np.complex128)
    H = np.array([[[4 + 1j, 5 + 0j, 6 - 2j]]], dtype=np.complex128)

    assert np.allclose(near_field_component(E, H, "real Ex"), [[1.0]])
    assert np.allclose(near_field_component(E, H, "real Hz"), [[6.0]])
    assert np.allclose(
        near_field_component(E, H, "abs E"), [[np.sqrt(np.abs(E[0, 0, :]) @ np.abs(E[0, 0, :]))]]
    )


def test_far_field_intensity_combines_te_tm():
    te = {"coeff": np.array([[1 + 1j, 2 + 0j]], dtype=np.complex128)}
    tm = {"coeff": np.array([[0 + 1j, 1 + 0j]], dtype=np.complex128)}
    I = far_field_intensity(te, tm)
    expected = np.abs(te["coeff"]) ** 2 + np.abs(tm["coeff"]) ** 2
    np.testing.assert_allclose(I, expected)


def test_far_field_intensity_from_result_uses_explicit_channel():
    te = {"coeff": np.array([[1 + 0j, 2 + 0j]], dtype=np.complex128)}
    tm = {"coeff": np.array([[3 + 0j, 4 + 0j]], dtype=np.complex128)}
    run = type(
        "Run",
        (),
        {"farfield": type("FF", (), {"scattered_te": te, "scattered_tm": tm})},
    )()
    np.testing.assert_allclose(
        far_field_intensity_from_result(run),
        far_field_intensity(te, tm),
    )


def test_far_field_intensity_from_result_rejects_periodic_placeholder():
    run = type("Run", (), {"periodic": object()})()

    with pytest.raises(ValueError, match="discrete diffraction orders"):
        far_field_intensity_from_result(run)


def test_unpolarized_far_field_intensity_averages_te_tm_channels():
    te = {"coeff": np.array([[1 + 0j, 2 + 0j]], dtype=np.complex128)}
    tm = {"coeff": np.array([[3 + 0j, 4 + 0j]], dtype=np.complex128)}
    ff_te = type("FF", (), {"scattered_te": te, "scattered_tm": tm})()
    ff_tm = type("FF", (), {"scattered_te": tm, "scattered_tm": te})()
    result = type(
        "PolarizationResult",
        (),
        {
            "te": type("Run", (), {"farfield": ff_te})(),
            "tm": type("Run", (), {"farfield": ff_tm})(),
        },
    )()
    np.testing.assert_allclose(
        unpolarized_far_field_intensity(result),
        0.5 * (far_field_intensity(te, tm) + far_field_intensity(tm, te)),
    )


def test_plot_nearfield_panels_channels_returns_expected_axes_shape():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="xy")
    E = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    H = np.zeros_like(E)
    fig, axes = plot_nearfield_panels_channels(
        axis_0,
        axis_1,
        {"TE": (E, H), "TM": (E, H)},
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=0.2, refractive_index=1.5 + 0j)],
        plane="y",
        plane_value=0.0,
    )
    assert axes.shape == (4, 4)
    fig.clf()


def test_plot_nearfield_panels_channels_rejects_empty_mapping():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="xy")
    with pytest.raises(ValueError, match="must not be empty"):
        plot_nearfield_panels_channels(axis_0, axis_1, {}, particles=[])


def test_plot_nearfield_panels_returns_expected_axes_shape():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="xy")
    E = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    H = np.zeros_like(E)
    fig, axes = plot_nearfield_panels(
        axis_0,
        axis_1,
        E,
        H,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=0.2, refractive_index=1.5 + 0j)],
        plane="y",
        plane_value=0.0,
    )
    assert axes.shape == (2, 4)
    plt.close(fig)


def test_plot_nearfield_panels_rejects_invalid_plane():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), indexing="xy")
    E = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    H = np.zeros_like(E)
    with pytest.raises(ValueError, match="plane must be one of"):
        plot_nearfield_panels(axis_0, axis_1, E, H, particles=[], plane="bad")


def test_plot_spheres_can_overlay_layered_shells():
    fig, ax = plt.subplots()
    plot_spheres(
        ax,
        particles=[
            LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(1.0, 2.0),
                layer_refractive_indices=(1.5 + 0j, 1.2 + 0j),
            )
        ],
        plane="z",
        plane_value=0.0,
    )
    assert len(ax.patches) == 2
    plt.close(fig)


def test_plot_spheres_rejects_invalid_plane_and_unsupported_particle():
    fig, ax = plt.subplots()
    sphere = Sphere(position=(0.0, 0.0, 0.0), radius=1.0, refractive_index=1.5 + 0j)
    with pytest.raises(ValueError, match="plane must be one of"):
        plot_spheres(ax, [sphere], plane="bad")
    with pytest.raises(TypeError, match="Unsupported particle type"):
        plot_spheres(ax, [Particle(position=(0.0, 0.0, 0.0))], plane="y")
    plt.close(fig)


def test_plot_spheres_draws_exact_spheroid_slice_and_circumscribing_circle():
    fig, ax = plt.subplots()
    plot_spheres(
        ax,
        particles=[
            Spheroid(
                position=(0.0, 0.0, 0.0),
                equatorial_radius=2.0,
                polar_radius=8.0,
                refractive_index=1.5 + 0j,
            )
        ],
        plane="y",
        plane_value=0.0,
    )
    assert len(ax.patches) == 2
    ellipse = next(p for p in ax.patches if isinstance(p, Ellipse) and not isinstance(p, Circle))
    circle = next(p for p in ax.patches if isinstance(p, Circle))
    np.testing.assert_allclose(ellipse.center, (0.0, 0.0), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose([ellipse.width, ellipse.height], [16.0, 4.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(circle.center, (0.0, 0.0), rtol=0.0, atol=1e-12)
    assert np.isclose(circle.radius, 8.0)
    assert circle.get_linestyle() == "--"
    plt.close(fig)


def test_plot_spheres_scales_off_axis_spheroid_slice():
    fig, ax = plt.subplots()
    plot_spheres(
        ax,
        particles=[
            Spheroid(
                position=(0.0, 0.0, 0.0),
                equatorial_radius=2.0,
                polar_radius=8.0,
                refractive_index=1.5 + 0j,
            )
        ],
        plane="y",
        plane_value=1.0,
    )
    ellipse = next(p for p in ax.patches if isinstance(p, Ellipse) and not isinstance(p, Circle))
    scale = np.sqrt(1.0 - (1.0 / 2.0) ** 2)
    np.testing.assert_allclose(
        [ellipse.width, ellipse.height],
        [16.0 * scale, 4.0 * scale],
        rtol=0.0,
        atol=1e-12,
    )
    plt.close(fig)


def test_plot_spheres_draws_tilted_spheroid_slice_on_offset_plane():
    fig, ax = plt.subplots()
    plot_spheres(
        ax,
        particles=[
            Spheroid(
                position=(0.2, -0.1, 0.3),
                equatorial_radius=2.0,
                polar_radius=8.0,
                refractive_index=1.5 + 0j,
                euler_angles=(0.4, 0.9, -0.3),
            )
        ],
        plane="x",
        plane_value=0.7,
    )
    assert len(ax.patches) == 2
    ellipse = next(p for p in ax.patches if isinstance(p, Ellipse) and not isinstance(p, Circle))
    circle = next(p for p in ax.patches if isinstance(p, Circle))
    assert abs(float(ellipse.angle)) > 1e-6
    assert np.all(np.isfinite(ellipse.center))
    assert ellipse.width > 0.0
    assert ellipse.height > 0.0
    assert circle.radius > 0.0
    plt.close(fig)


def test_plot_spheres_keeps_circumscribing_circle_when_plane_misses_spheroid():
    fig, ax = plt.subplots()
    plot_spheres(
        ax,
        particles=[
            Spheroid(
                position=(0.0, 0.0, 0.0),
                equatorial_radius=2.0,
                polar_radius=8.0,
                refractive_index=1.5 + 0j,
            )
        ],
        plane="y",
        plane_value=7.0,
    )
    assert len(ax.patches) == 1
    circle = ax.patches[0]
    assert isinstance(circle, Circle)
    assert circle.get_linestyle() == "--"
    plt.close(fig)


def test_unpolarized_near_field_intensity_is_incoherent_average():
    E_te = np.array([[[1 + 1j, 2 + 0j, 0 + 0j]]], dtype=np.complex128)
    E_tm = np.array([[[2 + 0j, 0 + 1j, 1 + 0j]]], dtype=np.complex128)
    Iu = unpolarized_near_field_intensity(E_te, E_tm)
    expected = 0.5 * (np.sum(np.abs(E_te) ** 2, axis=-1) + np.sum(np.abs(E_tm) ** 2, axis=-1))
    np.testing.assert_allclose(Iu, expected)


def test_near_field_component_rejects_unknown_component():
    E = np.zeros((2, 2, 3), dtype=np.complex128)
    H = np.zeros_like(E)
    with pytest.raises(ValueError, match="Unsupported component"):
        near_field_component(E, H, "phase Ex")


def test_plot_field_component_uses_center_based_extent_edges():
    axis_0, axis_1 = np.meshgrid(np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0]), indexing="xy")
    F = np.zeros(axis_0.shape, dtype=float)
    fig, ax = plt.subplots()
    im = plot_field_component(ax, axis_0, axis_1, F)
    np.testing.assert_allclose(im.get_extent(), [-0.5, 2.5, 9.5, 11.5], rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_plot_field_component_single_pixel_uses_unit_span_extent():
    axis_0 = np.array([[2.0]], dtype=float)
    axis_1 = np.array([[5.0]], dtype=float)
    fig, ax = plt.subplots()
    im = plot_field_component(ax, axis_0, axis_1, np.array([[1.0]], dtype=float))
    np.testing.assert_allclose(im.get_extent(), [1.5, 2.5, 4.5, 5.5], rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_plot_farfield_hemispheres_and_poynting_smokes():
    beta = np.linspace(0.0, np.pi, 5)
    alpha = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    intensity = np.arange(alpha.size * beta.size, dtype=float).reshape(alpha.size, beta.size)
    fig_ff, axes_ff = plot_farfield_hemispheres(
        beta,
        alpha,
        intensity,
        independent_scales=True,
        vmin_zero=False,
    )
    assert len(axes_ff) == 2
    plt.close(fig_ff)

    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-2, 2, 3), indexing="xy")
    sx = np.ones_like(axis_0)
    sz = -np.ones_like(axis_1)
    intensity_bg = axis_0**2 + axis_1**2
    fig_p, ax_p = plt.subplots()
    out_ax = plot_poynting(
        ax_p,
        axis_0,
        axis_1,
        sx,
        sz,
        intensity=intensity_bg,
        stride=1,
        axis_0_label="x",
        axis_1_label="z",
    )
    assert out_ax is ax_p
    plt.close(fig_p)


def test_plot_farfield_hemispheres_shared_scale_adds_single_colorbar():
    beta = np.linspace(0.0, np.pi, 5)
    alpha = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    intensity = np.arange(alpha.size * beta.size, dtype=float).reshape(alpha.size, beta.size)
    fig, axes = plot_farfield_hemispheres(beta, alpha, intensity, independent_scales=False)
    assert len(axes) == 2
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_farfield_hemispheres_rejects_wrong_intensity_shape():
    beta = np.linspace(0.0, np.pi, 5)
    alpha = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    with np.testing.assert_raises(ValueError):
        plot_farfield_hemispheres(beta, alpha, np.zeros((beta.size, alpha.size), dtype=float))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"field_component": "bad"}, "`field_component` must be one of"),
        ({"phase_component": "bad"}, "`phase_component` must be one of"),
        ({"rowwise_percentile": 0.0}, "`rowwise_percentile` must lie in"),
        ({"plane_values": (0.0, 1.0)}, "`plane_values` must contain exactly three entries"),
    ],
)
def test_plot_source_showcase_slices_validates_public_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        plot_source_showcase_slices(object(), show_progress=False, **kwargs)


def test_plot_source_showcase_slices_returns_3x5_layout(monkeypatch):
    axis_0, axis_1 = np.meshgrid(
        np.linspace(-1.0, 1.0, 4), np.linspace(-2.0, 2.0, 3), indexing="xy"
    )
    E = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    E[..., 0] = 1.0 + 1.0j
    H = np.zeros_like(E)

    def _fake_slice(*args, plane: str, plane_value: float, **kwargs):
        del args, kwargs
        return NearFieldSlice(
            axis_0=axis_0,
            axis_1=axis_1,
            inside=np.zeros(axis_0.shape, dtype=bool),
            field_maps={
                "initial": (E, H),
                "scattered": (E, H),
                "internal": (E, H),
                "total": (E, H),
            },
            plane=str(plane),
            plane_value=float(plane_value),
            axis_0_label="u",
            axis_1_label="v",
        )

    monkeypatch.setattr(
        "pyceles.postprocessing.nearfield.workflows.compute_near_field_slice", _fake_slice
    )
    run = type(
        "Run",
        (),
        {
            "particles": tuple(),
        },
    )()
    fig, axes = plot_source_showcase_slices(
        run,
        field_component="initial",
        plane_values=(0.0, 0.0, 0.0),
        show_progress=False,
    )
    assert axes.shape == (3, 5)
    assert axes[0, 3].axison is True
    assert axes[1, 4].axison is True
    assert axes[2, 3].axison is True
    plt.close(fig)


def test_plot_source_showcase_slices_handles_nonfinite_auto_limits(monkeypatch):
    axis_0, axis_1 = np.meshgrid(
        np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2), indexing="xy"
    )
    nan_map = np.full((*axis_0.shape, 3), np.nan + 0.0j, dtype=np.complex128)
    slice_result = NearFieldSlice(
        axis_0=axis_0,
        axis_1=axis_1,
        inside=np.zeros(axis_0.shape, dtype=bool),
        field_maps={
            "initial": (nan_map, nan_map),
            "scattered": (nan_map, nan_map),
            "internal": (nan_map, nan_map),
            "total": (nan_map, nan_map),
        },
        plane="x",
        plane_value=0.0,
        axis_0_label="u",
        axis_1_label="v",
    )

    monkeypatch.setattr(
        "pyceles.postprocessing.nearfield.workflows.compute_near_field_slice",
        lambda *args, **kwargs: slice_result,
    )
    run = type("Run", (), {"particles": tuple()})()
    fig, axes = plot_source_showcase_slices(run, show_progress=False)
    assert axes.shape == (3, 5)
    plt.close(fig)


def test_plot_nearfield_poynting_overlay_returns_figure_and_axes():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), indexing="xy")
    e = np.zeros((*axis_0.shape, 3), dtype=np.complex128)
    h = np.zeros_like(e)
    e[..., 0] = 1.0 + 0.0j
    h[..., 2] = 1.0 + 0.0j
    fig, ax = plot_nearfield_poynting_overlay(
        axis_0,
        axis_1,
        e,
        h,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=0.25, refractive_index=1.5 + 0j)],
        plane="y",
        plane_value=0.0,
        stride=1,
    )
    assert ax.get_xlabel() == r"$x$"
    assert ax.get_ylabel() == r"$z$"
    plt.close(fig)
