import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.io.plotting import (
    far_field_intensity,
    far_field_intensity_from_result,
    near_field_component,
    plot_field_component,
    plot_nearfield_panels_channels,
    plot_source_showcase_slices,
    plot_spheres,
    unpolarized_near_field_intensity,
)
from pyceles.postprocessing.workflows import NearFieldSlice


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


def test_far_field_intensity_from_result_unpolarized_averages_basis():
    te = {"coeff": np.array([[1 + 0j, 2 + 0j]], dtype=np.complex128)}
    tm = {"coeff": np.array([[3 + 0j, 4 + 0j]], dtype=np.complex128)}
    run = type(
        "Run",
        (),
        {
            "farfield": type("FF", (), {"scattered_te": te, "scattered_tm": tm}),
            "farfield_basis": {
                "te": type("FF", (), {"scattered_te": te, "scattered_tm": tm}),
                "tm": type("FF", (), {"scattered_te": tm, "scattered_tm": te}),
            },
        },
    )()
    Iu = far_field_intensity_from_result(run, channel="unpolarized")
    I_te = far_field_intensity(
        run.farfield_basis["te"].scattered_te, run.farfield_basis["te"].scattered_tm
    )
    I_tm = far_field_intensity(
        run.farfield_basis["tm"].scattered_te, run.farfield_basis["tm"].scattered_tm
    )
    np.testing.assert_allclose(Iu, 0.5 * (I_te + I_tm))


def test_plot_nearfield_panels_channels_returns_expected_axes_shape():
    axis_0, axis_1 = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="xy")
    E = np.zeros(axis_0.shape + (3,), dtype=np.complex128)
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


def test_unpolarized_near_field_intensity_is_incoherent_average():
    E_te = np.array([[[1 + 1j, 2 + 0j, 0 + 0j]]], dtype=np.complex128)
    E_tm = np.array([[[2 + 0j, 0 + 1j, 1 + 0j]]], dtype=np.complex128)
    Iu = unpolarized_near_field_intensity(E_te, E_tm)
    expected = 0.5 * (np.sum(np.abs(E_te) ** 2, axis=-1) + np.sum(np.abs(E_tm) ** 2, axis=-1))
    np.testing.assert_allclose(Iu, expected)


def test_plot_field_component_uses_center_based_extent_edges():
    axis_0, axis_1 = np.meshgrid(np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0]), indexing="xy")
    F = np.zeros(axis_0.shape, dtype=float)
    fig, ax = plt.subplots()
    im = plot_field_component(ax, axis_0, axis_1, F)
    np.testing.assert_allclose(im.get_extent(), [-0.5, 2.5, 9.5, 11.5], rtol=0.0, atol=1e-12)
    plt.close(fig)


def test_plot_source_showcase_slices_returns_3x5_layout(monkeypatch):
    axis_0, axis_1 = np.meshgrid(
        np.linspace(-1.0, 1.0, 4), np.linspace(-2.0, 2.0, 3), indexing="xy"
    )
    E = np.zeros(axis_0.shape + (3,), dtype=np.complex128)
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

    monkeypatch.setattr("pyceles.postprocessing.workflows.compute_near_field_slice", _fake_slice)
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
        channel="mixed",
        plane_values=(0.0, 0.0, 0.0),
        show_progress=False,
    )
    assert axes.shape == (3, 5)
    assert axes[0, 3].axison is True
    assert axes[1, 4].axison is True
    assert axes[2, 3].axison is True
    plt.close(fig)
