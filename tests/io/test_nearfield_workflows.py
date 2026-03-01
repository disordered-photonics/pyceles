import numpy as np

from pyceles.postprocessing.nearfield import NearFieldComponents
from pyceles.postprocessing.workflows import (
    NearFieldSlice,
    mix_near_field_components,
    mix_near_field_slices,
)


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
    te_map = np.ones(axis_0.shape + (3,), dtype=np.complex128)
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
