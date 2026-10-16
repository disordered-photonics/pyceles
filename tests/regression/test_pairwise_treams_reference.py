"""Selected TREAMS parity-basis matrix entries, converted to CELES ordering.

References: TREAMS 0.4.8.dev1, revision 1f5d0d6. Nonmagnetic spheres at vacuum
wavelength 600, with exp(-i*omega*t) and outgoing h^(1). For each mode,
c_treams=D*c_celes, where D=-i*sqrt(pi)*(-1)^m for positive m and -i*sqrt(pi)
otherwise. The inline W entries include that change of basis.
"""

import numpy as np
import pytest

from pyceles._optional import asnumpy
from pyceles.core.indexing import index_vswf, n_modes
from pyceles.core.operators import prepare_matvec
from pyceles.core.particles import spheres_from_arrays

pytestmark = pytest.mark.reference


@pytest.mark.parametrize(
    ("host", "w_ref", "t_ref"),
    [
        (
            1.33,
            (
                -0.13297635613285808 - 0.008853007256746134j,
                -0.03284403863672149 + 0.08050663058775974j,
                0.018613292531111544 + 0.061663521848497924j,
                -0.05158587264418905 + 0.007404488838917166j,
            ),
            (
                -0.023104548218111427 + 0.09493203885895896j,
                -2.006124533244557e-6 + 1.094310233767443e-5j,
                -3.048443862726251e-6 + 2.6022529926404725e-5j,
                -0.0005189147875429463 + 0.0026849809378243217j,
            ),
        ),
        (
            1.33 + 0.01j,
            (
                -0.12544911383085652 - 0.007389273281345088j,
                -0.030486337398880514 + 0.07656525611030436j,
                0.01854029376832629 + 0.059235354652420316j,
                -0.048106751429781894 + 0.007302008379137123j,
            ),
            (
                -0.023110017701301933 + 0.09468535673306967j,
                -2.2777892016788327e-6 + 1.0868937741945384e-5j,
                -3.937632327160509e-6 + 2.5846009935937597e-5j,
                -0.0005432962757986664 + 0.0026747988795982293j,
            ),
        ),
        (
            1.33 + 0.5j,
            (
                -0.006247726831460682 + 0.0020210373884519575j,
                6.24003065298499e-5 + 0.005466891599234698j,
                0.006673378995844997 + 0.0047890389007442784j,
                -0.001251157164509147 + 0.0008549429727712319j,
            ),
            (
                -0.043335563827598574 + 0.13698208784779536j,
                -1.7988858626444878e-5 + 2.9813985760880384e-6j,
                -4.8028158500631384e-5 - 9.650000938934539e-6j,
                -0.0024269876991445103 + 0.0031219413310707936j,
            ),
        ),
    ],
    ids=["real", "weak_loss", "strong_loss"],
)
@pytest.mark.parametrize("compute_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    ("backend", "cached"),
    [("numpy", False), ("numpy", True), pytest.param("cupy", False, marks=pytest.mark.gpu)],
)
def test_selected_pairwise_and_particle_responses_against_treams(
    host, w_ref, t_ref, compute_dtype, backend, cached, request
):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    positions = np.array([[-312, -108, -132], [246, -186, -48], [-114, 288, 126], [198, 162, 264]])
    operator = prepare_matvec(
        lmax=3,
        k=2 * np.pi / 600 * host,
        n_medium=host,
        particles=spheres_from_arrays(
            positions=positions,
            radii=np.array([45, 57, 39, 51]),
            refractive_indices=2 + 0.1j,
        ),
        radial_lut_dr=600 / 4096,
        operator_dtype=compute_dtype,
        backend=backend,
        cache_translation_blocks=cached,
    )
    # (particle, l, m, tau): span both polarizations and degrees 1 through 3.
    targets = [(1, 1, 0, 2), (2, 2, -1, 1), (3, 3, 2, 2), (0, 1, -1, 1)]
    sources = [(0, 1, 1, 1), (0, 1, 0, 1), (1, 2, -2, 1), (3, 3, 0, 2)]
    rows = np.array([p * n_modes(3) + index_vswf(l, m, tau, 3) for p, l, m, tau in targets])
    columns = np.array([p * n_modes(3) + index_vswf(l, m, tau, 3) for p, l, m, tau in sources])
    basis = np.zeros((4 * n_modes(3), 4), dtype=compute_dtype)
    basis[columns, np.arange(4)] = 1
    dual = np.zeros_like(basis)
    dual[rows, np.arange(4)] = 1
    t = asnumpy(operator.particle_t.apply(np.ones(basis.shape[0], dtype=compute_dtype)))[rows]
    np.testing.assert_allclose(t, t_ref, rtol=3e-7, atol=1e-12)
    a_ref = -np.asarray(t_ref) * w_ref
    for i, (row, column) in enumerate(zip(rows, columns, strict=True)):
        np.testing.assert_allclose(
            asnumpy(operator.apply_W(basis[:, i]))[row], w_ref[i], rtol=2e-6, atol=1e-12
        )
        np.testing.assert_allclose(
            asnumpy(operator.apply_A(basis[:, i]))[row], a_ref[i], rtol=2e-6, atol=1e-12
        )
        np.testing.assert_allclose(
            asnumpy(operator.apply_adjoint(dual[:, i]))[column],
            a_ref[i].conjugate(),
            rtol=2e-6,
            atol=1e-12,
        )
    if backend == "cupy":
        np.testing.assert_allclose(
            asnumpy(operator.apply_W(basis))[rows, np.arange(4)], w_ref, rtol=2e-6, atol=1e-12
        )
        np.testing.assert_allclose(
            asnumpy(operator.apply_adjoint(dual))[columns, np.arange(4)],
            a_ref.conjugate(),
            rtol=2e-6,
            atol=1e-12,
        )
