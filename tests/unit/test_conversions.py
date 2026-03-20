from __future__ import annotations

import numpy as np

from pyceles.core.conversions import (
    pwp_to_svwf_regular,
    svwf_outgoing_to_pwp,
    svwf_regular_to_pwp,
)
from pyceles.core.indexing import n_modes
from pyceles.core.translation import translation_ab5_table, translation_block_regular


def _angular_grid() -> tuple[np.ndarray, np.ndarray]:
    polar = np.linspace(0.0, np.pi, 181)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    return polar, azimuthal


def _phase_factor(pwp: dict, shift: np.ndarray, *, sign: float) -> np.ndarray:
    dot = pwp["kx"] * shift[0] + pwp["ky"] * shift[1] + pwp["kz"] * shift[2]
    return np.exp(1j * sign * dot)


def _with_coeff_phase(pwp: dict, phase: np.ndarray) -> dict:
    out = dict(pwp)
    out["coeff"] = np.asarray(pwp["coeff"], dtype=np.complex128) * np.asarray(
        phase, dtype=np.complex128
    )
    return out


def test_outgoing_pattern_translation_is_phase_shift_in_pwp_space() -> None:
    """Outgoing SVWF pattern translation law in PWP form.

    Reference context:
    Dufva et al., PIER B 4 (2008) 79-99, translational addition-theorem setting
    for SVWF coefficient transport.
    """
    lmax = 3
    k = 2.0 * np.pi / 550.0
    nm = n_modes(lmax)
    rng = np.random.default_rng(11)

    coeffs = rng.standard_normal((1, nm)) + 1j * rng.standard_normal((1, nm))
    shift = np.array([120.0, 35.0, -60.0], dtype=float)
    pos_origin = np.array([[0.0, 0.0, 0.0]], dtype=float)
    pos_shifted = shift[None, :]
    polar, azimuthal = _angular_grid()

    pwp0_te, pwp0_tm = svwf_outgoing_to_pwp(
        pos_origin,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    pwp1_te, pwp1_tm = svwf_outgoing_to_pwp(
        pos_shifted,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )

    phase = _phase_factor(pwp0_te, shift, sign=-1.0)
    np.testing.assert_allclose(pwp1_te["coeff"], pwp0_te["coeff"] * phase, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(pwp1_tm["coeff"], pwp0_tm["coeff"] * phase, rtol=2e-12, atol=2e-12)


def test_regular_pattern_translation_is_phase_shift_in_pwp_space() -> None:
    """Incoming/regular SVWF pattern translation law in PWP form.

    Reference context:
    Dufva et al., PIER B 4 (2008) 79-99, translational addition-theorem setting
    for SVWF coefficient transport.
    """
    lmax = 3
    k = 2.0 * np.pi / 550.0
    nm = n_modes(lmax)
    rng = np.random.default_rng(19)

    coeffs = rng.standard_normal((1, nm)) + 1j * rng.standard_normal((1, nm))
    shift = np.array([95.0, -40.0, 75.0], dtype=float)
    pos_origin = np.array([[0.0, 0.0, 0.0]], dtype=float)
    pos_shifted = shift[None, :]
    polar, azimuthal = _angular_grid()

    pwp0_te, pwp0_tm = svwf_regular_to_pwp(
        pos_origin,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )
    pwp1_te, pwp1_tm = svwf_regular_to_pwp(
        pos_shifted,
        coeffs,
        k=k,
        lmax=lmax,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )

    phase = _phase_factor(pwp0_te, shift, sign=-1.0)
    np.testing.assert_allclose(pwp1_te["coeff"], pwp0_te["coeff"] * phase, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(pwp1_tm["coeff"], pwp0_tm["coeff"] * phase, rtol=2e-12, atol=2e-12)


def test_regular_pwp_phase_shift_matches_exact_regular_translation() -> None:
    """Pattern-space phase mediation agrees with exact regular translation.

    References:
    Dufva et al., PIER B 4 (2008) 79-99.
    Exact translation benchmark is evaluated with the same addition-theorem
    coefficient framework used in the low-order oracle tests.
    """
    lmax = 3
    k = 2.0 * np.pi / 550.0
    nm = n_modes(lmax)
    rng = np.random.default_rng(23)

    coeff = rng.standard_normal((1, nm)) + 1j * rng.standard_normal((1, nm))
    shift = np.array([120.0, 35.0, -60.0], dtype=float)
    origin = np.array([[0.0, 0.0, 0.0]], dtype=float)
    polar, azimuthal = _angular_grid()

    pwp_te, pwp_tm = svwf_regular_to_pwp(
        origin,
        coeff,
        k=k,
        lmax=lmax,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
    )

    phase = _phase_factor(pwp_te, shift, sign=1.0)
    pwp_te_shifted = _with_coeff_phase(pwp_te, phase)
    pwp_tm_shifted = _with_coeff_phase(pwp_tm, phase)

    from_phase = pwp_to_svwf_regular(
        origin,
        lmax,
        k=k,
        pwp_te=pwp_te_shifted,
        pwp_tm=pwp_tm_shifted,
    )[0]
    from_position = pwp_to_svwf_regular(
        shift[None, :],
        lmax,
        k=k,
        pwp_te=pwp_te,
        pwp_tm=pwp_tm,
    )[0]
    np.testing.assert_allclose(from_phase, from_position, rtol=2e-12, atol=2e-12)

    exact = translation_block_regular(lmax, k, shift, ab5=translation_ab5_table(lmax)) @ coeff[0]
    np.testing.assert_allclose(from_phase, exact, rtol=2e-4, atol=2e-4)
