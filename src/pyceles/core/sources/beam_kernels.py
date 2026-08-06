from __future__ import annotations

import numpy as np

from ..angular import beam_axis_and_frame, trapezoidal_weights
from ..plane_wave_spectrum import PlaneWaveSpectrum
from ..polarization import (
    normalize_global_polarization_vector,
    project_global_cartesian_to_te_tm,
    pure_polarization_label,
)
from .base import Polarization, is_normal_incidence
from .common import _as_float_triplet, _laguerre_profile_factor


def _combine_jones_spectra(
    te_basis: PlaneWaveSpectrum,
    tm_basis: PlaneWaveSpectrum,
    *,
    a_te: complex,
    a_tm: complex,
) -> PlaneWaveSpectrum:
    """Coherently combine spectra generated for pure TE/TM source states."""
    dtype = np.result_type(
        te_basis.coeff_te,
        te_basis.coeff_tm,
        tm_basis.coeff_te,
        tm_basis.coeff_tm,
        a_te,
        a_tm,
    )
    return te_basis.linear_combination(
        tm_basis,
        weight_self=a_te,
        weight_other=a_tm,
        dtype=dtype,
    )


def _gaussian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> PlaneWaveSpectrum:
    """Evaluate Gaussian-beam TE/TM angular-spectrum coefficients on a grid."""
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_basis = _gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_basis = _gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TM",
            )
            return _combine_jones_spectra(
                te_basis,
                tm_basis,
                a_te=a_te,
                a_tm=a_tm,
            )
        polarization_override = pure

    beta = np.asarray(polar_angles, float)
    alpha = np.asarray(azimuthal_angles, float)
    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb
    sx = sb * ca
    sy = sb * sa
    sz = cb
    s = np.stack([sx, sy, sz], axis=2)
    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)
    rho2_l = np.maximum(0.0, 1.0 - sz_l**2)
    alpha_l = np.arctan2(sy_l, sx_l)
    sin_beta_l = np.sqrt(rho2_l)
    cos_beta_l = sz_l
    rg = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), float)
    e0 = float(getattr(beam, "amplitude", 1.0))
    w = float(getattr(beam, "beam_width", np.inf))
    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()
    alpha_pol = float(getattr(beam, "azimuthal_angle", 0.0))
    if pol != "te":
        alpha_pol -= np.pi / 2.0
    phase = np.exp(-1j * (kx * rg[0] + ky * rg[1] + kz * rg[2]))
    pref = e0 * (k**2) * (w**2) / (4.0 * np.pi)
    envelope = pref * cos_beta_l * np.exp(-(w**2) / 4.0 * (k**2) * rho2_l)
    envelope = envelope * (cos_beta_l > 0.0)
    g_te_l = np.cos(alpha_l - alpha_pol) * envelope
    g_tm_l = np.sin(alpha_l - alpha_pol) * envelope
    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)
    sin_alpha_l = np.sin(alpha_l)
    cos_alpha_l = np.cos(alpha_l)
    ephi_l = (-sin_alpha_l)[..., None] * u[None, None, :] + cos_alpha_l[..., None] * v[
        None, None, :
    ]
    etheta_l = (
        (cos_beta_l * cos_alpha_l)[..., None] * u[None, None, :]
        + (cos_beta_l * sin_alpha_l)[..., None] * v[None, None, :]
        - sin_beta_l[..., None] * n0[None, None, :]
    )
    m11 = np.einsum("abi,abi->ab", ephi_g, ephi_l)
    m12 = np.einsum("abi,abi->ab", ephi_g, etheta_l)
    m21 = np.einsum("abi,abi->ab", etheta_g, ephi_l)
    m22 = np.einsum("abi,abi->ab", etheta_g, etheta_l)
    coeff_te = (m11 * g_te_l + m12 * g_tm_l) * phase
    coeff_tm = (m21 * g_te_l + m22 * g_tm_l) * phase
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def _laguerre_gaussian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> PlaneWaveSpectrum:
    """Evaluate Maxwell-consistent collimated LG TE/TM angular-spectrum coefficients."""
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_basis = _laguerre_gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_basis = _laguerre_gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TM",
            )
            return _combine_jones_spectra(
                te_basis,
                tm_basis,
                a_te=a_te,
                a_tm=a_tm,
            )
        polarization_override = pure

    beta = np.asarray(polar_angles, float).reshape(-1)
    alpha = np.asarray(azimuthal_angles, float).reshape(-1)
    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb
    sx = sb * ca
    sy = sb * sa
    sz = cb
    s = np.stack([sx, sy, sz], axis=2)
    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)
    rho2_l = np.maximum(0.0, 1.0 - sz_l**2)
    alpha_l = np.arctan2(sy_l, sx_l)
    sin_beta_l = np.sqrt(rho2_l)
    cos_beta_l = np.clip(sz_l, -1.0, 1.0)
    rg = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), float)
    e0 = float(getattr(beam, "amplitude", 1.0))
    w = float(getattr(beam, "beam_width", np.inf))
    p = int(getattr(beam, "radial_order_p", 0))
    l = int(getattr(beam, "azimuthal_order_l", 0))
    az_phase = float(getattr(beam, "azimuthal_phase", 0.0))
    x_arg = 0.5 * k * w * sin_beta_l
    ppl = _laguerre_profile_factor(
        radial_argument=x_arg,
        radial_order_p=p,
        azimuthal_order_l=l,
    )
    pref = e0 * (k**2) * (w**2) / (4.0 * np.pi)
    envelope = pref * cos_beta_l * ppl
    envelope = envelope * (cos_beta_l > 0.0)
    envelope = envelope * np.exp(1j * (l * alpha_l + az_phase))
    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()
    alpha_pol = float(getattr(beam, "azimuthal_angle", 0.0))
    if pol != "te":
        alpha_pol -= np.pi / 2.0
    g_te_l = np.cos(alpha_l - alpha_pol) * envelope
    g_tm_l = np.sin(alpha_l - alpha_pol) * envelope
    phase = np.exp(-1j * (kx * rg[0] + ky * rg[1] + kz * rg[2]))
    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)
    sin_alpha_l = np.sin(alpha_l)
    cos_alpha_l = np.cos(alpha_l)
    ephi_l = (-sin_alpha_l)[..., None] * u[None, None, :] + cos_alpha_l[..., None] * v[
        None, None, :
    ]
    etheta_l = (
        (cos_beta_l * cos_alpha_l)[..., None] * u[None, None, :]
        + (cos_beta_l * sin_alpha_l)[..., None] * v[None, None, :]
        - sin_beta_l[..., None] * n0[None, None, :]
    )
    m11 = np.einsum("abi,abi->ab", ephi_g, ephi_l)
    m12 = np.einsum("abi,abi->ab", ephi_g, etheta_l)
    m21 = np.einsum("abi,abi->ab", etheta_g, ephi_l)
    m22 = np.einsum("abi,abi->ab", etheta_g, etheta_l)
    coeff_te = (m11 * g_te_l + m12 * g_tm_l) * phase
    coeff_tm = (m21 * g_te_l + m22 * g_tm_l) * phase
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def _focused_laguerre_geometry(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return shared angular/basis/envelope tensors for focused-LG sources."""
    beta = np.asarray(polar_angles, float).reshape(-1)
    alpha = np.asarray(azimuthal_angles, float).reshape(-1)
    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb
    sx = sb * ca
    sy = sb * sa
    sz = cb
    s = np.stack([sx, sy, sz], axis=2)
    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)
    rho2_l = np.maximum(0.0, 1.0 - sz_l**2)
    alpha_l = np.arctan2(sy_l, sx_l)
    sin_beta_l = np.sqrt(rho2_l)
    cos_beta_l = np.clip(sz_l, -1.0, 1.0)
    rg = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), float)
    e0 = float(getattr(beam, "amplitude", 1.0))
    w = float(getattr(beam, "beam_width", np.inf))
    f = float(getattr(beam, "focal_length", np.inf))
    p = int(getattr(beam, "radial_order_p", 0))
    l = int(getattr(beam, "azimuthal_order_l", 0))
    az_phase = float(getattr(beam, "azimuthal_phase", 0.0))
    n_medium_real = float(np.real(complex(getattr(beam, "medium_n", 1.0 + 0j))))
    sin_alpha_max = float(getattr(beam, "numerical_aperture", 0.0)) / n_medium_real
    apodize = bool(getattr(beam, "sine_condition_apodization", True))
    x_arg = (f / w) * sin_beta_l
    ppl = _laguerre_profile_factor(
        radial_argument=x_arg,
        radial_order_p=p,
        azimuthal_order_l=l,
    )
    aperture_mask = (cos_beta_l > 0.0) & (sin_beta_l <= (sin_alpha_max + 1e-12))
    apod = np.sqrt(np.clip(cos_beta_l, 0.0, None)) if apodize else np.ones_like(cos_beta_l)
    envelope = e0 * ppl * apod * aperture_mask
    envelope = envelope * np.exp(1j * (l * alpha_l + az_phase))
    phase = np.exp(-1j * (kx * rg[0] + ky * rg[1] + kz * rg[2]))
    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "alpha_l": alpha_l,
        "sin_beta_l": sin_beta_l,
        "cos_beta_l": cos_beta_l,
        "u": u,
        "v": v,
        "n0": n0,
        "ephi_g": ephi_g,
        "etheta_g": etheta_g,
        "envelope": envelope,
        "phase": phase,
    }


def _focused_laguerre_gaussian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> PlaneWaveSpectrum:
    """Evaluate Debye/aplanatic-focused LG TE/TM angular-spectrum coefficients."""
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_basis = _focused_laguerre_gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_basis = _focused_laguerre_gaussian_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TM",
            )
            return _combine_jones_spectra(
                te_basis,
                tm_basis,
                a_te=a_te,
                a_tm=a_tm,
            )
        polarization_override = pure

    geom = _focused_laguerre_geometry(
        beam=beam,
        k=float(k),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    alpha = np.asarray(geom["alpha"], dtype=float)
    beta = np.asarray(geom["beta"], dtype=float)
    kx = np.asarray(geom["kx"], dtype=float)
    ky = np.asarray(geom["ky"], dtype=float)
    kz = np.asarray(geom["kz"], dtype=float)
    alpha_l = np.asarray(geom["alpha_l"], dtype=float)
    sin_beta_l = np.asarray(geom["sin_beta_l"], dtype=float)
    cos_beta_l = np.asarray(geom["cos_beta_l"], dtype=float)
    u = np.asarray(geom["u"], dtype=float)
    v = np.asarray(geom["v"], dtype=float)
    n0 = np.asarray(geom["n0"], dtype=float)
    ephi_g = np.asarray(geom["ephi_g"], dtype=float)
    etheta_g = np.asarray(geom["etheta_g"], dtype=float)
    envelope = np.asarray(geom["envelope"], dtype=np.complex128)
    phase = np.asarray(geom["phase"], dtype=np.complex128)
    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()
    alpha_pol = float(getattr(beam, "azimuthal_angle", 0.0))
    if pol != "te":
        alpha_pol -= np.pi / 2.0
    g_te_l = np.cos(alpha_l - alpha_pol) * envelope
    g_tm_l = np.sin(alpha_l - alpha_pol) * envelope
    sin_alpha_l = np.sin(alpha_l)
    cos_alpha_l = np.cos(alpha_l)
    ephi_l = (-sin_alpha_l)[..., None] * u[None, None, :] + cos_alpha_l[..., None] * v[
        None, None, :
    ]
    etheta_l = (
        (cos_beta_l * cos_alpha_l)[..., None] * u[None, None, :]
        + (cos_beta_l * sin_alpha_l)[..., None] * v[None, None, :]
        - sin_beta_l[..., None] * n0[None, None, :]
    )
    m11 = np.einsum("abi,abi->ab", ephi_g, ephi_l)
    m12 = np.einsum("abi,abi->ab", ephi_g, etheta_l)
    m21 = np.einsum("abi,abi->ab", etheta_g, ephi_l)
    m22 = np.einsum("abi,abi->ab", etheta_g, etheta_l)
    coeff_te = (m11 * g_te_l + m12 * g_tm_l) * phase
    coeff_tm = (m21 * g_te_l + m22 * g_tm_l) * phase
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def _focused_laguerre_cartesian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
) -> PlaneWaveSpectrum:
    """Focused-LG spectrum with one lab-frame polarization projected per ray."""
    geom = _focused_laguerre_geometry(
        beam=beam,
        k=float(k),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    alpha = np.asarray(geom["alpha"], dtype=float)
    beta = np.asarray(geom["beta"], dtype=float)
    kx = np.asarray(geom["kx"], dtype=float)
    ky = np.asarray(geom["ky"], dtype=float)
    kz = np.asarray(geom["kz"], dtype=float)
    sx = np.asarray(geom["sx"], dtype=float)
    sy = np.asarray(geom["sy"], dtype=float)
    sz = np.asarray(geom["sz"], dtype=float)
    ephi_g = np.asarray(geom["ephi_g"], dtype=float)
    etheta_g = np.asarray(geom["etheta_g"], dtype=float)
    envelope = np.asarray(geom["envelope"], dtype=np.complex128)
    phase = np.asarray(geom["phase"], dtype=np.complex128)
    g_te, g_tm = project_global_cartesian_to_te_tm(
        global_polarization=beam.global_polarization,
        sx=sx,
        sy=sy,
        sz=sz,
        ephi_g=ephi_g,
        etheta_g=etheta_g,
    )
    coeff_te = envelope * g_te * phase
    coeff_tm = envelope * g_tm * phase
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def _bessel_ring_beta_kernel(beta: np.ndarray, beta0: float) -> np.ndarray:
    """Return a narrow beta-kernel approximating delta(beta-beta0)."""
    b = np.asarray(beta, dtype=float).reshape(-1)
    if b.size < 2:
        raise ValueError("`polar_angles` must contain at least two samples for BesselBeam.")
    if float(beta0) < float(b[0]) or float(beta0) > float(b[-1]):
        raise ValueError(
            f"`cone_angle`={beta0!r} must lie within supplied polar grid range "
            f"[{float(b[0])!r}, {float(b[-1])!r}]."
        )
    wb = trapezoidal_weights(b) * np.sin(b)
    ker = np.zeros_like(b, dtype=float)
    atol = 1e-14
    if np.isclose(beta0, b[0], atol=atol, rtol=0.0):
        if wb[0] <= 0.0:
            raise ValueError(
                "BesselBeam cone intersects beta endpoint with zero quadrature weight."
            )
        ker[0] = 1.0 / wb[0]
        return ker
    if np.isclose(beta0, b[-1], atol=atol, rtol=0.0):
        if wb[-1] <= 0.0:
            raise ValueError(
                "BesselBeam cone intersects beta endpoint with zero quadrature weight."
            )
        ker[-1] = 1.0 / wb[-1]
        return ker
    j1 = int(np.searchsorted(b, beta0, side="left"))
    j1 = min(max(1, j1), b.size - 1)
    j0 = j1 - 1
    b0 = float(b[j0])
    b1 = float(b[j1])
    if not (b1 > b0):
        raise ValueError("`polar_angles` must be strictly increasing for BesselBeam.")
    t = (float(beta0) - b0) / (b1 - b0)
    p0 = 1.0 - t
    p1 = t
    if wb[j0] <= 0.0 or wb[j1] <= 0.0:
        raise ValueError(
            "BesselBeam cone too close to beta endpoints for current polar grid; "
            "use a denser grid away from 0/pi."
        )
    ker[j0] = p0 / wb[j0]
    ker[j1] = p1 / wb[j1]
    return ker


def _bessel_add_beta_spike(
    *,
    kernel: np.ndarray,
    beta: np.ndarray,
    wb: np.ndarray,
    beta_star: float,
    scale: float,
) -> None:
    """Deposit one weighted beta spike onto the discrete beta quadrature grid."""
    if scale == 0.0:
        return
    b = np.asarray(beta, dtype=float).reshape(-1)
    atol = 1e-14
    if np.isclose(beta_star, b[0], atol=atol, rtol=0.0):
        if wb[0] <= 0.0:
            raise ValueError(
                "Tilted Bessel ring intersects beta endpoint with zero quadrature weight."
            )
        kernel[0] += scale / wb[0]
        return
    if np.isclose(beta_star, b[-1], atol=atol, rtol=0.0):
        if wb[-1] <= 0.0:
            raise ValueError(
                "Tilted Bessel ring intersects beta endpoint with zero quadrature weight."
            )
        kernel[-1] += scale / wb[-1]
        return
    j1 = int(np.searchsorted(b, beta_star, side="left"))
    j1 = min(max(1, j1), b.size - 1)
    j0 = j1 - 1
    b0 = float(b[j0])
    b1 = float(b[j1])
    if not (b1 > b0):
        raise ValueError("`polar_angles` must be strictly increasing for BesselBeam.")
    if wb[j0] <= 0.0 or wb[j1] <= 0.0:
        raise ValueError(
            "Tilted Bessel ring intersects beta samples with zero quadrature weight; "
            "use a denser polar grid away from 0/pi."
        )
    t = (float(beta_star) - b0) / (b1 - b0)
    p0 = 1.0 - t
    p1 = t
    kernel[j0] += scale * p0 / wb[j0]
    kernel[j1] += scale * p1 / wb[j1]


def _bessel_beta_roots_for_alpha(
    *,
    alpha_value: float,
    polar_angle: float,
    azimuthal_angle: float,
    cone_cos: float,
) -> list[tuple[float, float]]:
    """Return `(beta_root, jacobian_weight)` intersections for one alpha slice."""
    theta0 = float(polar_angle)
    phi0 = float(azimuthal_angle)
    d_alpha = float(alpha_value - phi0)
    a = float(np.sin(theta0) * np.cos(d_alpha))
    b = float(np.cos(theta0))
    r = float(np.hypot(a, b))
    if r <= 1e-15:
        return []
    c = float(cone_cos)
    if abs(c) > r + 1e-12:
        return []
    ratio = float(np.clip(c / r, -1.0, 1.0))
    gamma = float(np.arccos(ratio))
    delta = float(np.arctan2(a, b))
    roots: list[tuple[float, float]] = []
    for base in (delta - gamma, delta + gamma):
        for k in (-1, 0, 1):
            beta_star = float(base + 2.0 * np.pi * k)
            if beta_star < -1e-12 or beta_star > np.pi + 1e-12:
                continue
            beta_star = float(np.clip(beta_star, 0.0, np.pi))
            fp = float(a * np.cos(beta_star) - b * np.sin(beta_star))
            if abs(fp) <= 1e-14:
                continue
            w = float(np.sin(np.arccos(abs(c))) / abs(fp))
            roots.append((beta_star, w))
    deduped: list[tuple[float, float]] = []
    for beta_star, w in sorted(roots, key=lambda t: t[0]):
        if deduped and abs(beta_star - deduped[-1][0]) < 1e-10:
            deduped[-1] = (deduped[-1][0], deduped[-1][1] + w)
        else:
            deduped.append((beta_star, w))
    return deduped


def _bessel_tilted_ring_kernel(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    polar_angle: float,
    azimuthal_angle: float,
    cone_angle: float,
    forward_only: bool,
) -> np.ndarray:
    """Return sparse `(Na,Nb)` ring kernel for a tilted Bessel cone."""
    alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)
    wb = trapezoidal_weights(beta_arr) * np.sin(beta_arr)
    ring = np.zeros((alpha_arr.size, beta_arr.size), dtype=float)
    cone_cos = float(np.cos(float(cone_angle)))
    cone_signs = (1.0,) if bool(forward_only) else (1.0, -1.0)
    for ia, alpha_value in enumerate(alpha_arr):
        row = ring[ia]
        for sign in cone_signs:
            roots = _bessel_beta_roots_for_alpha(
                alpha_value=float(alpha_value),
                polar_angle=float(polar_angle),
                azimuthal_angle=float(azimuthal_angle),
                cone_cos=sign * cone_cos,
            )
            for beta_star, w in roots:
                _bessel_add_beta_spike(
                    kernel=row,
                    beta=beta_arr,
                    wb=wb,
                    beta_star=beta_star,
                    scale=float(w),
                )
    return ring


def _bessel_ring_geometry(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build common ring-spectrum geometry for Bessel-like sources."""
    beta = np.asarray(polar_angles, dtype=float).reshape(-1)
    alpha = np.asarray(azimuthal_angles, dtype=float).reshape(-1)
    agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
    sb = np.sin(bgrid)
    cb = np.cos(bgrid)
    ca = np.cos(agrid)
    sa = np.sin(agrid)
    kx = k * sb * ca
    ky = k * sb * sa
    kz = k * cb
    n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
    s = np.stack([sb * ca, sb * sa, cb], axis=2)
    sx_l = np.einsum("abi,i->ab", s, u)
    sy_l = np.einsum("abi,i->ab", s, v)
    sz_l = np.einsum("abi,i->ab", s, n0)
    alpha_l = np.arctan2(sy_l, sx_l)
    cos_beta_l = np.clip(sz_l, -1.0, 1.0)
    sin_beta_l = np.sqrt(np.maximum(0.0, 1.0 - cos_beta_l**2))
    if is_normal_incidence(float(beam.polar_angle)):
        ring_1d = _bessel_ring_beta_kernel(beta, float(beam.cone_angle))
        if not bool(beam.forward_only):
            ring_1d = ring_1d + _bessel_ring_beta_kernel(
                beta, float(np.pi - float(beam.cone_angle))
            )
        ring = np.broadcast_to(ring_1d[None, :], (alpha.size, beta.size))
    else:
        ring = _bessel_tilted_ring_kernel(
            alpha=alpha,
            beta=beta,
            polar_angle=float(beam.polar_angle),
            azimuthal_angle=float(beam.azimuthal_angle),
            cone_angle=float(beam.cone_angle),
            forward_only=bool(beam.forward_only),
        )
    m = int(beam.order_m)
    az_phase = float(beam.azimuthal_phase)
    phase_mode = np.exp(1j * (m * alpha_l + az_phase))
    cx, cy, cz = _as_float_triplet("center", beam.center)
    phase_center = np.exp(-1j * (kx * cx + ky * cy + kz * cz))
    envelope = float(beam.amplitude) * phase_mode * phase_center * ring
    ephi_g = np.stack([-sa, ca, np.zeros_like(agrid)], axis=2)
    etheta_g = np.stack([cb * ca, cb * sa, -sb], axis=2)
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "sx": sb * ca,
        "sy": sb * sa,
        "sz": cb,
        "alpha_l": alpha_l,
        "cos_beta_l": cos_beta_l,
        "sin_beta_l": sin_beta_l,
        "ephi_g": ephi_g,
        "etheta_g": etheta_g,
        "u": u,
        "v": v,
        "n0": n0,
        "envelope": envelope,
    }


def _bessel_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    polarization_override: Polarization | None = None,
) -> PlaneWaveSpectrum:
    """Evaluate exact non-paraxial Bessel-beam ring spectrum on one alpha-beta grid."""
    if polarization_override is None:
        a_te, a_tm = beam.jones_coefficients()
        pure = pure_polarization_label(a_te, a_tm)
        if pure is None:
            te_basis = _bessel_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TE",
            )
            tm_basis = _bessel_angular_spectrum_coeffs(
                beam=beam,
                k=k,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
                polarization_override="TM",
            )
            return _combine_jones_spectra(
                te_basis,
                tm_basis,
                a_te=a_te,
                a_tm=a_tm,
            )
        polarization_override = pure
    geom = _bessel_ring_geometry(
        beam=beam,
        k=float(k),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    alpha = np.asarray(geom["alpha"], dtype=float)
    beta = np.asarray(geom["beta"], dtype=float)
    kx = np.asarray(geom["kx"], dtype=float)
    ky = np.asarray(geom["ky"], dtype=float)
    kz = np.asarray(geom["kz"], dtype=float)
    alpha_l = np.asarray(geom["alpha_l"], dtype=float)
    cos_beta_l = np.asarray(geom["cos_beta_l"], dtype=float)
    sin_beta_l = np.asarray(geom["sin_beta_l"], dtype=float)
    ephi_g = np.asarray(geom["ephi_g"], dtype=float)
    etheta_g = np.asarray(geom["etheta_g"], dtype=float)
    u = np.asarray(geom["u"], dtype=float)
    v = np.asarray(geom["v"], dtype=float)
    n0 = np.asarray(geom["n0"], dtype=float)
    envelope = np.asarray(geom["envelope"], dtype=np.complex128)
    pol = str(polarization_override or getattr(beam, "polarization", "TE")).lower()
    if pol == "te":
        g_te_l = envelope
        g_tm_l = np.zeros_like(envelope)
    else:
        g_te_l = np.zeros_like(envelope)
        g_tm_l = envelope
    sin_alpha_l = np.sin(alpha_l)
    cos_alpha_l = np.cos(alpha_l)
    ephi_l = (-sin_alpha_l)[..., None] * u[None, None, :] + cos_alpha_l[..., None] * v[
        None, None, :
    ]
    etheta_l = (
        (cos_beta_l * cos_alpha_l)[..., None] * u[None, None, :]
        + (cos_beta_l * sin_alpha_l)[..., None] * v[None, None, :]
        - sin_beta_l[..., None] * n0[None, None, :]
    )
    m11 = np.einsum("abi,abi->ab", ephi_g, ephi_l)
    m12 = np.einsum("abi,abi->ab", ephi_g, etheta_l)
    m21 = np.einsum("abi,abi->ab", etheta_g, ephi_l)
    m22 = np.einsum("abi,abi->ab", etheta_g, etheta_l)
    coeff_te = m11 * g_te_l + m12 * g_tm_l
    coeff_tm = m21 * g_te_l + m22 * g_tm_l
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)


def _bessel_cartesian_angular_spectrum_coeffs(
    *,
    beam,
    k: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
) -> PlaneWaveSpectrum:
    """Evaluate Bessel ring spectrum with global-Cartesian polarization transport."""
    geom = _bessel_ring_geometry(
        beam=beam,
        k=float(k),
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
    )
    alpha = np.asarray(geom["alpha"], dtype=float)
    beta = np.asarray(geom["beta"], dtype=float)
    kx = np.asarray(geom["kx"], dtype=float)
    ky = np.asarray(geom["ky"], dtype=float)
    kz = np.asarray(geom["kz"], dtype=float)
    sx = np.asarray(geom["sx"], dtype=float)
    sy = np.asarray(geom["sy"], dtype=float)
    sz = np.asarray(geom["sz"], dtype=float)
    ephi_g = np.asarray(geom["ephi_g"], dtype=float)
    etheta_g = np.asarray(geom["etheta_g"], dtype=float)
    envelope = np.asarray(geom["envelope"], dtype=np.complex128)
    p = normalize_global_polarization_vector(beam.global_polarization)
    dot_ps = p[0] * sx + p[1] * sy + p[2] * sz
    ex_t = p[0] - dot_ps * sx
    ey_t = p[1] - dot_ps * sy
    ez_t = p[2] - dot_ps * sz
    g_te = ex_t * ephi_g[..., 0] + ey_t * ephi_g[..., 1] + ez_t * ephi_g[..., 2]
    g_tm = ex_t * etheta_g[..., 0] + ey_t * etheta_g[..., 1] + ez_t * etheta_g[..., 2]
    coeff_te = envelope * g_te
    coeff_tm = envelope * g_tm
    return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)
