from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import jv, spherical_jn, spherical_yn
from tqdm.auto import tqdm

from pyceles.core.angular import (
    beam_axis_and_frame,
    is_uniform_periodic_azimuth,
    periodic_azimuthal_weights,
    trapezoidal_weights,
)
from pyceles.core.geometry_bounds import conservative_cross_set_max_distance
from pyceles.core.indexing import index_vswf, n_modes
from pyceles.core.particles import LayeredSphere, Particle, Sphere
from pyceles.core.sources import (
    DipoleCollection,
    DipoleSource,
    GaussianBeam,
    PlaneWave,
    PolarizationInput,
    is_normal_incidence,
    polarization_to_jones,
)
from pyceles.core.spherical import spherical_functions_trigon
from pyceles.core.tmatrix import layered_internal_ab_ratios, sphere_internal_ratios


@dataclass(frozen=True)
class InternalPointClassification:
    """Broad-phase particle ownership of near-field sample points.

    `inside_any` marks points inside at least one circumscribing sphere.
    `point_indices_by_particle[j]` stores the global point indices associated
    with particle `j` under the same broad-phase rule.
    """

    inside_any: np.ndarray
    point_indices_by_particle: tuple[np.ndarray, ...]


def classify_internal_points(
    field_points: np.ndarray,
    particles: Sequence[Particle],
) -> InternalPointClassification:
    """Classify near-field points against particle circumscribing spheres.

    This broad-phase classification is exact for `Sphere` and `LayeredSphere`.
    """
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    part = list(particles)
    Np = pts.shape[0]
    inside_any = np.zeros((Np,), dtype=bool)
    by_particle: list[np.ndarray] = []

    supported = (Sphere, LayeredSphere)
    bad = [type(p).__name__ for p in part if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            f"Internal point classification currently supports Sphere and LayeredSphere. Got {bad}."
        )

    for p in part:
        center = np.asarray(p.position, dtype=float).reshape(3)
        rr = float(p.circumscribing_radius())
        R = pts - center[None, :]
        idx = np.flatnonzero(np.sum(R * R, axis=1) < (rr**2)).astype(np.intp, copy=False)
        by_particle.append(idx)
        inside_any[idx] = True

    return InternalPointClassification(
        inside_any=inside_any,
        point_indices_by_particle=tuple(by_particle),
    )


@cache
def _mode_indices_by_l(
    lmax: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Cache per-l arrays for vectorized mode contractions.

    Returns per l=1..lmax:
      (m_values, abs_m_values, tau1_indices, tau2_indices)
    """
    lmax = int(lmax)
    out = []
    for l in range(1, lmax + 1):
        m_vals = np.arange(-l, l + 1, dtype=np.int32)
        abs_m = np.abs(m_vals).astype(np.int32)
        n1_idx = np.array([index_vswf(l, int(m), 1, lmax) for m in m_vals], dtype=np.int32)
        n2_idx = np.array([index_vswf(l, int(m), 2, lmax) for m in m_vals], dtype=np.int32)
        out.append((m_vals, abs_m, n1_idx, n2_idx))
    return tuple(out)


def _sph_hankel1(l: int, x: np.ndarray) -> np.ndarray:
    """Spherical Hankel function of first kind h_l^(1)(x)."""
    return spherical_jn(l, x) + 1j * spherical_yn(l, x)


def _dx_xz_hankel1(l: int, x: np.ndarray) -> np.ndarray:
    """Return d/dx (x z_l(x)) for z_l = h_l^(1)."""
    z = _sph_hankel1(l, x)
    dzdx = spherical_jn(l, x, derivative=True) + 1j * spherical_yn(l, x, derivative=True)
    return z + x * dzdx


def _contract_modes(mode_coeffs: np.ndarray, mode_tensor: np.ndarray) -> np.ndarray:
    """Contract mode axis: (M,) x (B,M,3) -> (B,3)."""
    # Implementation choice is benchmark-driven:
    # we compared `einsum` (optimize=True/False), `tensordot`, and `matmul`
    # on this exact contraction. Performance is similar up to about lmax~3,
    # while matmul is consistently faster for larger lmax, and also easier to read.
    return np.matmul(np.transpose(mode_tensor, (0, 2, 1)), mode_coeffs)


def _build_internal_mode_tensors(
    *,
    l: int,
    m_vals: np.ndarray,
    abs_m: np.ndarray,
    phi: np.ndarray,
    e_r: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    PI: np.ndarray,
    TAU: np.ndarray,
    P: np.ndarray,
    z_l: np.ndarray,
    dxxz: np.ndarray,
    kr: np.ndarray,
    compute_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(M_l, N_l)` mode tensors for one degree and radial basis."""
    pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))

    z_over_kr = z_l / kr
    dxxz_over_kr = dxxz / kr

    P_lm = P[l, abs_m, :].T
    pi_lm = PI[l, abs_m, :].T
    tau_lm = TAU[l, abs_m, :].T

    eimphi = np.asarray(np.exp(1j * phi[:, None] * m_vals[None, :]), dtype=compute_dtype)
    impi = np.asarray((1j * m_vals[None, :]) * pi_lm, dtype=compute_dtype)

    theta_phi_m = impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
    Mv_all = pref * z_l[:, None, None] * theta_phi_m * eimphi[:, :, None]

    radial_er = (l * (l + 1.0) * z_over_kr)[:, None] * P_lm
    mix_theta_phi = tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
    Nv_all = (
        pref
        * (radial_er[:, :, None] * e_r[:, None, :] + dxxz_over_kr[:, None, None] * mix_theta_phi)
        * eimphi[:, :, None]
    )

    return Mv_all, Nv_all


def _compute_initial_field_gaussian_normal_incidence_analytic(
    field_points_local: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam_width: float,
    amplitude: float,
    polarization: PolarizationInput,
    azimuthal_angle: float,
    propagation_sign: float,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fast initial-field kernel for Gaussian wavebundles aligned with local +z.

    This analytically integrates the azimuthal angle using Bessel identities and
    leaves only a numerical beta quadrature.
    """
    if not np.isfinite(float(beam_width)) or float(beam_width) <= 0.0:
        return None
    if not is_uniform_periodic_azimuth(azimuthal_angles):
        return None

    pts = np.asarray(field_points_local, dtype=np.float64)
    beta = np.asarray(polar_angles, dtype=float).reshape(-1)
    if beta.size < 2:
        return None

    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    real_compute_dtype = np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64
    nM = complex(n_medium)

    x = pts[:, 0].astype(real_compute_dtype, copy=False)
    y = pts[:, 1].astype(real_compute_dtype, copy=False)
    z = pts[:, 2].astype(real_compute_dtype, copy=False)
    rho = np.hypot(x, y).astype(real_compute_dtype, copy=False)
    phi = np.arctan2(y, x)
    sin_phi = np.sin(phi).astype(real_compute_dtype, copy=False)
    cos_phi = np.cos(phi).astype(real_compute_dtype, copy=False)
    sin2_phi = (2.0 * sin_phi * cos_phi).astype(real_compute_dtype, copy=False)
    cos2_phi = (cos_phi * cos_phi - sin_phi * sin_phi).astype(real_compute_dtype, copy=False)

    sb = np.sin(beta).astype(real_compute_dtype, copy=False)
    cb = np.cos(beta).astype(real_compute_dtype, copy=False)
    beta_w = trapezoidal_weights(beta).astype(real_compute_dtype, copy=False)

    E0 = float(amplitude)
    w = float(beam_width)
    pref = E0 * (k**2) * (w**2) / (4.0 * np.pi)
    envelope = pref * cb * np.exp(-(w**2) / 4.0 * (k**2) * (sb**2))
    envelope *= np.sign(cb) == np.sign(float(propagation_sign))
    beta_weighted = envelope * sb * beta_w

    a_te, a_tm = polarization_to_jones(polarization)
    if not (np.isclose(abs(a_te), 0.0) or np.isclose(abs(a_tm), 0.0)):
        out_te = _compute_initial_field_gaussian_normal_incidence_analytic(
            field_points_local,
            k=k,
            n_medium=n_medium,
            beam_width=beam_width,
            amplitude=amplitude,
            polarization="TE",
            azimuthal_angle=azimuthal_angle,
            propagation_sign=propagation_sign,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        out_tm = _compute_initial_field_gaussian_normal_incidence_analytic(
            field_points_local,
            k=k,
            n_medium=n_medium,
            beam_width=beam_width,
            amplitude=amplitude,
            polarization="TM",
            azimuthal_angle=azimuthal_angle,
            propagation_sign=propagation_sign,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
            show_progress=False,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        if out_te is None or out_tm is None:
            return None
        e_te, h_te = out_te
        e_tm, h_tm = out_tm
        return (
            np.asarray(a_te * e_te + a_tm * e_tm, dtype=accum_dtype),
            np.asarray(a_te * h_te + a_tm * h_tm, dtype=accum_dtype),
        )

    pol = "te" if abs(a_te) > 0 else "tm"
    alpha_g = float(azimuthal_angle) if pol == "te" else float(azimuthal_angle) - np.pi / 2.0
    C = float(np.cos(alpha_g))
    S = float(np.sin(alpha_g))
    s = float(np.sign(propagation_sign))
    if s == 0.0:
        s = 1.0

    E = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    H = np.zeros_like(E)

    # Geometry-only combinations reused for each beta:
    # this avoids rebuilding multiple intermediate I_* arrays per iteration.
    u1 = (C * sin2_phi - S * cos2_phi).astype(real_compute_dtype, copy=False)
    u2 = (C * cos2_phi + S * sin2_phi).astype(real_compute_dtype, copy=False)
    u3 = (S * cos_phi - C * sin_phi).astype(real_compute_dtype, copy=False)
    u4 = (C * cos_phi + S * sin_phi).astype(real_compute_dtype, copy=False)
    # Keep scalar prefactors as Python floats for mypy clarity; array dtypes are
    # still governed by explicit astype(..., real_compute_dtype/compute_dtype).
    pi_r = float(np.pi)
    two_i_pi = np.asarray(2j * np.pi, dtype=compute_dtype)
    c_plus = (1.0 + s * cb).astype(real_compute_dtype, copy=False)
    c_minus = (1.0 - s * cb).astype(real_compute_dtype, copy=False)
    c_splus = (s + cb).astype(real_compute_dtype, copy=False)
    c_sminus = (s - cb).astype(real_compute_dtype, copy=False)

    # Only one propagation hemisphere contributes for a one-way beam.
    # In finite precision, very oblique components can underflow to exactly zero
    # in `beta_weighted`; those entries are skipped exactly (no thresholding).
    if s >= 0.0:
        hemi_mask = cb > 0.0
        hemi_label = "fwd"
    else:
        hemi_mask = cb < 0.0
        hemi_label = "bwd"
    nonzero_mask = beta_weighted != 0.0
    active_beta = np.flatnonzero(hemi_mask & nonzero_mask)
    beta_iter = active_beta
    if show_progress:
        hemi_total = int(np.count_nonzero(hemi_mask))
        desc = f"Initial field (non-zero {hemi_label} betas {int(active_beta.size)}/{hemi_total})"
        beta_iter = tqdm(
            active_beta,
            desc=desc,
            total=int(active_beta.size),
        )

    for ib_raw in beta_iter:
        ib = int(ib_raw)
        bw = beta_weighted[ib]
        if bw == 0.0:
            continue
        sb_i = sb[ib]
        cb_i = cb[ib]
        # Scalar casts avoid object-typed indices in strict type checking.
        # This does not change the compute/accum dtype policy of vector kernels.
        c_plus_i = float(c_plus[ib])
        c_minus_i = float(c_minus[ib])
        c_splus_i = float(c_splus[ib])
        c_sminus_i = float(c_sminus[ib])

        q = (k * sb_i) * rho
        J0 = jv(0, q).astype(real_compute_dtype, copy=False)
        J1 = jv(1, q).astype(real_compute_dtype, copy=False)
        J2 = jv(2, q).astype(real_compute_dtype, copy=False)
        phase_z = np.exp(1j * (k * cb_i) * z).astype(compute_dtype, copy=False)
        wgt = (bw * phase_z).astype(compute_dtype, copy=False)

        # Equivalent closed-form contraction (same kernel, fewer temporaries).
        ex = (-S * pi_r * c_plus_i) * J0 + (pi_r * c_minus_i) * (u1 * J2)
        ey = (C * pi_r * c_plus_i) * J0 - (pi_r * c_minus_i) * (u2 * J2)
        ez = (two_i_pi * (s * sb_i)) * (u3 * J1)

        hx = (-C * pi_r * c_splus_i) * J0 - (pi_r * c_sminus_i) * (u2 * J2)
        hy = (-S * pi_r * c_splus_i) * J0 - (pi_r * c_sminus_i) * (u1 * J2)
        hz = (two_i_pi * sb_i) * (u4 * J1)

        E[:, 0] += (wgt * ex).astype(accum_dtype, copy=False)
        E[:, 1] += (wgt * ey).astype(accum_dtype, copy=False)
        E[:, 2] += (wgt * ez).astype(accum_dtype, copy=False)
        H[:, 0] += nM * (wgt * hx).astype(accum_dtype, copy=False)
        H[:, 1] += nM * (wgt * hy).astype(accum_dtype, copy=False)
        H[:, 2] += nM * (wgt * hz).astype(accum_dtype, copy=False)

    return E, H


def _compute_initial_field_gaussian_rotated_fast(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam: GaussianBeam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fast path: rotate tilted Gaussian beams to local +z, evaluate, rotate back.

    This is intended for regular (non-periodic) near-field evaluation where the
    incident beam can be represented in a rotated local frame.
    """
    pts = np.asarray(field_points, dtype=np.float64)
    fp = np.asarray(getattr(beam, "focal_point", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
    rel = pts - fp[None, :]

    if is_normal_incidence(float(beam.polar_angle)):
        local = rel
        rot_back = np.eye(3, dtype=float)
        prop_sign = float(np.sign(np.cos(float(beam.polar_angle))))
    else:
        n0, u, v = beam_axis_and_frame(float(beam.polar_angle), float(beam.azimuthal_angle))
        Q = np.stack([u, v, n0], axis=0)  # rows are local basis vectors in global coordinates
        local = rel @ Q.T
        rot_back = Q
        prop_sign = 1.0  # local frame aligns beam propagation with +z

    out_local = _compute_initial_field_gaussian_normal_incidence_analytic(
        local,
        k=float(k),
        n_medium=complex(n_medium),
        beam_width=float(beam.beam_width),
        amplitude=float(beam.amplitude),
        polarization=beam.polarization,
        azimuthal_angle=float(beam.azimuthal_angle),
        propagation_sign=prop_sign,
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        show_progress=show_progress,
        compute_dtype=np.dtype(compute_dtype),
        accum_dtype=np.dtype(accum_dtype),
    )
    if out_local is None:
        return None
    E_local, H_local = out_local
    E_global = np.asarray(E_local, dtype=np.dtype(accum_dtype)) @ rot_back
    H_global = np.asarray(H_local, dtype=np.dtype(accum_dtype)) @ rot_back
    return E_global, H_global


class NearFieldRadialLUT:
    """Near-field lookup table for z_l(kr) and d/d(kr) (kr z_l(kr)) on uniform r.

    This LUT is specific to near/internal-field SVWF evaluation.
    It is distinct from `pyceles.core.translation.RadialLUT`, which stores
    translation-kernel Hankel tables for pair-interaction blocks.
    """

    def __init__(
        self,
        *,
        lmax: int,
        k: float,
        r_max: float,
        dr: float = 1.0,
        dtype: npt.DTypeLike = np.complex128,
    ):
        """Precompute near-field radial kernels on a uniform radius grid.

        Stored quantities are `h_l^(1)(kr)` and `d/d(kr)[kr h_l^(1)(kr)]` for
        all `l<=lmax`, later interpolated during scattered-field evaluation.
        """
        self.lmax = int(lmax)
        self.k = float(k)
        self.dr = float(dr)
        self.dtype = np.dtype(dtype)
        self.ri = np.arange(0.0, r_max + dr, dr, dtype=np.float64)
        x = self.k * self.ri
        # Avoid evaluating Hankel at exactly x=0 (singular)
        if x.size > 1:
            x = x.copy()
            x[0] = x[1]
        self.h: list[np.ndarray] = [
            np.zeros_like(x, dtype=self.dtype) for _ in range(self.lmax + 1)
        ]
        self.dxxz: list[np.ndarray] = [
            np.zeros_like(x, dtype=self.dtype) for _ in range(self.lmax + 1)
        ]
        for l in range(1, self.lmax + 1):
            self.h[l] = _sph_hankel1(l, x).astype(self.dtype, copy=False)
            self.dxxz[l] = _dx_xz_hankel1(l, x).astype(self.dtype, copy=False)

    def interp(self, l: int, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate precomputed radial kernels at requested radii."""
        # Keep this tiny kernel local: the LUT radius grid is uniform, so we can
        # jump in O(1) to each interval via floor(r/dr) and reuse the same
        # interpolation weights for both radial arrays (`h` and `dxxz`).
        # Generic interpolation helpers would add avoidable index-search overhead
        # in this hot inner loop.
        r = np.asarray(r, np.float64)
        ri = self.ri
        idx = np.clip((r / self.dr).astype(np.int64), 0, len(ri) - 2)
        t = (r - ri[idx]) / self.dr
        z0 = self.h[l][idx]
        z1 = self.h[l][idx + 1]
        d0 = self.dxxz[l][idx]
        d1 = self.dxxz[l][idx + 1]
        z = (1 - t) * z0 + t * z1
        d = (1 - t) * d0 + t * d1
        return z, d


def compute_scattered_field(
    field_points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_medium: complex = 1.0 + 0j,
    particle_distance_resolution: float = 1.0,
    lut: NearFieldRadialLUT | None = None,
    active_mask: np.ndarray | None = None,
    batch_size: int = 8192,
    show_progress: bool = True,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
):
    """CELES-like scattered field evaluation at points.

    Implements the same SVWF formula as CELES `compute_scattered_field.m`:
      M = 1/sqrt(2 l(l+1)) * z_l(kr) * ((i m pi) e_theta - tau e_phi) e^{i m phi}
      N = 1/sqrt(2 l(l+1)) * ( (l(l+1) z/(kr) P) e_r + (dxxz/(kr)) (tau e_theta + i m pi e_phi) ) e^{i m phi}
      E += a * M + b * N
      H += -i n_medium (a * N + b * M)
    where coefficients a,b come from tau=1 (M) and tau=2 (N) blocks in our ordering.

    `active_mask`, when provided, marks the subset of points to evaluate.
    Points outside the mask are left as zero. This is used by higher-level
    workflows to skip interior-particle points where the exterior scattered
    expansion is not physically meaningful.
    """
    pts = np.asarray(field_points, np.float64)
    pos = np.asarray(positions, np.float64)
    lmax = int(lmax)
    k = float(k)
    nM = complex(n_medium)
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)

    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    c = np.asarray(coeffs, compute_dtype).reshape(Ns, Nm)
    E = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    H = np.zeros_like(E)

    if Ns == 0:
        # Source-only (no scatterers): scattered contribution is identically zero.
        return E, H

    idx_eval: np.ndarray | None = None
    if active_mask is None:
        pts_eval = pts
        E_eval = E
        H_eval = H
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != pts.shape[0]:
            raise ValueError(f"`active_mask` must have length {pts.shape[0]}. Got {mask.shape[0]}.")
        idx_eval = np.flatnonzero(mask)
        if idx_eval.size == 0:
            return E, H
        pts_eval = pts[idx_eval]
        E_eval = np.zeros((pts_eval.shape[0], 3), dtype=accum_dtype)
        H_eval = np.zeros_like(E_eval)

    if lut is None:
        # Size the near-field radial LUT with an O(N+M) conservative bound
        # instead of an exact O(N*M) center-point scan.
        rmax = conservative_cross_set_max_distance(pos, pts_eval)
        lut = NearFieldRadialLUT(
            lmax=lmax,
            k=k,
            r_max=rmax,
            dr=particle_distance_resolution,
            dtype=compute_dtype,
        )

    mode_by_l = _mode_indices_by_l(lmax)

    for jS in tqdm(
        range(Ns),
        desc="Scattered field (spheres)",
        total=Ns,
        disable=not show_progress,
    ):
        for s in range(0, pts_eval.shape[0], batch_size):
            e = min(pts_eval.shape[0], s + batch_size)
            p = pts_eval[s:e]
            R = p - pos[jS]

            r = np.linalg.norm(R, axis=1)
            # avoid division by zero exactly at sphere center
            r_safe = np.where(r == 0, 1e-30, r)
            e_r = R / r_safe[:, None]
            ct = np.clip(e_r[:, 2], -1.0, 1.0)
            st = np.sqrt(np.maximum(0.0, 1.0 - ct**2))
            phi = np.arctan2(R[:, 1], R[:, 0])

            e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
            e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)

            kr = k * r_safe

            # Compute P, pi, tau in one recurrence pass.
            PI, TAU, P = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

            for l in range(1, lmax + 1):
                m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
                pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))

                z, dxxz = lut.interp(l, r)
                z_over_kr = z / kr
                dxxz_over_kr = dxxz / kr

                # Shape to (B, M) for vectorized contractions over all m for this l.
                P_lm = P[l, abs_m, :].T
                pi_lm = PI[l, abs_m, :].T
                tau_lm = TAU[l, abs_m, :].T

                eimphi = np.asarray(
                    np.exp(1j * phi[:, None] * m_vals[None, :]), dtype=compute_dtype
                )
                impi = np.asarray((1j * m_vals[None, :]) * pi_lm, dtype=compute_dtype)

                # Mode coefficients for this sphere and all m at fixed l.
                a_vec = c[jS, n1_idx].astype(compute_dtype, copy=False)
                b_vec = c[jS, n2_idx].astype(compute_dtype, copy=False)

                theta_phi_m = (
                    impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
                )
                Mv_all = pref * z[:, None, None] * theta_phi_m * eimphi[:, :, None]

                radial_er = (l * (l + 1.0) * z_over_kr)[:, None] * P_lm
                mix_theta_phi = (
                    tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
                )
                Nv_all = (
                    pref
                    * (
                        radial_er[:, :, None] * e_r[:, None, :]
                        + dxxz_over_kr[:, None, None] * mix_theta_phi
                    )
                    * eimphi[:, :, None]
                )

                E_eval[s:e] += _contract_modes(a_vec, Mv_all)
                E_eval[s:e] += _contract_modes(b_vec, Nv_all)
                H_eval[s:e] += (-1j * nM) * _contract_modes(a_vec, Nv_all)
                H_eval[s:e] += (-1j * nM) * _contract_modes(b_vec, Mv_all)

    if idx_eval is not None:
        E[idx_eval] = E_eval
        H[idx_eval] = H_eval
        return E, H
    return E_eval, H_eval


def compute_internal_field(
    field_points: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    *,
    particles: Sequence[Particle],
    _point_classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the *total* (physical) field inside particles.

    This mirrors CELES `compute_internal_field.m`:

    - Detect which evaluation points are inside which sphere (non-overlapping spheres assumed).
    - Convert scattered VSWF coefficients to internal coefficients via the CELES ratio
      (T_entry internal / T_entry scattered).
    - Evaluate regular SVWFs (nu=1, spherical Bessel j_l) at k_particle.
    - Return E,H defined on the full set of points, and a boolean mask indicating
      which points were inside a sphere.

    Parameters
    ----------
    field_points:
        (Np,3) points at which to evaluate fields.
    coeffs:
        (Ns, Nm) scattered-field expansion coefficients (CELES ordering).
    k:
        Medium wavenumber k_medium = k0 * n_medium.
    lmax:
        Multipole truncation.
    particles:
        Explicit particle descriptors (`Sphere` and/or `LayeredSphere`).
    n_medium:
        Medium refractive index.

    Returns
    -------
    E_internal, H_internal, internal_mask
        Arrays have shape (Np,3). Values are non-zero only where internal_mask is True.

    Notes
    -----
    The particle API is canonical. Mixed `Sphere` + `LayeredSphere` lists are
    supported.
    """
    return _compute_internal_field_particles(
        field_points,
        particles=particles,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        n_medium=n_medium,
        classification=_point_classification,
        show_progress=show_progress,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )


def _compute_internal_field_homogeneous_spheres(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    inside_indices_by_sphere: Sequence[np.ndarray] | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical homogeneous-sphere internal-field kernel used by dispatchers."""

    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    Ns = pos.shape[0]
    Np = pts.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)

    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != Ns:
        raise ValueError(f"radii must have length Ns={Ns}, got {rad.shape}")
    if inside_indices_by_sphere is not None and len(inside_indices_by_sphere) != Ns:
        raise ValueError(
            "`inside_indices_by_sphere` length must match number of spheres "
            f"({Ns}). Got {len(inside_indices_by_sphere)}."
        )

    # Allow scalar or per-sphere refractive index
    if np.ndim(n_particle) == 0:
        n_particle_arr = np.full(Ns, complex(n_particle), dtype=complex)
    else:
        n_particle_arr = np.asarray(n_particle, dtype=complex).reshape(-1)
        if n_particle_arr.shape[0] != Ns:
            raise ValueError(f"n_particle must have length Ns={Ns}, got {n_particle_arr.shape}")

    n_medium_c = complex(n_medium)

    E = np.zeros((Np, 3), dtype=accum_dtype)
    H = np.zeros((Np, 3), dtype=accum_dtype)
    inside = np.zeros(Np, dtype=bool)

    sphere_iter = range(Ns)
    if show_progress:
        sphere_iter = tqdm(sphere_iter, desc="Internal field (spheres)", leave=True)

    eps = 1e-12
    mode_by_l = _mode_indices_by_l(lmax)

    for jS in sphere_iter:
        if inside_indices_by_sphere is None:
            R_full = pts - pos[jS]
            r2_full = np.sum(R_full * R_full, axis=1)
            idx = np.flatnonzero(r2_full < (rad[jS] ** 2))
        else:
            idx = np.asarray(inside_indices_by_sphere[jS], dtype=np.intp).reshape(-1)
        if idx.size == 0:
            continue

        inside[idx] = True
        R = pts[idx] - pos[jS]
        r2 = np.sum(R * R, axis=1)
        r = np.sqrt(r2)
        r_safe = np.where(r < eps, eps, r)

        x = R[:, 0]
        y = R[:, 1]
        z = R[:, 2]

        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = np.arctan2(y, x)

        e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
        e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)

        # Compute P, pi, tau in one recurrence pass.
        PI, TAU, P = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

        # internal wavenumber in particle
        nS = n_particle_arr[jS]
        kS = k * (nS / n_medium_c)
        kr = kS * r_safe

        # ratios converting scattered -> internal coefficients for this sphere
        ratios = sphere_internal_ratios(lmax, k, rad[jS], nS, n_medium_c)
        ratio_M = ratios[1]
        ratio_N = ratios[2]

        for l in range(1, lmax + 1):
            # regular radial functions (nu=1 in CELES SVWF notation)
            z_l = spherical_jn(l, kr)
            dz_l = spherical_jn(l, kr, derivative=True)
            dxxz = z_l + kr * dz_l

            z_over_kr = z_l / kr
            dxxz_over_kr = dxxz / kr

            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))

            # (B, M) arrays
            Plm = P[l, abs_m, :].T
            pilm = PI[l, abs_m, :].T
            taulm = TAU[l, abs_m, :].T

            eimphi = np.asarray(np.exp(1j * phi[:, None] * m_vals[None, :]), dtype=compute_dtype)
            impi = np.asarray((1j * m_vals[None, :]) * pilm, dtype=compute_dtype)

            theta_phi_m = (
                impi[:, :, None] * e_theta[:, None, :] - taulm[:, :, None] * e_phi[:, None, :]
            )
            Mv_all = pref * z_l[:, None, None] * theta_phi_m * eimphi[:, :, None]

            radial_er = (l * (l + 1.0) * z_over_kr)[:, None] * Plm
            mix_theta_phi = (
                taulm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
            )
            Nv_all = (
                pref
                * (
                    radial_er[:, :, None] * e_r[:, None, :]
                    + dxxz_over_kr[:, None, None] * mix_theta_phi
                )
                * eimphi[:, :, None]
            )

            a_int = coeffs[jS, n1_idx].astype(compute_dtype, copy=False) * ratio_M[l]
            b_int = coeffs[jS, n2_idx].astype(compute_dtype, copy=False) * ratio_N[l]

            E[idx] += _contract_modes(a_int, Mv_all)
            E[idx] += _contract_modes(b_int, Nv_all)

            # CELES convention: H = -i * (kS/k0) * (a*N + b*M) = -i*nS*(a*N + b*M)
            H[idx] += (-1j * nS) * _contract_modes(a_int, Nv_all)
            H[idx] += (-1j * nS) * _contract_modes(b_int, Mv_all)

    return E, H, inside


def _compute_internal_field_particles(
    field_points: np.ndarray,
    particles: Sequence[Particle],
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute internal fields for explicit particle descriptors.

    Internal helper backing the particle-dispatch path in
    :func:`compute_internal_field`.

    It supports mixed particle lists with `Sphere` and `LayeredSphere`
    entries. For layered spheres, each shell uses the physically correct piecewise radial basis
    `A*j_l(k_j r) + B*h_l^(1)(k_j r)`, with `B=0` in the core.
    """
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    part = list(particles)
    Ns = len(part)
    Np = pts.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    n_medium_c = complex(n_medium)
    lmax = int(lmax)
    Nm = n_modes(lmax)

    E = np.zeros((Np, 3), dtype=accum_dtype)
    H = np.zeros((Np, 3), dtype=accum_dtype)
    inside = np.zeros(Np, dtype=bool)
    if Ns == 0:
        return E, H, inside
    if classification is not None:
        if classification.inside_any.shape != (Np,):
            raise ValueError(
                "`classification.inside_any` must have shape "
                f"({Np},). Got {classification.inside_any.shape}."
            )
        if len(classification.point_indices_by_particle) != Ns:
            raise ValueError(
                "`classification.point_indices_by_particle` length must match "
                f"particle count ({Ns}). Got {len(classification.point_indices_by_particle)}."
            )

    c = np.asarray(coeffs, dtype=compute_dtype)
    if c.size != Ns * Nm:
        raise ValueError(
            f"`coeffs` must have {Ns * Nm} entries for {Ns} particles and lmax={lmax}. Got {c.size}."
        )
    c = c.reshape(Ns, Nm)

    # Reuse the established homogeneous-sphere kernel if possible.
    spheres = [sp for sp in part if isinstance(sp, Sphere)]
    if len(spheres) == Ns:
        positions = np.asarray([sp.position for sp in spheres], dtype=float).reshape(Ns, 3)
        radii = np.asarray([sp.radius for sp in spheres], dtype=float).reshape(Ns)
        n_particle = np.asarray(
            [complex(sp.refractive_index) for sp in spheres], dtype=np.complex128
        )
        inside_idx = (
            [np.asarray(v, dtype=np.intp) for v in classification.point_indices_by_particle]
            if classification is not None
            else None
        )
        return _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            inside_indices_by_sphere=inside_idx,
            n_medium=n_medium_c,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )

    supported = (Sphere, LayeredSphere)
    bad = [type(p).__name__ for p in part if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            "compute_internal_field currently supports Sphere and LayeredSphere in "
            f"particle-dispatch mode. Got {bad}."
        )

    sphere_idx = [j for j, p in enumerate(part) if isinstance(p, Sphere)]
    layered_idx = [j for j, p in enumerate(part) if isinstance(p, LayeredSphere)]

    # Reuse the canonical homogeneous-sphere kernel to avoid sphere-path drift.
    if sphere_idx:
        sphere_part = [p for p in part if isinstance(p, Sphere)]
        positions = np.asarray([sp.position for sp in sphere_part], dtype=float).reshape(-1, 3)
        radii = np.asarray([sp.radius for sp in sphere_part], dtype=float).reshape(-1)
        n_particle = np.asarray(
            [complex(sp.refractive_index) for sp in sphere_part], dtype=np.complex128
        )
        c_sphere = c[np.asarray(sphere_idx, dtype=int), :]
        inside_idx = (
            [
                np.asarray(classification.point_indices_by_particle[j], dtype=np.intp)
                for j in sphere_idx
            ]
            if classification is not None
            else None
        )
        E_s, H_s, inside_s = _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c_sphere,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            inside_indices_by_sphere=inside_idx,
            n_medium=n_medium_c,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        E += E_s
        H += H_s
        inside |= inside_s

    if not layered_idx:
        return E, H, inside

    eps = 1e-12
    mode_by_l = _mode_indices_by_l(lmax)
    l_iter = layered_idx
    if show_progress:
        l_iter = tqdm(layered_idx, desc="Internal field (layered particles)", leave=True)

    for jS in l_iter:
        p = part[jS]
        if not isinstance(p, LayeredSphere):
            continue
        center = np.asarray(p.position, dtype=float).reshape(3)
        outer_radius = float(p.circumscribing_radius())

        if classification is None:
            R_full = pts - center[None, :]
            r2_full = np.sum(R_full * R_full, axis=1)
            # TODO(spheroids): this is exact for concentric layered spheres.
            # Introduce particle-native point-containment capability before adding
            # non-spherical internal-field kernels.
            idx = np.flatnonzero(r2_full < (outer_radius**2))
        else:
            idx = np.asarray(classification.point_indices_by_particle[jS], dtype=np.intp).reshape(
                -1
            )
        if idx.size == 0:
            continue

        inside[idx] = True
        R = pts[idx] - center[None, :]
        r2 = np.sum(R * R, axis=1)
        r = np.sqrt(r2)
        r_safe = np.where(r < eps, eps, r)

        x = R[:, 0]
        y = R[:, 1]
        z = R[:, 2]
        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = np.arctan2(y, x)

        e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
        e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)

        PI, TAU, P = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

        layer_radii = np.asarray(p.layer_radii, dtype=float).reshape(-1)
        layer_n = np.asarray(p.layer_refractive_indices, dtype=np.complex128).reshape(-1)
        layer_idx = np.searchsorted(layer_radii, r, side="right")
        layer_idx = np.clip(layer_idx, 0, layer_radii.size - 1)
        k_layers = float(k) * (layer_n / n_medium_c)
        layered_ratios = layered_internal_ab_ratios(
            lmax=lmax,
            k_medium=float(k),
            layer_radii=p.layer_radii,
            layer_refractive_indices=p.layer_refractive_indices,
            n_medium=n_medium_c,
        )
        A_m = layered_ratios[1]["A"]
        B_m = layered_ratios[1]["B"]
        A_n = layered_ratios[2]["A"]
        B_n = layered_ratios[2]["B"]

        for l in range(1, lmax + 1):
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            a_out = c[jS, n1_idx].astype(compute_dtype, copy=False)
            b_out = c[jS, n2_idx].astype(compute_dtype, copy=False)

            for g in range(layer_radii.size):
                gmask = layer_idx == g
                if not np.any(gmask):
                    continue
                idx_g = idx[gmask]
                r_g = r_safe[gmask]
                phi_g = phi[gmask]
                kr = k_layers[g] * r_g

                jl = spherical_jn(l, kr)
                djl = spherical_jn(l, kr, derivative=True)
                use_h = not (
                    np.isclose(B_m[g, l], 0.0, rtol=0.0, atol=0.0)
                    and np.isclose(B_n[g, l], 0.0, rtol=0.0, atol=0.0)
                )
                if use_h:
                    yl = spherical_yn(l, kr)
                    hl = jl + 1j * yl
                    dyl = spherical_yn(l, kr, derivative=True)
                    dhl = djl + 1j * dyl
                else:
                    hl = np.zeros_like(jl, dtype=np.complex128)
                    dhl = np.zeros_like(djl, dtype=np.complex128)

                z_m = A_m[g, l] * jl + B_m[g, l] * hl
                dxxz_m = A_m[g, l] * (jl + kr * djl) + B_m[g, l] * (hl + kr * dhl)

                z_n = A_n[g, l] * jl + B_n[g, l] * hl
                dxxz_n = A_n[g, l] * (jl + kr * djl) + B_n[g, l] * (hl + kr * dhl)

                M_m, N_m = _build_internal_mode_tensors(
                    l=l,
                    m_vals=m_vals,
                    abs_m=abs_m,
                    phi=phi_g,
                    e_r=e_r[gmask],
                    e_theta=e_theta[gmask],
                    e_phi=e_phi[gmask],
                    PI=PI[:, :, gmask],
                    TAU=TAU[:, :, gmask],
                    P=P[:, :, gmask],
                    z_l=np.asarray(z_m, dtype=compute_dtype),
                    dxxz=np.asarray(dxxz_m, dtype=compute_dtype),
                    kr=np.asarray(kr, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )
                M_n, N_n = _build_internal_mode_tensors(
                    l=l,
                    m_vals=m_vals,
                    abs_m=abs_m,
                    phi=phi_g,
                    e_r=e_r[gmask],
                    e_theta=e_theta[gmask],
                    e_phi=e_phi[gmask],
                    PI=PI[:, :, gmask],
                    TAU=TAU[:, :, gmask],
                    P=P[:, :, gmask],
                    z_l=np.asarray(z_n, dtype=compute_dtype),
                    dxxz=np.asarray(dxxz_n, dtype=compute_dtype),
                    kr=np.asarray(kr, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )

                E[idx_g] += _contract_modes(a_out, M_m)
                E[idx_g] += _contract_modes(b_out, N_n)

                n_loc = complex(layer_n[g])
                H[idx_g] += (-1j * n_loc) * _contract_modes(a_out, N_m)
                H[idx_g] += (-1j * n_loc) * _contract_modes(b_out, M_n)

    return E, H, inside


def compute_initial_field(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    batch_size: int = 2048,
    show_progress: bool = True,
    force_general_initial_field: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
):
    """Compute initial (incident) field.

    Strategy:
    - Plane-wave: direct closed form.
    - Gaussian sources in regular (non-periodic) workflows: hidden rotated-frame
      fast path with analytic azimuth integration.
    - Fallback: general alpha-beta quadrature (`_compute_initial_field_general`),
      kept for flexible/periodic workflows and non-periodic azimuth grids.

    Notes
    -----
    `field_points` is a generic `(N, 3)` point cloud; it is not restricted to a
    planar slice. Plane slicing is handled by higher-level convenience wrappers.

    Developer note:
    We benchmarked several alternatives on the 500-sphere / dx=40 case.
    `z*kz` phase precompute gave only modest gains (~8%) at large memory payload;
    full-alpha giant batches did not help consistently; multiprocessing achieved
    speedups but with substantial RAM/process overhead and extra plumbing.
    Progress reporting in the fast Gaussian kernel is over non-zero beta
    components in the propagation hemisphere (finite-precision underflow can
    reduce the count below half the full beta grid).
    """
    if (not bool(force_general_initial_field)) and isinstance(beam, GaussianBeam):
        fast = _compute_initial_field_gaussian_rotated_fast(
            field_points,
            k=float(k),
            n_medium=complex(n_medium),
            beam=beam,
            polar_angles=np.asarray(polar_angles, float),
            azimuthal_angles=np.asarray(azimuthal_angles, float),
            show_progress=show_progress,
            compute_dtype=np.dtype(compute_dtype),
            accum_dtype=np.dtype(accum_dtype),
        )
        if fast is not None:
            return fast

    return _compute_initial_field_general(
        field_points,
        k=float(k),
        n_medium=complex(n_medium),
        beam=beam,
        polar_angles=np.asarray(polar_angles, float),
        azimuthal_angles=np.asarray(azimuthal_angles, float),
        batch_size=int(batch_size),
        show_progress=show_progress,
        compute_dtype=np.dtype(compute_dtype),
        accum_dtype=np.dtype(accum_dtype),
    )


def _compute_initial_field_general(
    field_points: np.ndarray,
    *,
    k: float,
    n_medium: complex,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    batch_size: int,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """General alpha-beta quadrature evaluator kept for flexible fallback paths."""
    pts = np.asarray(field_points, np.float64)
    nM = complex(n_medium)

    if isinstance(beam, (DipoleSource, DipoleCollection)):
        dip_pos = np.asarray(beam.dipole_positions(), dtype=float).reshape(-1, 3)
        dip_coeffs = np.asarray(beam.outgoing_coeffs(1, dtype=compute_dtype), dtype=compute_dtype)
        E, H = compute_scattered_field(
            pts,
            dip_pos,
            dip_coeffs,
            k=float(k),
            lmax=1,
            n_medium=nM,
            particle_distance_resolution=float(getattr(beam, "radial_lut_dr", 1.0)),
            batch_size=int(batch_size),
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        # Explicitly mask exact singular points if a sampling point hits a dipole center.
        if dip_pos.shape[0] > 0 and pts.shape[0] > 0:
            hit = np.any(
                np.all(np.isclose(pts[:, None, :], dip_pos[None, :, :], atol=1e-12), axis=2),
                axis=1,
            )
            if np.any(hit):
                E = np.asarray(E, dtype=accum_dtype).copy()
                H = np.asarray(H, dtype=accum_dtype).copy()
                E[hit] = np.nan + 0j
                H[hit] = np.nan + 0j
        return np.asarray(E, dtype=accum_dtype), np.asarray(H, dtype=accum_dtype)

    if isinstance(beam, PlaneWave):
        alpha_pw = float(beam.azimuthal_angle)
        beta_pw = float(beam.polar_angle)
        amp = complex(beam.amplitude)
        fp = np.asarray(beam.focal_point, dtype=np.float64).reshape(3)

        ca = np.cos(alpha_pw)
        sa = np.sin(alpha_pw)
        cb = np.cos(beta_pw)
        sb = np.sin(beta_pw)
        khat = np.array([sb * ca, sb * sa, cb], dtype=np.float64)
        kvec = float(k) * khat

        ehat_te = np.array([-sa, ca, 0.0], dtype=np.float64)
        ehat_tm = np.array([cb * ca, cb * sa, -sb], dtype=np.float64)
        a_te, a_tm = polarization_to_jones(beam.polarization)
        ehat = a_te * ehat_te + a_tm * ehat_tm
        hhat = np.cross(khat, ehat)

        phase = np.exp(1j * ((pts - fp[None, :]) @ kvec))
        E = amp * phase[:, None] * ehat[None, :]
        H = amp * phase[:, None] * (nM * hhat)[None, :]
        return E.astype(accum_dtype, copy=False), H.astype(accum_dtype, copy=False)

    if hasattr(beam, "angular_spectrum"):
        pwp_te, pwp_tm = beam.angular_spectrum(
            k=float(k),
            polar_angles=np.asarray(polar_angles, float),
            azimuthal_angles=np.asarray(azimuthal_angles, float),
        )
    else:
        raise TypeError(
            "Initial-field evaluation is unavailable for source type "
            f"{type(beam).__name__}. Supported initial-field sources are PlaneWave, "
            "DipoleSource/DipoleCollection, and angular-spectrum sources."
        )

    E = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
    H = np.zeros_like(E)
    # Keep alpha quadrature consistent with RHS source projection.
    alpha_weights = periodic_azimuthal_weights(np.asarray(azimuthal_angles, float))

    beta = np.asarray(pwp_te["beta"], dtype=float)
    alpha = np.asarray(pwp_te["alpha"], dtype=float)
    coeff_te = np.asarray(pwp_te["coeff"], dtype=compute_dtype)
    coeff_tm = np.asarray(pwp_tm["coeff"], dtype=compute_dtype)

    sinb = np.sin(beta).astype(np.float64)
    cosb = np.cos(beta).astype(np.float64)
    beta_weights = trapezoidal_weights(beta).astype(np.float64)

    kx_all = np.asarray(pwp_te["kx"], dtype=np.float64)
    ky_all = np.asarray(pwp_te["ky"], dtype=np.float64)
    kz_all = np.asarray(pwp_te["kz"], dtype=np.float64)

    Np = pts.shape[0]
    Na = alpha.shape[0]
    if alpha_weights.shape[0] != Na:
        raise ValueError("azimuthal angle grid mismatch in alpha integration.")
    if beta.size == 0:
        return E, H

    batch_size_eff = int(batch_size)
    target_phase_bytes = 128 * 1024**2
    max_by_memory = max(
        1, target_phase_bytes // (int(beta.size) * int(np.dtype(compute_dtype).itemsize))
    )
    batch_size_eff = max(batch_size_eff, int(max_by_memory))

    sin_a = np.sin(alpha)[:, None]
    cos_a = np.cos(alpha)[:, None]
    sinb_row = sinb[None, :]
    cosb_row = cosb[None, :]

    weighted_beta = (sinb_row * beta_weights[None, :]).astype(compute_dtype, copy=False)
    w_te_all = coeff_te * weighted_beta
    w_tm_all = coeff_tm * weighted_beta

    khx_all = kx_all / float(k)
    khy_all = ky_all / float(k)
    khz_all = kz_all / float(k)

    ex_te = -sin_a
    ey_te = cos_a
    ez_te = np.zeros_like(ex_te)
    ex_tm = cosb_row * cos_a
    ey_tm = cosb_row * sin_a
    ez_tm = -sinb_row

    hx_te = khy_all * ez_te - khz_all * ey_te
    hy_te = khz_all * ex_te - khx_all * ez_te
    hz_te = khx_all * ey_te - khy_all * ex_te

    hx_tm = khy_all * ez_tm - khz_all * ey_tm
    hy_tm = khz_all * ex_tm - khx_all * ez_tm
    hz_tm = khx_all * ey_tm - khy_all * ex_tm

    V_all = np.empty((Na, beta.size, 6), dtype=compute_dtype)
    V_all[:, :, 0] = w_te_all * ex_te + w_tm_all * ex_tm
    V_all[:, :, 1] = w_te_all * ey_te + w_tm_all * ey_tm
    V_all[:, :, 2] = w_te_all * ez_te + w_tm_all * ez_tm
    V_all[:, :, 3] = w_te_all * hx_te + w_tm_all * hx_tm
    V_all[:, :, 4] = w_te_all * hy_te + w_tm_all * hy_tm
    V_all[:, :, 5] = w_te_all * hz_te + w_tm_all * hz_tm

    for ja in tqdm(range(Na), desc="Initial field (alpha)", total=Na, disable=not show_progress):
        alpha_w = alpha_weights[ja]
        if alpha_w == 0.0:
            continue
        active_beta = np.flatnonzero((w_te_all[ja, :] != 0) | (w_tm_all[ja, :] != 0))
        if active_beta.size == 0:
            continue
        kx = kx_all[ja, active_beta]
        ky = ky_all[ja, active_beta]
        kz = kz_all[ja, active_beta]
        V = V_all[ja, active_beta, :]
        for s in range(0, Np, batch_size_eff):
            e = min(Np, s + batch_size_eff)
            p = pts[s:e, :]
            phase_arg = (
                p[:, 0, None] * kx[None, :]
                + p[:, 1, None] * ky[None, :]
                + p[:, 2, None] * kz[None, :]
            )
            phase = np.exp(1j * phase_arg).astype(compute_dtype, copy=False)
            weighted = phase @ V

            weighted_acc = (alpha_w * weighted).astype(accum_dtype, copy=False)
            E[s:e, 0] += weighted_acc[:, 0]
            E[s:e, 1] += weighted_acc[:, 1]
            E[s:e, 2] += weighted_acc[:, 2]
            H[s:e, 0] += nM * weighted_acc[:, 3]
            H[s:e, 1] += nM * weighted_acc[:, 4]
            H[s:e, 2] += nM * weighted_acc[:, 5]

    return E, H
