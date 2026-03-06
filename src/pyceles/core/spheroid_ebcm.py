"""Axisymmetric spherical-basis EBCM helpers for spheroid-like particles.

This module is the particle-local preparation layer for future axisymmetric
`T`/`R` backends in `pyceles`. The current focus is homogeneous spheroids in a
spherical-wave basis, but the internal split is intentionally narrower:
- axisymmetric shape/profile geometry,
- radial-function products,
- per-`m` angular functions,
- per-`m` `P/Q` assembly.

That organization lets us keep the cluster solver centered on spherical SVWF
coupling while still preparing more specialized axisymmetric particle models.

Implemented so far:
- `mu = cos(theta)`-centric meridian quadrature,
- direct reference radial products in CELES/SMUTHI outgoing-wave conventions,
- one-`m` angular-function preparation,
- one-`m` raw `P/Q` block assembly.

Not implemented yet:
- `P/Q -> T/R` solves,
- conversion to final particle spherical-basis `T` blocks,
- rotated axisymmetric particle handling in the solver path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn, spherical_yn

Array = np.ndarray


@dataclass(frozen=True)
class AxisymmetricShapeQuadrature:
    """Meridian geometry and polar quadrature for an axisymmetric profile.

    The returned samples are stored in both `mu = cos(theta)` and `theta`
    form. The EBCM algebra is often cleaner in `mu`, while geometric factors
    and legacy references are usually written in `theta`. Keeping both avoids
    repeated trig conversions during `P/Q` assembly.

    When the quadrature is generated internally, the weights are doubled so
    integrating over `theta in (0, pi/2)` reproduces the full
    `theta in (0, pi)` integral for reflection-symmetric integrands.
    """

    mu: Array
    theta: Array
    sin_theta: Array
    cos_theta: Array
    weights: Array
    radius: Array
    dr_dmu: Array
    dr_dtheta: Array
    equivalent_volume_radius: float | None
    uses_gauss_half_space: bool


class AxisymmetricShapeProfile(Protocol):
    """Minimal profile contract for axisymmetric EBCM geometry preparation."""

    def radius_from_mu(self, mu: Array) -> Array: ...

    def dr_dmu(self, mu: Array) -> Array: ...

    @property
    def equivalent_volume_radius(self) -> float | None: ...


@dataclass(frozen=True)
class SpheroidShapeProfile:
    """Axisymmetric shape profile for a homogeneous spheroid.

    This keeps the particle-local geometry formulas separate from quadrature
    policy. Later axisymmetric particles can provide the same narrow interface
    without forcing the `P/Q` assembly to know about concrete particle types.
    """

    equatorial_radius: float
    polar_radius: float

    def __post_init__(self) -> None:
        if float(self.equatorial_radius) <= 0.0:
            raise ValueError("equatorial_radius must be positive.")
        if float(self.polar_radius) <= 0.0:
            raise ValueError("polar_radius must be positive.")

    @property
    def aspect_ratio(self) -> float:
        """Return the ratio of major to minor semi-axis."""

        return max(float(self.equatorial_radius), float(self.polar_radius)) / min(
            float(self.equatorial_radius), float(self.polar_radius)
        )

    @property
    def equivalent_volume_radius(self) -> float:
        """Return the radius of the equal-volume sphere."""

        a = float(self.equatorial_radius)
        c = float(self.polar_radius)
        return (a * a * c) ** (1.0 / 3.0)

    def radius_from_mu(self, mu: Array) -> Array:
        """Return the spheroid radius at `mu = cos(theta)`."""

        a = float(self.equatorial_radius)
        c = float(self.polar_radius)
        mu_arr = np.asarray(mu, dtype=float)
        sin2 = 1.0 - mu_arr * mu_arr
        denom = np.sqrt((c * c) * sin2 + (a * a) * (mu_arr * mu_arr))
        return (a * c) / denom

    def dr_dmu(self, mu: Array) -> Array:
        """Return `dr/dmu` for the spheroid profile."""

        a = float(self.equatorial_radius)
        c = float(self.polar_radius)
        mu_arr = np.asarray(mu, dtype=float)
        radius = self.radius_from_mu(mu_arr)
        return -((a * a) - (c * c)) * mu_arr * (radius**3) / ((a * c) ** 2)


@dataclass(frozen=True)
class ModifiedBesselProducts:
    """Modified radial-function products for axisymmetric EBCM integrals.

    The tensor layout follows the natural integral indices:
    - first axis: multipole order `n = 0..N+1`,
    - second axis: multipole order `k = 0..N+1`,
    - third axis: surface sample index.

    Only parity-compatible entries are populated for the base products:
    `n + k` even for `xipsi` / `psipsi`, `n + k` odd for mixed derivative
    products. This mirrors the axisymmetric selection rules and lets later
    assembly avoid redundant work.

    The current implementation computes the products directly from SciPy's
    spherical Bessel functions. The public shape is future-proof: a more
    stable SMARTIES-style `F^+` recurrence can replace the internal kernel
    later without changing the downstream `P/Q` assembly code.
    """

    xipsi: Array
    psipsi: Array
    xiprimepsi: Array
    xipsiprime: Array
    xipsi_over_sx2: Array
    xiprimepsiprime_plus_nnp1_xipsi_over_sx2: Array
    xiprimepsiprime_plus_kkp1_xipsi_over_sx2: Array
    psiprimepsi: Array
    psipsiprime: Array
    psipsi_over_sx2: Array
    psiprimepsiprime_plus_nnp1_psipsi_over_sx2: Array
    psiprimepsiprime_plus_kkp1_psipsi_over_sx2: Array
    psi_x: Array
    chi_x: Array
    psi_sx: Array
    xi_x: Array
    q11_diag_kernel: Array
    q22_diag_kernel: Array
    q22_diag_coupling_kernel: Array
    p11_diag_kernel: Array
    p22_diag_kernel: Array
    p22_diag_coupling_kernel: Array


@dataclass(frozen=True)
class AxisymmetricAngularFunctions:
    """Angular functions for one non-negative azimuthal order `m`.

    The arrays are indexed by the retained degrees `n = n_min..nmax`, where
    `n_min = max(m, 1)`. This is the natural indexing for the axisymmetric
    `m`-block algebra.
    """

    m: int
    n_values: Array
    pi_nm: Array
    tau_nm: Array
    d_nm: Array


@dataclass(frozen=True)
class AxisymmetricPQBlock:
    """Raw `P/Q` submatrices for one non-negative azimuthal order `m`."""

    m: int
    n_values: Array
    Q11: Array
    Q12: Array
    Q21: Array
    Q22: Array
    P11: Array
    P12: Array
    P21: Array
    P22: Array


def axisymmetric_shape_quadrature(
    profile: AxisymmetricShapeProfile,
    n_theta: int,
    *,
    mu: Array | None = None,
    theta: Array | None = None,
) -> AxisymmetricShapeQuadrature:
    """Return the generating-meridian geometry for one axisymmetric profile.

    Parameters
    ----------
    profile:
        Axisymmetric profile sampled on the generating meridian.
    n_theta:
        Number of polar samples on the upper half of the meridian when
        neither `mu` nor `theta` is provided.
    mu:
        Optional explicit `mu = cos(theta)` samples.
    theta:
        Optional explicit polar angles in radians. When omitted, a
        Gauss-Legendre rule on `(0, pi/2)` is generated and its weights are
        doubled to account for reflection symmetry.
    """

    if theta is not None and mu is not None:
        raise ValueError("Provide either theta or mu, not both.")

    if theta is None and mu is None:
        n_theta = int(n_theta)
        if n_theta < 1:
            raise ValueError("n_theta must be >= 1.")
        nodes, weights = leggauss(n_theta)
        mu_arr = 0.5 * (nodes + 1.0)
        theta_arr = np.arccos(mu_arr)
        weight_arr = 0.5 * np.pi * weights
        uses_gauss_half_space = True
    elif mu is not None:
        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        if mu_arr.size == 0:
            raise ValueError("mu must contain at least one sample.")
        if np.any(~np.isfinite(mu_arr)):
            raise ValueError("mu must contain only finite values.")
        if np.any((mu_arr < 0.0) | (mu_arr > 1.0)):
            raise ValueError("mu samples must lie in [0, 1] for the upper meridian.")
        theta_arr = np.arccos(mu_arr)
        weight_arr = np.zeros_like(mu_arr)
        uses_gauss_half_space = False
    else:
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        if theta_arr.size == 0:
            raise ValueError("theta must contain at least one sample.")
        if np.any(~np.isfinite(theta_arr)):
            raise ValueError("theta must contain only finite values.")
        if np.any((theta_arr < 0.0) | (theta_arr > 0.5 * np.pi)):
            raise ValueError("theta samples must lie in [0, pi/2] for the upper meridian.")
        mu_arr = np.cos(theta_arr)
        weight_arr = np.zeros_like(theta_arr)
        uses_gauss_half_space = False

    sin_theta = np.sin(theta_arr)
    cos_theta = mu_arr
    radius = np.asarray(profile.radius_from_mu(mu_arr), dtype=float)
    dr_dmu = np.asarray(profile.dr_dmu(mu_arr), dtype=float)
    dr_dtheta = -sin_theta * dr_dmu

    return AxisymmetricShapeQuadrature(
        mu=mu_arr,
        theta=theta_arr,
        sin_theta=sin_theta,
        cos_theta=cos_theta,
        weights=weight_arr,
        radius=radius,
        dr_dmu=dr_dmu,
        dr_dtheta=dr_dtheta,
        equivalent_volume_radius=profile.equivalent_volume_radius,
        uses_gauss_half_space=uses_gauss_half_space,
    )


def spheroid_geometry_quadrature(
    n_theta: int,
    equatorial_radius: float,
    polar_radius: float,
    *,
    mu: Array | None = None,
    theta: Array | None = None,
) -> AxisymmetricShapeQuadrature:
    """Return the generating-meridian geometry for a spheroid.

    This is a thin convenience wrapper around the generic axisymmetric shape
    quadrature. Spheroids remain the first concrete profile, but `P/Q`
    assembly should build on the generic `AxisymmetricShapeProfile` contract.
    """

    return axisymmetric_shape_quadrature(
        SpheroidShapeProfile(
            equatorial_radius=float(equatorial_radius),
            polar_radius=float(polar_radius),
        ),
        n_theta=n_theta,
        mu=mu,
        theta=theta,
    )


def _riccati_psi_table(max_order: int, z: Array) -> Array:
    """Return `psi_n(z) = z j_n(z)` for `n = 0..max_order`."""

    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.empty((z_arr.size, max_order + 1), dtype=np.complex128)
    for n in range(max_order + 1):
        out[:, n] = z_arr * spherical_jn(n, z_arr)
    return out


def _riccati_chi_table(max_order: int, z: Array) -> Array:
    """Return `chi_n(z) = z y_n(z)` for `n = 0..max_order`.

    The sign convention matches the outgoing combination `xi = psi + i chi`
    used by CELES/SMUTHI and by the SMARTIES EBCM derivation.
    """

    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.empty((z_arr.size, max_order + 1), dtype=np.complex128)
    for n in range(max_order + 1):
        out[:, n] = z_arr * spherical_yn(n, z_arr)
    return out


def _riccati_dpsi_table(max_order: int, z: Array) -> Array:
    """Return derivatives `psi'_n(z)` with respect to the argument `z`."""

    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.empty((z_arr.size, max_order + 1), dtype=np.complex128)
    for n in range(max_order + 1):
        out[:, n] = spherical_jn(n, z_arr) + z_arr * spherical_jn(n, z_arr, derivative=True)
    return out


def _riccati_dchi_table(max_order: int, z: Array) -> Array:
    """Return derivatives `chi'_n(z)` with respect to the argument `z`."""

    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.empty((z_arr.size, max_order + 1), dtype=np.complex128)
    for n in range(max_order + 1):
        out[:, n] = spherical_yn(n, z_arr) + z_arr * spherical_yn(n, z_arr, derivative=True)
    return out


def _parity_masks(max_order: int) -> tuple[Array, Array]:
    """Return boolean masks for even and odd `n + k` parity."""

    orders = np.arange(max_order + 1, dtype=np.int64)
    parity = (orders[:, None] + orders[None, :]) % 2
    even_mask = parity == 0
    return even_mask, ~even_mask


def modified_bessel_products(
    nmax: int,
    relative_refractive_index: complex,
    x: Array,
) -> ModifiedBesselProducts:
    """Return axisymmetric radial-function products up to order `nmax`.

    Parameters
    ----------
    nmax:
        Maximum multipole degree used in the EBCM integrals. The returned base
        products include the extra `n = N+1` and `k = N+1` rows/columns needed
        by the derivative identities.
    relative_refractive_index:
        Ratio `s = n_particle / n_medium`.
    x:
        Size parameters on the surface quadrature nodes, typically `x = k r`.

    Notes
    -----
    The returned products follow the same selection rules used by axisymmetric
    EBCM derivations:
    - `xipsi` / `psipsi` are populated only for `n + k` even,
    - `xiprimepsi` / `xipsiprime` and their regular-field analogues are
      populated only for `n + k` odd.

    This keeps the tensor layout close to the final `P/Q` assembly while
    preserving a straightforward SciPy-based reference implementation.
    """

    nmax = int(nmax)
    if nmax < 1:
        raise ValueError("nmax must be >= 1.")

    s = complex(relative_refractive_index)
    if s == 0:
        raise ValueError("relative_refractive_index must be non-zero.")

    x_arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x_arr.size == 0:
        raise ValueError("x must contain at least one sample.")
    if np.any(np.isclose(x_arr, 0.0)):
        raise ValueError("x must stay away from zero for the current reference kernels.")

    max_order = nmax + 1
    psi_x = _riccati_psi_table(max_order, x_arr)
    chi_x = _riccati_chi_table(max_order, x_arr)
    dpsi_x = _riccati_dpsi_table(max_order, x_arr)
    dchi_x = _riccati_dchi_table(max_order, x_arr)

    sx = s * x_arr
    psi_sx = _riccati_psi_table(max_order, sx)
    dpsi_sx = _riccati_dpsi_table(max_order, sx)

    xi_x = psi_x + 1j * chi_x
    dxi_x = dpsi_x + 1j * dchi_x

    shape = (max_order + 1, max_order + 1, x_arr.size)
    xipsi = np.zeros(shape, dtype=np.complex128)
    psipsi = np.zeros(shape, dtype=np.complex128)
    xiprimepsi = np.zeros(shape, dtype=np.complex128)
    xipsiprime = np.zeros(shape, dtype=np.complex128)
    xipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    xiprimepsiprime_plus_nnp1_xipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    xiprimepsiprime_plus_kkp1_xipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    psiprimepsi = np.zeros(shape, dtype=np.complex128)
    psipsiprime = np.zeros(shape, dtype=np.complex128)
    psipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    psiprimepsiprime_plus_nnp1_psipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    psiprimepsiprime_plus_kkp1_psipsi_over_sx2 = np.zeros(shape, dtype=np.complex128)
    q11_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    q22_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    q22_diag_coupling_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p11_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p22_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p22_diag_coupling_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)

    even_mask, odd_mask = _parity_masks(max_order)
    inv_sx2 = 1.0 / (s * (x_arr * x_arr))

    for n in range(max_order + 1):
        xi_n = xi_x[:, n]
        psi_n = psi_x[:, n]
        dxi_n = dxi_x[:, n]
        dpsi_n = dpsi_x[:, n]
        nnp1 = n * (n + 1)
        for k in range(max_order + 1):
            psi_k = psi_sx[:, k]
            dpsi_k = dpsi_sx[:, k]
            kkp1 = k * (k + 1)
            if even_mask[n, k]:
                base_xi = xi_n * psi_k
                base_psi = psi_n * psi_k
                xipsi[n, k, :] = base_xi
                psipsi[n, k, :] = base_psi
                xipsi_over_sx2[n, k, :] = base_xi * inv_sx2
                psipsi_over_sx2[n, k, :] = base_psi * inv_sx2
                xiprimepsiprime = dxi_n * dpsi_k
                psiprimepsiprime = dpsi_n * dpsi_k
                xiprimepsiprime_plus_nnp1_xipsi_over_sx2[n, k, :] = (
                    xiprimepsiprime + nnp1 * xipsi_over_sx2[n, k, :]
                )
                xiprimepsiprime_plus_kkp1_xipsi_over_sx2[n, k, :] = (
                    xiprimepsiprime + kkp1 * xipsi_over_sx2[n, k, :]
                )
                psiprimepsiprime_plus_nnp1_psipsi_over_sx2[n, k, :] = (
                    psiprimepsiprime + nnp1 * psipsi_over_sx2[n, k, :]
                )
                psiprimepsiprime_plus_kkp1_psipsi_over_sx2[n, k, :] = (
                    psiprimepsiprime + kkp1 * psipsi_over_sx2[n, k, :]
                )
            elif odd_mask[n, k]:
                xiprimepsi[n, k, :] = dxi_n * psi_k
                xipsiprime[n, k, :] = xi_n * dpsi_k
                psiprimepsi[n, k, :] = dpsi_n * psi_k
                psipsiprime[n, k, :] = psi_n * dpsi_k

    inv_x = 1.0 / x_arr
    for n in range(1, nmax + 1):
        psi_n_x = psi_x[:, n]
        psi_np1_x = psi_x[:, n + 1]
        psi_n_sx = psi_sx[:, n]
        psi_np1_sx = psi_sx[:, n + 1]
        xi_n_x = xi_x[:, n]
        xi_np1_x = xi_x[:, n + 1]

        p11_diag_kernel[n - 1, :] = s * psi_n_x * psi_np1_sx - psi_np1_x * psi_n_sx
        q11_diag_kernel[n - 1, :] = s * xi_n_x * psi_np1_sx - xi_np1_x * psi_n_sx

        common_prefactor = ((s - 1.0) * (s + 1.0) / s) * (n + 1) * inv_x
        p22_diag_kernel[n - 1, :] = (
            psi_n_x * psi_np1_sx - s * psi_np1_x * psi_n_sx + common_prefactor * psi_n_x * psi_n_sx
        )
        q22_diag_kernel[n - 1, :] = (
            xi_n_x * psi_np1_sx - s * xi_np1_x * psi_n_sx + common_prefactor * xi_n_x * psi_n_sx
        )
        p22_diag_coupling_kernel[n - 1, :] = (psi_n_x * psi_n_sx) / (s * (x_arr * x_arr))
        q22_diag_coupling_kernel[n - 1, :] = (xi_n_x * psi_n_sx) / (s * (x_arr * x_arr))

    return ModifiedBesselProducts(
        xipsi=xipsi,
        psipsi=psipsi,
        xiprimepsi=xiprimepsi,
        xipsiprime=xipsiprime,
        xipsi_over_sx2=xipsi_over_sx2,
        xiprimepsiprime_plus_nnp1_xipsi_over_sx2=(xiprimepsiprime_plus_nnp1_xipsi_over_sx2),
        xiprimepsiprime_plus_kkp1_xipsi_over_sx2=(xiprimepsiprime_plus_kkp1_xipsi_over_sx2),
        psiprimepsi=psiprimepsi,
        psipsiprime=psipsiprime,
        psipsi_over_sx2=psipsi_over_sx2,
        psiprimepsiprime_plus_nnp1_psipsi_over_sx2=(psiprimepsiprime_plus_nnp1_psipsi_over_sx2),
        psiprimepsiprime_plus_kkp1_psipsi_over_sx2=(psiprimepsiprime_plus_kkp1_psipsi_over_sx2),
        psi_x=psi_x,
        chi_x=chi_x,
        psi_sx=psi_sx,
        xi_x=xi_x,
        q11_diag_kernel=q11_diag_kernel,
        q22_diag_kernel=q22_diag_kernel,
        q22_diag_coupling_kernel=q22_diag_coupling_kernel,
        p11_diag_kernel=p11_diag_kernel,
        p22_diag_kernel=p22_diag_kernel,
        p22_diag_coupling_kernel=p22_diag_coupling_kernel,
    )


def axisymmetric_angular_functions(
    nmax: int,
    m: int,
    quadrature: AxisymmetricShapeQuadrature,
) -> AxisymmetricAngularFunctions:
    """Return `pi_nm`, `tau_nm`, and `d_n` for one non-negative `m`.

    The recurrence follows the SMARTIES/Mishchenko convention but is stored in
    a simpler `(n_index, sample_index)` layout tailored to the subsequent
    axisymmetric `P/Q` assembly.
    """

    nmax = int(nmax)
    m = int(m)
    if nmax < 1:
        raise ValueError("nmax must be >= 1.")
    if m < 0 or m > nmax:
        raise ValueError("m must satisfy 0 <= m <= nmax.")

    mu = np.asarray(quadrature.mu, dtype=float)
    sin_theta = np.asarray(quadrature.sin_theta, dtype=float)
    n_min = max(m, 1)
    n_values = np.arange(n_min, nmax + 1, dtype=np.int64)
    n_count = int(n_values.size)

    pi_nm = np.zeros((n_count, mu.size), dtype=np.float64)
    tau_nm = np.zeros((n_count, mu.size), dtype=np.float64)
    d_nm = np.zeros((n_count, mu.size), dtype=np.float64)

    if m == 0:
        p_prev = np.ones_like(mu)
        p_curr = mu.copy()
        t_curr = -sin_theta.copy()
        for idx, n in enumerate(n_values):
            if n == 1:
                p_n = p_curr
                t_n = t_curr
            else:
                p_next = ((2 * n - 1) / n) * mu * p_curr - ((n - 1) / n) * p_prev
                t_next = mu * t_curr - n * sin_theta * p_curr
                p_prev, p_curr = p_curr, p_next
                t_curr = t_next
                p_n = p_curr
                t_n = t_curr
            d_nm[idx, :] = p_n
            tau_nm[idx, :] = t_n
        return AxisymmetricAngularFunctions(
            m=m,
            n_values=n_values,
            pi_nm=pi_nm,
            tau_nm=tau_nm,
            d_nm=d_nm,
        )

    am_sin_m_minus_1 = np.sqrt((2 * m - 1) / (2 * m)) * np.power(sin_theta, m - 1)
    piaux = np.zeros((n_count + 1, mu.size), dtype=np.float64)
    piaux[1, :] = m * am_sin_m_minus_1
    for jj in range(2, n_count + 1):
        n_recur = m + jj - 1
        piaux[jj, :] = (
            (2 * n_recur - 1) * mu * piaux[jj - 1, :]
            - np.sqrt((n_recur - 1 - m) * (n_recur - 1 + m)) * piaux[jj - 2, :]
        ) / np.sqrt((n_recur - m) * (n_recur + m))

    for idx, n in enumerate(n_values):
        col = idx + (n_min - m) + 1
        pi_n = piaux[col, :]
        prev = piaux[col - 1, :]
        tau_n = (-np.sqrt((n - m) * (n + m)) / m) * prev + mu * (n / m) * pi_n
        pi_nm[idx, :] = pi_n
        tau_nm[idx, :] = tau_n
        d_nm[idx, :] = (sin_theta / m) * pi_n

    return AxisymmetricAngularFunctions(
        m=m,
        n_values=n_values,
        pi_nm=pi_nm,
        tau_nm=tau_nm,
        d_nm=d_nm,
    )


def assemble_axisymmetric_pq_block(
    relative_refractive_index: complex,
    quadrature: AxisymmetricShapeQuadrature,
    angular: AxisymmetricAngularFunctions,
    radial: ModifiedBesselProducts,
) -> AxisymmetricPQBlock:
    """Assemble the raw `P/Q` matrices for one axisymmetric `m` block.

    This implements the first spherical-basis reference path for spheroid-like
    axisymmetric particles. The returned matrices still use the compact `n`
    indexing of one `m` block; later steps can split even/odd sectors or
    convert them to the final particle `T/R` blocks.
    """

    s = complex(relative_refractive_index)
    n_values = np.asarray(angular.n_values, dtype=np.int64)
    n_count = int(n_values.size)
    if n_count == 0:
        raise ValueError("angular.n_values must be non-empty.")

    weights = np.asarray(quadrature.weights, dtype=np.float64)
    dxdtwt = np.asarray(quadrature.dr_dtheta, dtype=np.float64) * weights
    An = np.sqrt((2.0 * n_values + 1.0) / (2.0 * n_values * (n_values + 1.0)))
    prefactor1 = ((s - 1.0) * (s + 1.0) / s) * (An[:, None] * An[None, :])
    nnp1 = n_values * (n_values + 1)

    K1 = np.zeros((n_count, n_count), dtype=np.complex128)
    K2 = np.zeros((n_count, n_count), dtype=np.complex128)
    L5 = np.zeros((n_count, n_count), dtype=np.complex128)
    L6 = np.zeros((n_count, n_count), dtype=np.complex128)
    K1P = np.zeros((n_count, n_count), dtype=np.complex128)
    K2P = np.zeros((n_count, n_count), dtype=np.complex128)
    L5P = np.zeros((n_count, n_count), dtype=np.complex128)
    L6P = np.zeros((n_count, n_count), dtype=np.complex128)

    pinm = np.asarray(angular.pi_nm, dtype=np.float64)
    taunm = np.asarray(angular.tau_nm, dtype=np.float64)
    dn = np.asarray(angular.d_nm, dtype=np.float64)
    dntimesnnp1 = dn * nnp1[:, None]

    for k_idx, k in enumerate(n_values):
        dxdttauksint = dxdtwt * taunm[k_idx, :]
        dxdtdksint = dxdtwt * dn[k_idx, :]

        xiprimepsi = radial.xiprimepsi[n_values, k, :]
        xipsiprime = radial.xipsiprime[n_values, k, :]
        xipsi = radial.xipsi[n_values, k, :]
        xiprimepsiprime_plus_kkp1 = radial.xiprimepsiprime_plus_kkp1_xipsi_over_sx2[n_values, k, :]
        xiprimepsiprime_plus_nnp1 = radial.xiprimepsiprime_plus_nnp1_xipsi_over_sx2[n_values, k, :]

        psiprimepsi = radial.psiprimepsi[n_values, k, :]
        psipsiprime = radial.psipsiprime[n_values, k, :]
        psipsi = radial.psipsi[n_values, k, :]
        psiprimepsiprime_plus_kkp1 = radial.psiprimepsiprime_plus_kkp1_psipsi_over_sx2[
            n_values, k, :
        ]
        psiprimepsiprime_plus_nnp1 = radial.psiprimepsiprime_plus_nnp1_psipsi_over_sx2[
            n_values, k, :
        ]

        K1[:, k_idx] = (pinm * xipsiprime) @ dxdtdksint
        K2[:, k_idx] = (pinm * xiprimepsi) @ dxdtdksint
        K1P[:, k_idx] = (pinm * psipsiprime) @ dxdtdksint
        K2P[:, k_idx] = (pinm * psiprimepsi) @ dxdtdksint

        L5[:, k_idx] = (dntimesnnp1 * xipsi) @ dxdttauksint - (taunm * xipsi) @ dxdtdksint * (
            k * (k + 1)
        )
        L5P[:, k_idx] = (dntimesnnp1 * psipsi) @ dxdttauksint - (taunm * psipsi) @ dxdtdksint * (
            k * (k + 1)
        )

        L6[:, k_idx] = (dntimesnnp1 * xiprimepsiprime_plus_kkp1) @ dxdttauksint - (
            taunm * xiprimepsiprime_plus_nnp1
        ) @ dxdtdksint * (k * (k + 1))
        L6P[:, k_idx] = (dntimesnnp1 * psiprimepsiprime_plus_kkp1) @ dxdttauksint - (
            taunm * psiprimepsiprime_plus_nnp1
        ) @ dxdtdksint * (k * (k + 1))

    denom = nnp1[:, None] - nnp1[None, :]
    prefactor2 = np.zeros_like(prefactor1, dtype=np.complex128)
    off_diag = ~np.eye(n_count, dtype=bool)
    prefactor2[off_diag] = 1j * prefactor1[off_diag] / denom[off_diag]

    Q12 = prefactor1 * K1
    Q21 = -prefactor1 * K2
    Q11 = prefactor2 * L5
    Q22 = prefactor2 * L6
    P12 = prefactor1 * K1P
    P21 = -prefactor1 * K2P
    P11 = prefactor2 * L5P
    P22 = prefactor2 * L6P

    pref_diag1 = (-1j / s) * (2.0 * n_values + 1.0) / (2.0 * n_values * (n_values + 1.0))
    pref_diag2 = (-1j * (s - 1.0) * (s + 1.0) / s / 2.0) * (2.0 * n_values + 1.0)
    pi2ptau2 = pinm * pinm + taunm * taunm

    q11_diag = pref_diag1 * np.sum(
        pi2ptau2 * radial.q11_diag_kernel[n_values - 1, :] * weights, axis=1
    )
    p11_diag = pref_diag1 * np.sum(
        pi2ptau2 * radial.p11_diag_kernel[n_values - 1, :] * weights, axis=1
    )
    q22_diag = pref_diag1 * np.sum(
        pi2ptau2 * radial.q22_diag_kernel[n_values - 1, :] * weights, axis=1
    ) + (
        pref_diag2
        * np.sum(dn * taunm * radial.q22_diag_coupling_kernel[n_values - 1, :] * dxdtwt, axis=1)
    )
    p22_diag = pref_diag1 * np.sum(
        pi2ptau2 * radial.p22_diag_kernel[n_values - 1, :] * weights, axis=1
    ) + (
        pref_diag2
        * np.sum(dn * taunm * radial.p22_diag_coupling_kernel[n_values - 1, :] * dxdtwt, axis=1)
    )

    np.fill_diagonal(Q11, q11_diag)
    np.fill_diagonal(Q22, q22_diag)
    np.fill_diagonal(P11, p11_diag)
    np.fill_diagonal(P22, p22_diag)

    return AxisymmetricPQBlock(
        m=int(angular.m),
        n_values=n_values,
        Q11=Q11,
        Q12=Q12,
        Q21=Q21,
        Q22=Q22,
        P11=P11,
        P12=P12,
        P21=P21,
        P22=P22,
    )
