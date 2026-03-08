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
- one-`m` raw `P/Q` block assembly,
- parity-reduced `T/R` reference solves.

Not implemented yet:
- conversion to final particle spherical-basis `T` blocks,
- rotated axisymmetric particle handling in the solver path.

Precision policy
----------------
This preparation layer currently runs in `float64` / `complex128`
unconditionally. These are particle-local setup kernels and small dense solves,
not the repeated cluster-level matvec hot path, so the default here favors
robustness over `compute_dtype` plumbing for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn, spherical_yn

from .indexing import index_vswf, n_modes

Array = np.ndarray


def _double_factorial(n: int) -> int:
    """Return the double factorial for small integer normalization helpers."""

    n = int(n)
    if n in (-1, 0):
        return 1
    if n < -1:
        raise ValueError("double factorial is undefined for n < -1.")
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _axisymmetric_sector_scale(m: int) -> float:
    """Return the per-|m| normalization for compact axisymmetric `P/Q` blocks.

    The meridional EBCM reduction integrates out the azimuth analytically and
    produces one compact block per non-negative `|m|`. In the CELES/SMUTHI
    spherical-wave conventions used by `pyceles`, that compact block carries an
    extra scalar normalization that depends only on `|m|`.

    This factor cancels in `T = -P Q^{-1}`, so the scattering block is
    unaffected, but it matters for internal-response maps such as `-P^{-1}` or
    `Q^{-1}`. Applying it here keeps both the external and internal particle
    responses in the same CELES mode normalization.
    """

    m = abs(int(m))
    if m == 0:
        return 1.0
    return _double_factorial(2 * m - 3) / _double_factorial(2 * m - 2)


@dataclass(frozen=True)
class AxisymmetricShapeQuadrature:
    """Meridian geometry and polar quadrature for an axisymmetric profile.

    The returned samples are stored in both `mu = cos(theta)` and `theta`
    form. The EBCM algebra is often cleaner in `mu`, while geometric factors
    and legacy references are usually written in `theta`. Keeping both avoids
    repeated trig conversions during `P/Q` assembly.

    When the quadrature is generated internally, the nodes come from a
    Gauss-Legendre rule in `mu = cos(theta)` on the upper meridian
    `mu in (0, 1)`. The stored weights are doubled so the upper-meridian
    rule reproduces the full `mu in (-1, 1)` integral for reflection-
    symmetric integrands, matching the SMARTIES spheroid preparation
    convention.
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
    spherical Bessel functions because that path is both accurate and fast for
    the moderate aspect ratios currently targeted by `pyceles`. If extreme
    aspect ratios become a priority later, a more cancellation-resistant
    SMARTIES-style `F^+` recurrence can slot in behind the same tensor layout.
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


@dataclass(frozen=True)
class AxisymmetricParityBlock:
    """Even/odd parity block form for one axisymmetric `m` sector."""

    m: int
    even_indices: Array
    odd_indices: Array
    Q11: Array
    Q12: Array
    Q21: Array
    Q22: Array
    P11: Array
    P12: Array
    P21: Array
    P22: Array


@dataclass(frozen=True)
class AxisymmetricTRBlock:
    """Solved `T` and optional `R` blocks for one parity-reduced `m` sector."""

    m: int
    even_indices: Array
    odd_indices: Array
    T11: Array
    T12: Array
    T21: Array
    T22: Array
    R11: Array | None
    R12: Array | None
    R21: Array | None
    R22: Array | None


@dataclass(frozen=True)
class AxisymmetricResolvedMBlock:
    """Full fixed-`m` `T/R` block after recombining the parity subsectors."""

    m: int
    n_values: Array
    T11: Array
    T12: Array
    T21: Array
    T22: Array
    R11: Array | None
    R12: Array | None
    R21: Array | None
    R22: Array | None


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
        nodes, weights = leggauss(2 * n_theta)
        mu_arr = nodes[n_theta:]
        theta_arr = np.arccos(mu_arr)
        weight_arr = 2.0 * weights[n_theta:]
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


def _parity_masks(max_order: int) -> tuple[Array, Array]:
    """Return boolean masks for even and odd `n + k` parity."""

    orders = np.arange(max_order + 1, dtype=np.int64)
    parity = (orders[:, None] + orders[None, :]) % 2
    even_mask = parity == 0
    return even_mask, ~even_mask


def _bessel_products_with_derivatives(
    base_products: Array, s: complex, x: Array
) -> dict[str, Array]:
    """Return derivative-related tensors derived from parity-even base products."""

    base = np.asarray(base_products, dtype=np.complex128)
    nmax = base.shape[0] - 2
    x_arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    shape = (nmax + 2, nmax + 2, x_arr.size)
    odd_shape = (nmax + 2, nmax + 2, x_arr.size)
    xiprimepsi = np.zeros(odd_shape, dtype=np.complex128)
    xipsiprime = np.zeros(odd_shape, dtype=np.complex128)
    xiprimepsiprime_plus_nnp1 = np.zeros(shape, dtype=np.complex128)
    xiprimepsiprime_plus_kkp1 = np.zeros(shape, dtype=np.complex128)
    base_over_sx2 = np.zeros(shape, dtype=np.complex128)

    for n in range(1, nmax + 1):
        for k in range(2 - (n % 2), nmax + 1, 2):
            xiprimepsiprime_plus_kkp1[n, k, :] = (
                (k + (n + 1)) * (k + 1) * base[n - 1, k - 1, :]
                + (k * (k + 1) - k * (n + 1)) * base[n - 1, k + 1, :]
                + (k * (k + 1) - (k + 1) * n) * base[n + 1, k - 1, :]
                + (k * (k + 1) + k * n) * base[n + 1, k + 1, :]
            ) / ((2 * n + 1) * (2 * k + 1))
            xiprimepsiprime_plus_nnp1[n, k, :] = (
                (n * (n + 1) + (k + 1) * (n + 1)) * base[n - 1, k - 1, :]
                + (n - k) * (n + 1) * base[n - 1, k + 1, :]
                + ((n + 1) - (k + 1)) * n * base[n + 1, k - 1, :]
                + (n * (n + 1) + k * n) * base[n + 1, k + 1, :]
            ) / ((2 * n + 1) * (2 * k + 1))
        for k in range(1 + (n % 2), nmax + 1, 2):
            xiprimepsi[n, k, :] = ((n + 1) * base[n - 1, k, :] - n * base[n + 1, k, :]) / (
                2 * n + 1
            )
            xipsiprime[n, k, :] = ((k + 1) * base[n, k - 1, :] - k * base[n, k + 1, :]) / (
                2 * k + 1
            )

    base_over_sx2[1 : nmax + 1, 1 : nmax + 1, :] = (
        base[1 : nmax + 1, 1 : nmax + 1, :] / (s * x_arr * x_arr)[None, None, :]
    )
    return {
        "xipsi": base,
        "xiprimepsi": xiprimepsi,
        "xipsiprime": xipsiprime,
        "xipsi_over_sx2": base_over_sx2,
        "xiprimepsiprime_plus_nnp1": xiprimepsiprime_plus_nnp1,
        "xiprimepsiprime_plus_kkp1": xiprimepsiprime_plus_kkp1,
    }


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

    This reference implementation keeps the radial products explicit in terms
    of `xi_n(x) psi_k(sx)` and `psi_n(x) psi_k(sx)`. For the moderate aspect
    ratios currently targeted by `pyceles`, this simpler direct path is both
    accurate and slightly faster than the more elaborate SMARTIES `F^+` / `NB`
    machinery. If extreme aspect ratios become a priority later, that more
    cancellation-resistant scheme can be reintroduced without changing the
    downstream `P/Q` assembly.
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
    sx = s * x_arr
    psi_sx = _riccati_psi_table(max_order, sx)
    xi_x = psi_x + 1j * chi_x
    dpsi_x = _riccati_dpsi_table(max_order, x_arr)
    dpsi_sx = _riccati_dpsi_table(max_order, sx)

    shape = (max_order + 1, max_order + 1, x_arr.size)
    xipsi = np.zeros(shape, dtype=np.complex128)
    psipsi = np.zeros(shape, dtype=np.complex128)
    q11_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    q22_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    q22_diag_coupling_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p11_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p22_diag_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)
    p22_diag_coupling_kernel = np.zeros((nmax, x_arr.size), dtype=np.complex128)

    even_mask, odd_mask = _parity_masks(max_order)

    for n in range(max_order + 1):
        xi_n = xi_x[:, n]
        psi_n = psi_x[:, n]
        for k in range(max_order + 1):
            psi_k = psi_sx[:, k]
            if even_mask[n, k]:
                xipsi[n, k, :] = xi_n * psi_k
                base_psi = psi_n * psi_k
                psipsi[n, k, :] = base_psi

    xi_terms = _bessel_products_with_derivatives(xipsi, s, x_arr)
    psi_terms = _bessel_products_with_derivatives(psipsi, s, x_arr)

    psiprimepsi = np.zeros(shape, dtype=np.complex128)
    psipsiprime = np.zeros(shape, dtype=np.complex128)
    for n in range(max_order + 1):
        dpsi_n = dpsi_x[:, n]
        psi_n = psi_x[:, n]
        for k in range(max_order + 1):
            if odd_mask[n, k]:
                dpsi_k = dpsi_sx[:, k]
                psi_k = psi_sx[:, k]
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
        xiprimepsi=xi_terms["xiprimepsi"],
        xipsiprime=xi_terms["xipsiprime"],
        xipsi_over_sx2=xi_terms["xipsi_over_sx2"],
        xiprimepsiprime_plus_nnp1_xipsi_over_sx2=xi_terms["xiprimepsiprime_plus_nnp1"],
        xiprimepsiprime_plus_kkp1_xipsi_over_sx2=xi_terms["xiprimepsiprime_plus_kkp1"],
        psiprimepsi=psiprimepsi,
        psipsiprime=psipsiprime,
        psipsi_over_sx2=psi_terms["xipsi_over_sx2"],
        psiprimepsiprime_plus_nnp1_psipsi_over_sx2=psi_terms["xiprimepsiprime_plus_nnp1"],
        psiprimepsiprime_plus_kkp1_psipsi_over_sx2=psi_terms["xiprimepsiprime_plus_kkp1"],
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
    *,
    k_medium: complex,
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
    # SMARTIES writes this as dxdtwt with x(theta) = k * r(theta).
    # The derivative-coupled terms therefore need d(k r)/dtheta, not just dr/dtheta.
    dxdtwt = np.asarray(complex(k_medium) * quadrature.dr_dtheta, dtype=np.complex128) * weights
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

    sector_scale = _axisymmetric_sector_scale(int(angular.m))
    Q11 *= sector_scale
    Q12 *= sector_scale
    Q21 *= sector_scale
    Q22 *= sector_scale
    P11 *= sector_scale
    P12 *= sector_scale
    P21 *= sector_scale
    P22 *= sector_scale

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


def split_axisymmetric_pq_block_by_parity(
    block: AxisymmetricPQBlock,
) -> tuple[AxisymmetricParityBlock, AxisymmetricParityBlock]:
    """Split one raw `P/Q` block into the two reflection-parity subsectors.

    The raw `P/Q` matrices are indexed by consecutive `n` values for one `m`.
    For axisymmetric particles with reflection symmetries, these matrices admit
    a narrower block structure. We return the two complementary subsectors:

    - `even_odd`: the `11` family keeps even-degree rows/columns, while the
      `22` family keeps odd-degree rows/columns.
    - `odd_even`: the complementary sector, with odd degrees in the `11`
      family and even degrees in the `22` family.
    """

    n_values = np.asarray(block.n_values, dtype=np.int64)
    even_mask = (n_values % 2) == 0
    odd_mask = ~even_mask

    even_indices = np.flatnonzero(even_mask).astype(np.int64)
    odd_indices = np.flatnonzero(odd_mask).astype(np.int64)

    even_odd = AxisymmetricParityBlock(
        m=int(block.m),
        even_indices=even_indices,
        odd_indices=odd_indices,
        Q11=np.asarray(block.Q11[np.ix_(even_indices, even_indices)], dtype=np.complex128),
        Q12=np.asarray(block.Q12[np.ix_(even_indices, odd_indices)], dtype=np.complex128),
        Q21=np.asarray(block.Q21[np.ix_(odd_indices, even_indices)], dtype=np.complex128),
        Q22=np.asarray(block.Q22[np.ix_(odd_indices, odd_indices)], dtype=np.complex128),
        P11=np.asarray(block.P11[np.ix_(even_indices, even_indices)], dtype=np.complex128),
        P12=np.asarray(block.P12[np.ix_(even_indices, odd_indices)], dtype=np.complex128),
        P21=np.asarray(block.P21[np.ix_(odd_indices, even_indices)], dtype=np.complex128),
        P22=np.asarray(block.P22[np.ix_(odd_indices, odd_indices)], dtype=np.complex128),
    )
    odd_even = AxisymmetricParityBlock(
        m=int(block.m),
        even_indices=odd_indices,
        odd_indices=even_indices,
        Q11=np.asarray(block.Q11[np.ix_(odd_indices, odd_indices)], dtype=np.complex128),
        Q12=np.asarray(block.Q12[np.ix_(odd_indices, even_indices)], dtype=np.complex128),
        Q21=np.asarray(block.Q21[np.ix_(even_indices, odd_indices)], dtype=np.complex128),
        Q22=np.asarray(block.Q22[np.ix_(even_indices, even_indices)], dtype=np.complex128),
        P11=np.asarray(block.P11[np.ix_(odd_indices, odd_indices)], dtype=np.complex128),
        P12=np.asarray(block.P12[np.ix_(odd_indices, even_indices)], dtype=np.complex128),
        P21=np.asarray(block.P21[np.ix_(even_indices, odd_indices)], dtype=np.complex128),
        P22=np.asarray(block.P22[np.ix_(even_indices, even_indices)], dtype=np.complex128),
    )
    return even_odd, odd_even


def _solve_right_inverse(matrix: Array) -> Array:
    """Return the right inverse of a small dense block via `solve(A, I)`.

    The particle-local EBCM algebra needs `Q^{-1}` only as an operator acting
    on the right. Solving against the identity keeps that intent explicit and
    avoids forming an inverse through a separate matrix-inversion routine.
    """

    matrix_arr = np.asarray(matrix, dtype=np.complex128)
    ident = np.eye(matrix_arr.shape[0], dtype=np.complex128)
    return np.linalg.solve(matrix_arr, ident)


def solve_axisymmetric_tr_block(
    parity_block: AxisymmetricParityBlock,
    *,
    include_internal: bool = True,
) -> AxisymmetricTRBlock:
    """Solve one parity-reduced particle-local EBCM system for `T` and `R`.

    In the axisymmetric EBCM formulation, boundary matching first produces the
    particle-local matrices `P` and `Q`. The scattering and internal-response
    blocks are then defined by the dense linear systems

    `T Q = -P`
    `R Q = I`

    on each fixed-`m`, fixed-parity sector. These solves are completely local
    to one particle shape and one azimuthal order: their matrix dimension is
    the number of retained degrees in that sector, so it grows with `lmax`.
    """

    m = int(parity_block.m)
    Q11 = np.asarray(parity_block.Q11, dtype=np.complex128)
    Q12 = np.asarray(parity_block.Q12, dtype=np.complex128)
    Q21 = np.asarray(parity_block.Q21, dtype=np.complex128)
    Q22 = np.asarray(parity_block.Q22, dtype=np.complex128)
    P11 = np.asarray(parity_block.P11, dtype=np.complex128)
    P12 = np.asarray(parity_block.P12, dtype=np.complex128)
    P21 = np.asarray(parity_block.P21, dtype=np.complex128)
    P22 = np.asarray(parity_block.P22, dtype=np.complex128)

    n_even = int(Q11.shape[0])
    n_odd = int(Q22.shape[0])

    if m == 0:
        R11 = _solve_right_inverse(Q11) if include_internal else None
        R22 = _solve_right_inverse(Q22) if include_internal else None
        T11 = -P11 @ (R11 if R11 is not None else _solve_right_inverse(Q11))
        T22 = -P22 @ (R22 if R22 is not None else _solve_right_inverse(Q22))
        T12 = np.zeros((n_even, n_odd), dtype=np.complex128)
        T21 = np.zeros((n_odd, n_even), dtype=np.complex128)
        R12 = np.zeros((n_even, n_odd), dtype=np.complex128) if include_internal else None
        R21 = np.zeros((n_odd, n_even), dtype=np.complex128) if include_internal else None
    else:
        Q11_inv = _solve_right_inverse(Q11)
        G1 = P11 @ Q11_inv
        G3 = P21 @ Q11_inv
        G5 = Q21 @ Q11_inv
        F2 = _solve_right_inverse(Q22 - G5 @ Q12)
        G2 = P22 @ F2
        G4 = P12 @ F2
        G6 = Q12 @ F2

        T12 = G1 @ G6 - G4
        T22 = G3 @ G6 - G2
        T11 = -G1 - T12 @ G5
        T21 = -G3 - T22 @ G5

        if include_internal:
            R12 = -Q11_inv @ G6
            R22 = F2
            R11 = Q11_inv - R12 @ G5
            R21 = -R22 @ G5
        else:
            R11 = None
            R12 = None
            R21 = None
            R22 = None

    return AxisymmetricTRBlock(
        m=m,
        even_indices=np.asarray(parity_block.even_indices, dtype=np.int64),
        odd_indices=np.asarray(parity_block.odd_indices, dtype=np.int64),
        T11=T11,
        T12=T12,
        T21=T21,
        T22=T22,
        R11=R11,
        R12=R12,
        R21=R21,
        R22=R22,
    )


def _solve_internal_from_scattered_parity_block(
    parity_block: AxisymmetricParityBlock,
) -> AxisymmetricTRBlock:
    """Solve the parity-reduced scattered-to-internal map `c = C p`.

    The boundary-matching matrices satisfy `p = -P c` on each fixed-`m`,
    fixed-parity sector, so the internal regular coefficients follow from the
    dense local solve `C = -P^{-1}`.
    """

    m = int(parity_block.m)
    P11 = np.asarray(parity_block.P11, dtype=np.complex128)
    P12 = np.asarray(parity_block.P12, dtype=np.complex128)
    P21 = np.asarray(parity_block.P21, dtype=np.complex128)
    P22 = np.asarray(parity_block.P22, dtype=np.complex128)

    n_even = int(P11.shape[0])
    n_odd = int(P22.shape[0])
    zero_11 = np.zeros_like(P11)
    zero_12 = np.zeros_like(P12)
    zero_21 = np.zeros_like(P21)
    zero_22 = np.zeros_like(P22)

    if m == 0:
        C11 = -_solve_right_inverse(P11)
        C22 = -_solve_right_inverse(P22)
        C12 = np.zeros((n_even, n_odd), dtype=np.complex128)
        C21 = np.zeros((n_odd, n_even), dtype=np.complex128)
    else:
        P_full = np.block([[P11, P12], [P21, P22]])
        C_full = -_solve_right_inverse(P_full)
        C11 = C_full[:n_even, :n_even]
        C12 = C_full[:n_even, n_even:]
        C21 = C_full[n_even:, :n_even]
        C22 = C_full[n_even:, n_even:]

    return AxisymmetricTRBlock(
        m=m,
        even_indices=np.asarray(parity_block.even_indices, dtype=np.int64),
        odd_indices=np.asarray(parity_block.odd_indices, dtype=np.int64),
        T11=zero_11,
        T12=zero_12,
        T21=zero_21,
        T22=zero_22,
        R11=C11,
        R12=C12,
        R21=C21,
        R22=C22,
    )


def combine_axisymmetric_parity_blocks(
    even_odd: AxisymmetricTRBlock,
    odd_even: AxisymmetricTRBlock,
) -> AxisymmetricResolvedMBlock:
    """Recombine the complementary parity solves into one full fixed-`m` block.

    The parity-reduced solves split the axisymmetric algebra into two decoupled
    sectors. This helper stitches them back together in the natural degree
    ordering `n = max(m, 1)..lmax`, which is the form needed to populate the
    final spherical-basis particle `T` matrix.
    """

    if int(even_odd.m) != int(odd_even.m):
        raise ValueError("Parity blocks must belong to the same azimuthal order m.")

    m = int(even_odd.m)
    even_idx = np.asarray(even_odd.even_indices, dtype=np.int64)
    odd_idx = np.asarray(even_odd.odd_indices, dtype=np.int64)
    n_count = even_idx.size + odd_idx.size
    n_values = np.arange(max(m, 1), max(m, 1) + n_count, dtype=np.int64)

    def assemble_full(
        even_even: Array,
        even_odd_block: Array,
        odd_even_block: Array,
        odd_odd: Array,
    ) -> Array:
        full = np.zeros((n_count, n_count), dtype=np.complex128)
        full[np.ix_(even_idx, even_idx)] = np.asarray(even_even, dtype=np.complex128)
        full[np.ix_(even_idx, odd_idx)] = np.asarray(even_odd_block, dtype=np.complex128)
        full[np.ix_(odd_idx, even_idx)] = np.asarray(odd_even_block, dtype=np.complex128)
        full[np.ix_(odd_idx, odd_idx)] = np.asarray(odd_odd, dtype=np.complex128)
        return full

    T11 = assemble_full(even_odd.T11, even_odd.T12 * 0.0, even_odd.T21 * 0.0, odd_even.T11)
    T22 = assemble_full(odd_even.T22, odd_even.T21 * 0.0, odd_even.T12 * 0.0, even_odd.T22)
    T12 = assemble_full(
        np.zeros((even_idx.size, even_idx.size), dtype=np.complex128),
        even_odd.T12,
        odd_even.T12,
        np.zeros((odd_idx.size, odd_idx.size), dtype=np.complex128),
    )
    T21 = assemble_full(
        np.zeros((even_idx.size, even_idx.size), dtype=np.complex128),
        odd_even.T21,
        even_odd.T21,
        np.zeros((odd_idx.size, odd_idx.size), dtype=np.complex128),
    )

    if even_odd.R11 is None or odd_even.R11 is None:
        R11 = None
        R12 = None
        R21 = None
        R22 = None
    else:
        missing: list[str] = []
        if even_odd.R12 is None:
            missing.append("even_odd.R12")
        if even_odd.R21 is None:
            missing.append("even_odd.R21")
        if even_odd.R22 is None:
            missing.append("even_odd.R22")
        if odd_even.R12 is None:
            missing.append("odd_even.R12")
        if odd_even.R21 is None:
            missing.append("odd_even.R21")
        if odd_even.R22 is None:
            missing.append("odd_even.R22")
        if missing:
            raise RuntimeError(
                "Axisymmetric block parity merge received incomplete internal sub-blocks "
                f"for m={m}: missing {', '.join(missing)}."
            )
        r11_eo = cast(Array, even_odd.R11)
        r12_eo = cast(Array, even_odd.R12)
        r21_eo = cast(Array, even_odd.R21)
        r22_eo = cast(Array, even_odd.R22)
        r11_oe = cast(Array, odd_even.R11)
        r12_oe = cast(Array, odd_even.R12)
        r21_oe = cast(Array, odd_even.R21)
        r22_oe = cast(Array, odd_even.R22)
        R11 = assemble_full(r11_eo, r12_eo * 0.0, r21_eo * 0.0, r11_oe)
        R22 = assemble_full(
            r22_oe,
            r21_oe * 0.0,
            r12_oe * 0.0,
            r22_eo,
        )
        R12 = assemble_full(
            np.zeros((even_idx.size, even_idx.size), dtype=np.complex128),
            r12_eo,
            r12_oe,
            np.zeros((odd_idx.size, odd_idx.size), dtype=np.complex128),
        )
        R21 = assemble_full(
            np.zeros((even_idx.size, even_idx.size), dtype=np.complex128),
            r21_oe,
            r21_eo,
            np.zeros((odd_idx.size, odd_idx.size), dtype=np.complex128),
        )

    return AxisymmetricResolvedMBlock(
        m=m,
        n_values=n_values,
        T11=T11,
        T12=T12,
        T21=T21,
        T22=T22,
        R11=R11,
        R12=R12,
        R21=R21,
        R22=R22,
    )


def assemble_axisymmetric_tmatrix_block(
    lmax: int,
    resolved_blocks: list[AxisymmetricResolvedMBlock] | tuple[AxisymmetricResolvedMBlock, ...],
) -> Array:
    """Assemble a CELES-ordered dense particle `T` block from fixed-`m` sectors.

    The fixed-`m` axisymmetric solve preserves azimuthal order. This helper
    embeds those solved sectors into the full spherical-basis matrix used by
    the generic solver path. Positive and negative `m` share the same same-
    polarization couplings, while the cross-polarization couplings change sign
    with `m`.
    """

    Nm = n_modes(int(lmax))
    T = np.zeros((Nm, Nm), dtype=np.complex128)

    for block in resolved_blocks:
        m = int(block.m)
        n_values = np.asarray(block.n_values, dtype=np.int64)
        for row_idx, l1 in enumerate(n_values):
            for col_idx, l2 in enumerate(n_values):
                mm_values = (0,) if m == 0 else (m, -m)
                for mm in mm_values:
                    sign_m = 1 if mm >= 0 else -1
                    idx_m_1 = index_vswf(int(l1), int(mm), 1, int(lmax))
                    idx_m_2 = index_vswf(int(l1), int(mm), 2, int(lmax))
                    idx_n_1 = index_vswf(int(l2), int(mm), 1, int(lmax))
                    idx_n_2 = index_vswf(int(l2), int(mm), 2, int(lmax))

                    T[idx_m_1, idx_n_1] = block.T11[row_idx, col_idx]
                    T[idx_m_2, idx_n_2] = block.T22[row_idx, col_idx]
                    T[idx_m_1, idx_n_2] = sign_m * block.T12[row_idx, col_idx]
                    T[idx_m_2, idx_n_1] = sign_m * block.T21[row_idx, col_idx]

    return T


def assemble_axisymmetric_internal_block(
    lmax: int,
    resolved_blocks: list[AxisymmetricResolvedMBlock] | tuple[AxisymmetricResolvedMBlock, ...],
) -> Array:
    """Assemble a CELES-ordered dense internal-response block from fixed-`m` sectors.

    The returned matrix maps incident regular spherical-wave coefficients to
    internal regular spherical-wave coefficients in the same CELES ordering.
    """

    Ns = lmax * (lmax + 2)
    Nm = 2 * Ns
    R = np.zeros((Nm, Nm), dtype=np.complex128)

    for block in resolved_blocks:
        if block.R11 is None or block.R12 is None or block.R21 is None or block.R22 is None:
            raise ValueError("Resolved axisymmetric blocks do not contain internal-response data.")
        m = int(block.m)
        n_values = np.asarray(block.n_values, dtype=np.int64)
        for row_idx, l1 in enumerate(n_values):
            for col_idx, l2 in enumerate(n_values):
                mm_values = (0,) if m == 0 else (m, -m)
                for mm in mm_values:
                    sign_m = 1 if mm >= 0 else -1
                    row_tau1 = index_vswf(int(l1), int(mm), 1, int(lmax))
                    row_tau2 = index_vswf(int(l1), int(mm), 2, int(lmax))
                    col_tau1 = index_vswf(int(l2), int(mm), 1, int(lmax))
                    col_tau2 = index_vswf(int(l2), int(mm), 2, int(lmax))
                    R[row_tau1, col_tau1] = block.R11[row_idx, col_idx]
                    R[row_tau1, col_tau2] = sign_m * block.R12[row_idx, col_idx]
                    R[row_tau2, col_tau1] = sign_m * block.R21[row_idx, col_idx]
                    R[row_tau2, col_tau2] = block.R22[row_idx, col_idx]

    return R


def solve_axisymmetric_tmatrix_blocks(
    nmax: int,
    relative_refractive_index: complex,
    k_medium: complex,
    quadrature: AxisymmetricShapeQuadrature,
    radial: ModifiedBesselProducts,
    *,
    include_internal: bool = False,
) -> tuple[AxisymmetricResolvedMBlock, ...]:
    """Solve all fixed-`m` sectors needed for one axisymmetric particle block.

    This is the current reference orchestration layer for the spherical-basis
    spheroid backend:
    1. assemble raw `P/Q` blocks for each `m = 0..nmax`,
    2. split each block into the two reflection-parity sectors,
    3. solve the particle-local `TQ=-P` and optional `RQ=I` systems,
    4. recombine the parity sectors into one full fixed-`m` block.
    """

    out: list[AxisymmetricResolvedMBlock] = []
    for m in range(int(nmax) + 1):
        angular = axisymmetric_angular_functions(int(nmax), int(m), quadrature)
        pq = assemble_axisymmetric_pq_block(
            relative_refractive_index,
            quadrature,
            angular,
            radial,
            k_medium=k_medium,
        )
        even_odd_pq, odd_even_pq = split_axisymmetric_pq_block_by_parity(pq)
        even_odd = solve_axisymmetric_tr_block(even_odd_pq, include_internal=include_internal)
        odd_even = solve_axisymmetric_tr_block(odd_even_pq, include_internal=include_internal)
        out.append(combine_axisymmetric_parity_blocks(even_odd, odd_even))
    return tuple(out)


def recommend_spheroid_n_theta(
    lmax: int,
    equatorial_radius: float,
    polar_radius: float,
) -> int:
    """Return a conservative internal meridian quadrature order for spheroids.

    The user-facing spheroid API should stay centered on `lmax`. This helper
    therefore chooses a particle-local quadrature size internally from the
    truncation order and aspect ratio, rather than forcing an additional knob
    into the default solver workflow.
    """

    lmax = int(lmax)
    if lmax < 1:
        raise ValueError("lmax must be >= 1.")
    profile = SpheroidShapeProfile(
        equatorial_radius=float(equatorial_radius),
        polar_radius=float(polar_radius),
    )

    aspect_ratio = profile.aspect_ratio
    aspect_penalty = int(np.ceil(18.0 * np.log1p(aspect_ratio - 1.0)))
    base = max(64, 16 * lmax)
    return int(base + aspect_penalty)


def spheroid_tmatrix_block(
    lmax: int,
    k_medium: complex,
    equatorial_radius: float,
    polar_radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
    *,
    n_theta: int | None = None,
) -> Array:
    """Return the aligned-spheroid spherical-basis `T` block in CELES ordering.

    This is the current reference spheroid backend for the generic solver path.
    It keeps the cluster solver in spherical waves and prepares one dense
    particle-local `T` block by:
    1. assembling axisymmetric EBCM `P/Q` blocks,
    2. solving the fixed-`m` particle-local systems,
    3. embedding the result into CELES mode ordering.

    `n_theta` is kept as an internal expert override. When omitted, a
    conservative heuristic based on `lmax` and aspect ratio is used.
    """

    if n_theta is None:
        n_theta = recommend_spheroid_n_theta(
            lmax=int(lmax),
            equatorial_radius=float(equatorial_radius),
            polar_radius=float(polar_radius),
        )
    T, _ = spheroid_tmatrix_and_internal_block(
        lmax=int(lmax),
        k_medium=k_medium,
        equatorial_radius=equatorial_radius,
        polar_radius=polar_radius,
        n_particle=n_particle,
        n_medium=n_medium,
        n_theta=int(n_theta),
    )
    return T


def spheroid_tmatrix_and_internal_block(
    lmax: int,
    k_medium: complex,
    equatorial_radius: float,
    polar_radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
    *,
    n_theta: int | None = None,
) -> tuple[Array, Array]:
    """Return aligned spheroid `T` and scattered-to-internal dense blocks.

    The second returned matrix maps solved scattered/outgoing spherical-wave
    coefficients to internal regular spherical-wave coefficients. This mirrors
    the role played by `sphere_internal_ratios(...)` for diagonal particles.
    """

    if n_theta is None:
        n_theta = recommend_spheroid_n_theta(
            lmax=int(lmax),
            equatorial_radius=float(equatorial_radius),
            polar_radius=float(polar_radius),
        )

    geom = spheroid_geometry_quadrature(
        n_theta=int(n_theta),
        equatorial_radius=float(equatorial_radius),
        polar_radius=float(polar_radius),
    )
    rel_index = complex(n_particle) / complex(n_medium)
    radial = modified_bessel_products(
        nmax=int(lmax),
        relative_refractive_index=rel_index,
        x=np.asarray(complex(k_medium) * geom.radius, dtype=np.complex128),
    )
    resolved_t: list[AxisymmetricResolvedMBlock] = []
    resolved_internal: list[AxisymmetricResolvedMBlock] = []
    for m in range(0, int(lmax) + 1):
        angular = axisymmetric_angular_functions(int(lmax), int(m), geom)
        pq = assemble_axisymmetric_pq_block(
            rel_index,
            geom,
            angular,
            radial,
            k_medium=k_medium,
        )
        even_odd_pq, odd_even_pq = split_axisymmetric_pq_block_by_parity(pq)
        even_odd_t = solve_axisymmetric_tr_block(even_odd_pq, include_internal=False)
        odd_even_t = solve_axisymmetric_tr_block(odd_even_pq, include_internal=False)
        resolved_t.append(combine_axisymmetric_parity_blocks(even_odd_t, odd_even_t))

        even_odd_internal = _solve_internal_from_scattered_parity_block(even_odd_pq)
        odd_even_internal = _solve_internal_from_scattered_parity_block(odd_even_pq)
        resolved_internal.append(
            combine_axisymmetric_parity_blocks(even_odd_internal, odd_even_internal)
        )

    T = assemble_axisymmetric_tmatrix_block(int(lmax), tuple(resolved_t))
    internal_from_scattered = assemble_axisymmetric_internal_block(
        int(lmax), tuple(resolved_internal)
    )
    return T, internal_from_scattered
