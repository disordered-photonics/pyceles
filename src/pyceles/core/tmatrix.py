"""Spherical response kernels and CELES-style T-entry dispatch.

CPU-first implementation using NumPy + SciPy.

Conventions
-----------
- Input `k_medium` is the **wavenumber in the embedding medium**:

    k_medium = 2*pi/lambda * n_medium

  (CELES uses this convention throughout for translations and field evaluation.)

- `mie_ab` returns the standard Mie scattering coefficients (a_l, b_l)
  for l=1..lmax.

- `sphere_T_diagonal` and `pec_sphere_T_diagonal` map homogeneous and
  perfect-conductor sphere Mie coefficients into CELES' diagonal T entries.

- `layered_sphere_T_diagonal` does the same for concentric multilayer spheres
  through interface transfer matrices.

- `layered_internal_ab_ratios` provides per-layer radial coefficients for
  piecewise internal near-field evaluation in layered spheres.

- CELES' diagonal scattering entries follow `T_entry.m` in the MATLAB code:

    tau=1 (M/TE-like) uses -b_l
    tau=2 (N/TM-like) uses -a_l

- `sphere_internal_ratios` provides the per-l conversion factors used by CELES
  for homogeneous spheres. PEC spheres have zero physical interior field and
  therefore use zero internal/scattered conversion ratios.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .indexing import n_modes
from .particles import (
    LayeredSphere,
    Particle,
    PECSphere,
    Sphere,
    Spheroid,
    particle_intrinsic_t_signature,
    particle_t_signature,
)
from .spheroid_ebcm import spheroid_tmatrix_and_internal_block
from .svwf_rotation import rotate_svwf_tmatrix_block


def _riccati_jh(
    l: int,
    x: complex,
) -> tuple[complex, complex, complex, complex]:
    """Return (psi, dpsi, xi, dxi) for one order/argument.

    Here `psi=x*j_l(x)` and `xi=x*h_l^(1)(x)`, with derivatives w.r.t. `x`.
    """
    jl = spherical_jn(l, x)
    yl = spherical_yn(l, x)
    hl = jl + 1j * yl

    djl = spherical_jn(l, x, derivative=True)
    dyl = spherical_yn(l, x, derivative=True)
    dhl = djl + 1j * dyl

    psi = x * jl
    xi = x * hl
    dpsi = jl + x * djl
    dxi = hl + x * dhl
    return psi, dpsi, xi, dxi


def _layered_transfer_matrices(
    lmax: int,
    k_medium: complex,
    layer_radii: tuple[float, ...],
    layer_refractive_indices: tuple[complex, ...],
    n_medium: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build forward/backward transfer-matrix products for layered spheres.

    Returns
    -------
    T_tm, T_te, T_mm, T_me
        Arrays of shape `(lmax+1, n_layers, 2, 2)`.
        - `T_tm`, `T_te`: forward products (inner -> outer interfaces).
        - `T_mm`, `T_me`: backward products (each layer -> host side).
    """
    lmax = int(lmax)
    radii = np.asarray(layer_radii, dtype=float).reshape(-1)
    n_layers = np.asarray(layer_refractive_indices, dtype=np.complex128).reshape(-1)
    if radii.size == 0:
        raise ValueError("layer_radii must be non-empty.")
    if n_layers.size != radii.size:
        raise ValueError("layer_radii and layer_refractive_indices must have the same length.")
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("layer_radii must be finite and strictly positive.")
    if np.any(np.diff(radii) <= 0.0):
        raise ValueError("layer_radii must be strictly increasing.")

    n_medium_c = complex(n_medium)
    if n_medium_c == 0:
        raise ValueError("n_medium must be non-zero.")

    L = int(radii.size)
    n_full = np.empty(L + 1, dtype=np.complex128)
    n_full[:-1] = n_layers
    n_full[-1] = n_medium_c

    kM = complex(k_medium)

    T_tm = np.zeros((lmax + 1, L, 2, 2), dtype=np.complex128)
    T_te = np.zeros((lmax + 1, L, 2, 2), dtype=np.complex128)
    T_mm = np.zeros((lmax + 1, L, 2, 2), dtype=np.complex128)
    T_me = np.zeros((lmax + 1, L, 2, 2), dtype=np.complex128)

    for l in range(1, lmax + 1):
        tmf = np.zeros((L, 2, 2), dtype=np.complex128)
        tef = np.zeros((L, 2, 2), dtype=np.complex128)
        tmb = np.zeros((L, 2, 2), dtype=np.complex128)
        teb = np.zeros((L, 2, 2), dtype=np.complex128)

        for j in range(L):
            rj = float(radii[j])
            ns = complex(n_full[j])
            np1 = complex(n_full[j + 1])
            eta = ns / np1

            xs = kM * (ns / n_medium_c) * rj
            xp = kM * (np1 / n_medium_c) * rj

            psi_s, dpsi_s, xi_s, dxi_s = _riccati_jh(l, xs)
            psi_p, dpsi_p, xi_p, dxi_p = _riccati_jh(l, xp)

            # Magnetic (M / tau=1): mu-ratio assumed 1.
            tmb[j, 0, 0] = -1j * (dxi_s * psi_p * eta - xi_s * dpsi_p)
            tmb[j, 0, 1] = -1j * (dxi_s * xi_p * eta - xi_s * dxi_p)
            tmb[j, 1, 0] = -1j * (-dpsi_s * psi_p * eta + psi_s * dpsi_p)
            tmb[j, 1, 1] = -1j * (-dpsi_s * xi_p * eta + psi_s * dxi_p)

            tmf[j, 0, 0] = -1j * (dxi_p * psi_s / eta - xi_p * dpsi_s)
            tmf[j, 0, 1] = -1j * (dxi_p * xi_s / eta - xi_p * dxi_s)
            tmf[j, 1, 0] = -1j * (-dpsi_p * psi_s / eta + psi_p * dpsi_s)
            tmf[j, 1, 1] = -1j * (-dpsi_p * xi_s / eta + psi_p * dxi_s)

            # Electric (N / tau=2): eta/mu terms swap, mu-ratio still 1.
            teb[j, 0, 0] = -1j * (dxi_s * psi_p - xi_s * dpsi_p * eta)
            teb[j, 0, 1] = -1j * (dxi_s * xi_p - xi_s * dxi_p * eta)
            teb[j, 1, 0] = -1j * (-dpsi_s * psi_p + psi_s * dpsi_p * eta)
            teb[j, 1, 1] = -1j * (-dpsi_s * xi_p + psi_s * dxi_p * eta)

            tef[j, 0, 0] = -1j * (dxi_p * psi_s - xi_p * dpsi_s / eta)
            tef[j, 0, 1] = -1j * (dxi_p * xi_s - xi_p * dxi_s / eta)
            tef[j, 1, 0] = -1j * (-dpsi_p * psi_s + psi_p * dpsi_s / eta)
            tef[j, 1, 1] = -1j * (-dpsi_p * xi_s + psi_p * dxi_s / eta)

        for j in range(L):
            if j == 0:
                T_tm[l, j] = tmf[j]
                T_te[l, j] = tef[j]
            else:
                T_tm[l, j] = tmf[j] @ T_tm[l, j - 1]
                T_te[l, j] = tef[j] @ T_te[l, j - 1]

        for j in range(L - 1, -1, -1):
            if j == L - 1:
                T_mm[l, j] = tmb[j]
                T_me[l, j] = teb[j]
            else:
                T_mm[l, j] = tmb[j] @ T_mm[l, j + 1]
                T_me[l, j] = teb[j] @ T_me[l, j + 1]

    return T_tm, T_te, T_mm, T_me


def mie_ab(
    lmax: int,
    k_medium: complex,
    radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mie coefficients a_l, b_l for a sphere.

    Parameters
    ----------
    lmax:
        Maximum degree (returns arrays of length lmax+1; index 0 is unused).
    k_medium:
        Wavenumber in embedding medium.
    radius:
        Sphere radius.
    n_particle, n_medium:
        Complex refractive indices.

    Returns
    -------
    a, b:
        Complex arrays of shape (lmax+1,) with a[0]=b[0]=0.
    """

    lmax = int(lmax)
    radius = float(radius)
    kM = complex(k_medium)
    nM = complex(n_medium)
    nS = complex(n_particle)

    # relative index (mu=1 assumed)
    m = nS / nM

    x = kM * radius
    mx = m * x

    a = np.zeros(lmax + 1, dtype=np.complex128)
    b = np.zeros(lmax + 1, dtype=np.complex128)

    for l in range(1, lmax + 1):
        jl_x = spherical_jn(l, x)
        yl_x = spherical_yn(l, x)
        hl_x = jl_x + 1j * yl_x

        jl_mx = spherical_jn(l, mx)

        # derivatives d/dx
        djl_x = spherical_jn(l, x, derivative=True)
        dyl_x = spherical_yn(l, x, derivative=True)
        dhl_x = djl_x + 1j * dyl_x

        djl_mx = spherical_jn(l, mx, derivative=True)

        # Riccati-Bessel functions psi = x j_l(x), xi = x h_l(x)
        psi_x = x * jl_x
        psi_mx = mx * jl_mx
        xi_x = x * hl_x

        # derivatives psi' = j + x j', xi' = h + x h'
        dpsi_x = jl_x + x * djl_x
        dpsi_mx = jl_mx + mx * djl_mx
        dxi_x = hl_x + x * dhl_x

        # Standard Mie formulas
        num_a = m * psi_mx * dpsi_x - psi_x * dpsi_mx
        den_a = m * psi_mx * dxi_x - xi_x * dpsi_mx

        num_b = psi_mx * dpsi_x - m * psi_x * dpsi_mx
        den_b = psi_mx * dxi_x - m * xi_x * dpsi_mx

        a[l] = num_a / den_a
        b[l] = num_b / den_b

    return a, b


def pec_mie_ab(
    lmax: int,
    k_medium: complex,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mie coefficients for a perfect-electric-conductor sphere.

    For ``x = k_medium * radius`` and Riccati-Bessel functions
    ``psi_l(x) = x j_l(x)`` and ``xi_l(x) = x h_l^(1)(x)``, the conducting
    boundary condition gives

    ``a_l = psi_l'(x) / xi_l'(x)`` and ``b_l = psi_l(x) / xi_l(x)``.

    This is the stable large-relative-index limit of the non-magnetic Mie
    coefficients; it avoids representing a perfect conductor with an artificial
    extreme complex refractive index.
    """
    lmax = int(lmax)
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    x = complex(k_medium) * radius
    if x == 0:
        raise ValueError("k_medium * radius must be non-zero for PEC Mie coefficients")

    a = np.zeros(lmax + 1, dtype=np.complex128)
    b = np.zeros(lmax + 1, dtype=np.complex128)
    for l in range(1, lmax + 1):
        psi, dpsi, xi, dxi = _riccati_jh(l, x)
        a[l] = dpsi / dxi
        b[l] = psi / xi
    return a, b


def layered_mie_ab(
    lmax: int,
    k_medium: complex,
    layer_radii: tuple[float, ...],
    layer_refractive_indices: tuple[complex, ...],
    n_medium: complex = 1.0 + 0j,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multilayer-sphere Mie coefficients `a_l`, `b_l`.

    Parameters
    ----------
    lmax:
        Maximum degree (index 0 is unused).
    k_medium:
        Embedding-medium wavenumber.
    layer_radii:
        Strictly increasing outer radii of concentric layers (inner -> outer).
    layer_refractive_indices:
        Layer refractive indices, same ordering/length as `layer_radii`.
    n_medium:
        Embedding-medium refractive index.
    """
    lmax = int(lmax)
    T_tm, T_te, _, _ = _layered_transfer_matrices(
        lmax=lmax,
        k_medium=complex(k_medium),
        layer_radii=layer_radii,
        layer_refractive_indices=layer_refractive_indices,
        n_medium=complex(n_medium),
    )
    a = np.zeros(lmax + 1, dtype=np.complex128)
    b = np.zeros(lmax + 1, dtype=np.complex128)
    for l in range(1, lmax + 1):
        # Stratify/scattnlay convention: a from electric branch, b from magnetic branch.
        a[l] = -(T_te[l, -1, 1, 0] / T_te[l, -1, 0, 0])
        b[l] = -(T_tm[l, -1, 1, 0] / T_tm[l, -1, 0, 0])
    return a, b


def layered_internal_ab_ratios(
    lmax: int,
    k_medium: complex,
    layer_radii: tuple[float, ...],
    layer_refractive_indices: tuple[complex, ...],
    n_medium: complex = 1.0 + 0j,
) -> dict[int, dict[str, np.ndarray]]:
    """Return per-layer radial `(A,B)` multipliers relative to scattered modes.

    For each polarization branch (`tau=1` magnetic/M, `tau=2` electric/N),
    each layer `j`, and each degree `l`, this returns

    - `A[j,l]`: multiplier for regular radial basis `j_l(k_j r)`
    - `B[j,l]`: multiplier for outgoing radial basis `h_l^(1)(k_j r)`

    such that layer fields can be evaluated from solved outgoing coefficients
    (`Q_scat` basis used in CELES). The innermost layer has `B=0` by regularity
    at `r=0`.
    """
    lmax = int(lmax)
    L = len(layer_radii)
    T_tm, T_te, T_mm, T_me = _layered_transfer_matrices(
        lmax=lmax,
        k_medium=complex(k_medium),
        layer_radii=layer_radii,
        layer_refractive_indices=layer_refractive_indices,
        n_medium=complex(n_medium),
    )

    A_m = np.zeros((L, lmax + 1), dtype=np.complex128)
    B_m = np.zeros((L, lmax + 1), dtype=np.complex128)
    A_n = np.zeros((L, lmax + 1), dtype=np.complex128)
    B_n = np.zeros((L, lmax + 1), dtype=np.complex128)

    for l in range(1, lmax + 1):
        q_scat_m = T_tm[l, -1, 1, 0] / T_tm[l, -1, 0, 0]  # = -b_l
        q_scat_n = T_te[l, -1, 1, 0] / T_te[l, -1, 0, 0]  # = -a_l
        for j in range(L):
            tmm = T_mm[l, j]
            tme = T_me[l, j]
            A_m[j, l] = (tmm[0, 0] + tmm[0, 1] * q_scat_m) / q_scat_m
            A_n[j, l] = (tme[0, 0] + tme[0, 1] * q_scat_n) / q_scat_n
            if j > 0:
                B_m[j, l] = (tmm[1, 0] + tmm[1, 1] * q_scat_m) / q_scat_m
                B_n[j, l] = (tme[1, 0] + tme[1, 1] * q_scat_n) / q_scat_n

    return {1: {"A": A_m, "B": B_m}, 2: {"A": A_n, "B": B_n}}


def sphere_T_diagonal(
    lmax: int,
    k_medium: complex,
    radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
    *,
    sign: int = -1,
) -> dict[int, np.ndarray]:
    """Sphere diagonal T entries in CELES' (M,N) ordering.

    Returns a dict mapping:
      tau=1 -> T_M[l] for l=0..lmax
      tau=2 -> T_N[l] for l=0..lmax

    In CELES (see `T_entry.m`), the scattering diagonal entries are:
      tau=1: Q = -b_l
      tau=2: Q = -a_l

    `sign=-1` reproduces that default.
    """

    a, b = mie_ab(lmax, k_medium, radius, n_particle, n_medium)
    return {1: sign * b, 2: sign * a}


def pec_sphere_T_diagonal(
    lmax: int,
    k_medium: complex,
    radius: float,
    *,
    sign: int = -1,
) -> dict[int, np.ndarray]:
    """PEC-sphere diagonal T entries in CELES' (M,N) ordering.

    Uses the same CELES mapping as homogeneous spheres:
      tau=1: ``Q = -b_l``
      tau=2: ``Q = -a_l``
    """
    a, b = pec_mie_ab(lmax=lmax, k_medium=k_medium, radius=radius)
    return {1: sign * b, 2: sign * a}


def layered_sphere_T_diagonal(
    lmax: int,
    k_medium: complex,
    layer_radii: tuple[float, ...],
    layer_refractive_indices: tuple[complex, ...],
    n_medium: complex = 1.0 + 0j,
    *,
    sign: int = -1,
) -> dict[int, np.ndarray]:
    """Layered-sphere diagonal T entries in CELES' (M,N) ordering.

    Uses the same CELES mapping as homogeneous spheres:
      tau=1: `Q = -b_l`
      tau=2: `Q = -a_l`
    """
    a, b = layered_mie_ab(
        lmax=lmax,
        k_medium=k_medium,
        layer_radii=layer_radii,
        layer_refractive_indices=layer_refractive_indices,
        n_medium=n_medium,
    )
    return {1: sign * b, 2: sign * a}


def sphere_internal_ratios(
    lmax: int,
    k_medium: complex,
    radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
) -> dict[int, np.ndarray]:
    """Ratios to convert scattered -> internal coefficients for a sphere.

    CELES computes internal field coefficients from the scattered coefficients via

        ratio_M(l) = Q_int_M(l) / Q_scat_M(l)
        ratio_N(l) = Q_int_N(l) / Q_scat_N(l)

    where Q_scat_* are the CELES scattering diagonal entries (=-b_l,=-a_l) and
    Q_int_* are the corresponding internal entries.

    This function follows the MATLAB implementation in `src/scattering/T_entry.m`.

    Returns
    -------
    {1: ratio_M, 2: ratio_N}
        Arrays indexed by l (0..lmax). Entry 0 is unused.
    """

    lmax = int(lmax)

    kM = complex(k_medium)
    radius = float(radius)
    nM = complex(n_medium)
    nS = complex(n_particle)

    x = kM * radius
    m = nS / nM
    mx = m * x

    ratio_M = np.zeros(lmax + 1, dtype=np.complex128)
    ratio_N = np.zeros(lmax + 1, dtype=np.complex128)

    for l in range(1, lmax + 1):
        # outside: j_l(x), h_l(x)
        jl_x = spherical_jn(l, x)
        yl_x = spherical_yn(l, x)
        hl_x = jl_x + 1j * yl_x

        djl_x = spherical_jn(l, x, derivative=True)
        dyl_x = spherical_yn(l, x, derivative=True)
        dhl_x = djl_x + 1j * dyl_x

        # d/dx (x*z_l(x))
        djx_x = jl_x + x * djl_x
        dhx_x = hl_x + x * dhl_x

        # inside: only j_l(mx)
        jl_mx = spherical_jn(l, mx)
        djl_mx = spherical_jn(l, mx, derivative=True)
        djmx_x = jl_mx + mx * djl_mx

        # tau=1 (M)
        den_M = jl_mx * dhx_x - hl_x * djmx_x
        Q_scat_M = -(jl_mx * djx_x - jl_x * djmx_x) / den_M
        Q_int_M = (jl_x * dhx_x - hl_x * djx_x) / den_M
        ratio_M[l] = Q_int_M / Q_scat_M

        # tau=2 (N)
        den_N = (m**2) * jl_mx * dhx_x - hl_x * djmx_x
        Q_scat_N = -(((m**2) * jl_mx * djx_x - jl_x * djmx_x) / den_N)
        Q_int_N = (m * jl_x * dhx_x - m * hl_x * djx_x) / den_N
        ratio_N[l] = Q_int_N / Q_scat_N

    return {1: ratio_M, 2: ratio_N}


def _unsupported_particle_message(particle: Particle) -> str:
    """Human-readable dispatch error for not-yet-supported particle types."""
    return (
        f"T-matrix for particle type '{type(particle).__name__}' is not implemented yet. "
        "Supported particle backends are Sphere, PECSphere, LayeredSphere, and Spheroid "
        "(via spherical-basis T blocks). "
        "Non-spherical solvers are planned separately."
    )


def particle_T_diagonal(
    lmax: int,
    k_medium: complex,
    particle: Particle,
    n_medium: complex = 1.0 + 0j,
    *,
    sign: int = -1,
) -> dict[int, np.ndarray]:
    """Dispatch diagonal T entries by particle type."""
    if isinstance(particle, PECSphere):
        return pec_sphere_T_diagonal(
            lmax=lmax,
            k_medium=k_medium,
            radius=particle.radius,
            sign=sign,
        )
    if isinstance(particle, Sphere):
        return sphere_T_diagonal(
            lmax=lmax,
            k_medium=k_medium,
            radius=particle.radius,
            n_particle=particle.refractive_index,
            n_medium=n_medium,
            sign=sign,
        )
    if isinstance(particle, LayeredSphere):
        return layered_sphere_T_diagonal(
            lmax=lmax,
            k_medium=k_medium,
            layer_radii=particle.layer_radii,
            layer_refractive_indices=particle.layer_refractive_indices,
            n_medium=n_medium,
            sign=sign,
        )
    if isinstance(particle, Spheroid):
        raise NotImplementedError(
            "Spheroid diagonal T entries are not available. Use the spherical-basis "
            "particle T-block path instead."
        )
    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


def _mode_diagonal_from_degree_diagonal(
    lmax: int,
    T_M: np.ndarray,
    T_N: np.ndarray,
) -> np.ndarray:
    """Expand per-degree TE/TM diagonals into CELES-order mode diagonals.

    This is the canonical spherical-basis bridge from compact diagonal particle
    kernels to a full `(Nm,)` operator row. It is used when diagonal particles
    are forced through block-based solver paths, and will also serve as the
    reference representation for future non-diagonal particles implemented
    directly in the spherical basis.
    """

    lmax = int(lmax)
    T_M = np.asarray(T_M)
    T_N = np.asarray(T_N)
    if T_M.shape != T_N.shape:
        raise ValueError(f"T_M and T_N must have the same shape, got {T_M.shape} and {T_N.shape}.")

    Nm = n_modes(lmax)
    Nscl = lmax * (lmax + 2)
    out = np.zeros((Nm,), dtype=np.result_type(T_M.dtype, T_N.dtype, np.complex64))
    for l in range(1, lmax + 1):
        start = (l - 1) * (l + 1)
        end = start + (2 * l + 1)
        out[start:end] = T_M[l]
        out[Nscl + start : Nscl + end] = T_N[l]
    return out


def particle_T_matrix_block(
    lmax: int,
    k_medium: complex,
    particle: Particle,
    n_medium: complex = 1.0 + 0j,
    *,
    sign: int = -1,
) -> np.ndarray:
    """Return one particle-local spherical-basis T block in CELES mode ordering.

    Notes
    -----
    This is the canonical full-block representation consumed by dense and
    baseline axisymmetric solver paths. Diagonal particles are expanded exactly
    into a dense diagonal matrix, so non-diagonal backends can be regression
    tested against the fast path without re-implementing particle physics.
    """

    if isinstance(particle, Spheroid):
        block = _aligned_spheroid_tmatrix_block(
            lmax=lmax,
            k_medium=k_medium,
            particle=particle,
            n_medium=n_medium,
        )
        angles: tuple[float, float, float] = (
            float(particle.euler_angles[0]),
            float(particle.euler_angles[1]),
            float(particle.euler_angles[2]),
        )
        return rotate_svwf_tmatrix_block(
            block,
            int(lmax),
            angles,
        )

    Td = particle_T_diagonal(
        lmax=lmax,
        k_medium=k_medium,
        particle=particle,
        n_medium=n_medium,
        sign=sign,
    )
    diag = _mode_diagonal_from_degree_diagonal(
        int(lmax),
        np.asarray(Td[1], dtype=np.complex128),
        np.asarray(Td[2], dtype=np.complex128),
    )
    return np.diag(diag)


def _aligned_spheroid_tmatrix_block(
    lmax: int,
    k_medium: complex,
    particle: Spheroid,
    n_medium: complex,
) -> np.ndarray:
    """Return the aligned/body-frame spherical-basis T block for one spheroid."""

    return spheroid_tmatrix_and_internal_block(
        lmax=lmax,
        k_medium=k_medium,
        equatorial_radius=particle.equatorial_radius,
        polar_radius=particle.polar_radius,
        n_particle=particle.refractive_index,
        n_medium=n_medium,
    )[0]


def _aligned_spheroid_internal_block(
    lmax: int,
    k_medium: complex,
    particle: Spheroid,
    n_medium: complex,
) -> np.ndarray:
    """Return the aligned/body-frame scattered-to-internal block for one spheroid."""

    return spheroid_tmatrix_and_internal_block(
        lmax=lmax,
        k_medium=k_medium,
        equatorial_radius=particle.equatorial_radius,
        polar_radius=particle.polar_radius,
        n_particle=particle.refractive_index,
        n_medium=n_medium,
    )[1]


def _spheroid_internal_block(
    lmax: int,
    k_medium: complex,
    particle: Spheroid,
    n_medium: complex,
) -> np.ndarray:
    """Return the lab-frame scattered-to-internal block for one spheroid."""

    aligned = _aligned_spheroid_internal_block(
        lmax=lmax,
        k_medium=k_medium,
        particle=particle,
        n_medium=n_medium,
    )
    angles: tuple[float, float, float] = (
        float(particle.euler_angles[0]),
        float(particle.euler_angles[1]),
        float(particle.euler_angles[2]),
    )
    return rotate_svwf_tmatrix_block(aligned, int(lmax), angles)


def particle_T_matrix_blocks(
    lmax: int,
    k_medium: complex,
    particles: list[Particle] | tuple[Particle, ...],
    n_medium: complex = 1.0 + 0j,
    *,
    sign: int = -1,
) -> np.ndarray:
    """Return stacked spherical-basis T blocks for a particle subset.

    This helper intentionally mirrors `particle_T_diagonal(...)` but returns the
    denser `(Ng, Nm, Nm)` form expected by general prepared-operator paths.
    Homogeneous spheroids use their axisymmetric EBCM block and are rotated into
    the lab-frame spherical basis here; diagonal particle families are expanded
    exactly for mixed-family and dense-operator paths.

    Identical particles are prepared once per `(particle signature, medium,
    truncation)` tuple and then reused across the subset. That matters for the
    dense and axisymmetric fallback paths, where rebuilding a full block for
    every repeated particle would be avoidable overhead.
    """

    memo: dict[tuple[object, ...], np.ndarray] = {}
    intrinsic_memo: dict[tuple[object, ...], np.ndarray] = {}
    blocks: list[np.ndarray] = []
    for particle in particles:
        key = (
            particle_t_signature(particle),
            int(lmax),
            complex(k_medium),
            complex(n_medium),
            int(sign),
        )
        block = memo.get(key)
        if block is None:
            if isinstance(particle, Spheroid):
                intrinsic_key = (
                    particle_intrinsic_t_signature(particle),
                    int(lmax),
                    complex(k_medium),
                    complex(n_medium),
                    int(sign),
                )
                aligned = intrinsic_memo.get(intrinsic_key)
                if aligned is None:
                    aligned = _aligned_spheroid_tmatrix_block(
                        lmax=lmax,
                        k_medium=k_medium,
                        particle=particle,
                        n_medium=n_medium,
                    )
                    intrinsic_memo[intrinsic_key] = aligned
                block = rotate_svwf_tmatrix_block(
                    aligned,
                    int(lmax),
                    (
                        float(particle.euler_angles[0]),
                        float(particle.euler_angles[1]),
                        float(particle.euler_angles[2]),
                    ),
                )
            else:
                block = particle_T_matrix_block(
                    lmax=lmax,
                    k_medium=k_medium,
                    particle=particle,
                    n_medium=n_medium,
                    sign=sign,
                )
            memo[key] = block
        blocks.append(block)

    return np.stack(blocks, axis=0)


def particle_internal_ratios(
    lmax: int,
    k_medium: complex,
    particle: Particle,
    n_medium: complex = 1.0 + 0j,
) -> dict[int, np.ndarray]:
    """Dispatch internal/scattered conversion ratios by particle type.

    Notes
    -----
    For `LayeredSphere`, this returns the innermost-core regular-field ratios
    (`A` for layer index 0, where `B=0` by regularity). Full layer-wise `(A,B)`
    data are available via :func:`layered_internal_ab_ratios`.
    """
    if isinstance(particle, PECSphere):
        zeros = np.zeros(int(lmax) + 1, dtype=np.complex128)
        return {1: zeros.copy(), 2: zeros.copy()}
    if isinstance(particle, Sphere):
        return sphere_internal_ratios(
            lmax=lmax,
            k_medium=k_medium,
            radius=particle.radius,
            n_particle=particle.refractive_index,
            n_medium=n_medium,
        )
    if isinstance(particle, LayeredSphere):
        ratios = layered_internal_ab_ratios(
            lmax=lmax,
            k_medium=k_medium,
            layer_radii=particle.layer_radii,
            layer_refractive_indices=particle.layer_refractive_indices,
            n_medium=n_medium,
        )
        return {
            1: np.asarray(ratios[1]["A"][0, :], dtype=np.complex128),
            2: np.asarray(ratios[2]["A"][0, :], dtype=np.complex128),
        }
    if isinstance(particle, Spheroid):
        raise NotImplementedError(_unsupported_particle_message(particle))
    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


def mie_cross_sections(
    lmax: int,
    k_medium: complex,
    radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
) -> dict[str, float]:
    """Single-sphere Mie cross sections.

    Returns
    -------
    dict with keys:
      - ``C_ext``: extinction cross section
      - ``C_sca``: scattering cross section
      - ``C_abs``: absorption cross section (= C_ext - C_sca)

    Notes
    -----
    These expressions match the standard Mie series for non-magnetic spheres.
    """
    lmax = int(lmax)
    k = complex(k_medium)
    if k == 0:
        raise ValueError("k_medium must be non-zero")

    a, b = mie_ab(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    return _mie_cross_sections_from_coefficients(k, a, b)


def pec_mie_cross_sections(
    lmax: int,
    k_medium: complex,
    radius: float,
) -> dict[str, float]:
    """Single PEC-sphere cross sections from the perfect-conductor Mie series."""
    k = complex(k_medium)
    if k == 0:
        raise ValueError("k_medium must be non-zero")
    a, b = pec_mie_ab(lmax=lmax, k_medium=k_medium, radius=radius)
    return _mie_cross_sections_from_coefficients(k, a, b)


def _mie_cross_sections_from_coefficients(
    k: complex,
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, float]:
    """Evaluate standard single-sphere cross sections from Mie coefficients."""
    if a.shape != b.shape:
        raise ValueError(f"a and b must have matching shapes. Got {a.shape} and {b.shape}.")
    lmax = int(a.shape[0] - 1)
    l = np.arange(1, lmax + 1, dtype=np.float64)
    w = 2.0 * l + 1.0

    pref = 2.0 * np.pi / (abs(k) ** 2)
    C_ext = float(np.real(pref * np.sum(w * np.real(a[1:] + b[1:]))))
    C_sca = float(np.real(pref * np.sum(w * (np.abs(a[1:]) ** 2 + np.abs(b[1:]) ** 2))))
    C_abs = float(C_ext - C_sca)
    return {"C_ext": C_ext, "C_sca": C_sca, "C_abs": C_abs}


def mie_efficiencies(
    lmax: int,
    k_medium: complex,
    radius: float,
    n_particle: complex,
    n_medium: complex = 1.0 + 0j,
) -> dict[str, float]:
    """Single-sphere Mie efficiencies normalized by geometric area pi*radius^2."""
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")

    cs = mie_cross_sections(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
        n_particle=n_particle,
        n_medium=n_medium,
    )
    geom = np.pi * radius * radius
    return {
        "Q_ext": float(cs["C_ext"] / geom),
        "Q_sca": float(cs["C_sca"] / geom),
        "Q_abs": float(cs["C_abs"] / geom),
    }


def pec_mie_efficiencies(
    lmax: int,
    k_medium: complex,
    radius: float,
) -> dict[str, float]:
    """PEC-sphere Mie efficiencies normalized by geometric area pi*radius^2."""
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")

    cs = pec_mie_cross_sections(
        lmax=lmax,
        k_medium=k_medium,
        radius=radius,
    )
    geom = np.pi * radius * radius
    return {
        "Q_ext": float(cs["C_ext"] / geom),
        "Q_sca": float(cs["C_sca"] / geom),
        "Q_abs": float(cs["C_abs"] / geom),
    }
