"""Spherical response kernels: homogeneous/layered Mie and CELES T entries.

CPU-first implementation using NumPy + SciPy.

Conventions
-----------
- Input `k_medium` is the **wavenumber in the embedding medium**:

    k_medium = 2*pi/lambda * n_medium

  (CELES uses this convention throughout for translations and field evaluation.)

- `mie_ab` returns the standard Mie scattering coefficients (a_l, b_l)
  for l=1..lmax.

- `sphere_T_diagonal` maps homogeneous-sphere Mie coefficients into CELES'
  diagonal T entries.

- `layered_sphere_T_diagonal` does the same for concentric multilayer spheres
  through interface transfer matrices.

- `layered_internal_ab_ratios` provides per-layer radial coefficients for
  piecewise internal near-field evaluation in layered spheres.

- `sphere_internal_ratios` provides the per-l conversion factors used by CELES
  for homogeneous spheres.
  following `T_entry.m` in the CELES MATLAB code:

    tau=1 (M/TE-like) uses -b_l
    tau=2 (N/TM-like) uses -a_l

"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .particles import LayeredSphere, Particle, Sphere, Spheroid


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
        "Supported particle backends are Sphere and LayeredSphere. "
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
        raise NotImplementedError(_unsupported_particle_message(particle))
    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


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
