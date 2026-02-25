"""Single-sphere response: Mie coefficients and CELES diagonal T entries.

CPU-first implementation using NumPy + SciPy.

Conventions
-----------
- Input `k_medium` is the **wavenumber in the embedding medium**:

    k_medium = 2*pi/lambda * n_medium

  (CELES uses this convention throughout for translations and field evaluation.)

- `mie_ab` returns the standard Mie scattering coefficients (a_l, b_l)
  for l=1..lmax.

- `sphere_T_diagonal` maps Mie coefficients into CELES' diagonal T entries
  following `T_entry.m` in the CELES MATLAB code:

    tau=1 (M/TE-like) uses -b_l
    tau=2 (N/TM-like) uses -a_l

- `sphere_internal_ratios` provides the per-l conversion factors used by CELES
  to convert scattered SVWF coefficients into internal (regular) coefficients.

This module intentionally contains **no JAX** code.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .particles import Ellipsoid, LayeredSphere, Particle, Sphere


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
        "Only Sphere is currently supported. Planned future backends include layered-sphere "
        "and non-spherical solvers."
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
    if isinstance(particle, (LayeredSphere, Ellipsoid)):
        raise NotImplementedError(_unsupported_particle_message(particle))
    raise TypeError(f"Unsupported particle instance: {type(particle)!r}")


def particle_internal_ratios(
    lmax: int,
    k_medium: complex,
    particle: Particle,
    n_medium: complex = 1.0 + 0j,
) -> dict[int, np.ndarray]:
    """Dispatch internal/scattered conversion ratios by particle type."""
    if isinstance(particle, Sphere):
        return sphere_internal_ratios(
            lmax=lmax,
            k_medium=k_medium,
            radius=particle.radius,
            n_particle=particle.refractive_index,
            n_medium=n_medium,
        )
    if isinstance(particle, (LayeredSphere, Ellipsoid)):
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
