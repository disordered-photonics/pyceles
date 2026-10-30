"""Nonmagnetic spherical interface responses to incident and internal fields.

Regular core and outgoing host conditions leave two unknowns per interface.
Ordering the unknowns by material region makes the continuity system banded,
with two sub- and super-diagonals. Setup and solution scale linearly with the
number of layers and do not form products of transfer matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import spherical_jn

from .indexing import n_modes
from .particles import LayeredSphere, Sphere
from .radial import dx_xz_hankel1, passive_wavenumber, sph_hankel1
from .svwf import mode_indices_by_l


def spherical_material_profile(
    particle: Sphere | LayeredSphere, n_medium: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Return radii and absolute indices, dropping identical internal boundaries."""
    if isinstance(particle, Sphere):
        radii = np.array([particle.radius], dtype=float)
        indices = np.array([particle.refractive_index, n_medium], dtype=np.complex128)
    else:
        radii = np.asarray(particle.layer_radii, dtype=float)
        indices = np.array([*particle.layer_refractive_indices, n_medium], dtype=np.complex128)
    for n in indices:
        passive_wavenumber(n)
    keep = np.r_[indices[:-2] != indices[1:-1], True]
    return radii[keep], np.r_[indices[:-1][keep], indices[-1]]


def _tangential_trace(l: int, x: complex, n: complex, tau: int, outgoing: bool) -> np.ndarray:
    if outgoing:
        z = complex(sph_hankel1(l, x))
        derivative = complex(dx_xz_hankel1(l, x)) / x
    else:
        z = spherical_jn(l, x)
        derivative = z / x + spherical_jn(l, x, derivative=True)
    # Z0*H has the same -i factor on both sides, which cancels in continuity.
    return np.array([z, n * derivative] if tau == 1 else [derivative, n * z])


def spherical_response_coefficients(
    *,
    wavelength: float,
    radii: np.ndarray,
    indices: np.ndarray,
    lmax: int,
    incident: np.ndarray | None = None,
    source_layer: int | None = None,
    source_regular: np.ndarray | None = None,
    source_outgoing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve homogeneous regular/outgoing amplitudes in every material region.

    `incident` is the regular external exciting field in the host. An internal
    particular field has regular/outgoing expansions at its layer's inner/outer
    boundaries, respectively. Those are RHS jumps, not stored homogeneous
    amplitudes. All radial evaluation and matching use complex128 setup values.
    """
    layers = len(radii)
    modes = n_modes(lmax)
    k0 = 2 * np.pi / wavelength
    regular = np.zeros((layers + 1, modes), dtype=np.complex128)
    outgoing = np.zeros_like(regular)
    if incident is not None:
        regular[-1] = incident
    unknowns = (
        [(0, False)]
        + [(region, family) for region in range(1, layers) for family in (False, True)]
        + [(layers, True)]
    )
    width = min(2, len(unknowns) - 1)
    for l, (_, _, m_columns, n_columns) in enumerate(mode_indices_by_l(lmax), 1):
        for tau, columns in ((1, m_columns), (2, n_columns)):
            band = np.zeros((2 * width + 1, 2 * layers), dtype=np.complex128)
            rhs = np.zeros((2 * layers, len(columns)), dtype=np.complex128)
            for boundary, radius in enumerate(radii):
                rows = slice(2 * boundary, 2 * boundary + 2)
                for col in range(max(0, 2 * boundary - 1), min(2 * layers, 2 * boundary + 3)):
                    region, family = unknowns[col]
                    trace = _tangential_trace(
                        l, k0 * indices[region] * radius, indices[region], tau, family
                    )
                    sign = 1 if region == boundary else -1
                    for offset in (0, 1):
                        row = 2 * boundary + offset
                        band[width + row - col, col] = sign * trace[offset]
                if source_layer is not None:
                    n_local = indices[source_layer]
                    if boundary == source_layer - 1 and source_regular is not None:
                        rhs[rows] += (
                            _tangential_trace(l, k0 * n_local * radius, n_local, tau, False)[
                                :, None
                            ]
                            * source_regular[columns]
                        )
                    if boundary == source_layer and source_outgoing is not None:
                        rhs[rows] -= (
                            _tangential_trace(l, k0 * n_local * radius, n_local, tau, True)[:, None]
                            * source_outgoing[columns]
                        )
                if boundary == layers - 1 and incident is not None:
                    n_host = indices[-1]
                    rhs[rows] += (
                        _tangential_trace(l, k0 * n_host * radius, n_host, tau, False)[:, None]
                        * regular[-1, columns]
                    )
            solution = solve_banded((width, width), band, rhs)
            for col, (region, family) in enumerate(unknowns):
                coefficients = outgoing if family else regular
                coefficients[region, columns] = solution[col]
    return regular, outgoing
