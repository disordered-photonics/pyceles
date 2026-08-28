"""Hybrid Rayleigh/Ewald machinery for two-dimensional periodic coupling.

The global Rayleigh part is applied with z-sorted semiseparable scans.  A
uniform vertical exclusion band is omitted from those scans and is evaluated
with the exact Ewald structural sums by the coupling operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from pyceles.core.conversions import transformation_coefficients
from pyceles.core.indexing import index_vswf, iter_modes, n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic.scalar import chebyshev_shell_indices
from pyceles.core.periodic.types import PeriodicSpec
from pyceles.core.spherical import legendre_normalized_trigon, spherical_functions_trigon

Array = np.ndarray
# Per-batch temporary budget; persistent cache residency uses the shared
# guarded CuPy allocator policy rather than this fixed value.
RAYLEIGH_TRANSIENT_WORKSPACE_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class RayleighNearCacheEstimate:
    """Compact exact-near cache shape and persistent byte estimate."""

    pair_count: int
    structural_channels: int
    dtype: np.dtype
    structural_bytes: int


def rayleigh_near_cache_estimate(
    *, pair_count: int, lmax: int, dtype: npt.DTypeLike
) -> RayleighNearCacheEstimate:
    """Return the compact exact-near structural-cache byte estimate."""
    count = int(pair_count)
    if count < 0:
        raise ValueError(f"`pair_count` must be non-negative. Got {pair_count!r}.")
    ctype = np.dtype(dtype)
    if ctype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise TypeError(f"Rayleigh near caches require complex64 or complex128. Got {ctype!r}.")
    channels = int(valid_structural_indices(int(lmax))[0].size)
    return RayleighNearCacheEstimate(
        pair_count=count,
        structural_channels=channels,
        dtype=ctype,
        structural_bytes=count * channels * int(ctype.itemsize),
    )


def _compact_particle_index_dtype(*sizes: int) -> np.dtype:
    """Return the narrowest signed dtype that can index all requested rows."""
    maximum = max((int(size) for size in sizes), default=0)
    return np.dtype(np.int32 if maximum <= np.iinfo(np.int32).max else np.int64)


def resolve_rayleigh_z_cut(
    *,
    k: float,
    circumscribing_radii: npt.ArrayLike | None,
    requested: float | None,
) -> float:
    """Resolve a conservative uniform half-band for exact Ewald coupling.

    The default is one medium wavelength in ``|dz|`` and never smaller than
    twice the largest circumscribing radius.  The latter guarantees vertical
    separation of every pair delegated to the plane-wave representation.
    """
    k_f = float(k)
    if not np.isfinite(k_f) or k_f <= 0.0:
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    radii = (
        np.zeros((0,), dtype=float)
        if circumscribing_radii is None
        else np.asarray(circumscribing_radii, dtype=float).reshape(-1)
    )
    if radii.size and (np.any(~np.isfinite(radii)) or np.any(radii < 0.0)):
        raise ValueError("`circumscribing_radii` must be finite and non-negative.")
    minimum = 2.0 * float(np.max(radii, initial=0.0))
    if requested is None:
        value = max(2.0 * math.pi / k_f, minimum)
        # The exact-near/far partition is inclusive at ``|dz| == z_cut``.
        # Nondimensional wavelength conversions can land one ulp below a
        # simple coordinate difference (for example 2*pi/(2*pi/500)).
        return float(np.nextafter(value, np.inf))
    value = float(requested)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"`rayleigh_z_cut` must be finite and positive. Got {requested!r}.")
    if value + 32.0 * np.finfo(float).eps * max(1.0, minimum) < minimum:
        raise ValueError(
            "`rayleigh_z_cut` must be at least twice the largest particle "
            f"circumscribing radius ({minimum:g}). Got {value:g}."
        )
    # Treat the user-facing boundary as inclusive despite harmless binary
    # roundoff in coordinate differences.
    return float(np.nextafter(value, np.inf))


def reciprocal_modes(
    *,
    lattice: RectangularLattice2D,
    k_parallel: npt.ArrayLike,
    half_width: int,
) -> Array:
    """Return ``k_parallel + p*b1 + q*b2`` on a square reciprocal window."""
    h = int(half_width)
    if h < 0:
        raise ValueError(f"`half_width` must be non-negative. Got {half_width!r}.")
    indices = np.arange(-h, h + 1, dtype=np.int64)
    p, q = np.meshgrid(indices, indices, indexing="ij")
    reciprocal = (
        p.reshape(-1, 1) * np.asarray(lattice.b1, dtype=float)[None, :]
        + q.reshape(-1, 1) * np.asarray(lattice.b2, dtype=float)[None, :]
    )
    return np.asarray(
        np.asarray(k_parallel, dtype=float).reshape(1, 2) + reciprocal,
        dtype=float,
    )


def _reciprocal_gamma(k: float, q: Array) -> Array:
    rho = np.linalg.norm(np.asarray(q, dtype=float), axis=1)
    gamma = np.sqrt((float(k) ** 2 - rho**2) + 0.0j)
    if np.any(gamma == 0.0):
        raise ValueError(
            "The Rayleigh reciprocal basis contains a grazing diffraction order "
            "(Wood anomaly). Move away from the anomaly or use method='ewald'."
        )
    return np.asarray(gamma, dtype=np.complex128)


def resolve_rayleigh_half_width(
    *,
    lattice: RectangularLattice2D,
    k: float,
    k_parallel: npt.ArrayLike,
    lmax: int,
    z_cut: float,
    tolerance: float,
    max_shells: int,
    requested: int | None,
) -> int:
    """Choose a reciprocal half-width from a conservative evanescent envelope."""
    if requested is not None:
        value = int(requested)
        if value != requested or value < 0:
            raise ValueError(
                "`rayleigh_reciprocal_shells` must be a non-negative integer or None. "
                f"Got {requested!r}."
            )
        return value
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"`shell_tolerance` must be finite and positive. Got {tolerance!r}.")
    cap = int(max_shells)
    if cap <= 0:
        raise ValueError(f"`max_shells` must be positive. Got {max_shells!r}.")
    k_f = float(k)
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    lmax_i = int(lmax)
    if lmax_i < 0:
        raise ValueError(f"`lmax` must be non-negative. Got {lmax!r}.")
    degree_max = 2 * lmax_i
    prefactor = 2.0 * math.pi / (float(lattice.area) * k_f)
    power = max(1, degree_max - 1)
    for shell in range(cap + 1):
        idx = np.asarray(chebyshev_shell_indices(shell), dtype=np.int64)
        q = (
            kp[None, :]
            + idx[:, 0, None] * np.asarray(lattice.b1, dtype=float)[None, :]
            + idx[:, 1, None] * np.asarray(lattice.b2, dtype=float)[None, :]
        )
        rho = np.linalg.norm(q, axis=1)
        gamma = _reciprocal_gamma(k_f, q)
        if np.any(np.imag(gamma) <= 0.0):
            continue
        # A Rayleigh structural term contains a normalized associated-Legendre
        # factor evaluated at the complex angle of each evanescent order. At
        # high structural degree this factor can exceed the old radial-power
        # proxy by many orders of magnitude. Bound the same degree range used
        # by the evaluator, together with its scalar prefactor and 1/gamma
        # decay. Keep the historical power proxy as a conservative fallback
        # for the asymptotic polynomial growth.
        angular = legendre_normalized_trigon(
            gamma / k_f,
            rho / k_f,
            degree_max,
            xp=np,
        )
        angular_bound = np.max(np.abs(np.asarray(angular)), axis=(0, 1))
        scaled = np.maximum(1.0, rho / k_f)
        envelope = (
            prefactor
            * np.exp(-np.imag(gamma) * float(z_cut))
            * np.maximum(angular_bound, scaled**power)
            / np.maximum(np.abs(gamma), 1.0e-14)
        )
        shell_small = float(idx.shape[0]) * float(np.max(envelope, initial=0.0)) <= tol
        if shell_small:
            return shell
    raise ValueError(
        "Rayleigh reciprocal truncation did not reach the requested tolerance "
        f"{tol:g} by shell {cap}. Increase `max_shells`, enlarge "
        "`rayleigh_z_cut`, or set `rayleigh_reciprocal_shells` explicitly."
    )


@cache
def valid_structural_indices(lmax: int) -> tuple[Array, Array]:
    """Return dense-table indices for all valid ``(p, m)`` channels."""
    order = 2 * int(lmax)
    degrees: list[int] = []
    orders: list[int] = []
    for degree in range(order + 1):
        for azimuthal_order in range(-degree, degree + 1):
            degrees.append(degree)
            orders.append(azimuthal_order + order)
    degree_array = np.asarray(degrees, dtype=np.int32)
    order_array = np.asarray(orders, dtype=np.int32)
    degree_array.flags.writeable = False
    order_array.flags.writeable = False
    return degree_array, order_array


@dataclass(frozen=True)
class RayleighPlan:
    """Reusable reciprocal-mode and multipole projection tables."""

    lmax: int
    gamma: Array
    reciprocal_xy: Array
    sorted_xy_phase: Array
    source_tables: Array
    destination_tables: Array
    weights: Array
    sort_order: Array
    inverse_order: Array
    sorted_z: Array
    z_cut: float
    half_width: int

    @property
    def n_modes_reciprocal(self) -> int:
        return int(self.gamma.size)


def build_rayleigh_plan(
    *,
    lmax: int,
    k: float,
    positions: npt.ArrayLike,
    lattice: RectangularLattice2D,
    k_parallel: npt.ArrayLike,
    z_cut: float,
    half_width: int,
    dtype: npt.DTypeLike,
) -> RayleighPlan:
    """Build NumPy tables for the global far-Rayleigh operator."""
    lmax_i = int(lmax)
    ctype = np.dtype(dtype)
    if ctype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise TypeError(f"Rayleigh plans require complex64 or complex128. Got {ctype!r}.")
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    q = reciprocal_modes(lattice=lattice, k_parallel=k_parallel, half_width=int(half_width))
    rho = np.linalg.norm(q, axis=1)
    gamma = _reciprocal_gamma(float(k), q)
    alpha = np.arctan2(q[:, 1], q[:, 0])
    nm = n_modes(lmax_i)
    mode_m = np.zeros((nm,), dtype=np.int32)
    source = np.zeros((2, q.shape[0], 2, nm), dtype=np.complex128)
    destination = np.zeros_like(source)
    for direction_index, direction in enumerate((1.0, -1.0)):
        ct = direction * gamma / float(k)
        st = rho / float(k)
        pi, tau = spherical_functions_trigon(ct, st, lmax_i, xp=np)
        for tau_mode, degree, order, mode_index in iter_modes(lmax_i):
            mode_m[mode_index] = int(order)
            for polarization in (1, 2):
                source[direction_index, :, polarization - 1, mode_index] = (
                    transformation_coefficients(
                        pi,
                        tau,
                        tau_mode,
                        degree,
                        order,
                        polarization,
                        dagger=False,
                    )
                )
                destination[direction_index, :, polarization - 1, mode_index] = (
                    transformation_coefficients(
                        pi,
                        tau,
                        tau_mode,
                        degree,
                        order,
                        polarization,
                        dagger=True,
                    )
                )
    source *= np.exp(1j * alpha[:, None] * mode_m[None, :])[None, :, None, :]
    destination *= np.exp(-1j * alpha[:, None] * mode_m[None, :])[None, :, None, :]
    xy_phase = np.exp(1j * (pos[:, :2] @ q.T)).astype(ctype, copy=False)
    weights = np.asarray(
        8.0 * np.pi / (float(lattice.area) * float(k) * gamma),
        dtype=ctype,
    )
    index_dtype = _compact_particle_index_dtype(pos.shape[0])
    sort_order = np.argsort(pos[:, 2], kind="stable").astype(index_dtype, copy=False)
    inverse_order = np.empty_like(sort_order)
    inverse_order[sort_order] = np.arange(sort_order.size, dtype=index_dtype)
    return RayleighPlan(
        lmax=lmax_i,
        gamma=np.asarray(gamma, dtype=np.complex128),
        reciprocal_xy=np.asarray(q, dtype=float),
        sorted_xy_phase=np.asarray(xy_phase[sort_order], dtype=ctype),
        source_tables=np.asarray(source, dtype=ctype),
        destination_tables=np.asarray(destination, dtype=ctype),
        weights=weights,
        sort_order=sort_order,
        inverse_order=inverse_order,
        sorted_z=np.asarray(pos[sort_order, 2], dtype=float),
        z_cut=float(z_cut),
        half_width=int(half_width),
    )


def rayleigh_structural_sums_2d_batch(
    *,
    max_degree: int,
    k: float,
    displacements: npt.ArrayLike,
    lattice: RectangularLattice2D,
    k_parallel: npt.ArrayLike,
    tolerance: float,
    max_shells: int,
    requested_half_width: int | None = None,
    matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> Array:
    """Return off-plane periodic structural sums from reciprocal orders.

    The shifted two-dimensional lattice sum has an absolutely convergent
    Rayleigh representation whenever the vertical displacement is nonzero.
    Evaluating that representation directly avoids the separately enormous
    factors that occur in a split-Ewald shifted recurrence for tall cells.

    Displacements sharing one absolute height are evaluated together as a
    matrix product over reciprocal orders.  The result uses pyceles's dense
    CELES-normalized structural-table layout.
    """
    if matmul_backend not in {"numpy", "cupy"}:
        raise ValueError(f"`matmul_backend` must be 'numpy' or 'cupy'. Got {matmul_backend!r}.")
    cupy = None
    if matmul_backend == "cupy":
        from pyceles._optional import import_cupy

        cupy, _ = import_cupy()

    degree_max = int(max_degree)
    if degree_max < 0:
        raise ValueError(f"`max_degree` must be >= 0. Got {max_degree!r}.")
    k_f = float(k)
    if not np.isfinite(k_f) or k_f <= 0.0:
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    delta = np.asarray(displacements, dtype=float).reshape(-1, 3)
    if np.any(~np.isfinite(delta)):
        raise ValueError("`displacements` must be finite.")
    if np.any(delta[:, 2] == 0.0):
        raise ValueError(
            "Rayleigh structural sums require nonzero vertical displacements; "
            "same-plane offsets must use Ewald summation."
        )
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    out = np.zeros(
        (delta.shape[0], degree_max + 1, 2 * degree_max + 1),
        dtype=np.complex128,
    )
    if delta.shape[0] == 0:
        return out

    # The existing truncation envelope is parameterized by particle lmax,
    # whose translation table reaches degree 2*lmax.  Map the explicit box
    # degree to the smallest equivalent value.
    envelope_lmax = max(1, (degree_max + 1) // 2)
    prefactor = 2.0 * math.pi / (float(lattice.area) * k_f)
    table_offset = degree_max
    channel_count = (degree_max + 1) ** 2
    degrees = np.empty((channel_count,), dtype=np.int32)
    orders = np.empty((channel_count,), dtype=np.int32)
    channel = 0
    for degree in range(degree_max + 1):
        for order in range(-degree, degree + 1):
            degrees[channel] = degree
            orders[channel] = order
            channel += 1
    channel_prefactors = prefactor * (-1j) ** degrees
    reflection_parity = (-1.0) ** (degrees + np.abs(orders))

    abs_heights = np.abs(delta[:, 2])
    for height in np.unique(abs_heights):
        height_mask = abs_heights == height
        group_indices = np.flatnonzero(height_mask)
        half_width = resolve_rayleigh_half_width(
            lattice=lattice,
            k=k_f,
            k_parallel=kp,
            lmax=envelope_lmax,
            z_cut=float(height),
            tolerance=float(tolerance),
            max_shells=int(max_shells),
            requested=requested_half_width,
        )
        wave_xy = reciprocal_modes(
            lattice=lattice,
            k_parallel=kp,
            half_width=int(half_width),
        )
        rho = np.linalg.norm(wave_xy, axis=1)
        gamma = _reciprocal_gamma(k_f, wave_xy)
        alpha = np.arctan2(wave_xy[:, 1], wave_xy[:, 0])
        st = rho / k_f
        lateral_phase = np.exp(1j * (wave_xy @ delta[group_indices, :2].T))
        propagation = np.exp(1j * gamma * float(height)) / gamma
        radial_phase = propagation[:, None] * lateral_phase

        plm = np.asarray(
            legendre_normalized_trigon(gamma / k_f, st, degree_max, xp=np),
            dtype=np.complex128,
        )
        angular = np.empty((channel_count, wave_xy.shape[0]), dtype=np.complex128)
        order_phases = {
            order: np.exp(1j * order * alpha) for order in range(-degree_max, degree_max + 1)
        }
        for channel_index, (degree_raw, order_raw) in enumerate(zip(degrees, orders, strict=True)):
            degree_index = int(degree_raw)
            order_index = int(order_raw)
            angular[channel_index] = plm[degree_index, abs(order_index)] * order_phases[order_index]
        contracted = (
            angular @ radial_phase
            if cupy is None
            else cupy.asnumpy(cupy.asarray(angular) @ cupy.asarray(radial_phase))
        )
        values = contracted.T * channel_prefactors[None, :]
        negative_height = delta[group_indices, 2] < 0.0
        if np.any(negative_height):
            values[negative_height] *= reflection_parity[None, :]
        out[
            group_indices[:, None],
            degrees[None, :],
            (orders + table_offset)[None, :],
        ] = values

    if not np.all(np.isfinite(out)):
        raise FloatingPointError(
            "Periodic Rayleigh structural evaluation produced non-finite coefficients "
            f"at order={degree_max}."
        )
    return out


def prepare_rayleigh_plan(
    *,
    lmax: int,
    k: float,
    positions: npt.ArrayLike,
    circumscribing_radii: npt.ArrayLike | None,
    periodic: PeriodicSpec,
    k_parallel: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> RayleighPlan:
    """Resolve hybrid-policy defaults and build one reusable Rayleigh plan."""
    options = periodic.options
    z_cut = resolve_rayleigh_z_cut(
        k=float(k),
        circumscribing_radii=circumscribing_radii,
        requested=options.rayleigh_z_cut,
    )
    half_width = resolve_rayleigh_half_width(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=k_parallel,
        lmax=int(lmax),
        z_cut=z_cut,
        tolerance=float(options.shell_tolerance),
        max_shells=int(options.max_shells),
        requested=options.rayleigh_reciprocal_shells,
    )
    return build_rayleigh_plan(
        lmax=int(lmax),
        k=float(k),
        positions=positions,
        lattice=periodic.lattice,
        k_parallel=k_parallel,
        z_cut=z_cut,
        half_width=half_width,
        dtype=dtype,
    )


def _scan_far_numpy(
    *,
    source_amplitudes: Array,
    z: Array,
    gamma: Array,
    z_cut: float,
    upward: bool,
) -> Array:
    """Apply one delayed-entry semiseparable scan on a shared z grid."""
    return _scan_far_to_destinations_numpy(
        source_amplitudes=source_amplitudes,
        source_z=z,
        destination_z=z,
        gamma=gamma,
        z_cut=z_cut,
        upward=upward,
    )


def _scan_far_to_destinations_numpy(
    *,
    source_amplitudes: Array,
    source_z: Array,
    destination_z: Array,
    gamma: Array,
    z_cut: float,
    upward: bool,
) -> Array:
    """Scan sorted source amplitudes onto a distinct sorted destination grid."""
    src = np.asarray(source_amplitudes)
    src_z = np.asarray(source_z, dtype=float).reshape(-1)
    dst_z = np.asarray(destination_z, dtype=float).reshape(-1)
    gamma_arr = np.asarray(gamma, dtype=np.complex128).reshape(-1)
    if src.shape[0] != src_z.size:
        raise ValueError("Rayleigh source amplitudes and source z coordinates disagree.")
    out = np.zeros((dst_z.size, *src.shape[1:]), dtype=src.dtype)
    acc = np.zeros(src.shape[1:], dtype=src.dtype)
    gamma_shape = (gamma_arr.size,) + (1,) * (src.ndim - 2)
    gamma_b = gamma_arr.reshape(gamma_shape)
    if upward:
        pointer = 0
        for i in range(dst_z.size):
            if i:
                acc *= np.exp(1j * gamma_b * (dst_z[i] - dst_z[i - 1]))
            while pointer < src_z.size and dst_z[i] - src_z[pointer] > float(z_cut):
                acc += src[pointer] * np.exp(1j * gamma_b * (dst_z[i] - src_z[pointer]))
                pointer += 1
            out[i] = acc
        return out
    pointer = src_z.size - 1
    for i in range(dst_z.size - 1, -1, -1):
        if i < dst_z.size - 1:
            acc *= np.exp(1j * gamma_b * (dst_z[i + 1] - dst_z[i]))
        while pointer >= 0 and src_z[pointer] - dst_z[i] > float(z_cut):
            acc += src[pointer] * np.exp(1j * gamma_b * (src_z[pointer] - dst_z[i]))
            pointer -= 1
        out[i] = acc
    return out


def resolve_rayleigh_mode_chunk_size(
    *,
    n_modes_reciprocal: int,
    n_particles: int,
    n_destinations: int | None = None,
    n_phase_rows: int = 0,
    n_rhs: int,
    dtype: npt.DTypeLike,
    workspace_bytes: int = RAYLEIGH_TRANSIENT_WORKSPACE_BYTES,
) -> int:
    """Bound one reciprocal chunk by the live source and destination work arrays."""
    cap = max(1, int(n_modes_reciprocal))
    sources = max(1, int(n_particles))
    destinations = sources if n_destinations is None else max(1, int(n_destinations))
    phase_rows = max(0, int(n_phase_rows))
    rhs = max(1, int(n_rhs))
    itemsize = int(np.dtype(dtype).itemsize)
    bytes_per_mode = ((sources + destinations) * 2 * rhs + phase_rows) * itemsize
    memory_cap = max(1, int(workspace_bytes) // max(1, bytes_per_mode))
    return min(cap, memory_cap)


def apply_rayleigh_far_numpy(plan: RayleighPlan, x: npt.ArrayLike) -> Array:
    """Apply the far-only Rayleigh operator to one or more right-hand sides."""
    arr_raw = np.asarray(x)
    squeezed = arr_raw.ndim == 2
    if squeezed:
        arr = arr_raw[:, :, None]
    elif arr_raw.ndim == 3:
        arr = arr_raw
    else:
        raise ValueError("Rayleigh input must have shape (N, Nm) or (N, Nm, nrhs).")
    if arr.shape[0] != plan.sort_order.size:
        raise ValueError("Rayleigh input particle count does not match the plan.")
    order = plan.sort_order
    arr_sorted = np.asarray(arr[order], dtype=plan.source_tables.dtype)
    phase_sorted = plan.sorted_xy_phase
    y_sorted = np.zeros_like(arr_sorted)
    chunk = resolve_rayleigh_mode_chunk_size(
        n_modes_reciprocal=plan.n_modes_reciprocal,
        n_particles=arr.shape[0],
        n_rhs=arr.shape[2],
        dtype=arr_sorted.dtype,
    )
    for start in range(0, plan.n_modes_reciprocal, chunk):
        stop = min(plan.n_modes_reciprocal, start + chunk)
        phase = phase_sorted[:, start:stop]
        gamma = plan.gamma[start:stop]
        weight = plan.weights[start:stop]
        for direction, upward in ((0, True), (1, False)):
            source = np.einsum(
                "qpm,amr->aqpr",
                plan.source_tables[direction, start:stop],
                arr_sorted,
                optimize=True,
            )
            source *= np.conjugate(phase)[:, :, None, None]
            incoming = _scan_far_numpy(
                source_amplitudes=source,
                z=plan.sorted_z,
                gamma=gamma,
                z_cut=plan.z_cut,
                upward=upward,
            )
            y_sorted += np.einsum(
                "qpm,aqpr,aq,q->amr",
                plan.destination_tables[direction, start:stop],
                incoming,
                phase,
                weight,
                optimize=True,
            )
            del source, incoming
    y = y_sorted[plan.inverse_order]
    return cast(Array, y[:, :, 0] if squeezed else y)


def apply_rayleigh_far_to_points_numpy(
    plan: RayleighPlan,
    coeffs: npt.ArrayLike,
    points: npt.ArrayLike,
) -> Array:
    """Return far-source point-local regular ``l=1`` coefficients.

    Source particles and observation points use distinct z-sorted grids. Only
    sources with ``|z_point-z_source| > plan.z_cut`` participate; callers add
    exact Ewald contributions for the complementary source-point pairs.
    """
    coeff_raw = np.asarray(coeffs)
    squeezed = coeff_raw.ndim == 2
    if squeezed:
        coeff_arr = coeff_raw[:, :, None]
    elif coeff_raw.ndim == 3:
        coeff_arr = coeff_raw
    else:
        raise ValueError("Rayleigh coefficients must have shape (N, Nm) or (N, Nm, nrhs).")
    if coeff_arr.shape[0] != plan.sort_order.size:
        raise ValueError("Rayleigh coefficient particle count does not match the plan.")
    if coeff_arr.shape[1] != plan.source_tables.shape[-1]:
        raise ValueError("Rayleigh coefficient mode count does not match the plan.")

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n_rhs = int(coeff_arr.shape[2])
    if pts.shape[0] == 0:
        empty = np.zeros((0, 6, n_rhs), dtype=plan.source_tables.dtype)
        return empty[:, :, 0] if squeezed else empty

    destination_index_dtype = _compact_particle_index_dtype(pts.shape[0])
    destination_order = np.argsort(pts[:, 2], kind="stable").astype(
        destination_index_dtype, copy=False
    )
    destination_inverse = np.empty_like(destination_order)
    destination_inverse[destination_order] = np.arange(
        destination_order.size, dtype=destination_index_dtype
    )
    pts_sorted = pts[destination_order]
    coeff_sorted = np.asarray(coeff_arr[plan.sort_order], dtype=plan.source_tables.dtype)
    y_sorted = np.zeros((pts.shape[0], 6, n_rhs), dtype=plan.source_tables.dtype)
    chunk = resolve_rayleigh_mode_chunk_size(
        n_modes_reciprocal=plan.n_modes_reciprocal,
        n_particles=coeff_arr.shape[0],
        n_destinations=pts.shape[0],
        n_phase_rows=pts.shape[0],
        n_rhs=n_rhs,
        dtype=coeff_sorted.dtype,
    )
    l1_indices = np.asarray(
        [index_vswf(1, m, tau, plan.lmax) for tau in (1, 2) for m in (-1, 0, 1)],
        dtype=np.int64,
    )
    for start in range(0, plan.n_modes_reciprocal, chunk):
        stop = min(plan.n_modes_reciprocal, start + chunk)
        source_phase = plan.sorted_xy_phase[:, start:stop]
        destination_phase = np.exp(
            1j * (pts_sorted[:, :2] @ plan.reciprocal_xy[start:stop].T)
        ).astype(plan.source_tables.dtype, copy=False)
        gamma = plan.gamma[start:stop]
        weight = plan.weights[start:stop]
        for direction, upward in ((0, True), (1, False)):
            source = np.einsum(
                "qpm,amr->aqpr",
                plan.source_tables[direction, start:stop],
                coeff_sorted,
                optimize=True,
            )
            source *= np.conjugate(source_phase)[:, :, None, None]
            incoming = _scan_far_to_destinations_numpy(
                source_amplitudes=source,
                source_z=plan.sorted_z,
                destination_z=pts_sorted[:, 2],
                gamma=gamma,
                z_cut=plan.z_cut,
                upward=upward,
            )
            y_sorted += np.einsum(
                "qpm,dqpr,dq,q->dmr",
                np.take(
                    plan.destination_tables[direction, start:stop],
                    l1_indices,
                    axis=-1,
                ),
                incoming,
                destination_phase,
                weight,
                optimize=True,
            )
            del source, incoming
    y = y_sorted[destination_inverse]
    return cast(Array, y[:, :, 0] if squeezed else y)


def rayleigh_block(
    *,
    plan: RayleighPlan,
    destination_index: int,
    source_index: int,
) -> Array:
    """Return one truncated Rayleigh block using the plan's directional tables."""
    dst = int(destination_index)
    src = int(source_index)
    dz = float(plan.sorted_z[plan.inverse_order[dst]] - plan.sorted_z[plan.inverse_order[src]])
    direction = 0 if dz >= 0.0 else 1
    dst_sorted = int(plan.inverse_order[dst])
    src_sorted = int(plan.inverse_order[src])
    phase = plan.sorted_xy_phase[dst_sorted] * np.conjugate(plan.sorted_xy_phase[src_sorted])
    propagation = np.exp(1j * plan.gamma * abs(dz))
    return cast(
        Array,
        np.einsum(
            "qpn,qpm,q,q,q->nm",
            plan.destination_tables[direction],
            plan.source_tables[direction],
            phase,
            propagation,
            plan.weights,
            optimize=True,
        ),
    )


def near_pair_csr(positions: npt.ArrayLike, z_cut: float) -> tuple[Array, Array, Array]:
    """Return source-major CSR data for exact non-self near pairs.

    A pair belongs to the exact band when ``|z_dst-z_src| <= z_cut``. Self
    interactions are deliberately omitted because their periodic block is
    translationally invariant and can be prepared once for all particles. Pair
    indices use int32 whenever the particle count permits; CSR offsets remain
    int64 because the directed pair count can exceed the int32 range.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = int(pos.shape[0])
    index_dtype = _compact_particle_index_dtype(n)
    if n == 0:
        empty = np.zeros((0,), dtype=index_dtype)
        return np.zeros((1,), dtype=np.int64), empty, empty
    z = pos[:, 2]
    order = np.argsort(z, kind="stable").astype(index_dtype, copy=False)
    sorted_z = z[order]
    lo = np.searchsorted(sorted_z, z - float(z_cut), side="left")
    hi = np.searchsorted(sorted_z, z + float(z_cut), side="right")
    counts_with_self = np.asarray(hi - lo, dtype=np.int64)

    # The vertically sparse case is the intended large-N regime. Avoid
    # materializing an N-entry temporary pair table when every band contains
    # only the source itself.
    if np.all(counts_with_self == 1) and np.array_equal(order[lo], np.arange(n, dtype=index_dtype)):
        empty = np.zeros((0,), dtype=index_dtype)
        return np.zeros((n + 1,), dtype=np.int64), empty, empty

    indptr_with_self = np.empty((n + 1,), dtype=np.int64)
    indptr_with_self[0] = 0
    np.cumsum(counts_with_self, out=indptr_with_self[1:])
    total = int(indptr_with_self[-1])
    sources = np.repeat(np.arange(n, dtype=index_dtype), counts_with_self)
    offsets = np.arange(total, dtype=np.int64) - indptr_with_self[sources]
    destinations = order[lo[sources] + offsets]
    keep = destinations != sources
    sources = np.asarray(sources[keep], dtype=index_dtype)
    destinations = np.asarray(destinations[keep], dtype=index_dtype)

    counts = np.bincount(sources, minlength=n).astype(np.int64, copy=False)
    indptr = np.empty((n + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, destinations, sources


def near_point_source_csr(
    points: npt.ArrayLike,
    source_positions: npt.ArrayLike,
    z_cut: float,
) -> tuple[Array, Array, Array]:
    """Return source-major CSR data for exact point-source pairs in the z band."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    sources_pos = np.asarray(source_positions, dtype=float).reshape(-1, 3)
    n_sources = int(sources_pos.shape[0])
    n_points = int(pts.shape[0])
    index_dtype = _compact_particle_index_dtype(n_sources, n_points)
    if n_sources == 0 or n_points == 0:
        empty = np.zeros((0,), dtype=index_dtype)
        return np.zeros((n_sources + 1,), dtype=np.int64), empty, empty
    order = np.argsort(pts[:, 2], kind="stable").astype(index_dtype, copy=False)
    sorted_z = pts[order, 2]
    source_z = sources_pos[:, 2]
    lo = np.searchsorted(sorted_z, source_z - float(z_cut), side="left")
    hi = np.searchsorted(sorted_z, source_z + float(z_cut), side="right")
    counts = np.asarray(hi - lo, dtype=np.int64)
    indptr = np.empty((n_sources + 1,), dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    source_indices = np.repeat(np.arange(n_sources, dtype=index_dtype), counts)
    offsets = np.arange(total, dtype=np.int64) - indptr[source_indices]
    destination_indices = np.asarray(
        order[lo[source_indices] + offsets],
        dtype=index_dtype,
    )
    return indptr, destination_indices, source_indices


__all__ = [
    "RayleighNearCacheEstimate",
    "RayleighPlan",
    "apply_rayleigh_far_numpy",
    "apply_rayleigh_far_to_points_numpy",
    "build_rayleigh_plan",
    "near_pair_csr",
    "near_point_source_csr",
    "prepare_rayleigh_plan",
    "rayleigh_block",
    "rayleigh_near_cache_estimate",
    "reciprocal_modes",
    "resolve_rayleigh_half_width",
    "resolve_rayleigh_mode_chunk_size",
    "resolve_rayleigh_z_cut",
    "valid_structural_indices",
]
