"""Physics invariants for periodic mixed-particle scattering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

import pyceles as pcl

_WAVELENGTH = 550.0
_N_MEDIUM = 1.0 + 0.0j
_AX = 900.0
_AY = 850.0
# A fixed non-lattice displacement keeps the test reproducible while forcing
# several centers through the periodic seams.
_XY_SHIFT = np.asarray([487.25, -331.75], dtype=float)
_SEAM_SHIFT = np.asarray([795.0, 0.0], dtype=float)


def _mixed_periodic_particles() -> list[pcl.Particle]:
    """Return a small, disjoint cluster with distinct particle families."""
    return [
        pcl.Sphere(
            position=(100.0, 150.0, 50.0),
            radius=25.0,
            refractive_index=1.50 + 0.01j,
        ),
        pcl.LayeredSphere(
            position=(320.0, 420.0, 170.0),
            layer_radii=(15.0, 30.0),
            layer_refractive_indices=(1.70 + 0.01j, 1.30 + 0.005j),
        ),
        pcl.PECSphere(position=(550.0, 220.0, 300.0), radius=35.0),
    ]


def _periodically_shift_particles(
    particles: list[pcl.Particle],
    *,
    shift: np.ndarray = _XY_SHIFT,
) -> list[pcl.Particle]:
    """Translate every center in xy and wrap it back into the unit cell."""
    shifted: list[pcl.Particle] = []
    for particle in particles:
        position = np.asarray(particle.position, dtype=float).copy()
        position[:2] = np.mod(position[:2] + shift, np.asarray([_AX, _AY]))
        shifted.append(replace(particle, position=tuple(position)))
    return shifted


def _laterally_close_pair(z_separation: float) -> list[pcl.Particle]:
    """Return two mixed particles whose lateral pair straddles a seam after shifting."""
    z0 = 100.0
    return [
        pcl.Sphere(
            position=(100.0, 200.0, z0),
            radius=4.0,
            refractive_index=1.50 + 0.01j,
        ),
        pcl.LayeredSphere(
            position=(110.0, 200.0, z0 + float(z_separation)),
            layer_radii=(2.0, 4.0),
            layer_refractive_indices=(1.70 + 0.01j, 1.30 + 0.005j),
        ),
    ]


def _seam_pair_and_translation(
    z_separation: float,
) -> tuple[list[pcl.Particle], list[pcl.Particle]]:
    """Build a close pair and verify that the chosen shift crosses one seam."""
    particles = _laterally_close_pair(z_separation)
    translated = _periodically_shift_particles(particles, shift=_SEAM_SHIFT)

    # The translation deliberately makes the pair look far apart in the
    # stored Cartesian coordinates: x=(100, 110) becomes x=(895, 5).
    raw_lateral_separation = abs(
        float(translated[1].position[0]) - float(translated[0].position[0])
    )
    if raw_lateral_separation <= 0.5 * _AX:
        raise AssertionError("The seam-shift fixture did not separate the stored x coordinates.")
    return particles, translated


def _periodic_options(method: Literal["ewald", "rayleigh"]) -> pcl.PeriodicOptions:
    """Use a compact, deterministic split that exercises both periodic paths."""
    kwargs: dict[str, Any] = dict(
        method=method,
        eta=2.0e-3,
        real_shells=4,
        reciprocal_shells=4,
    )
    if method == "rayleigh":
        # The z coordinates intentionally put some pairs on the reciprocal
        # scan side of the hybrid half-band.
        kwargs.update(rayleigh_z_cut=100.0, rayleigh_reciprocal_shells=8)
    return pcl.PeriodicOptions(**kwargs)


def _run_periodic_case(
    particles: list[pcl.Particle],
    *,
    polar_angle: float,
    method: Literal["ewald", "rayleigh"],
    backend: Literal["numpy", "cupy"],
) -> pcl.SimulationResult:
    source = pcl.PlaneWave(
        wavelength=_WAVELENGTH,
        medium_n=_N_MEDIUM,
        polarization="TE",
        polar_angle=float(polar_angle),
        azimuthal_angle=0.41,
        amplitude=1.0,
    )
    compute_dtype: Literal["complex64", "complex128"] = (
        "complex128" if backend == "numpy" else "complex64"
    )
    config = pcl.SimulationConfig(
        wavelength=_WAVELENGTH,
        n_medium=_N_MEDIUM,
        lmax=1,
        periodic=pcl.PeriodicSpec(
            lattice=pcl.RectangularLattice2D(ax=_AX, ay=_AY),
            options=_periodic_options(method),
        ),
        solver_method="direct",
        operator_backend=backend,
        postprocessing_backend="inherit",
        compute_dtype=compute_dtype,
        accum_dtype="complex128",
        verbose=False,
    )
    return pcl.Simulation(config, particles=particles).run(
        source,
        include_farfield=False,
    )


def _assert_translation_invariant_observables(
    reference: pcl.SimulationResult,
    translated: pcl.SimulationResult,
    *,
    rtol: float,
    atol: float,
) -> None:
    if reference.periodic is None or translated.periodic is None:
        raise AssertionError("Periodic simulations must expose periodic far-field payloads.")

    first = reference.periodic
    second = translated.periodic
    np.testing.assert_array_equal(second.order_mn, first.order_mn)
    np.testing.assert_array_equal(second.order_propagating, first.order_propagating)

    for name in ("reflected_flux_per_order", "transmitted_flux_per_order"):
        np.testing.assert_allclose(
            getattr(second, name),
            getattr(first, name),
            rtol=rtol,
            atol=atol,
            err_msg=f"Periodic translation changed {name}.",
        )

    first_power = first.power
    second_power = second.power
    for name in (
        "incident_power",
        "reflectance",
        "transmittance",
        "local_absorptance",
        "flux_defect_fraction",
    ):
        first_value = getattr(first_power, name)
        second_value = getattr(second_power, name)
        if first_value is None or second_value is None:
            raise AssertionError(f"Periodic power field {name!r} must be available.")
        np.testing.assert_allclose(
            second_value,
            first_value,
            rtol=rtol,
            atol=atol,
            err_msg=f"Periodic translation changed power observable {name}.",
        )


@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
@pytest.mark.parametrize("polar_angle", [0.0, 0.35])
def test_periodic_observables_are_invariant_to_wrapped_xy_translation_numpy(
    method: Literal["ewald", "rayleigh"],
    polar_angle: float,
) -> None:
    particles = _mixed_periodic_particles()
    reference = _run_periodic_case(
        particles,
        polar_angle=polar_angle,
        method=method,
        backend="numpy",
    )
    translated = _run_periodic_case(
        _periodically_shift_particles(particles),
        polar_angle=polar_angle,
        method=method,
        backend="numpy",
    )
    _assert_translation_invariant_observables(reference, translated, rtol=2e-10, atol=2e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
@pytest.mark.parametrize("polar_angle", [0.0, 0.35])
def test_periodic_observables_are_invariant_to_wrapped_xy_translation_cupy(
    cupy_runtime: tuple[Any, Any],
    method: Literal["ewald", "rayleigh"],
    polar_angle: float,
) -> None:
    del cupy_runtime
    particles = _mixed_periodic_particles()
    reference = _run_periodic_case(
        particles,
        polar_angle=polar_angle,
        method=method,
        backend="cupy",
    )
    translated = _run_periodic_case(
        _periodically_shift_particles(particles),
        polar_angle=polar_angle,
        method=method,
        backend="cupy",
    )
    _assert_translation_invariant_observables(reference, translated, rtol=5e-5, atol=2e-8)


@pytest.mark.parametrize(
    "z_separation",
    [
        pytest.param(0.0, id="coincident-z"),
        pytest.param(0.25, id="tiny-z-separation"),
        pytest.param(240.0, id="large-z-separation"),
    ],
)
@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
def test_periodic_seam_crossing_preserves_close_pair_observables_numpy(
    z_separation: float,
    method: Literal["ewald", "rayleigh"],
) -> None:
    particles, translated_particles = _seam_pair_and_translation(z_separation)

    reference = _run_periodic_case(
        particles,
        polar_angle=0.35,
        method=method,
        backend="numpy",
    )
    translated = _run_periodic_case(
        translated_particles,
        polar_angle=0.35,
        method=method,
        backend="numpy",
    )
    _assert_translation_invariant_observables(reference, translated, rtol=2e-10, atol=2e-10)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "z_separation",
    [
        pytest.param(0.0, id="coincident-z"),
        pytest.param(0.25, id="tiny-z-separation"),
        pytest.param(240.0, id="large-z-separation"),
    ],
)
@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
def test_periodic_seam_crossing_preserves_close_pair_observables_cupy(
    cupy_runtime: tuple[Any, Any],
    z_separation: float,
    method: Literal["ewald", "rayleigh"],
) -> None:
    del cupy_runtime
    particles, translated_particles = _seam_pair_and_translation(z_separation)

    reference = _run_periodic_case(
        particles,
        polar_angle=0.35,
        method=method,
        backend="cupy",
    )
    translated = _run_periodic_case(
        translated_particles,
        polar_angle=0.35,
        method=method,
        backend="cupy",
    )
    _assert_translation_invariant_observables(reference, translated, rtol=5e-5, atol=2e-8)
