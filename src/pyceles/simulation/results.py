"""Typed simulation result containers.

A completed physical channel is represented independently from the solve that
produced it. Ordinary single-source runs add one matching solver report,
whereas multi-source and polarization envelopes own the shared block solve.
This keeps source cardinality and solver provenance explicit without copying
large coefficient arrays. Field bindings and labeled mappings are read-only;
numerical arrays remain mutable so result construction never requires deep
copies of potentially GiB-scale payloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import n_modes
from pyceles.core.particles import Particle, ParticleCollection
from pyceles.core.plane_wave_spectrum import PlaneWaveSpectrum
from pyceles.core.sources import JonesPolarizedSource, Source
from pyceles.linear.solvers import LinearSolveResult
from pyceles.postprocessing.farfield import (
    CrossSectionBalance,
    FarFieldPatterns,
    PeriodicFarFieldPayload,
    PowerBalance,
)

from .config import SimulationConfig


@dataclass(frozen=True, slots=True)
class ResultRetention:
    """Control optional large arrays retained by completed channel results.

    Solved coefficients are always retained because they are the restart and
    deferred-postprocessing state. Multi-source and polarization results retain
    every solved RHS channel; those vectors are essential state, not optional
    duplicated diagnostics.
    """

    initial_coeffs: bool = True
    rhs: bool = True
    residual_history: bool = True

    @classmethod
    def minimal(cls) -> ResultRetention:
        """Retain solved coefficients and compact solver diagnostics only."""
        return cls(
            initial_coeffs=False,
            rhs=False,
            residual_history=False,
        )


def _apply_solver_result_retention(
    result: LinearSolveResult,
    *,
    retention: ResultRetention,
) -> LinearSolveResult:
    if retention.residual_history:
        return result
    return replace(
        result,
        residual_history=None,
        preconditioned_residual_history=None,
        true_residual_history=None,
        block_residual_history=None,
    )


@dataclass(frozen=True, slots=True)
class UnpolarizedDiagnostics:
    """Incoherent averages of orthogonal TE/TM scalar diagnostics."""

    power: PowerBalance | None = None
    cross_sections: CrossSectionBalance | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return the compact canonical serialization mapping."""
        out: dict[str, object] = {}
        if self.power is not None:
            out["power"] = self.power.to_mapping()
        if self.cross_sections is not None:
            out["cross_sections"] = self.cross_sections.to_mapping()
        return out


def _solver_solution_matrix(result: LinearSolveResult) -> np.ndarray:
    """Return the authoritative solver solution as ``(unknowns, rhs_count)``."""
    rhs_count = int(result.rhs_count)
    if rhs_count < 1:
        raise ValueError(f"Solver RHS count must be positive. Got {rhs_count}.")
    x = np.asarray(result.x)
    if rhs_count == 1:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2 and x.shape[1] == 1:
            return x
        raise ValueError(
            f"Single-RHS solver solutions must be 1D or have shape (unknowns, 1). Got {x.shape}."
        )
    if x.ndim != 2 or x.shape[1] != rhs_count:
        raise ValueError(
            "Multi-RHS solver solution shape must match rhs_count: "
            f"shape={x.shape}, rhs_count={rhs_count}."
        )
    return x


def _validate_channel_solution_views(
    *,
    labels: tuple[str, ...],
    coeffs_by_label: Mapping[str, np.ndarray],
    solver_result: LinearSolveResult,
) -> None:
    """Require each solved channel to be a zero-copy view of its solver column."""
    solution = _solver_solution_matrix(solver_result)
    if solution.shape[1] != len(labels):
        raise ValueError(
            "Solver solution column count must match channel labels: "
            f"{solution.shape[1]} != {len(labels)}."
        )
    for column, label in enumerate(labels):
        coeffs = np.asarray(coeffs_by_label[label]).reshape(-1)
        solved = solution[:, column]
        if coeffs.shape != solved.shape:
            raise ValueError(
                f"Coefficient shape for channel '{label}' does not match solver column: "
                f"{coeffs.shape} != {solved.shape}."
            )
        if coeffs.size and not np.shares_memory(coeffs, solved):
            raise ValueError(
                f"Channel '{label}' coefficients must be a zero-copy view of the "
                "authoritative solver solution."
            )


def _compact_labels(labels: Sequence[str], *, limit: int = 4) -> str:
    values = tuple(labels)
    if len(values) <= limit:
        return repr(values)
    head = ", ".join(repr(label) for label in values[:limit])
    return f"({head}, ...; {len(values)} total)"


def _array_summary(value: object) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return f"type={type(value).__name__}"
    return f"shape={tuple(shape)}, dtype={getattr(value, 'dtype', None)}"


def _channel_repr_body(channel: ChannelResult) -> str:
    return (
        f"source={type(channel.source).__name__}, particles={channel.n_particles}, "
        f"coeffs=({_array_summary(channel.coeffs)}), "
        f"farfield={bool(channel.farfield.scattered.coeff_te.size)}, "
        f"periodic={channel.periodic is not None}, "
        f"power={channel.power is not None}, cross_sections={channel.cross_sections is not None}, "
        f"compute_dtype={channel.compute_dtype!r}, accum_dtype={channel.accum_dtype!r}"
    )


@dataclass(frozen=True, slots=True, repr=False)
class ChannelResult:
    """One physical source channel and its derived observables.

    This payload deliberately has no solver report. A channel may be solved
    directly, extracted from a shared block solve, or coherently derived from
    polarization-basis channels. The owning result envelope records that
    provenance explicitly.
    """

    config: SimulationConfig
    source: Source
    k: float
    k0: float
    coeffs: np.ndarray
    rhs: np.ndarray | None
    initial_coeffs: np.ndarray | None
    farfield: FarFieldPatterns
    power: PowerBalance | None
    cross_sections: CrossSectionBalance | None
    decomposition_forward: dict[str, float] | None
    decomposition_backward: dict[str, float] | None
    particles: ParticleCollection | Sequence[Particle]
    periodic: PeriodicFarFieldPayload | None
    compute_dtype: str
    accum_dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "particles",
            ParticleCollection.from_particles(self.particles),
        )

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def positions(self) -> np.ndarray:
        particles = self.particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("ChannelResult particles were not normalized.")
        return particles.positions

    @property
    def circumscribing_radii(self) -> np.ndarray:
        particles = self.particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("ChannelResult particles were not normalized.")
        return particles.circumscribing_radii

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_channel_repr_body(self)})"


@dataclass(frozen=True, slots=True, repr=False)
class SimulationResult(ChannelResult):
    """One directly solved source channel plus its matching solver report."""

    solver_result: LinearSolveResult

    def __post_init__(self) -> None:
        ChannelResult.__post_init__(self)
        _validate_channel_solution_views(
            labels=("source",),
            coeffs_by_label={"source": self.coeffs},
            solver_result=self.solver_result,
        )

    def __repr__(self) -> str:
        return f"SimulationResult({_channel_repr_body(self)}, solver={self.solver_result!r})"


def _validate_labeled_payload(
    *,
    labels: tuple[str, ...],
    payloads: Sequence[Mapping[str, object]],
    rhs_count: int,
) -> None:
    if not labels:
        raise ValueError("Labeled multi-source results require at least one channel.")
    if any(not isinstance(label, str) for label in labels):
        raise TypeError("Labeled result keys must be strings.")
    if any(label == "" for label in labels):
        raise ValueError("Labeled result keys must be non-empty strings.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Source labels must be unique. Got {labels!r}.")
    expected = set(labels)
    for payload in payloads:
        if set(payload) != expected:
            raise ValueError(
                "Labeled result keys must exactly match `labels`: "
                f"expected {labels!r}, got {tuple(payload)!r}."
            )
    if rhs_count != len(labels):
        raise ValueError(
            f"Solver RHS count must match the labeled channel count: {rhs_count} != {len(labels)}."
        )


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceSolveResult:
    """Solve-only outputs from one shared-operator multi-RHS solve."""

    config: SimulationConfig
    particles: ParticleCollection | Sequence[Particle]
    labels: tuple[str, ...]
    sources: Mapping[str, Source]
    solver_result: LinearSolveResult
    initial_coeffs: Mapping[str, np.ndarray]
    rhs: Mapping[str, np.ndarray]
    coeffs: Mapping[str, np.ndarray]
    k: float
    k0: float
    compute_dtype: str
    accum_dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "particles",
            ParticleCollection.from_particles(self.particles),
        )
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        object.__setattr__(
            self,
            "initial_coeffs",
            MappingProxyType(dict(self.initial_coeffs)),
        )
        object.__setattr__(self, "rhs", MappingProxyType(dict(self.rhs)))
        object.__setattr__(self, "coeffs", MappingProxyType(dict(self.coeffs)))
        _validate_labeled_payload(
            labels=self.labels,
            payloads=(self.sources, self.initial_coeffs, self.rhs, self.coeffs),
            rhs_count=int(self.solver_result.rhs_count),
        )
        expected_shape = (len(self.particles), n_modes(self.config.lmax))
        for payload_name, payload in (
            ("Initial coefficients", self.initial_coeffs),
            ("RHS", self.rhs),
            ("Solved coefficients", self.coeffs),
        ):
            for label in self.labels:
                shape = tuple(np.shape(payload[label]))
                if shape != expected_shape:
                    raise ValueError(
                        f"{payload_name} for channel '{label}' must have "
                        f"shape {expected_shape}. Got {shape}."
                    )
        _validate_channel_solution_views(
            labels=self.labels,
            coeffs_by_label=self.coeffs,
            solver_result=self.solver_result,
        )

    def __repr__(self) -> str:
        return (
            f"MultiSourceSolveResult(labels={_compact_labels(self.labels)}, "
            f"particles={len(self.particles)}, coeffs={len(self.coeffs)}, "
            f"compute_dtype={self.compute_dtype!r}, accum_dtype={self.accum_dtype!r}, "
            f"solver={self.solver_result!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class MultiSourceResult:
    """Postprocessed channels from one shared block solve.

    Channels contain physical per-source state only; ``solver_result`` is the
    single authoritative multi-RHS report for the whole block execution.
    """

    labels: tuple[str, ...]
    channels: Mapping[str, ChannelResult]
    solver_result: LinearSolveResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))
        _validate_labeled_payload(
            labels=self.labels,
            payloads=(self.channels,),
            rhs_count=int(self.solver_result.rhs_count),
        )
        _validate_channel_solution_views(
            labels=self.labels,
            coeffs_by_label={label: self.channels[label].coeffs for label in self.labels},
            solver_result=self.solver_result,
        )
        first = self.channels[self.labels[0]]
        for label in self.labels[1:]:
            channel = self.channels[label]
            if channel.config is not first.config:
                raise ValueError("All channels must share one SimulationConfig.")
            if channel.particles is not first.particles:
                raise ValueError("All channels must share one particle collection.")

    def __getitem__(self, label: str) -> ChannelResult:
        return self.channels[label]

    @property
    def config(self) -> SimulationConfig:
        """Return the shared immutable simulation configuration."""
        return self.channels[self.labels[0]].config

    @property
    def particles(self) -> ParticleCollection:
        """Return the shared immutable particle collection."""
        particles = self.channels[self.labels[0]].particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("MultiSourceResult particles were not normalized.")
        return particles

    @property
    def sources(self) -> Mapping[str, Source]:
        return MappingProxyType({label: self.channels[label].source for label in self.labels})

    def __repr__(self) -> str:
        return (
            f"MultiSourceResult(labels={_compact_labels(self.labels)}, "
            f"particles={len(self.particles)}, channels={len(self.channels)}, "
            f"solver={self.solver_result!r})"
        )


@dataclass(frozen=True, slots=True)
class _CoherentChannelPayload:
    farfield: FarFieldPatterns
    power: PowerBalance | None
    cross_sections: CrossSectionBalance | None
    decomposition_forward: dict[str, float] | None
    decomposition_backward: dict[str, float] | None
    periodic: PeriodicFarFieldPayload | None


@dataclass(frozen=True, slots=True, repr=False)
class PolarizationResult:
    """TE/TM basis channels from one block solve and their Jones combination.

    ``te`` and ``tm`` are the two solved channels and share the block solve in
    ``solver_result``. The requested coherent channel is exposed by ``mixed``;
    its large coefficient/RHS arrays are materialized on demand, so the result
    does not permanently retain a third coefficient vector.
    """

    source: JonesPolarizedSource
    te: ChannelResult
    tm: ChannelResult
    solver_result: LinearSolveResult
    unpolarized: UnpolarizedDiagnostics
    _mixed_payload: _CoherentChannelPayload

    def __post_init__(self) -> None:
        if self.te.config is not self.tm.config:
            raise ValueError("TE and TM channels must share one SimulationConfig.")
        if self.te.particles is not self.tm.particles:
            raise ValueError("TE and TM channels must share one particle collection.")
        if int(self.solver_result.rhs_count) != 2:
            raise ValueError(
                "PolarizationResult requires one two-RHS TE/TM solver report. "
                f"Got rhs_count={self.solver_result.rhs_count}."
            )
        _validate_channel_solution_views(
            labels=self.labels,
            coeffs_by_label={"te": self.te.coeffs, "tm": self.tm.coeffs},
            solver_result=self.solver_result,
        )

    @property
    def jones(self) -> tuple[complex, complex]:
        """Return the coherent Jones weights requested by ``source``."""
        a_te, a_tm = self.source.jones_coefficients()
        return complex(a_te), complex(a_tm)

    @property
    def config(self) -> SimulationConfig:
        """Return the shared immutable simulation configuration."""
        return self.te.config

    @property
    def particles(self) -> ParticleCollection:
        """Return the shared immutable particle collection."""
        particles = self.te.particles
        if not isinstance(particles, ParticleCollection):
            raise RuntimeError("PolarizationResult particles were not normalized.")
        return particles

    @property
    def labels(self) -> tuple[str, str]:
        """Return the deterministic solved-basis label order."""
        return ("te", "tm")

    @property
    def channels(self) -> Mapping[str, ChannelResult]:
        """Return the two solved orthogonal basis channels."""
        return MappingProxyType({"te": self.te, "tm": self.tm})

    def __getitem__(self, label: str) -> ChannelResult:
        if label == "te":
            return self.te
        if label == "tm":
            return self.tm
        raise KeyError(label)

    def __repr__(self) -> str:
        return (
            f"PolarizationResult(source={type(self.source).__name__}, jones={self.jones!r}, "
            f"particles={len(self.particles)}, periodic={self.te.periodic is not None}, "
            f"solver={self.solver_result!r})"
        )

    @property
    def mixed(self) -> ChannelResult:
        """Materialize the source-requested coherent Jones channel.

        The returned arrays are linear combinations of the retained basis
        vectors. They are allocated only when this property is requested and
        are not retained by the polarization envelope.
        """
        a_te, a_tm = self.jones
        compute_dtype = np.dtype(self.te.compute_dtype)
        accum_dtype = np.dtype(self.te.accum_dtype)

        coeffs = np.asarray(a_te * self.te.coeffs + a_tm * self.tm.coeffs, dtype=compute_dtype)
        rhs = None
        if self.te.rhs is not None and self.tm.rhs is not None:
            rhs = np.asarray(a_te * self.te.rhs + a_tm * self.tm.rhs, dtype=accum_dtype)
        initial_coeffs = None
        if self.te.initial_coeffs is not None and self.tm.initial_coeffs is not None:
            initial_coeffs = np.asarray(
                a_te * self.te.initial_coeffs + a_tm * self.tm.initial_coeffs,
                dtype=accum_dtype,
            )

        payload = self._mixed_payload
        return ChannelResult(
            config=self.te.config,
            source=self.source,
            particles=self.te.particles,
            k=self.te.k,
            k0=self.te.k0,
            coeffs=coeffs,
            rhs=rhs,
            initial_coeffs=initial_coeffs,
            farfield=payload.farfield,
            power=payload.power,
            cross_sections=payload.cross_sections,
            decomposition_forward=payload.decomposition_forward,
            decomposition_backward=payload.decomposition_backward,
            periodic=payload.periodic,
            compute_dtype=self.te.compute_dtype,
            accum_dtype=self.te.accum_dtype,
        )


def average_power_balances(first: PowerBalance, second: PowerBalance) -> PowerBalance:
    """Return the incoherent arithmetic average of two power balances."""

    def avg_optional(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(0.5 * (float(a) + float(b)))

    particles = None
    if (
        first.local_absorbed_power_per_particle is not None
        and second.local_absorbed_power_per_particle is not None
    ):
        a = np.asarray(first.local_absorbed_power_per_particle, dtype=np.float64)
        b = np.asarray(second.local_absorbed_power_per_particle, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(
                "Cannot average power balances with different per-particle shapes: "
                f"{a.shape} and {b.shape}."
            )
        particles = np.asarray(0.5 * (a + b), dtype=np.float64)
    return PowerBalance(
        incident_power=avg_optional(first.incident_power, second.incident_power),
        reflected_power=avg_optional(first.reflected_power, second.reflected_power),
        transmitted_power=avg_optional(first.transmitted_power, second.transmitted_power),
        local_absorbed_power=avg_optional(
            first.local_absorbed_power,
            second.local_absorbed_power,
        ),
        local_absorbed_power_per_particle=particles,
    )


def average_cross_section_balances(
    first: CrossSectionBalance,
    second: CrossSectionBalance,
) -> CrossSectionBalance:
    """Return the incoherent arithmetic average of two cross-section balances."""
    return CrossSectionBalance(
        extinction=0.5 * (first.extinction + second.extinction),
        scattering=0.5 * (first.scattering + second.scattering),
        local_absorption=0.5 * (first.local_absorption + second.local_absorption),
    )


def empty_farfield_patterns(dtype: npt.DTypeLike) -> FarFieldPatterns:
    """Return an empty far-field payload for solve-only workflows."""
    return FarFieldPatterns(
        initial=None,
        scattered=PlaneWaveSpectrum.empty(dtype),
    )


def simulation_result_from_channel(
    channel: ChannelResult,
    solver_result: LinearSolveResult,
) -> SimulationResult:
    """Attach a matching single-RHS solver report to one solved channel."""
    return SimulationResult(
        config=channel.config,
        source=channel.source,
        k=channel.k,
        k0=channel.k0,
        coeffs=channel.coeffs,
        rhs=channel.rhs,
        initial_coeffs=channel.initial_coeffs,
        farfield=channel.farfield,
        power=channel.power,
        cross_sections=channel.cross_sections,
        decomposition_forward=channel.decomposition_forward,
        decomposition_backward=channel.decomposition_backward,
        particles=channel.particles,
        periodic=channel.periodic,
        compute_dtype=channel.compute_dtype,
        accum_dtype=channel.accum_dtype,
        solver_result=solver_result,
    )


__all__ = [
    "ChannelResult",
    "MultiSourceResult",
    "MultiSourceSolveResult",
    "PolarizationResult",
    "ResultRetention",
    "SimulationResult",
    "UnpolarizedDiagnostics",
    "average_cross_section_balances",
    "average_power_balances",
    "empty_farfield_patterns",
    "simulation_result_from_channel",
]
