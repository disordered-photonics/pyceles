from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere
from pyceles.postprocessing.nearfield import components


class _ProgressRecorder:
    instances: ClassVar[list[_ProgressRecorder]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.total = int(kwargs["total"])
        self.desc = str(kwargs["desc"])
        self.unit = str(kwargs["unit"])
        self.statuses: list[str] = []
        self.updates: list[int] = []
        self.closed = False
        self.instances.append(self)

    def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
        assert refresh
        self.statuses.append(str(value))

    def update(self, value: int) -> None:
        self.updates.append(int(value))

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _beam() -> PlaneWave:
    return PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )


def test_high_level_nearfield_progress_reports_physical_stages(monkeypatch) -> None:
    _ProgressRecorder.instances.clear()
    monkeypatch.setattr(components, "tqdm", _ProgressRecorder)

    nested_progress: list[tuple[str, bool]] = []
    points = np.array([[0.0, 0.0, 0.0], [250.0, 0.0, 0.0]], dtype=float)
    particles = [Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0j)]

    def fake_initial(field_points: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        nested_progress.append(("initial", bool(kwargs["show_progress"])))
        shape = (np.asarray(field_points).shape[0], 3)
        return np.ones(shape, dtype=np.complex128), np.ones(shape, dtype=np.complex128)

    def fake_scattered(
        field_points: np.ndarray, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        del args
        nested_progress.append(("scattered", bool(kwargs["show_progress"])))
        shape = (np.asarray(field_points).shape[0], 3)
        return np.full(shape, 2.0 + 0j), np.full(shape, 3.0 + 0j)

    def fake_internal(
        field_points: np.ndarray, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del args
        nested_progress.append(("internal", bool(kwargs["show_progress"])))
        shape = (np.asarray(field_points).shape[0], 3)
        inside = np.array([True, False])
        return np.full(shape, 4.0 + 0j), np.full(shape, 5.0 + 0j), inside

    monkeypatch.setattr(components, "compute_initial_field", fake_initial)
    monkeypatch.setattr(components, "compute_scattered_field", fake_scattered)
    monkeypatch.setattr(components, "compute_internal_field", fake_internal)

    out = components.compute_near_field_components(
        points,
        coeffs=np.zeros((1, 6), dtype=np.complex128),
        k=2.0 * np.pi / 550.0,
        lmax=1,
        beam=_beam(),
        polar_angles=np.linspace(0.0, np.pi, 5),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False),
        particles=particles,
        show_progress=True,
    )

    assert nested_progress == [("initial", False), ("scattered", False), ("internal", False)]
    assert len(_ProgressRecorder.instances) == 1
    progress = _ProgressRecorder.instances[0]
    assert (progress.desc, progress.unit, progress.total) == ("Near field", "stage", 3)
    assert progress.statuses == ["initial", "classifying points", "scattered", "internal"]
    assert progress.updates == [1, 1, 1]
    assert progress.closed
    np.testing.assert_allclose(out.E_total[0], 4.0)
    np.testing.assert_allclose(out.E_total[1], 3.0)


def test_high_level_nearfield_skips_internal_kernel_when_no_points_are_inside(monkeypatch) -> None:
    _ProgressRecorder.instances.clear()
    monkeypatch.setattr(components, "tqdm", _ProgressRecorder)
    points = np.array([[250.0, 0.0, 0.0], [400.0, 0.0, 0.0]], dtype=float)
    particles = [Sphere(position=(0.0, 0.0, 0.0), radius=100.0, refractive_index=1.5 + 0j)]

    def fail_internal(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("internal evaluator should not run without interior sample points")

    monkeypatch.setattr(components, "compute_internal_field", fail_internal)
    out = components.compute_near_field_components(
        points,
        coeffs=np.zeros((1, 6), dtype=np.complex128),
        k=2.0 * np.pi / 550.0,
        lmax=1,
        beam=_beam(),
        polar_angles=np.linspace(0.0, np.pi, 5),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False),
        particles=particles,
        show_progress=True,
    )

    assert not np.any(out.inside_mask)
    progress = _ProgressRecorder.instances[0]
    assert progress.total == 2
    assert progress.statuses == ["initial", "classifying points", "scattered"]
    assert progress.updates == [1, 1]
