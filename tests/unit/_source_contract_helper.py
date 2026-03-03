from __future__ import annotations

import numpy as np

from pyceles.core.sources import (
    AngularSpectrumSource,
    JonesPolarizedSource,
    Source,
)


def assert_source_compliance(
    source: object,
    *,
    expect_angular_spectrum: bool,
    expect_finite_incident_power: bool,
    expect_jones_polarization: bool,
) -> None:
    """Assert that one source object satisfies pyceles source capability contract."""
    assert isinstance(source, Source), (
        f"{type(source).__name__} must satisfy the Source protocol "
        "(required methods/properties for solver integration)."
    )
    src = source
    assert callable(getattr(source, "incident_coeffs", None))
    assert callable(getattr(source, "has_finite_incident_power", None))

    wl = float(src.wavelength)
    amp = float(src.amplitude)
    n_medium = complex(src.medium_n)
    assert np.isfinite(wl)
    assert np.isfinite(amp)
    assert np.isfinite(n_medium.real) and np.isfinite(n_medium.imag)

    assert isinstance(source, AngularSpectrumSource) is expect_angular_spectrum
    assert bool(source.has_finite_incident_power()) is expect_finite_incident_power
    assert isinstance(source, JonesPolarizedSource) is expect_jones_polarization

    if expect_jones_polarization:
        if not isinstance(source, JonesPolarizedSource):
            raise AssertionError("Expected source to satisfy JonesPolarizedSource protocol.")
        assert callable(getattr(source, "jones_coefficients", None))
        assert callable(getattr(source, "with_polarization", None))
        a_te, a_tm = source.jones_coefficients()
        assert np.isfinite(a_te.real) and np.isfinite(a_te.imag)
        assert np.isfinite(a_tm.real) and np.isfinite(a_tm.imag)
        polarized = source.with_polarization("TE")
        assert isinstance(polarized, Source)
    else:
        assert not callable(getattr(source, "jones_coefficients", None))
        assert not callable(getattr(source, "with_polarization", None))
