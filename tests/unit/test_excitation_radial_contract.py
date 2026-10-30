"""Selected internal reception must honor the outgoing radial table domain."""

from typing import Any

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.particles import ParticleCollection, Sphere
from pyceles.core.sources import PlaneWave
from pyceles.postprocessing.nearfield import excitation


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_selected_reception_rejects_nonrepresentable_outgoing_padding(backend):
    particles = ParticleCollection.from_particles(
        tuple(Sphere((x, 0, 0), 0.001, 1.5) for x in (0, 0.1, 100))
    )
    kwargs: dict[str, Any] = dict(
        particles=particles,
        host_source=PlaneWave(550),
        coeffs=np.ones((3, n_modes(7))),
        k=2 * np.pi / 550,
        lmax=7,
        lut_dr=0,
        compute_dtype=np.complex64,
        accum_dtype=np.complex128,
    )
    with pytest.raises(ValueError, match="includes padding"):
        excitation.exciting_coefficients(np.array([0]), backend=backend, **kwargs)
    if backend == "numpy":
        # The close pair does not invalidate reception at the distant owner.
        assert np.isfinite(excitation.exciting_coefficients(np.array([2]), **kwargs)).all()
