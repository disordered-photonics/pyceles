"""Near-plane reciprocal routing must preserve the singular real-space field."""

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic.ewald import ewald_structural_constant_2d, ewald_structural_sums_2d_batch
from pyceles.core.periodic.scalar import same_plane_z_tolerance
from pyceles.core.periodic.structural import free_space_structural_sums


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("unit", [1.0, 1e-9])
@pytest.mark.parametrize("host", [1.33, 1.33 + 0.15j])
def test_near_coplanar_distinct_centres_retain_their_bulk_green_field(request, backend, unit, host):
    k = 2 * np.pi / (550 * unit) * host
    eta = 0.003 / unit
    lattice = RectangularLattice2D(600 * unit, 650 * unit)
    displacement = np.array([0.0, 0.0, same_plane_z_tolerance(k) / 4])
    q = np.zeros(2)
    expected = free_space_structural_sums(max_degree=2, k=k, displacement=displacement)
    if backend == "numpy":
        sums = ewald_structural_sums_2d_batch(
            lmax_struct=1,
            k=k,
            destinations=displacement[None, :],
            source=np.zeros(3),
            lattice=lattice,
            k_parallel=q,
            eta=eta,
            real_shells=4,
            reciprocal_shells=6,
        )[0]
        for degree in (1, 2):
            scalar = ewald_structural_constant_2d(
                degree,
                0,
                k=k,
                destination=displacement,
                source=np.zeros(3),
                lattice=lattice,
                k_parallel=q,
                eta=eta,
                real_shells=4,
                reciprocal_shells=6,
            )
            np.testing.assert_allclose(scalar, expected[degree, 2], rtol=2e-10, atol=0)
    else:
        cp, _ = request.getfixturevalue("cupy_runtime")
        from pyceles.core.periodic.ewald_cupy import (
            CupyEwaldShellWorkspace,
            ewald_structural_sums_2d_fixed_cupy,
        )

        workspace = CupyEwaldShellWorkspace(cupy=cp, lattice=lattice, k=k, k_parallel=q, eta=eta)
        sums = cp.asnumpy(
            ewald_structural_sums_2d_fixed_cupy(
                relative_source_minus_destination=cp.asarray(-displacement[None, :]),
                lmax_struct=1,
                workspace=workspace,
                real_shell_count=4,
                reciprocal_shell_count=6,
            )
        )[0]
    # Nonzero-image terms are bounded; the near-coincident singular term
    # dominates these degrees in either length unit.
    np.testing.assert_allclose(sums[[1, 2], 2], expected[[1, 2], 2], rtol=2e-10, atol=0)
