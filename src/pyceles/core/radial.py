"""Spherical outgoing radial functions for the exp(-i*omega*t) convention."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import hankel1


def sph_hankel1(l: int | np.ndarray, z: npt.ArrayLike) -> np.ndarray:
    """Evaluate spherical h_l^(1)(z) in double precision.

    Half-integer cylindrical Hankel functions avoid cancellation between the
    exponentially growing j_l and i*y_l terms for attenuating arguments.
    Outgoing waves are singular at zero; callers must exclude their centres.
    """
    argument = np.asarray(z, dtype=np.complex128)
    return np.asarray(np.sqrt(np.pi / (2 * argument)) * hankel1(np.asarray(l) + 0.5, argument))


def dx_xz_hankel1(l: int | np.ndarray, z: npt.ArrayLike) -> np.ndarray:
    """Evaluate d[z*h_l^(1)(z)]/dz, including order zero, in double precision."""
    argument = np.asarray(z, dtype=np.complex128)
    order = np.asarray(l)
    scale = np.sqrt(np.pi / (2 * argument))
    return np.asarray(
        scale * (argument * hankel1(order - 0.5, argument) - order * hankel1(order + 0.5, argument))
    )
