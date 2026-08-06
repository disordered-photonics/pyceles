from __future__ import annotations

from typing import cast

import numpy as np


def integrate_periodic_alpha(values: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Integrate azimuthal samples while enforcing 2*pi periodic closure."""
    values = np.asarray(values)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.size < 2:
        return np.zeros(values.shape[1:], dtype=np.result_type(values, np.float64))
    span = alpha[-1] - alpha[0]
    if np.isclose(span, 2.0 * np.pi):
        return cast(np.ndarray, np.trapezoid(values, alpha, axis=0))
    alpha_ext = np.concatenate([alpha, [alpha[0] + 2.0 * np.pi]])
    values_ext = np.concatenate([values, values[0:1, ...]], axis=0)
    return cast(np.ndarray, np.trapezoid(values_ext, alpha_ext, axis=0))


__all__ = ["integrate_periodic_alpha"]
