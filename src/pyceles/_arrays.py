"""Small ownership helpers for immutable public NumPy state."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def expose_read_only_view(owner: np.ndarray) -> np.ndarray:
    """Expose a private owning array through a non-owning read-only view.

    Use this only when ``owner`` was allocated by pyceles and will remain
    private.  Locking the owner matters: a read-only array that owns its data
    can otherwise be made writable again by a caller with ``setflags``.
    """
    owner.setflags(write=False)
    view = owner.view()
    view.setflags(write=False)
    return view


def owned_read_only_view(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike | None = None,
    order: str = "C",
) -> np.ndarray:
    """Take one private copy and expose it as a non-owning read-only view."""
    owner = np.array(values, dtype=dtype, copy=True, order=order)
    return expose_read_only_view(owner)
