"""Tests for metadata exposed by an installed pyceles distribution."""

from __future__ import annotations

from importlib.metadata import version

import pyceles


def test_runtime_version_matches_distribution_metadata() -> None:
    """The import-time and installed-distribution versions stay identical."""
    assert pyceles.__version__ == version("pyceles")
