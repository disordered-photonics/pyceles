from __future__ import annotations

import os

_LOGO = r"""
           ____^____                          ___________   ________
          / / / / /                ___  __ __/ ___/ __/ /  / __/ __/
          \_\_\_\_\ <\            / _ \/ // / /__/ _// /__/ _/_\ \
        (\_____!_____))          / .__/\_, /\___/___/____/___/___/
         \\\\\\\\\\\\/          /_/   /___/
      ----^~^~^~^~^~^~^~~-~--                          version {version}
"""


def print_logo(version: str = "unknown", *, force: bool = False) -> None:
    """Print the pyCELES ASCII logo.

    By default, printing can be disabled with env var `PYCELES_NO_LOGO`.
    Pass `force=True` to ignore that env var and print anyway.
    """
    if (not force) and os.environ.get("PYCELES_NO_LOGO", "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }:
        return
    print(_LOGO.format(version=version))
